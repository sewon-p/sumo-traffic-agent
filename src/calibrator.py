"""
Calibration loop for SUMO traffic simulation.

Adjusts vehicles_per_hour, sigma, and tau iteratively so that
simulated average speed converges toward the FT-predicted target (speed_kmh).

Uses proportional control with fixed gains derived from SUMO Krauss model
sensitivity analysis. Drift from original FT values is bounded.
"""

import os
import re
import subprocess

from src.config import get_sumo_bin
from src.validator import parse_sumo_statistics
from tools.sumo_generator import (
    SimulationConfig,
    TrafficDemand,
    build_vtypes_from_ft,
    generate_all,
)

# Proportional gains (Krauss model sensitivity)
GAIN_VOLUME = 0.4
GAIN_SIGMA = 0.15
GAIN_TAU = 0.25

# Maximum drift from original FT values
MAX_DRIFT_VOLUME_PCT = 0.20
MAX_DRIFT_SIGMA = 0.15
MAX_DRIFT_TAU = 0.3

MAX_ITERATIONS = 3
TOLERANCE_PCT = 10.0


def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _run_sumo_headless(cfg_path, warmup_seconds=0):
    """Run SUMO and return stats dict with avg_speed_kmh."""
    tripinfo_path = os.path.join(os.path.dirname(cfg_path), "tripinfo.xml")
    cmd = [
        get_sumo_bin(),
        "-c", cfg_path,
        "--duration-log.statistics",
        "--no-step-log",
        "--tripinfo-output", tripinfo_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    stats = {}
    for line in output.split("\n"):
        line = line.strip()
        if "Inserted:" in line:
            m = re.search(r'Inserted:\s*(\d+)', line)
            if m:
                stats["vehicles_inserted"] = int(m.group(1))
        elif "Speed:" in line and "Statistics" not in line:
            m = re.search(r'Speed:\s*([\d.]+)', line)
            if m:
                stats["avg_speed_ms"] = float(m.group(1))
                stats["avg_speed_kmh"] = round(float(m.group(1)) * 3.6, 1)

    # Prefer detector/trip stats (with warmup filtering)
    detailed = parse_sumo_statistics(cfg_path, warmup_seconds=warmup_seconds)
    if "detector_avg_speed_kmh" in detailed:
        stats["avg_speed_kmh"] = detailed["detector_avg_speed_kmh"]
        stats["avg_speed_ms"] = detailed.get(
            "detector_avg_speed_ms",
            round(stats["avg_speed_kmh"] / 3.6, 2),
        )
    elif "trip_avg_speed_kmh" in detailed:
        stats["avg_speed_kmh"] = detailed["trip_avg_speed_kmh"]
        stats["avg_speed_ms"] = detailed.get(
            "trip_avg_speed_ms",
            round(stats["avg_speed_kmh"] / 3.6, 2),
        )

    return stats


def run_calibration(
    ft,
    params,
    net_path,
    output_dir,
    initial_sim_speed,
    warmup_seconds=0,
    max_iterations=MAX_ITERATIONS,
    tolerance_pct=TOLERANCE_PCT,
    on_iteration=None,
):
    """
    Closed-loop calibration: nudge volume/sigma/tau so sim_speed ≈ ft.speed_kmh.

    Args:
        ft: FT metadata dict (must contain "speed_kmh")
        params: SimulationParams object
        net_path: path to .net.xml (read-only, not regenerated)
        output_dir: base output directory (iterations go in cal_1/, cal_2/, ...)
        initial_sim_speed: avg_speed_kmh from the first simulation run
        warmup_seconds: SUMO warmup period
        max_iterations: maximum calibration iterations
        tolerance_pct: convergence threshold (%)
        on_iteration: optional callback(iter_data_dict) for progress streaming

    Returns:
        dict with keys: status, converged, iterations, original, calibrated,
                        drift, target_speed_kmh, final_speed_kmh, final_error_pct
    """
    target_speed = ft.get("speed_kmh")
    if not target_speed or target_speed <= 0:
        return {"status": "skipped", "converged": False, "reason": "no_target_speed",
                "iterations": []}

    # Snapshot originals (drift caps are relative to these)
    orig_vph = params.vehicles_per_hour
    orig_sigma = float(ft.get("sigma") or 0.5)
    orig_tau = float(ft.get("tau") or 1.0)

    cur_vph = orig_vph
    cur_sigma = orig_sigma
    cur_tau = orig_tau
    current_speed = initial_sim_speed

    iterations = []

    for i in range(max_iterations):
        error_pct = (current_speed - target_speed) / target_speed * 100

        # Already converged from previous iteration result (or initial)
        if i > 0 and abs(error_pct) <= tolerance_pct:
            break

        # First iteration: check if initial run already converged
        if i == 0 and abs(error_pct) <= tolerance_pct:
            return _build_result(
                True, [], orig_vph, orig_sigma, orig_tau,
                cur_vph, cur_sigma, cur_tau, target_speed,
                error_pct, error_pct,
            )

        # Proportional adjustment
        magnitude = error_pct / 100.0

        cur_vph = int(cur_vph * (1.0 + GAIN_VOLUME * magnitude))
        cur_vph = _clamp(
            cur_vph,
            int(orig_vph * (1 - MAX_DRIFT_VOLUME_PCT)),
            int(orig_vph * (1 + MAX_DRIFT_VOLUME_PCT)),
        )
        cur_vph = max(cur_vph, 100)

        cur_sigma = cur_sigma + GAIN_SIGMA * magnitude
        cur_sigma = _clamp(
            cur_sigma,
            max(orig_sigma - MAX_DRIFT_SIGMA, 0.0),
            min(orig_sigma + MAX_DRIFT_SIGMA, 1.0),
        )

        cur_tau = cur_tau + GAIN_TAU * magnitude
        cur_tau = _clamp(
            cur_tau,
            max(orig_tau - MAX_DRIFT_TAU, 0.5),
            min(orig_tau + MAX_DRIFT_TAU, 3.0),
        )
        cur_sigma = round(cur_sigma, 3)
        cur_tau = round(cur_tau, 3)

        # Detect stall: if params didn't change, all drift caps are hit
        if iterations:
            prev = iterations[-1].get("params", {})
            if (prev.get("vehicles_per_hour") == cur_vph
                    and prev.get("sigma") == cur_sigma
                    and prev.get("tau") == cur_tau):
                initial_error = (initial_sim_speed - target_speed) / target_speed * 100
                return _build_result(
                    False, iterations, orig_vph, orig_sigma, orig_tau,
                    cur_vph, cur_sigma, cur_tau, target_speed,
                    initial_error, iterations[-1]["error_pct"],
                    error_message="Drift limits reached, cannot adjust further",
                )

        # Build and run simulation
        iter_dir = os.path.join(output_dir, f"cal_{i + 1}")
        os.makedirs(iter_dir, exist_ok=True)

        try:
            cur_ft = {**ft, "sigma": cur_sigma, "tau": cur_tau}
            vtypes = build_vtypes_from_ft(cur_ft, speed_limit_kmh=params.speed_limit_kmh)

            demand = TrafficDemand(
                total_vehicles_per_hour=cur_vph,
                passenger_ratio=params.passenger_ratio,
                truck_ratio=params.truck_ratio,
                bus_ratio=params.bus_ratio,
            )

            h1 = int(params.time_start.split(":")[0])
            h2 = int(params.time_end.split(":")[0])
            eval_duration = max((h2 - h1) * 3600, 3600)

            sim_config = SimulationConfig(
                begin_time=0,
                end_time=eval_duration + warmup_seconds,
                warmup_seconds=warmup_seconds,
            )

            files = generate_all(
                net_path, iter_dir,
                demand=demand, vtypes=vtypes, sim_config=sim_config,
            )

            stats = _run_sumo_headless(files["cfg"], warmup_seconds=warmup_seconds)
            sim_speed = stats.get("avg_speed_kmh", 0)
        except Exception as e:
            iterations.append({
                "iteration": i + 1,
                "error": str(e),
                "params": {"vehicles_per_hour": cur_vph,
                           "sigma": cur_sigma, "tau": cur_tau},
            })
            final_err = iterations[-2]["error_pct"] if len(iterations) >= 2 else error_pct
            return _build_result(
                False, iterations, orig_vph, orig_sigma, orig_tau,
                cur_vph, cur_sigma, cur_tau, target_speed,
                (initial_sim_speed - target_speed) / target_speed * 100,
                final_err, error_message=str(e),
            )

        if sim_speed <= 0:
            iterations.append({
                "iteration": i + 1,
                "error": "no_speed_output",
                "params": {"vehicles_per_hour": cur_vph,
                           "sigma": cur_sigma, "tau": cur_tau},
            })
            return _build_result(
                False, iterations, orig_vph, orig_sigma, orig_tau,
                cur_vph, cur_sigma, cur_tau, target_speed,
                (initial_sim_speed - target_speed) / target_speed * 100,
                error_pct, error_message="SUMO produced no speed output",
            )

        new_error = (sim_speed - target_speed) / target_speed * 100

        iter_data = {
            "iteration": i + 1,
            "params": {"vehicles_per_hour": cur_vph,
                       "sigma": cur_sigma, "tau": cur_tau},
            "sim_speed_kmh": sim_speed,
            "target_speed_kmh": target_speed,
            "error_pct": round(new_error, 1),
            "converged": abs(new_error) <= tolerance_pct,
        }
        iterations.append(iter_data)

        if on_iteration:
            on_iteration(iter_data)

        current_speed = sim_speed

        if abs(new_error) <= tolerance_pct:
            break

    initial_error = (initial_sim_speed - target_speed) / target_speed * 100
    final_error = iterations[-1]["error_pct"] if iterations else initial_error
    converged = bool(iterations) and abs(final_error) <= tolerance_pct

    return _build_result(
        converged, iterations, orig_vph, orig_sigma, orig_tau,
        cur_vph, cur_sigma, cur_tau, target_speed,
        initial_error, final_error,
    )


def _build_result(
    converged, iterations, orig_vph, orig_sigma, orig_tau,
    final_vph, final_sigma, final_tau, target, initial_err, final_err,
    error_message="",
):
    status = "converged" if converged else ("error" if error_message else "max_iterations")
    return {
        "status": status,
        "converged": converged,
        "iterations": iterations,
        "target_speed_kmh": target,
        "initial_error_pct": round(initial_err, 1),
        "final_error_pct": round(final_err, 1),
        "final_speed_kmh": round(
            target * (1 + final_err / 100), 1,
        ) if final_err is not None else None,
        "original": {
            "vehicles_per_hour": orig_vph,
            "sigma": orig_sigma,
            "tau": orig_tau,
        },
        "calibrated": {
            "vehicles_per_hour": final_vph,
            "sigma": final_sigma,
            "tau": final_tau,
        },
        "drift": {
            "volume_pct": round((final_vph - orig_vph) / orig_vph * 100, 1) if orig_vph else 0,
            "sigma": round(final_sigma - orig_sigma, 3),
            "tau": round(final_tau - orig_tau, 3),
        },
        "error_message": error_message,
    }
