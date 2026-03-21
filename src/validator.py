"""
Simulation Validation & Quality Assessment Module

Compares SUMO simulation results against real data (statistics/TOPIS/CSV)
and generates an error analysis report.
"""

import json
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, asdict


@dataclass
class ValidationResult:
    """Validation result"""
    # Simulation results
    sim_speed_kmh: float = 0
    sim_volume_vph: int = 0
    sim_avg_waiting_s: float = 0
    sim_avg_timeloss_s: float = 0
    sim_vehicles_inserted: int = 0

    # Real data (comparison target)
    real_speed_kmh: float = 0
    real_volume_vph: int = 0

    # Error
    speed_error_pct: float = 0  # (sim - real) / real * 100
    volume_error_pct: float = 0

    # Assessment
    grade: str = ""  # A, B, C, D, F
    issues: list = None
    suggestions: list = None

    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.suggestions is None:
            self.suggestions = []

    def to_dict(self) -> dict:
        return asdict(self)


def parse_sumo_statistics(cfg_path: str, warmup_seconds: int = 0) -> dict:
    """
    Extract detailed results from a SUMO simulation.
    Parses detector output XML and tripinfo.
    Only uses data after warmup_seconds for evaluation.
    """
    cfg_dir = os.path.dirname(cfg_path)
    stats = {"warmup_seconds": max(int(warmup_seconds or 0), 0)}

    # 1) Parse detector output
    detector_file = os.path.join(cfg_dir, "detector_output.xml")
    if os.path.exists(detector_file):
        try:
            tree = ET.parse(detector_file)
            root = tree.getroot()

            intervals = root.findall("interval")
            if intervals:
                filtered_intervals = []
                skipped_intervals = 0
                for interval in intervals:
                    begin = float(interval.get("begin", 0) or 0)
                    end = float(interval.get("end", begin) or begin)
                    if warmup_seconds and begin < warmup_seconds:
                        skipped_intervals += 1
                        continue
                    if warmup_seconds and end <= warmup_seconds:
                        skipped_intervals += 1
                        continue
                    filtered_intervals.append(interval)

                speeds = []
                flows = []
                occupancies = []

                for interval in filtered_intervals:
                    speed = float(interval.get("speed", -1))
                    flow = float(interval.get("flow", 0))
                    occ = float(interval.get("occupancy", 0))

                    if speed >= 0:
                        speeds.append(speed)
                    flows.append(flow)
                    occupancies.append(occ)

                if speeds:
                    avg_speed_ms = sum(speeds) / len(speeds)
                    stats["detector_avg_speed_ms"] = round(avg_speed_ms, 2)
                    stats["detector_avg_speed_kmh"] = round(avg_speed_ms * 3.6, 1)

                if flows:
                    # flow is veh/interval -> convert to per hour
                    # if interval freq is 300s (5min), multiply by 12
                    freq = int(filtered_intervals[0].get("freq", 300) if filtered_intervals and hasattr(filtered_intervals[0], 'get') else 300)
                    multiplier = 3600 / freq if freq > 0 else 12
                    avg_flow = sum(flows) / len(flows)
                    stats["detector_avg_flow_vph"] = round(avg_flow * multiplier)
                    stats["detector_total_count"] = sum(int(i.get("nVehContrib", 0)) for i in filtered_intervals)

                if occupancies:
                    stats["detector_avg_occupancy"] = round(sum(occupancies) / len(occupancies), 2)

                stats["detector_intervals"] = len(filtered_intervals)
                if warmup_seconds:
                    stats["detector_intervals_skipped"] = skipped_intervals
        except Exception as e:
            stats["detector_error"] = str(e)

    # 2) Parse tripinfo (if available)
    tripinfo_file = os.path.join(cfg_dir, "tripinfo.xml")
    if os.path.exists(tripinfo_file):
        try:
            tree = ET.parse(tripinfo_file)
            root = tree.getroot()
            trips = root.findall("tripinfo")

            if trips:
                filtered_trips = []
                skipped_trips = 0
                for trip in trips:
                    depart = float(trip.get("depart", 0) or 0)
                    if warmup_seconds and depart < warmup_seconds:
                        skipped_trips += 1
                        continue
                    filtered_trips.append(trip)

                durations = [float(t.get("duration", 0)) for t in filtered_trips]
                wait_times = [float(t.get("waitingTime", 0)) for t in filtered_trips]
                time_losses = [float(t.get("timeLoss", 0)) for t in filtered_trips]
                route_lengths = [float(t.get("routeLength", 0)) for t in filtered_trips]

                stats["trip_count"] = len(filtered_trips)
                if warmup_seconds:
                    stats["trip_count_skipped"] = skipped_trips
                stats["trip_avg_duration_s"] = round(sum(durations) / len(durations), 1)
                stats["trip_avg_waiting_s"] = round(sum(wait_times) / len(wait_times), 1)
                stats["trip_avg_timeloss_s"] = round(sum(time_losses) / len(time_losses), 1)
                stats["trip_avg_length_m"] = round(sum(route_lengths) / len(route_lengths), 1)

                # Calculate average speed
                speeds = []
                for t in filtered_trips:
                    length = float(t.get("routeLength", 0))
                    duration = float(t.get("duration", 1))
                    if duration > 0 and length > 0:
                        speeds.append(length / duration)
                if speeds:
                    avg_speed = sum(speeds) / len(speeds)
                    stats["trip_avg_speed_ms"] = round(avg_speed, 2)
                    stats["trip_avg_speed_kmh"] = round(avg_speed * 3.6, 1)
        except Exception as e:
            stats["tripinfo_error"] = str(e)

    return stats


def validate(
    sim_stats: dict,
    real_speed_kmh: float = None,
    real_volume_vph: int = None,
    detailed_stats: dict = None,
) -> ValidationResult:
    """
    Validate simulation results by comparing against real data.

    Args:
        sim_stats: Statistics returned by run_sumo()
        real_speed_kmh: Real data average speed (km/h)
        real_volume_vph: Real data traffic volume (veh/h)
        detailed_stats: Detailed results from parse_sumo_statistics()

    Returns:
        ValidationResult
    """
    result = ValidationResult()

    # Fill simulation results
    result.sim_speed_kmh = sim_stats.get("avg_speed_kmh", 0)
    result.sim_volume_vph = sim_stats.get("vehicles_inserted", 0)
    result.sim_avg_waiting_s = sim_stats.get("avg_waiting_time_s", 0)
    result.sim_avg_timeloss_s = sim_stats.get("avg_time_loss_s", 0)
    result.sim_vehicles_inserted = sim_stats.get("vehicles_inserted", 0)

    # Detector-based speed may be more accurate
    if detailed_stats and "detector_avg_speed_kmh" in detailed_stats:
        result.sim_speed_kmh = detailed_stats["detector_avg_speed_kmh"]

    # Set real data
    if real_speed_kmh is not None:
        result.real_speed_kmh = real_speed_kmh
    if real_volume_vph is not None:
        result.real_volume_vph = real_volume_vph

    # Calculate error
    if result.real_speed_kmh > 0:
        result.speed_error_pct = round(
            (result.sim_speed_kmh - result.real_speed_kmh) / result.real_speed_kmh * 100, 1
        )
    if result.real_volume_vph > 0:
        result.volume_error_pct = round(
            (result.sim_volume_vph - result.real_volume_vph) / result.real_volume_vph * 100, 1
        )

    # Grade assessment
    speed_err_abs = abs(result.speed_error_pct)
    if speed_err_abs <= 10:
        result.grade = "A"
    elif speed_err_abs <= 20:
        result.grade = "B"
    elif speed_err_abs <= 30:
        result.grade = "C"
    elif speed_err_abs <= 50:
        result.grade = "D"
    else:
        result.grade = "F"

    # Issue diagnosis & improvement suggestions
    if result.speed_error_pct > 15:
        result.issues.append("Simulation speed is higher than real data (congestion underestimated)")
        result.suggestions.append("Increase vehicles_per_hour by 10~20%")
        result.suggestions.append("Increase sigma value (driver variability up -> capacity decrease)")
    elif result.speed_error_pct < -15:
        result.issues.append("Simulation speed is lower than real data (congestion overestimated)")
        result.suggestions.append("Decrease vehicles_per_hour by 10~20%")
        result.suggestions.append("Decrease tau value (headway time down -> capacity increase)")

    if result.sim_avg_waiting_s > 60:
        result.issues.append(f"Average waiting time is excessive at {result.sim_avg_waiting_s} seconds")
        result.suggestions.append("Check signal cycle optimization or actuated signal application")

    if result.sim_avg_timeloss_s > 120:
        result.issues.append(f"Average time loss is excessive at {result.sim_avg_timeloss_s} seconds")
        result.suggestions.append("Check network connectivity (unnecessary detour routes)")

    return result


def generate_report(
    validation: ValidationResult,
    params: dict = None,
    output_path: str = None,
) -> str:
    """
    Generate a text report from validation results.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("Simulation Validation Report")
    lines.append("=" * 60)

    lines.append(f"\nGrade: {validation.grade}")

    lines.append(f"\n{'Item':<25} {'Simulation':>12} {'Real Data':>12} {'Error':>10}")
    lines.append("-" * 60)

    if validation.real_speed_kmh > 0:
        lines.append(
            f"{'Avg Speed (km/h)':<25} {validation.sim_speed_kmh:>12.1f} "
            f"{validation.real_speed_kmh:>12.1f} {validation.speed_error_pct:>9.1f}%"
        )
    else:
        lines.append(f"{'Avg Speed (km/h)':<25} {validation.sim_speed_kmh:>12.1f} {'N/A':>12}")

    if validation.real_volume_vph > 0:
        lines.append(
            f"{'Volume (veh/h)':<25} {validation.sim_volume_vph:>12d} "
            f"{validation.real_volume_vph:>12d} {validation.volume_error_pct:>9.1f}%"
        )

    lines.append(f"{'Avg Waiting (s)':<25} {validation.sim_avg_waiting_s:>12.1f}")
    lines.append(f"{'Avg Time Loss (s)':<25} {validation.sim_avg_timeloss_s:>12.1f}")

    if validation.issues:
        lines.append(f"\nIssues Found:")
        for issue in validation.issues:
            lines.append(f"  - {issue}")

    if validation.suggestions:
        lines.append(f"\nImprovement Suggestions:")
        for sug in validation.suggestions:
            lines.append(f"  - {sug}")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        # Also save as JSON
        json_path = output_path.replace(".txt", ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(validation.to_dict(), f, ensure_ascii=False, indent=2)

    return report


def calibration_suggestion(validation: ValidationResult) -> dict:
    """
    Suggest parameter adjustments based on error analysis results.
    The LLM agent can use these results to re-run the simulation.
    """
    adjustments = {}

    speed_err = validation.speed_error_pct

    if abs(speed_err) <= 10:
        return {"status": "good", "message": "Error within 10%. No further adjustment needed."}

    # Traffic volume adjustment based on speed error
    if speed_err > 0:
        # Simulation is faster -> inject more vehicles
        volume_adj = min(speed_err * 1.5, 30)  # Max 30% increase
        adjustments["vehicles_per_hour_change_pct"] = round(volume_adj)
        adjustments["sigma_change"] = 0.05  # Increase variability
    else:
        # Simulation is slower -> reduce vehicles
        volume_adj = max(speed_err * 1.5, -30)  # Max 30% decrease
        adjustments["vehicles_per_hour_change_pct"] = round(volume_adj)
        adjustments["tau_change"] = -0.1  # Decrease headway time

    # If waiting time is too long, signal adjustment may be needed
    if validation.sim_avg_waiting_s > 60:
        adjustments["signal_note"] = "Signal cycle may be too long. Check actuated signal settings."

    adjustments["expected_improvement"] = f"Speed error {abs(speed_err):.0f}% -> ~{abs(speed_err)*0.5:.0f}% expected"

    return {
        "status": "needs_adjustment",
        "current_error_pct": speed_err,
        "adjustments": adjustments,
    }


if __name__ == "__main__":
    # Test: validate with dummy results
    sim_stats = {
        "avg_speed_kmh": 24.0,
        "vehicles_inserted": 1800,
        "avg_waiting_time_s": 15.2,
        "avg_time_loss_s": 45.7,
    }

    result = validate(sim_stats, real_speed_kmh=18.0, real_volume_vph=1900)
    report = generate_report(result)
    print(report)

    print("\n\n--- Calibration Suggestion ---")
    cal = calibration_suggestion(result)
    print(json.dumps(cal, ensure_ascii=False, indent=2))
