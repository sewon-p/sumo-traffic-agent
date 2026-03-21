"""
LLM-Powered Traffic Simulation Agent - Main Pipeline

Natural language input -> Parameter extraction -> OSM download -> SUMO file generation -> Simulation run
"""

import os
import sys
import subprocess
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.llm_parser import parse_user_input, SimulationParams
from tools.osm_network import build_network
from tools.sumo_generator import (
    generate_all,
    TrafficDemand,
    SimulationConfig,
)


def run_simulation(cfg_path: str, sumo_bin: str = "sumo") -> dict:
    """Run SUMO simulation and return results."""
    cmd = [
        sumo_bin,
        "-c", cfg_path,
        "--duration-log.statistics",
        "--no-step-log",
    ]

    print("\n🚗 Running SUMO simulation...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    output = result.stdout + result.stderr
    stats = {}

    # Parse results
    import re as _re
    for line in output.split("\n"):
        line = line.strip()
        if "Inserted:" in line:
            m = _re.search(r'Inserted:\s*(\d+)', line)
            if m:
                stats["vehicles_inserted"] = int(m.group(1))
        elif "Speed:" in line and "Statistics" not in line:
            m = _re.search(r'Speed:\s*([\d.]+)', line)
            if m:
                stats["avg_speed_ms"] = float(m.group(1))
                stats["avg_speed_kmh"] = round(float(m.group(1)) * 3.6, 1)
        elif "Duration:" in line and "Performance" not in line:
            m = _re.search(r'Duration:\s*([\d.]+)', line)
            if m:
                stats["avg_trip_duration_s"] = float(m.group(1))
        elif "WaitingTime:" in line:
            m = _re.search(r'WaitingTime:\s*([\d.]+)', line)
            if m:
                stats["avg_waiting_time_s"] = float(m.group(1))
        elif "TimeLoss:" in line:
            m = _re.search(r'TimeLoss:\s*([\d.]+)', line)
            if m:
                stats["avg_time_loss_s"] = float(m.group(1))

    return stats


def pipeline(user_input: str, api_key: str = None) -> dict:
    """
    End-to-end pipeline: natural language -> simulation results

    Args:
        user_input: User natural language input
        api_key: Anthropic API key (rule-based if not provided)

    Returns:
        Simulation result dictionary
    """
    print("=" * 60)
    print(f"📝 Input: {user_input}")
    print("=" * 60)

    # 1) Natural language -> parameters
    print("\n[1/4] Extracting parameters...")
    params = parse_user_input(user_input, api_key)
    print(f"  Location: {params.location}")
    print(f"  Time: {params.time_start} ~ {params.time_end}")
    print(f"  Traffic volume: {params.vehicles_per_hour} veh/h")
    print(f"  Speed limit: {params.speed_limit_kmh} km/h")
    if params.weather != "clear":
        print(f"  Weather: {params.weather}")
    if params.incident:
        print(f"  Special condition: {params.incident} (lane closure: {params.lane_closure})")

    # 2) OSM -> SUMO network
    print(f"\n[2/4] Building {params.location} road network...")
    output_dir = os.path.join("output", params.location.replace(" ", "_"))
    net_path = build_network(params.location, params.radius_m, output_dir)

    # 3) Generate SUMO files
    print(f"\n[3/4] Generating SUMO simulation files...")
    demand = TrafficDemand(
        total_vehicles_per_hour=params.vehicles_per_hour,
        passenger_ratio=params.passenger_ratio,
        truck_ratio=params.truck_ratio,
        bus_ratio=params.bus_ratio,
    )

    # Convert time to seconds (simulation always starts from 0, only duration is set)
    h1 = int(params.time_start.split(":")[0])
    h2 = int(params.time_end.split(":")[0])
    duration = max((h2 - h1) * 3600, 3600)  # Minimum 1 hour

    sim_config = SimulationConfig(
        begin_time=0,
        end_time=duration,
    )

    files = generate_all(net_path, output_dir, demand=demand, sim_config=sim_config)

    # 4) Run simulation
    print(f"\n[4/4] Running simulation...")
    stats = run_simulation(files["cfg"])

    # Organize results
    result = {
        "input": user_input,
        "params": params.to_dict(),
        "files": files,
        "statistics": stats,
    }

    # Print results
    print("\n" + "=" * 60)
    print("📊 Simulation Results")
    print("=" * 60)
    print(f"  Location: {params.location} (radius {params.radius_m}m)")
    print(f"  Time: {params.time_start} ~ {params.time_end}")
    if stats:
        print(f"  Vehicles inserted: {stats.get('vehicles_inserted', 'N/A')}")
        print(f"  Average speed: {stats.get('avg_speed_kmh', 'N/A')} km/h")
        print(f"  Average waiting time: {stats.get('avg_waiting_time_s', 'N/A')}s")
        print(f"  Average time loss: {stats.get('avg_time_loss_s', 'N/A')}s")
    print(f"\n  Generated files:")
    for k, v in files.items():
        print(f"    {k}: {v}")

    # Save result JSON
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved: {result_path}")

    return result


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        user_input = "강남역 퇴근시간 시뮬레이션해줘"

    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        # API key available -> agent mode (LLM auto-selects tools)
        from src.agent import run_agent
        print("🤖 Agent mode (Claude tool-calling)")
        run_agent(user_input, api_key)
    else:
        # No API key -> rule-based pipeline
        print("📏 Rule-based mode (no API key)")
        pipeline(user_input)
