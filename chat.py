#!/usr/bin/env python3
"""
LLM-Powered Traffic Simulation Agent - Interactive Interface

Usage:
  python chat.py                      # Rule-based mode
  ANTHROPIC_API_KEY=sk-... python chat.py  # Agent mode (Claude tool-calling)
"""

import os
import sys
import json
import readline  # Arrow key and history support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import SUMO_BIN, get_sumo_bin, get_sumo_gui_bin

ENV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")


def load_env():
    """Load project .env file"""
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, val = line.split("=", 1)
                    os.environ.setdefault(key.strip(), val.strip())


def _save_env(key: str, value: str):
    """Save key=value to .env (update if key already exists)."""
    lines = []
    found = False
    if os.path.exists(ENV_PATH):
        with open(ENV_PATH) as f:
            for line in f:
                if line.strip().startswith(f"{key}="):
                    lines.append(f"{key}={value}\n")
                    found = True
                else:
                    lines.append(line)
    if not found:
        lines.append(f"{key}={value}\n")
    with open(ENV_PATH, "w") as f:
        f.writelines(lines)
    os.environ[key] = value


def _detect_cli():
    """List installed CLI tools"""
    import shutil
    found = []
    for name in ["gemini", "claude", "codex"]:
        if shutil.which(name):
            found.append(name)
    return found


def _detect_llm():
    """Detect configured LLM (API key or CLI)"""
    # Check mode set in .env
    mode = os.environ.get("LLM_MODE", "")
    provider = os.environ.get("LLM_PROVIDER", "")
    if mode and provider:
        return provider, mode  # ("gemini", "cli") or ("gemini", "api")

    # Check API keys
    for name, key in [("gemini", "GEMINI_API_KEY"), ("claude", "ANTHROPIC_API_KEY"), ("gpt", "OPENAI_API_KEY")]:
        if os.environ.get(key):
            return name, "api"

    # Check CLI
    clis = _detect_cli()
    if clis:
        return clis[0], "cli"

    return None, None


def setup_wizard():
    """Initial setup wizard. Skipped if already configured."""
    provider, mode = _detect_llm()
    if provider and mode:
        return provider, mode

    print()
    print("=" * 60)
    print("  Initial Setup")
    print("=" * 60)

    # 1) Connection method
    clis = _detect_cli()
    print("\n  Select LLM connection method:\n")
    print("  1) CLI mode (locally installed tools)")
    if clis:
        print(f"     Found: {', '.join(clis)}")
    else:
        print("     (No CLI installed)")
    print("  2) Enter API key")
    print()

    choice = input("  Selection (1/2): ").strip()

    if choice == "1" and clis:
        # CLI selection
        print()
        for i, name in enumerate(clis, 1):
            print(f"  {i}) {name}")
        print()
        idx = input(f"  Select LLM (1~{len(clis)}): ").strip()
        try:
            selected = clis[int(idx) - 1]
        except (ValueError, IndexError):
            selected = clis[0]

        _save_env("LLM_MODE", "cli")
        _save_env("LLM_PROVIDER", selected)
        print(f"\n  ✓ {selected} (CLI) setup complete")
        return selected, "cli"

    else:
        # API key input
        print()
        print("  Enter API key (Enter = skip):\n")

        key_map = [
            ("gemini", "GEMINI_API_KEY", "Gemini"),
            ("claude", "ANTHROPIC_API_KEY", "Claude"),
            ("gpt", "OPENAI_API_KEY", "GPT"),
        ]

        selected = None
        for name, env_key, label in key_map:
            key = input(f"  {label} API Key: ").strip()
            if key:
                _save_env(env_key, key)
                if not selected:
                    selected = name

        if selected:
            _save_env("LLM_MODE", "api")
            _save_env("LLM_PROVIDER", selected)
            print(f"\n  ✓ {selected} (API) setup complete. Saved to .env.")
            return selected, "api"
        else:
            print("\n  No key entered -> running in rule-based mode.")
            return None, None


def print_banner(provider, mode):
    print()
    print("=" * 60)
    print("  🚗 LLM Traffic Simulation Agent")
    print("=" * 60)

    if provider:
        icon = "🖥️" if mode == "cli" else "🔑"
        print(f"  LLM: {icon} {provider} ({mode})")
    else:
        print("  LLM: 📏 Rule-based")

    print(f"  SUMO: ✓")
    print(f"  Traffic: {'✓ Seoul API' if os.environ.get('TOPIS_API_KEY') else 'Stats-based'}")
    print()
    print("  Commands:")
    print("    /settings          Change LLM settings")
    print("    /status            Check settings")
    print("    /traffic 강남역      Query traffic info")
    print("    /view 강남역         SUMO-GUI")
    print("    /quit              Exit")
    print("=" * 60)
    print()


def cmd_status():
    """Print current configuration status"""
    provider, mode = _detect_llm()
    print("\n📋 Current settings:")
    print(f"  SUMO: {SUMO_BIN or 'Not installed'}")
    print(f"  LLM: {provider} ({mode})" if provider and mode else "  LLM: None (rule-based)")
    for name, key in [("Gemini", "GEMINI_API_KEY"), ("Claude", "ANTHROPIC_API_KEY"), ("GPT", "OPENAI_API_KEY")]:
        has = "✓" if os.environ.get(key) else "✗"
        print(f"    {has} {name}")
    print(f"  Seoul API: {'✓' if os.environ.get('TOPIS_API_KEY') else '✗'}")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    if os.path.exists(output_dir):
        sims = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if sims:
            print(f"  Simulations: {', '.join(sims)}")
    print()


def cmd_traffic(location: str):
    """Query real-time traffic information"""
    from tools.topis_api import query_realtime_traffic

    print(f"\n🔍 Querying real-time traffic info for {location}...")
    result = query_realtime_traffic(location)

    if result["status"] != "success":
        print(f"  ⚠ {result.get('message', 'Query failed')}")
        return

    print(f"\n  📍 {result['area_name']}")
    print(f"  🕐 {result['timestamp']}")
    print(f"  🚦 Overall: {result['traffic_index']} (avg {result['avg_speed_kmh']} km/h)")
    print(f"  💬 {result['traffic_message']}")

    if result.get("road_summary"):
        print(f"\n  Road-level status:")
        for road in result["road_summary"]:
            print(f"    {road['road_name']:<12} {road['avg_speed_kmh']:>5.1f} km/h  "
                  f"({road['link_count']} segments, {road['total_distance_m']}m)")
    print()


def run_rule_based(user_input: str):
    """Run rule-based pipeline"""
    from src.llm_parser import parse_user_input
    from tools.osm_network import build_network
    from tools.sumo_generator import build_vtypes_from_ft, generate_all, TrafficDemand, SimulationConfig
    from tools.topis_api import query_realtime_traffic
    from src.validator import validate, generate_report

    # 1) Extract parameters
    print("\n[1/5] Extracting parameters...")
    params = parse_user_input(user_input)
    print(f"  Location: {params.location}")
    print(f"  Time: {params.time_start} ~ {params.time_end}")
    print(f"  Traffic volume: {params.vehicles_per_hour} veh/h")

    ft = {}
    if params.notes:
        try:
            ft = json.loads(params.notes)
        except (TypeError, ValueError, json.JSONDecodeError):
            ft = {}

    # 2) Query real-time traffic data
    print(f"\n[2/5] Querying real-time traffic data...")
    real_data = query_realtime_traffic(params.location)
    real_speed = None

    if real_data["status"] == "success":
        real_speed = real_data["avg_speed_kmh"]
        print(f"  ✓ {real_data['area_name']}: {real_speed} km/h ({real_data['traffic_index']})")
        if real_data.get("road_summary"):
            for rd in real_data["road_summary"][:3]:
                print(f"    {rd['road_name']}: {rd['avg_speed_kmh']} km/h")
    else:
        print(f"  ⚠ No real-time data available -> using statistics-based estimation")

    # 3) Build road network
    print(f"\n[3/5] Building {params.location} road network...")
    output_dir = os.path.join("output", params.location.replace(" ", "_"))
    net_path = build_network(
        params.location,
        params.radius_m,
        output_dir,
        speed_limit_kmh=params.speed_limit_kmh,
    )

    # 4) Generate simulation files
    print(f"\n[4/5] Generating SUMO simulation files...")
    demand = TrafficDemand(
        total_vehicles_per_hour=params.vehicles_per_hour,
        passenger_ratio=params.passenger_ratio,
        truck_ratio=params.truck_ratio,
        bus_ratio=params.bus_ratio,
    )
    h1 = int(params.time_start.split(":")[0])
    h2 = int(params.time_end.split(":")[0])
    duration = max((h2 - h1) * 3600, 3600)
    sim_config = SimulationConfig(begin_time=0, end_time=duration)

    vtypes = build_vtypes_from_ft(ft, speed_limit_kmh=params.speed_limit_kmh)
    files = generate_all(net_path, output_dir, demand=demand, vtypes=vtypes, sim_config=sim_config)

    # 5) Run simulation
    print(f"\n[5/5] Running simulation...")
    import subprocess, re
    cmd = [get_sumo_bin(), "-c", files["cfg"], "--duration-log.statistics", "--no-step-log"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    output = result.stdout + result.stderr

    stats = {}
    for line in output.split("\n"):
        line = line.strip()
        if "Inserted:" in line:
            m = re.search(r'Inserted:\s*(\d+)', line)
            if m: stats["vehicles_inserted"] = int(m.group(1))
        elif "Speed:" in line and "Statistics" not in line:
            m = re.search(r'Speed:\s*([\d.]+)', line)
            if m:
                stats["avg_speed_ms"] = float(m.group(1))
                stats["avg_speed_kmh"] = round(float(m.group(1)) * 3.6, 1)
        elif "WaitingTime:" in line:
            m = re.search(r'WaitingTime:\s*([\d.]+)', line)
            if m: stats["avg_waiting_time_s"] = float(m.group(1))
        elif "TimeLoss:" in line:
            m = re.search(r'TimeLoss:\s*([\d.]+)', line)
            if m: stats["avg_time_loss_s"] = float(m.group(1))

    # Print results
    print("\n" + "=" * 60)
    print("📊 Simulation Results")
    print("=" * 60)
    print(f"  Location: {params.location} (radius {params.radius_m}m)")
    print(f"  Time: {params.time_start} ~ {params.time_end}")
    print(f"  Vehicles inserted: {stats.get('vehicles_inserted', 'N/A')}")
    print(f"  Average speed: {stats.get('avg_speed_kmh', 'N/A')} km/h")
    print(f"  Average waiting time: {stats.get('avg_waiting_time_s', 'N/A')}s")

    # Validation
    if real_speed and stats.get("avg_speed_kmh"):
        print(f"\n📋 Validation (vs real-time data)")
        v = validate(stats, real_speed_kmh=real_speed)
        report = generate_report(v)
        print(report)

    print(f"\n  Generated files: {output_dir}/")
    print()

    # Save results
    result_data = {
        "input": user_input,
        "params": params.to_dict(),
        "files": files,
        "statistics": stats,
        "real_data": real_data if real_data["status"] == "success" else None,
    }
    with open(os.path.join(output_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)


def run_agent_mode(user_input: str, messages: list, provider: str = None):
    """Run agent mode (multi-LLM support)"""
    from src.agent import run_agent
    run_agent(user_input, provider=provider, verbose=True, messages=messages)


def main():
    load_env()


    # Initial setup
    provider, mode = setup_wizard()
    print_banner(provider, mode)

    messages = []

    while True:
        try:
            user_input = input("🗣️  ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Exiting.")
            break

        if not user_input:
            continue

        # Commands
        if user_input.lower() in ("/quit", "/exit", "/q"):
            print("\n👋 Exiting.")
            break
        elif user_input.lower() in ("/settings", "/setup"):
            # Reset settings and run again
            for k in ["LLM_MODE", "LLM_PROVIDER"]:
                os.environ.pop(k, None)
            provider, mode = setup_wizard()
            messages = []
            print_banner(provider, mode)
            continue
        elif user_input.lower() == "/status":
            cmd_status()
            continue
        elif user_input.lower().startswith("/traffic"):
            parts = user_input.split(maxsplit=1)
            location = parts[1] if len(parts) > 1 else "강남역"
            cmd_traffic(location)
            continue
        elif user_input.lower().startswith("/view"):
            parts = user_input.split(maxsplit=1)
            location = parts[1].strip() if len(parts) > 1 else "강남역"
            safe = location.replace(" ", "_")
            cfg = os.path.join("output", safe, f"{safe}.sumocfg")
            if os.path.exists(cfg):
                sumo_gui = get_sumo_gui_bin()
                import subprocess
                subprocess.Popen([sumo_gui, "-c", cfg])
                print(f"  SUMO-GUI launched: {cfg}")
            else:
                print(f"  ⚠ File not found: {cfg}")
            continue
        elif user_input.startswith("/"):
            print("  /settings /status /traffic /view /quit")
            continue

        # Run simulation
        try:
            if provider:
                run_agent_mode(user_input, messages, provider=provider)
            else:
                run_rule_based(user_input)
        except KeyboardInterrupt:
            print("\n\n⚠ Interrupted")
        except Exception as e:
            print(f"\n❌ {e}")


if __name__ == "__main__":
    main()
