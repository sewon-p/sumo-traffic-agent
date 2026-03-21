"""
LLM Tool-Calling Agent

Uses Claude API's tool-use feature to automatically select and execute
the necessary tools based on user's natural language requests.

Flow:
  User input -> Claude selects tool -> tool execution -> result returned to Claude
  -> Claude selects next tool or gives final response -> repeat
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import ANTHROPIC_API_KEY, OUTPUT_DIR, get_sumo_bin

# ──────────────────────────────────────────────
# Tool definitions: Available tools to inform Claude about
# ──────────────────────────────────────────────

TOOLS = [
    {
        "name": "search_location",
        "description": "Search for latitude/longitude coordinates by area name. Used to verify location before extracting OSM road network.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location_name": {
                    "type": "string",
                    "description": "Area name to search (e.g., 강남역, 테헤란로, 방배동)"
                }
            },
            "required": ["location_name"]
        }
    },
    {
        "name": "build_road_network",
        "description": "Download road network from OSM given an area name and radius, and generate a SUMO network file (.net.xml).",
        "input_schema": {
            "type": "object",
            "properties": {
                "location_name": {
                    "type": "string",
                    "description": "Area name to extract road network from"
                },
                "radius_m": {
                    "type": "number",
                    "description": "Radius from center in meters. Recommended: 300 for side streets, 500 for regular roads, 800 for highways.",
                    "default": 500
                }
            },
            "required": ["location_name"]
        }
    },
    {
        "name": "get_traffic_stats",
        "description": "Query Seoul traffic statistics data. Returns average traffic volume and speed by road type and time period. Used for statistics-based estimation when real data is unavailable.",
        "input_schema": {
            "type": "object",
            "properties": {
                "road_type": {
                    "type": "string",
                    "enum": ["고속도로", "도시고속도로", "간선도로", "보조간선", "이면도로"],
                    "description": "Road type"
                },
                "time_period": {
                    "type": "string",
                    "enum": ["출근(07-09)", "오전(09-12)", "점심(12-14)", "오후(14-18)", "퇴근(18-20)", "야간(20-24)", "심야(00-06)"],
                    "description": "Time period"
                },
                "day_type": {
                    "type": "string",
                    "enum": ["평일", "토요일", "일요일"],
                    "default": "평일"
                }
            },
            "required": ["road_type", "time_period"]
        }
    },
    {
        "name": "generate_simulation",
        "description": "Generate a set of SUMO simulation files (.rou.xml, .add.xml, .sumocfg). Takes network file path and traffic parameters as input.",
        "input_schema": {
            "type": "object",
            "properties": {
                "net_path": {
                    "type": "string",
                    "description": "Path to SUMO network file (.net.xml)"
                },
                "vehicles_per_hour": {
                    "type": "integer",
                    "description": "Number of vehicles to inject per hour"
                },
                "duration_seconds": {
                    "type": "integer",
                    "description": "Simulation duration in seconds",
                    "default": 3600
                },
                "passenger_ratio": {
                    "type": "number",
                    "description": "Passenger car ratio (0~1)",
                    "default": 0.85
                },
                "truck_ratio": {
                    "type": "number",
                    "description": "Truck ratio (0~1)",
                    "default": 0.10
                },
                "bus_ratio": {
                    "type": "number",
                    "description": "Bus ratio (0~1)",
                    "default": 0.05
                }
            },
            "required": ["net_path", "vehicles_per_hour"]
        }
    },
    {
        "name": "run_sumo",
        "description": "Run a SUMO simulation and return result statistics (average speed, waiting time, inserted vehicles, etc.).",
        "input_schema": {
            "type": "object",
            "properties": {
                "cfg_path": {
                    "type": "string",
                    "description": "Path to .sumocfg file"
                }
            },
            "required": ["cfg_path"]
        }
    },
    {
        "name": "query_topis_speed",
        "description": "Query real-time link-level average speed in Seoul via the TOPIS API. Returns a guidance message if no API key is available.",
        "input_schema": {
            "type": "object",
            "properties": {
                "road_name": {
                    "type": "string",
                    "description": "Road name to query (e.g., 강남대로, 테헤란로)"
                },
                "link_id": {
                    "type": "string",
                    "description": "TOPIS link ID (when known)"
                }
            },
            "required": []
        }
    },
    {
        "name": "load_csv_data",
        "description": "Load a user-provided traffic data CSV file. Automatically maps columns such as speed, traffic volume, and road name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "csv_path": {
                    "type": "string",
                    "description": "Path to CSV file"
                }
            },
            "required": ["csv_path"]
        }
    },
    {
        "name": "recommend_road",
        "description": "Given abstract traffic conditions (congestion level, road atmosphere, etc.), recommend representative Seoul roads matching those conditions. Used when the user makes abstract requests like 'congested commute road' or 'empty road' without specifying a location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "congestion_level": {
                    "type": "string",
                    "enum": ["매우혼잡", "혼잡", "보통", "원활", "한산"],
                    "description": "Congestion level"
                },
                "road_type": {
                    "type": "string",
                    "enum": ["고속도로", "도시고속도로", "간선도로", "보조간선", "이면도로", "any"],
                    "description": "Road type ('any' if unknown)",
                    "default": "any"
                },
                "time_context": {
                    "type": "string",
                    "description": "Time context (e.g., evening rush, morning rush, late night, weekend)"
                }
            },
            "required": ["congestion_level"]
        }
    },
    {
        "name": "find_similar_roads",
        "description": "When traffic data is unavailable for the requested road, search for roads with similar road type, lane count, and speed limit to estimate traffic volume and speed. A fallback tool.",
        "input_schema": {
            "type": "object",
            "properties": {
                "road_type": {
                    "type": "string",
                    "enum": ["고속도로", "도시고속도로", "간선도로", "보조간선", "이면도로"],
                    "description": "Road type"
                },
                "num_lanes": {
                    "type": "integer",
                    "description": "Number of lanes"
                },
                "speed_limit_kmh": {
                    "type": "number",
                    "description": "Speed limit (km/h)"
                },
                "district": {
                    "type": "string",
                    "description": "District name (e.g., 강남구, 서초구). Prioritizes roads in the same district."
                }
            },
            "required": ["road_type", "num_lanes", "speed_limit_kmh"]
        }
    },
    {
        "name": "validate_simulation",
        "description": "Compare simulation results against real data (speed, traffic volume), analyze errors, and assign a grade (A~F). Used for quality verification after simulation execution.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sim_speed_kmh": {
                    "type": "number",
                    "description": "Simulation average speed (km/h)"
                },
                "sim_volume_vph": {
                    "type": "integer",
                    "description": "Simulation inserted vehicle count (veh/h)"
                },
                "sim_waiting_s": {
                    "type": "number",
                    "description": "Simulation average waiting time (seconds)"
                },
                "sim_timeloss_s": {
                    "type": "number",
                    "description": "Simulation average time loss (seconds)"
                },
                "real_speed_kmh": {
                    "type": "number",
                    "description": "Real data average speed (km/h). Value retrieved from get_traffic_stats."
                },
                "real_volume_vph": {
                    "type": "integer",
                    "description": "Real data traffic volume (veh/h)"
                }
            },
            "required": ["sim_speed_kmh", "real_speed_kmh"]
        }
    },
    {
        "name": "calibrate_params",
        "description": "Calculate parameter adjustments based on validation error. If error exceeds 10%, suggest adjustments to traffic volume, sigma, tau, etc. Use the results to re-run generate_simulation for re-simulation.",
        "input_schema": {
            "type": "object",
            "properties": {
                "speed_error_pct": {
                    "type": "number",
                    "description": "Speed error percentage ((sim-real)/real*100)"
                },
                "current_vehicles_per_hour": {
                    "type": "integer",
                    "description": "Currently configured vehicles per hour"
                },
                "sim_waiting_s": {
                    "type": "number",
                    "description": "Current simulation average waiting time (seconds)",
                    "default": 0
                }
            },
            "required": ["speed_error_pct", "current_vehicles_per_hour"]
        }
    },
]

# ──────────────────────────────────────────────
# Tool execution functions
# ──────────────────────────────────────────────

def _tool_search_location(location_name: str) -> dict:
    from tools.osm_network import geocode_location
    lat, lon = geocode_location(location_name)
    return {"location": location_name, "lat": lat, "lon": lon}


def _tool_build_road_network(location_name: str, radius_m: float = 500) -> dict:
    from tools.osm_network import build_network
    output_dir = os.path.join(OUTPUT_DIR, location_name.replace(" ", "_"))
    net_path = build_network(
        location_name, radius_m, output_dir,
    )
    return {"net_path": net_path, "output_dir": output_dir}


def _tool_get_traffic_stats(road_type: str, time_period: str, day_type: str = "평일") -> dict:
    from traffic_data.traffic_db import get_traffic_stats
    return get_traffic_stats(road_type, time_period, day_type)


def _tool_generate_simulation(
    net_path: str,
    vehicles_per_hour: int,
    duration_seconds: int = 3600,
    passenger_ratio: float = 0.85,
    truck_ratio: float = 0.10,
    bus_ratio: float = 0.05,
) -> dict:
    from tools.sumo_generator import generate_all, TrafficDemand, SimulationConfig

    demand = TrafficDemand(
        total_vehicles_per_hour=vehicles_per_hour,
        passenger_ratio=passenger_ratio,
        truck_ratio=truck_ratio,
        bus_ratio=bus_ratio,
    )
    sim_config = SimulationConfig(begin_time=0, end_time=duration_seconds)
    output_dir = os.path.dirname(net_path)
    files = generate_all(net_path, output_dir, demand=demand, sim_config=sim_config)
    return files


def _tool_run_sumo(cfg_path: str) -> dict:
    import subprocess
    import re

    cmd = [get_sumo_bin(), "-c", cfg_path, "--duration-log.statistics", "--no-step-log"]
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
        elif "Duration:" in line and "Performance" not in line:
            m = re.search(r'Duration:\s*([\d.]+)', line)
            if m: stats["avg_trip_duration_s"] = float(m.group(1))

    if result.returncode != 0 and not stats:
        stats["error"] = result.stderr[:500]

    return stats


def _tool_query_topis_speed(road_name: str = None, link_id: str = None) -> dict:
    from tools.topis_api import query_realtime_traffic, query_road_speed
    location = road_name or link_id or ""
    if road_name:
        # Specific road name -> filter speed for that road
        return query_road_speed(location_name=road_name, road_name=road_name)
    return query_realtime_traffic(location_name=location)


def _tool_recommend_road(congestion_level: str, road_type: str = "any", time_context: str = "") -> dict:
    """Recommend representative roads matching abstract conditions."""

    # Speed range by congestion level (peak hour basis, km/h)
    congestion_speed = {
        "매우혼잡": (0, 20),
        "혼잡": (15, 30),
        "보통": (25, 45),
        "원활": (40, 70),
        "한산": (60, 100),
    }

    from tools.similar_road import ROAD_PROFILES

    speed_range = congestion_speed.get(congestion_level, (15, 40))

    candidates = []
    for road in ROAD_PROFILES:
        # Road type filter
        if road_type != "any" and road.road_type != road_type:
            continue
        # Check if peak speed falls within congestion range
        if speed_range[0] <= road.peak_speed_kmh <= speed_range[1]:
            candidates.append(road)

    # If no exact match, add closest roads
    if not candidates:
        target_speed = (speed_range[0] + speed_range[1]) / 2
        all_roads = ROAD_PROFILES if road_type == "any" else [r for r in ROAD_PROFILES if r.road_type == road_type]
        all_roads_sorted = sorted(all_roads, key=lambda r: abs(r.peak_speed_kmh - target_speed))
        candidates = all_roads_sorted[:3]

    # Format results
    recommendations = []
    for road in candidates[:3]:
        recommendations.append({
            "name": road.name,
            "district": road.district,
            "road_type": road.road_type,
            "num_lanes": road.num_lanes,
            "speed_limit_kmh": road.speed_limit_kmh,
            "peak_speed_kmh": road.peak_speed_kmh,
            "peak_volume_vph": road.peak_volume_vph,
            "description": f"{road.district} {road.name}, {road.num_lanes} lanes, "
                          f"speed limit {road.speed_limit_kmh}km/h, "
                          f"peak avg {road.peak_speed_kmh}km/h",
        })

    return {
        "congestion_level": congestion_level,
        "road_type_filter": road_type,
        "recommendations": recommendations,
        "note": "Select one of the recommended roads to generate a simulation with its actual geometry.",
    }


def _tool_validate_simulation(
    sim_speed_kmh: float,
    real_speed_kmh: float,
    sim_volume_vph: int = 0,
    sim_waiting_s: float = 0,
    sim_timeloss_s: float = 0,
    real_volume_vph: int = 0,
) -> dict:
    from src.validator import validate, generate_report
    sim_stats = {
        "avg_speed_kmh": sim_speed_kmh,
        "vehicles_inserted": sim_volume_vph,
        "avg_waiting_time_s": sim_waiting_s,
        "avg_time_loss_s": sim_timeloss_s,
    }
    result = validate(sim_stats, real_speed_kmh=real_speed_kmh, real_volume_vph=real_volume_vph or None)
    report = generate_report(result)
    return {
        "grade": result.grade,
        "speed_error_pct": result.speed_error_pct,
        "volume_error_pct": result.volume_error_pct,
        "issues": result.issues,
        "suggestions": result.suggestions,
        "report": report,
    }


def _tool_calibrate_params(
    speed_error_pct: float,
    current_vehicles_per_hour: int,
    sim_waiting_s: float = 0,
) -> dict:
    from src.validator import ValidationResult, calibration_suggestion
    vr = ValidationResult(speed_error_pct=speed_error_pct, sim_avg_waiting_s=sim_waiting_s)
    suggestion = calibration_suggestion(vr)

    # Calculate specific new parameters
    if suggestion.get("status") == "needs_adjustment":
        adj = suggestion["adjustments"]
        change_pct = adj.get("vehicles_per_hour_change_pct", 0)
        new_vph = int(current_vehicles_per_hour * (1 + change_pct / 100))
        suggestion["new_vehicles_per_hour"] = new_vph

    return suggestion


def _tool_find_similar_roads(road_type: str, num_lanes: int, speed_limit_kmh: float, district: str = "") -> list:
    from tools.similar_road import find_similar_roads
    return find_similar_roads(road_type, num_lanes, speed_limit_kmh, district)


def _tool_load_csv_data(csv_path: str) -> dict:
    from tools.topis_api import load_traffic_csv
    rows = load_traffic_csv(csv_path)
    return {
        "status": "success",
        "total_rows": len(rows),
        "sample": rows[:5],  # Return only first 5 rows (token saving)
        "columns": list(rows[0].keys()) if rows else [],
    }


# Tool name -> execution function mapping
TOOL_HANDLERS = {
    "search_location": lambda **kw: _tool_search_location(**kw),
    "build_road_network": lambda **kw: _tool_build_road_network(**kw),
    "get_traffic_stats": lambda **kw: _tool_get_traffic_stats(**kw),
    "generate_simulation": lambda **kw: _tool_generate_simulation(**kw),
    "run_sumo": lambda **kw: _tool_run_sumo(**kw),
    "query_topis_speed": lambda **kw: _tool_query_topis_speed(**kw),
    "load_csv_data": lambda **kw: _tool_load_csv_data(**kw),
    "recommend_road": lambda **kw: _tool_recommend_road(**kw),
    "find_similar_roads": lambda **kw: _tool_find_similar_roads(**kw),
    "validate_simulation": lambda **kw: _tool_validate_simulation(**kw),
    "calibrate_params": lambda **kw: _tool_calibrate_params(**kw),
}


# ──────────────────────────────────────────────
# Agent main loop
# ──────────────────────────────────────────────

AGENT_SYSTEM_PROMPT = """You are a traffic simulation expert agent.
You receive natural language requests from the user and automatically generate and run SUMO traffic simulations.

## Request Type Determination

User requests fall into two types:

### A. Specific request (location is specified)
Examples: "Simulate 강남역 during evening rush hour", "Simulate accident on 올림픽대로 여의도 section"
-> Proceed directly in the work order

### B. Abstract request (only conditions, no location)
Examples: "Moderately congested commute road", "Empty road", "Heavily jammed morning commute"
-> You must follow the steps below:
1. Use the recommend_road tool to search for representative road candidates matching the conditions
2. Present 2~3 candidates to the user with a brief description of each road's characteristics
   Example: "Here are roads matching your conditions:
   1) 강남대로 (강남구, 5 lanes, peak 18km/h) - Seoul's representative congested arterial
   2) 테헤란로 (강남구, 4 lanes, peak 20km/h) - IT valley evening congestion
   3) 종로 (종로구, 4 lanes, peak 16km/h) - Downtown congestion
   Which road would you like to proceed with?"
3. Once the user selects, generate simulation with that road's actual geometry
4. If the user responds with "any" or "number 1", proceed with the first recommendation

## Work Order (after location is determined)

1. Query traffic statistics for the road/time period with get_traffic_stats -> obtain real_speed_kmh, real_volume_vph
2. Attempt real-time data query with query_topis_speed (use statistics if it fails)
3. If no data for the road, use find_similar_roads to search similar roads for estimation
4. Extract OSM road network with build_road_network -> generate SUMO network
5. Generate simulation files reflecting traffic demand with generate_simulation
6. Run simulation with run_sumo
7. Compare simulation speed vs real data speed with validate_simulation
8. If grade is C or below (error > 20%):
   a. Calculate parameter adjustments with calibrate_params
   b. Re-run generate_simulation with adjusted traffic volume
   c. Re-run run_sumo -> re-validate with validate_simulation
   d. Repeat up to 2 more times (3 attempts total)
9. Report final results to the user (grade, error, key statistics)

## Decision Criteria

Congestion level interpretation:
- "Gridlocked/standstill" -> Very congested (peak 0~15km/h)
- "Congested/jammed" -> Congested (15~25km/h)
- "Moderately congested/normal" -> Normal (25~40km/h)
- "Flowing smoothly" -> Smooth (40~60km/h)
- "Empty/deserted" -> Light (60~100km/h)

Road types:
- Boulevard/arterial: road_type="간선도로", radius 500m
- Side street/alley: road_type="이면도로", radius 300m
- Highway/올림픽대로: road_type="도시고속도로", radius 800m

Time periods:
- Evening rush/commute home: time_period="퇴근(18-20)"
- Morning rush/commute to work: time_period="출근(07-09)"

## When Data is Insufficient
- Use find_similar_roads to estimate based on similar road data
- When using estimated values, always inform the user that it is "based on similar road estimation"
- Also provide uncertainty ranges (e.g., "traffic volume 400±100 veh/h")

Always use tools to make data-driven parameter decisions.
Do not guess; verify with tools."""


def run_agent(
    user_input: str,
    provider: str = None,
    api_key: str = None,
    verbose: bool = True,
    messages: list = None,
) -> dict:
    """
    Agent main loop (multi-LLM support).
    Works with tool-calling on Gemini/Claude/GPT.

    Args:
        user_input: User input
        provider: "gemini", "claude", "gpt" (None for auto-detection)
        api_key: API key (None to use environment variable)
        verbose: Print execution progress
        messages: Existing conversation history (for conversation continuity)
    """
    from src.llm_client import create_client

    try:
        client = create_client(provider=provider, api_key=api_key)
    except ValueError as e:
        print(f"⚠ {e}")
        return {"error": str(e)}
    except ImportError as e:
        print(f"⚠ Please install the required package: {e}")
        return {"error": str(e)}

    if messages is None:
        messages = []
    messages.append({"role": "user", "content": user_input})

    if verbose:
        print(f"\n{'='*60}")
        print(f"🤖 Agent ({client.provider_name()})")
        print(f"   Input: {user_input}")
        print(f"{'='*60}")

    max_iterations = 15
    final_result = {}

    for i in range(max_iterations):
        try:
            response = client.chat(messages, AGENT_SYSTEM_PROMPT, TOOLS)
        except Exception as e:
            print(f"❌ LLM call error: {e}")
            final_result["error"] = str(e)
            break

        # Save assistant message
        if response.get("_raw_content"):
            messages.append({"role": "assistant", "content": response["_raw_content"]})
        elif response["text"]:
            messages.append({"role": "assistant", "content": response["text"]})

        # If no tool calls, this is the final response
        if not response["tool_calls"]:
            if verbose and response["text"]:
                print(f"\n📋 Agent response:\n{response['text']}")
            final_result["agent_response"] = response["text"]
            break

        # Execute tools
        tool_results = []
        for tc in response["tool_calls"]:
            tool_name = tc["name"]
            tool_input = tc["input"]

            if verbose:
                print(f"\n🔧 Tool [{i+1}]: {tool_name}")
                print(f"   Input: {json.dumps(tool_input, ensure_ascii=False)[:150]}")

            handler = TOOL_HANDLERS.get(tool_name)
            if handler is None:
                result = {"error": f"Unknown tool: {tool_name}"}
            else:
                try:
                    result = handler(**tool_input)
                except Exception as e:
                    result = {"error": str(e)}

            result_str = json.dumps(result, ensure_ascii=False)
            if verbose:
                print(f"   Result: {result_str[:200]}{'...' if len(result_str) > 200 else ''}")

            tr = client.format_tool_result(tc["id"], result_str)
            tr["_tool_name"] = tool_name  # For Gemini
            tool_results.append(tr)

            final_result[tool_name] = result

        messages.append({"role": "user", "content": tool_results})

    return final_result


if __name__ == "__main__":
    user_input = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "강남역 퇴근시간 시뮬레이션해줘"
    result = run_agent(user_input)
