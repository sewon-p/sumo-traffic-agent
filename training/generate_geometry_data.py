#!/usr/bin/env python3
"""
Geometry training data generator.

Generates geometry parameter training pairs from road structure facility
design rules and traffic engineering design standards.

Usage:
    python -m training.generate_geometry_data
"""

import json
import os
import random
import itertools

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")

# ──────────────────────────────────────────────
# Road design standard tables (MOLIT road structure facility design rules)
# ──────────────────────────────────────────────

# Minimum curve radius by design speed (m)
MIN_CURVE_RADIUS = {
    120: 710, 100: 460, 80: 280, 60: 150, 50: 90, 40: 60, 30: 30, 20: 15,
}

# Lane width by road class (m)
LANE_WIDTH = {
    "고속도로": 3.6, "도시고속": 3.5, "간선": 3.25, "보조간선": 3.0, "집산": 3.0, "국지": 2.75,
}

# Design speed range by road class
DESIGN_SPEED_RANGE = {
    "고속도로": [100, 120], "도시고속": [80, 100],
    "간선": [60, 80], "보조간선": [50, 60],
    "집산": [40, 50], "국지": [30, 40],
}

# Network type definitions
NETWORK_TYPES = {
    "straight": {"desc": "직선 구간", "curve_radius": None},
    "curve": {"desc": "곡선 구간", "curve_radius": "설계속도 기반"},
    "intersection": {"desc": "교차로", "curve_radius": None},
    "ramp": {"desc": "진출입 램프", "curve_radius": "설계속도 기반"},
    "roundabout": {"desc": "회전교차로", "curve_radius": "15~30m"},
    "weaving": {"desc": "엇갈림 구간", "curve_radius": None},
}


def generate_straight_scenarios():
    """Generate straight road scenarios."""
    scenarios = []
    configs = [
        # (lanes(one direction), road_class, length_range, description_keywords)
        (1, "국지", (200, 500), ["좁은", "주택가", "이면"]),
        (2, "집산", (300, 800), ["2차선", "소로"]),
        (2, "보조간선", (500, 1500), ["왕복4차선", "보조간선"]),
        (3, "간선", (500, 2000), ["왕복6차선", "간선"]),
        (4, "간선", (500, 2000), ["왕복8차선", "대로"]),
        (5, "간선", (800, 2000), ["왕복10차선", "대로"]),
        (3, "도시고속", (1000, 3000), ["도시고속", "자동차전용"]),
        (4, "고속도로", (1000, 5000), ["고속도로"]),
    ]

    for lanes, grade, (len_min, len_max), keywords in configs:
        speed_range = DESIGN_SPEED_RANGE[grade]
        speed_limit = random.choice(speed_range)
        length = random.randint(len_min // 100, len_max // 100) * 100

        flow_type = "연속류" if grade in ("고속도로", "도시고속") else "단속류"
        total_lanes = lanes * 2

        # Natural language prompt variants
        prompts = [
            f"직선 {grade} 왕복 {total_lanes}차선 {length}m",
            f"{random.choice(keywords)} 직선 도로 {length}m 시뮬레이션",
            f"왕복 {total_lanes}차선 직선 {flow_type} {length}m",
        ]

        entry_exit = 1 if flow_type == "연속류" else random.choice([2, 3, 4])

        params = {
            "network_mode": "generate",
            "network_type": "straight",
            "length_m": length,
            "curve_radius_m": None,
            "lanes": lanes,
            "total_lanes": total_lanes,
            "speed_limit_kmh": speed_limit,
            "lane_width_m": LANE_WIDTH[grade],
            "road_grade": grade,
            "flow_type": flow_type,
            "entry_points": entry_exit,
            "exit_points": entry_exit,
            "reasoning": (
                f"직선 {grade}({flow_type}). "
                f"설계속도 {speed_limit}km/h, 편도 {lanes}차로(폭 {LANE_WIDTH[grade]}m). "
                f"{'유입/유출 각 1개(연속류)' if flow_type == '연속류' else f'교차로 {entry_exit}개(단속류)'}."
            ),
        }

        for p in prompts:
            scenarios.append({"prompt": p, "params": params, "meta": {"type": "straight", "grade": grade}})

    return scenarios


def generate_curve_scenarios():
    """Generate curved road scenarios."""
    scenarios = []

    curve_descriptions = [
        ("급커브", 0.8),   # 0.8x minimum radius
        ("커브", 1.2),     # 1.2x minimum radius
        ("완만한 곡선", 2.5),  # 2.5x minimum radius
        ("아주 완만한 곡선", 5.0),
    ]

    for grade in ["간선", "보조간선", "도시고속", "고속도로"]:
        speed_range = DESIGN_SPEED_RANGE[grade]
        for speed_limit in speed_range:
            min_r = MIN_CURVE_RADIUS[speed_limit]

            for curve_desc, r_factor in curve_descriptions:
                radius = int(min_r * r_factor)
                lanes = random.choice([2, 3, 4]) if grade != "고속도로" else random.choice([3, 4])
                total_lanes = lanes * 2
                length = random.choice([300, 500, 800, 1000, 1500])
                flow_type = "연속류" if grade in ("고속도로", "도시고속") else "단속류"

                prompts = [
                    f"{curve_desc} {length}m {flow_type} 왕복 {total_lanes}차선",
                    f"{curve_desc} 도로 {length}m 시뮬레이션",
                    f"반경 {radius}m 곡선 {grade} {length}m",
                    f"왕복 {total_lanes}차선 {curve_desc}부 {length}m {flow_type}",
                ]

                params = {
                    "network_mode": "generate",
                    "network_type": "curve",
                    "length_m": length,
                    "curve_radius_m": radius,
                    "lanes": lanes,
                    "total_lanes": total_lanes,
                    "speed_limit_kmh": speed_limit,
                    "lane_width_m": LANE_WIDTH[grade],
                    "road_grade": grade,
                    "flow_type": flow_type,
                    "entry_points": 1,
                    "exit_points": 1,
                    "reasoning": (
                        f"{curve_desc}({grade}, {flow_type}). "
                        f"설계속도 {speed_limit}km/h → 최소곡선반경 {min_r}m, "
                        f"실제 반경 {radius}m(최소의 {r_factor}배). "
                        f"편도 {lanes}차로."
                    ),
                }

                for p in prompts:
                    scenarios.append({"prompt": p, "params": params, "meta": {"type": "curve", "grade": grade}})

    return scenarios


def generate_intersection_scenarios():
    """Generate intersection scenarios."""
    scenarios = []

    configs = [
        # (main_road_lanes, sub_road_lanes, intersection_type, keywords)
        (2, 1, "비신호", ["비신호 교차로", "소규모 교차로"]),
        (2, 2, "신호", ["신호 교차로", "사거리"]),
        (3, 2, "신호", ["큰 사거리", "간선 교차로"]),
        (4, 3, "신호", ["대형 교차로", "주요 교차로"]),
        (5, 3, "신호", ["대규모 교차로"]),
    ]

    for main_lanes, sub_lanes, signal_type, keywords in configs:
        approach_length = random.choice([200, 300, 500])
        main_total = main_lanes * 2
        sub_total = sub_lanes * 2
        main_limit = 50 if main_lanes <= 3 else 60
        sub_limit = 40 if sub_lanes <= 2 else 50

        prompts = [
            f"{random.choice(keywords)} 주도로 왕복{main_total}차선 부도로 왕복{sub_total}차선",
            f"{signal_type} 사거리 {main_total}차선×{sub_total}차선",
            f"4방향 {signal_type} 교차로 시뮬레이션",
        ]

        params = {
            "network_mode": "generate",
            "network_type": "intersection",
            "length_m": approach_length,
            "curve_radius_m": None,
            "lanes": main_lanes,
            "total_lanes": main_total,
            "sub_lanes": sub_lanes,
            "sub_total_lanes": sub_total,
            "speed_limit_kmh": main_limit,
            "sub_speed_limit_kmh": sub_limit,
            "lane_width_m": 3.25,
            "road_grade": "간선" if main_lanes >= 3 else "보조간선",
            "flow_type": "단속류",
            "signal_type": signal_type,
            "entry_points": 4,
            "exit_points": 4,
            "approach_length_m": approach_length,
            "reasoning": (
                f"4방향 {signal_type} 교차로. "
                f"주도로 편도{main_lanes}차로(제한{main_limit}), "
                f"부도로 편도{sub_lanes}차로(제한{sub_limit}). "
                f"접근로 {approach_length}m."
            ),
        }

        for p in prompts:
            scenarios.append({"prompt": p, "params": params, "meta": {"type": "intersection"}})

    return scenarios


def generate_ramp_scenarios():
    """Generate ramp/merge section scenarios."""
    scenarios = []

    configs = [
        (3, "진입램프", "합류"),
        (3, "진출램프", "분류"),
        (4, "진입램프", "합류"),
        (4, "진출램프", "분류"),
    ]

    for main_lanes, ramp_type, merge_type in configs:
        total = main_lanes * 2
        length = random.choice([500, 800, 1000])
        ramp_length = random.choice([200, 300, 400])

        prompts = [
            f"고속도로 {ramp_type} {merge_type} 구간 시뮬레이션",
            f"왕복 {total}차선 고속도로 {merge_type}부 {length}m",
            f"{ramp_type} 구간 본선 {total}차선",
        ]

        params = {
            "network_mode": "generate",
            "network_type": "ramp",
            "length_m": length,
            "ramp_length_m": ramp_length,
            "curve_radius_m": None,
            "lanes": main_lanes,
            "total_lanes": total,
            "ramp_lanes": 1,
            "speed_limit_kmh": 100,
            "ramp_speed_limit_kmh": 60,
            "lane_width_m": 3.6,
            "road_grade": "고속도로",
            "flow_type": "연속류",
            "merge_type": merge_type,
            "entry_points": 2 if merge_type == "합류" else 1,
            "exit_points": 1 if merge_type == "합류" else 2,
            "reasoning": (
                f"고속도로 {ramp_type}({merge_type}구간). "
                f"본선 편도{main_lanes}차로(100km/h), "
                f"램프 1차로({ramp_length}m, 60km/h)."
            ),
        }

        for p in prompts:
            scenarios.append({"prompt": p, "params": params, "meta": {"type": "ramp"}})

    return scenarios


def generate_roundabout_scenarios():
    """Generate roundabout scenarios."""
    scenarios = []

    configs = [
        (1, 20, "소형"),
        (1, 30, "일반"),
        (2, 40, "대형"),
    ]

    for circ_lanes, radius, size in configs:
        approach_lanes = random.choice([1, 2])
        arms = random.choice([3, 4, 5])
        approach_length = random.choice([150, 200, 300])

        prompts = [
            f"{size} 회전교차로 {arms}방향",
            f"로터리 {arms}방향 시뮬레이션",
            f"반경 {radius}m 회전교차로 접근로 {approach_lanes}차로",
        ]

        params = {
            "network_mode": "generate",
            "network_type": "roundabout",
            "length_m": approach_length,
            "curve_radius_m": radius,
            "lanes": approach_lanes,
            "total_lanes": approach_lanes * 2,
            "circulatory_lanes": circ_lanes,
            "arms": arms,
            "speed_limit_kmh": 30,
            "lane_width_m": 3.25,
            "road_grade": "보조간선",
            "flow_type": "단속류",
            "entry_points": arms,
            "exit_points": arms,
            "approach_length_m": approach_length,
            "reasoning": (
                f"{size} 회전교차로(반경{radius}m). "
                f"{arms}방향, 회전차로 {circ_lanes}차로, "
                f"접근로 편도{approach_lanes}차로({approach_length}m). "
                f"회전부 제한속도 30km/h."
            ),
        }

        for p in prompts:
            scenarios.append({"prompt": p, "params": params, "meta": {"type": "roundabout"}})

    return scenarios


def add_traffic_to_geometry(scenarios: list) -> list:
    """Add traffic parameters to geometry scenarios.

    For cross-learning between geometry and traffic parameters,
    combines the same geometry with various traffic conditions.
    """
    time_conditions = [
        ("출근시간", 0.85, 0.65),   # (time_period, V/C, sigma)
        ("퇴근시간", 0.80, 0.60),
        ("한산한 시간", 0.25, 0.30),
        ("보통 시간", 0.50, 0.45),
    ]

    enriched = []
    for s in scenarios:
        p = s["params"]
        speed_limit = p.get("speed_limit_kmh", 60)
        lanes = p.get("lanes", 2)
        grade = p.get("road_grade", "간선")
        flow_type = p.get("flow_type", "단속류")

        capacity_per_lane = {
            "고속도로": 2200, "도시고속": 2000,
            "간선": 1800, "보조간선": 1600,
            "집산": 1200, "국지": 800,
        }.get(grade, 1400)

        for time_label, vc, sigma in time_conditions:
            free_speed = speed_limit * 0.9
            # Apply curve speed reduction
            curve_r = p.get("curve_radius_m")
            if curve_r and curve_r > 0:
                curve_factor = min(1.0, curve_r / (MIN_CURVE_RADIUS.get(speed_limit, 200) * 3))
                free_speed *= (0.85 + 0.15 * curve_factor)

            speed = round(free_speed * (1 - vc * 0.7), 1)
            speed = max(speed, 5)
            volume = int(capacity_per_lane * lanes * vc)
            tau = round(0.8 + (1 - vc) * 1.5, 1)

            traffic_params = {
                "speed_kmh": speed,
                "volume_vph": volume,
                "sigma": round(sigma, 2),
                "tau": tau,
                "accel": 2.0 if grade in ("고속도로", "도시고속") else 2.6,
                "decel": 3.5 if grade in ("고속도로", "도시고속") else 4.5,
                "minGap": 4.0 if grade in ("고속도로", "도시고속") else 2.5,
                "speedFactor": round(speed / speed_limit, 2) if speed_limit > 0 else 0.8,
                "passenger_ratio": 0.78 if grade == "고속도로" else 0.85,
                "truck_ratio": 0.18 if grade == "고속도로" else 0.08,
                "bus_ratio": 0.04 if grade == "고속도로" else 0.07,
            }

            # Combine geometry + traffic parameters
            combined = {**p, **traffic_params}
            combined["reasoning"] = (
                p["reasoning"] + f" {time_label}: V/C {vc:.2f}, "
                f"평균속도 {speed}km/h, 교통량 {volume}vph."
            )

            # Add time period to prompt
            new_prompt = f"{s['prompt']} {time_label}"
            enriched.append({
                "prompt": new_prompt,
                "params": combined,
                "meta": {**s["meta"], "time": time_label, "has_traffic": True},
            })

    return enriched


def main():
    random.seed(42)

    print("=" * 50)
    print("  Geometry training data generation")
    print("=" * 50)

    straight = generate_straight_scenarios()
    curve = generate_curve_scenarios()
    intersection = generate_intersection_scenarios()
    ramp = generate_ramp_scenarios()
    roundabout = generate_roundabout_scenarios()

    geometry_only = straight + curve + intersection + ramp + roundabout
    print(f"  Straight: {len(straight)} records")
    print(f"  Curve: {len(curve)} records")
    print(f"  Intersection: {len(intersection)} records")
    print(f"  Ramp: {len(ramp)} records")
    print(f"  Roundabout: {len(roundabout)} records")
    print(f"  Geometry total: {len(geometry_only)} records")

    # Combine traffic parameters (for cross-learning)
    enriched = add_traffic_to_geometry(geometry_only)
    print(f"  + Traffic conditions combined: {len(enriched)} records")

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    filepath = os.path.join(OUTPUT_DIR, "geometry_raw.jsonl")
    with open(filepath, "w", encoding="utf-8") as f:
        for s in enriched:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"\n  Saved: {filepath} ({len(enriched)} records)")

    # Sample output
    print(f"\nSamples:")
    for s in random.sample(enriched, min(5, len(enriched))):
        print(f"  Input: {s['prompt']}")
        p = s['params']
        print(f"  -> type={p['network_type']}, length={p.get('length_m')}m, "
              f"R={p.get('curve_radius_m')}m, lanes={p.get('lanes')}, "
              f"speed={p.get('speed_kmh')}km/h, vol={p.get('volume_vph')}vph")
        print()


if __name__ == "__main__":
    main()
