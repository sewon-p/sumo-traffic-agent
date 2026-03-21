#!/usr/bin/env python3
"""
Automatic training dataset (JSONL) generator for fine-tuning.

Generates diverse traffic scenarios by combining scenario templates with variable
combinations, and matches each scenario with expert-level ground-truth parameters.

Usage:
    python -m training.generate_dataset                    # Generate all
    python -m training.generate_dataset --verify gemini    # Verify with LLM
    python -m training.generate_dataset --count 50         # Generate only 50
"""

import json
import itertools
import random
import os
import argparse
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "data")
SYSTEM_PROMPT = (
    "너는 교통공학 전문가이자 SUMO 시뮬레이션 엔지니어다. "
    "사용자가 도로/교통 상황을 설명하면, SUMO 시뮬레이션에 필요한 "
    "파라미터를 JSON으로 반환한다. 모든 값은 현실적인 교통공학 근거에 기반해야 한다."
)

PARAM_SCHEMA = {
    "speed_kmh": "구간 평균속도 (km/h)",
    "volume_vph": "편도 시간당 교통량 (veh/h)",
    "lanes": "편도 차로수",
    "speed_limit_kmh": "제한속도 (km/h)",
    "sigma": "운전자 무작위성 (0~1, 높을수록 불규칙)",
    "tau": "안전 차두시간 (초)",
    "accel": "최대 가속도 (m/s²)",
    "decel": "최대 감속도 (m/s²)",
    "minGap": "최소 차간거리 (m)",
    "speedFactor": "제한속도 대비 실제속도 비율",
    "passenger_ratio": "승용차 비율 (0~1)",
    "truck_ratio": "화물차 비율 (0~1)",
    "bus_ratio": "버스 비율 (0~1)",
    "reasoning": "파라미터 결정 근거 (한국어)",
}

# ──────────────────────────────────────────────
# Scenario components
# ──────────────────────────────────────────────

ROADS_KOREA = [
    {"name": "강남대로", "lanes": 5, "limit": 50, "type": "도심간선"},
    {"name": "테헤란로", "lanes": 4, "limit": 50, "type": "도심간선"},
    {"name": "올림픽대로", "lanes": 4, "limit": 80, "type": "도시고속"},
    {"name": "강변북로", "lanes": 4, "limit": 80, "type": "도시고속"},
    {"name": "세종대로", "lanes": 6, "limit": 50, "type": "도심간선"},
    {"name": "남부순환로", "lanes": 3, "limit": 60, "type": "보조간선"},
    {"name": "경부고속도로 서울-수원", "lanes": 4, "limit": 100, "type": "고속도로"},
    {"name": "서울역 앞 도로", "lanes": 3, "limit": 50, "type": "도심간선"},
    {"name": "홍대입구역 주변", "lanes": 2, "limit": 50, "type": "보조간선"},
    {"name": "판교테크노밸리 내부도로", "lanes": 2, "limit": 40, "type": "집산도로"},
]

ROADS_INTERNATIONAL = [
    {"name": "Manhattan 5th Avenue", "lanes": 3, "limit": 40, "type": "도심간선"},
    {"name": "Tokyo Shinjuku main road", "lanes": 3, "limit": 50, "type": "도심간선"},
    {"name": "LA Interstate 405", "lanes": 5, "limit": 105, "type": "고속도로"},
    {"name": "London M25 motorway", "lanes": 4, "limit": 112, "type": "고속도로"},
    {"name": "Paris Champs-Élysées", "lanes": 4, "limit": 50, "type": "도심간선"},
]

TIME_SCENARIOS = [
    {"label": "출근시간", "hour": "08:00", "desc": "오전 출근 피크"},
    {"label": "퇴근시간", "hour": "18:00", "desc": "오후 퇴근 피크"},
    {"label": "점심시간", "hour": "12:00", "desc": "낮 시간대"},
    {"label": "심야", "hour": "02:00", "desc": "새벽 한산"},
    {"label": "주말 오후", "hour": "15:00", "desc": "주말 여가 통행"},
    {"label": "금요일 저녁", "hour": "19:00", "desc": "주말 전야 혼잡"},
]

WEATHER_CONDITIONS = ["맑음", "비", "눈", "안개"]

ABSTRACT_SCENARIOS = [
    "적당히 막히는 서울 퇴근길",
    "텅 빈 새벽 고속도로",
    "꽉 막힌 도심 교차로 주변",
    "비 오는 날 미끄러운 도로",
    "스쿨존 등하교 시간",
    "고속도로 졸음운전 구간",
    "공사 중인 2차로 도로",
    "버스전용차로가 있는 간선도로",
    "신도시 아파트단지 내부도로",
    "대형마트 주변 주말 도로",
    "공항 연결 도로 새벽 시간",
    "대학교 캠퍼스 내부 도로",
    "산업단지 출퇴근 도로",
    "해안도로 관광시즌",
    "터널 구간",
]


def _estimate_params(road: dict, time: dict, weather: str = "맑음") -> dict:
    """Estimate ground-truth parameters using rule-based logic.

    These values serve as targets for LLM fine-tuning, so they must be grounded
    in traffic engineering principles. Can be refined later with real data (CSV)
    or improved through LLM verification.
    """
    rt = road["type"]
    lanes = road["lanes"]
    limit = road["limit"]
    hour = int(time["hour"].split(":")[0])

    # Estimate V/C ratio (by time of day)
    if 7 <= hour <= 9 or 17 <= hour <= 19:
        vc = random.uniform(0.75, 0.95)  # Peak
    elif 10 <= hour <= 16:
        vc = random.uniform(0.45, 0.65)  # Off-peak
    elif 20 <= hour <= 23:
        vc = random.uniform(0.30, 0.50)  # Evening
    else:
        vc = random.uniform(0.10, 0.25)  # Late night

    # Capacity per lane by road type (one direction, veh/h/lane)
    capacity_per_lane = {
        "고속도로": 2200,
        "도시고속": 2000,
        "도심간선": 1800,
        "보조간선": 1600,
        "집산도로": 1200,
    }.get(rt, 1600)

    total_capacity = capacity_per_lane * lanes
    volume = int(total_capacity * vc)

    # Speed estimation (simplified BPR function)
    # t = t0 * (1 + alpha * (v/c)^beta), alpha=0.15, beta=4
    free_speed = limit * 0.9  # Free-flow speed
    bpr_ratio = 1 + 0.15 * (vc ** 4)
    speed = free_speed / bpr_ratio

    # Weather adjustment
    weather_factor = {"맑음": 1.0, "비": 0.85, "눈": 0.70, "안개": 0.80}.get(weather, 1.0)
    speed *= weather_factor

    speed = round(max(speed, 5), 1)  # Minimum 5 km/h

    # Car-following parameters
    if vc > 0.8:  # Congested
        sigma = round(random.uniform(0.6, 0.8), 2)
        tau = round(random.uniform(0.8, 1.2), 1)
    elif vc > 0.5:  # Moderate
        sigma = round(random.uniform(0.4, 0.6), 2)
        tau = round(random.uniform(1.0, 1.5), 1)
    else:  # Light traffic
        sigma = round(random.uniform(0.2, 0.4), 2)
        tau = round(random.uniform(1.5, 2.5), 1)

    # Vehicle composition
    if rt in ("고속도로",):
        passenger, truck, bus = 0.78, 0.18, 0.04
    elif rt in ("도시고속",):
        passenger, truck, bus = 0.82, 0.12, 0.06
    elif rt in ("도심간선", "보조간선"):
        passenger, truck, bus = 0.85, 0.08, 0.07
    else:
        passenger, truck, bus = 0.90, 0.05, 0.05

    accel = 2.6 if rt != "고속도로" else 2.0
    decel = 4.5 if rt != "고속도로" else 3.5
    min_gap = 2.5 if vc > 0.7 else 3.0
    speed_factor = round(speed / limit, 2) if limit > 0 else 0.8

    reasoning_parts = [
        f"{road['name']}은(는) {rt} 도로(편도 {lanes}차로, 제한속도 {limit}km/h).",
        f"{time['desc']} 시간대이므로 V/C비 약 {vc:.2f} 추정.",
        f"BPR 함수 적용 시 평균속도 약 {speed}km/h.",
    ]
    if weather != "맑음":
        reasoning_parts.append(f"{weather} 조건 반영하여 속도 {int((1-weather_factor)*100)}% 감소.")

    return {
        "speed_kmh": speed,
        "volume_vph": volume,
        "lanes": lanes,
        "speed_limit_kmh": limit,
        "sigma": sigma,
        "tau": tau,
        "accel": accel,
        "decel": decel,
        "minGap": min_gap,
        "speedFactor": speed_factor,
        "passenger_ratio": passenger,
        "truck_ratio": truck,
        "bus_ratio": bus,
        "reasoning": " ".join(reasoning_parts),
    }


def generate_specific_scenarios() -> list:
    """Generate scenarios combining real road names, time periods, and weather."""
    scenarios = []
    all_roads = ROADS_KOREA + ROADS_INTERNATIONAL

    for road in all_roads:
        for time in TIME_SCENARIOS:
            for weather in WEATHER_CONDITIONS:
                # Generate natural language input (varied expressions)
                templates = [
                    f"{road['name']} {time['label']} 시뮬레이션",
                    f"{road['name']}에서 {time['hour']}시 {weather} 상황",
                    f"{time['label']}에 {road['name']} {weather}일 때 교통 시뮬레이션 해줘",
                    f"{road['name']} {road['lanes']}차로 {time['desc']}",
                ]
                prompt = random.choice(templates)
                params = _estimate_params(road, time, weather)

                scenarios.append({
                    "prompt": prompt,
                    "params": params,
                    "meta": {
                        "road": road["name"],
                        "time": time["label"],
                        "weather": weather,
                        "road_type": road["type"],
                    },
                })
    return scenarios


def generate_abstract_scenarios() -> list:
    """Generate abstract expression scenarios (situation descriptions without road names)."""
    scenarios = []
    abstract_params = {
        "적당히 막히는 서울 퇴근길": {
            "speed_kmh": 18, "volume_vph": 3600, "lanes": 4, "speed_limit_kmh": 50,
            "sigma": 0.65, "tau": 1.0, "accel": 2.6, "decel": 4.5, "minGap": 2.5,
            "speedFactor": 0.36, "passenger_ratio": 0.85, "truck_ratio": 0.08, "bus_ratio": 0.07,
            "reasoning": "서울 도심 간선도로 퇴근시간 기준. V/C 0.85~0.9 수준의 정체 상황.",
        },
        "텅 빈 새벽 고속도로": {
            "speed_kmh": 95, "volume_vph": 400, "lanes": 4, "speed_limit_kmh": 100,
            "sigma": 0.3, "tau": 2.0, "accel": 2.0, "decel": 3.5, "minGap": 4.0,
            "speedFactor": 0.95, "passenger_ratio": 0.75, "truck_ratio": 0.22, "bus_ratio": 0.03,
            "reasoning": "새벽 2~4시 고속도로. V/C 0.05 수준. 화물차 비율이 높음.",
        },
        "꽉 막힌 도심 교차로 주변": {
            "speed_kmh": 8, "volume_vph": 2400, "lanes": 3, "speed_limit_kmh": 50,
            "sigma": 0.8, "tau": 0.8, "accel": 2.6, "decel": 4.5, "minGap": 2.0,
            "speedFactor": 0.16, "passenger_ratio": 0.87, "truck_ratio": 0.06, "bus_ratio": 0.07,
            "reasoning": "도심 교차로 극심한 정체. V/C 0.95 이상. 신호 대기 포함 평균 8km/h.",
        },
        "비 오는 날 미끄러운 도로": {
            "speed_kmh": 30, "volume_vph": 2000, "lanes": 3, "speed_limit_kmh": 60,
            "sigma": 0.5, "tau": 1.5, "accel": 2.0, "decel": 3.5, "minGap": 3.5,
            "speedFactor": 0.50, "passenger_ratio": 0.88, "truck_ratio": 0.07, "bus_ratio": 0.05,
            "reasoning": "우천 시 마찰계수 저하로 속도 15~20% 감소, 차두시간 증가.",
        },
        "스쿨존 등하교 시간": {
            "speed_kmh": 20, "volume_vph": 600, "lanes": 1, "speed_limit_kmh": 30,
            "sigma": 0.7, "tau": 1.5, "accel": 1.5, "decel": 5.0, "minGap": 3.0,
            "speedFactor": 0.67, "passenger_ratio": 0.95, "truck_ratio": 0.02, "bus_ratio": 0.03,
            "reasoning": "스쿨존 30km/h 제한. 보행자 횡단 빈번, 급정거 대비 높은 감속도.",
        },
        "고속도로 졸음운전 구간": {
            "speed_kmh": 85, "volume_vph": 1200, "lanes": 3, "speed_limit_kmh": 100,
            "sigma": 0.6, "tau": 1.8, "accel": 2.0, "decel": 3.0, "minGap": 4.0,
            "speedFactor": 0.85, "passenger_ratio": 0.80, "truck_ratio": 0.16, "bus_ratio": 0.04,
            "reasoning": "장거리 고속도로 단조 구간. sigma 높게 설정하여 주의력 저하 반영.",
        },
        "공사 중인 2차로 도로": {
            "speed_kmh": 25, "volume_vph": 800, "lanes": 1, "speed_limit_kmh": 30,
            "sigma": 0.7, "tau": 1.2, "accel": 1.5, "decel": 4.0, "minGap": 3.0,
            "speedFactor": 0.83, "passenger_ratio": 0.85, "truck_ratio": 0.12, "bus_ratio": 0.03,
            "reasoning": "공사로 2→1차로 축소. 병목 발생, 교통량 대비 높은 지체.",
        },
        "버스전용차로가 있는 간선도로": {
            "speed_kmh": 22, "volume_vph": 3200, "lanes": 3, "speed_limit_kmh": 50,
            "sigma": 0.55, "tau": 1.1, "accel": 2.6, "decel": 4.5, "minGap": 2.5,
            "speedFactor": 0.44, "passenger_ratio": 0.82, "truck_ratio": 0.06, "bus_ratio": 0.12,
            "reasoning": "버스전용차로 운영으로 일반차로 실질 용량 감소. 버스 비율 높음.",
        },
        "신도시 아파트단지 내부도로": {
            "speed_kmh": 25, "volume_vph": 500, "lanes": 1, "speed_limit_kmh": 30,
            "sigma": 0.5, "tau": 1.5, "accel": 2.0, "decel": 4.5, "minGap": 3.0,
            "speedFactor": 0.83, "passenger_ratio": 0.92, "truck_ratio": 0.03, "bus_ratio": 0.05,
            "reasoning": "단지 내부 30km/h 제한, 과속방지턱, 보행자 혼재.",
        },
        "대형마트 주변 주말 도로": {
            "speed_kmh": 15, "volume_vph": 1800, "lanes": 2, "speed_limit_kmh": 50,
            "sigma": 0.7, "tau": 1.0, "accel": 2.6, "decel": 4.5, "minGap": 2.5,
            "speedFactor": 0.30, "passenger_ratio": 0.93, "truck_ratio": 0.04, "bus_ratio": 0.03,
            "reasoning": "주말 대형마트 진출입 차량으로 주변 도로 극심한 정체.",
        },
        "공항 연결 도로 새벽 시간": {
            "speed_kmh": 70, "volume_vph": 800, "lanes": 3, "speed_limit_kmh": 80,
            "sigma": 0.35, "tau": 1.8, "accel": 2.0, "decel": 3.5, "minGap": 3.5,
            "speedFactor": 0.88, "passenger_ratio": 0.80, "truck_ratio": 0.05, "bus_ratio": 0.15,
            "reasoning": "새벽 공항 리무진 및 택시 비율 높음. 한산하여 자유류 근접.",
        },
        "대학교 캠퍼스 내부 도로": {
            "speed_kmh": 15, "volume_vph": 300, "lanes": 1, "speed_limit_kmh": 20,
            "sigma": 0.6, "tau": 2.0, "accel": 1.5, "decel": 5.0, "minGap": 3.0,
            "speedFactor": 0.75, "passenger_ratio": 0.90, "truck_ratio": 0.05, "bus_ratio": 0.05,
            "reasoning": "캠퍼스 내 20km/h 제한, 보행자 우선, 급정거 빈번.",
        },
        "산업단지 출퇴근 도로": {
            "speed_kmh": 30, "volume_vph": 2500, "lanes": 2, "speed_limit_kmh": 60,
            "sigma": 0.55, "tau": 1.2, "accel": 2.0, "decel": 4.0, "minGap": 3.0,
            "speedFactor": 0.50, "passenger_ratio": 0.75, "truck_ratio": 0.20, "bus_ratio": 0.05,
            "reasoning": "산업단지 출퇴근 시 화물차 비율 높고, 단일 진출입로에 병목.",
        },
        "해안도로 관광시즌": {
            "speed_kmh": 35, "volume_vph": 1500, "lanes": 2, "speed_limit_kmh": 60,
            "sigma": 0.5, "tau": 1.5, "accel": 2.0, "decel": 4.0, "minGap": 3.0,
            "speedFactor": 0.58, "passenger_ratio": 0.92, "truck_ratio": 0.03, "bus_ratio": 0.05,
            "reasoning": "관광시즌 주말 해안도로. 관광버스와 느린 관광 차량 혼재.",
        },
        "터널 구간": {
            "speed_kmh": 55, "volume_vph": 2000, "lanes": 2, "speed_limit_kmh": 60,
            "sigma": 0.4, "tau": 1.3, "accel": 2.0, "decel": 4.0, "minGap": 3.0,
            "speedFactor": 0.92, "passenger_ratio": 0.85, "truck_ratio": 0.10, "bus_ratio": 0.05,
            "reasoning": "터널 내 제한속도 60km/h, 차로변경 금지, 균일한 속도 유지.",
        },
    }

    for desc, params in abstract_params.items():
        scenarios.append({
            "prompt": desc,
            "params": params,
            "meta": {"type": "abstract", "scenario": desc},
        })

        # Add variant expressions
        variants = [
            f"{desc} 시뮬레이션 해줘",
            f"{desc} 조건으로 SUMO 돌려줘",
            f"{desc}에서 교통 시뮬레이션",
        ]
        for v in variants:
            scenarios.append({
                "prompt": v,
                "params": params,
                "meta": {"type": "abstract_variant", "scenario": desc},
            })

    return scenarios


def to_openai_jsonl(scenarios: list) -> list:
    """Convert to OpenAI fine-tuning JSONL format."""
    records = []
    for s in scenarios:
        records.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": s["prompt"]},
                {"role": "assistant", "content": json.dumps(s["params"], ensure_ascii=False)},
            ]
        })
    return records


def to_gemini_jsonl(scenarios: list) -> list:
    """Convert to Gemini tuning format."""
    records = []
    for s in scenarios:
        records.append({
            "text_input": f"[시스템] {SYSTEM_PROMPT}\n[사용자] {s['prompt']}",
            "output": json.dumps(s["params"], ensure_ascii=False),
        })
    return records


def save_jsonl(records: list, filepath: str):
    """Save records to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {filepath} ({len(records)} records)")


def verify_with_llm(scenarios: list, provider: str = "gemini", sample_n: int = 5):
    """Request LLM to verify the realism of generated data."""
    import subprocess

    samples = random.sample(scenarios, min(sample_n, len(scenarios)))
    prompt = "다음 교통 시뮬레이션 파라미터가 현실적인지 검증해줘. 비현실적인 값이 있으면 지적하고 수정값을 제안해.\n\n"
    for i, s in enumerate(samples):
        prompt += f"[{i+1}] 입력: {s['prompt']}\n"
        prompt += f"    파라미터: speed={s['params']['speed_kmh']}km/h, volume={s['params']['volume_vph']}vph, "
        prompt += f"lanes={s['params']['lanes']}, sigma={s['params']['sigma']}, tau={s['params']['tau']}\n\n"

    if provider == "gemini":
        result = subprocess.run(
            ["gemini"], input=prompt, capture_output=True, text=True, timeout=30
        )
    elif provider == "claude":
        result = subprocess.run(
            ["claude", "-p", prompt], capture_output=True, text=True, timeout=30
        )
    else:
        print(f"  Unsupported provider: {provider}")
        return

    print(f"\n{'='*50}")
    print(f"  LLM verification result ({provider})")
    print(f"{'='*50}")
    print(result.stdout)


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning training dataset generator")
    parser.add_argument("--count", type=int, default=0, help="Maximum number of scenarios to generate (0=all)")
    parser.add_argument("--verify", type=str, default=None, help="Verify with LLM (gemini/claude)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    print("Starting training dataset generation...")

    # Generate scenarios
    specific = generate_specific_scenarios()
    abstract = generate_abstract_scenarios()
    all_scenarios = specific + abstract

    if args.count > 0:
        all_scenarios = random.sample(all_scenarios, min(args.count, len(all_scenarios)))

    print(f"  Specific scenarios: {len(specific)} records")
    print(f"  Abstract scenarios: {len(abstract)} records")
    print(f"  Total: {len(all_scenarios)} records")

    # Save
    openai_records = to_openai_jsonl(all_scenarios)
    gemini_records = to_gemini_jsonl(all_scenarios)

    save_jsonl(openai_records, os.path.join(OUTPUT_DIR, "train_openai.jsonl"))
    save_jsonl(gemini_records, os.path.join(OUTPUT_DIR, "train_gemini.jsonl"))

    # Save raw data (for debugging/analysis)
    save_jsonl(all_scenarios, os.path.join(OUTPUT_DIR, "train_raw.jsonl"))

    # Statistics
    print(f"\n  Parameter ranges:")
    speeds = [s["params"]["speed_kmh"] for s in all_scenarios]
    volumes = [s["params"]["volume_vph"] for s in all_scenarios]
    print(f"    Speed: {min(speeds):.0f} ~ {max(speeds):.0f} km/h")
    print(f"    Volume: {min(volumes)} ~ {max(volumes)} vph")

    # LLM verification
    if args.verify:
        verify_with_llm(all_scenarios, provider=args.verify)


if __name__ == "__main__":
    main()
