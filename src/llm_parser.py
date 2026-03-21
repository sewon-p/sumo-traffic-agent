"""
LLM-based Natural Language -> Simulation Parameter Extraction

Parses user natural language input and returns structured parameters
required for SUMO traffic simulation.

- Uses Claude API if API key is available
- Falls back to rule-based parsing otherwise
"""

import json
import os
import re
from dataclasses import dataclass, asdict


@dataclass
class SimulationParams:
    """Simulation parameters extracted from natural language"""
    location: str = ""
    radius_m: float = 500
    time_start: str = "08:00"  # HH:MM
    time_end: str = "09:00"
    vehicles_per_hour: int = 1000
    passenger_ratio: float = 0.85
    truck_ratio: float = 0.10
    bus_ratio: float = 0.05
    speed_limit_kmh: float = 50.0
    # Special conditions
    weather: str = "clear"  # clear, rain, snow
    incident: str = ""  # Description of accidents/construction etc.
    lane_closure: int = 0  # Number of closed lanes
    notes: str = ""  # Additional notes determined by LLM

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = """You are a traffic simulation expert.
Analyze the user's natural language request and extract the parameters needed for SUMO traffic simulation in JSON format.

You must respond only in the JSON format below. Output only JSON without any explanation.

{
  "location": "Area name (e.g., 강남역, 테헤란로)",
  "radius_m": radius in meters (default 500),
  "time_start": "Start time HH:MM",
  "time_end": "End time HH:MM",
  "vehicles_per_hour": vehicles per hour (integer),
  "passenger_ratio": passenger car ratio (0~1),
  "truck_ratio": truck ratio (0~1),
  "bus_ratio": bus ratio (0~1),
  "speed_limit_kmh": speed limit (km/h),
  "weather": "clear|rain|snow",
  "incident": "Accident/construction description (empty string if none)",
  "lane_closure": number of closed lanes (integer, 0 if none),
  "notes": "Additional observations"
}

Parameter determination criteria:
- Evening rush hour: 18:00~19:00, high traffic volume (1500~2000 veh/h)
- Morning rush hour: 08:00~09:00, high traffic volume (1500~2000 veh/h)
- Late night: 00:00~05:00, low traffic volume (200~400 veh/h)
- Daytime: 10:00~16:00, moderate traffic volume (800~1200 veh/h)
- Major road/arterial: 50~60km/h, high traffic volume
- Side street: 30km/h, low traffic volume (300~600 veh/h)
- Highway: 80~100km/h
- Rainy conditions: estimated 20% speed reduction
- Accident: describe in incident, set lane_closure count"""
SYSTEM_PROMPT_VERSION = "rule-v1"


def parse_with_claude(user_input: str, api_key: str = None) -> SimulationParams:
    """Convert natural language to parameters using the Claude API."""
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        print("⚠ ANTHROPIC_API_KEY not found -> using rule-based fallback")
        return parse_with_rules(user_input)

    try:
        import anthropic
    except ImportError:
        print("⚠ anthropic package not installed -> run pip install anthropic and retry")
        return parse_with_rules(user_input)

    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_input}],
    )

    response_text = message.content[0].text.strip()

    # Extract JSON (handles ```json ... ``` wrapping)
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        print(f"⚠ JSON parsing failed -> using rule-based fallback")
        return parse_with_rules(user_input)

    data = json.loads(json_match.group())
    return SimulationParams(**{k: v for k, v in data.items() if k in SimulationParams.__dataclass_fields__})


def parse_with_rules(user_input: str) -> SimulationParams:
    """
    Rule-based parameter extraction (fallback when API is unavailable).
    Determines basic parameters through keyword matching.
    """
    params = SimulationParams()
    text = user_input.strip()

    # Location extraction: match known place names
    known_places = [
        "강남역", "강남대로", "테헤란로", "여의도", "올림픽대로",
        "서울역", "홍대입구", "잠실역", "방배동", "서초동",
        "종로", "을지로", "한강대로", "양재", "삼성역",
    ]
    for place in known_places:
        if place in text:
            params.location = place
            break

    # If location not found, use first word
    if not params.location:
        # Extract XX from "XX simulation" pattern
        match = re.search(r'([가-힣]+(?:역|로|대로|동|구간|도로))', text)
        if match:
            params.location = match.group(1)
        else:
            params.location = "강남역"  # Default value

    # Time period extraction
    time_match = re.search(r'(\d{1,2})[:\s~\-]+(\d{1,2})시', text)
    if time_match:
        h1, h2 = int(time_match.group(1)), int(time_match.group(2))
        params.time_start = f"{h1:02d}:00"
        params.time_end = f"{h2:02d}:00"
    elif "퇴근" in text:
        params.time_start = "18:00"
        params.time_end = "19:00"
        params.vehicles_per_hour = 1800
    elif "출근" in text:
        params.time_start = "08:00"
        params.time_end = "09:00"
        params.vehicles_per_hour = 1800
    elif "심야" in text or "새벽" in text:
        params.time_start = "00:00"
        params.time_end = "05:00"
        params.vehicles_per_hour = 300
    elif "점심" in text:
        params.time_start = "12:00"
        params.time_end = "13:00"
        params.vehicles_per_hour = 1200

    # HH~HH시 or HH시~HH시 pattern
    time_match2 = re.search(r'(\d{1,2})시?\s*[~\-]\s*(\d{1,2})시', text)
    if time_match2:
        h1, h2 = int(time_match2.group(1)), int(time_match2.group(2))
        params.time_start = f"{h1:02d}:00"
        params.time_end = f"{h2:02d}:00"

    # Speed/traffic volume adjustment by road type
    if "이면도로" in text or "골목" in text:
        params.speed_limit_kmh = 30
        params.vehicles_per_hour = min(params.vehicles_per_hour, 500)
        params.radius_m = 300
    elif "고속도로" in text or "올림픽대로" in text:
        params.speed_limit_kmh = 80
        params.vehicles_per_hour = max(params.vehicles_per_hour, 2000)
        params.radius_m = 800
    elif "대로" in text or "간선" in text:
        params.speed_limit_kmh = 60

    # Weather conditions
    if "비" in text or "우천" in text or "비올" in text:
        params.weather = "rain"
    elif "눈" in text or "폭설" in text:
        params.weather = "snow"

    # Accidents/construction
    if "사고" in text:
        params.incident = "교통사고"
        lane_match = re.search(r'(\d+)\s*차로\s*(차단|폐쇄|막)', text)
        if lane_match:
            params.lane_closure = int(lane_match.group(1))
        else:
            params.lane_closure = 1
    elif "공사" in text:
        params.incident = "도로공사"
        params.lane_closure = 1

    return params


FT_SYSTEM_PROMPT = (
    "You are a traffic engineering expert and SUMO simulation engineer. "
    "When the user describes a road/traffic situation, return only JSON with the parameters needed for SUMO simulation. "
    "You must fill all 8 fields below with numbers. Never use empty values or the string '-'.\n\n"
    "Calibration rules:\n"
    "- School/school zone: speed_limit_kmh=30, sigma high (0.6+)\n"
    "- Highway/expressway: speed_limit_kmh=80~100, avg_block_m 500+\n"
    "- Side street/alley: speed_limit_kmh=30, lanes=1, avg_block_m 50~80\n"
    "- Morning/evening rush: volume high, speed low\n"
    "- Late night: volume very low, speed high\n\n"
    "Output format:\n"
    '{"speed_kmh": number, "volume_vph": number, "lanes": one-way lane count, '
    '"speed_limit_kmh": number, "sigma": between 0~1, "tau": between 0.5~3, '
    '"avg_block_m": intersection spacing (m), "reasoning": "rationale"}'
)
FT_SYSTEM_PROMPT_VERSION = "ft-v1"

# Load from .env
_env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if "=" in _line and not _line.startswith("#"):
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


def parse_with_finetuned(user_input: str, api_key: str = None) -> SimulationParams:
    """Extract parameters using a fine-tuned model."""
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    ft_model = os.environ.get(
        "OPENAI_FT_MODEL",
        "ft:gpt-4.1-mini-2025-04-14:university-of-seoul:sumo-traffic-v2:DL1ENw1g"
    )

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=ft_model,
            messages=[
                {"role": "system", "content": FT_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.2,
            max_tokens=300,
        )
        text = resp.choices[0].message.content.strip()

        # Extract JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return None

        d = json.loads(json_match.group())

        # Fine-tuned output -> SimulationParams conversion (location extracted directly from input)
        params = SimulationParams()
        params.vehicles_per_hour = int(d.get("volume_vph", 1000))
        params.speed_limit_kmh = float(d.get("speed_limit_kmh", 50))

        # Location: extract only proper place names (empty string if none -> use generator)
        # Exclude generic nouns (arterial road, highway, side street, etc.)
        generic = {
            '간선도로', '보조간선도로', '주간선도로', '고속도로', '이면도로', '도시고속도로',
            '교차로', '사거리', '직선도로', '곡선도로', '도로', '격자도로',
            '개로', '으로', '해로', '거로',  # False positives from verb conjugations
        }
        # Proper road names: at least 2-char name + station/boulevard/road/street/district suffix
        loc_match = re.search(r'([가-힣]{2,}(?:역|대로)|[가-힣]{2,}로(?![\s만들해바꿔줘]))', user_input)
        loc = loc_match.group(0) if loc_match else ""
        params.location = "" if loc in generic else loc

        # Time: extract specific time -> keyword-based in order
        time_match = re.search(r'(\d{1,2})\s*[시:]\s*(?:~|-|부터)?\s*(\d{1,2})\s*시?', user_input)
        hour_match = re.search(r'(\d{1,2})\s*시(?:쯤|경|경에|에|의)?', user_input)
        if time_match:
            h1, h2 = int(time_match.group(1)), int(time_match.group(2))
            params.time_start, params.time_end = f"{h1:02d}:00", f"{h2:02d}:00"
        elif hour_match:
            h = int(hour_match.group(1))
            params.time_start, params.time_end = f"{h:02d}:00", f"{h+1:02d}:00"
        elif "퇴근" in user_input:
            params.time_start, params.time_end = "17:00", "20:00"
        elif "출근" in user_input:
            params.time_start, params.time_end = "07:00", "09:00"
        elif "심야" in user_input or "새벽" in user_input:
            params.time_start, params.time_end = "00:00", "05:00"
        elif "점심" in user_input:
            params.time_start, params.time_end = "12:00", "14:00"
        elif "야간" in user_input:
            params.time_start, params.time_end = "20:00", "23:00"

        # Store fine-tuning results in notes (for later use in SUMO configuration)
        params.notes = json.dumps({
            "ft_model": ft_model,
            "speed_kmh": d.get("speed_kmh"),
            "sigma": d.get("sigma"),
            "tau": d.get("tau"),
            "lanes": d.get("lanes"),
            "avg_block_m": d.get("avg_block_m"),
            "reasoning": d.get("reasoning", ""),
        }, ensure_ascii=False)

        return params

    except Exception as e:
        print(f"⚠ Fine-tuned model call failed: {e}")
        return None


def get_prompt_metadata(ft_used=False):
    return {
        "parser_prompt_version": FT_SYSTEM_PROMPT_VERSION if ft_used else SYSTEM_PROMPT_VERSION,
        "parser_prompt_type": "fine_tuned" if ft_used else "rule_or_claude",
        "ft_model": os.environ.get("OPENAI_FT_MODEL", "") if ft_used else "",
        "base_llm_mode": os.environ.get("BASE_LLM_MODE", ""),
        "base_llm_model": os.environ.get("BASE_LLM_MODEL", ""),
    }


def parse_user_input(user_input: str, api_key: str = None) -> SimulationParams:
    """
    Main parsing function.
    Priority: Fine-tuned model -> Claude API -> Rule-based fallback.
    """
    # 1) Try fine-tuned model
    ft_result = parse_with_finetuned(user_input)
    if ft_result:
        print("✓ Using fine-tuned model")
        return ft_result

    # 2) Try Claude API
    return parse_with_claude(user_input, api_key)


if __name__ == "__main__":
    # Test
    test_inputs = [
        "강남역 퇴근시간 시뮬레이션해줘",
        "테헤란로 18~19시 시뮬레이션",
        "방배동 이면도로 퇴근시간 시뮬레이션",
        "올림픽대로 여의도구간 사고 발생시 시뮬레이션. 3차로 중 1차로 차단, 30분간",
        "강남대로 출근시간 비오는 날 시뮬레이션",
    ]

    for text in test_inputs:
        print(f"\nInput: {text}")
        result = parse_user_input(text)
        print(f"Result: {json.dumps(result.to_dict(), ensure_ascii=False, indent=2)}")
