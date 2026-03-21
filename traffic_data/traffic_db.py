"""
Traffic statistics representative values DB

Provides representative traffic statistics by road type and time period for Seoul.
Uses stable representative values instead of real-time data to determine simulation parameters.
"""

import json
import os

_DB_PATH = os.path.join(os.path.dirname(__file__), "seoul_traffic_stats.json")
_db = None


def _load_db() -> dict:
    global _db
    if _db is None:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            _db = json.load(f)
    return _db


def get_traffic_stats(
    road_type: str,
    time_period: str,
    day_type: str = "평일",
    num_lanes: int = None,
) -> dict:
    """
    Return representative traffic statistics for the given road type and time period.

    Args:
        road_type: "도시고속도로", "간선도로", "보조간선", "이면도로"
        time_period: "출근(07-09)", "퇴근(18-20)", etc.
        day_type: "평일", "토요일", "일요일"
        num_lanes: Number of lanes (uses road type default if None)

    Returns:
        {
            "speed_kmh": representative speed,
            "volume_per_lane": volume per lane,
            "total_volume_vph": total volume (reflecting number of lanes),
            "v_c_ratio": V/C ratio,
            "speed_limit": speed limit,
            "num_lanes": number of lanes used,
            "vehicle_composition": {"passenger": 0.85, ...},
            "source": "Seoul traffic statistics representative values"
        }
    """
    db = _load_db()

    if road_type not in db:
        available = [k for k in db if not k.startswith("_") and k != "vehicle_composition"]
        return {"error": f"'{road_type}' not found. Available: {available}"}

    road_data = db[road_type]

    # Select data by day type
    day_map = {"평일": "weekday", "토요일": "saturday", "일요일": "sunday"}
    day_key = day_map.get(day_type, "weekday")
    time_data = road_data.get(day_key, road_data.get("weekday", {}))

    if time_period not in time_data:
        return {"error": f"'{time_period}' not found. Available: {list(time_data.keys())}"}

    stats = time_data[time_period]
    lanes = num_lanes or road_data.get("typical_lanes", 2)

    # Vehicle composition
    composition = db.get("vehicle_composition", {}).get(road_type, {
        "passenger": 0.85, "truck": 0.10, "bus": 0.05
    })

    return {
        "road_type": road_type,
        "time_period": time_period,
        "day_type": day_type,
        "speed_kmh": stats["speed"],
        "volume_per_lane": stats["volume"],
        "total_volume_vph": stats["volume"] * lanes,
        "v_c_ratio": stats["v_c_ratio"],
        "speed_limit_kmh": road_data["speed_limit"],
        "num_lanes": lanes,
        "examples": road_data.get("examples", []),
        "vehicle_composition": composition,
        "source": "Seoul traffic statistics representative values",
    }


def classify_road_type(location_name: str) -> str:
    """Estimate the road type from the area/road name."""
    db = _load_db()

    for road_type in ["도시고속도로", "간선도로", "보조간선", "이면도로"]:
        if road_type not in db:
            continue
        examples = db[road_type].get("examples", [])
        for ex in examples:
            if ex in location_name or location_name in ex:
                return road_type

    # Keyword-based estimation
    if any(k in location_name for k in ["고속", "올림픽", "강변", "내부순환", "동부간선"]):
        return "도시고속도로"
    elif any(k in location_name for k in ["대로", "테헤란", "종로", "한강"]):
        return "간선도로"
    elif any(k in location_name for k in ["이면", "골목", "주택가"]):
        return "이면도로"
    elif any(k in location_name for k in ["로"]):
        return "보조간선"

    return "간선도로"  # Default


def classify_time_period(time_str: str = None, keyword: str = None) -> str:
    """Classify the time period from a time string or keyword."""
    if keyword:
        kw_map = {
            "출근": "출근(07-09)", "퇴근": "퇴근(18-20)",
            "심야": "심야(00-06)", "새벽": "심야(00-06)",
            "점심": "점심(12-14)", "오전": "오전(09-12)",
            "오후": "오후(14-18)", "야간": "야간(20-24)",
            "저녁": "야간(20-24)",
        }
        for k, v in kw_map.items():
            if k in keyword:
                return v

    if time_str:
        try:
            hour = int(time_str.split(":")[0])
            if 0 <= hour < 6: return "심야(00-06)"
            elif 7 <= hour < 9: return "출근(07-09)"
            elif 9 <= hour < 12: return "오전(09-12)"
            elif 12 <= hour < 14: return "점심(12-14)"
            elif 14 <= hour < 18: return "오후(14-18)"
            elif 18 <= hour < 20: return "퇴근(18-20)"
            else: return "야간(20-24)"
        except (ValueError, IndexError):
            pass

    return "오후(14-18)"  # Default


if __name__ == "__main__":
    print("=== Gangnam-daero, evening rush hour ===")
    r = get_traffic_stats("간선도로", "퇴근(18-20)")
    print(json.dumps(r, ensure_ascii=False, indent=2))

    print("\n=== Olympic Expressway, morning rush hour ===")
    r = get_traffic_stats("도시고속도로", "출근(07-09)")
    print(json.dumps(r, ensure_ascii=False, indent=2))

    print(f"\nRoad classification: Teheran-ro -> {classify_road_type('테헤란로')}")
    print(f"Road classification: Bangbae-dong local road -> {classify_road_type('방배동 이면도로')}")
    print(f"Time classification: evening rush -> {classify_time_period(keyword='퇴근')}")
