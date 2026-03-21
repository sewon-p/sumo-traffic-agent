"""
Similar Road Search (Fallback Logic)

When traffic data is unavailable for the requested road,
finds roads with similar conditions (road type, number of lanes,
speed limit, etc.) and returns adjusted data.
"""

from dataclasses import dataclass


@dataclass
class RoadProfile:
    """Road profile"""
    name: str
    district: str  # district (gu)
    road_type: str  # arterial, sub-arterial, local road, etc.
    num_lanes: int
    speed_limit_kmh: float
    # Representative data by time period (peak hour basis)
    peak_speed_kmh: float = 0
    peak_volume_vph: int = 0


# Seoul major road profile database
ROAD_PROFILES = [
    # Arterial roads
    RoadProfile("강남대로", "강남구", "간선도로", 5, 60, peak_speed_kmh=18, peak_volume_vph=2200),
    RoadProfile("테헤란로", "강남구", "간선도로", 4, 60, peak_speed_kmh=20, peak_volume_vph=1900),
    RoadProfile("한강대로", "용산구", "간선도로", 5, 60, peak_speed_kmh=22, peak_volume_vph=2100),
    RoadProfile("종로", "종로구", "간선도로", 4, 50, peak_speed_kmh=16, peak_volume_vph=1800),
    RoadProfile("을지로", "중구", "간선도로", 3, 50, peak_speed_kmh=18, peak_volume_vph=1500),
    RoadProfile("도산대로", "강남구", "간선도로", 4, 60, peak_speed_kmh=22, peak_volume_vph=1700),
    RoadProfile("봉은사로", "강남구", "보조간선", 3, 50, peak_speed_kmh=20, peak_volume_vph=1200),
    RoadProfile("삼성로", "강남구", "보조간선", 2, 50, peak_speed_kmh=22, peak_volume_vph=1000),

    # Urban expressways
    RoadProfile("올림픽대로", "영등포구", "도시고속도로", 4, 80, peak_speed_kmh=35, peak_volume_vph=4500),
    RoadProfile("강변북로", "성동구", "도시고속도로", 4, 80, peak_speed_kmh=30, peak_volume_vph=4200),
    RoadProfile("내부순환로", "성북구", "도시고속도로", 3, 80, peak_speed_kmh=40, peak_volume_vph=3500),
    RoadProfile("동부간선도로", "성동구", "도시고속도로", 3, 80, peak_speed_kmh=45, peak_volume_vph=3000),

    # Sub-arterial roads
    RoadProfile("논현로", "강남구", "보조간선", 3, 50, peak_speed_kmh=18, peak_volume_vph=1300),
    RoadProfile("역삼로", "강남구", "보조간선", 2, 50, peak_speed_kmh=20, peak_volume_vph=900),
    RoadProfile("서초대로", "서초구", "보조간선", 4, 50, peak_speed_kmh=16, peak_volume_vph=1500),
    RoadProfile("반포대로", "서초구", "보조간선", 3, 50, peak_speed_kmh=20, peak_volume_vph=1200),
    RoadProfile("양재대로", "서초구", "보조간선", 3, 60, peak_speed_kmh=25, peak_volume_vph=1400),

    # Local roads
    RoadProfile("서초동 이면도로", "서초구", "이면도로", 1, 30, peak_speed_kmh=12, peak_volume_vph=400),
    RoadProfile("방배동 이면도로", "서초구", "이면도로", 1, 30, peak_speed_kmh=14, peak_volume_vph=350),
    RoadProfile("역삼동 이면도로", "강남구", "이면도로", 1, 30, peak_speed_kmh=10, peak_volume_vph=500),
    RoadProfile("신사동 이면도로", "강남구", "이면도로", 2, 30, peak_speed_kmh=15, peak_volume_vph=450),
    RoadProfile("성수동 이면도로", "성동구", "이면도로", 1, 30, peak_speed_kmh=12, peak_volume_vph=350),
]


def _similarity_score(target: dict, candidate: RoadProfile) -> float:
    """
    Calculates a similarity score (0 to 1) between target conditions and a candidate road.

    Matching criteria:
    - Road type (most important, 40%)
    - Number of lanes (25%)
    - Speed limit (25%)
    - Same district (10%)
    """
    score = 0.0

    # Road type matching
    if target.get("road_type") == candidate.road_type:
        score += 0.40
    elif target.get("road_type") in candidate.road_type or candidate.road_type in target.get("road_type", ""):
        score += 0.20

    # Number of lanes matching
    target_lanes = target.get("num_lanes", 2)
    lane_diff = abs(target_lanes - candidate.num_lanes)
    if lane_diff == 0:
        score += 0.25
    elif lane_diff == 1:
        score += 0.15
    elif lane_diff == 2:
        score += 0.05

    # Speed limit matching
    target_speed = target.get("speed_limit_kmh", 50)
    speed_diff = abs(target_speed - candidate.speed_limit_kmh)
    if speed_diff <= 5:
        score += 0.25
    elif speed_diff <= 10:
        score += 0.15
    elif speed_diff <= 20:
        score += 0.05

    # Same district bonus
    if target.get("district") and target["district"] == candidate.district:
        score += 0.10

    return score


def find_similar_roads(
    road_type: str = "간선도로",
    num_lanes: int = 2,
    speed_limit_kmh: float = 50,
    district: str = "",
    top_k: int = 3,
) -> list[dict]:
    """
    Searches for roads with similar conditions.

    Returns:
        List of road information sorted by similarity score
    """
    target = {
        "road_type": road_type,
        "num_lanes": num_lanes,
        "speed_limit_kmh": speed_limit_kmh,
        "district": district,
    }

    scored = []
    for road in ROAD_PROFILES:
        score = _similarity_score(target, road)
        scored.append((score, road))

    scored.sort(key=lambda x: x[0], reverse=True)

    results = []
    for score, road in scored[:top_k]:
        correction_factor = 1.0

        # Volume correction based on lane count difference
        if num_lanes != road.num_lanes:
            correction_factor *= (num_lanes / road.num_lanes)

        # Speed correction based on speed limit difference
        speed_ratio = speed_limit_kmh / road.speed_limit_kmh if road.speed_limit_kmh else 1.0
        speed_correction = min(max(speed_ratio, 0.7), 1.3)

        estimated_speed = round(road.peak_speed_kmh * speed_correction, 1)
        estimated_volume = int(road.peak_volume_vph * correction_factor)

        results.append({
            "reference_road": road.name,
            "reference_district": road.district,
            "similarity_score": round(score, 2),
            "reference_data": {
                "speed_kmh": road.peak_speed_kmh,
                "volume_vph": road.peak_volume_vph,
                "num_lanes": road.num_lanes,
                "speed_limit_kmh": road.speed_limit_kmh,
            },
            "estimated_data": {
                "speed_kmh": estimated_speed,
                "volume_vph": estimated_volume,
                "correction_factor": round(correction_factor, 2),
            },
            "confidence": "high" if score >= 0.7 else "medium" if score >= 0.5 else "low",
        })

    return results


if __name__ == "__main__":
    import json

    print("=== Similar road search for Bangbae-dong local road (1 lane, 30km/h) ===")
    results = find_similar_roads("이면도로", num_lanes=1, speed_limit_kmh=30, district="서초구")
    print(json.dumps(results, ensure_ascii=False, indent=2))

    print("\n=== Similar road search for 3-lane arterial 60km/h ===")
    results = find_similar_roads("간선도로", num_lanes=3, speed_limit_kmh=60, district="강남구")
    print(json.dumps(results, ensure_ascii=False, indent=2))
