"""
Seoul Real-time Traffic Data API Connection Tool

Uses the Seoul Open Data Plaza citydata API to query
real-time link-level speeds and road congestion levels.

API: http://openapi.seoul.go.kr:8088/{key}/json/citydata/1/1/{area_name}
Response: ROAD_TRAFFIC_STTS -> link-level SPD, IDX, ROAD_NM, etc.
"""

import json
import os
import urllib.request
import urllib.parse


SEOUL_API_BASE = "http://openapi.seoul.go.kr:8088"

# Area name mapping used by Seoul citydata
# User input -> POI name used for citydata API calls
AREA_NAME_MAP = {
    "강남역": "강남 MICE 관광특구",
    "강남": "강남 MICE 관광특구",
    "강남대로": "강남 MICE 관광특구",
    "테헤란로": "강남 MICE 관광특구",
    "삼성역": "강남 MICE 관광특구",
    "코엑스": "강남 MICE 관광특구",
    "여의도": "여의도",
    "홍대": "홍대 관광특구",
    "홍대입구": "홍대 관광특구",
    "명동": "명동 관광특구",
    "잠실": "잠실 관광특구",
    "잠실역": "잠실 관광특구",
    "서울역": "서울역",
    "광화문": "광화문·덕수궁",
    "종로": "종로·청계 관광특구",
    "동대문": "동대문 관광특구",
    "이태원": "이태원 관광특구",
    "서초": "서초",
    "서초동": "서초",
    "방배동": "서초",
    "양재": "양재",
    "신촌": "신촌·이대 관광특구",
    "건대": "건대입구역",
    "왕십리": "왕십리역",
    "성수": "성수카페거리",
}


def _load_api_key() -> str:
    """Loads the API key from environment variables or the .env file."""
    key = os.environ.get("TOPIS_API_KEY", "")
    if key:
        return key

    # Load from .env file
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TOPIS_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


def query_realtime_traffic(
    location_name: str,
    api_key: str = None,
) -> dict:
    """
    Queries real-time road traffic information for a specific area
    via the Seoul citydata API.

    Args:
        location_name: Area name (e.g., 강남역, 여의도, 홍대)
        api_key: Seoul Open Data Plaza API key

    Returns:
        {
            "status": "success",
            "area_name": "...",
            "avg_speed_kmh": 27,
            "traffic_index": "원활",
            "traffic_message": "...",
            "links": [{"road_name": "...", "speed": 37, "index": "원활", ...}, ...],
            "timestamp": "2026-03-19 01:45"
        }
    """
    if api_key is None:
        api_key = _load_api_key()

    if not api_key:
        return {
            "status": "no_api_key",
            "message": "No Seoul API key found. Please set TOPIS_API_KEY in the .env file.",
        }

    # Area name mapping
    area_name = AREA_NAME_MAP.get(location_name, location_name)

    # Partial matching
    if area_name == location_name:
        for key, val in AREA_NAME_MAP.items():
            if key in location_name or location_name in key:
                area_name = val
                break

    encoded_area = urllib.parse.quote(area_name)
    url = f"{SEOUL_API_BASE}/{api_key}/json/citydata/1/1/{encoded_area}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SUMO-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

    except Exception as e:
        return {"status": "error", "message": str(e), "location": location_name}

    # Handle error responses
    result_info = data.get("RESULT", {})
    result_code = result_info.get("RESULT.CODE", result_info.get("CODE", ""))
    if result_code and result_code != "INFO-000":
        return {
            "status": "api_error",
            "message": result_info.get("RESULT.MESSAGE", result_info.get("MESSAGE", "")),
            "location": location_name,
        }

    citydata = data.get("CITYDATA", {})
    road_data = citydata.get("ROAD_TRAFFIC_STTS", {})

    if not road_data:
        return {
            "status": "no_traffic_data",
            "message": f"No traffic data available for '{area_name}'",
            "location": location_name,
        }

    # Average data
    avg_data = road_data.get("AVG_ROAD_DATA", {})
    # Per-link detailed data
    link_list = road_data.get("ROAD_TRAFFIC_STTS", [])

    links = []
    for link in link_list:
        links.append({
            "link_id": link.get("LINK_ID", ""),
            "road_name": link.get("ROAD_NM", ""),
            "start_name": link.get("START_ND_NM", ""),
            "end_name": link.get("END_ND_NM", ""),
            "distance_m": float(link.get("DIST", 0)),
            "speed_kmh": float(link.get("SPD", 0)),
            "index": link.get("IDX", ""),  # smooth, slow, congested, etc.
        })

    # Aggregate average speed by road name
    road_summary = {}
    for link in links:
        rn = link["road_name"]
        if rn not in road_summary:
            road_summary[rn] = {"speeds": [], "distances": [], "count": 0}
        road_summary[rn]["speeds"].append(link["speed_kmh"])
        road_summary[rn]["distances"].append(link["distance_m"])
        road_summary[rn]["count"] += 1

    road_averages = []
    for rn, info in road_summary.items():
        avg_spd = sum(info["speeds"]) / len(info["speeds"]) if info["speeds"] else 0
        total_dist = sum(info["distances"])
        road_averages.append({
            "road_name": rn,
            "avg_speed_kmh": round(avg_spd, 1),
            "total_distance_m": round(total_dist),
            "link_count": info["count"],
        })
    road_averages.sort(key=lambda x: x["link_count"], reverse=True)

    return {
        "status": "success",
        "source": "Seoul Real-time Traffic Information",
        "area_name": area_name,
        "avg_speed_kmh": avg_data.get("ROAD_TRAFFIC_SPD", 0),
        "traffic_index": avg_data.get("ROAD_TRAFFIC_IDX", ""),
        "traffic_message": avg_data.get("ROAD_MSG", ""),
        "timestamp": avg_data.get("ROAD_TRAFFIC_TIME", ""),
        "road_summary": road_averages[:10],  # top 10 roads
        "total_links": len(links),
        "links_sample": links[:5],  # sample of 5
    }


def query_road_speed(
    location_name: str,
    road_name: str = None,
    api_key: str = None,
) -> dict:
    """
    Queries real-time speed for a specific road in a given area.
    Filters the full traffic data by road name.

    Args:
        location_name: Area name
        road_name: Specific road name (e.g., "강남대로", "테헤란로")

    Returns:
        Per-link speed information for the specified road
    """
    result = query_realtime_traffic(location_name, api_key)

    if result["status"] != "success":
        return result

    if not road_name:
        return result

    # Fetch full data again for filtering
    api_key = api_key or _load_api_key()
    area_name = AREA_NAME_MAP.get(location_name, location_name)
    for key, val in AREA_NAME_MAP.items():
        if key in location_name or location_name in key:
            area_name = val
            break

    encoded_area = urllib.parse.quote(area_name)
    url = f"{SEOUL_API_BASE}/{api_key}/json/citydata/1/1/{encoded_area}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "SUMO-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        return {"status": "error", "message": str(e)}

    citydata = data.get("CITYDATA", {})
    link_list = citydata.get("ROAD_TRAFFIC_STTS", {}).get("ROAD_TRAFFIC_STTS", [])

    # Filter by road name
    matched_links = [
        {
            "link_id": l.get("LINK_ID"),
            "road_name": l.get("ROAD_NM"),
            "start": l.get("START_ND_NM"),
            "end": l.get("END_ND_NM"),
            "distance_m": float(l.get("DIST", 0)),
            "speed_kmh": float(l.get("SPD", 0)),
            "index": l.get("IDX"),
        }
        for l in link_list
        if road_name in l.get("ROAD_NM", "")
    ]

    if not matched_links:
        return {
            "status": "no_match",
            "message": f"No data for road '{road_name}'. Available roads in this area: {list(set(l.get('ROAD_NM','') for l in link_list[:20]))}",
            "area_avg_speed_kmh": result.get("avg_speed_kmh"),
        }

    speeds = [l["speed_kmh"] for l in matched_links]
    avg_speed = sum(speeds) / len(speeds)

    return {
        "status": "success",
        "source": "Seoul Real-time Traffic Information",
        "road_name": road_name,
        "avg_speed_kmh": round(avg_speed, 1),
        "min_speed_kmh": min(speeds),
        "max_speed_kmh": max(speeds),
        "link_count": len(matched_links),
        "links": matched_links,
        "timestamp": result.get("timestamp"),
    }


# CSV loading (kept for compatibility)
def load_traffic_csv(csv_path: str) -> list[dict]:
    """Loads a user-provided traffic data CSV file."""
    import csv

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return []

    column_map = {
        "speed": ["속도", "speed", "avg_speed", "평균속도", "SPEED", "SPD"],
        "volume": ["교통량", "volume", "traffic_volume", "VOLUME", "통행량"],
        "road_name": ["도로명", "road_name", "ROAD_NAME", "도로", "ROAD_NM"],
        "link_id": ["링크ID", "link_id", "LINK_ID", "linkId"],
        "time": ["시간", "time", "TIME", "시간대"],
    }

    def find_column(target_keys, fieldnames):
        for key in target_keys:
            if key in fieldnames:
                return key
        return None

    fieldnames = rows[0].keys()
    mapped = {}
    for standard, candidates in column_map.items():
        col = find_column(candidates, fieldnames)
        if col:
            mapped[standard] = col

    result = []
    for row in rows:
        entry = {}
        for standard, original in mapped.items():
            val = row.get(original, "")
            if standard in ("speed", "volume"):
                try:
                    entry[standard] = float(val)
                except (ValueError, TypeError):
                    entry[standard] = 0
            else:
                entry[standard] = val
        for k, v in row.items():
            if k not in mapped.values():
                entry[k] = v
        result.append(entry)

    return result


if __name__ == "__main__":
    print("=== Gangnam Station Real-time Traffic Information ===")
    r = query_realtime_traffic("강남역")
    print(json.dumps(r, ensure_ascii=False, indent=2))
