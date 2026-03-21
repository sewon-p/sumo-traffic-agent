"""
Seoul traffic data periodic collector

Collects per-link speed data from the citydata API for 14 areas and stores it in the DB.
Collecting at 15-minute intervals allows computing representative hourly values after a few days to 2 weeks.

Usage:
  # Single collection
  python -m data_pipeline.collector

  # Continuous collection at 15-minute intervals
  python -m data_pipeline.collector --loop

  # Check collection status
  python -m data_pipeline.collector --status
"""

import json
import os
import sys
import time
import urllib.request
import urllib.parse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_pipeline.schema import get_connection, init_db

# Target areas for collection (POI names used in citydata API calls)
COLLECT_AREAS = [
    "강남 MICE 관광특구",
    "여의도",
    "서울역",
    "홍대 관광특구",
    "종로·청계 관광특구",
    "잠실 관광특구",
    "동대문 관광특구",
    "이태원 관광특구",
    "명동 관광특구",
    "건대입구역",
    "왕십리역",
    "성수카페거리",
    "광화문·덕수궁",
    "가로수길",
]

SEOUL_API_BASE = "http://openapi.seoul.go.kr:8088"
COLLECTION_INTERVAL_SECONDS = 15 * 60  # 15 minutes


def _load_api_key() -> str:
    key = os.environ.get("TOPIS_API_KEY", "")
    if key:
        return key
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("TOPIS_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return ""


def collect_once(api_key: str = None, verbose: bool = True) -> int:
    """
    Collect traffic data for all areas once and store it in the DB.

    Returns:
        Number of links saved
    """
    if not api_key:
        api_key = _load_api_key()
    if not api_key:
        print("TOPIS_API_KEY is not set.")
        return 0

    now = datetime.now()
    timestamp = now.isoformat(timespec="seconds")
    day_of_week = now.weekday()  # 0=Mon ~ 6=Sun
    hour = now.hour
    # Holiday detection is simplified to weekends only (can be extended later)
    is_holiday = 1 if day_of_week >= 5 else 0

    conn = get_connection()
    total_links = 0

    for area_name in COLLECT_AREAS:
        encoded = urllib.parse.quote(area_name)
        url = f"{SEOUL_API_BASE}/{api_key}/json/citydata/1/1/{encoded}"

        try:
            req = urllib.request.Request(url, headers={"User-Agent": "SUMO-Agent/1.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as e:
            if verbose:
                print(f"  ✗ {area_name}: {e}")
            continue

        # Check response
        result_info = data.get("RESULT", {})
        result_code = result_info.get("RESULT.CODE", result_info.get("CODE", ""))
        if result_code and result_code != "INFO-000":
            if verbose:
                print(f"  x {area_name}: API error {result_code}")
            continue

        citydata = data.get("CITYDATA", {})
        road_data = citydata.get("ROAD_TRAFFIC_STTS", {})
        links = road_data.get("ROAD_TRAFFIC_STTS", [])

        if not links:
            continue

        rows = []
        for link in links:
            rows.append((
                timestamp,
                area_name,
                link.get("LINK_ID", ""),
                link.get("ROAD_NM", ""),
                link.get("START_ND_NM", ""),
                link.get("END_ND_NM", ""),
                float(link.get("DIST", 0)),
                float(link.get("SPD", 0)),
                link.get("IDX", ""),
                day_of_week,
                hour,
                is_holiday,
            ))

        conn.executemany("""
            INSERT INTO raw_snapshots
            (timestamp, area_name, link_id, road_name, start_node, end_node,
             distance_m, speed_kmh, congestion_index, day_of_week, hour, is_holiday)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, rows)

        total_links += len(rows)
        if verbose:
            avg_spd = sum(r[7] for r in rows) / len(rows)
            print(f"  o {area_name}: {len(rows)} links, avg {avg_spd:.1f}km/h")

        time.sleep(0.3)  # Avoid API overload

    conn.commit()
    conn.close()

    if verbose:
        print(f"\nTotal {total_links} links saved ({timestamp})")

    return total_links


def show_status():
    """Print the collection status."""
    conn = get_connection()

    total = conn.execute("SELECT COUNT(*) as cnt FROM raw_snapshots").fetchone()["cnt"]
    if total == 0:
        print("No data has been collected yet.")
        conn.close()
        return

    first = conn.execute("SELECT MIN(timestamp) as ts FROM raw_snapshots").fetchone()["ts"]
    last = conn.execute("SELECT MAX(timestamp) as ts FROM raw_snapshots").fetchone()["ts"]
    snapshots = conn.execute("SELECT COUNT(DISTINCT timestamp) as cnt FROM raw_snapshots").fetchone()["cnt"]
    roads = conn.execute("SELECT COUNT(DISTINCT road_name) as cnt FROM raw_snapshots").fetchone()["cnt"]
    areas = conn.execute("SELECT COUNT(DISTINCT area_name) as cnt FROM raw_snapshots").fetchone()["cnt"]

    print(f"\nData Collection Status")
    print(f"  Total records: {total:,}")
    print(f"  Collection count: {snapshots}")
    print(f"  Period: {first} ~ {last}")
    print(f"  Areas: {areas}, Roads: {roads}")

    # Distribution by hour
    print(f"\n  Collection count by hour:")
    hourly = conn.execute("""
        SELECT hour, COUNT(*) as cnt, ROUND(AVG(speed_kmh), 1) as avg_spd
        FROM raw_snapshots GROUP BY hour ORDER BY hour
    """).fetchall()
    for row in hourly:
        bar = "█" * (row["cnt"] // 100)
        print(f"    {row['hour']:02d}h: {bar} {row['cnt']:>6} records (avg {row['avg_spd']}km/h)")

    conn.close()


def run_loop(interval_seconds: int = COLLECTION_INTERVAL_SECONDS):
    """Continuously collect at the specified interval."""
    print(f"Starting continuous collection ({interval_seconds}s interval)")
    print(f"   Press Ctrl+C to stop\n")

    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting collection...")
        try:
            count = collect_once()
            print(f"   -> {count} links collected")
        except Exception as e:
            print(f"   -> Collection error: {e}")

        print(f"   Next collection in {interval_seconds} seconds")
        try:
            time.sleep(interval_seconds)
        except KeyboardInterrupt:
            print("\n\nCollection stopped")
            break


if __name__ == "__main__":
    init_db()

    if "--loop" in sys.argv:
        run_loop()
    elif "--status" in sys.argv:
        show_status()
    else:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Running single collection...")
        collect_once()
        print("\nContinuous collection: python -m data_pipeline.collector --loop")
        print("Check status: python -m data_pipeline.collector --status")
