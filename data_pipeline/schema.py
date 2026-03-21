"""
Traffic data SQLite schema and DB connection management

Tables:
- raw_snapshots: Raw per-link speed data collected from the API
- hourly_profiles: Aggregated representative values by hour
- road_metadata: Road basic information (number of lanes, grade, etc. - user provided)
- simulation_runs: SUMO execution logs and validation results
"""

import os
import sqlite3

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, "traffic_data", "traffic.db")


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Create the database tables."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS raw_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,           -- ISO 8601 (2026-03-19T18:15:00)
            area_name TEXT NOT NULL,           -- citydata area name
            link_id TEXT NOT NULL,
            road_name TEXT NOT NULL,
            start_node TEXT,
            end_node TEXT,
            distance_m REAL,
            speed_kmh REAL NOT NULL,
            congestion_index TEXT,             -- smooth, slow, congested
            day_of_week INTEGER,              -- 0=Mon ~ 6=Sun
            hour INTEGER,                     -- 0~23
            is_holiday INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_snapshots_road_hour
            ON raw_snapshots(road_name, hour, day_of_week);

        CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp
            ON raw_snapshots(timestamp);

        CREATE TABLE IF NOT EXISTS hourly_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            road_name TEXT NOT NULL,
            area_name TEXT NOT NULL,
            day_type TEXT NOT NULL,            -- weekday, Saturday, Sunday
            hour INTEGER NOT NULL,            -- 0~23
            speed_mean REAL,
            speed_std REAL,
            speed_p15 REAL,                   -- 15th percentile
            speed_p50 REAL,                   -- median
            speed_p85 REAL,                   -- 85th percentile
            congestion_prob_clear REAL,       -- P(smooth)
            congestion_prob_slow REAL,        -- P(slow)
            congestion_prob_jam REAL,         -- P(congested)
            sample_count INTEGER,
            last_updated TEXT,
            UNIQUE(road_name, day_type, hour)
        );

        CREATE TABLE IF NOT EXISTS road_metadata (
            link_id TEXT PRIMARY KEY,
            road_name TEXT NOT NULL,
            area_name TEXT,
            num_lanes INTEGER,
            speed_limit_kmh REAL,
            road_type TEXT,                   -- urban expressway, arterial road, collector road, local road
            district TEXT,
            distance_m REAL,
            source TEXT                       -- data source
        );

        CREATE TABLE IF NOT EXISTS user_volume_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            road_name TEXT NOT NULL,
            link_id TEXT,
            day_type TEXT,
            hour INTEGER,
            volume_vph INTEGER,               -- vehicles per hour
            passenger_ratio REAL,
            truck_ratio REAL,
            bus_ratio REAL,
            source TEXT
        );

        CREATE TABLE IF NOT EXISTS simulation_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            user_input TEXT,
            location TEXT,
            road_type TEXT,
            time_period TEXT,
            -- Input parameters
            param_vehicles_per_hour INTEGER,
            param_speed_limit REAL,
            param_sigma REAL,
            param_tau REAL,
            param_passenger_ratio REAL,
            param_truck_ratio REAL,
            param_bus_ratio REAL,
            -- Simulation results
            sim_speed_kmh REAL,
            sim_vehicles_inserted INTEGER,
            sim_waiting_s REAL,
            sim_timeloss_s REAL,
            -- Real data comparison
            real_speed_kmh REAL,
            real_volume_vph INTEGER,
            -- Validation
            grade TEXT,
            speed_error_pct REAL,
            -- Metadata
            prediction_source TEXT,           -- 'hardcoded', 'few_shot', 'fine_tuned'
            output_dir TEXT
        );
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    print(f"DB initialized: {DB_PATH}")
    conn = get_connection()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    for t in tables:
        print(f"  Table: {t['name']}")
    conn.close()
