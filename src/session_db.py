#!/usr/bin/env python3
"""
Simulation Session + Modification History DB

Stores all simulations and modification requests for later use as retraining data.

Tables:
  simulations: Record of each simulation run
  modifications: Modification request history (what parameters were changed and how)
"""

import json
import os
import sqlite3
from collections import Counter
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions.db")


def _get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS simulations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            user_input TEXT NOT NULL,
            params_json TEXT,
            ft_output_json TEXT,
            initial_ft_output_json TEXT,
            prompt_meta_json TEXT,
            network_type TEXT,
            output_dir TEXT,
            sim_speed_kmh REAL,
            ft_speed_kmh REAL,
            error_pct REAL,
            grade TEXT
        )
    """)
    sim_cols = {row["name"] for row in conn.execute("PRAGMA table_info(simulations)").fetchall()}
    if "initial_ft_output_json" not in sim_cols:
        conn.execute("ALTER TABLE simulations ADD COLUMN initial_ft_output_json TEXT")
    if "prompt_meta_json" not in sim_cols:
        conn.execute("ALTER TABLE simulations ADD COLUMN prompt_meta_json TEXT")
    if "calibrated_params_json" not in sim_cols:
        conn.execute("ALTER TABLE simulations ADD COLUMN calibrated_params_json TEXT")
    if "calibration_meta_json" not in sim_cols:
        conn.execute("ALTER TABLE simulations ADD COLUMN calibration_meta_json TEXT")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS modifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sim_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            user_input TEXT NOT NULL,
            field_changed TEXT,
            old_value TEXT,
            new_value TEXT,
            modification_type TEXT,
            edit_intent TEXT,
            trainable INTEGER DEFAULT 0,
            details_json TEXT,
            sim_speed_kmh REAL,
            FOREIGN KEY (sim_id) REFERENCES simulations(id)
        )
    """)
    cols = {row["name"] for row in conn.execute("PRAGMA table_info(modifications)").fetchall()}
    if "modification_type" not in cols:
        conn.execute("ALTER TABLE modifications ADD COLUMN modification_type TEXT")
    if "edit_intent" not in cols:
        conn.execute("ALTER TABLE modifications ADD COLUMN edit_intent TEXT")
    if "trainable" not in cols:
        conn.execute("ALTER TABLE modifications ADD COLUMN trainable INTEGER DEFAULT 0")
    if "details_json" not in cols:
        conn.execute("ALTER TABLE modifications ADD COLUMN details_json TEXT")
    conn.commit()
    return conn


def save_simulation(user_input, params_dict, ft_dict, network_type,
                    output_dir, sim_speed=None, ft_speed=None,
                    error_pct=None, grade=None, prompt_meta=None):
    """Save a simulation run record. Returns the sim_id."""
    conn = _get_conn()
    cur = conn.execute("""
        INSERT INTO simulations
        (created_at, user_input, params_json, ft_output_json, initial_ft_output_json,
         prompt_meta_json, network_type, output_dir, sim_speed_kmh, ft_speed_kmh, error_pct, grade)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        user_input,
        json.dumps(params_dict, ensure_ascii=False),
        json.dumps(ft_dict, ensure_ascii=False),
        json.dumps(ft_dict, ensure_ascii=False),
        json.dumps(prompt_meta or {}, ensure_ascii=False),
        network_type,
        output_dir,
        sim_speed,
        ft_speed,
        error_pct,
        grade,
    ))
    conn.commit()
    sim_id = cur.lastrowid
    conn.close()
    return sim_id


def save_modification(sim_id, user_input, field, old_val, new_val, sim_speed=None,
                      modification_type=None, edit_intent=None, trainable=False, details=None):
    """Save a modification history entry."""
    conn = _get_conn()
    conn.execute("""
        INSERT INTO modifications
        (sim_id, created_at, user_input, field_changed, old_value, new_value,
         modification_type, edit_intent, trainable, details_json, sim_speed_kmh)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        sim_id,
        datetime.now().isoformat(),
        user_input,
        field,
        str(old_val),
        str(new_val),
        modification_type or field,
        edit_intent,
        1 if trainable else 0,
        json.dumps(details, ensure_ascii=False) if details is not None else None,
        sim_speed,
    ))
    conn.commit()
    conn.close()


def update_simulation_params(sim_id, params_dict, ft_dict, sim_speed=None, error_pct=None, grade=None):
    """Update the simulation record with final parameters after modification."""
    conn = _get_conn()
    conn.execute("""
        UPDATE simulations
        SET params_json = ?, ft_output_json = ?, sim_speed_kmh = ?, error_pct = ?, grade = ?
        WHERE id = ?
    """, (
        json.dumps(params_dict, ensure_ascii=False),
        json.dumps(ft_dict, ensure_ascii=False),
        sim_speed,
        error_pct,
        grade,
        sim_id,
    ))
    conn.commit()
    conn.close()


def save_calibration(sim_id, calibrated_params, calibration_meta):
    """Save calibration results to the simulation record."""
    conn = _get_conn()
    conn.execute("""
        UPDATE simulations
        SET calibrated_params_json = ?, calibration_meta_json = ?
        WHERE id = ?
    """, (
        json.dumps(calibrated_params, ensure_ascii=False),
        json.dumps(calibration_meta, ensure_ascii=False),
        sim_id,
    ))
    conn.commit()
    conn.close()


SYSTEM_PROMPT = (
    "You are a traffic engineering expert and SUMO simulation engineer. "
    "When the user describes a road/traffic situation, return only JSON with the parameters needed for SUMO simulation. "
    "You must fill all 8 fields below with numbers. Never use empty values or the string '-'.\n\n"
    "Output format:\n"
    '{"speed_kmh": number, "volume_vph": number, "lanes": one-way lane count, '
    '"speed_limit_kmh": number, "sigma": between 0~1, "tau": between 0.5~3, '
    '"avg_block_m": intersection spacing (m), "reasoning": "rationale"}'
)


def _simulation_to_training_record(row):
    ft_out = json.loads(row["initial_ft_output_json"] or row["ft_output_json"] or "{}") if (row["initial_ft_output_json"] or row["ft_output_json"]) else {}
    params = json.loads(row["params_json"]) if row["params_json"] else {}
    if not ft_out:
        return None

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": row["user_input"]},
            {"role": "assistant", "content": json.dumps({
                "speed_kmh": ft_out.get("speed_kmh"),
                "volume_vph": params.get("vehicles_per_hour"),
                "lanes": ft_out.get("lanes"),
                "speed_limit_kmh": params.get("speed_limit_kmh"),
                "sigma": ft_out.get("sigma"),
                "tau": ft_out.get("tau"),
                "avg_block_m": ft_out.get("avg_block_m"),
                "reasoning": ft_out.get("reasoning", ""),
            }, ensure_ascii=False)},
        ]
    }


def _safe_json_loads(value, default=None):
    if not value:
        return {} if default is None else default
    try:
        return json.loads(value)
    except Exception:
        return {} if default is None else default


def _write_jsonl(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _classify_error_pattern(row):
    """Classify simulation results into simple error patterns."""
    err = row["error_pct"]
    if err is None:
        return "unverified"
    if abs(err) <= 10:
        return "well_calibrated"
    if err > 30:
        return "speed_overestimate_severe"
    if err > 10:
        return "speed_overestimate"
    if err < -30:
        return "speed_underestimate_severe"
    return "speed_underestimate"


def export_for_training(output_path=None):
    """Export saved simulations as OpenAI fine-tuning JSONL."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(DB_PATH), "sessions_openai.jsonl")

    conn = _get_conn()
    rows = conn.execute("SELECT * FROM simulations ORDER BY id").fetchall()

    records = []
    for row in rows:
        record = _simulation_to_training_record(row)
        if record:
            records.append(record)

    conn.close()
    _write_jsonl(records, output_path)

    print(f"Export complete: {output_path} ({len(records)} records, OpenAI JSONL)")
    return output_path


def export_corrections_for_training(output_path=None):
    """Export only verified simulations with error corrections as training JSONL."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(DB_PATH), "sessions_corrections_openai.jsonl")

    conn = _get_conn()
    rows = conn.execute("""
        SELECT DISTINCT s.*
        FROM simulations s
        JOIN modifications m ON m.sim_id = s.id
        WHERE m.trainable = 1 AND m.edit_intent = 'correction'
        ORDER BY s.id
    """).fetchall()

    records = []
    for row in rows:
        record = _simulation_to_training_record(row)
        if record:
            records.append(record)

    conn.close()
    _write_jsonl(records, output_path)

    print(f"Correction data export complete: {output_path} ({len(records)} records)")
    return output_path


def build_evaluation_summary():
    """Return summary statistics based on simulation/modification history."""
    conn = _get_conn()
    sims = conn.execute("SELECT * FROM simulations ORDER BY id").fetchall()
    mods = conn.execute("SELECT * FROM modifications ORDER BY id").fetchall()
    conn.close()

    grade_counts = Counter()
    error_patterns = Counter()
    trainable_corrections_by_type = Counter()
    modifications_by_intent = Counter()
    modifications_by_type = Counter()

    abs_errors = []
    corrected_sim_ids = set()

    for row in sims:
        grade = row["grade"] or "ungraded"
        grade_counts[grade] += 1
        error_patterns[_classify_error_pattern(row)] += 1
        if row["error_pct"] is not None:
            abs_errors.append(abs(row["error_pct"]))

    for row in mods:
        modifications_by_intent[row["edit_intent"] or "unknown"] += 1
        modifications_by_type[row["modification_type"] or row["field_changed"] or "unknown"] += 1
        if row["trainable"]:
            corrected_sim_ids.add(row["sim_id"])
            trainable_corrections_by_type[row["modification_type"] or row["field_changed"] or "unknown"] += 1

    summary = {
        "simulations": {
            "total": len(sims),
            "with_error_pct": len(abs_errors),
            "avg_abs_error_pct": round(sum(abs_errors) / len(abs_errors), 2) if abs_errors else None,
            "grade_distribution": dict(grade_counts),
            "error_patterns": dict(error_patterns),
            "corrected_simulations": len(corrected_sim_ids),
        },
        "modifications": {
            "total": len(mods),
            "by_intent": dict(modifications_by_intent),
            "by_type": dict(modifications_by_type),
            "trainable_corrections_by_type": dict(trainable_corrections_by_type),
        },
    }
    return summary


def _classify_geometry_request(text):
    text = (text or "").strip()
    if not text:
        return "unknown"
    geometry_keywords = [
        ("rotary_or_roundabout", ["로터리", "roundabout", "원형"]),
        ("curvature_change", ["휘게", "곡선", "굽", "curv", "bend"]),
        ("intersection_change", ["사거리", "삼거리", "교차로", "intersection", "junction"]),
        ("lane_structure_change", ["차로", "lane"]),
        ("connection_change", ["연결", "이어", "접속", "merge"]),
        ("length_or_position_change", ["위로", "아래로", "왼쪽", "오른쪽", "길게", "짧게", "이동"]),
    ]
    lower = text.lower()
    for name, patterns in geometry_keywords:
        if any(p in text or p in lower for p in patterns):
            return name
    return "other_geometry"


def build_llm_evaluation_summary():
    """Summarize LLM parameter extraction quality based on error correction logs."""
    conn = _get_conn()
    sims = conn.execute("""
        SELECT id, user_input, params_json, ft_output_json, initial_ft_output_json, prompt_meta_json
        FROM simulations ORDER BY id
    """).fetchall()
    mods = conn.execute("""
        SELECT sim_id, user_input, modification_type, edit_intent, trainable, details_json
        FROM modifications
        WHERE edit_intent = 'correction'
        ORDER BY id
    """).fetchall()
    conn.close()

    sim_by_id = {}
    for row in sims:
        sim_by_id[row["id"]] = {
            "user_input": row["user_input"],
            "params": _safe_json_loads(row["params_json"]),
            "final_ft": _safe_json_loads(row["ft_output_json"]),
            "initial_ft": _safe_json_loads(row["initial_ft_output_json"]),
            "prompt_meta": _safe_json_loads(row["prompt_meta_json"]),
        }

    total_corrections = 0
    parameter_corrections = 0
    geometry_corrections = 0
    parameter_field_counts = Counter()
    parameter_delta_totals = Counter()
    parameter_delta_counts = Counter()
    geometry_kind_counts = Counter()
    prompt_versions = Counter()
    sample_corrections = []

    for row in mods:
        total_corrections += 1
        mod_type = row["modification_type"] or "unknown"
        details = {}
        if row["details_json"]:
            try:
                details = json.loads(row["details_json"])
            except Exception:
                details = {}
        if mod_type == "parameter":
            parameter_corrections += 1
            changes = details.get("changes") or {}
            before = ((details.get("before") or {}).get("params") or {})
            before_ft = ((details.get("before") or {}).get("ft") or {})

            for field, new_value in changes.items():
                normalized_field = field
                parameter_field_counts[normalized_field] += 1

                if field == "volume_vph":
                    old_value = before.get("vehicles_per_hour")
                elif field in {"speed_limit_kmh"}:
                    old_value = before.get(field)
                else:
                    old_value = before_ft.get(field)

                try:
                    delta = float(new_value) - float(old_value)
                    parameter_delta_totals[normalized_field] += delta
                    parameter_delta_counts[normalized_field] += 1
                except (TypeError, ValueError):
                    continue
        elif mod_type == "geometry":
            geometry_corrections += 1
            geometry_kind_counts[_classify_geometry_request(row["user_input"])] += 1
            ft_train_fields = details.get("ft_train_fields") or {}
            before_ft = ((details.get("before") or {}).get("ft") or {})
            before_params = ((details.get("before") or {}).get("params") or {})
            for field, new_value in ft_train_fields.items():
                parameter_field_counts[field] += 1
                if field == "speed_limit_kmh":
                    old_value = before_params.get(field)
                else:
                    old_value = before_ft.get(field)
                try:
                    delta = float(new_value) - float(old_value)
                    parameter_delta_totals[field] += delta
                    parameter_delta_counts[field] += 1
                except (TypeError, ValueError):
                    continue
        elif mod_type == "mixed":
            parameter_corrections += 1
            geometry_corrections += 1
            geometry_kind_counts[_classify_geometry_request(row["user_input"])] += 1
            changes = details.get("changes") or {}
            ft_train_fields = details.get("ft_train_fields") or {}
            before = ((details.get("before") or {}).get("params") or {})
            before_ft = ((details.get("before") or {}).get("ft") or {})
            merged = {**changes, **ft_train_fields}
            for field, new_value in merged.items():
                parameter_field_counts[field] += 1
                if field == "volume_vph":
                    old_value = before.get("vehicles_per_hour")
                elif field == "speed_limit_kmh":
                    old_value = before.get(field)
                else:
                    old_value = before_ft.get(field)
                try:
                    delta = float(new_value) - float(old_value)
                    parameter_delta_totals[field] += delta
                    parameter_delta_counts[field] += 1
                except (TypeError, ValueError):
                    continue

        sim = sim_by_id.get(row["sim_id"]) or {}
        prompt_meta = sim.get("prompt_meta") or {}
        prompt_version = prompt_meta.get("parser_prompt_version") or "unknown"
        prompt_versions[prompt_version] += 1

        if len(sample_corrections) < 5:
            sample_corrections.append({
                "sim_id": row["sim_id"],
                "prompt_version": prompt_version,
                "user_input": sim.get("user_input", ""),
                "correction_request": row["user_input"],
                "modification_type": mod_type,
                "initial_output": sim.get("initial_ft", {}),
                "final_output": sim.get("final_ft", {}),
                "changes": details.get("changes", {}),
                "event_count": details.get("event_count", 1),
            })

    total_simulations = len(sims)
    corrected_sim_ids = {row["sim_id"] for row in mods}
    correction_rate = round(len(corrected_sim_ids) / total_simulations * 100, 1) if total_simulations else 0.0

    avg_delta_by_field = {}
    for field, total in parameter_delta_totals.items():
        count = parameter_delta_counts[field]
        if count:
            avg_delta_by_field[field] = round(total / count, 2)

    return {
        "overview": {
            "total_simulations": total_simulations,
            "corrected_simulations": len(corrected_sim_ids),
            "correction_rate_pct": correction_rate,
            "total_corrections": total_corrections,
            "parameter_corrections": parameter_corrections,
            "geometry_corrections": geometry_corrections,
        },
        "parameter_fields": dict(parameter_field_counts),
        "parameter_avg_delta": avg_delta_by_field,
        "geometry_categories": dict(geometry_kind_counts),
        "prompt_versions": dict(prompt_versions),
        "sample_corrections": sample_corrections,
    }


def export_llm_evaluation_report(output_path=None):
    if output_path is None:
        output_path = os.path.join(os.path.dirname(DB_PATH), "llm_evaluation_report.txt")

    summary = build_llm_evaluation_summary()
    overview = summary["overview"]

    lines = []
    lines.append("=" * 60)
    lines.append("LLM Evaluation Summary")
    lines.append("=" * 60)
    lines.append(f"Total simulations: {overview['total_simulations']}")
    lines.append(f"Corrected simulations: {overview['corrected_simulations']}")
    lines.append(f"Correction rate: {overview['correction_rate_pct']}%")
    lines.append(f"Parameter corrections: {overview['parameter_corrections']}")
    lines.append(f"Geometry corrections: {overview['geometry_corrections']}")

    lines.append("\nPrompt versions:")
    for name, count in sorted(summary["prompt_versions"].items()):
        lines.append(f"  {name}: {count}")

    lines.append("\nParameter field error counts:")
    for name, count in sorted(summary["parameter_fields"].items(), key=lambda x: (-x[1], x[0])):
        delta = summary["parameter_avg_delta"].get(name)
        delta_text = f" (avg delta {delta:+})" if delta is not None else ""
        lines.append(f"  {name}: {count}{delta_text}")

    lines.append("\nGeometry correction categories:")
    for name, count in sorted(summary["geometry_categories"].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"  {name}: {count}")

    lines.append("\nSample correction cases:")
    for sample in summary["sample_corrections"]:
        lines.append(f"  sim_id={sample['sim_id']} prompt={sample['prompt_version']} type={sample['modification_type']}")
        lines.append(f"    user_input: {sample['user_input']}")
        lines.append(f"    correction: {sample['correction_request']}")
        lines.append(f"    initial_output: {json.dumps(sample['initial_output'], ensure_ascii=False)}")
        lines.append(f"    final_output: {json.dumps(sample['final_output'], ensure_ascii=False)}")
        if sample["changes"]:
            lines.append(f"    changes: {json.dumps(sample['changes'], ensure_ascii=False)}")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return output_path


def export_evaluation_report(output_path=None):
    """Save an evaluation/error pattern report for cumulative run results."""
    if output_path is None:
        output_path = os.path.join(os.path.dirname(DB_PATH), "evaluation_report.txt")

    summary = build_evaluation_summary()
    sims = summary["simulations"]
    mods = summary["modifications"]

    lines = []
    lines.append("=" * 60)
    lines.append("Evaluation Summary")
    lines.append("=" * 60)
    lines.append(f"Total simulations: {sims['total']}")
    lines.append(f"Verified simulations: {sims['with_error_pct']}")
    lines.append(f"Average absolute error: {sims['avg_abs_error_pct'] if sims['avg_abs_error_pct'] is not None else 'N/A'}%")
    lines.append(f"Corrected simulations: {sims['corrected_simulations']}")

    lines.append("\nGrade distribution:")
    for grade, count in sorted(sims["grade_distribution"].items()):
        lines.append(f"  {grade}: {count}")

    lines.append("\nError patterns:")
    for name, count in sorted(sims["error_patterns"].items()):
        lines.append(f"  {name}: {count}")

    lines.append("\nModification intents:")
    for name, count in sorted(mods["by_intent"].items()):
        lines.append(f"  {name}: {count}")

    lines.append("\nModification types:")
    for name, count in sorted(mods["by_type"].items()):
        lines.append(f"  {name}: {count}")

    lines.append("\nTrainable corrections by type:")
    for name, count in sorted(mods["trainable_corrections_by_type"].items()):
        lines.append(f"  {name}: {count}")

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Evaluation report saved: {output_path}")
    return output_path


def list_simulations(limit=100):
    """List recent simulations."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT id, created_at, user_input, network_type, sim_speed_kmh, ft_speed_kmh, error_pct, grade, output_dir
        FROM simulations
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def list_modifications(limit=200):
    """List recent modification history."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT id, sim_id, created_at, user_input, field_changed, modification_type,
               edit_intent, trainable, sim_speed_kmh, details_json
        FROM modifications
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    result = []
    for row in rows:
        item = dict(row)
        if item.get("details_json"):
            try:
                item["details"] = json.loads(item["details_json"])
            except Exception:
                item["details"] = None
        result.append(item)
    return result
