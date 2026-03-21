import os
import sqlite3
import tempfile
import unittest
import json
from unittest import mock

from src import session_db


class SessionDbTest(unittest.TestCase):
    def test_save_modification_persists_structured_details(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            with mock.patch.object(session_db, "DB_PATH", db_path):
                sim_id = session_db.save_simulation(
                    "test", {"location": "강남역"}, {"sigma": 0.5}, "generated", tmpdir
                )
                session_db.save_modification(
                    sim_id,
                    "오른쪽 도로 좀 위로 휘게 해줘",
                    "geometry",
                    '{"before":1}',
                    '{"after":1}',
                    23.5,
                    modification_type="geometry",
                    edit_intent="alternative",
                    trainable=False,
                    details={"xml": {"old_nod_path": "a", "new_nod_path": "b"}},
                )

                conn = sqlite3.connect(db_path)
                row = conn.execute(
                    "SELECT field_changed, modification_type, edit_intent, trainable, details_json FROM modifications"
                ).fetchone()
                conn.close()

                self.assertEqual(row[0], "geometry")
                self.assertEqual(row[1], "geometry")
                self.assertEqual(row[2], "alternative")
                self.assertEqual(row[3], 0)
                self.assertIn('"new_nod_path": "b"', row[4])

    def test_export_corrections_for_training_filters_alternatives(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            out_path = os.path.join(tmpdir, "corrections.jsonl")
            with mock.patch.object(session_db, "DB_PATH", db_path):
                sim1 = session_db.save_simulation(
                    "초등학교 앞 사거리", 
                    {"location": "강남역", "vehicles_per_hour": 1200, "speed_limit_kmh": 30},
                    {"speed_kmh": 18, "lanes": 2, "sigma": 0.6, "tau": 1.4, "avg_block_m": 120, "reasoning": "school zone"},
                    "generated", tmpdir
                )
                sim2 = session_db.save_simulation(
                    "로터리 버전도 보여줘",
                    {"location": "강남역", "vehicles_per_hour": 1200, "speed_limit_kmh": 30},
                    {"speed_kmh": 18, "lanes": 2, "sigma": 0.6, "tau": 1.4, "avg_block_m": 120, "reasoning": "alt"},
                    "generated", tmpdir
                )

                session_db.save_modification(
                    sim1, "속도가 너무 높아, 스쿨존으로 수정", "parameter", "{}", "{}",
                    modification_type="parameter", edit_intent="correction", trainable=True,
                    details={"kind": "parameter"}
                )
                session_db.save_modification(
                    sim2, "로터리 버전도 보여줘", "geometry", "{}", "{}",
                    modification_type="geometry", edit_intent="alternative", trainable=False,
                    details={"kind": "geometry"}
                )

                session_db.export_corrections_for_training(out_path)

                with open(out_path, "r", encoding="utf-8") as f:
                    lines = [json.loads(line) for line in f if line.strip()]

                self.assertEqual(len(lines), 1)
                self.assertEqual(lines[0]["messages"][1]["content"], "초등학교 앞 사거리")

    def test_export_evaluation_report_summarizes_error_patterns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            report_path = os.path.join(tmpdir, "evaluation_report.txt")
            with mock.patch.object(session_db, "DB_PATH", db_path):
                sim1 = session_db.save_simulation(
                    "case1",
                    {"location": "A", "vehicles_per_hour": 1000, "speed_limit_kmh": 50},
                    {"speed_kmh": 20, "lanes": 2, "sigma": 0.5, "tau": 1.2, "avg_block_m": 100, "reasoning": "r1"},
                    "generated", tmpdir, sim_speed=30, ft_speed=20, error_pct=50.0, grade="F"
                )
                sim2 = session_db.save_simulation(
                    "case2",
                    {"location": "B", "vehicles_per_hour": 1000, "speed_limit_kmh": 50},
                    {"speed_kmh": 20, "lanes": 2, "sigma": 0.5, "tau": 1.2, "avg_block_m": 100, "reasoning": "r2"},
                    "generated", tmpdir, sim_speed=18, ft_speed=20, error_pct=-10.0, grade="A"
                )
                session_db.save_modification(
                    sim1, "속도 너무 빠름", "parameter", "{}", "{}",
                    modification_type="parameter", edit_intent="correction", trainable=True,
                    details={"kind": "parameter"}
                )
                session_db.save_modification(
                    sim2, "로터리 버전", "geometry", "{}", "{}",
                    modification_type="geometry", edit_intent="alternative", trainable=False,
                    details={"kind": "geometry"}
                )

                session_db.export_evaluation_report(report_path)

                with open(report_path.replace(".txt", ".json"), "r", encoding="utf-8") as f:
                    summary = json.load(f)

                self.assertEqual(summary["simulations"]["total"], 2)
                self.assertEqual(summary["simulations"]["grade_distribution"]["F"], 1)
                self.assertEqual(summary["simulations"]["error_patterns"]["speed_overestimate_severe"], 1)
                self.assertEqual(summary["modifications"]["by_intent"]["correction"], 1)
                self.assertEqual(summary["modifications"]["trainable_corrections_by_type"]["parameter"], 1)

    def test_build_llm_evaluation_summary_aggregates_parameter_and_geometry_corrections(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            with mock.patch.object(session_db, "DB_PATH", db_path):
                sim1 = session_db.save_simulation(
                    "초등학교 앞 사거리",
                    {"location": "강남역", "vehicles_per_hour": 1000, "speed_limit_kmh": 50},
                    {"speed_kmh": 20, "lanes": 2, "sigma": 0.5, "tau": 1.2, "avg_block_m": 120, "reasoning": "r1"},
                    "generated", tmpdir
                )
                sim2 = session_db.save_simulation(
                    "곡선 도로",
                    {"location": "강남역", "vehicles_per_hour": 1400, "speed_limit_kmh": 60},
                    {"speed_kmh": 25, "lanes": 3, "sigma": 0.4, "tau": 1.0, "avg_block_m": 180, "reasoning": "r2"},
                    "generated", tmpdir
                )

                session_db.save_modification(
                    sim1, "속도가 너무 높아", "parameter", "{}", "{}",
                    modification_type="parameter", edit_intent="correction", trainable=True,
                    details={
                        "kind": "parameter",
                        "changes": {"volume_vph": 1600, "sigma": 0.7},
                        "before": {
                            "params": {"vehicles_per_hour": 1000, "speed_limit_kmh": 50},
                            "ft": {"sigma": 0.5, "tau": 1.2},
                        },
                        "after": {},
                    }
                )
                session_db.save_modification(
                    sim2, "오른쪽 도로를 더 위로 휘게 해줘", "geometry", "{}", "{}",
                    modification_type="geometry", edit_intent="correction", trainable=True,
                    details={"kind": "geometry"}
                )
                session_db.save_modification(
                    sim2, "로터리 버전도 보여줘", "geometry", "{}", "{}",
                    modification_type="geometry", edit_intent="alternative", trainable=False,
                    details={"kind": "geometry"}
                )

                summary = session_db.build_llm_evaluation_summary()

                self.assertEqual(summary["overview"]["total_simulations"], 2)
                self.assertEqual(summary["overview"]["corrected_simulations"], 2)
                self.assertEqual(summary["overview"]["parameter_corrections"], 1)
                self.assertEqual(summary["overview"]["geometry_corrections"], 1)
                self.assertEqual(summary["parameter_fields"]["volume_vph"], 1)
                self.assertEqual(summary["parameter_fields"]["sigma"], 1)
                self.assertEqual(summary["parameter_avg_delta"]["volume_vph"], 600.0)
                self.assertEqual(summary["parameter_avg_delta"]["sigma"], 0.2)
                self.assertEqual(summary["geometry_categories"]["curvature_change"], 1)

    def test_geometry_corrections_can_contribute_ft_training_hints(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            with mock.patch.object(session_db, "DB_PATH", db_path):
                sim_id = session_db.save_simulation(
                    "long block road",
                    {"location": "A", "vehicles_per_hour": 1000, "speed_limit_kmh": 50},
                    {"speed_kmh": 20, "lanes": 2, "sigma": 0.5, "tau": 1.2, "avg_block_m": 120, "reasoning": "r"},
                    "generated", tmpdir
                )
                session_db.save_modification(
                    sim_id, "블록은 2km로", "geometry", "{}", "{}",
                    modification_type="geometry", edit_intent="correction", trainable=True,
                    details={
                        "kind": "geometry",
                        "before": {
                            "params": {"vehicles_per_hour": 1000, "speed_limit_kmh": 50},
                            "ft": {"lanes": 2, "avg_block_m": 120, "sigma": 0.5, "tau": 1.2},
                        },
                        "ft_train_fields": {"avg_block_m": 2000},
                    }
                )

                summary = session_db.build_llm_evaluation_summary()

                self.assertEqual(summary["geometry_categories"]["other_geometry"], 1)
                self.assertEqual(summary["parameter_fields"]["avg_block_m"], 1)
                self.assertEqual(summary["parameter_avg_delta"]["avg_block_m"], 1880.0)

    def test_export_llm_evaluation_report_includes_prompt_and_output_examples(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "sessions.db")
            report_path = os.path.join(tmpdir, "llm_eval.txt")
            with mock.patch.object(session_db, "DB_PATH", db_path):
                sim_id = session_db.save_simulation(
                    "초등학교 앞 사거리",
                    {"location": "강남역", "vehicles_per_hour": 1000, "speed_limit_kmh": 50},
                    {"speed_kmh": 20, "lanes": 2, "sigma": 0.5, "tau": 1.2, "avg_block_m": 120, "reasoning": "initial"},
                    "generated", tmpdir,
                    prompt_meta={"parser_prompt_version": "ft-v1"}
                )
                session_db.update_simulation_params(
                    sim_id,
                    {"location": "강남역", "vehicles_per_hour": 1600, "speed_limit_kmh": 30},
                    {"speed_kmh": 15, "lanes": 2, "sigma": 0.7, "tau": 1.2, "avg_block_m": 120, "reasoning": "final"},
                )
                session_db.save_modification(
                    sim_id, "속도 너무 높아", "parameter", "{}", "{}",
                    modification_type="parameter", edit_intent="correction", trainable=True,
                    details={
                        "changes": {"volume_vph": 1600, "sigma": 0.7},
                        "event_count": 1,
                    }
                )

                session_db.export_llm_evaluation_report(report_path)

                with open(report_path, "r", encoding="utf-8") as f:
                    text = f.read()
                with open(report_path.replace(".txt", ".json"), "r", encoding="utf-8") as f:
                    data = json.load(f)

                self.assertIn("Prompt versions:", text)
                self.assertIn("initial_output:", text)
                self.assertIn("final_output:", text)
                self.assertEqual(data["prompt_versions"]["ft-v1"], 1)
                self.assertEqual(data["sample_corrections"][0]["prompt_version"], "ft-v1")


if __name__ == "__main__":
    unittest.main()
