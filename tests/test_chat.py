import io
import unittest
from contextlib import redirect_stdout
from unittest import mock
from types import SimpleNamespace

import chat
import server
from src import base_llm


class ChatStatusTest(unittest.TestCase):
    def test_cmd_status_handles_tuple_return_from_detect_llm(self):
        with mock.patch.object(chat, "_detect_llm", return_value=("gemini", "cli")):
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                chat.cmd_status()
            output = buffer.getvalue()
        self.assertIn("gemini (cli)", output)


class ServerModificationTest(unittest.TestCase):
    def test_classify_modification_supports_mixed(self):
        with mock.patch.object(base_llm, "ask_base_llm", return_value="mixed"):
            self.assertEqual(base_llm.classify_modification("제한속도는 70, 교차로는 없애줘"), "mixed")

    def test_extract_ft_training_hints_has_regex_fallback_for_lanes_and_block(self):
        with mock.patch.object(base_llm, "ask_base_llm", return_value="{}"):
            hints = base_llm.extract_ft_training_hints("차선은 3개로 하고 블록은 2km로", "{}")
        self.assertEqual(hints["lanes"], 3)
        self.assertEqual(hints["avg_block_m"], 2000.0)

    def test_ft_runtime_status_requires_config_and_openai_package(self):
        with mock.patch.dict(server.os.environ, {"OPENAI_API_KEY": "sk-test", "OPENAI_FT_MODEL": "ft:test"}, clear=False):
            with mock.patch.object(server.importlib.util, "find_spec", return_value=SimpleNamespace()):
                status = server._ft_runtime_status()
        self.assertTrue(status["configured"])
        self.assertTrue(status["ready"])

    def test_apply_parameter_changes_returns_before_after_snapshots(self):
        params = mock.Mock()
        params.location = "강남역"
        params.radius_m = 500
        params.time_start = "07:00"
        params.time_end = "09:00"
        params.vehicles_per_hour = 1000
        params.speed_limit_kmh = 50
        params.weather = "clear"
        params.incident = ""
        params.lane_closure = 0
        ft = {"sigma": 0.5, "tau": 1.2, "lanes": 2, "avg_block_m": 150, "speed_kmh": 22}

        before, after = server._apply_parameter_changes(
            params, ft, {"volume_vph": 1800, "sigma": 0.7, "avg_block_m": 2000}
        )

        self.assertEqual(before["params"]["vehicles_per_hour"], 1000)
        self.assertEqual(after["params"]["vehicles_per_hour"], 1800)
        self.assertEqual(before["ft"]["sigma"], 0.5)
        self.assertEqual(after["ft"]["sigma"], 0.7)
        self.assertEqual(after["ft"]["avg_block_m"], 2000)

    def test_modify_request_without_session_should_not_fallback_to_new_parse(self):
        handler = server.Handler.__new__(server.Handler)
        events = []
        handler.send_sse = events.append
        with mock.patch.dict(server._session, {"net_path": None, "params": None, "ft": {}, "output_dir": None, "history": []}, clear=False):
            handler._run_simulation("오른쪽 도로 좀 위로 휘게 해줘", provider="modify", modify_intent="alternative")
        self.assertEqual(events[-1]["type"], "error")
        self.assertIn("No previous simulation", events[-1]["text"])


if __name__ == "__main__":
    unittest.main()
