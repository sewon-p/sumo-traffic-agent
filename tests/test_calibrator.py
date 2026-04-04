import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibrator import (
    GAIN_SIGMA,
    GAIN_TAU,
    GAIN_VOLUME,
    MAX_DRIFT_SIGMA,
    MAX_DRIFT_TAU,
    MAX_DRIFT_VOLUME_PCT,
    _clamp,
    run_calibration,
)
from src.llm_parser import SimulationParams


def _make_params(**overrides):
    defaults = dict(
        location="test",
        time_start="08:00",
        time_end="09:00",
        vehicles_per_hour=1500,
        speed_limit_kmh=50,
        passenger_ratio=0.85,
        truck_ratio=0.10,
        bus_ratio=0.05,
    )
    defaults.update(overrides)
    return SimulationParams(**defaults)


def _make_ft(**overrides):
    defaults = dict(speed_kmh=25.0, sigma=0.6, tau=1.2)
    defaults.update(overrides)
    return defaults


class TestClamp(unittest.TestCase):
    def test_within_range(self):
        self.assertEqual(_clamp(5, 0, 10), 5)

    def test_below_min(self):
        self.assertEqual(_clamp(-1, 0, 10), 0)

    def test_above_max(self):
        self.assertEqual(_clamp(15, 0, 10), 10)


class TestCalibrationConvergence(unittest.TestCase):
    """Test that calibration converges when SUMO returns predictable speeds."""

    def _mock_sumo_speed_sequence(self, speeds):
        """Return a factory that yields speeds in sequence."""
        call_count = [0]

        def mock_run(cfg_path, warmup_seconds=0):
            idx = min(call_count[0], len(speeds) - 1)
            call_count[0] += 1
            return {"avg_speed_kmh": speeds[idx], "vehicles_inserted": 1000}

        return mock_run

    @patch("src.calibrator._run_sumo_headless")
    @patch("src.calibrator.generate_all")
    @patch("src.calibrator.build_vtypes_from_ft")
    def test_converges_when_speed_approaches_target(self, mock_vtypes, mock_gen, mock_sumo):
        mock_vtypes.return_value = []
        mock_gen.return_value = {"cfg": "/tmp/test.sumocfg"}

        # Target=25, initial=32 (too fast), then SUMO returns closer values
        speeds = [28.0, 25.5]
        call_idx = [0]

        def sumo_side_effect(cfg_path, warmup_seconds=0):
            idx = min(call_idx[0], len(speeds) - 1)
            call_idx[0] += 1
            return {"avg_speed_kmh": speeds[idx], "vehicles_inserted": 1000}

        mock_sumo.side_effect = sumo_side_effect

        ft = _make_ft(speed_kmh=25.0)
        params = _make_params(vehicles_per_hour=1500)

        result = run_calibration(
            ft=ft, params=params, net_path="/tmp/test.net.xml",
            output_dir="/tmp/cal_test", initial_sim_speed=32.0,
        )

        self.assertTrue(result["converged"])
        self.assertLessEqual(abs(result["final_error_pct"]), 10.0)
        self.assertGreater(len(result["iterations"]), 0)

    @patch("src.calibrator._run_sumo_headless")
    @patch("src.calibrator.generate_all")
    @patch("src.calibrator.build_vtypes_from_ft")
    def test_already_converged(self, mock_vtypes, mock_gen, mock_sumo):
        ft = _make_ft(speed_kmh=25.0)
        params = _make_params()

        result = run_calibration(
            ft=ft, params=params, net_path="/tmp/test.net.xml",
            output_dir="/tmp/cal_test", initial_sim_speed=26.0,
        )

        self.assertTrue(result["converged"])
        self.assertEqual(len(result["iterations"]), 0)
        mock_sumo.assert_not_called()

    @patch("src.calibrator._run_sumo_headless")
    @patch("src.calibrator.generate_all")
    @patch("src.calibrator.build_vtypes_from_ft")
    def test_max_iterations_reached(self, mock_vtypes, mock_gen, mock_sumo):
        mock_vtypes.return_value = []
        mock_gen.return_value = {"cfg": "/tmp/test.sumocfg"}
        # SUMO always returns 35 km/h, never converges to target 25
        mock_sumo.return_value = {"avg_speed_kmh": 35.0, "vehicles_inserted": 1000}

        ft = _make_ft(speed_kmh=25.0)
        params = _make_params(vehicles_per_hour=1500)

        result = run_calibration(
            ft=ft, params=params, net_path="/tmp/test.net.xml",
            output_dir="/tmp/cal_test", initial_sim_speed=35.0,
        )

        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "max_iterations")
        self.assertEqual(len(result["iterations"]), 3)

    def test_no_target_speed(self):
        ft = _make_ft(speed_kmh=None)
        params = _make_params()

        result = run_calibration(
            ft=ft, params=params, net_path="/tmp/test.net.xml",
            output_dir="/tmp/cal_test", initial_sim_speed=30.0,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertFalse(result["converged"])


class TestDriftLimits(unittest.TestCase):
    """Verify calibrated params stay within drift bounds."""

    @patch("src.calibrator._run_sumo_headless")
    @patch("src.calibrator.generate_all")
    @patch("src.calibrator.build_vtypes_from_ft")
    def test_volume_drift_capped(self, mock_vtypes, mock_gen, mock_sumo):
        mock_vtypes.return_value = []
        mock_gen.return_value = {"cfg": "/tmp/test.sumocfg"}
        # Huge error — sim 50 vs target 25 — should try to increase volume a lot
        mock_sumo.return_value = {"avg_speed_kmh": 50.0, "vehicles_inserted": 1000}

        ft = _make_ft(speed_kmh=25.0, sigma=0.5, tau=1.0)
        params = _make_params(vehicles_per_hour=1000)

        result = run_calibration(
            ft=ft, params=params, net_path="/tmp/test.net.xml",
            output_dir="/tmp/cal_test", initial_sim_speed=50.0,
        )

        cal = result["calibrated"]
        # Volume must not exceed +20% of original 1000
        self.assertLessEqual(cal["vehicles_per_hour"], int(1000 * (1 + MAX_DRIFT_VOLUME_PCT)))
        self.assertGreaterEqual(cal["vehicles_per_hour"], int(1000 * (1 - MAX_DRIFT_VOLUME_PCT)))
        # Sigma within bounds
        self.assertLessEqual(cal["sigma"], min(0.5 + MAX_DRIFT_SIGMA, 1.0))
        self.assertGreaterEqual(cal["sigma"], max(0.5 - MAX_DRIFT_SIGMA, 0.0))
        # Tau within bounds
        self.assertLessEqual(cal["tau"], min(1.0 + MAX_DRIFT_TAU, 3.0))
        self.assertGreaterEqual(cal["tau"], max(1.0 - MAX_DRIFT_TAU, 0.5))


class TestCallbackCalled(unittest.TestCase):
    @patch("src.calibrator._run_sumo_headless")
    @patch("src.calibrator.generate_all")
    @patch("src.calibrator.build_vtypes_from_ft")
    def test_on_iteration_callback(self, mock_vtypes, mock_gen, mock_sumo):
        mock_vtypes.return_value = []
        mock_gen.return_value = {"cfg": "/tmp/test.sumocfg"}
        mock_sumo.return_value = {"avg_speed_kmh": 30.0, "vehicles_inserted": 1000}

        ft = _make_ft(speed_kmh=25.0)
        params = _make_params()
        collected = []

        run_calibration(
            ft=ft, params=params, net_path="/tmp/test.net.xml",
            output_dir="/tmp/cal_test", initial_sim_speed=35.0,
            on_iteration=lambda d: collected.append(d),
        )

        self.assertGreater(len(collected), 0)
        for item in collected:
            self.assertIn("iteration", item)
            self.assertIn("sim_speed_kmh", item)
            self.assertIn("error_pct", item)
            self.assertIn("params", item)


if __name__ == "__main__":
    unittest.main()
