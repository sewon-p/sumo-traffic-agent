import os
import tempfile
import unittest

from src.validator import parse_sumo_statistics


class ValidatorWarmupTest(unittest.TestCase):
    def test_parse_sumo_statistics_excludes_warmup_intervals_and_trips(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_path = os.path.join(tmpdir, "case.sumocfg")
            detector_path = os.path.join(tmpdir, "detector_output.xml")
            tripinfo_path = os.path.join(tmpdir, "tripinfo.xml")

            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("<configuration/>")
            with open(detector_path, "w", encoding="utf-8") as f:
                f.write(
                    "<detector>\n"
                    '  <interval begin="0.00" end="300.00" speed="5.00" flow="10" occupancy="0.20" nVehContrib="10"/>\n'
                    '  <interval begin="600.00" end="900.00" speed="10.00" flow="20" occupancy="0.30" nVehContrib="20"/>\n'
                    "</detector>\n"
                )
            with open(tripinfo_path, "w", encoding="utf-8") as f:
                f.write(
                    "<tripinfos>\n"
                    '  <tripinfo id="a" depart="100.00" duration="50.0" waitingTime="5.0" timeLoss="7.0" routeLength="500.0"/>\n'
                    '  <tripinfo id="b" depart="700.00" duration="100.0" waitingTime="10.0" timeLoss="15.0" routeLength="1000.0"/>\n'
                    "</tripinfos>\n"
                )

            stats = parse_sumo_statistics(cfg_path, warmup_seconds=600)

        self.assertEqual(stats["warmup_seconds"], 600)
        self.assertEqual(stats["detector_intervals"], 1)
        self.assertEqual(stats["detector_intervals_skipped"], 1)
        self.assertEqual(stats["detector_avg_speed_kmh"], 36.0)
        self.assertEqual(stats["detector_total_count"], 20)
        self.assertEqual(stats["trip_count"], 1)
        self.assertEqual(stats["trip_count_skipped"], 1)
        self.assertEqual(stats["trip_avg_waiting_s"], 10.0)
        self.assertEqual(stats["trip_avg_speed_kmh"], 36.0)


if __name__ == "__main__":
    unittest.main()
