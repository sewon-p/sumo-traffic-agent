import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from tools.osm_network import apply_speed_limit_to_net
from tools.sumo_generator import build_vtypes_from_ft


class RuntimeParameterWiringTest(unittest.TestCase):
    def test_apply_speed_limit_to_net_overrides_non_internal_lanes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            net_path = os.path.join(tmpdir, "sample.net.xml")
            with open(net_path, "w", encoding="utf-8") as f:
                f.write(
                    "<net>\n"
                    '  <edge id="e0" from="n0" to="n1" speed="13.89">\n'
                    '    <lane id="e0_0" index="0" speed="13.89" length="100.0"/>\n'
                    "  </edge>\n"
                    '  <edge id=":n1_0" from="n1" to="n1" speed="8.33">\n'
                    '    <lane id=":n1_0_0" index="0" speed="8.33" length="20.0"/>\n'
                    "  </edge>\n"
                    "</net>\n"
                )

            apply_speed_limit_to_net(net_path, 40)
            tree = ET.parse(net_path)
            root = tree.getroot()

        main_edge = root.find("./edge[@id='e0']")
        internal_edge = root.find("./edge[@id=':n1_0']")
        self.assertEqual(main_edge.get("speed"), "11.11")
        self.assertEqual(main_edge.find("lane").get("speed"), "11.11")
        self.assertEqual(internal_edge.get("speed"), "8.33")
        self.assertEqual(internal_edge.find("lane").get("speed"), "8.33")

    def test_build_vtypes_from_ft_reflects_sigma_tau(self):
        vtypes = build_vtypes_from_ft(
            {"sigma": 0.72, "tau": 1.4},
            speed_limit_kmh=50,
        )

        self.assertTrue(vtypes)
        for vt in vtypes:
            self.assertEqual(vt.sigma, 0.72)
            self.assertEqual(vt.tau, 1.4)
            self.assertLessEqual(vt.max_speed, round((50 / 3.6) * 1.05, 2))


if __name__ == "__main__":
    unittest.main()
