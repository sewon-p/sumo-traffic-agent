import importlib
import os
import unittest
from unittest import mock


class ConfigImportTest(unittest.TestCase):
    def test_config_import_does_not_raise_without_sumo(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            with mock.patch("shutil.which", return_value=None):
                with mock.patch("os.path.isfile", return_value=False):
                    config = importlib.import_module("src.config")
                    importlib.reload(config)
                    self.assertEqual(config.SUMO_BIN, "")
                    self.assertEqual(config.NETCONVERT_BIN, "")

    def test_get_sumo_bin_raises_when_required(self):
        config = importlib.import_module("src.config")
        with mock.patch("src.config._find_binary", return_value=""):
            with self.assertRaises(FileNotFoundError):
                config.get_sumo_bin()


if __name__ == "__main__":
    unittest.main()
