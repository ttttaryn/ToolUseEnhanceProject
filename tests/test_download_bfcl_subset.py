import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import download_bfcl_subset


class DownloadBfclSubsetTests(unittest.TestCase):
    def test_default_files_are_bfcl_json_files(self):
        self.assertIn("BFCL_v4_irrelevance.json", download_bfcl_subset.DEFAULT_FILES)
        self.assertTrue(all(name.endswith(".json") for name in download_bfcl_subset.DEFAULT_FILES))

    def test_default_base_url_points_to_bfcl_data(self):
        self.assertIn("berkeley-function-call-leaderboard/bfcl_eval/data", download_bfcl_subset.DEFAULT_BASE_URL)


if __name__ == "__main__":
    unittest.main()
