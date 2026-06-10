import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_grpo_tooluse.data_prep import iter_records


class DataPrepTests(unittest.TestCase):
    def test_missing_input_path_has_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "data" / "raw"
            with self.assertRaises(FileNotFoundError) as ctx:
                list(iter_records(missing))
        self.assertIn("Input path does not exist", str(ctx.exception))
        self.assertIn("BFCL", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
