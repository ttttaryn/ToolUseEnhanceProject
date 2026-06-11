import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_grpo_tooluse.data_prep import iter_records


class DataPrepBomTests(unittest.TestCase):
    def test_iter_records_accepts_utf8_bom_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.json"
            path.write_text("\ufeff" + json.dumps([{"question": "hello"}]), encoding="utf-8")
            records = list(iter_records(path))
        self.assertEqual(records, [{"question": "hello"}])

    def test_iter_records_accepts_utf8_bom_jsonl(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "records.jsonl"
            path.write_text("\ufeff" + json.dumps({"question": "hello"}) + "\n", encoding="utf-8")
            records = list(iter_records(path))
        self.assertEqual(records, [{"question": "hello"}])


if __name__ == "__main__":
    unittest.main()
