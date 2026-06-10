import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from generate_synthetic_tooluse_data import generate_dataset
from trl_grpo_tooluse.data_prep import convert_record


class SyntheticGeneratorTests(unittest.TestCase):
    def test_generates_requested_count(self):
        rows = generate_dataset(count=1000, seed=42)
        self.assertEqual(len(rows), 1000)
        self.assertEqual(len({row["id"] for row in rows}), 1000)

    def test_contains_expected_task_mix(self):
        rows = generate_dataset(count=1000, seed=42)
        task_counts = {}
        for row in rows:
            task_counts[row["task_type"]] = task_counts.get(row["task_type"], 0) + 1
        self.assertEqual(task_counts["single_turn"], 400)
        self.assertEqual(task_counts["hard_tool_selection"], 250)
        self.assertEqual(task_counts["irrelevance"], 250)
        self.assertEqual(task_counts["missing_param"], 100)

    def test_records_are_convertible(self):
        rows = generate_dataset(count=20, seed=7)
        converted = [convert_record(row) for row in rows]
        self.assertTrue(all(item is not None for item in converted))


if __name__ == "__main__":
    unittest.main()
