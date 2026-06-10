from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_grpo_tooluse.data_prep import convert_record, iter_records, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert BFCL-style JSON/JSONL into TRL GRPO JSONL.")
    parser.add_argument("--input", required=True, type=Path, help="Input BFCL file or directory.")
    parser.add_argument("--output", required=True, type=Path, help="Output JSONL path.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap after shuffling.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--irrelevance-ratio",
        type=float,
        default=None,
        help="Optional target ratio for task_type=irrelevance samples.",
    )
    args = parser.parse_args()

    converted = [item for item in (convert_record(record) for record in iter_records(args.input)) if item]
    rng = random.Random(args.seed)
    rng.shuffle(converted)

    if args.irrelevance_ratio is not None:
        converted = _resample_irrelevance(converted, args.irrelevance_ratio, rng)
    if args.max_samples is not None:
        converted = converted[: args.max_samples]

    count = write_jsonl(converted, args.output)
    print(f"Wrote {count} samples to {args.output}")


def _resample_irrelevance(records: list[dict], ratio: float, rng: random.Random) -> list[dict]:
    ratio = min(max(ratio, 0.0), 1.0)
    irrelevant = [record for record in records if record.get("task_type") == "irrelevance"]
    relevant = [record for record in records if record.get("task_type") != "irrelevance"]
    if not irrelevant or not relevant:
        return records

    max_irrelevant = int(len(relevant) * ratio / max(1.0 - ratio, 1e-6))
    selected = irrelevant[:max_irrelevant]
    mixed = relevant + selected
    rng.shuffle(mixed)
    return mixed


if __name__ == "__main__":
    main()
