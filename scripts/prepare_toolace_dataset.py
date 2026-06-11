from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_grpo_tooluse.data_prep import iter_records, write_jsonl
from trl_grpo_tooluse.toolace_prep import convert_toolace_record


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Team-ACE/ToolACE into TRL GRPO JSONL.")
    parser.add_argument("--input", type=Path, default=None, help="Optional local ToolACE JSON/JSONL file or directory.")
    parser.add_argument("--dataset-name", default="Team-ACE/ToolACE")
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-samples", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    raw_records = iter_local_or_hf(args)
    converted = [item for item in (convert_toolace_record(record) for record in raw_records) if item]
    if not converted:
        raise SystemExit("No convertible ToolACE records found.")

    rng = random.Random(args.seed)
    rng.shuffle(converted)
    if args.max_samples:
        converted = converted[: args.max_samples]
    count = write_jsonl(converted, args.output)
    print(f"Wrote {count} ToolACE samples to {args.output}")


def iter_local_or_hf(args: argparse.Namespace) -> Iterable[dict]:
    if args.input is not None:
        yield from iter_records(args.input)
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install datasets or pass --input with a local ToolACE file.") from exc

    try:
        dataset = load_dataset(args.dataset_name, split=args.split)
    except Exception as exc:
        raise SystemExit(
            f"Failed to download {args.dataset_name} from Hugging Face: {exc}\n"
            "If AutoDL cannot reach Hugging Face, download this file locally:\n"
            "  https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/data.json\n"
            "Upload it to AutoDL, for example:\n"
            "  /root/autodl-tmp/ToolUseEnhanceProject/data/toolace/data.json\n"
            "Then run:\n"
            "  python scripts/prepare_toolace_dataset.py "
            "--input data/toolace/data.json "
            "--output data/grpo_train_toolace.jsonl "
            "--max-samples 3000"
        ) from exc
    for record in dataset:
        yield dict(record)


if __name__ == "__main__":
    main()
