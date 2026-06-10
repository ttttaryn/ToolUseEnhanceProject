from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trl_grpo_tooluse.rewards import (
    argument_reward,
    brevity_reward,
    format_reward,
    hallucination_penalty,
    tool_selection_reward,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score generation JSONL with GRPO reward functions.")
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    completions = [row.get("completion") or row.get("response") or row.get("prediction") or "" for row in rows]
    ground_truth = [row.get("ground_truth") for row in rows]
    task_type = [row.get("task_type", "") for row in rows]
    tools = [row.get("tools", []) for row in rows]

    metrics = {
        "format": format_reward(completions, ground_truth, task_type=task_type),
        "tool_selection": tool_selection_reward(completions, ground_truth, task_type=task_type),
        "argument": argument_reward(completions, ground_truth, task_type=task_type),
        "hallucination": hallucination_penalty(completions, ground_truth, tools=tools),
        "brevity": brevity_reward(completions),
    }
    for name, values in metrics.items():
        print(f"{name}: {sum(values) / max(len(values), 1):.4f}")


if __name__ == "__main__":
    main()
