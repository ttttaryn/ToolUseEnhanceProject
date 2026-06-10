"""Rule-based rewards for BFCL-style tool-use GRPO training."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .tool_parser import is_no_tool_gold, parse_tool_calls, parse_tool_schema


def format_reward(completions: list[str], ground_truth: list[Any], task_type: list[str] | None = None, **_: Any) -> list[float]:
    """Reward parseable tool calls, or direct answers for no-tool samples."""

    task_type = _coerce_list(task_type, len(completions), "")
    rewards: list[float] = []
    for completion, gold, task in zip(completions, ground_truth, task_type):
        text = _completion_to_text(completion)
        calls = parse_tool_calls(text)
        no_tool = is_no_tool_gold(gold, task)
        if no_tool:
            rewards.append(1.0 if not calls and _has_direct_answer(text) else -0.5)
        else:
            rewards.append(1.0 if calls else -1.0)
    return rewards


def tool_selection_reward(completions: list[str], ground_truth: list[Any], task_type: list[str] | None = None, **_: Any) -> list[float]:
    """Reward selecting the correct tool and refusing tools on irrelevance tasks."""

    task_type = _coerce_list(task_type, len(completions), "")
    rewards: list[float] = []
    for completion, gold, task in zip(completions, ground_truth, task_type):
        pred_calls = parse_tool_calls(_completion_to_text(completion))
        gold_calls = parse_tool_calls(gold)
        if is_no_tool_gold(gold, task):
            rewards.append(1.0 if not pred_calls else -1.0)
            continue
        if not pred_calls:
            rewards.append(-1.0)
            continue
        gold_names = {call.name for call in gold_calls}
        pred_names = {call.name for call in pred_calls}
        if pred_names == gold_names:
            rewards.append(1.0)
        elif pred_names & gold_names:
            rewards.append(0.3)
        else:
            rewards.append(-1.0)
    return rewards


def argument_reward(completions: list[str], ground_truth: list[Any], task_type: list[str] | None = None, **_: Any) -> list[float]:
    """Reward argument overlap and exact values for the matched gold tool."""

    task_type = _coerce_list(task_type, len(completions), "")
    rewards: list[float] = []
    for completion, gold, task in zip(completions, ground_truth, task_type):
        if is_no_tool_gold(gold, task):
            rewards.append(0.0)
            continue
        pred_calls = parse_tool_calls(_completion_to_text(completion))
        gold_calls = parse_tool_calls(gold)
        if not pred_calls or not gold_calls:
            rewards.append(-0.5)
            continue
        pred_by_name = {call.name: call for call in pred_calls}
        sample_scores: list[float] = []
        for gold_call in gold_calls:
            pred = pred_by_name.get(gold_call.name)
            if pred is None:
                sample_scores.append(-0.5)
                continue
            sample_scores.append(_argument_score(pred.arguments, gold_call.arguments))
        rewards.append(sum(sample_scores) / len(sample_scores))
    return rewards


def hallucination_penalty(completions: list[str], ground_truth: list[Any], tools: list[Any] | None = None, **_: Any) -> list[float]:
    """Penalize invented tools and invented parameters."""

    tools = _coerce_list(tools, len(completions), [])
    rewards: list[float] = []
    for completion, tool_schema in zip(completions, tools):
        calls = parse_tool_calls(_completion_to_text(completion))
        allowed_tools = parse_tool_schema(tool_schema)
        if not calls:
            rewards.append(0.0)
            continue

        penalty = 0.0
        for call in calls:
            allowed_args = allowed_tools.get(call.name)
            if allowed_args is None:
                penalty -= 1.0
                continue
            extra_args = set(call.arguments) - allowed_args if allowed_args else set()
            penalty -= min(0.5, 0.15 * len(extra_args))
        rewards.append(penalty)
    return rewards


def brevity_reward(completions: list[str], **_: Any) -> list[float]:
    """Small guard against verbose templates that hide malformed calls."""

    rewards: list[float] = []
    for completion in completions:
        length = len(_completion_to_text(completion))
        if length <= 400:
            rewards.append(0.1)
        elif length <= 800:
            rewards.append(0.0)
        else:
            rewards.append(-0.2)
    return rewards


def selection_only_rewards() -> list[Any]:
    return [format_reward, tool_selection_reward]


def full_rewards(include_hallucination_penalty: bool = True) -> list[Any]:
    rewards = [format_reward, tool_selection_reward, argument_reward]
    if include_hallucination_penalty:
        rewards.append(hallucination_penalty)
    rewards.append(brevity_reward)
    return rewards


def _argument_score(pred_args: dict[str, Any], gold_args: dict[str, Any]) -> float:
    if not gold_args:
        return 1.0 if not pred_args else 0.5

    gold_keys = set(gold_args)
    pred_keys = set(pred_args)
    missing = gold_keys - pred_keys
    extra = pred_keys - gold_keys
    matching_values = sum(1 for key in gold_keys & pred_keys if _values_equal(pred_args[key], gold_args[key]))

    key_score = len(gold_keys & pred_keys) / max(len(gold_keys), 1)
    value_score = matching_values / max(len(gold_keys), 1)
    penalty = min(0.5, 0.2 * len(missing) + 0.1 * len(extra))
    return max(-0.5, 0.4 * key_score + 0.6 * value_score - penalty)


def _values_equal(left: Any, right: Any) -> bool:
    if left == right:
        return True
    return str(left).strip().lower() == str(right).strip().lower()


def _coerce_list(value: Any, size: int, default: Any) -> list[Any]:
    if value is None:
        return [default for _ in range(size)]
    if isinstance(value, str):
        return [value for _ in range(size)]
    if isinstance(value, Iterable):
        values = list(value)
        if len(values) == size:
            return values
    return [value for _ in range(size)]


def _has_direct_answer(text: str) -> bool:
    stripped = str(text or "").strip()
    return bool(stripped) and not stripped.startswith(("{", "["))


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content")
        return str(content if content is not None else completion)
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            content = last.get("content")
            return str(content if content is not None else last)
    return str(completion or "")
