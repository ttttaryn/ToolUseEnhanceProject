"""Dataset conversion helpers for BFCL-style records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from .tool_parser import is_no_tool_gold, load_jsonish


SYSTEM_PROMPT = (
    "You are a tool-use assistant. Given a user request and a list of available "
    "tools, either return a JSON tool call with the selected tool name and "
    "arguments, or answer directly if no tool is needed. Do not invent tools or "
    "parameters."
)


def iter_records(path: Path) -> Iterable[dict[str, Any]]:
    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.suffix.lower() in {".json", ".jsonl"}:
                yield from iter_records(child)
        return

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
        return

    with path.open("r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item
    elif isinstance(obj, dict):
        data = obj.get("data") or obj.get("records") or obj.get("examples")
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield item
        else:
            yield obj


def convert_record(record: dict[str, Any]) -> dict[str, Any] | None:
    user_request = _extract_user_request(record)
    tools = _first_present(record, "tools", "tool_list", "functions", "function", "available_tools")
    ground_truth = _first_present(
        record,
        "ground_truth",
        "answer",
        "answers",
        "possible_answer",
        "gold",
        "reference",
        "expected",
    )
    task_type = str(_first_present(record, "task_type", "category", "test_category", "data_source") or "")

    if not user_request or tools is None:
        return None

    no_tool = is_no_tool_gold(ground_truth, task_type)
    normalized_task = "irrelevance" if no_tool else (task_type or "single_turn")
    prompt = build_prompt(user_request=user_request, tools=tools)
    return {
        "prompt": prompt,
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        "tools": json.dumps(tools, ensure_ascii=False),
        "task_type": normalized_task,
        "source_id": str(_first_present(record, "id", "idx", "question_id") or ""),
    }


def build_prompt(user_request: str, tools: Any) -> str:
    tools_json = json.dumps(load_jsonish(tools, default=tools), ensure_ascii=False, indent=2)
    return (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        "<|user|>\n"
        f"Available tools:\n{tools_json}\n\n"
        f"User request:\n{user_request}\n"
        "<|assistant|>\n"
    )


def write_jsonl(records: Iterable[dict[str, Any]], output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _extract_user_request(record: dict[str, Any]) -> str:
    for key in ("question", "prompt", "user", "query", "instruction"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = record.get("messages") or record.get("conversation")
    messages = load_jsonish(messages, default=messages)
    if isinstance(messages, list):
        for message in reversed(messages):
            if not isinstance(message, dict):
                continue
            role = str(message.get("role", "")).lower()
            content = message.get("content")
            if role in {"user", "human"} and isinstance(content, str):
                return content.strip()
    return ""


def _first_present(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None
