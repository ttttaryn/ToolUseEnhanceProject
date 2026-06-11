"""Conversion helpers for Team-ACE/ToolACE records."""

from __future__ import annotations

import json
import re
from typing import Any

from .data_prep import build_prompt
from .tool_parser import ToolCall, load_jsonish, parse_tool_calls


def convert_toolace_record(record: dict[str, Any]) -> dict[str, Any] | None:
    system = str(record.get("system") or "")
    conversations = normalize_conversations(record.get("conversations") or record.get("messages") or [])
    user_request = first_message(conversations, {"human", "user"})
    assistant_text = first_message(conversations, {"gpt", "assistant"}, after_role={"human", "user"})
    if not user_request or assistant_text is None:
        return None

    gold_calls = parse_tool_calls(assistant_text)
    tools = extract_tools_from_system(system, gold_calls)
    if not tools and not gold_calls:
        return None

    ground_truth: Any
    task_type: str
    if gold_calls:
        ground_truth = [
            {"name": call.name, "arguments": call.arguments}
            for call in gold_calls
        ]
        task_type = "toolace"
    else:
        ground_truth = "none"
        task_type = "irrelevance"

    return {
        "prompt": build_prompt(user_request=user_request, tools=tools),
        "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        "tools": json.dumps(tools, ensure_ascii=False),
        "task_type": task_type,
        "source_id": str(record.get("id") or record.get("idx") or ""),
    }


def normalize_conversations(value: Any) -> list[dict[str, str]]:
    value = load_jsonish(value, default=value)
    if not isinstance(value, list):
        return []
    normalized: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            continue
        role = str(message.get("from") or message.get("role") or "").lower()
        content = message.get("value") if "value" in message else message.get("content")
        if role and content is not None:
            normalized.append({"role": role, "content": str(content)})
    return normalized


def first_message(
    conversations: list[dict[str, str]],
    roles: set[str],
    after_role: set[str] | None = None,
) -> str | None:
    seen_after = after_role is None
    for message in conversations:
        role = message["role"]
        if after_role and role in after_role:
            seen_after = True
            continue
        if seen_after and role in roles:
            return message["content"]
    return None


def extract_tools_from_system(system: str, gold_calls: list[ToolCall]) -> list[dict[str, Any]]:
    tools = extract_json_tools(system)
    if tools:
        return tools
    return [minimal_tool_schema(call) for call in gold_calls]


def extract_json_tools(system: str) -> list[dict[str, Any]]:
    for candidate in balanced_json_candidates(system):
        parsed = load_jsonish(candidate, default=None)
        tools = normalize_tool_container(parsed)
        if tools:
            return tools
    return []


def balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
    quote: str | None = None
    escaped = False
    pairs = {"{": "}", "[": "]"}
    for idx, char in enumerate(text):
        if quote:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
        elif char in pairs:
            if not stack:
                start = idx
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : idx + 1])
                start = None
    return candidates


def normalize_tool_container(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("tools") or value.get("functions") or value.get("apis") or [value]
    if not isinstance(value, list):
        return []
    tools: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        fn = item.get("function") if isinstance(item.get("function"), dict) else item
        name = fn.get("name") or item.get("name")
        if not name:
            continue
        parameters = fn.get("parameters") or item.get("parameters") or {}
        if not isinstance(parameters, dict):
            parameters = {"type": "object", "properties": {}}
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": str(name),
                    "description": str(fn.get("description") or item.get("description") or ""),
                    "parameters": normalize_parameters(parameters),
                },
            }
        )
    return tools


def minimal_tool_schema(call: ToolCall) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": call.name,
            "description": "ToolACE tool inferred from the gold tool call.",
            "parameters": {
                "type": "object",
                "properties": {
                    key: {"type": infer_json_type(value)}
                    for key, value in call.arguments.items()
                },
                "required": list(call.arguments),
            },
        },
    }


def normalize_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    properties = parameters.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = parameters.get("required")
    if not isinstance(required, list):
        required = []
    return {
        "type": parameters.get("type") or "object",
        "properties": properties,
        "required": required,
    }


def infer_json_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"
