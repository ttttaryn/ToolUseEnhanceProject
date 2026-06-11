"""Parsing helpers for BFCL-style tool calls.

The parser is deliberately permissive: during GRPO the model may emit partially
valid calls, and the reward functions should grade those attempts instead of
crashing on them.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any


NO_TOOL_NAMES = {"none", "no_tool", "no tool", "null", "direct_answer"}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]


def load_jsonish(value: Any, default: Any = None) -> Any:
    """Load JSON/Python-literal strings while leaving real objects untouched."""

    if value is None:
        return default
    if isinstance(value, (dict, list, tuple)):
        return value
    if not isinstance(value, str):
        return default if default is not None else value

    text = value.strip()
    if not text:
        return default

    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass
    return default


def normalize_name(name: Any) -> str:
    return str(name or "").strip()


def parse_tool_schema(tools: Any) -> dict[str, set[str]]:
    """Return a mapping of tool name to allowed parameter names."""

    tools_obj = load_jsonish(tools, default=[])
    if isinstance(tools_obj, dict):
        tools_obj = tools_obj.get("tools") or tools_obj.get("functions") or [tools_obj]

    schema: dict[str, set[str]] = {}
    for tool in tools_obj or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = normalize_name(fn.get("name") or tool.get("name"))
        if not name:
            continue
        params = fn.get("parameters") or tool.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else {}
        allowed = set(props.keys()) if isinstance(props, dict) else set()
        schema[name] = allowed
    return schema


def parse_tool_calls(text: Any) -> list[ToolCall]:
    """Extract tool calls from model output or gold labels."""

    if text is None:
        return []
    if isinstance(text, ToolCall):
        return [text]
    obj = load_jsonish(text, default=None)
    if obj is not None and not isinstance(obj, str):
        return _calls_from_obj(obj)

    raw = str(text).strip()
    if not raw:
        return []

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        parsed = load_jsonish(fenced.group(1), default=None)
        if parsed is not None:
            return _calls_from_obj(parsed)

    for candidate in _balanced_json_candidates(raw):
        parsed = load_jsonish(candidate, default=None)
        if parsed is not None:
            calls = _calls_from_obj(parsed)
            if calls:
                return calls

    function_style_calls = _parse_function_style_calls(raw)
    if function_style_calls:
        return function_style_calls

    return []


def is_no_tool_gold(gold: Any, task_type: str | None = None) -> bool:
    task = (task_type or "").lower()
    if any(token in task for token in ("irrelevance", "no_tool", "no-tool", "non_tool")):
        return True

    if gold is None:
        return True
    if isinstance(gold, str) and gold.strip().lower() in NO_TOOL_NAMES:
        return True
    obj = load_jsonish(gold, default=gold)
    if obj in (None, [], {}, ""):
        return True
    calls = parse_tool_calls(obj)
    if not calls:
        return False
    return all(call.name.strip().lower() in NO_TOOL_NAMES for call in calls)


def _calls_from_obj(obj: Any) -> list[ToolCall]:
    if obj is None:
        return []
    if isinstance(obj, ToolCall):
        return [obj]
    if isinstance(obj, str):
        return parse_tool_calls(obj)
    if isinstance(obj, tuple):
        obj = list(obj)
    if isinstance(obj, list):
        calls: list[ToolCall] = []
        for item in obj:
            calls.extend(_calls_from_obj(item))
        return calls
    if not isinstance(obj, dict):
        return []

    if "tool_calls" in obj:
        return _calls_from_obj(obj["tool_calls"])
    if "function_call" in obj:
        return _calls_from_obj(obj["function_call"])

    fn = obj.get("function") if isinstance(obj.get("function"), dict) else None
    name = (
        obj.get("name")
        or obj.get("tool_name")
        or obj.get("function_name")
        or (fn or {}).get("name")
    )
    args = (
        obj.get("arguments")
        or obj.get("args")
        or obj.get("parameters")
        or (fn or {}).get("arguments")
        or {}
    )
    args = load_jsonish(args, default=args)
    if not isinstance(args, dict):
        args = {}
    if name:
        return [ToolCall(name=normalize_name(name), arguments=dict(args))]
    return []


def _balanced_json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    stack: list[str] = []
    start: int | None = None
    pairs = {"{": "}", "[": "]"}

    for idx, char in enumerate(text):
        if char in pairs:
            if not stack:
                start = idx
            stack.append(pairs[char])
        elif stack and char == stack[-1]:
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start : idx + 1])
                start = None
    return candidates


def _parse_call_arguments(arg_text: str) -> dict[str, Any]:
    text = arg_text.strip()
    if not text:
        return {}

    parsed = load_jsonish("{" + text + "}", default=None)
    if isinstance(parsed, dict):
        return parsed

    ast_args = _parse_keyword_arguments_with_ast(text)
    if ast_args is not None:
        return ast_args

    args: dict[str, Any] = {}
    for match in re.finditer(r"([A-Za-z_]\w*)\s*=\s*([^,]+)", text):
        key, value = match.groups()
        args[key] = load_jsonish(value.strip(), default=value.strip().strip("\"'"))
    return args


def _parse_function_style_calls(text: str) -> list[ToolCall]:
    raw = text.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
    calls: list[ToolCall] = []
    idx = 0
    while idx < len(raw):
        while idx < len(raw) and raw[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(raw):
            break
        open_idx = raw.find("(", idx)
        if open_idx == -1:
            break
        name = raw[idx:open_idx].strip()
        if not name:
            break
        close_idx = _find_matching_paren(raw, open_idx)
        if close_idx == -1:
            break
        args = _parse_call_arguments(raw[open_idx + 1 : close_idx])
        calls.append(ToolCall(name=name, arguments=args))
        idx = close_idx + 1
    return calls


def _find_matching_paren(text: str, open_idx: int) -> int:
    depth = 0
    quote: str | None = None
    escaped = False
    for idx in range(open_idx, len(text)):
        char = text[idx]
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
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _parse_keyword_arguments_with_ast(text: str) -> dict[str, Any] | None:
    try:
        expression = ast.parse(f"f({text})", mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(expression, ast.Call):
        return None
    args: dict[str, Any] = {}
    for keyword in expression.keywords:
        if keyword.arg is None:
            return None
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except Exception:
            return None
    return args
