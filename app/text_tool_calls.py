from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from typing import Any


_DSML_INVOKE_PATTERN = re.compile(
    r"<[^>]*DSML[^>]*invoke\b(?P<attributes>[^>]*)>"
    r"(?P<body>.*?)"
    r"</[^>]*DSML[^>]*invoke\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DSML_PARAMETER_PATTERN = re.compile(
    r"<[^>]*DSML[^>]*parameter\b(?P<attributes>[^>]*)>"
    r"(?P<body>.*?)"
    r"</[^>]*DSML[^>]*parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DSML_SELF_CLOSING_PARAMETER_PATTERN = re.compile(
    r"<[^>]*DSML[^>]*parameter\b(?P<attributes>[^>]*)/\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SIMPLE_TOOL_PATTERN = re.compile(
    r"<(?P<name>[a-zA-Z_][\w.-]*)\b[^>]*>"
    r"(?P<body>.*?)"
    r"</(?P=name)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_SIMPLE_ARGUMENT_PATTERN = re.compile(
    r"<(?P<name>[a-zA-Z_][\w.-]*)\b[^>]*>"
    r"(?P<value>.*?)"
    r"</(?P=name)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_ATTRIBUTE_PATTERN = re.compile(
    r"(?P<name>[a-zA-Z_][\w.-]*)\s*=\s*"
    r"(?P<quote>[\"'])(?P<value>.*?)(?P=quote)",
    re.DOTALL,
)
_DSML_BLOCK_PATTERN = re.compile(
    r"<[^>]*DSML[^>]*(?:tool_calls|function_calls)\b[^>]*>.*?"
    r"</[^>]*DSML[^>]*(?:tool_calls|function_calls)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DSML_TAG_PATTERN = re.compile(r"</?[^>]*DSML[^>]*>", re.IGNORECASE)
_TOOL_BLOCK_PATTERN = re.compile(
    r"<(?P<name>[a-zA-Z_][\w.-]*(?:tool|_file|_files|_change)[\w.-]*)\b[^>]*>"
    r".*?</(?P=name)\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_CALLS_BLOCK_PATTERN = re.compile(
    r"<function_calls\b[^>]*>.*?</function_calls\s*>",
    re.IGNORECASE | re.DOTALL,
)
_FUNCTION_CALLS_TAG_PATTERN = re.compile(r"</?function_calls\s*>", re.IGNORECASE)
_INVOKE_BLOCK_PATTERN = re.compile(
    r"<invoke\b[^>]*>.*?</invoke\s*>",
    re.IGNORECASE | re.DOTALL,
)
_INVOKE_TAG_PATTERN = re.compile(r"</?invoke\b[^>]*/?>", re.IGNORECASE)
_PARAMETER_BLOCK_PATTERN = re.compile(
    r"<parameter\b[^>]*>.*?</parameter\s*>",
    re.IGNORECASE | re.DOTALL,
)
_PARAMETER_TAG_PATTERN = re.compile(r"</?parameter\b[^>]*/?>", re.IGNORECASE)


@dataclass(frozen=True)
class TextToolCall:
    name: str
    arguments: dict[str, Any]


def extract_text_tool_calls(text: str) -> list[TextToolCall]:
    calls = _extract_dsml_calls(text)
    if calls:
        return calls
    return _extract_simple_calls(text)


def strip_tool_call_markup(text: str) -> str:
    cleaned = _DSML_BLOCK_PATTERN.sub("", text)
    cleaned = _TOOL_BLOCK_PATTERN.sub("", cleaned)
    cleaned = _FUNCTION_CALLS_BLOCK_PATTERN.sub("", cleaned)
    cleaned = _INVOKE_BLOCK_PATTERN.sub("", cleaned)
    cleaned = _PARAMETER_BLOCK_PATTERN.sub("", cleaned)
    cleaned = _DSML_TAG_PATTERN.sub("", cleaned)
    cleaned = _FUNCTION_CALLS_TAG_PATTERN.sub("", cleaned)
    cleaned = _INVOKE_TAG_PATTERN.sub("", cleaned)
    cleaned = _PARAMETER_TAG_PATTERN.sub("", cleaned)
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).strip()


def _extract_dsml_calls(text: str) -> list[TextToolCall]:
    calls: list[TextToolCall] = []
    for match in _DSML_INVOKE_PATTERN.finditer(text):
        attributes = _attributes(match.group("attributes"))
        name = attributes.get("name", "").strip()
        if not name:
            continue
        calls.append(
            TextToolCall(
                name=name,
                arguments=_dsml_arguments(match.group("body")),
            )
        )
    return calls


def _dsml_arguments(body: str) -> dict[str, Any]:
    stripped = body.strip()
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed

    arguments: dict[str, Any] = {}
    parameter_matches = [
        *_DSML_PARAMETER_PATTERN.finditer(body),
        *_DSML_SELF_CLOSING_PARAMETER_PATTERN.finditer(body),
    ]
    for match in parameter_matches:
        attributes = _attributes(match.group("attributes"))
        name = attributes.pop("name", "").strip()
        if not name:
            continue
        value = _parameter_value(attributes, match.groupdict().get("body", ""))
        arguments[name] = value
    return arguments


def _extract_simple_calls(text: str) -> list[TextToolCall]:
    calls: list[TextToolCall] = []
    for match in _SIMPLE_TOOL_PATTERN.finditer(text):
        name = match.group("name")
        body = match.group("body")
        arguments = {
            argument.group("name"): html.unescape(argument.group("value").strip())
            for argument in _SIMPLE_ARGUMENT_PATTERN.finditer(body)
        }
        if arguments:
            calls.append(TextToolCall(name=name, arguments=arguments))
    return calls


def _attributes(text: str) -> dict[str, str]:
    return {
        match.group("name"): html.unescape(match.group("value"))
        for match in _ATTRIBUTE_PATTERN.finditer(text)
    }


def _parameter_value(attributes: dict[str, str], body: str) -> Any:
    for key in ("value", "string"):
        if key in attributes:
            return attributes[key]
    value = html.unescape(_DSML_TAG_PATTERN.sub("", body).strip())
    if not value:
        return ""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value
