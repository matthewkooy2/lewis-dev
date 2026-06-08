from __future__ import annotations

from typing import Any

from app.text_tool_calls import strip_tool_call_markup


def assistant_tool_call_api_message(
    source_message: object,
    tool_calls: list[Any],
) -> dict[str, Any]:
    """Build private API history for a DeepSeek assistant tool-call turn."""
    message: dict[str, Any] = {
        "role": "assistant",
        "content": strip_tool_call_markup(
            str(getattr(source_message, "content", "") or "")
        )
        or None,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in tool_calls
        ],
    }
    reasoning_content = _reasoning_content(source_message)
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return message


def _reasoning_content(source_message: object) -> str | None:
    reasoning_content = getattr(source_message, "reasoning_content", None)
    if isinstance(reasoning_content, str):
        return reasoning_content

    model_dump = getattr(source_message, "model_dump", None)
    if not callable(model_dump):
        return None
    dumped = model_dump()
    if not isinstance(dumped, dict):
        return None
    reasoning_content = dumped.get("reasoning_content")
    return reasoning_content if isinstance(reasoning_content, str) else None
