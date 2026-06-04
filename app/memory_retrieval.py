from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from app.config import DB_PATH
from app.memory_store import Memory, MemoryStore


DEFAULT_MEMORY_LIMIT = 5
DEFAULT_MEMORY_CONTENT_LIMIT = 240
DEFAULT_CONTEXT_LIMIT = 1400
MIN_RELEVANCE_TOKEN_LENGTH = 3


@dataclass(frozen=True)
class ActiveMemoryContext:
    prompt: str
    memory_count: int = 0


def retrieve_active_memories(
    user_input: str,
    *,
    db_path: Path = DB_PATH,
    limit: int = DEFAULT_MEMORY_LIMIT,
    content_limit: int = DEFAULT_MEMORY_CONTENT_LIMIT,
) -> list[Memory]:
    """Return approved active memories for transient model context only."""
    memories = MemoryStore(Path(db_path)).list_active()
    capped_limit = max(0, limit)
    if capped_limit == 0:
        return []

    query_tokens = _tokens(user_input)
    if query_tokens:
        scored = [
            (_relevance_score(memory, query_tokens), index, memory)
            for index, memory in enumerate(memories)
        ]
        relevant = [
            (score, index, memory) for score, index, memory in scored if score > 0
        ]
        if relevant:
            relevant.sort(key=lambda item: (-item[0], item[1]))
            return [
                _with_capped_content(memory, content_limit)
                for _, _, memory in relevant[:capped_limit]
            ]

    return [
        _with_capped_content(memory, content_limit)
        for memory in memories[:capped_limit]
    ]


def format_active_memory_context(
    user_input: str,
    *,
    db_path: Path = DB_PATH,
    limit: int = DEFAULT_MEMORY_LIMIT,
    content_limit: int = DEFAULT_MEMORY_CONTENT_LIMIT,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> str:
    return build_active_memory_context(
        user_input,
        db_path=db_path,
        limit=limit,
        content_limit=content_limit,
        context_limit=context_limit,
    ).prompt


def build_active_memory_context(
    user_input: str,
    *,
    db_path: Path = DB_PATH,
    limit: int = DEFAULT_MEMORY_LIMIT,
    content_limit: int = DEFAULT_MEMORY_CONTENT_LIMIT,
    context_limit: int = DEFAULT_CONTEXT_LIMIT,
) -> ActiveMemoryContext:
    memories = retrieve_active_memories(
        user_input,
        db_path=db_path,
        limit=limit,
        content_limit=content_limit,
    )
    if not memories:
        return ActiveMemoryContext("")

    header = (
        "Approved active local memories for this chat turn only. "
        "Use them as transient context; do not save or quote this block as chat history. "
        "Do not claim to remember archived, rejected, or pending memory proposals; "
        "they are not included here."
    )
    lines = [header]
    for memory in memories:
        line = (
            f"- #{memory.id} {memory.memory_type}: {_normalize_content(memory.content)}"
        )
        candidate = "\n".join([*lines, line])
        if len(candidate) > max(context_limit, 0):
            break
        lines.append(line)
    if len(lines) <= 1:
        return ActiveMemoryContext("")
    return ActiveMemoryContext("\n".join(lines), len(lines) - 1)


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) >= MIN_RELEVANCE_TOKEN_LENGTH
    }


def _relevance_score(memory: Memory, query_tokens: set[str]) -> int:
    haystack = " ".join(
        [
            memory.memory_type,
            memory.content,
            memory.source,
            " ".join(memory.tags),
        ]
    )
    memory_tokens = _tokens(haystack)
    return len(query_tokens & memory_tokens)


def _with_capped_content(memory: Memory, content_limit: int) -> Memory:
    content = _truncate(_normalize_content(memory.content), max(content_limit, 0))
    return Memory(
        id=memory.id,
        approval_id=memory.approval_id,
        memory_type=memory.memory_type,
        content=content,
        source=memory.source,
        tags=memory.tags,
        status=memory.status,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        archived_at=memory.archived_at,
        archive_reason=memory.archive_reason,
    )


def _normalize_content(value: str) -> str:
    return " ".join(value.split())


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return "." * limit
    return f"{value[: limit - 3].rstrip()}..."
