from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.approvals import ApprovalItem
from app.config import DB_PATH
from app.db import connect_database, initialize_database
from app.memory_store import Memory, MemoryEvent, MemoryStore


MEMORY_ACTION_TYPE = "propose_memory"
MEMORY_UPDATE_ACTION_TYPE = "propose_memory_update"
MEMORY_TYPES = {"fact", "preference", "project_context"}
MAX_MEMORY_CONTENT_LENGTH = 2000
MAX_TAG_LENGTH = 80
MAX_TAGS = 20
MAX_MEMORY_TITLE_LENGTH = 120

SECRET_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk|xox[pbar]|gh[pousr])_[A-Za-z0-9_=-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"\b(?:api[_ -]?key|password|passwd|pwd|secret|token)\b"
        r"\s*(?:is|=|:)\s*\S{8,}",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class ValidatedMemoryProposal:
    memory_type: str
    content: str
    source: str
    tags: list[str]


def propose_memory(
    *,
    memory_type: str,
    content: str,
    source: str = "manual",
    tags: list[str] | None = None,
    db_path: Path = DB_PATH,
    actor: str = "user",
) -> ApprovalItem:
    path = Path(db_path)
    initialize_database(path)
    proposal = validate_memory_payload(
        {
            "memory_type": memory_type,
            "content": content,
            "source": source,
            "tags_json": tags or [],
        }
    )
    payload: dict[str, Any] = {
        "memory_type": proposal.memory_type,
        "content": proposal.content,
        "source": proposal.source,
    }
    if proposal.tags:
        payload["tags_json"] = proposal.tags

    with connect_database(path) as conn:
        try:
            conn.execute("BEGIN")
            item = _create_approval(
                conn,
                MEMORY_ACTION_TYPE,
                _memory_title(proposal),
                payload,
            )
            MemoryStore.record_event(
                conn,
                approval_id=item.id,
                event_type="proposal_created",
                actor=actor,
                details={
                    "memory_type": proposal.memory_type,
                    "source": proposal.source,
                },
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return item


def propose_memory_update(
    *,
    memory_id: int,
    memory_type: str,
    content: str,
    source: str = "manual",
    tags: list[str] | None = None,
    reason: str | None = None,
    db_path: Path = DB_PATH,
    actor: str = "user",
) -> ApprovalItem:
    path = Path(db_path)
    initialize_database(path)
    replacement = validate_memory_payload(
        {
            "memory_type": memory_type,
            "content": content,
            "source": source,
            "tags_json": tags or [],
        }
    )
    normalized_reason = reason.strip() if reason else ""

    with connect_database(path) as conn:
        try:
            conn.execute("BEGIN")
            current = MemoryStore.get_for_update(conn, memory_id)
            _require_active_memory_for_update(current, memory_id)
            payload = _memory_update_payload(current, replacement, normalized_reason)
            item = _create_approval(
                conn,
                MEMORY_UPDATE_ACTION_TYPE,
                _memory_update_title(current, replacement),
                payload,
            )
            details = {
                "memory_id": current.id,
                "replacement_memory_type": replacement.memory_type,
                "source": replacement.source,
            }
            if normalized_reason:
                details["reason"] = normalized_reason
            MemoryStore.record_event(
                conn,
                memory_id=current.id,
                approval_id=item.id,
                event_type="update_proposal_created",
                actor=actor,
                details=details,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return item


def approve_memory_proposal(
    db_path: Path = DB_PATH, approval_id: int = 0, actor: str = "user"
) -> ApprovalItem:
    path = Path(db_path)
    initialize_database(path)
    with connect_database(path) as conn:
        try:
            conn.execute("BEGIN")
            item = _get_approval_for_update(conn, approval_id)
            _require_memory_action(item)
            proposal = validate_memory_payload(item.payload)
            memory_id, created = MemoryStore.create_active_memory_once(
                conn,
                approval_id=item.id,
                memory_type=proposal.memory_type,
                content=proposal.content,
                source=proposal.source,
                tags=proposal.tags,
            )
            updated = _update_approval_status(conn, item.id, "approved")
            if created:
                MemoryStore.record_event(
                    conn,
                    memory_id=memory_id,
                    approval_id=item.id,
                    event_type="approved",
                    actor=actor,
                    details={
                        "memory_type": proposal.memory_type,
                        "source": proposal.source,
                    },
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return updated


def approve_memory_update_proposal(
    db_path: Path = DB_PATH, approval_id: int = 0, actor: str = "user"
) -> ApprovalItem:
    path = Path(db_path)
    initialize_database(path)
    with connect_database(path) as conn:
        try:
            conn.execute("BEGIN")
            item = _get_approval_for_update(conn, approval_id)
            _require_memory_update_action(item)
            existing = MemoryStore.get_by_approval_id_for_update(conn, item.id)
            if existing is not None:
                updated = _update_approval_status(conn, item.id, "approved")
                conn.commit()
                return updated

            update = validate_memory_update_payload(item.payload)
            current = MemoryStore.get_for_update(conn, update["memory_id"])
            _require_active_memory_for_update(current, update["memory_id"])
            _require_memory_snapshot_match(current, update)
            replacement = validate_memory_payload(
                {
                    "memory_type": update["replacement_memory_type"],
                    "content": update["replacement_content"],
                    "source": update["replacement_source"],
                    "tags_json": update["replacement_tags"],
                }
            )

            archive_reason = f"Replaced by approval #{item.id}"
            if update.get("reason"):
                archive_reason = f"{archive_reason}: {update['reason']}"
            archived = MemoryStore.archive_active_memory(
                conn,
                memory_id=current.id,
                reason=archive_reason,
            )
            MemoryStore.record_event(
                conn,
                memory_id=archived.id,
                approval_id=item.id,
                event_type="memory_archived_for_replacement",
                actor=actor,
                details={"replacement_approval_id": item.id},
            )
            memory_id, created = MemoryStore.create_active_memory_once(
                conn,
                approval_id=item.id,
                memory_type=replacement.memory_type,
                content=replacement.content,
                source=replacement.source,
                tags=replacement.tags,
            )
            updated = _update_approval_status(conn, item.id, "approved")
            if created:
                MemoryStore.record_event(
                    conn,
                    memory_id=memory_id,
                    approval_id=item.id,
                    event_type="memory_replaced",
                    actor=actor,
                    details={
                        "archived_memory_id": archived.id,
                        "replacement_memory_type": replacement.memory_type,
                        "source": replacement.source,
                    },
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return updated


def reject_memory_proposal(
    db_path: Path = DB_PATH, approval_id: int = 0, actor: str = "user"
) -> ApprovalItem:
    path = Path(db_path)
    initialize_database(path)
    with connect_database(path) as conn:
        try:
            conn.execute("BEGIN")
            item = _get_approval_for_update(conn, approval_id)
            _require_memory_action(item)
            updated = _update_approval_status(conn, item.id, "rejected")
            MemoryStore.record_event(
                conn,
                approval_id=item.id,
                event_type="rejected",
                actor=actor,
                details={},
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return updated


def reject_memory_update_proposal(
    db_path: Path = DB_PATH, approval_id: int = 0, actor: str = "user"
) -> ApprovalItem:
    path = Path(db_path)
    initialize_database(path)
    with connect_database(path) as conn:
        try:
            conn.execute("BEGIN")
            item = _get_approval_for_update(conn, approval_id)
            _require_memory_update_action(item)
            update = validate_memory_update_payload(item.payload)
            updated = _update_approval_status(conn, item.id, "rejected")
            MemoryStore.record_event(
                conn,
                memory_id=update["memory_id"],
                approval_id=item.id,
                event_type="update_rejected",
                actor=actor,
                details={},
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return updated


def list_active_memories(db_path: Path = DB_PATH) -> list[Memory]:
    return MemoryStore(Path(db_path)).list_active()


def list_recent_memory_events(
    db_path: Path = DB_PATH, limit: int = 20
) -> list[MemoryEvent]:
    return MemoryStore(Path(db_path)).list_recent_events(limit)


def archive_memory(
    db_path: Path = DB_PATH,
    memory_id: int = 0,
    *,
    reason: str | None = None,
    actor: str = "user",
) -> Memory:
    path = Path(db_path)
    initialize_database(path)
    with connect_database(path) as conn:
        try:
            conn.execute("BEGIN")
            memory = MemoryStore.archive_active_memory(
                conn,
                memory_id=memory_id,
                reason=reason,
            )
            details = {}
            if memory.archive_reason:
                details["reason"] = memory.archive_reason
            MemoryStore.record_event(
                conn,
                memory_id=memory.id,
                approval_id=memory.approval_id,
                event_type="memory_archived",
                actor=actor,
                details=details,
            )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return memory


def validate_memory_payload(payload: dict[str, Any]) -> ValidatedMemoryProposal:
    memory_type = str(payload.get("memory_type", "")).strip()
    if memory_type not in MEMORY_TYPES:
        raise ValueError("Memory proposal has an unsupported memory_type.")

    content = str(payload.get("content", "")).strip()
    if not content:
        raise ValueError("Memory proposal content is required.")
    if len(content) > MAX_MEMORY_CONTENT_LENGTH:
        raise ValueError("Memory proposal content is too long.")
    if contains_obvious_secret(content):
        raise ValueError("Memory proposal content appears to contain a secret.")

    source = str(payload.get("source", "chat")).strip() or "chat"
    tags = _normalize_tags(payload.get("tags", payload.get("tags_json", [])))
    return ValidatedMemoryProposal(
        memory_type=memory_type,
        content=content,
        source=source,
        tags=tags,
    )


def validate_memory_update_payload(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        memory_id = int(payload.get("memory_id", 0))
    except (TypeError, ValueError) as exc:
        raise ValueError("Memory update proposal memory_id is required.") from exc
    if memory_id <= 0:
        raise ValueError("Memory update proposal memory_id is required.")

    replacement = validate_memory_payload(
        {
            "memory_type": payload.get("replacement_memory_type"),
            "content": payload.get("replacement_content"),
            "source": payload.get("replacement_source", "manual"),
            "tags_json": payload.get("replacement_tags_json", []),
        }
    )
    original_tags = _normalize_tags(payload.get("original_tags_json", []))
    original_memory_type = str(payload.get("original_memory_type", "")).strip()
    original_content = str(payload.get("original_content", "")).strip()
    original_source = str(payload.get("original_source", "")).strip()
    if original_memory_type not in MEMORY_TYPES:
        raise ValueError("Memory update proposal has an invalid original memory_type.")
    if not original_content:
        raise ValueError("Memory update proposal original content is required.")
    if not original_source:
        raise ValueError("Memory update proposal original source is required.")

    return {
        "memory_id": memory_id,
        "original_memory_type": original_memory_type,
        "original_content": original_content,
        "original_source": original_source,
        "original_tags": original_tags,
        "replacement_memory_type": replacement.memory_type,
        "replacement_content": replacement.content,
        "replacement_source": replacement.source,
        "replacement_tags": replacement.tags,
        "reason": str(payload.get("reason", "")).strip(),
    }


def contains_obvious_secret(content: str) -> bool:
    if any(pattern.search(content) for pattern in SECRET_PATTERNS):
        return True
    return _contains_payment_card_number(content)


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("Memory proposal tags_json must be a JSON list.") from exc
    if not isinstance(value, list):
        raise ValueError("Memory proposal tags must be a list.")
    tags: list[str] = []
    for item in value:
        tag = str(item).strip()
        if not tag:
            continue
        if len(tag) > MAX_TAG_LENGTH:
            raise ValueError("Memory proposal tag is too long.")
        tags.append(tag)
    return tags[:MAX_TAGS]


def _contains_payment_card_number(content: str) -> bool:
    for match in re.finditer(r"(?:\d[ -]?){13,19}", content):
        digits = re.sub(r"\D", "", match.group(0))
        if 13 <= len(digits) <= 19 and _luhn_valid(digits):
            return True
    return False


def _luhn_valid(digits: str) -> bool:
    total = 0
    reverse_digits = digits[::-1]
    for index, char in enumerate(reverse_digits):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _memory_title(proposal: ValidatedMemoryProposal) -> str:
    normalized = " ".join(proposal.content.split())
    title = f"Remember {proposal.memory_type}: {normalized}"
    if len(title) <= MAX_MEMORY_TITLE_LENGTH:
        return title
    return f"{title[: MAX_MEMORY_TITLE_LENGTH - 3].rstrip()}..."


def _memory_update_title(current: Memory, replacement: ValidatedMemoryProposal) -> str:
    normalized = " ".join(replacement.content.split())
    title = f"Replace memory #{current.id} with {replacement.memory_type}: {normalized}"
    if len(title) <= MAX_MEMORY_TITLE_LENGTH:
        return title
    return f"{title[: MAX_MEMORY_TITLE_LENGTH - 3].rstrip()}..."


def _memory_update_payload(
    current: Memory,
    replacement: ValidatedMemoryProposal,
    reason: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "memory_id": current.id,
        "original_memory_type": current.memory_type,
        "original_content": current.content,
        "original_source": current.source,
        "original_tags_json": current.tags,
        "replacement_memory_type": replacement.memory_type,
        "replacement_content": replacement.content,
        "replacement_source": replacement.source,
    }
    if replacement.tags:
        payload["replacement_tags_json"] = replacement.tags
    if reason:
        payload["reason"] = reason
    return payload


def _require_memory_action(item: ApprovalItem) -> None:
    if item.action_type != MEMORY_ACTION_TYPE:
        raise ValueError("Only propose_memory approvals can create memory.")


def _require_memory_update_action(item: ApprovalItem) -> None:
    if item.action_type != MEMORY_UPDATE_ACTION_TYPE:
        raise ValueError("Only propose_memory_update approvals can replace memory.")


def _require_active_memory_for_update(memory: Memory | None, memory_id: int) -> None:
    if memory is None:
        raise KeyError(f"Memory {memory_id} not found.")
    if memory.status != "active":
        raise ValueError(f"Memory {memory_id} is not active.")


def _require_memory_snapshot_match(memory: Memory, update: dict[str, Any]) -> None:
    if (
        memory.memory_type != update["original_memory_type"]
        or memory.content != update["original_content"]
        or memory.source != update["original_source"]
        or memory.tags != update["original_tags"]
    ):
        raise ValueError("Memory update proposal no longer matches the active memory.")


def _create_approval(
    conn: sqlite3.Connection,
    action_type: str,
    title: str,
    payload: dict[str, Any],
) -> ApprovalItem:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """
        INSERT INTO approvals (action_type, title, payload, status, created_at, updated_at)
        VALUES (?, ?, ?, 'pending', ?, ?)
        """,
        (action_type, title, json.dumps(payload, sort_keys=True), now, now),
    )
    return _get_approval_for_update(conn, int(cur.lastrowid))


def _get_approval_for_update(
    conn: sqlite3.Connection, approval_id: int
) -> ApprovalItem:
    row = conn.execute(
        "SELECT * FROM approvals WHERE id = ?", (approval_id,)
    ).fetchone()
    if row is None:
        raise KeyError(f"Approval item {approval_id} not found.")
    return ApprovalItem(
        id=int(row["id"]),
        action_type=str(row["action_type"]),
        title=str(row["title"]),
        payload=json.loads(str(row["payload"])),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _update_approval_status(
    conn: sqlite3.Connection, approval_id: int, status: str
) -> ApprovalItem:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "UPDATE approvals SET status = ?, updated_at = ? WHERE id = ?",
        (status, now, approval_id),
    )
    return _get_approval_for_update(conn, approval_id)
