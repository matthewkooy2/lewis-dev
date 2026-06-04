from __future__ import annotations

from pathlib import Path

from app.approvals import ApprovalItem, ApprovalStore
from app.memory_service import (
    MEMORY_ACTION_TYPE,
    MEMORY_UPDATE_ACTION_TYPE,
    approve_memory_proposal,
    approve_memory_update_proposal,
    reject_memory_proposal,
    reject_memory_update_proposal,
)


def approve_approval(store: ApprovalStore, approval_id: int) -> ApprovalItem:
    item = store.get(approval_id)
    if item is None:
        raise KeyError(f"Approval item {approval_id} not found.")
    if item.action_type == MEMORY_ACTION_TYPE:
        return approve_memory_proposal(Path(store.db_path), approval_id, actor="user")
    if item.action_type == MEMORY_UPDATE_ACTION_TYPE:
        return approve_memory_update_proposal(
            Path(store.db_path), approval_id, actor="user"
        )
    return store.update_status(approval_id, "approved")


def reject_approval(store: ApprovalStore, approval_id: int) -> ApprovalItem:
    item = store.get(approval_id)
    if item is None:
        raise KeyError(f"Approval item {approval_id} not found.")
    if item.action_type == MEMORY_ACTION_TYPE:
        return reject_memory_proposal(Path(store.db_path), approval_id, actor="user")
    if item.action_type == MEMORY_UPDATE_ACTION_TYPE:
        return reject_memory_update_proposal(
            Path(store.db_path), approval_id, actor="user"
        )
    return store.update_status(approval_id, "rejected")
