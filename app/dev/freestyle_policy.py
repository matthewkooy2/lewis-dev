"""Shared static policy data and pure predicates for freestyle modules.

Centralizes the excluded-path categories, approval-record markers,
sensitive-content patterns, recovery guidance, and no-integration statement
text that the freestyle slices previously duplicated. This module performs
no I/O, subprocess, model, network, or job-state access, and imports no
other freestyle module, so every slice can depend on it safely.
"""

from __future__ import annotations

import re

# Directory names whose contents are generated or cache state; freestyle
# slices never read them for model context and never accept them as edit
# targets. The review slice additionally excludes a plain "cache" part.
GENERATED_OR_CACHE_PARTS = frozenset(
    {
        ".cache",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "coverage",
        "dist",
        "generated",
        "htmlcov",
        "node_modules",
        "target",
        "vendor",
    }
)

APPROVAL_STORE_PATH_NAMES = frozenset(
    {
        "approval_service.py",
        "approvals.py",
        "test_approvals.py",
    }
)
# Concatenated so content scans for the class name never match the policy
# module's own source text.
APPROVAL_STORE_CLASS_NAME = "Approval" "Store"
APPROVAL_STORE_TOKEN = "approvalstore"

SENSITIVE_CONTENT_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"\b(?:api[_-]?key|credential|password|passwd|secret|token)\b"
        r"\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']",
        re.IGNORECASE,
    ),
)

NO_INTEGRATION_STATEMENT_RUN = (
    "No commit, merge, push, pull, fetch, remote operation, branch switch, PR "
    "creation, or auto-merge occurred."
)
NO_INTEGRATION_STATEMENT_MAINTENANCE = (
    "No commit, merge, push, pull, fetch, remote operation, PR creation, or "
    "auto-merge occurred."
)
NO_INTEGRATION_STATEMENT_HANDOFF = (
    "Lewis has not committed, merged, pushed, pulled, fetched, opened a PR, "
    "or auto-merged this job."
)


def contains_sensitive_content(text: str) -> bool:
    return any(pattern.search(text) for pattern in SENSITIVE_CONTENT_PATTERNS)


def recovery_message(action: str) -> str:
    """Build the standard stale/unsafe-state recovery instruction."""
    return (
        f"Recover or verify the isolated worktree and job metadata before {action}."
    )


def is_job_id(value: str) -> bool:
    """Return whether one value is a 32-character lowercase hex job id."""
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )
