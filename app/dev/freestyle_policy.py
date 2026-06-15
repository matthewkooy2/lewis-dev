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

# A file that *defines* the approval-store class is part of the approval
# record implementation and stays out of model context and likely-file
# suggestions entirely. A file that merely imports or calls it (routes,
# dashboards, tests) stays inspectable and proposable; only operations that
# touch its approval-sensitive regions are blocked, via the region markers
# below.
APPROVAL_STORE_DEFINITION_PATTERN = re.compile(
    r"\bclass\s+" + APPROVAL_STORE_CLASS_NAME + r"\b"
)

# Text markers for approval-sensitive code regions inside otherwise editable
# files: the class itself, the approval-store route helper, approval
# lifecycle mutators, approved-execution entry points, and imports of the
# approval modules. Edit operations whose edited text matches any marker are
# refused (fail closed); edits elsewhere in the same file remain allowed.
APPROVAL_SENSITIVE_REGION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "approval-store class reference",
        re.compile(r"\b" + APPROVAL_STORE_CLASS_NAME + r"\b"),
    ),
    ("approval-store helper reference", re.compile(r"\b_approval_store\b")),
    (
        "approval mutation call",
        re.compile(r"\b(?:approve|reject)_approval\b"),
    ),
    ("approved-execution call", re.compile(r"\bexecute_approved\b")),
    (
        "approval module reference",
        re.compile(r"\bapp[./]approval(?:s|_service)\b"),
    ),
)

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


def defines_approval_store(text: str) -> bool:
    """Report whether text defines the approval-store class itself.

    References (imports, calls, type annotations) intentionally do not
    match: route and dashboard files that merely use the store must stay
    inspectable for unrelated tasks, with their approval-sensitive regions
    protected separately at the edit-operation level.
    """
    return APPROVAL_STORE_DEFINITION_PATTERN.search(text) is not None


def approval_sensitive_region_labels(text: str) -> tuple[str, ...]:
    """Return the labels of every approval-sensitive marker found in text."""
    return tuple(
        label
        for label, pattern in APPROVAL_SENSITIVE_REGION_PATTERNS
        if pattern.search(text)
    )


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
