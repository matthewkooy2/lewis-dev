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
APPROVAL_STORE_CLASS_NAME = "ApprovalStore"
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

# The route-level ``def _approval_store(...)`` helper definition. Editing the
# helper itself (not merely calling it) rewires every route's store access and
# is always refused.
APPROVAL_STORE_HELPER_DEFINITION_PATTERN = re.compile(r"\bdef\s+_approval_store\b")

# Quoted approval route-path literals: the ``/approvals`` dashboard route, its
# approve/reject/execute-approved sub-routes, and ``/approvals`` return-to
# defaults. An edit whose text contains one is changing an approval route and
# is refused even when the approval-store helper call is left untouched.
APPROVAL_ROUTE_PATH_PATTERN = re.compile(r"""["']/approvals(?:/[^"'\s]*)?["']""")

# The route-level ``_approval_store`` helper *reference*, capturing the
# immediate single-level call argument list when one is present. A replace_text
# edit must preserve every such reference byte-for-byte, so the multiset of
# matches in old_text and new_text is compared rather than merely detected.
APPROVAL_STORE_HELPER_REFERENCE_PATTERN = re.compile(
    r"_approval_store\s*\([^()]*\)|_approval_store\b"
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

# The subset of approval-sensitive markers that must never appear in an edited
# region at all, whether the edit adds, removes, or merely preserves them: the
# approval-store class, the approval lifecycle mutators, approved execution,
# and approval-module imports. The route-level ``_approval_store`` helper
# reference is deliberately excluded; many non-approval feature routes call it
# to obtain the store, so it is governed by byte-for-byte preservation instead
# of outright refusal (see ``approval_edit_refusal``).
APPROVAL_HARD_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (label, pattern)
    for label, pattern in APPROVAL_SENSITIVE_REGION_PATTERNS
    if label != "approval-store helper reference"
)

# Markers that always signal an approval *behavior* change or assertion rather
# than the harmless proposal-record seeding a non-approval test may perform.
# Three are reused verbatim from the region markers above; the module marker is
# narrowed to the approval *service* module (the behavior layer) only. The
# approval-store class *reference* and the ``app.approvals`` store-module import
# are deliberately excluded here, because constructing the store to seed local
# proposal fixtures is exactly what the constrained test allowance permits.
_BEHAVIOR_REGION_LABELS = frozenset(
    {
        "approval-store helper reference",
        "approval mutation call",
        "approved-execution call",
    }
)
APPROVAL_SERVICE_MODULE_PATTERN = re.compile(r"\bapp[./]approval_service\b")
APPROVAL_BEHAVIOR_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    *(
        (label, pattern)
        for label, pattern in APPROVAL_SENSITIVE_REGION_PATTERNS
        if label in _BEHAVIOR_REGION_LABELS
    ),
    ("approval service reference", APPROVAL_SERVICE_MODULE_PATTERN),
)

# Repo-relative path prefixes that mark a test file. A test file is the one
# context where importing and constructing the approval-store class to seed
# local proposal records for an unrelated feature is legitimate.
_TEST_PATH_PREFIXES = ("tests/", "conftest")

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


def approval_edit_refusal(old_text: str, new_text: str) -> str | None:
    """Return a precise refusal reason if a replace_text edit changes approval code.

    Many non-approval feature routes (calendar, tasks, history, ...) call the
    route-level ``_approval_store(request)`` helper to obtain the store, so an
    edit that preserves every such reference byte-for-byte while changing only
    the surrounding feature logic must be allowed. Everything that actually
    defines or alters approval behavior stays refused, checked in fail-closed
    order: the helper definition, any ``/approvals`` route path, the approval
    hard markers (the approval-store class, the approve/reject mutators,
    approved execution, and approval-module imports), and finally any addition,
    removal, or alteration of an approval-store helper reference. Returns
    ``None`` only when the edit touches no approval-sensitive code.
    """
    for text in (old_text, new_text):
        if APPROVAL_STORE_HELPER_DEFINITION_PATTERN.search(text):
            return "approval helper definition edit blocked"
        if APPROVAL_ROUTE_PATH_PATTERN.search(text):
            return "approval route edit blocked"
    hard_labels = sorted(
        {
            label
            for label, pattern in APPROVAL_HARD_MARKER_PATTERNS
            for text in (old_text, new_text)
            if pattern.search(text)
        }
    )
    if hard_labels:
        return f"approval-sensitive region touched ({', '.join(hard_labels)})"
    old_references = sorted(APPROVAL_STORE_HELPER_REFERENCE_PATTERN.findall(old_text))
    new_references = sorted(APPROVAL_STORE_HELPER_REFERENCE_PATTERN.findall(new_text))
    if old_references != new_references:
        return "approval-store helper reference changed"
    return None


def is_test_file_path(path: str) -> bool:
    """Report whether one repo-relative POSIX path names a test file."""
    lowered = path.casefold()
    return lowered.startswith(_TEST_PATH_PREFIXES) or "/tests/" in lowered


def approval_fixture_allowed_for_path(path: str) -> bool:
    """Whether a path may use the constrained approval test-fixture allowance.

    Only a test file that is not itself approval-focused qualifies. A Calendar
    (or other feature) test legitimately constructs the approval-store class to
    seed local proposal records, but an approval-named test is held to the full
    approval-sensitive-region rules so it can never quietly rewrite approval
    behavior under the fixture exemption.
    """
    if not is_test_file_path(path):
        return False
    name = path.rsplit("/", 1)[-1].casefold()
    normalized = re.sub(r"[^a-z0-9]", "", name)
    return "approval" not in normalized


def approval_test_fixture_refusal(text: str) -> str | None:
    """Return a refusal reason if test content goes past safe proposal seeding.

    A non-approval test file may import and construct the approval-store class
    and import the approval *store* module solely to seed local proposal records
    for another feature. Anything that instead changes or asserts approval
    *behavior* is refused, checked in fail-closed order: an ``/approvals`` route
    literal, a ``def _approval_store`` helper definition, a ``class`` definition
    of the approval store, and finally any approval behavior marker (the
    ``_approval_store`` route-helper reference, an approve/reject lifecycle
    mutator, approved execution, or an approval *service* module reference).
    Returns ``None`` only when the content's approval usage is limited to the
    fixture-safe class reference and store-module import.
    """
    if APPROVAL_ROUTE_PATH_PATTERN.search(text):
        return "approval route literal"
    if APPROVAL_STORE_HELPER_DEFINITION_PATTERN.search(text):
        return "approval helper definition"
    if defines_approval_store(text):
        return "approval-store class definition"
    behavior_labels = sorted(
        {
            label
            for label, pattern in APPROVAL_BEHAVIOR_MARKER_PATTERNS
            if pattern.search(text)
        }
    )
    if behavior_labels:
        return f"approval behavior reference ({', '.join(behavior_labels)})"
    return None


def recovery_message(action: str) -> str:
    """Build the standard stale/unsafe-state recovery instruction."""
    return f"Recover or verify the isolated worktree and job metadata before {action}."


def is_job_id(value: str) -> bool:
    """Return whether one value is a 32-character lowercase hex job id."""
    return (
        isinstance(value, str)
        and len(value) == 32
        and all(character in "0123456789abcdef" for character in value)
    )
