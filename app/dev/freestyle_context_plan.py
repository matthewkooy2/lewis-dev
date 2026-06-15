"""Deterministic goal-aware context planning for freestyle model context.

Classifies one freestyle job goal/title into fixed implementation-surface
categories and exposes the matching high-priority file patterns, a readable
context-plan summary for artifacts, and advisory goal-fit warnings for
generated edit bundles. Everything here is pure string work: this module
performs no I/O, runs no processes or commands, makes no model or network
call, imports nothing from the rest of the application, and never relaxes
any safety exclusion.
Safety filtering happens before these priorities are consulted, so a pattern
here can only reorder files that were already safe to include.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fnmatch import fnmatchcase
import re
from typing import Any

MAX_PLAN_OMISSION_RECORDS = 50
MAX_NAMED_TARGET_PATHS = 20
MAX_NAMED_TARGET_PATH_CHARS = 200
# Files dropped only because a bounded context budget filled up, as opposed
# to safety or readability exclusions.
BUDGET_OMISSION_REASONS = frozenset(
    {"file_count_limit", "total_context_limit", "total_prompt_limit"}
)
DEFAULT_GOAL_CATEGORY = "core_app"

# Extensions recognised when extracting explicitly named repo-relative file
# paths from a goal. The set is deliberately narrow so arbitrary tokens that
# merely contain a slash are not treated as edit targets.
_NAMED_PATH_EXTENSIONS = frozenset(
    {
        "cfg",
        "css",
        "csv",
        "html",
        "ini",
        "js",
        "json",
        "jsx",
        "md",
        "py",
        "rst",
        "sql",
        "toml",
        "ts",
        "tsx",
        "txt",
        "yaml",
        "yml",
    }
)
# One repo-relative path: at least one "/" separator and a final extension,
# bounded to path-safe characters so surrounding quotes/backticks/prose are
# excluded by construction.
_NAMED_PATH_PATTERN = re.compile(
    r"(?<![\w./-])((?:[A-Za-z0-9_.+-]+/)+[A-Za-z0-9_.+-]+\.[A-Za-z0-9]+)"
)

NAMED_TARGET_BUDGET_BLOCKER = (
    "An explicitly named target file from the goal was safe and present in the "
    "isolated worktree but could not fit within the bounded model context; "
    "refusing to call DeepSeek without the file the goal explicitly requires "
    "inspecting or changing."
)

# Fixed canonical ordering keeps detected categories and the concatenated
# priority patterns deterministic for identical goals.
_CATEGORY_ORDER = (
    "web_ui",
    "tests",
    "docs",
    "cli_dev_tooling",
    "config",
    "freestyle_internal",
    DEFAULT_GOAL_CATEGORY,
)

_CATEGORY_KEYWORDS: dict[str, frozenset[str]] = {
    "web_ui": frozenset(
        {
            "browser",
            "button",
            "buttons",
            "css",
            "dashboard",
            "form",
            "forms",
            "frontend",
            "html",
            "javascript",
            "js",
            "layout",
            "menu",
            "nav",
            "navigation",
            "overview",
            "page",
            "pages",
            "route",
            "routes",
            "sidebar",
            "style",
            "styles",
            "stylesheet",
            "styling",
            "tab",
            "tabs",
            "template",
            "templates",
            "ui",
            "web",
            "webpage",
        }
    ),
    "tests": frozenset({"pytest", "regression", "test", "tests", "unittest"}),
    "docs": frozenset({"changelog", "doc", "docs", "documentation", "readme"}),
    "cli_dev_tooling": frozenset({"cli", "tooling"}),
    "config": frozenset({"config", "configuration", "settings"}),
    "freestyle_internal": frozenset(
        {"bundle", "deepseek", "freestyle", "handoff", "materialize", "worktree"}
    ),
}

# Per-category high-priority include patterns, most relevant first. Matching
# uses casefolded fnmatch on full relative POSIX paths, so "*" may cross "/".
# Patterns may name files that do not exist; only files that already passed
# every safety exclusion can be ranked by them.
_PRIORITY_PATTERNS: dict[str, tuple[str, ...]] = {
    # Core implementation surface first, then static assets and templates,
    # then tests: when the bounded context budget is tight, tests must be the
    # files that get dropped, never the implementation surface.
    "web_ui": (
        "app/web_pages.py",
        "app/web_components.py",
        "app/web_routes.py",
        "app/web.py",
        "app/web_rules.py",
        "app/dashboard.py",
        "app/static/lewis.css",
        "app/static/chat.js",
        "app/dev/freestyle_web.py",
        "app/static/*.css",
        "app/static/*.js",
        "app/web*.py",
        "templates/*.html",
        "app/templates/*.html",
        "app/static/*",
        "tests/test_web.py",
        "tests/test_web_rules.py",
        "tests/test_web_pages.py",
        "tests/test_freestyle_web.py",
        "tests/test_web*.py",
    ),
    "tests": ("tests/*.py", "conftest.py"),
    "docs": ("README.md", "docs/*", "*.md"),
    "cli_dev_tooling": ("app/main.py", "scripts/*"),
    "config": ("pyproject.toml", "setup.cfg", "app/config.py", "*.toml", "*.cfg"),
    "freestyle_internal": ("app/dev/freestyle_*.py", "tests/test_freestyle_*.py"),
    DEFAULT_GOAL_CATEGORY: (),
}

# Patterns naming test files: ranked after every implementation pattern for
# the same category, and reported separately in context-plan summaries.
_TEST_PATTERN_PREFIXES = ("tests/", "conftest")

# Recognized web entry points: route/navigation/page/component files that an
# edit bundle for a web/UI goal is expected to touch when it creates new
# page, static, or template files.
_WEB_ENTRY_POINT_PATTERNS = (
    "app/web_pages.py",
    "app/web_components.py",
    "app/web_routes.py",
    "app/web.py",
    "app/web_rules.py",
    "app/dashboard.py",
    "app/dev/freestyle_web.py",
    "app/web*.py",
)
# Page/static/template assets whose creation should be wired into one of the
# entry points above to be reachable from the existing UI.
_WEB_CREATED_ASSET_PATTERNS = (
    "app/static/*",
    "templates/*",
    "app/templates/*",
    "*.html",
    "*.css",
    "*.js",
)

PLAN_WARNING_NO_UI_FILES = (
    "This goal was classified as web/UI work, but no recognized web or UI "
    "implementation files were available as safe context candidates in the "
    "isolated worktree."
)
UI_CONTEXT_GAP_MESSAGE = (
    "This goal was classified as web/UI work and recognized web/UI files exist "
    "as safe context candidates, but none were included in the bounded model "
    "context; refusing to call DeepSeek with context that cannot ground the goal."
)
GOAL_FIT_WARNING_CREATE_ONLY = (
    "UI goal-fit warning: the generated bundle only creates new files and does "
    "not modify any existing route, navigation, page, component, static, or "
    "test file, so the new files are likely unreachable from the existing UI."
)
GOAL_FIT_WARNING_NO_WEB_SURFACE = (
    "UI goal-fit warning: the generated bundle does not touch any recognized "
    "web or UI surface path for this goal."
)
GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT = (
    "UI goal-fit warning: the generated bundle creates page, static, or "
    "template files but does not touch any recognized web entry point "
    "(route, navigation, page, or component file), so the new files are "
    "likely not wired into the existing UI."
)


@dataclass(frozen=True)
class ContextPlan:
    """One deterministic, model-free context plan for a single job goal."""

    goal_categories: tuple[str, ...]
    priority_patterns: tuple[str, ...]
    demote_freestyle_internals: bool
    named_target_paths: tuple[str, ...] = ()


def plan_context(
    goal: str,
    title: str | None = None,
    *,
    extra_named_paths: Sequence[str] = (),
) -> ContextPlan:
    """Classify one goal/title into fixed categories without calling a model.

    ``extra_named_paths`` lets a refinement contribute file paths named in the
    refinement instruction (or carried from a prior bundle/proposal) so they
    are protected alongside paths named in the goal text itself.
    """
    terms = _classification_terms(goal, title)
    detected = tuple(
        category
        for category in _CATEGORY_ORDER
        if category != DEFAULT_GOAL_CATEGORY
        and not _CATEGORY_KEYWORDS[category].isdisjoint(terms)
    )
    if not detected:
        detected = (DEFAULT_GOAL_CATEGORY,)
    patterns: list[str] = []
    for category in detected:
        for pattern in _PRIORITY_PATTERNS[category]:
            if pattern not in patterns:
                patterns.append(pattern)
    named = extract_named_paths(goal)
    if isinstance(title, str):
        named = _merge_named_paths(named, extract_named_paths(title))
    named = _merge_named_paths(named, _clean_named_paths(extra_named_paths))
    return ContextPlan(
        goal_categories=detected,
        priority_patterns=tuple(patterns),
        demote_freestyle_internals="freestyle_internal" not in detected,
        named_target_paths=named,
    )


def extract_named_paths(text: str) -> tuple[str, ...]:
    """Extract explicitly named, repo-relative-looking file paths from text.

    Deterministic and conservative: only forward/backslash separated tokens
    that end in a recognised code or text extension are returned, traversal
    and absolute paths are dropped, and the result is bounded. Path *safety*
    (worktree containment, excluded categories) is enforced separately by the
    caller; this only surfaces the candidate names a human wrote.
    """
    normalized = text.replace("\\", "/")
    found: list[str] = []
    for match in _NAMED_PATH_PATTERN.finditer(normalized):
        candidate = match.group(1)
        if _is_named_path_candidate(candidate):
            found.append(candidate)
    return _clean_named_paths(found)


def _is_named_path_candidate(candidate: str) -> bool:
    if len(candidate) > MAX_NAMED_TARGET_PATH_CHARS:
        return False
    parts = candidate.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        return False
    suffix = candidate.rsplit(".", 1)[-1].casefold()
    return suffix in _NAMED_PATH_EXTENSIONS


def _clean_named_paths(paths: Sequence[str]) -> tuple[str, ...]:
    """Casefold-dedupe while preserving first-seen order and the bound."""
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip().replace("\\", "/")
        if not candidate or len(candidate) > MAX_NAMED_TARGET_PATH_CHARS:
            continue
        key = candidate.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(candidate)
        if len(ordered) >= MAX_NAMED_TARGET_PATHS:
            break
    return tuple(ordered)


def _merge_named_paths(
    first: Sequence[str], second: Sequence[str]
) -> tuple[str, ...]:
    return _clean_named_paths((*first, *second))


def is_named_target(relative_path: str, plan: ContextPlan) -> bool:
    """Report whether one path was explicitly named in the goal/refinement."""
    lowered = relative_path.casefold()
    return any(named.casefold() == lowered for named in plan.named_target_paths)


def named_target_availability(
    plan: ContextPlan,
    included_paths: Sequence[str],
    omitted_records: Sequence[dict[str, str]],
    candidate_paths: Sequence[str],
) -> dict[str, Any]:
    """Classify every explicitly named target by how it reached model context.

    Buckets each named path as included, omitted only because a context budget
    filled up (a fail-closed blocker), excluded by a safety/readability policy,
    or simply absent from the worktree. This is deterministic and auditable and
    never inspects file contents, so it can only report on already-safe paths.
    """
    included_set = {path.casefold() for path in included_paths}
    candidate_set = {path.casefold() for path in candidate_paths}
    first_omission: dict[str, str] = {}
    for record in omitted_records:
        key = str(record.get("path", "")).casefold()
        if key and key not in first_omission:
            first_omission[key] = str(record.get("reason", ""))
    included: list[str] = []
    budget: list[dict[str, str]] = []
    excluded: list[str] = []
    not_found: list[str] = []
    for raw in plan.named_target_paths:
        key = raw.casefold()
        reason = first_omission.get(key)
        if key in included_set:
            included.append(raw)
        elif reason in BUDGET_OMISSION_REASONS:
            budget.append({"path": raw, "reason": reason})
        elif key in candidate_set or key in first_omission:
            excluded.append(raw)
        else:
            not_found.append(raw)
    return {
        "named_target_paths": list(plan.named_target_paths),
        "included": included,
        "omitted_due_to_budget": budget[:MAX_PLAN_OMISSION_RECORDS],
        "excluded_by_policy": excluded[:MAX_PLAN_OMISSION_RECORDS],
        "not_found": not_found[:MAX_PLAN_OMISSION_RECORDS],
        "has_budget_blocker": bool(budget),
    }


def priority_rank(relative_path: str, plan: ContextPlan) -> int | None:
    """Return the index of the first matching priority pattern, or None."""
    lowered = relative_path.casefold()
    for rank, pattern in enumerate(plan.priority_patterns):
        if fnmatchcase(lowered, pattern.casefold()):
            return rank
    return None


def summarize_context_plan(
    plan: ContextPlan,
    included_paths: Sequence[str],
    omitted_records: Sequence[dict[str, str]],
    *,
    priority_candidate_count: int,
    candidate_paths: Sequence[str] = (),
) -> dict[str, Any]:
    """Build the readable context-plan section recorded in artifacts."""
    included_priority = [
        path for path in included_paths if priority_rank(path, plan) is not None
    ]
    priority_records = [
        {"path": str(record.get("path", "")), "reason": str(record.get("reason", ""))}
        for record in omitted_records
        if priority_rank(str(record.get("path", "")), plan) is not None
    ]
    # A "content_truncated" record means the file was included with its tail
    # cut, not dropped; report it separately so omissions stay honest.
    omitted_priority = [
        record
        for record in priority_records
        if record["reason"] != "content_truncated"
    ][:MAX_PLAN_OMISSION_RECORDS]
    truncated_priority = [
        record["path"]
        for record in priority_records
        if record["reason"] == "content_truncated"
    ][:MAX_PLAN_OMISSION_RECORDS]
    warnings: list[str] = []
    if "web_ui" in plan.goal_categories and priority_candidate_count == 0:
        warnings.append(PLAN_WARNING_NO_UI_FILES)
    named_targets = named_target_availability(
        plan, included_paths, omitted_records, candidate_paths
    )
    if named_targets["has_budget_blocker"]:
        warnings.append(NAMED_TARGET_BUDGET_BLOCKER)
    return {
        "detected_goal_categories": list(plan.goal_categories),
        "high_priority_patterns": list(plan.priority_patterns),
        "core_priority_patterns": [
            pattern
            for pattern in plan.priority_patterns
            if not is_test_pattern(pattern)
        ],
        "test_priority_patterns": [
            pattern for pattern in plan.priority_patterns if is_test_pattern(pattern)
        ],
        "ordering_policy": (
            "Priority-ranked files are selected strictly by pattern rank before "
            "any generic score, so core implementation files always precede "
            "test files; when the context budget is tight, tests are dropped "
            "before the implementation surface."
        ),
        "priority_candidate_count": priority_candidate_count,
        "included_priority_files": included_priority,
        "omitted_priority_files": omitted_priority,
        "truncated_priority_files": truncated_priority,
        "priority_files_omitted_due_to_budget": any(
            record["reason"] in BUDGET_OMISSION_REASONS for record in omitted_priority
        ),
        "named_target_files": named_targets,
        "warnings": warnings,
    }


def ui_context_gap(plan_summary: dict[str, Any] | None) -> str | None:
    """Detect a UI-classified goal whose context includes no UI candidate."""
    if not isinstance(plan_summary, dict):
        return None
    categories = plan_summary.get("detected_goal_categories", ())
    if "web_ui" not in categories:
        return None
    candidate_count = plan_summary.get("priority_candidate_count", 0)
    included = plan_summary.get("included_priority_files", ())
    if isinstance(candidate_count, int) and candidate_count > 0 and not included:
        return UI_CONTEXT_GAP_MESSAGE
    return None


def goal_fit_warnings(
    plan: ContextPlan, operations: Sequence[tuple[str, str]]
) -> tuple[str, ...]:
    """Return advisory warnings for bundles that look unfit for a UI goal.

    Purely deterministic and separate from structural/safety validation; a
    warning never proves the bundle wrong and never blocks the deterministic
    applier by itself.
    """
    if "web_ui" not in plan.goal_categories or not operations:
        return ()
    warnings: list[str] = []
    if all(op == "create_file" for op, _ in operations):
        warnings.append(GOAL_FIT_WARNING_CREATE_ONLY)
    if all(priority_rank(path, plan) is None for _, path in operations):
        warnings.append(GOAL_FIT_WARNING_NO_WEB_SURFACE)
    creates_web_asset = any(
        op == "create_file" and _matches_any(path, _WEB_CREATED_ASSET_PATTERNS)
        for op, path in operations
    )
    touches_entry_point = any(
        _matches_any(path, _WEB_ENTRY_POINT_PATTERNS) for _, path in operations
    )
    if creates_web_asset and not touches_entry_point:
        warnings.append(GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT)
    return tuple(warnings)


def _matches_any(relative_path: str, patterns: Sequence[str]) -> bool:
    lowered = relative_path.casefold()
    return any(fnmatchcase(lowered, pattern.casefold()) for pattern in patterns)


def is_test_pattern(pattern: str) -> bool:
    """Report whether one priority pattern names test files rather than code."""
    return pattern.casefold().startswith(_TEST_PATTERN_PREFIXES)


def _classification_terms(goal: str, title: str | None) -> frozenset[str]:
    text_parts = [goal]
    if isinstance(title, str):
        text_parts.append(title)
    return frozenset(
        re.findall(r"[a-z0-9]+", " ".join(text_parts).casefold())
    )
