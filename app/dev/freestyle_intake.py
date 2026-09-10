"""Deterministic intake/intent planning for freestyle jobs.

Turns one job goal (and, for refinements, the refinement goal plus what a
previous attempt did and why it was rejected) into a fixed, auditable intake
summary: an intent classification, the explicitly named target files, the
files a human asked the model *not* to touch, and the files the job is
expected to inspect or change. Everything here is pure string work built on
top of the model-free context planner: this module performs no I/O, runs no
processes or commands, makes no model or network call, and never relaxes any
safety exclusion. It only surfaces what a human wrote so later stages can plan
context and critique generated bundles against it deterministically.
"""

from __future__ import annotations

from collections.abc import Sequence
import re

from app.dev.freestyle_context_plan import (
    extract_named_paths,
    is_test_pattern,
    plan_context,
)

# Intent categories, in canonical reporting order. A goal can carry several
# (a UI goal is also an implementation goal); the order keeps summaries
# deterministic for identical goals.
INTENT_FREESTYLE_INTERNAL = "freestyle_internal"
INTENT_UI = "ui"
INTENT_DIAGNOSTICS = "diagnostics"
INTENT_DOCS_ONLY = "docs_only"
INTENT_TESTS_ONLY = "tests_only"
INTENT_REFACTOR = "refactor"
INTENT_IMPLEMENTATION = "implementation"

_INTENT_ORDER = (
    INTENT_FREESTYLE_INTERNAL,
    INTENT_UI,
    INTENT_DIAGNOSTICS,
    INTENT_REFACTOR,
    INTENT_IMPLEMENTATION,
    INTENT_TESTS_ONLY,
    INTENT_DOCS_ONLY,
)

# Intents that, on their own, do not require any non-test/non-doc source
# change. A goal whose intents are a subset of these may legitimately produce a
# tests-only or docs-only bundle.
NON_IMPLEMENTATION_INTENTS = frozenset({INTENT_TESTS_ONLY, INTENT_DOCS_ONLY})

_DIAGNOSTICS_KEYWORDS = frozenset(
    {
        "debug",
        "diagnose",
        "diagnostic",
        "diagnostics",
        "instrument",
        "instrumentation",
        "logging",
        "observability",
        "telemetry",
        "trace",
        "tracing",
    }
)
_REFACTOR_KEYWORDS = frozenset(
    {
        "cleanup",
        "consolidate",
        "deduplicate",
        "extract",
        "refactor",
        "rename",
        "reorganize",
        "restructure",
        "simplify",
        "tidy",
    }
)

# Clause-level negation markers: a file path named inside one of these clauses
# is treated as a "do not touch" constraint for bundle critique.
_NEGATION_MARKERS = (
    "do not",
    "don't",
    "dont",
    "do n't",
    "must not",
    "never",
    "without modifying",
    "without changing",
    "without touching",
    "avoid",
    "leave",
    "keep",
    "preserve",
)
# Split on clause boundaries without breaking dotted file paths: a period only
# ends a clause when followed by whitespace or the end of the text, never when
# it sits inside "app/web_routes.py".
_CLAUSE_SPLIT_PATTERN = re.compile(
    r"[;\n]+|\.\s+|\.$|\s+(?:but|while|however|yet|though|although)\s+",
    re.IGNORECASE,
)


def classify_intent(goal: str, title: str | None = None) -> tuple[str, ...]:
    """Classify one goal/title into fixed intent categories without a model."""
    plan = plan_context(goal, title)
    categories = plan.goal_categories
    terms = _terms(goal, title)
    intents: list[str] = []
    if "freestyle_internal" in categories:
        intents.append(INTENT_FREESTYLE_INTERNAL)
    if "web_ui" in categories:
        intents.append(INTENT_UI)
    if not _DIAGNOSTICS_KEYWORDS.isdisjoint(terms):
        intents.append(INTENT_DIAGNOSTICS)
    if not _REFACTOR_KEYWORDS.isdisjoint(terms):
        intents.append(INTENT_REFACTOR)
    only_tests = categories == ("tests",)
    only_docs = categories == ("docs",)
    if only_tests:
        intents.append(INTENT_TESTS_ONLY)
    elif only_docs:
        intents.append(INTENT_DOCS_ONLY)
    elif not intents:
        intents.append(INTENT_IMPLEMENTATION)
    elif INTENT_UI in intents or INTENT_FREESTYLE_INTERNAL in intents:
        # A surface (UI/internal) goal still implies an implementation change
        # unless it is explicitly tests-only or docs-only above.
        intents.append(INTENT_IMPLEMENTATION)
    return tuple(intent for intent in _INTENT_ORDER if intent in intents)


def requires_implementation_change(intents: Sequence[str]) -> bool:
    """Report whether these intents demand a non-test/non-doc source change."""
    meaningful = [intent for intent in intents if intent]
    if not meaningful:
        return True
    return any(intent not in NON_IMPLEMENTATION_INTENTS for intent in meaningful)


def extract_forbidden_paths(text: str) -> tuple[str, ...]:
    """Extract repo-relative paths a human asked the model not to touch.

    Deterministic and conservative: a path is forbidden only when it is named
    inside a clause that also carries an explicit negation/preservation marker
    (``do not modify app/x.py``, ``without changing tests/y.py``). Path safety
    is enforced elsewhere; this only surfaces the human's stated constraint.
    """
    forbidden: list[str] = []
    seen: set[str] = set()
    for clause in _CLAUSE_SPLIT_PATTERN.split(text):
        lowered = clause.casefold()
        if not any(marker in lowered for marker in _NEGATION_MARKERS):
            continue
        for path in extract_named_paths(clause):
            key = path.casefold()
            if key not in seen:
                seen.add(key)
                forbidden.append(path)
    return tuple(forbidden)


def is_test_path(path: str) -> bool:
    """Report whether one repo-relative path names a test file."""
    return is_test_pattern(path) or "/tests/" in path.casefold()


def split_named_targets(
    paths: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split named targets into (implementation files, test files)."""
    implementation = tuple(path for path in paths if not is_test_path(path))
    tests = tuple(path for path in paths if is_test_path(path))
    return implementation, tests


def build_intake_summary(
    goal: str,
    title: str | None = None,
    *,
    refinement_goal: str | None = None,
    previous_bundle_touched_files: Sequence[str] = (),
    previous_proposal_likely_files: Sequence[str] = (),
    rejection_text: str | None = None,
) -> dict[str, object]:
    """Build the deterministic, auditable intake record stored in artifacts.

    For a refinement the named paths, forbidden paths, and intent are derived
    from the goal *and* the refinement instruction, and the previous attempt's
    touched files and proposal likely-files are carried forward as memory so
    later stages neither re-plan from scratch nor repeat a rejected pattern.
    """
    combined_named = list(extract_named_paths(goal))
    forbidden = list(extract_forbidden_paths(goal))
    if refinement_goal:
        combined_named.extend(extract_named_paths(refinement_goal))
        forbidden.extend(extract_forbidden_paths(refinement_goal))
    named = _dedupe(combined_named)
    forbidden_paths = _dedupe(forbidden)
    implementation_files, test_files = split_named_targets(named)
    intents = classify_intent(
        " ".join(part for part in (goal, refinement_goal) if part), title
    )
    must_inspect = _dedupe(
        [
            *named,
            *previous_proposal_likely_files,
            *previous_bundle_touched_files,
        ]
    )
    # A human who named implementation files to change clearly wants an
    # implementation change, even when fuzzy keyword classification (e.g. a
    # "do not touch tests/..." clause) would otherwise read as tests-only.
    needs_implementation = requires_implementation_change(intents) or bool(
        implementation_files
    )
    return {
        "intent_categories": list(intents),
        "requires_implementation_change": needs_implementation,
        "named_target_paths": list(named),
        "named_implementation_files": list(implementation_files),
        "named_test_files": list(test_files),
        "forbidden_paths": list(forbidden_paths),
        "must_inspect_files": must_inspect,
        "previous_bundle_touched_files": list(_dedupe(previous_bundle_touched_files)),
        "previous_proposal_likely_files": list(_dedupe(previous_proposal_likely_files)),
        "rejection_text_present": bool(rejection_text and rejection_text.strip()),
    }


def _dedupe(paths: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if not candidate:
            continue
        key = candidate.casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(candidate)
    return ordered


def _terms(goal: str, title: str | None) -> frozenset[str]:
    parts = [goal]
    if isinstance(title, str):
        parts.append(title)
    return frozenset(re.findall(r"[a-z0-9]+", " ".join(parts).casefold()))
