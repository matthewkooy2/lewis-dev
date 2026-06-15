"""Deterministic semantic critique of generated freestyle edit bundles.

Structural and safety validation prove a bundle is *well formed and safe to
apply*. This module answers a different, advisory question: does the bundle
*plausibly satisfy the goal it was generated for*? It compares the validated
operations (and their already-validated new file contents) against the
deterministic intake plan and proposal, and returns bounded human-readable
warnings when the bundle looks unfit: tests-only changes for an
implementation goal, explicitly named files left untouched, a proposal's
likely files ignored wholesale, a helper created but never called, or a
refinement constraint violated.

Everything here is pure string work: no I/O, no process or model call, no
relaxation of any safety check, and the warnings never block the deterministic
applier. They exist to be surfaced to a human before apply.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re

from app.dev.freestyle_intake import is_test_path

MAX_CRITIQUE_PATHS_IN_MESSAGE = 5
MAX_CRITIQUE_WARNINGS = 12

CRITIQUE_TESTS_ONLY_FOR_IMPLEMENTATION = (
    "Bundle goal-fit warning: this goal requires an implementation change, but "
    "the generated bundle only creates or edits test files and changes no "
    "implementation file."
)
CRITIQUE_NAMED_TARGETS_NOT_TOUCHED = (
    "Bundle goal-fit warning: the goal explicitly named implementation files "
    "to change, but the generated bundle does not touch any of them: {paths}."
)
CRITIQUE_LIKELY_FILES_IGNORED = (
    "Bundle goal-fit warning: the proposal listed likely files to edit, but "
    "the generated bundle touches none of them and offers no other named "
    "target instead: {paths}."
)
CRITIQUE_HELPER_CREATED_NOT_CALLED = (
    "Bundle goal-fit warning: the bundle creates {paths} defining new "
    "symbols, but none of those symbols are called by any other file the "
    "bundle touches, so the new code is likely dead."
)
CRITIQUE_FORBIDDEN_PATH_TOUCHED = (
    "Bundle goal-fit warning: the refinement asked not to modify {paths}, but "
    "the generated bundle changes it anyway."
)

_TOP_LEVEL_SYMBOL_PATTERN = re.compile(
    r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", re.MULTILINE
)


@dataclass(frozen=True)
class CritiqueOperation:
    """One validated bundle operation reduced to what critique can inspect."""

    op: str
    path: str
    new_content: str


def critique_bundle(
    *,
    requires_implementation: bool,
    named_implementation_files: Sequence[str],
    proposal_likely_files: Sequence[str],
    forbidden_paths: Sequence[str],
    operations: Sequence[CritiqueOperation],
) -> tuple[str, ...]:
    """Return advisory goal-fit warnings for one validated bundle."""
    if not operations:
        return ()
    touched = {op.path.casefold() for op in operations if op.op != "delete_file"}
    impl_touched = {path for path in touched if not _is_test_or_doc(path)}
    warnings: list[str] = []

    if requires_implementation and touched and not impl_touched:
        warnings.append(CRITIQUE_TESTS_ONLY_FOR_IMPLEMENTATION)

    untouched_named = [
        path
        for path in _dedupe(named_implementation_files)
        if path.casefold() not in touched
    ]
    if untouched_named and not _any_touched(named_implementation_files, touched):
        warnings.append(
            CRITIQUE_NAMED_TARGETS_NOT_TOUCHED.format(
                paths=_format_paths(untouched_named)
            )
        )

    likely = _dedupe(proposal_likely_files)
    if (
        likely
        and not _any_touched(likely, touched)
        and not _any_touched(named_implementation_files, touched)
    ):
        warnings.append(
            CRITIQUE_LIKELY_FILES_IGNORED.format(paths=_format_paths(likely))
        )

    dead_helpers = _uncalled_created_helpers(operations)
    if dead_helpers:
        warnings.append(
            CRITIQUE_HELPER_CREATED_NOT_CALLED.format(
                paths=_format_paths(dead_helpers)
            )
        )

    touched_forbidden = [
        path for path in _dedupe(forbidden_paths) if path.casefold() in touched
    ]
    if touched_forbidden:
        warnings.append(
            CRITIQUE_FORBIDDEN_PATH_TOUCHED.format(
                paths=_format_paths(touched_forbidden)
            )
        )

    return tuple(warnings[:MAX_CRITIQUE_WARNINGS])


def _uncalled_created_helpers(
    operations: Sequence[CritiqueOperation],
) -> list[str]:
    """Find created non-test Python files whose symbols nothing else calls."""
    other_content_by_path: dict[str, str] = {
        op.path: op.new_content for op in operations
    }
    dead: list[str] = []
    for op in operations:
        if op.op != "create_file" or not op.path.casefold().endswith(".py"):
            continue
        if is_test_path(op.path):
            continue
        symbols = _top_level_symbols(op.new_content)
        if not symbols:
            continue
        called_elsewhere = any(
            _symbol_called(symbol, content)
            for symbol in symbols
            for path, content in other_content_by_path.items()
            if path != op.path
        )
        if not called_elsewhere:
            dead.append(op.path)
    return dead


def _top_level_symbols(content: str) -> tuple[str, ...]:
    return tuple(
        name
        for name in _TOP_LEVEL_SYMBOL_PATTERN.findall(content)
        if not name.startswith("__")
    )


def _symbol_called(symbol: str, content: str) -> bool:
    return re.search(r"\b" + re.escape(symbol) + r"\s*\(", content) is not None


def _is_test_or_doc(path: str) -> bool:
    return is_test_path(path) or path.casefold().endswith((".md", ".rst", ".txt"))


def _any_touched(paths: Sequence[str], touched: set[str]) -> bool:
    return any(path.casefold() in touched for path in paths)


def _dedupe(paths: Sequence[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for raw in paths:
        if not isinstance(raw, str):
            continue
        candidate = raw.strip()
        if candidate and candidate.casefold() not in seen:
            seen.add(candidate.casefold())
            ordered.append(candidate)
    return ordered


def _format_paths(paths: Sequence[str]) -> str:
    shown = list(paths)[:MAX_CRITIQUE_PATHS_IN_MESSAGE]
    suffix = "" if len(paths) <= MAX_CRITIQUE_PATHS_IN_MESSAGE else ", …"
    return ", ".join(shown) + suffix
