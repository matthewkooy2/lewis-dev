"""Deterministic exact-anchor snippet harvesting for freestyle edit generation.

Extracts small verbatim snippets from the current text of one
already-safety-screened isolated-worktree file so the edit-generation prompt
can tell the model to copy ``replace_text.old_text`` exactly from real file
content instead of inventing approximate anchors. Everything here is pure
string work: this module performs no file, process, model, or network I/O and
never relaxes any safety policy. Callers must apply the existing path and
content safety policy before passing text in, and every harvested snippet is
itself re-screened so approval-sensitive regions and sensitive content are
never quoted back to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.dev.freestyle_policy import (
    approval_sensitive_region_labels,
    contains_sensitive_content,
)

MAX_SNIPPETS_PER_FILE = 4
MAX_SNIPPET_CHARS = 700
MAX_TOTAL_SNIPPET_CHARS = 6_000
MAX_OMITTED_SNIPPET_RECORDS_PER_FILE = 8
_MAX_IMPORT_BLOCKS = 2
_MAX_TEST_BLOCKS = 3

# Route paths the edit-generation guidance names as canonical wiring examples;
# harvested right after any route mentioned in the goal itself.
_GUIDANCE_ROUTES = frozenset({"/dashboard", "/usage", "/chat"})

_IMPORT_BLOCK_OPEN_PATTERN = re.compile(r"^from\s+([\w.]+)\s+import\s+\(\s*$")
_ROUTE_DECORATOR_PATTERN = re.compile(
    r"^(?P<indent>\s*)@\w+\.(?:get|post|put|patch|delete)\(\s*"
    r"(?P<quote>[\"'])(?P<route>[^\"']*)(?P=quote)"
)
_TEST_FUNCTION_PATTERN = re.compile(r"^def\s+(test_\w+)\s*\(")
_GOAL_ROUTE_PATTERN = re.compile(r"/[A-Za-z0-9_-]+")
_GOAL_TERM_PATTERN = re.compile(r"[a-z0-9_]{4,}")


@dataclass(frozen=True)
class AnchorSnippet:
    """One bounded snippet copied verbatim from current file text."""

    label: str
    exact_text: str


@dataclass(frozen=True)
class SnippetHarvest:
    """Selected snippets plus bounded records of what was filtered out."""

    snippets: tuple[AnchorSnippet, ...]
    omitted: tuple[dict[str, str], ...]


def harvest_anchor_snippets(relative_path: str, text: str, goal: str) -> SnippetHarvest:
    """Harvest bounded verbatim anchor snippets from one file's full text.

    Python route/implementation files yield multi-line import blocks and
    route-decorator blocks (goal-mentioned routes first); test files yield
    nearby test functions; anything else falls back to one file-tail snippet.
    Snippets containing approval-sensitive region markers or sensitive content
    are always dropped, never weakened.
    """
    lines = text.split("\n")
    lowered = relative_path.casefold()
    name = lowered.rpartition("/")[2]
    candidates: list[tuple[str, str]] = []
    if lowered.endswith(".py"):
        if lowered.startswith("tests/") or name.startswith("test_"):
            candidates.extend(_test_function_blocks(lines, goal))
        else:
            candidates.extend(_import_blocks(lines))
            candidates.extend(_route_blocks(lines, goal))
    selected: list[AnchorSnippet] = []
    omitted: list[dict[str, str]] = []
    _select_candidates(candidates, text, selected, omitted)
    if not selected:
        tail = _tail_block(lines)
        if tail is not None:
            _select_candidates([tail], text, selected, omitted)
    return SnippetHarvest(
        snippets=tuple(selected),
        omitted=tuple(omitted[:MAX_OMITTED_SNIPPET_RECORDS_PER_FILE]),
    )


def _select_candidates(
    candidates: list[tuple[str, str]],
    text: str,
    selected: list[AnchorSnippet],
    omitted: list[dict[str, str]],
) -> None:
    for label, candidate_text in candidates:
        if len(selected) >= MAX_SNIPPETS_PER_FILE:
            omitted.append({"label": label, "reason": "file_snippet_limit"})
            continue
        bounded = _bounded_line_prefix(candidate_text)
        if not bounded.strip():
            omitted.append({"label": label, "reason": "empty"})
            continue
        if approval_sensitive_region_labels(bounded):
            omitted.append({"label": label, "reason": "approval_sensitive_region"})
            continue
        if contains_sensitive_content(bounded):
            omitted.append({"label": label, "reason": "sensitive_content"})
            continue
        if bounded not in text:
            omitted.append({"label": label, "reason": "not_verbatim"})
            continue
        if any(
            bounded in existing.exact_text or existing.exact_text in bounded
            for existing in selected
        ):
            omitted.append({"label": label, "reason": "duplicate"})
            continue
        selected.append(AnchorSnippet(label=label, exact_text=bounded))


def _bounded_line_prefix(candidate_text: str) -> str:
    """Bound one snippet to MAX_SNIPPET_CHARS at a line boundary.

    A prefix of contiguous lines is still a verbatim substring of the file,
    so the exact-match property of the anchor is preserved.
    """
    if len(candidate_text) <= MAX_SNIPPET_CHARS:
        return candidate_text
    kept: list[str] = []
    length = 0
    for line in candidate_text.split("\n"):
        added = len(line) + (1 if kept else 0)
        if length + added > MAX_SNIPPET_CHARS:
            break
        kept.append(line)
        length += added
    return "\n".join(kept)


def _import_blocks(lines: list[str]) -> list[tuple[str, str]]:
    """Harvest multi-line ``from module import (...)`` blocks, web pages first."""
    blocks: list[tuple[str, str]] = []
    for index, line in enumerate(lines):
        match = _IMPORT_BLOCK_OPEN_PATTERN.match(line)
        if match is None:
            continue
        for end in range(index + 1, len(lines)):
            if lines[end].lstrip().startswith(")"):
                blocks.append((match.group(1), "\n".join(lines[index : end + 1])))
                break
    blocks.sort(key=lambda item: 0 if item[0].endswith("web_pages") else 1)
    return [
        (f"import-block:{module}", text) for module, text in blocks[:_MAX_IMPORT_BLOCKS]
    ]


def _route_blocks(lines: list[str], goal: str) -> list[tuple[str, str]]:
    """Harvest decorated route blocks: goal routes, guidance routes, the rest."""
    goal_routes = frozenset(
        route.casefold() for route in _GOAL_ROUTE_PATTERN.findall(goal)
    )
    found: list[tuple[int, int, str, str]] = []
    for index, line in enumerate(lines):
        match = _ROUTE_DECORATOR_PATTERN.match(line)
        if match is None:
            continue
        indent = len(match.group("indent"))
        end = _decorated_block_end(lines, index, indent)
        route = match.group("route")
        lowered = route.casefold()
        if lowered in goal_routes:
            rank = 0
        elif lowered in _GUIDANCE_ROUTES:
            rank = 1
        else:
            rank = 2
        found.append((rank, index, route, "\n".join(lines[index : end + 1])))
    found.sort(key=lambda item: (item[0], item[1]))
    return [(f"route:{route}", text) for _, _, route, text in found]


def _decorated_block_end(lines: list[str], start: int, indent: int) -> int:
    """Find the last line of one decorated function block.

    The block continues through stacked decorators, the (possibly multi-line)
    signature, and the indented body; it ends at the next sibling decorator or
    definition at the same indent, or at any dedent below the decorator.
    """
    seen_def = False
    end = start
    for index in range(start + 1, len(lines)):
        line = lines[index]
        stripped = line.strip()
        if not stripped:
            continue
        line_indent = len(line) - len(line.lstrip())
        if line_indent < indent:
            break
        if line_indent == indent:
            if stripped.startswith("@") or stripped.startswith(("def ", "async def ")):
                if seen_def:
                    break
                seen_def = stripped.startswith(("def ", "async def "))
                end = index
                continue
            if seen_def and not stripped.startswith((")", "]", "}")):
                break
        end = index
    return end


def _test_function_blocks(lines: list[str], goal: str) -> list[tuple[str, str]]:
    """Harvest top-level test functions, goal-term-matching names first.

    Names are matched without their "test_" prefix and the generic
    test/tests terms are ignored, so every test does not trivially match a
    goal that merely mentions testing.
    """
    goal_terms = frozenset(_GOAL_TERM_PATTERN.findall(goal.casefold())) - {
        "test",
        "tests",
    }
    found: list[tuple[int, int, str, str]] = []
    for index, line in enumerate(lines):
        match = _TEST_FUNCTION_PATTERN.match(line)
        if match is None:
            continue
        start = index
        while start > 0 and lines[start - 1].lstrip().startswith("@"):
            start -= 1
        end = index
        for body_index in range(index + 1, len(lines)):
            stripped = lines[body_index].strip()
            if not stripped:
                continue
            if lines[body_index][0] not in " \t":
                break
            end = body_index
        name = match.group(1)
        stripped_name = name.removeprefix("test_")
        rank = 0 if any(term in stripped_name for term in goal_terms) else 1
        found.append((rank, index, name, "\n".join(lines[start : end + 1])))
    found.sort(key=lambda item: (item[0], item[1]))
    return [(f"test:{name}", text) for _, _, name, text in found[:_MAX_TEST_BLOCKS]]


def _tail_block(lines: list[str]) -> tuple[str, str] | None:
    """Build one bounded snippet of the file's trailing lines as a fallback."""
    kept: list[str] = []
    length = 0
    for line in reversed(lines):
        added = len(line) + (1 if kept else 0)
        if length + added > MAX_SNIPPET_CHARS:
            break
        kept.append(line)
        length += added
    while kept and not kept[-1].strip():
        kept.pop()
    while kept and not kept[0].strip():
        kept.pop(0)
    if not kept:
        return None
    return ("file-tail", "\n".join(reversed(kept)))
