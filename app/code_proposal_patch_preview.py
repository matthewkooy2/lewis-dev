from __future__ import annotations

import re
from dataclasses import dataclass

from app.project_search import ProjectFileExcerpt, ProjectFileSearch


PATCH_PREVIEW_UNAVAILABLE_NOTE = (
    "A grounded patch preview could not be generated confidently; review-only "
    "planning is available."
)
PATCH_PREVIEW_REVIEW_ONLY_NOTE = (
    "Patch preview is inert review text only; Lewis does not apply it."
)
_SIMPLE_LABEL_TERMS = ("label", "badge", "caption")
_UI_TERMS = ("ui", "chat", "command center", "interface")
_PYTHON_STRING_RETURN_PATTERN = re.compile(
    r"^(?P<indent>\s*)return\s+(?P<quote>['\"])(?P<value>[^'\"]*"
    r"(?:chat|ui|message)[^'\"]*)(?P=quote)\s*$",
    re.IGNORECASE,
)
_CHAT_SHELL_LINE = "\"<section class='chat-shell'>\","
_CHAT_LABEL_LINE = "\"<p class='muted'>Command Center Chat</p>\","
_EXCERPT_FOCUS_TERMS = ("chat-shell", "chat ui")


@dataclass(frozen=True)
class GroundedPatchPreview:
    patch_preview: str
    path: str


def generate_grounded_patch_preview(
    *,
    requested_goal: str,
    relevant_files: list[str],
    search: ProjectFileSearch,
) -> GroundedPatchPreview | None:
    """Generate a conservative review-only patch preview from verified files."""
    if not _is_simple_ui_label_request(requested_goal):
        return None

    verified_files = [path for path in relevant_files if search.is_safe_file_path(path)]
    for path in verified_files:
        excerpt = search.read_excerpt(path, focus_terms=_EXCERPT_FOCUS_TERMS)
        preview = _preview_for_excerpt(excerpt)
        if preview is not None:
            return GroundedPatchPreview(patch_preview=preview, path=path)
    return None


def _preview_for_excerpt(excerpt: ProjectFileExcerpt | None) -> str | None:
    if excerpt is None:
        return None
    if excerpt.path.endswith(".py"):
        return _python_label_preview(excerpt)
    return None


def _python_label_preview(excerpt: ProjectFileExcerpt) -> str | None:
    lines = excerpt.text.splitlines()
    for index, line in enumerate(lines, start=excerpt.start_line):
        match = _PYTHON_STRING_RETURN_PATTERN.match(line)
        if match is None:
            if line.strip() == _CHAT_SHELL_LINE:
                indent = line[: len(line) - len(line.lstrip())]
                return _unified_single_line_addition(
                    excerpt.path,
                    index,
                    line,
                    f"{indent}{_CHAT_LABEL_LINE}",
                )
            continue
        old_line = line
        value = match.group("value")
        new_value = _with_chat_label(value)
        if new_value == value:
            continue
        new_line = (
            f"{match.group('indent')}return "
            f"{match.group('quote')}{new_value}{match.group('quote')}"
        )
        return _unified_single_line_replacement(excerpt.path, index, old_line, new_line)
    return None


def _with_chat_label(value: str) -> str:
    stripped = value.strip()
    if stripped.lower().startswith("chat:"):
        return value
    return value.replace(stripped, f"Chat: {stripped}", 1)


def _unified_single_line_replacement(
    path: str,
    line_number: int,
    old_line: str,
    new_line: str,
) -> str:
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            f"--- a/{path}",
            f"+++ b/{path}",
            f"@@ -{line_number},1 +{line_number},1 @@",
            f"-{old_line}",
            f"+{new_line}",
        ]
    )


def _unified_single_line_addition(
    path: str,
    line_number: int,
    context_line: str,
    new_line: str,
) -> str:
    return "\n".join(
        [
            f"diff --git a/{path} b/{path}",
            f"--- a/{path}",
            f"+++ b/{path}",
            f"@@ -{line_number},1 +{line_number},2 @@",
            f" {context_line}",
            f"+{new_line}",
        ]
    )


def _is_simple_ui_label_request(requested_goal: str) -> bool:
    normalized = requested_goal.lower()
    return any(term in normalized for term in _SIMPLE_LABEL_TERMS) and any(
        term in normalized for term in _UI_TERMS
    )
