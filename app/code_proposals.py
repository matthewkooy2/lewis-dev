from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from app.approvals import ApprovalItem, ApprovalStore
from app.config import ROOT_DIR
from app.code_proposal_path_safety import validate_relative_project_path
from app.code_proposal_paths import align_relevant_files_with_project_structure
from app.code_proposal_previews import (
    classify_patch_preview_paths,
    patch_preview_is_implementation_ready,
    patch_preview_warning,
    validate_patch_preview,
)
from app.code_proposal_test_commands import normalize_suggested_test_commands


CODE_PROPOSAL_ACTION_TYPE = "propose_code_change"
MAX_TEXT_LENGTH = 4000
MAX_LIST_ITEMS = 20
PATCH_MARKER_PATTERN = re.compile(r"(?m)^(?:diff --git|@@\s|---\s|\+\+\+\s)|```diff\b")
SECRET_VALUE_PATTERNS = (
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(?:sk|xox[pbar]|gh[pousr])[_-][A-Za-z0-9_=-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"\b(api[_ -]?key|password|passwd|pwd|secret|token)\b"
        r"(\s*(?:is|=|:)\s*)\S{8,}",
        re.IGNORECASE,
    ),
)


@dataclass(frozen=True)
class CodeProposal:
    requested_goal: str
    target_surface: str
    relevant_files: list[str]
    proposed_approach: str
    patch_preview: str
    patch_preview_path_classifications: list[dict[str, str]]
    patch_preview_implementation_ready: bool
    patch_preview_warning: str
    safety_notes: list[str]
    suggested_test_commands: list[str]


def propose_code_change(
    store: ApprovalStore,
    requested_goal: str,
    relevant_files: Iterable[object] | None,
    proposed_approach: str,
    patch_preview: object | None = None,
    target_surface: object | None = None,
    safety_notes: Iterable[object] | None = None,
    suggested_test_commands: Iterable[object] | None = None,
    extra_payload: dict[str, Any] | None = None,
    project_root: Path = ROOT_DIR,
    align_relevant_files: bool = True,
) -> ApprovalItem:
    proposal = validate_code_proposal(
        requested_goal=requested_goal,
        relevant_files=relevant_files,
        proposed_approach=proposed_approach,
        patch_preview=patch_preview,
        target_surface=target_surface,
        safety_notes=safety_notes,
        suggested_test_commands=suggested_test_commands,
        project_root=project_root,
        align_relevant_files=align_relevant_files,
    )
    payload = {
        "requested_goal": proposal.requested_goal,
        "target_surface": proposal.target_surface,
        "relevant_files": proposal.relevant_files,
        "proposed_approach": proposal.proposed_approach,
        "safety_notes": proposal.safety_notes,
        "suggested_test_commands": proposal.suggested_test_commands,
    }
    if proposal.patch_preview:
        payload["patch_preview"] = proposal.patch_preview
        payload["patch_preview_path_classifications"] = (
            proposal.patch_preview_path_classifications
        )
        payload["patch_preview_implementation_ready"] = (
            proposal.patch_preview_implementation_ready
        )
        if proposal.patch_preview_warning:
            payload["patch_preview_warning"] = proposal.patch_preview_warning
    if extra_payload:
        payload.update(extra_payload)
    return store.create(
        action_type=CODE_PROPOSAL_ACTION_TYPE,
        title=_proposal_title(proposal.requested_goal),
        payload=payload,
    )


def validate_code_proposal(
    requested_goal: str,
    relevant_files: Iterable[object] | None,
    proposed_approach: str,
    patch_preview: object | None = None,
    target_surface: object | None = None,
    safety_notes: Iterable[object] | None = None,
    suggested_test_commands: Iterable[object] | None = None,
    project_root: Path = ROOT_DIR,
    align_relevant_files: bool = True,
) -> CodeProposal:
    goal = _required_text(requested_goal, "Code proposal requested_goal")
    _reject_patch_content(goal, "requested_goal")
    approach = _required_text(proposed_approach, "Code proposal proposed_approach")
    _reject_patch_content(approach, "proposed_approach")
    preview = validate_patch_preview(
        _optional_text(patch_preview, "Code proposal patch_preview"),
        project_root,
    )
    preview_path_classification_items = classify_patch_preview_paths(
        preview, project_root
    )
    preview_path_classifications = [
        item.as_payload() for item in preview_path_classification_items
    ]
    preview_implementation_ready = patch_preview_is_implementation_ready(
        preview_path_classification_items
    )
    preview_warning = patch_preview_warning(preview_path_classification_items)
    notes = _normalized_text_list(safety_notes, "safety_notes")
    for note in notes:
        _reject_patch_content(note, "safety_notes")
    tests = _normalized_text_list(suggested_test_commands, "suggested_test_commands")
    for command in tests:
        _reject_patch_content(command, "suggested_test_commands")
    tests = normalize_suggested_test_commands(tests, project_root)
    files = _validated_relevant_files(relevant_files, project_root)
    surface = _target_surface(goal, approach, files, target_surface)
    if align_relevant_files:
        files = align_relevant_files_with_project_structure(
            files,
            goal=goal,
            approach=approach,
            target_surface=surface,
            project_root=project_root,
        )
    notes = _with_contextual_safety_notes(notes, goal, approach, surface)
    return CodeProposal(
        requested_goal=goal,
        target_surface=surface,
        relevant_files=files,
        proposed_approach=approach,
        patch_preview=preview,
        patch_preview_path_classifications=preview_path_classifications,
        patch_preview_implementation_ready=preview_implementation_ready,
        patch_preview_warning=preview_warning,
        safety_notes=notes,
        suggested_test_commands=tests,
    )


def _validated_relevant_files(
    relevant_files: Iterable[object] | None, project_root: Path
) -> list[str]:
    if isinstance(relevant_files, (str, bytes)):
        raise ValueError("Code proposal relevant_files must be a list.")
    root = Path(project_root).resolve(strict=False)
    files: list[str] = []
    for value in relevant_files or []:
        raw_path = str(value).strip().replace("\\", "/")
        if not raw_path:
            continue
        path = validate_relative_project_path(raw_path, root)
        if path not in files:
            files.append(path)
        if len(files) > MAX_LIST_ITEMS:
            raise ValueError("Code proposal relevant_files has too many entries.")
    return files


def _required_text(value: object, field_name: str) -> str:
    text = _redact_secret_values(str(value or "").strip())
    if not text:
        raise ValueError(f"{field_name} is required.")
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name} is too long.")
    return text


def _optional_text(value: object | None, field_name: str) -> str:
    text = _redact_secret_values(str(value or "").strip())
    if len(text) > MAX_TEXT_LENGTH:
        raise ValueError(f"{field_name} is too long.")
    return text


def _normalized_text_list(
    values: Iterable[object] | None, field_name: str
) -> list[str]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"Code proposal {field_name} must be a list.")
    normalized: list[str] = []
    for value in values or []:
        text = _redact_secret_values(str(value).strip())
        if not text:
            continue
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Code proposal {field_name} contains an item too long.")
        normalized.append(text)
        if len(normalized) > MAX_LIST_ITEMS:
            raise ValueError(f"Code proposal {field_name} has too many entries.")
    return normalized


def _reject_patch_content(text: str, field_name: str) -> None:
    if PATCH_MARKER_PATTERN.search(text):
        raise ValueError(
            f"Code proposal {field_name} must be planning text, not a patch or diff."
        )


def _redact_secret_values(text: str) -> str:
    redacted = text
    for pattern in SECRET_VALUE_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(
                lambda match: f"{match[1]}{match[2]}[redacted]", redacted
            )
        else:
            redacted = pattern.sub("[redacted]", redacted)
    return redacted


def _infer_target_surface(goal: str, approach: str, files: list[str]) -> str:
    text = _searchable_text(goal, approach, *files)
    if _mentions_database_or_migration(text):
        return "database/migrations"
    if _mentions_cli_terminal(text):
        return "CLI/terminal UI"
    if _mentions_command_center_ui(text):
        return "command center web UI"
    if _mentions_web_ui(text):
        return "command center web UI"
    if _mentions_backend_or_config(text):
        return "backend/service logic"
    return "unspecified"


def _target_surface(
    goal: str, approach: str, files: list[str], supplied_surface: object | None
) -> str:
    inferred_surface = _infer_target_surface(goal, approach, files)
    if inferred_surface != "unspecified":
        return inferred_surface
    return _normalized_target_surface(supplied_surface) or inferred_surface


def _normalized_target_surface(value: object | None) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    if text in {
        "command center web ui",
        "web command center ui",
        "web ui",
        "browser ui",
    }:
        return "command center web UI"
    if text in {"cli/terminal ui", "cli", "terminal", "terminal ui", "console"}:
        return "CLI/terminal UI"
    if text in {"backend/service logic", "backend", "service", "service logic"}:
        return "backend/service logic"
    if text in {"database/migrations", "database", "migrations", "migration"}:
        return "database/migrations"
    return "unspecified"


def _with_contextual_safety_notes(
    notes: list[str], goal: str, approach: str, target_surface: str
) -> list[str]:
    contextual_notes = list(notes)
    text = _searchable_text(goal, approach)

    if target_surface == "command center web UI":
        _append_unique(
            contextual_notes,
            "Assumption: this targets the command center web UI.",
        )
        _append_unique(
            contextual_notes,
            "Do not reinterpret this as a CLI/terminal ASCII UI unless explicitly requested.",
        )
    if _mentions_sensitive_config(text):
        _append_unique(
            contextual_notes,
            "Do not inspect, expose, or request .env, secret, credential, token, database, or runtime data contents.",
        )
        _append_unique(
            contextual_notes,
            "Prefer .env.example documentation and environment-variable based runtime configuration.",
        )
        _append_unique(
            contextual_notes,
            "Frame this as runtime config support, not as exposing .env contents or secret values.",
        )
    return contextual_notes


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _searchable_text(*values: object) -> str:
    return " ".join(str(value).lower() for value in values if value)


def _mentions_command_center_ui(text: str) -> bool:
    return "command center" in text and _mentions_ui(text)


def _mentions_web_ui(text: str) -> bool:
    return _mentions_ui(text) and any(
        marker in text
        for marker in (
            "web",
            "browser",
            "dashboard",
            "frontend",
            "front end",
            "css",
            "html",
            "page",
        )
    )


def _mentions_ui(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "ui",
            "interface",
            "avatar",
            "animation",
            "animate",
            "button",
            "layout",
            "screen",
            "component",
            "visual",
        )
    )


def _mentions_cli_terminal(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "cli",
            "terminal",
            "console",
            "command-line",
            "command line",
            "ascii",
            "tty",
        )
    )


def _mentions_backend_or_config(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "backend",
            "service",
            "config",
            "configuration",
            "environment variable",
            "env var",
            "runtime",
        )
    ) or _mentions_sensitive_config(text)


def _mentions_database_or_migration(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            "migration",
            "migrations",
            "schema",
            "table",
            "database schema",
        )
    )


def _mentions_sensitive_config(text: str) -> bool:
    return any(
        marker in text
        for marker in (
            ".env",
            "secret",
            "secrets",
            "credential",
            "credentials",
            "password",
            "token",
            "api key",
            "api_key",
            "database file",
            "database files",
            "runtime data",
        )
    )


def _proposal_title(requested_goal: str) -> str:
    normalized = " ".join(requested_goal.split())
    title = f"Code proposal: {normalized}"
    return title[:120]
