from __future__ import annotations

import shlex
from pathlib import Path

from app.code_proposal_file_structure import (
    ProjectPathClassification,
    SUSPICIOUS_OR_INVALID_PROJECT_STRUCTURE,
    classify_project_paths,
)
from app.code_proposal_path_safety import validate_relative_project_path


PATCH_PREVIEW_PATH_LABEL = "Code proposal patch_preview file paths"
PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING = "Patch preview contains suspicious paths and should be revised before implementation."


def validate_patch_preview(patch_preview: str, project_root: Path) -> str:
    preview = patch_preview.strip()
    if not preview:
        return ""

    root = Path(project_root).resolve(strict=False)
    for path in extract_patch_preview_paths(preview):
        validate_relative_project_path(path, root, PATCH_PREVIEW_PATH_LABEL)
    return preview


def classify_patch_preview_paths(
    patch_preview: str, project_root: Path
) -> list[ProjectPathClassification]:
    preview = patch_preview.strip()
    if not preview:
        return []
    return classify_project_paths(
        extract_patch_preview_paths(preview),
        project_root,
        PATCH_PREVIEW_PATH_LABEL,
    )


def patch_preview_is_implementation_ready(
    classifications: list[ProjectPathClassification],
) -> bool:
    return bool(classifications) and not patch_preview_has_suspicious_paths(
        classifications
    )


def patch_preview_has_suspicious_paths(
    classifications: list[ProjectPathClassification] | list[dict[str, str]],
) -> bool:
    for item in classifications:
        classification = (
            item.classification
            if isinstance(item, ProjectPathClassification)
            else item.get("classification")
        )
        if classification == SUSPICIOUS_OR_INVALID_PROJECT_STRUCTURE:
            return True
    return False


def patch_preview_warning(
    classifications: list[ProjectPathClassification] | list[dict[str, str]],
) -> str:
    if patch_preview_has_suspicious_paths(classifications):
        return PATCH_PREVIEW_SUSPICIOUS_PATHS_WARNING
    return ""


def extract_patch_preview_paths(patch_preview: str) -> list[str]:
    paths: list[str] = []
    for line in patch_preview.splitlines():
        for path in _paths_from_patch_line(line.strip()):
            if path not in paths:
                paths.append(path)
    return paths


def _paths_from_patch_line(line: str) -> list[str]:
    if not line:
        return []

    tokens = _split_patch_line(line)
    if len(tokens) >= 4 and tokens[0] == "diff" and tokens[1] == "--git":
        return _present_paths(
            [_clean_diff_path(tokens[2]), _clean_diff_path(tokens[3])]
        )
    if len(tokens) >= 2 and tokens[0] in {"---", "+++", "Index:"}:
        return _present_paths([_clean_diff_path(tokens[1])])
    if len(tokens) >= 3 and (tokens[0], tokens[1]) in {
        ("rename", "from"),
        ("rename", "to"),
        ("copy", "from"),
        ("copy", "to"),
    }:
        return _present_paths([_clean_diff_path(tokens[2])])
    return []


def _split_patch_line(line: str) -> list[str]:
    try:
        return shlex.split(line)
    except ValueError:
        return line.split()


def _clean_diff_path(raw_path: str) -> str:
    path = raw_path.strip()
    if "\t" in path:
        path = path.split("\t", 1)[0].strip()
    if path == "/dev/null":
        return ""
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


def _present_paths(paths: list[str]) -> list[str]:
    return [path for path in paths if path]
