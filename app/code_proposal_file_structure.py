from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.code_proposal_path_safety import validate_relative_project_path


EXISTING_PROJECT_FILE = "existing project file"
NEW_FILE_UNDER_EXISTING_SAFE_PROJECT_DIRECTORY = (
    "new file under an existing safe project directory"
)
SUSPICIOUS_OR_INVALID_PROJECT_STRUCTURE = "suspicious or invalid project structure"


@dataclass(frozen=True)
class ProjectPathClassification:
    path: str
    classification: str

    def as_payload(self) -> dict[str, str]:
        return {
            "path": self.path,
            "classification": self.classification,
        }


def classify_project_paths(
    raw_paths: list[str],
    project_root: Path,
    field_label: str,
) -> list[ProjectPathClassification]:
    root = Path(project_root).resolve(strict=False)
    classified: list[ProjectPathClassification] = []
    for raw_path in raw_paths:
        path = validate_relative_project_path(raw_path, root, field_label)
        if any(item.path == path for item in classified):
            continue
        classified.append(classify_validated_project_path(path, root))
    return classified


def classify_validated_project_path(
    path: str, project_root: Path
) -> ProjectPathClassification:
    root = Path(project_root).resolve(strict=False)
    candidate = root / PurePosixPath(path)
    if candidate.is_file():
        classification = EXISTING_PROJECT_FILE
    elif not candidate.exists() and candidate.parent.is_dir():
        classification = NEW_FILE_UNDER_EXISTING_SAFE_PROJECT_DIRECTORY
    else:
        classification = SUSPICIOUS_OR_INVALID_PROJECT_STRUCTURE
    return ProjectPathClassification(path=path, classification=classification)


def path_has_valid_project_structure(path: str, project_root: Path) -> bool:
    return (
        classify_validated_project_path(path, project_root).classification
        != SUSPICIOUS_OR_INVALID_PROJECT_STRUCTURE
    )
