"""Shared bounded artifact path validation and atomic writes for freestyle slices.

Every freestyle artifact lives under one ignored local dev-state directory as
``<state>/<category>/<job-id>/<name>``. This module centralizes the symlink
and containment validation of those directories, the temporary-file-then-
replace write, and the safe relative artifact references that job metadata
records. It writes only inside the supplied dev-state root and refuses
rather than repairs unexpected paths.
"""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any

from app.dev.freestyle_coding import FreestyleJobRefusedError

MAX_ARTIFACT_REFERENCE_CHARS = 500


def encode_json_artifact(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def validated_artifact_dir(
    state: Path, category: str, job_id: str, *, label: str
) -> Path:
    """Create and validate one per-job artifact directory inside dev state."""
    parent = state / category
    job_dir = parent / job_id
    for directory in (parent, job_dir):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise FreestyleJobRefusedError(
                f"Freestyle {label} artifact path is unsafe."
            )
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.resolve(strict=True).relative_to(state)
        except (OSError, ValueError) as exc:
            raise FreestyleJobRefusedError(
                f"Freestyle {label} artifact path must stay inside local dev state."
            ) from exc
    return job_dir


def write_artifact_bytes(
    state: Path,
    category: str,
    job_id: str,
    name: str,
    data: bytes,
    *,
    label: str,
    allow_replace: bool = False,
) -> str:
    """Atomically write one artifact and return its state-relative POSIX path."""
    job_dir = validated_artifact_dir(state, category, job_id, label=label)
    path = job_dir / name
    temporary = job_dir / f"{name}.tmp"
    if allow_replace:
        for candidate in (path, temporary):
            if candidate.is_symlink() or (
                candidate.exists() and not candidate.is_file()
            ):
                raise FreestyleJobRefusedError(
                    f"Freestyle {label} artifact path is unsafe."
                )
    elif path.exists() or path.is_symlink():
        raise FreestyleJobRefusedError(f"Freestyle {label} artifact path is unsafe.")
    if temporary.exists() or temporary.is_symlink():
        raise FreestyleJobRefusedError(
            f"Freestyle {label} temporary artifact path is unsafe."
        )
    temporary.write_bytes(data)
    temporary.replace(path)
    return path.relative_to(state).as_posix()


def safe_artifact_reference(value: Any) -> str | None:
    """Return one bounded, relative, traversal-free artifact reference or None."""
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_ARTIFACT_REFERENCE_CHARS
        or "\x00" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()
