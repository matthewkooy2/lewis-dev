from __future__ import annotations

import re
from pathlib import Path, PurePosixPath, PureWindowsPath


BLOCKED_PATH_PARTS = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "data",
    "env",
    "venv",
}
BLOCKED_FILE_NAMES = {
    ".env",
    "lewis.sqlite3",
}
BLOCKED_SUFFIXES = {
    ".db",
    ".key",
    ".pem",
    ".pfx",
    ".pyo",
    ".pyc",
    ".sqlite",
    ".sqlite3",
}
SECRET_NAME_PATTERN = re.compile(
    r"(?:secret|credential|password|passwd|token|api[_-]?key|private[_-]?key|"
    r"id_rsa|id_dsa|id_ecdsa|id_ed25519)",
    re.IGNORECASE,
)


def validate_relative_project_path(
    raw_path: str, project_root: Path, field_label: str = "Code proposal relevant files"
) -> str:
    if "\x00" in raw_path:
        raise ValueError(f"{field_label} paths cannot contain NUL bytes.")
    posix_path = PurePosixPath(raw_path)
    windows_path = PureWindowsPath(raw_path)
    if posix_path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise ValueError(f"{field_label} must be relative paths.")
    if any(part in {"", ".", ".."} for part in posix_path.parts):
        raise ValueError(f"{field_label} cannot use dot segments.")
    normalized_parts = tuple(part.lower() for part in posix_path.parts)
    if any(part in BLOCKED_PATH_PARTS for part in normalized_parts):
        raise ValueError(f"{field_label} include a blocked runtime path.")
    name = normalized_parts[-1] if normalized_parts else ""
    if (
        name in BLOCKED_FILE_NAMES
        or name.startswith(".")
        or any(part.startswith(".") for part in normalized_parts)
    ):
        raise ValueError(f"{field_label} include a blocked file name.")
    if PurePosixPath(raw_path).suffix.lower() in BLOCKED_SUFFIXES:
        raise ValueError(f"{field_label} include a blocked file type.")
    if any(SECRET_NAME_PATTERN.search(part) for part in posix_path.parts):
        raise ValueError(f"{field_label} include a sensitive path name.")
    try:
        # Path.resolve() raises OSError (not ValueError) on Windows for
        # reserved device names (CONIN$, CONOUT$) and wildcard characters.
        resolved = (project_root / posix_path).resolve(strict=False)
    except OSError as exc:
        raise ValueError(
            f"{field_label} include an invalid or reserved file system name."
        ) from exc
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(
            f"{field_label} must stay inside the Lewis project root."
        ) from exc
    return posix_path.as_posix()
