from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.code_proposal_path_safety import validate_relative_project_path
from app.config import ROOT_DIR
from app.secret_redaction import redact_secret_values


PROJECT_INSPECTION_MODE_NAME = "developer_code_workspace"
MAX_PROJECT_FILE_BYTES = 64_000
MAX_RETURNED_CHARS = 20_000
MAX_LISTED_PROJECT_FILES = 500
MAX_SEARCH_RESULTS = 25
MAX_SEARCH_QUERY_LENGTH = 120
SEARCH_SNIPPET_CHARS = 240
PROJECT_INSPECTION_PATH_LABEL = "Project inspection paths"

SAFE_SUFFIXES = {
    ".css",
    ".cjs",
    ".js",
    ".jsx",
    ".json",
    ".md",
    ".mjs",
    ".py",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}
BLOCKED_RUNTIME_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "cache",
    "caches",
    "data",
    "dist",
    "env",
    "node_modules",
    "venv",
}
BLOCKED_RUNTIME_NAMES = {
    ".env",
    ".env.example",
    "lewis.sqlite3",
}
# Runtime-data directories that project inspection reports as "runtime files"
# before the shared path validator labels them generic blocked runtime paths.
# Version-control, virtualenv, and cache directories such as .git, .venv, and
# __pycache__ intentionally keep the shared validator's wording.
RUNTIME_DATA_PARTS = {
    "cache",
    "caches",
    "data",
    "dist",
    "node_modules",
}


@dataclass(frozen=True)
class ProjectFileRead:
    path: str
    content: str
    truncated: bool
    size_bytes: int


@dataclass(frozen=True)
class ProjectSearchResult:
    path: str
    score: int
    snippet: str


class ProjectInspector:
    def __init__(self, project_root: Path = ROOT_DIR):
        self.project_root = Path(project_root).resolve(strict=False)

    def list_files(self) -> list[str]:
        if not self.project_root.exists() or not self.project_root.is_dir():
            return []

        files: list[str] = []
        for candidate in self._iter_project_file_candidates():
            path = self._safe_existing_file_path(candidate)
            if path is None:
                continue
            files.append(path)
            if len(files) >= MAX_LISTED_PROJECT_FILES:
                break
        return files

    def read_file(self, raw_path: str) -> ProjectFileRead:
        path = self._validate_requested_path(raw_path)
        resolved = self._resolve_existing_project_file(path)
        size_bytes = resolved.stat().st_size
        if size_bytes > MAX_PROJECT_FILE_BYTES:
            raise ValueError(
                f"{path} is too large for read-only inspection "
                f"({size_bytes} bytes; limit is {MAX_PROJECT_FILE_BYTES})."
            )

        data = resolved.read_bytes()
        text = _decode_text(data, path)
        redacted = redact_secret_values(text)
        truncated = len(redacted) > MAX_RETURNED_CHARS
        content = redacted[:MAX_RETURNED_CHARS]
        return ProjectFileRead(
            path=path,
            content=content,
            truncated=truncated,
            size_bytes=size_bytes,
        )

    def search(self, query: str) -> list[ProjectSearchResult]:
        terms = _search_terms(query)
        results: list[ProjectSearchResult] = []
        for path in self.list_files():
            try:
                read = self.read_file(path)
            except ValueError:
                continue
            lower_content = read.content.lower()
            score = sum(lower_content.count(term) for term in terms)
            if score <= 0:
                continue
            results.append(
                ProjectSearchResult(
                    path=path,
                    score=score,
                    snippet=_snippet(read.content, terms),
                )
            )
        results.sort(key=lambda item: (-item.score, item.path))
        return results[:MAX_SEARCH_RESULTS]

    def _safe_existing_file_path(self, candidate: Path) -> str | None:
        try:
            relative_path = candidate.relative_to(self.project_root).as_posix()
            path = self._validate_requested_path(relative_path)
            resolved = self._resolve_existing_project_file(path)
        except (OSError, ValueError):
            return None
        if resolved.stat().st_size > MAX_PROJECT_FILE_BYTES:
            return None
        try:
            _decode_text(resolved.read_bytes(), path)
        except ValueError:
            return None
        return path

    def _iter_project_file_candidates(self) -> list[Path]:
        return self._iter_allowed_directory(self.project_root)

    def _iter_allowed_directory(self, directory: Path) -> list[Path]:
        candidates: list[Path] = []
        for child in _sorted_children(directory):
            if child.is_dir():
                if child.is_symlink() or _blocked_directory_parts(
                    child.relative_to(self.project_root).parts
                ):
                    continue
                candidates.extend(self._iter_allowed_directory(child))
                continue
            candidates.append(child)
        return candidates

    def _validate_requested_path(self, raw_path: str) -> str:
        normalized = _normalize_raw_path(raw_path)
        # Classify the clearest refusal reasons first so users get specific
        # messages; the shared validator still runs afterwards and enforces
        # every remaining protection, so nothing is unblocked by the ordering.
        self._reject_blocked_file_names(normalized)
        self._reject_runtime_data_paths(normalized)
        path = validate_relative_project_path(
            normalized,
            self.project_root,
            PROJECT_INSPECTION_PATH_LABEL,
        )
        self._validate_allowed_project_area(path)
        return path

    def _reject_blocked_file_names(self, normalized_path: str) -> None:
        name = PurePosixPath(normalized_path).name.lower()
        if name in BLOCKED_RUNTIME_NAMES or name.startswith("."):
            raise ValueError(
                f"{PROJECT_INSPECTION_PATH_LABEL} include a blocked file name."
            )

    def _reject_runtime_data_paths(self, normalized_path: str) -> None:
        parts = tuple(
            part.lower() for part in PurePosixPath(normalized_path).parts
        )
        if any(part in RUNTIME_DATA_PARTS for part in parts):
            raise ValueError(f"{PROJECT_INSPECTION_PATH_LABEL} include runtime files.")

    def _validate_allowed_project_area(self, path: str) -> None:
        posix_path = PurePosixPath(path)
        parts = tuple(part.lower() for part in posix_path.parts)
        if any(part in BLOCKED_RUNTIME_PARTS for part in parts):
            raise ValueError(f"{PROJECT_INSPECTION_PATH_LABEL} include runtime files.")
        name = posix_path.name.lower()
        if name in BLOCKED_RUNTIME_NAMES or name.startswith(".env"):
            raise ValueError(f"{PROJECT_INSPECTION_PATH_LABEL} include env files.")
        if posix_path.suffix.lower() not in SAFE_SUFFIXES:
            raise ValueError(f"{PROJECT_INSPECTION_PATH_LABEL} require text files.")

    def _resolve_existing_project_file(self, path: str) -> Path:
        candidate = self.project_root / PurePosixPath(path)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.project_root)
        except FileNotFoundError as exc:
            raise ValueError(f"{path} does not exist.") from exc
        except (OSError, ValueError) as exc:
            raise ValueError(
                f"{PROJECT_INSPECTION_PATH_LABEL} must stay inside the project root."
            ) from exc
        if not resolved.is_file():
            raise ValueError(f"{path} is not a regular project file.")
        return resolved


def _normalize_raw_path(raw_path: str) -> str:
    return str(raw_path or "").strip().replace("\\", "/")


def _sorted_children(directory: Path) -> list[Path]:
    try:
        return sorted(directory.iterdir(), key=lambda path: path.name.lower())
    except OSError:
        return []


def _blocked_directory_parts(parts: tuple[str, ...]) -> bool:
    normalized_parts = tuple(part.lower() for part in parts)
    return any(
        part.startswith(".")
        or part in BLOCKED_RUNTIME_PARTS
        or "secret" in part
        or "credential" in part
        or "password" in part
        or "token" in part
        or "api_key" in part
        or "apikey" in part
        for part in normalized_parts
    )


def _decode_text(data: bytes, path: str) -> str:
    if b"\x00" in data:
        raise ValueError(f"{path} appears to be binary and cannot be read.")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path} is not valid UTF-8 text.") from exc


def _search_terms(query: str) -> list[str]:
    text = str(query or "").strip()
    if not text:
        raise ValueError("search_project query is required.")
    if len(text) > MAX_SEARCH_QUERY_LENGTH:
        raise ValueError("search_project query is too long.")
    return [term.lower() for term in text.split() if term.strip()]


def _snippet(text: str, terms: list[str]) -> str:
    lower_text = text.lower()
    positions = [lower_text.find(term) for term in terms if lower_text.find(term) >= 0]
    start = min(positions) if positions else 0
    start = max(0, start - 60)
    return " ".join(text[start : start + SEARCH_SNIPPET_CHARS].split())
