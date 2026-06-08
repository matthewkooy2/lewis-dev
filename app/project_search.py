from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.code_proposal_path_safety import validate_relative_project_path
from app.config import ROOT_DIR


MAX_FILE_BYTES = 512 * 1024
MAX_RESULTS = 10
MAX_MATCHES_PER_FILE = 3
TEXT_SAMPLE_BYTES = 2048
MAX_EXCERPT_LINES = 80
MAX_EXCERPT_CHARS = 8000
EXCERPT_CONTEXT_LINES = 20


@dataclass(frozen=True)
class ProjectSearchResult:
    path: str
    line_number: int | None
    snippet: str


@dataclass(frozen=True)
class ProjectFileExcerpt:
    path: str
    start_line: int
    text: str


class ProjectFileSearch:
    """Read-only inventory and text search for safe Lewis project files."""

    def __init__(
        self,
        project_root: Path = ROOT_DIR,
        *,
        max_file_bytes: int = MAX_FILE_BYTES,
    ):
        self.project_root = Path(project_root)
        self._resolved_project_root = self.project_root.resolve(strict=False)
        self.max_file_bytes = max_file_bytes

    def list_files(self) -> list[str]:
        if not self._project_root_is_usable():
            return []

        files: list[str] = []
        for path in sorted(self.project_root.rglob("*")):
            safe_path = self._validated_text_file(path)
            if safe_path is None:
                continue
            files.append(self._relative_path(safe_path))
        return sorted(files)

    def search(self, query: str, limit: int = MAX_RESULTS) -> list[ProjectSearchResult]:
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        max_results = max(1, min(limit, MAX_RESULTS))
        relative_paths = self.list_files()
        path_matches = [
            ProjectSearchResult(
                path=relative_path,
                line_number=None,
                snippet="path match",
            )
            for relative_path in relative_paths
            if any(term in relative_path.lower() for term in terms)
        ]
        results = path_matches[:max_results]
        if len(results) >= max_results:
            return results

        for relative_path in relative_paths:
            path = self.project_root / PurePosixPath(relative_path)
            safe_path = self._validated_text_file(path)
            if safe_path is None:
                continue

            text = self._read_text(safe_path)
            matches_for_file = 0
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not any(term in line.lower() for term in terms):
                    continue
                results.append(
                    ProjectSearchResult(
                        path=relative_path,
                        line_number=line_number,
                        snippet=self._snippet(line),
                    )
                )
                matches_for_file += 1
                if (
                    len(results) >= max_results
                    or matches_for_file >= MAX_MATCHES_PER_FILE
                ):
                    break
            if len(results) >= max_results:
                break
        return results

    def is_safe_relative_path(self, raw_path: str) -> bool:
        return self._safe_relative_path(raw_path) is not None

    def is_safe_file_path(self, raw_path: str) -> bool:
        safe_path = self._safe_relative_path(raw_path)
        if safe_path is None:
            return False
        return (
            self._validated_text_file(self.project_root / PurePosixPath(safe_path))
            is not None
        )

    def read_excerpt(
        self,
        raw_path: str,
        *,
        focus_terms: tuple[str, ...] = (),
        max_lines: int = MAX_EXCERPT_LINES,
        max_chars: int = MAX_EXCERPT_CHARS,
    ) -> ProjectFileExcerpt | None:
        """Return bounded context from a verified safe project text file."""
        safe_path = self._safe_relative_path(raw_path)
        if safe_path is None:
            return None
        path = self._validated_text_file(self.project_root / PurePosixPath(safe_path))
        if path is None:
            return None
        lines = self._read_text(path).splitlines()
        if not lines:
            return ProjectFileExcerpt(path=safe_path, start_line=1, text="")

        line_limit = max(1, min(max_lines, MAX_EXCERPT_LINES))
        char_limit = max(1, min(max_chars, MAX_EXCERPT_CHARS))
        match_index = self._first_matching_line(lines, focus_terms)
        start_index = max(0, match_index - EXCERPT_CONTEXT_LINES)
        excerpt_lines = lines[start_index : start_index + line_limit]
        text = "\n".join(excerpt_lines)[:char_limit]
        return ProjectFileExcerpt(
            path=safe_path,
            start_line=start_index + 1,
            text=text,
        )

    def _project_root_is_usable(self) -> bool:
        return (
            self.project_root.exists()
            and self.project_root.is_dir()
            and not self.project_root.is_symlink()
        )

    def _validated_text_file(self, path: Path) -> Path | None:
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(self._resolved_project_root)
        except (OSError, ValueError):
            return None
        if not resolved_path.is_file() or resolved_path.is_symlink():
            return None
        relative_path = self._relative_path(resolved_path)
        if self._safe_relative_path(relative_path) is None:
            return None
        try:
            if resolved_path.stat().st_size > self.max_file_bytes:
                return None
        except OSError:
            return None
        if not self._is_text_file(resolved_path):
            return None
        return resolved_path

    def _safe_relative_path(self, raw_path: str) -> str | None:
        normalized = raw_path.strip().replace("\\", "/")
        if not normalized:
            return None
        try:
            return validate_relative_project_path(
                normalized,
                self._resolved_project_root,
                "Project file search",
            )
        except ValueError:
            return None

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self._resolved_project_root).as_posix()

    def _is_text_file(self, path: Path) -> bool:
        try:
            sample = path.read_bytes()[:TEXT_SAMPLE_BYTES]
        except OSError:
            return False
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            return False
        return True

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return ""

    @staticmethod
    def _first_matching_line(lines: list[str], focus_terms: tuple[str, ...]) -> int:
        normalized_terms = tuple(term.lower() for term in focus_terms if term.strip())
        for index, line in enumerate(lines):
            if any(term in line.lower() for term in normalized_terms):
                return index
        return 0

    @staticmethod
    def _snippet(line: str, size: int = 160) -> str:
        return " ".join(line.split())[:size]
