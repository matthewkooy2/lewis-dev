from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}
NO_DOCUMENTS_MESSAGE = "No documents found in data/documents/."
NO_MATCHES_MESSAGE = "I searched your local documents but did not find a clear match."
SKIPPED_DIRECTORY_NAMES = {
    "$recycle.bin",
    ".git",
    "__pycache__",
    "system volume information",
}
SKIPPED_FILE_NAMES = {
    ".env",
    "desktop.ini",
    "thumbs.db",
}
SECRET_NAME_PATTERN = re.compile(
    r"(?:secret|credential|password|passwd|token|api[_-]?key|private[_-]?key|"
    r"id_rsa|id_dsa|id_ecdsa|id_ed25519|\.pem$|\.key$|\.pfx$|\.crt$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SearchResult:
    path: str
    score: int
    snippet: str


class DocumentSearch:
    def __init__(self, documents_dir: Path):
        self.documents_dir = Path(documents_dir)
        self._resolved_documents_dir = self.documents_dir.resolve(strict=False)

    def list_files(self) -> list[str]:
        if not self._documents_root_is_usable():
            return []
        files: list[str] = []
        for path in sorted(self.documents_dir.rglob("*")):
            safe_path = self._validated_supported_file(path)
            if safe_path is None:
                continue
            files.append(self._relative_path(safe_path))
        return files

    def search(self, query: str, limit: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []

        terms = [term.lower() for term in query.split() if term.strip()]
        results: list[SearchResult] = []

        for relative_path in self.list_files():
            path = self.documents_dir / relative_path
            safe_path = self._validated_supported_file(path)
            if safe_path is None:
                continue
            text = self._read_text(safe_path)
            if not text.strip():
                continue
            lower_text = text.lower()
            score = sum(lower_text.count(term) for term in terms)
            if score <= 0:
                continue
            results.append(
                SearchResult(
                    path=relative_path,
                    score=score,
                    snippet=self._snippet(text, terms),
                )
            )

        results.sort(key=lambda item: item.score, reverse=True)
        return results[:limit]

    def has_documents(self) -> bool:
        return bool(self.list_files())

    def _documents_root_is_usable(self) -> bool:
        return (
            self.documents_dir.exists()
            and self.documents_dir.is_dir()
            and not self.documents_dir.is_symlink()
        )

    def _validated_supported_file(self, path: Path) -> Path | None:
        if self._should_skip_by_name(path):
            return None
        try:
            resolved_path = path.resolve(strict=True)
            resolved_path.relative_to(self._resolved_documents_dir)
        except (OSError, ValueError):
            return None
        if (
            not resolved_path.is_file()
            or resolved_path.suffix.lower() not in SUPPORTED_EXTENSIONS
        ):
            return None
        return resolved_path

    def _should_skip_by_name(self, path: Path) -> bool:
        relative_parts = self._relative_parts(path)
        if not relative_parts:
            return True
        for part in relative_parts:
            normalized_part = part.lower()
            if (
                normalized_part.startswith(".")
                or normalized_part in SKIPPED_DIRECTORY_NAMES
            ):
                return True
            if SECRET_NAME_PATTERN.search(part):
                return True
        name = relative_parts[-1].lower()
        return name in SKIPPED_FILE_NAMES

    def _relative_parts(self, path: Path) -> tuple[str, ...]:
        try:
            return path.relative_to(self.documents_dir).parts
        except ValueError:
            return ()

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self._resolved_documents_dir).as_posix()

    def _read_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".pdf":
            return self._read_pdf(path)
        return ""

    def _read_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            return ""

        try:
            reader = PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            return ""

    @staticmethod
    def _snippet(text: str, terms: list[str], size: int = 240) -> str:
        lower_text = text.lower()
        positions = [
            lower_text.find(term) for term in terms if lower_text.find(term) >= 0
        ]
        start = min(positions) if positions else 0
        start = max(0, start - 60)
        snippet = " ".join(text[start : start + size].split())
        return snippet
