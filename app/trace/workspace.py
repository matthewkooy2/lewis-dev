"""Safe, bounded, read-only file access for code tracing.

Reuses the shared Lewis path validator and layers trace-specific rules on
top: only analyzable source suffixes, no hidden/runtime/dependency/generated
paths, no symlinks, no binaries, size caps, and a hard file-count bound.
The workspace never writes, executes, or touches anything outside its root.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from app.code_proposal_path_safety import validate_relative_project_path
from app.config import ROOT_DIR
from app.trace.model import (
    LANGUAGE_JAVASCRIPT,
    LANGUAGE_PYTHON,
    LANGUAGE_TYPESCRIPT,
    TraceLimits,
)


TRACE_PATH_LABEL = "Code trace paths"

LANGUAGE_BY_SUFFIX = {
    ".py": LANGUAGE_PYTHON,
    ".js": LANGUAGE_JAVASCRIPT,
    ".jsx": LANGUAGE_JAVASCRIPT,
    ".mjs": LANGUAGE_JAVASCRIPT,
    ".cjs": LANGUAGE_JAVASCRIPT,
    ".ts": LANGUAGE_TYPESCRIPT,
    ".tsx": LANGUAGE_TYPESCRIPT,
}

# Directories that hold dependencies, runtime data, or generated output.
# The shared validator already blocks .git, data, env, venv, caches, and any
# hidden component; these extend the same policy for JS/TS project layouts.
BLOCKED_DIRECTORY_PARTS = {
    "bower_components",
    "build",
    "cache",
    "caches",
    "coverage",
    "dist",
    "node_modules",
    "out",
    "site-packages",
    "vendor",
}

# Generated or declaration-only files that would pollute call tracing.
BLOCKED_FILE_ENDINGS = (
    ".min.js",
    ".min.mjs",
    ".bundle.js",
    ".d.ts",
    ".d.mts",
    ".d.cts",
)


@dataclass(frozen=True)
class WorkspaceFile:
    path: str
    language: str
    size_bytes: int


class TraceWorkspace:
    """Discovers and reads analyzable source files inside one root."""

    def __init__(
        self,
        root: Path = ROOT_DIR,
        *,
        limits: TraceLimits | None = None,
    ):
        self.root = Path(root)
        self.resolved_root = self.root.resolve(strict=False)
        self.limits = limits or TraceLimits()
        self._discovery_truncated = False

    @property
    def discovery_truncated(self) -> bool:
        return self._discovery_truncated

    def language_for_path(self, raw_path: str) -> str | None:
        return LANGUAGE_BY_SUFFIX.get(PurePosixPath(raw_path).suffix.lower())

    def safe_relative_path(self, raw_path: str) -> str | None:
        """Validate one requested path; None when any rule rejects it."""
        normalized = str(raw_path or "").strip().replace("\\", "/")
        if not normalized:
            return None
        try:
            validated = validate_relative_project_path(
                normalized, self.resolved_root, TRACE_PATH_LABEL
            )
        except ValueError:
            return None
        posix = PurePosixPath(validated)
        lowered_parts = tuple(part.lower() for part in posix.parts)
        if any(part in BLOCKED_DIRECTORY_PARTS for part in lowered_parts):
            return None
        lowered_name = posix.name.lower()
        if any(lowered_name.endswith(ending) for ending in BLOCKED_FILE_ENDINGS):
            return None
        if posix.suffix.lower() not in LANGUAGE_BY_SUFFIX:
            return None
        return validated

    def list_source_files(self) -> list[WorkspaceFile]:
        """Bounded discovery of analyzable files, sorted for determinism."""
        self._discovery_truncated = False
        if (
            not self.resolved_root.exists()
            or not self.resolved_root.is_dir()
            or self.root.is_symlink()
        ):
            return []

        found: list[WorkspaceFile] = []
        for candidate in self._iter_candidates(self.resolved_root):
            record = self._validated_file(candidate)
            if record is None:
                continue
            found.append(record)
            if len(found) >= self.limits.max_files:
                self._discovery_truncated = True
                break
        return found

    def read_source(self, raw_path: str) -> str | None:
        """Full text of one verified safe source file; None when rejected."""
        validated = self.safe_relative_path(raw_path)
        if validated is None:
            return None
        resolved = self._resolve_inside_root(validated)
        if resolved is None:
            return None
        return self._read_text(resolved)

    def _iter_candidates(self, directory: Path) -> list[Path]:
        candidates: list[Path] = []
        try:
            children = sorted(directory.iterdir(), key=lambda path: path.name.lower())
        except OSError:
            return []
        for child in children:
            name_lower = child.name.lower()
            if name_lower.startswith("."):
                continue
            if child.is_symlink():
                continue
            if child.is_dir():
                if name_lower in BLOCKED_DIRECTORY_PARTS:
                    continue
                candidates.extend(self._iter_candidates(child))
                continue
            if child.suffix.lower() in LANGUAGE_BY_SUFFIX:
                candidates.append(child)
        return candidates

    def _validated_file(self, candidate: Path) -> WorkspaceFile | None:
        try:
            relative = candidate.relative_to(self.resolved_root).as_posix()
        except ValueError:
            return None
        validated = self.safe_relative_path(relative)
        if validated is None:
            return None
        resolved = self._resolve_inside_root(validated)
        if resolved is None:
            return None
        try:
            size = resolved.stat().st_size
        except OSError:
            return None
        if size > self.limits.max_file_bytes:
            return None
        if not self._looks_like_text(resolved):
            return None
        language = self.language_for_path(validated)
        if language is None:
            return None
        return WorkspaceFile(path=validated, language=language, size_bytes=size)

    @staticmethod
    def _looks_like_text(resolved: Path) -> bool:
        try:
            with resolved.open("rb") as handle:
                sample = handle.read(2048)
        except OSError:
            return False
        if b"\x00" in sample:
            return False
        try:
            sample.decode("utf-8")
        except UnicodeDecodeError:
            # The sample may split a multi-byte character; tolerate the tail.
            try:
                sample[:-3].decode("utf-8")
            except UnicodeDecodeError:
                return False
        return True

    def _resolve_inside_root(self, validated_path: str) -> Path | None:
        candidate = self.resolved_root / PurePosixPath(validated_path)
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.resolved_root)
        except (OSError, ValueError):
            return None
        if not resolved.is_file() or candidate.is_symlink():
            return None
        return resolved

    def _read_text(self, resolved: Path) -> str | None:
        try:
            if resolved.stat().st_size > self.limits.max_file_bytes:
                return None
            data = resolved.read_bytes()
        except OSError:
            return None
        if b"\x00" in data:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None
