"""Deterministic, read-only review artifacts for isolated freestyle worktrees.

This developer-only slice inspects one approved, successfully materialized
freestyle worktree and persists a bounded review artifact in ignored local dev
state. It does not call a model, run repository code or checks, mutate source,
touch approvals or SQLite, or perform any Git write, remote, or network action.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import difflib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
from typing import Any
import uuid

from app.code_proposal_path_safety import (
    BLOCKED_FILE_NAMES,
    BLOCKED_PATH_PARTS,
    BLOCKED_SUFFIXES,
    SECRET_NAME_PATTERN,
)
from app.config import ROOT_DIR
from app.dev.freestyle_coding import (
    DEV_STATE_DIR_NAME,
    FreestyleJob,
    FreestyleJobRefusedError,
    _bounded_error,
    _is_commit_id,
    _require_explicit_developer_approval,
    _require_state_inside_project,
    _utc_now,
    _validated_job_worktree,
    _write_job,
    get_freestyle_job,
)

REVIEW_SCHEMA_VERSION = 1
MAX_REVIEW_CHANGED_FILES = 500
MAX_REVIEW_PATH_CHARS = 500
MAX_REVIEW_TEXT_FILE_BYTES = 64 * 1024
MAX_REVIEW_DIFF_SNIPPET_BYTES = 12 * 1024
MAX_REVIEW_TOTAL_DIFF_BYTES = 96 * 1024
MAX_REVIEW_ARTIFACT_BYTES = 500 * 1024
MAX_REVIEW_REFERENCE_CHARS = 500
_GIT_TIMEOUT_SECONDS = 30

_REVIEW_GIT_PREFIX = (
    "git",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.excludesFile=/dev/null",
)
_STATUS_SUFFIX = (
    "status",
    "--porcelain=v1",
    "-z",
    "--untracked-files=all",
    "--ignored=no",
    "--no-renames",
)
_IGNORED_SUFFIX = (
    "ls-files",
    "--others",
    "--ignored",
    "--exclude-standard",
    "-z",
)
_GENERATED_OR_CACHE_PARTS = {
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "cache",
    "coverage",
    "dist",
    "generated",
    "htmlcov",
    "node_modules",
    "target",
    "vendor",
}
_BINARY_SUFFIXES = {
    ".7z",
    ".a",
    ".avi",
    ".bmp",
    ".bz2",
    ".class",
    ".dll",
    ".dylib",
    ".eot",
    ".exe",
    ".gif",
    ".gz",
    ".ico",
    ".jar",
    ".jpeg",
    ".jpg",
    ".mov",
    ".mp3",
    ".mp4",
    ".o",
    ".otf",
    ".pdf",
    ".png",
    ".pyc",
    ".so",
    ".tar",
    ".tiff",
    ".ttf",
    ".wasm",
    ".webp",
    ".woff",
    ".woff2",
    ".zip",
}
_SENSITIVE_CONTENT_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}", re.IGNORECASE),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"),
    re.compile(
        r"\b(?:api[_-]?key|credential|password|passwd|secret|token)\b"
        r"\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']",
        re.IGNORECASE,
    ),
)
_CONFLICT_STATUSES = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
_RECOVERY_MESSAGE = (
    "Recover or verify the isolated worktree and job metadata before retrying review."
)
_EXECUTION_STATEMENT = (
    "This review is read-only: merge, push, pull, fetch, commit, PR creation, "
    "auto-merge, checks, formatters, package managers, repository scripts, and "
    "model calls were not performed."
)


@dataclass(frozen=True)
class _GitResult:
    args: tuple[str, ...]
    returncode: int
    stdout: bytes
    stderr: str


@dataclass(frozen=True)
class _ChangedPath:
    path: str
    status: str
    git_status: str


def generate_freestyle_review(
    job_id: str,
    *,
    project_root: Path | None = None,
    state_root: Path | None = None,
) -> FreestyleJob:
    """Persist one bounded review artifact without mutating the job worktree."""
    root = (project_root or ROOT_DIR).resolve()
    state = (state_root or root / DEV_STATE_DIR_NAME).resolve()
    _require_state_inside_project(state, root)
    job = get_freestyle_job(job_id, project_root=root, state_root=state)

    started_at = _utc_now()
    try:
        _require_review_prerequisites(job)
    except Exception as exc:
        _persist_review_failure(job, state, exc, started_at=started_at)
        raise FreestyleJobRefusedError(
            f"Could not review freestyle job {job_id}; the error was recorded in "
            "local dev state."
        ) from exc

    started = replace(
        job,
        review_status="reviewing",
        review_started_at=started_at,
        review_completed_at=None,
        review_artifact_path=None,
        review_changed_file_count=0,
        review_errors=(),
        updated_at=started_at,
    )
    _write_job(started, state)
    changed_count = 0
    try:
        worktree = _validated_job_worktree(started, root, state)
        current_head = _read_current_head(worktree)
        changed = _read_changed_paths(worktree)
        changed_count = len(changed)
        artifact = _build_review_artifact(started, worktree, current_head, changed)
        artifact_path = _write_review_artifact(artifact, state, started.job_id)
    except Exception as exc:
        failed = replace(
            started,
            review_status="error",
            review_completed_at=_utc_now(),
            review_changed_file_count=changed_count,
            review_errors=(_bounded_error(str(exc), "Review generation failed."),),
            updated_at=_utc_now(),
        )
        _write_job(failed, state)
        raise FreestyleJobRefusedError(
            f"Could not review freestyle job {job_id}; the error was recorded in "
            "local dev state."
        ) from exc

    completed_at = artifact["generated_at"]
    completed = replace(
        started,
        review_status="reviewed",
        review_completed_at=completed_at,
        review_artifact_path=artifact_path,
        review_changed_file_count=changed_count,
        review_errors=(),
        updated_at=completed_at,
    )
    _write_job(completed, state)
    return completed


def _require_review_prerequisites(job: FreestyleJob) -> None:
    _require_explicit_developer_approval(job, "review")
    if job.status != "materialized" or job.materialization_status != "materialized":
        raise FreestyleJobRefusedError(
            f"Freestyle job {job.job_id!r} must be successfully materialized before "
            "review."
        )
    if job.edit_apply_status in {"applying", "inconsistent"}:
        raise FreestyleJobRefusedError(
            f"Freestyle job {job.job_id!r} has an in-progress or inconsistent edit "
            f"application. {_RECOVERY_MESSAGE}"
        )


def _persist_review_failure(
    job: FreestyleJob, state: Path, exc: Exception, *, started_at: str
) -> None:
    completed_at = _utc_now()
    failed = replace(
        job,
        review_status="error",
        review_started_at=started_at,
        review_completed_at=completed_at,
        review_artifact_path=None,
        review_changed_file_count=0,
        review_errors=(_bounded_error(str(exc), "Review generation failed."),),
        updated_at=completed_at,
    )
    _write_job(failed, state)


def _read_current_head(worktree: Path) -> dict[str, str]:
    commit_result = _run_review_git(
        [*_REVIEW_GIT_PREFIX, "rev-parse", "HEAD"], worktree
    )
    branch_result = _run_review_git(
        [*_REVIEW_GIT_PREFIX, "rev-parse", "--abbrev-ref", "HEAD"], worktree
    )
    commit = _decode_short_git_text(commit_result, "Could not read review HEAD.")
    branch = _decode_short_git_text(branch_result, "Could not read review branch.")
    if not _is_commit_id(commit):
        raise FreestyleJobRefusedError("Review HEAD is not a valid commit id.")
    return {"commit": commit, "branch": branch}


def _read_changed_paths(worktree: Path) -> tuple[_ChangedPath, ...]:
    result = _run_review_git([*_REVIEW_GIT_PREFIX, *_STATUS_SUFFIX], worktree)
    if result.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(result.stderr, "Could not read isolated worktree status.")
        )
    entries: list[_ChangedPath] = []
    seen: set[str] = set()
    for raw_entry in result.stdout.split(b"\x00"):
        if not raw_entry:
            continue
        if len(raw_entry) < 4 or raw_entry[2:3] != b" ":
            raise FreestyleJobRefusedError("Git status returned malformed output.")
        try:
            git_status = raw_entry[:2].decode("ascii")
            path = raw_entry[3:].decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FreestyleJobRefusedError(
                "Git status returned a non-UTF-8 path."
            ) from exc
        path = _validated_review_path(path)
        if git_status in _CONFLICT_STATUSES or "U" in git_status:
            raise FreestyleJobRefusedError(
                f"Isolated worktree has unresolved Git state at {path!r}. "
                f"{_RECOVERY_MESSAGE}"
            )
        _append_changed_path(entries, seen, path, git_status)
    ignored = _run_review_git([*_REVIEW_GIT_PREFIX, *_IGNORED_SUFFIX], worktree)
    if ignored.returncode != 0:
        raise FreestyleJobRefusedError(
            _bounded_error(ignored.stderr, "Could not read ignored worktree paths.")
        )
    for raw_path in ignored.stdout.split(b"\x00"):
        if not raw_path:
            continue
        try:
            path = raw_path.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise FreestyleJobRefusedError(
                "Git ignored-path listing returned a non-UTF-8 path."
            ) from exc
        _append_changed_path(entries, seen, _validated_review_path(path), "!!")
    return tuple(sorted(entries, key=lambda item: item.path.casefold()))


def _append_changed_path(
    entries: list[_ChangedPath], seen: set[str], path: str, git_status: str
) -> None:
    if path.casefold() in seen:
        raise FreestyleJobRefusedError("Git status returned duplicate paths.")
    seen.add(path.casefold())
    entries.append(
        _ChangedPath(
            path=path,
            status=_summary_status(git_status),
            git_status=git_status,
        )
    )
    if len(entries) > MAX_REVIEW_CHANGED_FILES:
        raise FreestyleJobRefusedError(
            "Review contains too many changed files for a bounded artifact."
        )


def _summary_status(git_status: str) -> str:
    if git_status in {"??", "!!"} or "A" in git_status:
        return "added"
    if "D" in git_status:
        return "deleted"
    if any(marker in git_status for marker in ("M", "T", "R", "C")):
        return "modified"
    raise FreestyleJobRefusedError(
        f"Git status returned unsupported state {git_status!r}."
    )


def _validated_review_path(path: str) -> str:
    if (
        not path
        or len(path) > MAX_REVIEW_PATH_CHARS
        or "\x00" in path
        or "\\" in path
        or any(ord(character) < 32 for character in path)
    ):
        raise FreestyleJobRefusedError("Git status returned an unsafe changed path.")
    posix = PurePosixPath(path)
    if posix.is_absolute() or any(part in {"", ".", ".."} for part in posix.parts):
        raise FreestyleJobRefusedError("Git status returned an unsafe changed path.")
    return posix.as_posix()


def _build_review_artifact(
    job: FreestyleJob,
    worktree: Path,
    current_head: dict[str, str],
    changed: tuple[_ChangedPath, ...],
) -> dict[str, Any]:
    if current_head["commit"] != job.base_commit or current_head["branch"] != job.branch:
        raise FreestyleJobRefusedError(
            "Isolated worktree HEAD metadata changed during review. "
            f"{_RECOVERY_MESSAGE}"
        )

    files: list[dict[str, Any]] = []
    snippets: list[dict[str, Any]] = []
    omissions: list[dict[str, str]] = []
    additions = 0
    deletions = 0
    unavailable_stats = 0
    total_diff_bytes = 0
    for item in changed:
        summary, snippet, omission = _review_changed_file(
            item, worktree, current_head["commit"]
        )
        if summary["additions"] is None:
            unavailable_stats += 1
        else:
            additions += summary["additions"]
            deletions += summary["deletions"]
        if snippet is not None:
            snippet_bytes = len(snippet["unified_diff"].encode("utf-8"))
            if total_diff_bytes + snippet_bytes <= MAX_REVIEW_TOTAL_DIFF_BYTES:
                snippets.append(snippet)
                total_diff_bytes += snippet_bytes
            else:
                summary["diff_snippet_included"] = False
                summary["diff_snippet_omission_reason"] = "total_diff_limit"
                omissions.append({"path": item.path, "reason": "total_diff_limit"})
        elif omission is not None:
            omissions.append({"path": item.path, "reason": omission})
        files.append(summary)

    generated_at = _utc_now()
    artifact = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review_id": uuid.uuid4().hex[:16],
        "generated_at": generated_at,
        "job": {
            "job_id": job.job_id,
            "goal": job.goal,
            "branch": job.branch,
            "worktree_path": str(worktree),
        },
        "base_commit": job.base_commit,
        "current_head": current_head,
        "no_changes": not changed,
        "changed_file_count": len(changed),
        "changed_files": [item.path for item in changed],
        "files": files,
        "diffstat": {
            "files_changed": len(changed),
            "additions": additions,
            "deletions": deletions,
            "files_without_line_stats": unavailable_stats,
        },
        "diff_snippets": snippets,
        "files_omitted_from_diff_snippets": omissions,
        "artifact_references": _artifact_references(job),
        "human_review_checklist": [
            "Confirm every changed file and omission aligns with the approved goal.",
            "Inspect omitted sensitive, binary, generated, or oversized files safely.",
            "Review additions and deletions for unintended behavior or data exposure.",
            "Run appropriate checks separately before any integration decision.",
            "Keep merge, push, PR creation, and auto-merge as separate explicit actions.",
        ],
        "execution_statement": _EXECUTION_STATEMENT,
        "source_mutation_performed": False,
    }
    encoded = _encoded_artifact(artifact)
    if len(encoded) > MAX_REVIEW_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError("Review artifact exceeds the total size limit.")
    return artifact


def _review_changed_file(
    item: _ChangedPath, worktree: Path, head: str
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    summary: dict[str, Any] = {
        "path": item.path,
        "status": item.status,
        "git_status": item.git_status,
        "additions": None,
        "deletions": None,
        "diff_snippet_included": False,
        "diff_snippet_omission_reason": None,
    }
    path_reason = _path_omission_reason(item.path)
    if path_reason is not None:
        summary["diff_snippet_omission_reason"] = path_reason
        return summary, None, path_reason

    current: bytes | None = None
    base: bytes | None = None
    if item.status != "deleted":
        current, reason = _read_current_file(worktree, item.path)
        if reason is not None:
            summary["diff_snippet_omission_reason"] = reason
            return summary, None, reason
    if item.status != "added":
        base, reason = _read_head_blob(worktree, head, item.path)
        if reason is not None:
            summary["diff_snippet_omission_reason"] = reason
            return summary, None, reason

    before = base or b""
    after = current or b""
    reason = _content_omission_reason(before, after)
    if reason is not None:
        summary["diff_snippet_omission_reason"] = reason
        return summary, None, reason
    before_text = before.decode("utf-8")
    after_text = after.decode("utf-8")
    additions, deletions = _line_diff_counts(before_text, after_text)
    summary["additions"] = additions
    summary["deletions"] = deletions
    snippet_text, truncated = _unified_diff(item.path, item.status, before_text, after_text)
    summary["diff_snippet_included"] = True
    snippet = {
        "path": item.path,
        "status": item.status,
        "truncated": truncated,
        "unified_diff": snippet_text,
    }
    return summary, snippet, None


def _path_omission_reason(path: str) -> str | None:
    posix = PurePosixPath(path)
    parts = tuple(part.casefold() for part in posix.parts)
    name = parts[-1]
    if ".git" in parts:
        return "git_metadata"
    if DEV_STATE_DIR_NAME.casefold() in parts:
        return "dev_state"
    if (
        name in BLOCKED_FILE_NAMES
        or name == ".env"
        or name.startswith(".env.")
        or any(SECRET_NAME_PATTERN.search(part) for part in posix.parts)
    ):
        return "secret_or_env"
    if posix.suffix.casefold() in BLOCKED_SUFFIXES or "data" in parts:
        return "database_or_runtime_data"
    if any(part in _GENERATED_OR_CACHE_PARTS for part in parts):
        return "generated_or_cache"
    if any(part in BLOCKED_PATH_PARTS for part in parts):
        return "blocked_runtime_path"
    if posix.suffix.casefold() in _BINARY_SUFFIXES:
        return "binary"
    return None


def _read_current_file(worktree: Path, path: str) -> tuple[bytes | None, str | None]:
    target = worktree.joinpath(*PurePosixPath(path).parts)
    current = worktree
    for part in PurePosixPath(path).parts:
        current = current / part
        if current.is_symlink():
            return None, "symlink"
    try:
        target.resolve(strict=True).relative_to(worktree.resolve(strict=True))
        if not target.is_file():
            return None, "unsupported_file_type"
        if target.stat().st_size > MAX_REVIEW_TEXT_FILE_BYTES:
            return None, "oversized"
        return target.read_bytes(), None
    except (OSError, ValueError):
        return None, "unreadable_or_outside_worktree"


def _read_head_blob(
    worktree: Path, head: str, path: str
) -> tuple[bytes | None, str | None]:
    expression = f"{head}:{path}".encode("utf-8") + b"\n"
    checked = _run_review_git(
        [*_REVIEW_GIT_PREFIX, "cat-file", "--batch-check"],
        worktree,
        input_bytes=expression,
    )
    if checked.returncode != 0:
        return None, "unreadable_base_blob"
    if checked.stdout.endswith(b" missing\n"):
        return None, "missing_base_blob"
    fields = checked.stdout.strip().split(b" ")
    try:
        object_id = fields[0].decode("ascii")
        object_type = fields[1].decode("ascii")
        size = int(fields[2])
    except (IndexError, UnicodeDecodeError, ValueError):
        return None, "malformed_base_blob"
    if (
        len(fields) != 3
        or not _is_commit_id(object_id)
        or object_type != "blob"
        or size < 0
    ):
        return None, "malformed_base_blob"
    if size > MAX_REVIEW_TEXT_FILE_BYTES:
        return None, "oversized"
    read = _run_review_git(
        [*_REVIEW_GIT_PREFIX, "cat-file", "--batch"],
        worktree,
        input_bytes=expression,
    )
    if read.returncode != 0:
        return None, "unreadable_base_blob"
    header_end = read.stdout.find(b"\n")
    content_end = header_end + 1 + size
    if (
        header_end < 0
        or content_end >= len(read.stdout)
        or read.stdout[content_end : content_end + 1] != b"\n"
        or content_end + 1 != len(read.stdout)
    ):
        return None, "malformed_base_blob"
    return read.stdout[header_end + 1 : content_end], None


def _content_omission_reason(before: bytes, after: bytes) -> str | None:
    for content in (before, after):
        if b"\x00" in content:
            return "binary"
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            return "binary"
        if any(pattern.search(text) for pattern in _SENSITIVE_CONTENT_PATTERNS):
            return "sensitive_content"
    return None


def _line_diff_counts(before: str, after: str) -> tuple[int, int]:
    matcher = difflib.SequenceMatcher(
        a=before.splitlines(), b=after.splitlines(), autojunk=False
    )
    additions = 0
    deletions = 0
    for tag, before_start, before_end, after_start, after_end in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            deletions += before_end - before_start
        if tag in {"replace", "insert"}:
            additions += after_end - after_start
    return additions, deletions


def _unified_diff(
    path: str, status: str, before: str, after: str
) -> tuple[str, bool]:
    fromfile = "/dev/null" if status == "added" else f"a/{path}"
    tofile = "/dev/null" if status == "deleted" else f"b/{path}"
    text = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
            n=3,
        )
    )
    encoded = text.encode("utf-8")
    if len(encoded) <= MAX_REVIEW_DIFF_SNIPPET_BYTES:
        return text, False
    marker = "\n[diff snippet truncated]\n"
    budget = MAX_REVIEW_DIFF_SNIPPET_BYTES - len(marker.encode("utf-8"))
    truncated = encoded[:budget].decode("utf-8", errors="ignore")
    return truncated + marker, True


def _artifact_references(job: FreestyleJob) -> dict[str, str | None]:
    return {
        "proposal_artifact_path": _safe_artifact_reference(job.proposal_artifact_path),
        "edit_generation_artifact_path": _safe_artifact_reference(
            job.edit_generation_artifact_path
        ),
        "generated_edit_bundle_path": _safe_artifact_reference(job.edit_bundle_path),
        "applied_edit_artifact_path": _safe_artifact_reference(job.edit_artifact_path),
    }


def _safe_artifact_reference(value: Any) -> str | None:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > MAX_REVIEW_REFERENCE_CHARS
        or "\x00" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
    ):
        return None
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        return None
    return path.as_posix()


def _write_review_artifact(
    artifact: dict[str, Any], state: Path, job_id: str
) -> str:
    reviews_dir = state / "reviews"
    job_dir = reviews_dir / job_id
    for directory in (reviews_dir, job_dir):
        if directory.exists() and (directory.is_symlink() or not directory.is_dir()):
            raise FreestyleJobRefusedError("Freestyle review artifact path is unsafe.")
        directory.mkdir(parents=True, exist_ok=True)
        try:
            directory.resolve(strict=True).relative_to(state)
        except (OSError, ValueError) as exc:
            raise FreestyleJobRefusedError(
                "Freestyle review artifact path must stay inside local dev state."
            ) from exc
    name = f"review-{artifact['review_id']}.json"
    path = job_dir / name
    temporary = job_dir / f"{name}.tmp"
    for candidate in (path, temporary):
        if candidate.exists() or candidate.is_symlink():
            raise FreestyleJobRefusedError("Freestyle review artifact path is unsafe.")
    encoded = _encoded_artifact(artifact)
    if len(encoded) > MAX_REVIEW_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError("Review artifact exceeds the total size limit.")
    temporary.write_bytes(encoded)
    temporary.replace(path)
    return path.relative_to(state).as_posix()


def _encoded_artifact(artifact: dict[str, Any]) -> bytes:
    return (json.dumps(artifact, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _decode_short_git_text(result: _GitResult, fallback: str) -> str:
    if result.returncode != 0 or len(result.stdout) > 1_000:
        raise FreestyleJobRefusedError(_bounded_error(result.stderr, fallback))
    try:
        text = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise FreestyleJobRefusedError(fallback) from exc
    if not text or any(ord(character) < 32 for character in text):
        raise FreestyleJobRefusedError(fallback)
    return text


def _run_review_git(
    args: list[str], worktree: Path, *, input_bytes: bytes | None = None
) -> _GitResult:
    command = tuple(args)
    if not _is_allowed_review_git_command(command, input_bytes):
        raise FreestyleJobRefusedError("Internal error: forbidden review git operation.")
    environment = {
        # Avoid user config, external prompts, and optional index refresh writes.
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LC_ALL": "C",
        "PATH": os.defpath,
    }
    try:
        completed = subprocess.run(
            args,
            cwd=str(worktree),
            env=environment,
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            check=False,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return _GitResult(command, 124, b"", "Review Git operation timed out.")
    except OSError:
        return _GitResult(command, 1, b"", "Review Git operation could not start.")
    return _GitResult(
        args=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _is_allowed_review_git_command(
    args: tuple[str, ...], input_bytes: bytes | None
) -> bool:
    if args[: len(_REVIEW_GIT_PREFIX)] != _REVIEW_GIT_PREFIX:
        return False
    suffix = args[len(_REVIEW_GIT_PREFIX) :]
    if suffix in {
        ("rev-parse", "HEAD"),
        ("rev-parse", "--abbrev-ref", "HEAD"),
        _STATUS_SUFFIX,
        _IGNORED_SUFFIX,
    }:
        return input_bytes is None
    if suffix not in {("cat-file", "--batch-check"), ("cat-file", "--batch")}:
        return False
    if (
        not isinstance(input_bytes, bytes)
        or len(input_bytes) > MAX_REVIEW_PATH_CHARS + 42
        or input_bytes.count(b"\n") != 1
        or not input_bytes.endswith(b"\n")
        or b"\x00" in input_bytes
    ):
        return False
    try:
        expression = input_bytes[:-1].decode("utf-8")
        commit, path = expression.split(":", 1)
    except (UnicodeDecodeError, ValueError):
        return False
    return _is_commit_id(commit) and _validated_review_path_or_false(path)


def _validated_review_path_or_false(path: str) -> bool:
    try:
        _validated_review_path(path)
    except FreestyleJobRefusedError:
        return False
    return True
