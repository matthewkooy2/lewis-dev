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
import subprocess
from typing import Any, Sequence
import uuid

from app.code_proposal_path_safety import (
    BLOCKED_FILE_NAMES,
    BLOCKED_PATH_PARTS,
    BLOCKED_SUFFIXES,
    SECRET_NAME_PATTERN,
)
from app.config import ROOT_DIR
from app.dev.freestyle_artifacts import (
    encode_json_artifact,
    safe_artifact_reference,
    write_artifact_bytes,
)
from app.dev.freestyle_coding import (
    DEV_STATE_DIR_NAME,
    FreestyleJob,
    FreestyleJobRefusedError,
    _bounded_error,
    _is_commit_id,
    _require_explicit_developer_approval,
    _utc_now,
    _validated_job_worktree,
    _write_job,
    get_freestyle_job,
)
from app.dev.freestyle_policy import (
    GENERATED_OR_CACHE_PARTS,
    contains_sensitive_content,
    recovery_message,
)
from app.dev.freestyle_state import (
    require_materialized,
    require_no_active_edit_application,
    resolved_roots,
)
from app.dev.freestyle_ui_review import (
    KIND_HTML,
    KIND_JS,
    KIND_PYTHON_MARKUP,
    ReferencedJsAsset,
    UiSource,
    classify_web_source,
    is_static_asset_path,
    referenced_static_js_srcs,
    review_ui,
    static_js_reference_candidates,
)

REVIEW_SCHEMA_VERSION = 1
MAX_REVIEW_CHANGED_FILES = 500
MAX_REVIEW_PATH_CHARS = 500
MAX_REVIEW_TEXT_FILE_BYTES = 64 * 1024
MAX_REVIEW_DIFF_SNIPPET_BYTES = 12 * 1024
MAX_REVIEW_TOTAL_DIFF_BYTES = 96 * 1024
MAX_REVIEW_ARTIFACT_BYTES = 500 * 1024
_GIT_TIMEOUT_SECONDS = 30

# Bounds for the advisory web-UI review pass.
MAX_UI_REVIEW_SOURCES = 40
MAX_UI_STATIC_ASSETS = 300
MAX_UI_STATIC_SCAN_ENTRIES = 4000
_UI_ASSET_SUFFIXES = frozenset({".js", ".mjs", ".css"})
_UI_REVIEW_STATEMENT = (
    "Web-UI findings are advisory: they never block edit-bundle validation, edit "
    "application, review, or approval, and no source file was modified."
)
_REPO_MAP_REVIEW_STATEMENT = (
    "Repo-map findings are advisory: they reflect the persistent local repo map "
    "and never block edit-bundle validation, edit application, review, or "
    "approval. The repo map never bypasses path safety, approval gates, the "
    "static check allowlist, or human merge authority."
)

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
# The review slice also omits plain "cache" directories, beyond the shared
# generated-or-cache exclusion the proposal and edit slices use.
_GENERATED_OR_CACHE_PARTS = GENERATED_OR_CACHE_PARTS | {"cache"}
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
_CONFLICT_STATUSES = {"DD", "AU", "UD", "UA", "DU", "AA", "UU"}
_RECOVERY_MESSAGE = recovery_message("retrying review")
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
    root, state = resolved_roots(project_root or ROOT_DIR, state_root)
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
        review_ui_warnings=(),
        review_repo_map_warnings=(),
        review_plan_warnings=(),
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
        artifact = _build_review_artifact(
            started,
            worktree,
            current_head,
            changed,
            render_local=_is_host_repo(root),
            root=root,
            state=state,
        )
        artifact_path = _write_review_artifact(artifact, state, started.job_id)
    except Exception as exc:
        failed = replace(
            started,
            review_status="error",
            review_completed_at=_utc_now(),
            review_changed_file_count=changed_count,
            review_ui_warnings=(),
            review_repo_map_warnings=(),
            review_plan_warnings=(),
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
        review_ui_warnings=_artifact_ui_warnings(artifact),
        review_repo_map_warnings=_artifact_repo_map_warnings(artifact),
        review_plan_warnings=_artifact_plan_warnings(artifact),
        review_errors=(),
        updated_at=completed_at,
    )
    _write_job(completed, state)
    return completed


def _require_review_prerequisites(job: FreestyleJob) -> None:
    _require_explicit_developer_approval(job, "review")
    require_materialized(job, "must be successfully materialized before review.")
    require_no_active_edit_application(job, f". {_RECOVERY_MESSAGE}")


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
        review_ui_warnings=(),
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
    *,
    render_local: bool = False,
    root: Path | None = None,
    state: Path | None = None,
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
        "ui_review": _web_ui_review(
            worktree, current_head["commit"], changed, render_local=render_local
        ),
        "repo_map_review": _repo_map_review(worktree, changed, root, state),
        "implementation_plan_review": _implementation_plan_review(
            job, changed, state
        ),
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
    encoded = encode_json_artifact(artifact)
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
        if contains_sensitive_content(text):
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
        "proposal_artifact_path": safe_artifact_reference(job.proposal_artifact_path),
        "edit_generation_artifact_path": safe_artifact_reference(
            job.edit_generation_artifact_path
        ),
        "generated_edit_bundle_path": safe_artifact_reference(job.edit_bundle_path),
        "applied_edit_artifact_path": safe_artifact_reference(job.edit_artifact_path),
    }


def _is_host_repo(root: Path) -> bool:
    """Whether ``root`` is the live application checkout.

    Only then does rendering the in-process app correspond to the repository
    under review, so route rendering is enabled there and skipped for isolated
    test repositories.
    """
    try:
        return root.resolve() == ROOT_DIR.resolve()
    except (OSError, ValueError):
        return False


def _artifact_ui_warnings(artifact: dict[str, Any]) -> tuple[str, ...]:
    ui_review = artifact.get("ui_review")
    if not isinstance(ui_review, dict):
        return ()
    warnings = ui_review.get("warnings")
    if not isinstance(warnings, list):
        return ()
    return tuple(
        str(warning)[:500] for warning in warnings if isinstance(warning, str) and warning
    )


def _artifact_repo_map_warnings(artifact: dict[str, Any]) -> tuple[str, ...]:
    section = artifact.get("repo_map_review")
    if not isinstance(section, dict):
        return ()
    warnings = section.get("warnings")
    if not isinstance(warnings, list):
        return ()
    return tuple(
        str(warning)[:500]
        for warning in warnings
        if isinstance(warning, str) and warning
    )


def _artifact_plan_warnings(artifact: dict[str, Any]) -> tuple[str, ...]:
    section = artifact.get("implementation_plan_review")
    if not isinstance(section, dict):
        return ()
    warnings = section.get("warnings")
    if not isinstance(warnings, list):
        return ()
    return tuple(
        str(warning)[:500]
        for warning in warnings
        if isinstance(warning, str) and warning
    )


_IMPLEMENTATION_PLAN_REVIEW_STATEMENT = (
    "Implementation-plan findings are advisory: they compare the applied change "
    "against the deterministic plan built before edit generation and never block "
    "edit-bundle validation, edit application, review, or approval."
)


def _implementation_plan_review(
    job: FreestyleJob,
    changed: tuple[_ChangedPath, ...],
    state: Path | None,
) -> dict[str, Any]:
    """Build the advisory implementation-plan review section, fail-open.

    Loads the plan persisted before edit generation and flags whether the
    applied change contradicts it: a CSS-only change where the plan called for
    DOM/JS/CSS wiring, a file the plan said not to touch, or a plan whose
    expected surfaces were all skipped. Advisory only; never blocks review.
    """
    base: dict[str, Any] = {
        "warnings": [],
        "advisory": True,
        "statement": _IMPLEMENTATION_PLAN_REVIEW_STATEMENT,
        "plan_available": False,
        "plan_summary": {},
    }
    plan_artifact = _read_plan_artifact(job, state)
    if plan_artifact is None:
        return base
    try:
        from app.dev.freestyle_implementation_plan import (
            plan_deviation_warnings,
            summarize_implementation_plan,
        )

        changed_paths = [item.path for item in changed]
        base["plan_available"] = True
        base["plan_summary"] = summarize_implementation_plan(plan_artifact)
        base["warnings"] = list(
            plan_deviation_warnings(plan_artifact, changed_paths)
        )
    except Exception:
        return base
    return base


def _read_plan_artifact(
    job: FreestyleJob, state: Path | None
) -> dict[str, Any] | None:
    """Read the full first-class implementation-plan artifact, fail-open."""
    if state is None:
        return None
    relative = safe_artifact_reference(job.implementation_plan_artifact_path)
    if relative is None:
        return None
    path = state / relative
    try:
        if path.is_symlink() or not path.is_file():
            return None
        path.resolve(strict=True).relative_to(state.resolve())
        if path.stat().st_size > MAX_REVIEW_ARTIFACT_BYTES:
            return None
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _repo_map_review(
    worktree: Path,
    changed: tuple[_ChangedPath, ...],
    root: Path | None,
    state: Path | None,
) -> dict[str, Any]:
    """Build the advisory repo-map review section, fail-open to empty findings.

    Resolves the persistent repo map for this job (worktree-preferred, host
    fallback marked stale), then reports advisory warnings about implementation
    surfaces the changed files appear to ignore (related CSS/JS/tests, missing
    assets) and the connected files the map links to each change. Never mutates
    the worktree and never blocks the review.
    """
    base = {
        "warnings": [],
        "advisory": True,
        "statement": _REPO_MAP_REVIEW_STATEMENT,
        "provenance": {"available": False, "stale": True, "fallback": True},
        "connected_files": {},
    }
    if root is None or state is None:
        return base
    try:
        from app.dev.repo_map_context import (
            connected_files_for_paths,
            repo_map_review_warnings,
            resolve_freestyle_index,
        )

        index, provenance = resolve_freestyle_index(worktree, root, state)
        base["provenance"] = provenance
        if index is None:
            return base
        changed_paths = [item.path for item in changed]
        base["warnings"] = repo_map_review_warnings(index, changed_paths)
        base["connected_files"] = connected_files_for_paths(index, changed_paths)
    except Exception:
        return base
    return base


def _web_ui_review(
    worktree: Path,
    head: str,
    changed: tuple[_ChangedPath, ...],
    *,
    render_local: bool,
) -> dict[str, Any]:
    """Build the advisory web-UI review section for the artifact.

    Reads only the changed web files (HTML, page-builder Python, CSS, static
    JS) plus a bounded listing of static assets, optionally enriched with HTML
    from locally rendered routes. Never mutates the worktree, never blocks the
    review, and degrades to empty findings on any rendering failure.
    """
    sources = _collect_ui_sources(worktree, head, changed)
    if not sources:
        return {"warnings": [], "summary": {"web_source_count": 0}, "advisory": True}
    present_assets = _scan_static_asset_paths(worktree, changed)
    rendered_pages = _safe_render_local_pages() if render_local else ()
    referenced_js_assets = _collect_referenced_js_assets(
        worktree, sources, rendered_pages
    )
    result = review_ui(
        sources,
        rendered_pages=rendered_pages,
        present_static_assets=frozenset(present_assets),
        referenced_js_assets=referenced_js_assets,
    )
    return {
        "warnings": list(result.warnings),
        "summary": result.summary,
        "rendered_routes": [page.route for page in rendered_pages],
        "advisory": True,
        "statement": _UI_REVIEW_STATEMENT,
    }


def _collect_ui_sources(
    worktree: Path, head: str, changed: tuple[_ChangedPath, ...]
) -> list[UiSource]:
    sources: list[UiSource] = []
    for item in changed:
        if len(sources) >= MAX_UI_REVIEW_SOURCES:
            break
        if item.status == "deleted" or _path_omission_reason(item.path) is not None:
            continue
        current, reason = _read_current_file(worktree, item.path)
        if current is None or reason is not None:
            continue
        try:
            text = current.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if contains_sensitive_content(text):
            continue
        kind = classify_web_source(item.path, text)
        if kind is None:
            continue
        sources.append(
            UiSource(
                path=item.path,
                kind=kind,
                text=text,
                status=item.status,
                added_text=_introduced_text(worktree, head, item, text),
            )
        )
    return sources


def _collect_referenced_js_assets(
    worktree: Path,
    sources: Sequence[UiSource],
    rendered_pages: Sequence[Any],
) -> tuple[ReferencedJsAsset, ...]:
    """Read same-app static JS assets referenced by the rendered/markup HTML."""
    markup_texts = [
        source.text
        for source in sources
        if source.kind in (KIND_HTML, KIND_PYTHON_MARKUP)
    ] + [str(getattr(page, "html", "")) for page in rendered_pages]
    source_by_path = {
        source.path: source
        for source in sources
        if source.kind == KIND_JS and source.status != "deleted"
    }
    assets: list[ReferencedJsAsset] = []
    seen: set[str] = set()
    for src in referenced_static_js_srcs(markup_texts):
        for candidate in static_js_reference_candidates(src):
            if candidate in seen:
                continue
            source = source_by_path.get(candidate)
            if source is not None:
                assets.append(
                    ReferencedJsAsset(path=source.path, src=src, text=source.text)
                )
                seen.add(candidate)
                break
            current, reason = _read_current_file(worktree, candidate)
            if current is None or reason is not None:
                continue
            try:
                text = current.decode("utf-8")
            except UnicodeDecodeError:
                continue
            if contains_sensitive_content(text):
                continue
            assets.append(ReferencedJsAsset(path=candidate, src=src, text=text))
            seen.add(candidate)
            break
    return tuple(assets)


def _introduced_text(
    worktree: Path, head: str, item: _ChangedPath, text: str
) -> str | None:
    """The lines a modification added, so warnings focus on this change.

    Returns ``None`` (treat the whole file as introduced) for additions or when
    the base blob is unavailable.
    """
    if item.status == "added":
        return None
    base, reason = _read_head_blob(worktree, head, item.path)
    if reason is not None or base is None:
        return None
    try:
        return _added_lines(base.decode("utf-8"), text)
    except UnicodeDecodeError:
        return None


def _added_lines(before: str, after: str) -> str:
    after_lines = after.splitlines()
    matcher = difflib.SequenceMatcher(
        a=before.splitlines(), b=after_lines, autojunk=False
    )
    added: list[str] = []
    for tag, _before_start, _before_end, after_start, after_end in matcher.get_opcodes():
        if tag in {"insert", "replace"}:
            added.extend(after_lines[after_start:after_end])
    return "\n".join(added)


def _scan_static_asset_paths(
    worktree: Path, changed: tuple[_ChangedPath, ...]
) -> set[str]:
    """Bounded, symlink-safe listing of static JS/CSS assets in the worktree."""
    assets: set[str] = {
        item.path
        for item in changed
        if item.status != "deleted"
        and _is_ui_asset_path(item.path)
        and is_static_asset_path(item.path)
    }
    try:
        worktree_resolved = worktree.resolve(strict=True)
    except (OSError, ValueError):
        return assets
    visited = 0
    for root_dir in _candidate_static_roots(worktree, worktree_resolved, changed):
        for dirpath, dirnames, filenames in os.walk(root_dir, followlinks=False):
            dirnames[:] = [
                name
                for name in dirnames
                if not (Path(dirpath) / name).is_symlink()
            ]
            for name in filenames:
                visited += 1
                if (
                    visited > MAX_UI_STATIC_SCAN_ENTRIES
                    or len(assets) >= MAX_UI_STATIC_ASSETS
                ):
                    return assets
                file_path = Path(dirpath) / name
                if file_path.is_symlink() or not _is_ui_asset_path(name):
                    continue
                try:
                    relative = (
                        file_path.resolve(strict=True)
                        .relative_to(worktree_resolved)
                        .as_posix()
                    )
                except (OSError, ValueError):
                    continue
                assets.add(relative)
    return assets


def _candidate_static_roots(
    worktree: Path, worktree_resolved: Path, changed: tuple[_ChangedPath, ...]
) -> list[Path]:
    candidates = [worktree / "app" / "static", worktree / "static"]
    for item in changed:
        parts = PurePosixPath(item.path).parts
        if "static" in parts:
            index = parts.index("static")
            candidates.append(worktree.joinpath(*parts[: index + 1]))
    roots: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except (OSError, ValueError):
            continue
        if candidate.is_symlink() or not candidate.is_dir() or resolved in seen:
            continue
        try:
            resolved.relative_to(worktree_resolved)
        except ValueError:
            continue
        seen.add(resolved)
        roots.append(candidate)
    return roots


def _is_ui_asset_path(path: str) -> bool:
    return PurePosixPath(path).suffix.casefold() in _UI_ASSET_SUFFIXES


def _safe_render_local_pages() -> tuple[Any, ...]:
    """Render the allowlisted local routes, fail-open to no pages."""
    try:
        from app.dev.freestyle_route_render import render_local_pages

        return render_local_pages()
    except Exception:
        return ()


def _write_review_artifact(
    artifact: dict[str, Any], state: Path, job_id: str
) -> str:
    encoded = encode_json_artifact(artifact)
    if len(encoded) > MAX_REVIEW_ARTIFACT_BYTES:
        raise FreestyleJobRefusedError("Review artifact exceeds the total size limit.")
    return write_artifact_bytes(
        state,
        "reviews",
        job_id,
        f"review-{artifact['review_id']}.json",
        encoded,
        label="review",
    )


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
