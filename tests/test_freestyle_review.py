"""Tests for deterministic, read-only freestyle review artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_review as review
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    get_freestyle_job,
    materialize_freestyle_job,
    review_freestyle_job,
)


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    )
    return completed.stdout.decode().strip()


def _init_repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / ".gitignore").write_text(
        ".env\n.lewis-dev/\n__pycache__/\n*.db\n", encoding="utf-8"
    )
    (root / "delete.txt").write_text("delete me\n", encoding="utf-8")
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _materialized_job(root: Path):
    created = create_freestyle_job(
        "Review isolated freestyle edits.",
        developer_approved=True,
        project_root=root,
    )
    return materialize_freestyle_job(created.job_id, project_root=root)


def _metadata_path(root: Path, job_id: str) -> Path:
    return root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"


def _set_metadata(root: Path, job_id: str, **values) -> None:
    path = _metadata_path(root, job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _artifact(root: Path, job) -> dict:
    assert job.review_artifact_path
    path = root / freestyle.DEV_STATE_DIR_NAME / job.review_artifact_path
    return json.loads(path.read_text(encoding="utf-8"))


def _source_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.relative_to(root).parts
        and freestyle.DEV_STATE_DIR_NAME not in path.relative_to(root).parts
    }


@pytest.mark.parametrize("approval", [None, False, "yes"])
def test_review_requires_explicit_approval_and_persists_bounded_error(
    tmp_path, approval
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(root, job.job_id, developer_approved=approval)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        review_freestyle_job(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.review_status == "error"
    assert failed.review_started_at
    assert failed.review_completed_at
    assert failed.review_artifact_path is None
    assert len(failed.review_errors[0]) <= 500
    assert "approval" in failed.review_errors[0]


def test_review_requires_successfully_materialized_job(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Review only after materialization.",
        developer_approved=True,
        project_root=root,
    )

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        review_freestyle_job(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.review_status == "error"
    assert "successfully materialized" in failed.review_errors[0]


def test_no_changes_produces_valid_review_artifact_and_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(
        root,
        job.job_id,
        proposal_artifact_path=f"proposals/{job.job_id}/proposal.json",
        edit_generation_artifact_path=(
            f"edit-bundles/{job.job_id}/generation-abc.json"
        ),
        edit_bundle_path=f"edit-bundles/{job.job_id}/bundle-abc.json",
        edit_artifact_path=f"edits/{job.job_id}/apply-abc.json",
    )

    completed = review_freestyle_job(job.job_id, project_root=root)
    artifact = _artifact(root, completed)

    assert completed.review_status == "reviewed"
    assert completed.review_changed_file_count == 0
    assert completed.review_errors == ()
    assert artifact["job"]["job_id"] == job.job_id
    assert artifact["job"]["goal"] == job.goal
    assert artifact["job"]["branch"] == job.branch
    assert artifact["job"]["worktree_path"] == job.worktree_path
    assert artifact["base_commit"] == job.base_commit
    assert artifact["current_head"] == {
        "branch": job.branch,
        "commit": job.base_commit,
    }
    assert artifact["no_changes"] is True
    assert artifact["changed_files"] == []
    assert artifact["diffstat"] == {
        "additions": 0,
        "deletions": 0,
        "files_changed": 0,
        "files_without_line_stats": 0,
    }
    assert artifact["artifact_references"]["proposal_artifact_path"]
    assert artifact["artifact_references"]["edit_generation_artifact_path"]
    assert artifact["artifact_references"]["generated_edit_bundle_path"]
    assert artifact["artifact_references"]["applied_edit_artifact_path"]
    assert artifact["source_mutation_performed"] is False
    statement = artifact["execution_statement"]
    for forbidden_action in ("merge", "push", "PR creation", "auto-merge"):
        assert forbidden_action in statement


def test_changed_created_deleted_files_are_summarized_from_worktree_only(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    (root / "tracked.txt").write_text("dirty main marker\n", encoding="utf-8")
    main_before = _source_snapshot(root)
    (worktree / "tracked.txt").write_text("isolated marker\n", encoding="utf-8")
    (worktree / "created.txt").write_text("new file\n", encoding="utf-8")
    (worktree / "delete.txt").unlink()
    worktree_before = _source_snapshot(worktree)

    completed = review_freestyle_job(job.job_id, project_root=root)
    artifact = _artifact(root, completed)

    by_path = {item["path"]: item for item in artifact["files"]}
    assert {path: item["status"] for path, item in by_path.items()} == {
        "created.txt": "added",
        "delete.txt": "deleted",
        "tracked.txt": "modified",
    }
    assert by_path["created.txt"]["additions"] == 1
    assert by_path["created.txt"]["deletions"] == 0
    assert by_path["delete.txt"]["additions"] == 0
    assert by_path["delete.txt"]["deletions"] == 1
    assert by_path["tracked.txt"]["additions"] == 1
    assert by_path["tracked.txt"]["deletions"] == 1
    snippets = "\n".join(item["unified_diff"] for item in artifact["diff_snippets"])
    assert "isolated marker" in snippets
    assert "dirty main marker" not in snippets
    assert _source_snapshot(root) == main_before
    assert _source_snapshot(worktree) == worktree_before
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_diff_snippets_and_total_artifact_are_bounded(tmp_path):
    root = _init_repo(tmp_path)
    (root / "large.txt").write_text("old\n" * 8_000, encoding="utf-8")
    _git(root, "add", "large.txt")
    _git(root, "commit", "-m", "add large text")
    job = _materialized_job(root)
    (Path(job.worktree_path) / "large.txt").write_text(
        "new\n" * 8_000, encoding="utf-8"
    )

    completed = review_freestyle_job(job.job_id, project_root=root)
    artifact_path = root / freestyle.DEV_STATE_DIR_NAME / completed.review_artifact_path
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    snippet = artifact["diff_snippets"][0]

    assert snippet["truncated"] is True
    assert len(snippet["unified_diff"].encode("utf-8")) <= (
        review.MAX_REVIEW_DIFF_SNIPPET_BYTES
    )
    assert artifact_path.stat().st_size <= review.MAX_REVIEW_ARTIFACT_BYTES


def test_sensitive_generated_binary_and_oversized_files_are_omitted(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)
    files = {
        ".env": b"ENV_SECRET_MARKER=bad\n",
        ".lewis-dev/state.json": b"DEV_STATE_MARKER\n",
        "__pycache__/cache.py": b"CACHE_MARKER\n",
        "cache/output.txt": b"PLAIN_CACHE_MARKER\n",
        "data/runtime.db": b"DB_MARKER\n",
        "generated/output.txt": b"GENERATED_MARKER\n",
        "secret-token.txt": b"SECRET_MARKER\n",
        "binary.bin": b"\x00BINARY_MARKER",
        "oversized.txt": b"x" * (review.MAX_REVIEW_TEXT_FILE_BYTES + 1),
        "sensitive.txt": b'password = "this-is-a-secret-value"\n',
    }
    for relative, content in files.items():
        target = worktree / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    completed = review_freestyle_job(job.job_id, project_root=root)
    artifact = _artifact(root, completed)
    omissions = {
        item["path"]: item["reason"]
        for item in artifact["files_omitted_from_diff_snippets"]
    }

    assert omissions == {
        ".env": "secret_or_env",
        ".lewis-dev/state.json": "dev_state",
        "__pycache__/cache.py": "generated_or_cache",
        "binary.bin": "binary",
        "cache/output.txt": "generated_or_cache",
        "data/runtime.db": "database_or_runtime_data",
        "generated/output.txt": "generated_or_cache",
        "oversized.txt": "oversized",
        "secret-token.txt": "secret_or_env",
        "sensitive.txt": "sensitive_content",
    }
    encoded = json.dumps(artifact)
    for marker in (
        "ENV_SECRET_MARKER",
        "DEV_STATE_MARKER",
        "CACHE_MARKER",
        "PLAIN_CACHE_MARKER",
        "DB_MARKER",
        "GENERATED_MARKER",
        "SECRET_MARKER",
        "BINARY_MARKER",
        "this-is-a-secret-value",
    ):
        assert marker not in encoded
    assert review._path_omission_reason("nested/.git/config") == "git_metadata"


@pytest.mark.parametrize("edit_status", ["applying", "inconsistent"])
def test_in_progress_or_inconsistent_edit_state_fails_closed(tmp_path, edit_status):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(root, job.job_id, edit_apply_status=edit_status)

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        review_freestyle_job(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.review_status == "error"
    assert failed.review_artifact_path is None
    assert "Recover or verify" in failed.review_errors[0]


def test_unsafe_recorded_worktree_path_fails_and_persists_error(tmp_path):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    _set_metadata(root, job.job_id, worktree_path=str(root))

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        review_freestyle_job(job.job_id, project_root=root)

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.review_status == "error"
    assert "worktree path does not match" in failed.review_errors[0]


def test_review_uses_only_constrained_readonly_git_and_no_model_or_shell(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    (Path(job.worktree_path) / "tracked.txt").write_text("review me\n", encoding="utf-8")
    calls: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        calls.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(freestyle.subprocess, "run", recording_run)
    monkeypatch.setattr(review.subprocess, "run", recording_run)

    completed = review_freestyle_job(job.job_id, project_root=root)

    assert completed.review_status == "reviewed"
    assert calls
    assert all(call[0][0] == "git" for call in calls)
    assert all(call[1]["shell"] is False for call in calls)
    assert all(Path(call[1]["cwd"]) == Path(job.worktree_path) for call in calls)
    forbidden = {
        "apply",
        "checkout",
        "commit",
        "fetch",
        "merge",
        "pull",
        "push",
        "remote",
        "reset",
        "worktree",
    }
    assert all(forbidden.isdisjoint(call[0]) for call in calls)
    review_source = Path(review.__file__).read_text(encoding="utf-8")
    for dependency in (
        "deepseek_client",
        "ApprovalStore",
        "sqlite",
        "load_dotenv",
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web",
        "requests",
        "httpx",
    ):
        assert dependency not in review_source


@pytest.mark.parametrize(
    "command",
    [
        ["git", "push"],
        ["git", "checkout", "main"],
        ["git", "status"],
        [*review._REVIEW_GIT_PREFIX, "diff"],
        [*review._REVIEW_GIT_PREFIX, "remote", "-v"],
    ],
)
def test_forbidden_review_git_operations_remain_forbidden(tmp_path, command):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)

    with pytest.raises(FreestyleJobRefusedError, match="forbidden review git"):
        review._run_review_git(command, Path(job.worktree_path))


def test_review_cli_dispatches_only_the_requested_job_id(tmp_path, monkeypatch, capsys):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    calls: list[str] = []

    def fake_review(job_id):
        calls.append(job_id)
        return job

    monkeypatch.setattr(freestyle, "review_freestyle_job", fake_review)
    exit_code = freestyle.main(["review", job.job_id])

    assert exit_code == 0
    assert calls == [job.job_id]
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id
