"""Tests for developer-only freestyle cleanup and stale-lock recovery."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_maintenance as maintenance
from app.dev.freestyle_coding import FreestyleJobRefusedError


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
    (root / ".gitignore").write_text(".lewis-dev/\n", encoding="utf-8")
    (root / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".gitignore", "tracked.txt")
    _git(root, "commit", "-m", "init")
    return root


def _job(root: Path):
    return freestyle.create_freestyle_job(
        "Maintain this local freestyle job.",
        developer_approved=True,
        project_root=root,
    )


def _metadata_path(root: Path, job_id: str) -> Path:
    return root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"


def _set_metadata(root: Path, job_id: str, **updates) -> None:
    path = _metadata_path(root, job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(updates)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _source_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.relative_to(root).parts
        and freestyle.DEV_STATE_DIR_NAME not in path.relative_to(root).parts
    }


def _maintenance_artifacts(root: Path, job_id: str) -> list[Path]:
    directory = root / freestyle.DEV_STATE_DIR_NAME / "maintenance" / job_id
    return sorted(directory.glob("*.json"))


def test_cleanup_status_is_read_only_and_reports_key_cases(tmp_path):
    root = _init_repo(tmp_path)
    clean = _job(root)
    active = _job(root)
    _set_metadata(
        root,
        active.job_id,
        latest_run_status="running",
        updated_at="2000-01-01T00:00:00+00:00",
    )
    materialized_clean = freestyle.materialize_freestyle_job(
        _job(root).job_id, project_root=root
    )
    dirty = freestyle.materialize_freestyle_job(_job(root).job_id, project_root=root)
    (Path(dirty.worktree_path) / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    orphan_id = "f" * 32
    orphan_dir = root / freestyle.DEV_STATE_DIR_NAME / "reviews" / orphan_id
    orphan_dir.mkdir(parents=True)
    (orphan_dir / "review.json").write_text("{}\n", encoding="utf-8")
    state_before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / freestyle.DEV_STATE_DIR_NAME).rglob("*")
        if path.is_file() and not path.is_symlink()
    }

    statuses = maintenance.get_freestyle_cleanup_status(project_root=root)
    by_id = {item["job_id"]: item for item in statuses}

    assert by_id[clean.job_id]["cleanable"] is True
    assert by_id[clean.job_id]["dirty"] is False
    assert by_id[clean.job_id]["worktree_state"] == "not_materialized_worktree"
    assert by_id[clean.job_id]["worktree_not_checked_out"] is True
    assert by_id[clean.job_id]["dirty_reason"] is None
    assert by_id[active.job_id]["active"] is True
    assert by_id[active.job_id]["stale"] is True
    assert by_id[materialized_clean.job_id]["worktree_state"] == "materialized_worktree"
    assert by_id[materialized_clean.job_id]["dirty"] is False
    assert by_id[dirty.job_id]["dirty"] is True
    assert by_id[dirty.job_id]["worktree_state"] == "materialized_worktree"
    assert by_id[dirty.job_id]["dirty_reason"] == "user_reviewable_worktree_changes"
    assert by_id[orphan_id]["orphaned"] is True
    assert by_id[orphan_id]["cleanable"] is False
    state_after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / freestyle.DEV_STATE_DIR_NAME).rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert state_after == state_before


def test_ready_no_checkout_worktree_status_and_cleanup_are_safe(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    worktree = Path(job.worktree_path)

    raw_status = _git(worktree, "status", "--short")
    assert "tracked.txt" in raw_status
    status = next(
        item
        for item in maintenance.get_freestyle_cleanup_status(project_root=root)
        if item["job_id"] == job.job_id
    )

    assert status["worktree_state"] == "not_materialized_worktree"
    assert status["worktree_not_checked_out"] is True
    assert status["dirty"] is False
    assert status["dirty_paths"] == []
    assert "no-checkout worktree" in status["recommended_manual_action"]

    result = maintenance.cleanup_freestyle_job(
        job.job_id, confirm=True, project_root=root
    )

    assert result["cleanup_mode"] == "normal"
    assert result["worktree_removal_mode"] == "forced_not_materialized_worktree"
    assert not worktree.exists()
    assert not _metadata_path(root, job.job_id).exists()
    artifact = json.loads(
        _maintenance_artifacts(root, job.job_id)[-1].read_text(encoding="utf-8")
    )
    assert artifact["cleanup_mode"] == "normal"
    assert artifact["worktree_removal_mode"] == "forced_not_materialized_worktree"
    remove = artifact["git_operations_attempted"][0]["args"]
    assert remove[-3:] == ["remove", "--force", str(worktree)]


def test_cleanup_requires_confirmation_and_refuses_active_state(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)

    with pytest.raises(FreestyleJobRefusedError, match="--confirm"):
        maintenance.cleanup_freestyle_job(job.job_id, project_root=root)

    _set_metadata(root, job.job_id, checks_status="running")
    with pytest.raises(FreestyleJobRefusedError, match="recover"):
        maintenance.cleanup_freestyle_job(job.job_id, confirm=True, project_root=root)

    assert Path(job.worktree_path).exists()
    assert _metadata_path(root, job.job_id).exists()
    artifact = json.loads(
        _maintenance_artifacts(root, job.job_id)[-1].read_text(encoding="utf-8")
    )
    assert artifact["operation"] == "cleanup"
    assert artifact["refusal_reasons"]


def test_cleanup_refuses_dirty_worktree_without_explicit_discard(tmp_path):
    root = _init_repo(tmp_path)
    job = freestyle.materialize_freestyle_job(_job(root).job_id, project_root=root)
    (Path(job.worktree_path) / "tracked.txt").write_text("dirty\n", encoding="utf-8")

    with pytest.raises(FreestyleJobRefusedError, match="discard-worktree-changes"):
        maintenance.cleanup_freestyle_job(job.job_id, confirm=True, project_root=root)

    assert Path(job.worktree_path).exists()
    assert _git(root, "branch", "--list", job.branch).removeprefix("+ ") == job.branch


def test_cleanup_removes_clean_materialized_worktree_normally(tmp_path):
    root = _init_repo(tmp_path)
    job = freestyle.materialize_freestyle_job(_job(root).job_id, project_root=root)

    result = maintenance.cleanup_freestyle_job(
        job.job_id, confirm=True, project_root=root
    )

    assert result["cleanup_mode"] == "normal"
    assert result["worktree_removal_mode"] == "normal"
    artifact = json.loads(
        _maintenance_artifacts(root, job.job_id)[-1].read_text(encoding="utf-8")
    )
    remove = artifact["git_operations_attempted"][0]["args"]
    assert remove[-2:] == ["remove", job.worktree_path]
    assert "--force" not in remove


def test_cleanup_refuses_unexpected_ready_worktree_content_without_discard(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    unexpected = Path(job.worktree_path) / "unexpected.txt"
    unexpected.write_text("review me\n", encoding="utf-8")

    status = next(
        item
        for item in maintenance.get_freestyle_cleanup_status(project_root=root)
        if item["job_id"] == job.job_id
    )
    assert status["worktree_state"] == "not_materialized_worktree"
    assert status["dirty"] is True
    assert status["dirty_paths"] == ["unexpected.txt"]
    assert status["dirty_reason"] == "unexpected_paths_in_not_materialized_worktree"

    with pytest.raises(FreestyleJobRefusedError, match="discard-worktree-changes"):
        maintenance.cleanup_freestyle_job(job.job_id, confirm=True, project_root=root)

    assert unexpected.exists()


def test_cleanup_removes_only_matching_local_job_and_preserves_main(tmp_path):
    root = _init_repo(tmp_path)
    job = freestyle.materialize_freestyle_job(_job(root).job_id, project_root=root)
    other = _job(root)
    (Path(job.worktree_path) / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    artifact_dir = root / freestyle.DEV_STATE_DIR_NAME / "reviews" / job.job_id
    artifact_dir.mkdir(parents=True)
    (artifact_dir / "review.json").write_text("{}\n", encoding="utf-8")
    main_before = _source_snapshot(root)

    result = maintenance.cleanup_freestyle_job(
        job.job_id,
        confirm=True,
        discard_worktree_changes=True,
        project_root=root,
    )

    assert _source_snapshot(root) == main_before
    assert not Path(job.worktree_path).exists()
    assert not _metadata_path(root, job.job_id).exists()
    assert not artifact_dir.exists()
    assert _git(root, "branch", "--list", job.branch) == ""
    assert Path(other.worktree_path).exists()
    assert _metadata_path(root, other.job_id).exists()
    assert _maintenance_artifacts(root, job.job_id)
    assert "No commit, merge, push" in result["integration_statement"]
    assert result["cleanup_mode"] == "forced_explicit_discard"
    assert result["worktree_removal_mode"] == "forced_explicit_discard"
    artifact = json.loads(
        _maintenance_artifacts(root, job.job_id)[-1].read_text(encoding="utf-8")
    )
    remove = artifact["git_operations_attempted"][0]["args"]
    assert remove[-3:] == ["remove", "--force", str(Path(job.worktree_path))]
    assert str(Path(other.worktree_path)) not in remove


def test_cleanup_orphaned_metadata_only_job_records_artifact(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    _git(root, "worktree", "remove", "--force", job.worktree_path)
    _git(root, "branch", "-D", job.branch)

    status = next(
        item
        for item in maintenance.get_freestyle_cleanup_status(project_root=root)
        if item["job_id"] == job.job_id
    )
    assert status["orphaned"] is True
    assert status["worktree_state"] == "missing_worktree"
    assert status["branch_exists"] is False

    result = maintenance.cleanup_freestyle_job(
        job.job_id, confirm=True, project_root=root
    )

    assert result["cleanup_mode"] == "metadata_only"
    assert result["worktree_removal_mode"] == "not_present"
    assert result["actions_taken"] == ["removed_job_metadata"]
    assert not _metadata_path(root, job.job_id).exists()
    artifact = json.loads(
        _maintenance_artifacts(root, job.job_id)[-1].read_text(encoding="utf-8")
    )
    assert artifact["cleanup_mode"] == "metadata_only"
    assert artifact["git_operations_attempted"] == []


@pytest.mark.parametrize(
    "updates",
    [
        {"worktree_path": "/tmp/not-the-job-worktree"},
        {"branch": "lewis/freestyle-not-the-job"},
    ],
)
def test_cleanup_refuses_mismatched_worktree_or_branch(tmp_path, updates, monkeypatch):
    root = _init_repo(tmp_path)
    job = _job(root)
    _set_metadata(root, job.job_id, **updates)
    calls: list[tuple[str, ...]] = []
    real_run = maintenance._run_maintenance_git

    def recording_run(args, cwd):
        calls.append(tuple(args))
        return real_run(args, cwd)

    monkeypatch.setattr(maintenance, "_run_maintenance_git", recording_run)

    with pytest.raises(FreestyleJobRefusedError, match="mismatched"):
        maintenance.cleanup_freestyle_job(
            job.job_id,
            confirm=True,
            discard_worktree_changes=True,
            project_root=root,
        )

    assert Path(job.worktree_path).exists()
    assert not any("worktree" in call and "remove" in call for call in calls)


def test_cleanup_preflights_unsafe_artifact_parent_before_git_removal(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    state = root / freestyle.DEV_STATE_DIR_NAME
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (state / "reviews").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is not supported in this environment")
    (outside / job.job_id).mkdir()

    with pytest.raises(FreestyleJobRefusedError, match="parent is unsafe"):
        maintenance.cleanup_freestyle_job(job.job_id, confirm=True, project_root=root)

    assert Path(job.worktree_path).exists()
    assert _metadata_path(root, job.job_id).exists()
    assert (outside / job.job_id).exists()


def test_recover_requires_confirmation_and_marks_locks_failed(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    _set_metadata(
        root,
        job.job_id,
        latest_run_status="running",
        checks_status="running",
        edit_apply_status="applying",
    )
    source_before = _source_snapshot(root)

    with pytest.raises(FreestyleJobRefusedError, match="--confirm"):
        maintenance.recover_freestyle_job(job.job_id, project_root=root)

    recovered = maintenance.recover_freestyle_job(
        job.job_id, confirm=True, project_root=root
    )

    assert recovered.latest_run_status == "failed"
    assert recovered.checks_status == "error"
    assert recovered.edit_apply_status == "error"
    assert recovered.latest_run_status != "completed"
    assert _source_snapshot(root) == source_before
    artifact = json.loads(
        _maintenance_artifacts(root, job.job_id)[-1].read_text(encoding="utf-8")
    )
    assert artifact["operation"] == "recover"
    assert artifact["actions_taken"] == [
        "marked_stale_run_failed",
        "marked_stale_checks_failed",
        "marked_stale_apply_failed",
    ]
    assert "not marked successful" in artifact["recovery_note"]


def test_recover_requires_stronger_confirmation_for_inconsistent(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    _set_metadata(root, job.job_id, edit_apply_status="inconsistent")

    with pytest.raises(FreestyleJobRefusedError, match="confirm-inconsistent"):
        maintenance.recover_freestyle_job(job.job_id, confirm=True, project_root=root)
    assert (
        freestyle.get_freestyle_job(job.job_id, project_root=root).edit_apply_status
        == "inconsistent"
    )

    recovered = maintenance.recover_freestyle_job(
        job.job_id,
        confirm=True,
        confirm_inconsistent=True,
        project_root=root,
    )
    assert recovered.edit_apply_status == "error"
    artifacts = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in _maintenance_artifacts(root, job.job_id)
    ]
    artifact = next(
        item for item in artifacts if item.get("manual_verification_claimed") is True
    )
    assert artifact["manual_verification_claimed"] is True


def test_maintenance_git_allowlist_has_no_remotes_or_unrestricted_commands():
    allowed = maintenance._is_allowed_maintenance_git
    prefix = maintenance._MAINTENANCE_GIT_PREFIX
    assert allowed((*prefix, "worktree", "list", "--porcelain"))
    assert not allowed(("git", "status"))
    assert not allowed((*prefix, "worktree", "remove", f"/tmp/{'a' * 32}"))
    for suffix in (
        ("fetch",),
        ("pull",),
        ("push",),
        ("merge", "main"),
        ("commit", "-m", "no"),
        ("remote", "-v"),
        ("checkout", "main"),
    ):
        assert not allowed((*prefix, *suffix))


def test_cli_commands_are_explicit_and_do_not_call_execution_slices(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    def forbidden(*args, **kwargs):
        raise AssertionError("maintenance must not call execution or model slices")

    for name in (
        "propose_freestyle_job",
        "generate_freestyle_edit_bundle",
        "apply_freestyle_edit_bundle",
        "review_freestyle_job",
        "run_freestyle_checks",
        "run_freestyle_job",
    ):
        monkeypatch.setattr(freestyle, name, forbidden)

    assert freestyle.main(["cleanup-status"]) == 0
    assert job.job_id in capsys.readouterr().out
    assert freestyle.main(["recover", job.job_id]) == 1
    assert "--confirm" in capsys.readouterr().err
