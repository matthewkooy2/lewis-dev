"""CLI presentation tests for expected freestyle developer refusals."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

import pytest

import app.dev.freestyle_checks as checks
import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_maintenance as maintenance
import app.dev.freestyle_proposal as proposal
import app.dev.freestyle_run as orchestration


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


def _created_job(root: Path) -> freestyle.FreestyleJob:
    return freestyle.create_freestyle_job(
        "Exercise clean CLI refusals.",
        developer_approved=True,
        project_root=root,
    )


def _materialized_job(root: Path) -> freestyle.FreestyleJob:
    job = _created_job(root)
    return freestyle.materialize_freestyle_job(job.job_id, project_root=root)


def _json_error(capsys) -> dict:
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    payload = json.loads(captured.err)
    assert payload["ok"] is False
    assert len(payload["message"]) <= 500
    return payload


def test_propose_missing_deepseek_config_is_clean_and_preserves_failure_metadata(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)
    monkeypatch.setattr(proposal, "ROOT_DIR", root)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert freestyle.main(["propose", job.job_id]) == 1

    error = _json_error(capsys)
    assert error["error_type"] == "freestyle_job_refused"
    assert error["command"] == "propose"
    assert error["job_id"] == job.job_id
    assert "DEEPSEEK_API_KEY must be set" in error["message"]
    failed = freestyle.get_freestyle_job(job.job_id, project_root=root)
    assert failed.proposal_status == "error"
    assert failed.proposal_errors == (
        "DEEPSEEK_API_KEY must be set in the process environment.",
    )


def test_run_missing_deepseek_config_is_clean_and_reports_recorded_artifact(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)
    monkeypatch.setattr(orchestration, "ROOT_DIR", root)
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    assert freestyle.main(["run", job.job_id]) == 1

    error = _json_error(capsys)
    assert error["command"] == "run"
    assert error["artifact_path"]
    failed = freestyle.get_freestyle_job(job.job_id, project_root=root)
    assert failed.latest_run_status == "failed"
    assert failed.latest_run_artifact_path == error["artifact_path"]
    assert (root / freestyle.DEV_STATE_DIR_NAME / error["artifact_path"]).is_file()


@pytest.mark.parametrize(
    "arguments",
    [
        ["propose"],
        ["show", "not-a-job-id"],
        ["run-checks", "0" * 32, "--check", "unknown-check"],
        ["run", "0" * 32, "--check", "pytest-freestyle"],
    ],
)
def test_expected_input_and_validation_refusals_are_clean(arguments, capsys):
    assert freestyle.main(arguments) != 0
    error = _json_error(capsys)
    assert error["error_type"] in {"freestyle_job_refused", "invalid_cli_input"}


def test_dirty_create_refusal_is_clean_and_bounded(tmp_path, monkeypatch, capsys):
    root = _init_repo(tmp_path)
    (root / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    assert freestyle.main(["create", "--approve", "--goal", "Use clean HEAD."]) == 1

    error = _json_error(capsys)
    assert error["error_type"] == "freestyle_job_refused"
    assert error["command"] == "create"
    assert "main checkout has uncommitted" in error["message"]
    assert not (root / freestyle.DEV_STATE_DIR_NAME).exists()


def test_dirty_cleanup_refusal_is_clean_and_bounded(tmp_path, monkeypatch, capsys):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    (Path(job.worktree_path) / "tracked.txt").write_text("dirty\n", encoding="utf-8")
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    assert freestyle.main(["cleanup", job.job_id, "--confirm"]) == 1

    error = _json_error(capsys)
    assert error["error_type"] == "freestyle_job_refused"
    assert error["command"] == "cleanup"
    assert error["job_id"] == job.job_id
    assert "--discard-worktree-changes" in error["message"]


def test_git_cleanup_removal_failure_is_clean_and_bounded(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)
    real_run = maintenance._run_maintenance_git

    def refuse_remove(args, cwd):
        if "worktree" in args and "remove" in args:
            return freestyle.GitCommandResult(
                tuple(args),
                1,
                "",
                f"fatal: {job.worktree_path!r} contains modified or untracked files",
            )
        return real_run(args, cwd)

    monkeypatch.setattr(maintenance, "_run_maintenance_git", refuse_remove)

    assert freestyle.main(["cleanup", job.job_id, "--confirm"]) == 1

    error = _json_error(capsys)
    assert error["error_type"] == "freestyle_job_refused"
    assert error["command"] == "cleanup"
    assert "contains modified or untracked files" in error["message"]
    assert Path(job.worktree_path).exists()


@pytest.mark.parametrize("bundle_path", ["../unsafe.json", "malformed.json"])
def test_invalid_or_malformed_bundle_is_clean(
    bundle_path, tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)
    (Path(job.worktree_path) / "malformed.json").write_text(
        "not json", encoding="utf-8"
    )

    assert freestyle.main(["apply-edits", job.job_id, bundle_path]) == 1
    assert _json_error(capsys)["command"] == "apply-edits"


def test_unknown_check_id_is_refused_before_check_execution(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)
    monkeypatch.setattr(checks, "ROOT_DIR", root)

    def forbidden_run(*args, **kwargs):
        raise AssertionError("check subprocess must not execute")

    monkeypatch.setattr(checks.subprocess, "run", forbidden_run)

    assert freestyle.main(["run-checks", job.job_id, "--check", "unknown-check"]) == 1
    assert _json_error(capsys)["command"] == "run-checks"


def test_running_state_refusal_is_clean_and_does_not_clear_state(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _materialized_job(root)
    state = root / freestyle.DEV_STATE_DIR_NAME
    freestyle._write_job(
        replace(job, proposal_status="proposing", updated_at=freestyle._utc_now()),
        state,
    )
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    assert freestyle.main(["run", job.job_id]) == 1
    assert _json_error(capsys)["command"] == "run"
    persisted = freestyle.get_freestyle_job(job.job_id, project_root=root)
    assert persisted.proposal_status == "proposing"


def test_expected_error_output_redacts_secret_environment_values(
    monkeypatch, capsys
):
    secret = "secret-value-that-must-not-appear"
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)

    def refuse(job_id):
        raise freestyle.FreestyleJobRefusedError(f"provider refused {secret}")

    monkeypatch.setattr(freestyle, "propose_freestyle_job", refuse)

    assert freestyle.main(["propose", "0" * 32]) == 1
    captured = capsys.readouterr()
    assert secret not in captured.err
    assert "[REDACTED]" in captured.err


def test_secret_environment_value_is_redacted_from_job_id(monkeypatch, capsys):
    secret = "a" * 32
    monkeypatch.setenv("DEEPSEEK_API_KEY", secret)

    assert freestyle.main(["show", secret]) == 1

    captured = capsys.readouterr()
    assert secret not in captured.err
    assert json.loads(captured.err)["job_id"] == "[REDACTED]"


def test_status_refusal_uses_human_stderr(tmp_path, monkeypatch, capsys):
    root = _init_repo(tmp_path)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    assert freestyle.main(["status", "invalid"]) == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ERROR [freestyle_job_refused]" in captured.err
    assert "Traceback" not in captured.err


def test_unexpected_exception_is_not_hidden(monkeypatch):
    def fail_loudly(job_id):
        raise RuntimeError("unexpected implementation bug")

    monkeypatch.setattr(freestyle, "propose_freestyle_job", fail_loudly)

    with pytest.raises(RuntimeError, match="unexpected implementation bug"):
        freestyle.main(["propose", "0" * 32])
