"""Tests for read-only developer freestyle status/handoff output."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_handoff as handoff
import app.dev.freestyle_review as review
from app.dev.freestyle_coding import (
    apply_freestyle_edit_bundle,
    create_freestyle_job,
    get_freestyle_handoff,
    materialize_freestyle_job,
    review_freestyle_job,
    run_freestyle_checks,
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
    (root / "src").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".gitignore").write_text(".lewis-dev/\n", encoding="utf-8")
    (root / "src" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_freestyle_sample.py").write_text(
        "def test_freestyle_sample():\n    assert True\n", encoding="utf-8"
    )
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _created_job(root: Path):
    return create_freestyle_job(
        "Summarize this freestyle job for a human.",
        title="Human handoff",
        developer_approved=True,
        project_root=root,
    )


def _metadata_path(root: Path, job_id: str) -> Path:
    return root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"


def _set_metadata(root: Path, job_id: str, **values) -> None:
    path = _metadata_path(root, job_id)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
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


def _bundle(root: Path, job_id: str) -> Path:
    path = root / freestyle.DEV_STATE_DIR_NAME / "test-bundle.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "job_id": job_id,
                "created_at": "2026-06-11T12:00:00+00:00",
                "operation_count": 1,
                "operations": [
                    {
                        "op": "replace_text",
                        "path": "src/sample.py",
                        "old_text": "VALUE = 1",
                        "new_text": "VALUE = 2",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_status_works_for_fresh_approved_job_and_cli_outputs_markdown_and_json(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)
    metadata_before = _metadata_path(root, job.job_id).read_bytes()

    status = get_freestyle_handoff(job.job_id, project_root=root)

    assert status["job"] == {
        "job_id": job.job_id,
        "title": "Human handoff",
        "goal": job.goal,
        "approval_state": "approved",
        "base_commit": job.base_commit,
        "base_source": "current clean repo HEAD at job creation",
        "branch": job.branch,
        "worktree_path": job.worktree_path,
    }
    assert status["stages"]["materialize"] == "not_materialized"
    assert status["changed_files"]["source"] == "not_materialized"
    assert status["human_next_steps"][0] == "Materialize the isolated job worktree."
    assert "has not committed, merged, pushed" in status["integration_statement"]
    assert _metadata_path(root, job.job_id).read_bytes() == metadata_before

    assert freestyle.main(["status", job.job_id]) == 0
    markdown = capsys.readouterr().out
    assert "# Freestyle Job" in markdown
    assert "current clean repo `HEAD` at job creation" in markdown
    assert "## Human Next Steps" in markdown
    assert "Materialize the isolated job worktree." in markdown
    assert freestyle.main(["status", job.job_id, "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["job"]["job_id"] == job.job_id


def test_status_after_materialize_apply_review_and_checks_surfaces_artifacts_and_steps(
    tmp_path,
):
    root = _init_repo(tmp_path)
    created = _created_job(root)
    materialize_freestyle_job(created.job_id, project_root=root)
    applied = apply_freestyle_edit_bundle(
        created.job_id, str(_bundle(root, created.job_id)), project_root=root
    )
    reviewed = review_freestyle_job(applied.job_id, project_root=root)
    checked = run_freestyle_checks(
        reviewed.job_id, check_ids=["pytest-freestyle"], project_root=root
    )

    status = get_freestyle_handoff(checked.job_id, project_root=root)

    assert status["stages"]["materialize"] == "materialized"
    assert status["stages"]["apply-edits"] == "applied"
    assert status["stages"]["review"] == "reviewed"
    assert status["stages"]["run-checks"] == "passed"
    assert status["artifacts"]["applied_edit"] == applied.edit_artifact_path
    assert status["artifacts"]["review"] == reviewed.review_artifact_path
    assert status["artifacts"]["checks"] == checked.checks_artifact_path
    assert status["changed_files"]["source"] == "latest_review_artifact"
    assert status["changed_files"]["files"] == [
        {"path": "src/sample.py", "status": "modified", "git_status": " M"}
    ]
    assert status["checks"]["overall_status"] == "passed"
    assert status["checks"]["passed"] == 1
    assert any(
        "Manually merge outside Lewis" in step for step in status["human_next_steps"]
    )

    _set_metadata(
        root,
        checked.job_id,
        proposal_status="error",
        proposal_errors=["DEEPSEEK_API_KEY must be set in the process environment."],
    )
    status_with_stale_error = get_freestyle_handoff(checked.job_id, project_root=root)
    assert status_with_stale_error["blocked"]["missing_deepseek_config"] is False
    assert status_with_stale_error["blocked"]["is_blocked"] is False
    assert "DEEPSEEK_API_KEY" in status_with_stale_error["bounded_errors"][0]


def test_status_surfaces_every_latest_artifact_reference(tmp_path):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    references = {
        "proposal_artifact_path": f"proposals/{job.job_id}/proposal.json",
        "proposal_failure_artifact_path": (
            f"proposals/{job.job_id}/proposal-failure.json"
        ),
        "edit_bundle_path": f"edit-bundles/{job.job_id}/bundle-abc.json",
        "edit_generation_artifact_path": (
            f"edit-bundles/{job.job_id}/generation-abc.json"
        ),
        "implementation_plan_artifact_path": (
            f"implementation-plans/{job.job_id}/abc-implementation-plan.json"
        ),
        "edit_artifact_path": f"edits/{job.job_id}/apply-abc.json",
        "review_artifact_path": f"reviews/{job.job_id}/review-abc.json",
        "checks_artifact_path": f"checks/{job.job_id}/checks-abc.json",
        "latest_run_artifact_path": f"runs/{job.job_id}/run-abc.json",
    }
    _set_metadata(root, job.job_id, **references)

    status = get_freestyle_handoff(job.job_id, project_root=root)

    assert status["artifacts"] == {
        "proposal": references["proposal_artifact_path"],
        "proposal_failure": references["proposal_failure_artifact_path"],
        "generated_bundle": references["edit_bundle_path"],
        "edit_generation": references["edit_generation_artifact_path"],
        "implementation_plan": references["implementation_plan_artifact_path"],
        "applied_edit": references["edit_artifact_path"],
        "review": references["review_artifact_path"],
        "checks": references["checks_artifact_path"],
        "orchestration_run": references["latest_run_artifact_path"],
        "latest_refinement": None,
    }


def test_status_and_handoff_surface_bounded_proposal_failure_diagnostics(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    reference = f"proposals/{job.job_id}/proposal-failure.json"
    artifact = root / freestyle.DEV_STATE_DIR_NAME / reference
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "proposal_attempts": [
                    {"attempt": 1, "status": "empty_response"},
                    {"attempt": 2, "status": "unsafe_likely_path"},
                ]
            }
        ),
        encoding="utf-8",
    )
    _set_metadata(
        root,
        job.job_id,
        proposal_status="error",
        proposal_failure_artifact_path=reference,
    )
    monkeypatch.setattr(freestyle, "ROOT_DIR", root)

    status = get_freestyle_handoff(job.job_id, project_root=root)
    markdown = handoff.render_freestyle_handoff_markdown(status)
    assert status["proposal_failure_diagnostics"]["artifact_path"] == reference
    assert (
        "attempt 1: empty_response" in status["proposal_failure_diagnostics"]["summary"]
    )
    assert "Proposal Failure Diagnostics" in markdown
    assert "Inspect proposal diagnostics before retrying." in markdown

    assert freestyle.main(["status", job.job_id, "--json"]) == 0
    cli_status = json.loads(capsys.readouterr().out)
    assert cli_status["proposal_failure_diagnostics"]["artifact_path"] == reference
    assert freestyle.main(["show", job.job_id]) == 0
    assert (
        json.loads(capsys.readouterr().out)["proposal_failure_artifact_path"]
        == reference
    )


@pytest.mark.parametrize(
    ("metadata", "flag", "message"),
    [
        (
            {
                "proposal_status": "error",
                "proposal_errors": [
                    "DEEPSEEK_API_KEY must be set in the process environment."
                ],
            },
            "missing_deepseek_config",
            "DEEPSEEK_API_KEY",
        ),
        (
            {"edit_apply_status": "inconsistent"},
            "inconsistent_edit_state",
            "inconsistent",
        ),
        (
            {"checks_status": "running"},
            "running_lock",
            "running",
        ),
        (
            {"checks_status": "failed", "checks_failed_count": 1},
            "failed_checks",
            "failed",
        ),
        (
            {"edit_apply_status": "applied", "review_status": "not_reviewed"},
            "missing_review",
            "review",
        ),
    ],
)
def test_status_reports_bounded_blocking_states_without_traceback(
    tmp_path, metadata, flag, message
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    _set_metadata(root, job.job_id, **metadata)

    status = get_freestyle_handoff(job.job_id, project_root=root)
    rendered = handoff.render_freestyle_handoff_markdown(status)

    assert status["blocked"][flag] is True
    assert status["blocked"]["is_blocked"] is True
    assert message.casefold() in rendered.casefold()
    assert "Traceback" not in rendered
    assert all(len(error) <= 500 for error in status["bounded_errors"])


def test_status_only_reads_dev_state_and_constrained_isolated_git_status(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = materialize_freestyle_job(_created_job(root).job_id, project_root=root)
    main_before = _source_snapshot(root)
    worktree_before = _source_snapshot(Path(job.worktree_path))
    state_before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / freestyle.DEV_STATE_DIR_NAME).rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    calls: list[list[str]] = []
    real_freestyle_run = freestyle.subprocess.run
    real_review_run = review.subprocess.run

    def record_freestyle(args, **kwargs):
        calls.append(list(args))
        return real_freestyle_run(args, **kwargs)

    def record_review(args, **kwargs):
        calls.append(list(args))
        return real_review_run(args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "status must not call model, apply, review, checks, or run"
        )

    monkeypatch.setattr(freestyle.subprocess, "run", record_freestyle)
    monkeypatch.setattr(review.subprocess, "run", record_review)
    monkeypatch.setattr(freestyle, "propose_freestyle_job", forbidden)
    monkeypatch.setattr(freestyle, "generate_freestyle_edit_bundle", forbidden)
    monkeypatch.setattr(freestyle, "apply_freestyle_edit_bundle", forbidden)
    monkeypatch.setattr(freestyle, "review_freestyle_job", forbidden)
    monkeypatch.setattr(freestyle, "run_freestyle_checks", forbidden)
    monkeypatch.setattr(freestyle, "run_freestyle_job", forbidden)

    status = get_freestyle_handoff(job.job_id, project_root=root)

    assert status["changed_files"]["source"] == "current_isolated_worktree_status"
    assert calls
    assert all(call[0] == "git" for call in calls)
    forbidden_git = {"commit", "merge", "push", "pull", "fetch", "remote", "switch"}
    assert all(not forbidden_git.intersection(call) for call in calls)
    assert _source_snapshot(root) == main_before
    assert _source_snapshot(Path(job.worktree_path)) == worktree_before
    state_after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in (root / freestyle.DEV_STATE_DIR_NAME).rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    assert state_after == state_before
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_handoff_surfaces_rendered_ui_review_warnings(tmp_path):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    warning = (
        "UI_REVIEW_BROKEN_CALENDAR_LAYOUT: weekday labels are rendered without "
        "any grid/flex container."
    )
    _set_metadata(root, job.job_id, review_ui_warnings=[warning])

    result = get_freestyle_handoff(job.job_id, project_root=root)

    assert warning in result["ui_review_warnings"]
    assert warning in result["risks"]

    markdown = handoff.render_freestyle_handoff_markdown(result)
    assert "Rendered UI Review (Advisory)" in markdown
    assert warning in markdown


def test_handoff_module_adds_no_model_env_runtime_or_mutating_dependencies():
    source = Path(handoff.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "DeepSeekProposalConfig",
        "create_deepseek_client",
        "run_freestyle_checks",
        "apply_freestyle_edit_bundle",
        "ApprovalStore",
        "sqlite",
        "load_dotenv",
        "dotenv",
        "os.environ",
        "os.getenv",
        "subprocess",
        "from app.main",
        "from app.web",
        "from app.tools",
    ):
        assert forbidden not in source
