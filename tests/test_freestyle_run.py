"""Tests for bounded developer-only freestyle orchestration."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_run as orchestration
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    get_freestyle_job,
    run_freestyle_job,
)
from app.dev.freestyle_proposal import DEEPSEEK_API_BASE_URL, DeepSeekProposalConfig


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
    (root / "tests" / "test_sample.py").write_text(
        "def test_sample():\n    assert True\n", encoding="utf-8"
    )
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _config() -> DeepSeekProposalConfig:
    return DeepSeekProposalConfig(
        api_key="test-deepseek-key",
        model="deepseek-test-code",
        base_url=DEEPSEEK_API_BASE_URL,
    )


def _proposal_content() -> str:
    return json.dumps(
        {
            "proposed_implementation_steps": ["Update the sample value safely."],
            "likely_files_to_edit": ["src/sample.py", "tests/test_sample.py"],
            "risks_unknowns": ["Keep the change isolated."],
            "suggested_allowlisted_checks": ["pytest"],
        }
    )


def _bundle_content(job_id: str) -> str:
    return json.dumps(
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
    )


class _SequentialCompletions:
    def __init__(self, contents: list[str], error: Exception | None = None):
        self.contents = contents
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        content = self.contents[len(self.calls) - 1]
        return SimpleNamespace(
            model="deepseek-test-code",
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150,
            ),
        )


class _FakeDeepSeekClient:
    def __init__(self, contents: list[str], error: Exception | None = None):
        self.completions = _SequentialCompletions(contents, error)
        self.chat = SimpleNamespace(completions=self.completions)


def _created_job(root: Path):
    return create_freestyle_job(
        "Update the sample value.",
        developer_approved=True,
        project_root=root,
    )


def _source_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
        and not path.is_symlink()
        and ".git" not in path.relative_to(root).parts
        and freestyle.DEV_STATE_DIR_NAME not in path.relative_to(root).parts
    }


def _run_artifact(root: Path, job) -> dict:
    assert job.latest_run_artifact_path
    path = root / freestyle.DEV_STATE_DIR_NAME / job.latest_run_artifact_path
    return json.loads(path.read_text(encoding="utf-8"))


def _set_metadata(root: Path, job_id: str, **values) -> None:
    path = root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_default_run_materializes_proposes_generates_and_stops_before_mutation(tmp_path):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    main_before = _source_snapshot(root)
    client = _FakeDeepSeekClient([_proposal_content(), _bundle_content(job.job_id)])

    completed = run_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_run_status == "completed"
    assert completed.latest_run_completed_stages == (
        "materialize",
        "propose",
        "generate_edits",
    )
    assert completed.edit_bundle_validation_status == "valid"
    assert completed.edit_apply_status == "not_applied"
    assert completed.review_status == "not_reviewed"
    assert completed.checks_status == "not_run"
    assert (Path(job.worktree_path) / "src" / "sample.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 1\n"
    assert _source_snapshot(root) == main_before
    artifact = _run_artifact(root, completed)
    assert artifact["edits_applied_by_run"] is False
    assert artifact["checks_run_by_run"] is False
    assert artifact["generated_bundle_validation_status"] == "valid"
    assert artifact["stages"][2]["status"] == "completed"
    assert [stage["status"] for stage in artifact["stages"][-3:]] == [
        "skipped",
        "skipped",
        "skipped",
    ]
    assert "No commit, merge, push" in artifact["integration_statement"]
    assert artifact["human_next_steps"]


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        ({"apply_latest_generated": True}, "requires both"),
        ({"confirm_apply": True}, "requires both"),
        ({"check_ids": ["pytest-freestyle"]}, "--check requires --run-checks"),
    ],
)
def test_contradictory_flags_fail_before_any_stage_starts(tmp_path, flags, message):
    root = _init_repo(tmp_path)
    job = _created_job(root)

    with pytest.raises(FreestyleJobRefusedError, match=message):
        run_freestyle_job(job.job_id, project_root=root, **flags)

    persisted = get_freestyle_job(job.job_id, project_root=root)
    assert persisted.latest_run_status == "not_run"
    assert persisted.materialization_status == "not_materialized"
    assert not (root / freestyle.DEV_STATE_DIR_NAME / "runs").exists()


def test_explicit_apply_uses_generated_bundle_applier_then_creates_review(tmp_path):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    main_before = _source_snapshot(root)
    client = _FakeDeepSeekClient([_proposal_content(), _bundle_content(job.job_id)])

    completed = run_freestyle_job(
        job.job_id,
        apply_latest_generated=True,
        confirm_apply=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_apply_status == "applied"
    assert completed.edit_artifact_path
    assert completed.review_status == "reviewed"
    assert completed.review_artifact_path
    assert completed.checks_status == "not_run"
    assert (Path(job.worktree_path) / "src" / "sample.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    assert _source_snapshot(root) == main_before
    artifact = _run_artifact(root, completed)
    assert artifact["edits_applied_by_run"] is True
    assert artifact["final_changed_file_summary"]["changed_file_count"] == 1
    assert artifact["artifact_references"]["applied_edits"]
    assert artifact["artifact_references"]["review"]


def test_apply_latest_never_reads_or_applies_a_tampered_generated_bundle_path(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    client = _FakeDeepSeekClient([_proposal_content(), _bundle_content(job.job_id)])
    generated = run_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )
    outside = tmp_path / "outside-bundle.json"
    outside.write_text(_bundle_content(job.job_id), encoding="utf-8")
    _set_metadata(
        root,
        job.job_id,
        edit_bundle_path=str(outside),
        latest_run_status="completed",
    )

    def refuse_regeneration(*args, **kwargs):
        raise FreestyleJobRefusedError("tampered generated reference refused")

    def forbidden_apply(*args, **kwargs):
        raise AssertionError("tampered generated reference must never reach applier")

    monkeypatch.setattr(
        orchestration, "generate_freestyle_edit_bundle", refuse_regeneration
    )
    monkeypatch.setattr(orchestration, "apply_freestyle_edit_bundle", forbidden_apply)
    with pytest.raises(FreestyleJobRefusedError, match="bounded failure was recorded"):
        run_freestyle_job(
            generated.job_id,
            apply_latest_generated=True,
            confirm_apply=True,
            project_root=root,
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_apply_status == "not_applied"
    assert failed.latest_run_status == "failed"
    assert outside.read_text(encoding="utf-8") == _bundle_content(job.job_id)


def test_checks_require_applied_edits_and_dispatch_only_selected_ids(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    with pytest.raises(FreestyleJobRefusedError, match="require previously applied"):
        run_freestyle_job(job.job_id, run_checks=True, project_root=root)
    assert get_freestyle_job(job.job_id, project_root=root).latest_run_status == "not_run"

    client = _FakeDeepSeekClient([_proposal_content(), _bundle_content(job.job_id)])
    calls: list[tuple[str, tuple[str, ...]]] = []

    def fake_checks(job_id, *, check_ids=None, project_root=None, state_root=None):
        calls.append((job_id, tuple(check_ids or ())))
        current = get_freestyle_job(
            job_id, project_root=project_root, state_root=state_root
        )
        return replace(
            current,
            checks_status="passed",
            checks_artifact_path=f"checks/{job_id}/checks-test.json",
            checks_selected_ids=tuple(check_ids or ()),
        )

    monkeypatch.setattr(orchestration, "run_freestyle_checks", fake_checks)
    completed = run_freestyle_job(
        job.job_id,
        apply_latest_generated=True,
        confirm_apply=True,
        run_checks=True,
        check_ids=["pytest-freestyle"],
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert calls == [(job.job_id, ("pytest-freestyle",))]
    assert completed.checks_status == "passed"
    assert _run_artifact(root, completed)["checks_run_by_run"] is True


def test_unknown_check_selection_fails_before_materialization(tmp_path):
    root = _init_repo(tmp_path)
    job = _created_job(root)

    with pytest.raises(FreestyleJobRefusedError, match="Unknown check id"):
        run_freestyle_job(
            job.job_id,
            apply_latest_generated=True,
            confirm_apply=True,
            run_checks=True,
            check_ids=["not-allowlisted"],
            project_root=root,
        )

    assert get_freestyle_job(job.job_id, project_root=root).materialization_status == (
        "not_materialized"
    )


def test_completed_stages_are_skipped_idempotently(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    client = _FakeDeepSeekClient([_proposal_content(), _bundle_content(job.job_id)])
    first = run_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("completed stages must not rerun")

    monkeypatch.setattr(orchestration, "materialize_freestyle_job", forbidden)
    monkeypatch.setattr(orchestration, "propose_freestyle_job", forbidden)
    monkeypatch.setattr(orchestration, "generate_freestyle_edit_bundle", forbidden)
    second = run_freestyle_job(first.job_id, project_root=root)

    artifact = _run_artifact(root, second)
    assert second.latest_run_completed_stages == ()
    assert [stage["status"] for stage in artifact["stages"]] == [
        "skipped",
        "skipped",
        "skipped",
        "skipped",
        "skipped",
        "skipped",
    ]
    assert len(client.completions.calls) == 2


def test_stage_failure_persists_bounded_run_artifact_and_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    client = _FakeDeepSeekClient([], error=RuntimeError("proposal exploded " + "x" * 900))

    with pytest.raises(FreestyleJobRefusedError, match="bounded failure was recorded"):
        run_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.latest_run_status == "failed"
    assert failed.latest_run_artifact_path
    assert failed.latest_run_completed_stages == ("materialize",)
    assert len(failed.latest_run_errors[0]) <= 500
    artifact = _run_artifact(root, failed)
    assert artifact["status"] == "failed"
    assert artifact["errors"]
    assert artifact["stages"][-1]["status"] == "failed"
    assert artifact["artifact_references"]["proposal_failure"]
    assert "Inspect proposal diagnostics before retrying." in artifact["human_next_steps"]
    assert artifact["human_next_steps"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("edit_apply_status", "applying"),
        ("edit_apply_status", "inconsistent"),
        ("checks_status", "running"),
        ("latest_run_status", "running"),
    ],
)
def test_inconsistent_or_running_states_fail_closed_before_new_run(
    tmp_path, field, value
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    _set_metadata(root, job.job_id, **{field: value})

    with pytest.raises(FreestyleJobRefusedError, match="Recover or verify"):
        run_freestyle_job(job.job_id, project_root=root)

    persisted = get_freestyle_job(job.job_id, project_root=root)
    assert getattr(persisted, field) == value
    assert not (root / freestyle.DEV_STATE_DIR_NAME / "runs").exists()


def test_run_cli_dispatches_explicit_flags_and_rejects_contradiction(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = _created_job(root)
    calls: list[dict] = []

    def fake_run(job_id, **kwargs):
        calls.append({"job_id": job_id, **kwargs})
        return job

    monkeypatch.setattr(freestyle, "run_freestyle_job", fake_run)
    assert (
        freestyle.main(
            [
                "run",
                job.job_id,
                "--apply-latest-generated",
                "--confirm-apply",
                "--run-checks",
                "--check",
                "pytest-freestyle",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id
    assert calls == [
        {
            "job_id": job.job_id,
            "apply_latest_generated": True,
            "confirm_apply": True,
            "run_checks": True,
            "check_ids": ["pytest-freestyle"],
        }
    ]


def test_orchestration_module_adds_no_execution_or_runtime_dependency():
    source = Path(orchestration.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "subprocess",
        "ApprovalStore",
        "sqlite",
        "load_dotenv",
        "dotenv",
        "os.environ",
        "os.getenv",
        "from app.main",
        "from app.web",
        "from app.tools",
    ):
        assert forbidden not in source
