"""Tests for the developer-only freestyle command center."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess

from fastapi.testclient import TestClient

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_web as freestyle_web
from app.config import Settings
from app.dev.freestyle_coding import FreestyleJobRefusedError, create_freestyle_job
from app.dev.freestyle_web import MAX_WEB_PREVIEW_CHARS
from app.web import create_app


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
    (root / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _settings(tmp_path: Path, *, debug: bool) -> Settings:
    documents = tmp_path / "documents"
    documents.mkdir(exist_ok=True)
    return Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=debug,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=documents,
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
        deepseek_code_model="deepseek-code-test",
    )


def _client(tmp_path: Path, root: Path, *, debug: bool = True) -> TestClient:
    app = create_app(_settings(tmp_path, debug=debug))
    app.state.freestyle_project_root = root
    return TestClient(app)


def _job(root: Path):
    return create_freestyle_job(
        "Inspect a bounded freestyle web job.",
        title="Web command center",
        developer_approved=True,
        project_root=root,
    )


def _metadata_path(root: Path, job_id: str) -> Path:
    return root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"


def _set_metadata(root: Path, job_id: str, **values: object) -> None:
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


def test_freestyle_page_and_navigation_are_debug_only(tmp_path):
    root = _init_repo(tmp_path)
    normal = _client(tmp_path, root, debug=False)
    dev = _client(tmp_path, root)

    assert normal.get("/dev/freestyle").status_code == 404
    assert "/dev/freestyle" not in normal.get("/dashboard").text
    response = dev.get("/dev/freestyle")
    assert response.status_code == 200
    assert "Freestyle Command Center" in response.text
    assert "/dev/freestyle" in dev.get("/dashboard").text


def test_page_renders_empty_state_and_requires_explicit_create_approval(tmp_path):
    root = _init_repo(tmp_path)
    client = _client(tmp_path, root)

    empty = client.get("/dev/freestyle")
    refused = client.post(
        "/dev/freestyle/create",
        data={"title": "No approval", "goal": "Do not create this."},
    )

    assert "No freestyle jobs yet." in empty.text
    assert "explicitly approve" in empty.text
    assert "current clean repo HEAD" in empty.text
    assert "Explicit developer approval is required" in refused.text
    assert freestyle.list_freestyle_jobs(project_root=root) == []

    created = client.post(
        "/dev/freestyle/create",
        data={
            "title": "Approved web job",
            "goal": "Create one isolated job.",
            "developer_approved": "approved",
        },
    )
    assert "Created approved freestyle job" in created.text
    assert "Approved web job" in created.text
    assert "base commit (current clean repo HEAD at creation)" in created.text
    assert len(freestyle.list_freestyle_jobs(project_root=root)) == 1


def test_populated_detail_renders_handoff_artifacts_and_escaped_bounded_preview(
    tmp_path,
):
    root = _init_repo(tmp_path)
    job = _job(root)
    artifact_dir = root / freestyle.DEV_STATE_DIR_NAME / "proposals" / job.job_id
    artifact_dir.mkdir(parents=True)
    artifact_path = artifact_dir / "proposal.json"
    artifact_path.write_text(
        json.dumps(
            {
                "risks_unknowns": ["Review <script>alert('risk')</script> manually."],
                "preview": "<script>alert('preview')</script>" + "x" * 30_000,
            }
        ),
        encoding="utf-8",
    )
    _set_metadata(
        root,
        job.job_id,
        proposal_status="error",
        proposal_artifact_path=f"proposals/{job.job_id}/proposal.json",
        proposal_errors=["DeepSeek proposal configuration is missing."],
    )

    response = _client(tmp_path, root).get(f"/dev/freestyle/{job.job_id}")

    assert response.status_code == 200
    assert "Stage Status" in response.text
    assert "Latest Artifacts" in response.text
    assert "Blockers" in response.text
    assert "Errors And Risks" in response.text
    assert "Next Manual Steps" in response.text
    assert "Missing or invalid DeepSeek configuration." in response.text
    assert "proposal preview" in response.text
    assert "&lt;script&gt;alert" in response.text
    assert "<script>alert" not in response.text
    assert "[preview truncated]" in response.text
    assert len(response.text) < MAX_WEB_PREVIEW_CHARS + 40_000
    assert "does not commit, merge, push" in response.text
    assert f"/dev/freestyle/{job.job_id}/apply" in response.text
    assert f"/dev/freestyle/{job.job_id}/review" in response.text
    assert f"/dev/freestyle/{job.job_id}/run-checks" in response.text
    assert f"/dev/freestyle/{job.job_id}/cleanup" in response.text
    assert f"/dev/freestyle/{job.job_id}/recover" in response.text
    assert "Advanced Maintenance And Danger Actions" in response.text
    assert "Raw backend error" not in response.text


def test_web_surfaces_bounded_proposal_failure_summary_without_model_output(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    reference = f"proposals/{job.job_id}/proposal-failure.json"
    artifact = root / freestyle.DEV_STATE_DIR_NAME / reference
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "proposal_attempts": [
                    {
                        "attempt": 1,
                        "status": "unsafe_likely_path",
                        "model_output_preview": "DO_NOT_RENDER_FULL_MODEL_OUTPUT",
                    }
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

    response = _client(tmp_path, root).get(f"/dev/freestyle/{job.job_id}")

    assert "Proposal Failure Diagnostics" in response.text
    assert reference in response.text
    assert "attempt 1: unsafe_likely_path" in response.text
    assert "Full model output is not displayed here." in response.text
    assert "DO_NOT_RENDER_FULL_MODEL_OUTPUT" not in response.text


def test_default_run_uses_only_non_mutating_orchestration_flags(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[dict[str, object]] = []

    def fake_run(job_id: str, **kwargs):
        calls.append({"job_id": job_id, **kwargs})
        return job

    monkeypatch.setattr(freestyle_web, "run_freestyle_job", fake_run)
    client = _client(tmp_path, root)

    response = client.post(
        f"/dev/freestyle/{job.job_id}/run", follow_redirects=False
    )

    assert response.status_code == 303
    assert calls == [
        {
            "job_id": job.job_id,
            "apply_latest_generated": False,
            "confirm_apply": False,
            "run_checks": False,
            "check_ids": None,
            "project_root": root,
            "deepseek_config": calls[0]["deepseek_config"],
        }
    ]


def test_detail_page_renders_rendered_ui_review_warnings(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    warning = (
        "UI_REVIEW_JS_MISSING_TARGETS: the changed JavaScript references "
        "#calendar-grid but no rendered element provides it."
    )
    _set_metadata(root, job.job_id, review_ui_warnings=[warning])

    response = _client(tmp_path, root).get(f"/dev/freestyle/{job.job_id}")

    assert response.status_code == 200
    assert "Rendered UI review warnings" in response.text
    assert "UI_REVIEW_JS_MISSING_TARGETS" in response.text
    assert "#calendar-grid" in response.text


def test_materialize_route_uses_existing_isolated_operation_only(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    source_before = _source_snapshot(root)

    response = _client(tmp_path, root).post(
        f"/dev/freestyle/{job.job_id}/materialize"
    )

    updated = freestyle.get_freestyle_job(job.job_id, project_root=root)
    assert response.status_code == 200
    assert "Materialized the isolated worktree." in response.text
    assert updated.materialization_status == "materialized"
    assert Path(updated.worktree_path).is_dir()
    assert _source_snapshot(root) == source_before
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"


def test_actions_reject_arbitrary_fields_before_backend_calls(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("bounded action must not run")

    for name in (
        "apply_freestyle_edit_bundle",
        "cleanup_freestyle_job",
        "refine_freestyle_job",
        "recover_freestyle_job",
        "review_freestyle_job",
        "run_freestyle_checks",
        "run_freestyle_job",
    ):
        monkeypatch.setattr(freestyle_web, name, forbidden)
    client = _client(tmp_path, root)

    for suffix in ("run", "apply", "refine", "review", "run-checks", "cleanup", "recover"):
        rejected = client.post(
            f"/dev/freestyle/{job.job_id}/{suffix}",
            data={
                "command": "git push",
                "args": "--force",
                "git_args": "reset --hard",
                "cwd": "/tmp",
                "env": "SECRET=x",
                "executable": "bash",
                "provider": "other",
                "shell": "true",
            },
        )
        assert "Freestyle action stopped" in rejected.text
    assert calls == []

    malformed = client.post(
        f"/dev/freestyle/{job.job_id}/run",
        content=b"{not-json",
        headers={"content-type": "application/json"},
    )
    assert "Default run accepts no input fields." in malformed.text
    assert calls == []
    assert client.get("/dev/freestyle?command=git+push").status_code == 400


def test_action_routes_are_debug_only(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    normal = _client(tmp_path, root, debug=False)

    for suffix in ("apply", "refine", "review", "run-checks", "cleanup", "recover"):
        assert normal.post(f"/dev/freestyle/{job.job_id}/{suffix}").status_code == 404


def test_refine_web_requires_goal_and_approval_and_uses_safe_defaults(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[dict[str, object]] = []

    def fake_refine(job_id: str, goal: str, **kwargs):
        calls.append({"job_id": job_id, "goal": goal, **kwargs})
        return job

    monkeypatch.setattr(freestyle_web, "refine_freestyle_job", fake_refine)
    client = _client(tmp_path, root)
    refused = client.post(
        f"/dev/freestyle/{job.job_id}/refine", data={"goal": "Wire route."}
    )
    assert "Explicit developer approval is required" in refused.text
    assert calls == []

    response = client.post(
        f"/dev/freestyle/{job.job_id}/refine",
        data={"goal": "Wire route.", "developer_approved": "approved"},
    )
    assert "Generated a refinement bundle from the current isolated worktree" in response.text
    assert calls == [
        {
            "job_id": job.job_id,
            "goal": "Wire route.",
            "developer_approved": True,
            "observed_ui_feedback": None,
            "project_root": root,
            "deepseek_config": calls[0]["deepseek_config"],
        }
    ]


def test_detail_renders_original_and_latest_refinement_state(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    freestyle.materialize_freestyle_job(job.job_id, project_root=root)
    refinement_id = "1" * 16
    reference = (
        f"refinements/{job.job_id}/refinement-{refinement_id}-0001.json"
    )
    artifact = root / freestyle.DEV_STATE_DIR_NAME / reference
    artifact.parent.mkdir(parents=True)
    artifact.write_text("{}\n", encoding="utf-8")
    _set_metadata(
        root,
        job.job_id,
        original_artifacts={"proposal": f"proposals/{job.job_id}/proposal.json"},
        original_stage_statuses={"edit_apply": "applied", "review": "reviewed"},
        refinement_count=1,
        latest_refinement_id=refinement_id,
        latest_refinement_goal="Wire /shenanigans.",
        latest_refinement_status="generated",
        latest_refinement_artifact_path=reference,
        refinement_history=[
            {
                "iteration": 1,
                "refinement_id": refinement_id,
                "refinement_goal": "Wire /shenanigans.",
                "status": "generated",
                "revision": 1,
                "artifact_path": reference,
            }
        ],
    )

    response = _client(tmp_path, root).get(f"/dev/freestyle/{job.job_id}")

    assert "Refine This Job" in response.text
    assert "Refinement History" in response.text
    assert "Wire /shenanigans." in response.text
    assert "original edit apply status" in response.text
    assert "original proposal" in response.text


def test_apply_requires_confirmation_and_uses_validated_existing_applier(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[object] = []
    bundle_path = root / freestyle.DEV_STATE_DIR_NAME / "bundle.json"

    def fake_required_bundle(current_job, state_root):
        calls.append(("validate", current_job.job_id, state_root))
        return bundle_path

    def fake_apply(job_id: str, path: str, **kwargs):
        calls.append(("apply", job_id, path, kwargs))
        return job

    monkeypatch.setattr(freestyle_web, "_required_bundle_path", fake_required_bundle)
    monkeypatch.setattr(freestyle_web, "apply_freestyle_edit_bundle", fake_apply)
    client = _client(tmp_path, root)

    refused = client.post(f"/dev/freestyle/{job.job_id}/apply")
    assert "Explicit confirmation is required" in refused.text
    assert calls == []

    rejected = client.post(
        f"/dev/freestyle/{job.job_id}/apply",
        data={
            "confirm_apply": "apply-latest-generated",
            "bundle_path": "/tmp/untrusted.json",
        },
    )
    assert "Apply only accepts explicit confirmation" in rejected.text
    assert calls == []

    response = client.post(
        f"/dev/freestyle/{job.job_id}/apply",
        data={"confirm_apply": "apply-latest-generated"},
    )

    assert "Applied the latest validated generated bundle" in response.text
    assert calls == [
        ("validate", job.job_id, root / freestyle.DEV_STATE_DIR_NAME),
        ("apply", job.job_id, str(bundle_path), {"project_root": root}),
    ]


def test_apply_preflight_refusal_never_reaches_applier(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[str] = []

    def refuse_bundle(*args, **kwargs):
        raise FreestyleJobRefusedError("The latest generated bundle is invalid.")

    def forbidden_apply(*args, **kwargs):
        calls.append("applied")
        raise AssertionError("invalid generated bundle must not reach the applier")

    monkeypatch.setattr(freestyle_web, "_required_bundle_path", refuse_bundle)
    monkeypatch.setattr(freestyle_web, "apply_freestyle_edit_bundle", forbidden_apply)

    response = _client(tmp_path, root).post(
        f"/dev/freestyle/{job.job_id}/apply",
        data={"confirm_apply": "apply-latest-generated"},
    )

    assert "latest generated bundle is invalid" in response.text
    assert calls == []


def test_review_calls_deterministic_reviewer_only(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[tuple[str, Path]] = []

    def fake_review(job_id: str, *, project_root: Path):
        calls.append((job_id, project_root))
        return job

    def forbidden(*args, **kwargs):
        raise AssertionError("review route must not call orchestration or a model path")

    monkeypatch.setattr(freestyle_web, "review_freestyle_job", fake_review)
    monkeypatch.setattr(freestyle_web, "run_freestyle_job", forbidden)

    response = _client(tmp_path, root).post(
        f"/dev/freestyle/{job.job_id}/review"
    )

    assert "Created a deterministic review" in response.text
    assert calls == [(job.job_id, root)]


def test_run_checks_accepts_only_static_backend_ids_before_execution(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[tuple[str, tuple[str, ...], Path]] = []

    def fake_checks(job_id: str, *, check_ids, project_root: Path):
        calls.append((job_id, check_ids, project_root))
        return job

    monkeypatch.setattr(freestyle_web, "run_freestyle_checks", fake_checks)
    client = _client(tmp_path, root)

    for value in ("git push --force", "pytest-freestyle; rm -rf /", ""):
        refused = client.post(
            f"/dev/freestyle/{job.job_id}/run-checks",
            data={"check_id": value},
        )
        assert "only static backend check IDs" in refused.text
        assert calls == []

    response = client.post(
        f"/dev/freestyle/{job.job_id}/run-checks",
        content="check_id=pytest-freestyle&check_id=ruff-check",
        headers={"content-type": "application/x-www-form-urlencoded"},
    )

    assert "Completed allowlisted checks: pytest-freestyle, ruff-check." in response.text
    assert calls == [
        (job.job_id, ("pytest-freestyle", "ruff-check"), root)
    ]


def test_cleanup_requires_confirmation_and_separate_discard_confirmation(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[dict[str, object]] = []

    def fake_cleanup(job_id: str, **kwargs):
        calls.append({"job_id": job_id, **kwargs})
        return {"artifact_path": f"maintenance/{job_id}/cleanup-result.json"}

    monkeypatch.setattr(freestyle_web, "cleanup_freestyle_job", fake_cleanup)
    client = _client(tmp_path, root)

    refused = client.post(f"/dev/freestyle/{job.job_id}/cleanup")
    assert "Explicit confirmation is required" in refused.text
    assert calls == []

    bad_discard = client.post(
        f"/dev/freestyle/{job.job_id}/cleanup",
        data={
            "confirm_cleanup": "cleanup",
            "discard_worktree_changes": "yes",
        },
    )
    assert "separate explicit confirmation" in bad_discard.text
    assert calls == []

    clean = client.post(
        f"/dev/freestyle/{job.job_id}/cleanup",
        data={"confirm_cleanup": "cleanup"},
    )
    discarded = client.post(
        f"/dev/freestyle/{job.job_id}/cleanup",
        data={
            "confirm_cleanup": "cleanup",
            "discard_worktree_changes": "discard",
        },
    )

    assert "Result artifact: maintenance/" in clean.text
    assert "Result artifact: maintenance/" in discarded.text
    assert [call["discard_worktree_changes"] for call in calls] == [False, True]
    assert all(call["confirm"] is True for call in calls)


def test_recover_requires_confirmation_and_stronger_inconsistent_confirmation(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[dict[str, object]] = []

    def fake_recover(job_id: str, **kwargs):
        calls.append({"job_id": job_id, **kwargs})
        return job

    monkeypatch.setattr(freestyle_web, "recover_freestyle_job", fake_recover)
    client = _client(tmp_path, root)

    refused = client.post(f"/dev/freestyle/{job.job_id}/recover")
    assert "Explicit confirmation is required" in refused.text
    assert calls == []

    ordinary = client.post(
        f"/dev/freestyle/{job.job_id}/recover",
        data={"confirm_recover": "recover"},
    )
    assert "no work was marked successful" in ordinary.text
    assert calls == [
        {
            "job_id": job.job_id,
            "confirm": True,
            "confirm_inconsistent": False,
            "project_root": root,
        }
    ]
    calls.clear()

    _set_metadata(root, job.job_id, edit_apply_status="inconsistent")
    weak = client.post(
        f"/dev/freestyle/{job.job_id}/recover",
        data={"confirm_recover": "recover"},
    )
    assert "manually verified" in weak.text
    assert calls == []

    strong = client.post(
        f"/dev/freestyle/{job.job_id}/recover",
        data={
            "confirm_recover": "recover",
            "confirm_inconsistent": "verified-inconsistent",
        },
    )

    assert "no work was marked successful" in strong.text
    assert calls == [
        {
            "job_id": job.job_id,
            "confirm": True,
            "confirm_inconsistent": True,
            "project_root": root,
        }
    ]


def test_missing_deepseek_error_is_inline_without_traceback_or_secret(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _job(root)
    secret = "super-secret-deepseek-key"

    def refused(*args, **kwargs):
        raise FreestyleJobRefusedError(
            f"DeepSeek proposal configuration is missing; key={secret}"
        )

    monkeypatch.setattr(freestyle_web, "run_freestyle_job", refused)
    settings = replace(_settings(tmp_path, debug=True), deepseek_api_key=secret)
    app = create_app(settings)
    app.state.freestyle_project_root = root
    client = TestClient(app)

    response = client.post(f"/dev/freestyle/{job.job_id}/run")

    assert response.status_code == 200
    assert "Freestyle action stopped" in response.text
    assert "DeepSeek proposal configuration is missing" in response.text
    assert "[REDACTED]" in response.text
    assert secret not in response.text
    assert "Traceback" not in response.text
    assert "Raw backend error" in response.text


def test_detail_view_does_not_mutate_source_or_job_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    source_before = _source_snapshot(root)
    metadata_before = _metadata_path(root, job.job_id).read_bytes()

    response = _client(tmp_path, root).get(f"/dev/freestyle/{job.job_id}")

    assert response.status_code == 200
    assert _source_snapshot(root) == source_before
    assert _metadata_path(root, job.job_id).read_bytes() == metadata_before
