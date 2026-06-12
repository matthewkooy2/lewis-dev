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
    assert "/apply" not in response.text
    assert "/cleanup" not in response.text
    assert "/recover" not in response.text
    assert "/run-checks" not in response.text


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


def test_actions_reject_arbitrary_fields_and_forbidden_routes_do_not_exist(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = _job(root)
    calls: list[str] = []

    def forbidden(*args, **kwargs):
        calls.append("called")
        raise AssertionError("bounded action must not run")

    monkeypatch.setattr(freestyle_web, "run_freestyle_job", forbidden)
    client = _client(tmp_path, root)

    rejected = client.post(
        f"/dev/freestyle/{job.job_id}/run",
        data={
            "command": "git push",
            "args": "--force",
            "env": "SECRET=x",
            "provider": "other",
            "file_path": "/tmp/file",
        },
    )

    assert "Default run accepts no input fields." in rejected.text
    assert calls == []
    malformed = client.post(
        f"/dev/freestyle/{job.job_id}/run",
        content=b"{not-json",
        headers={"content-type": "application/json"},
    )
    assert "Default run accepts no input fields." in malformed.text
    assert calls == []
    assert client.get("/dev/freestyle?command=git+push").status_code == 400
    for suffix in ("apply", "run-checks", "cleanup", "recover"):
        assert client.post(f"/dev/freestyle/{job.job_id}/{suffix}").status_code == 404


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


def test_detail_view_does_not_mutate_source_or_job_metadata(tmp_path):
    root = _init_repo(tmp_path)
    job = _job(root)
    source_before = _source_snapshot(root)
    metadata_before = _metadata_path(root, job.job_id).read_bytes()

    response = _client(tmp_path, root).get(f"/dev/freestyle/{job.job_id}")

    assert response.status_code == 200
    assert _source_snapshot(root) == source_before
    assert _metadata_path(root, job.job_id).read_bytes() == metadata_before
