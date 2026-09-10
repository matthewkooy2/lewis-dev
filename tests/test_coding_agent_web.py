from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.config import Settings
import app.dev.coding_agents as agents
from app.dev.coding_agents import ApprovedRepository, LaunchRequest, ProviderStatus
from app.web import create_app


def _settings(tmp_path: Path, *, enabled: bool, debug: bool = False) -> Settings:
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
        coding_agents_web_ui=enabled,
        coding_agent_repositories=(tmp_path / "repo",),
    )


class FakeStore:
    def __init__(self):
        self.jobs = []
        self.events = []

    def list_jobs(self):
        return self.jobs

    def get_job(self, job_id):
        for job in self.jobs:
            if job.id == job_id:
                return job
        return None

    def list_events(self, job_id):
        return self.events


class FakeService:
    def __init__(self, tmp_path: Path):
        self.store = FakeStore()
        self.launches: list[LaunchRequest] = []
        self.cancelled: list[str] = []
        self.refreshed = 0
        self.repo = ApprovedRepository("repo1", tmp_path / "repo", "repo")

    def repositories(self):
        return [self.repo]

    def provider_statuses(self):
        return {
            "codex": ProviderStatus(
                "codex", installed=True, executable_path="codex.exe"
            ),
            "claude": ProviderStatus("claude", installed=False),
        }

    def refresh_provider_statuses(self):
        self.refreshed += 1
        return self.provider_statuses()

    def launch(self, request: LaunchRequest):
        self.launches.append(request)
        now = agents._utc_now()
        job = agents.CodingAgentJob(
            id="a" * 32,
            provider=request.provider,
            repository_root=str(self.repo.root),
            repository_label=self.repo.label,
            goal=request.goal,
            model=request.model,
            effort=request.effort,
            max_turns=request.max_turns,
            timeout_seconds=request.timeout_seconds,
            external_session_id=None,
            state="queued",
            created_at=now,
            started_at=None,
            completed_at=None,
            updated_at=now,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            reasoning_tokens=None,
            token_usage_source=None,
            estimated_cost_usd=None,
            cost_source=None,
            duration_seconds=None,
            failure_category=None,
            cancellation_requested=False,
            cancellation_requested_at=None,
            last_error=None,
        )
        self.store.jobs.append(job)
        return job

    def cancel(self, job_id: str):
        self.cancelled.append(job_id)


def _client_with_fake_service(tmp_path: Path):
    app = create_app(_settings(tmp_path, enabled=True))
    fake = FakeService(tmp_path)
    app.state.coding_agent_service = fake
    return TestClient(
        base_url="http://127.0.0.1:8787",
        headers={"Origin": "http://127.0.0.1:8787"},
        app=app,
    ), fake


def test_coding_agents_page_requires_feature_flag(tmp_path):
    normal = TestClient(
        base_url="http://127.0.0.1:8787",
        headers={"Origin": "http://127.0.0.1:8787"},
        app=create_app(_settings(tmp_path, enabled=False, debug=False)),
    )
    enabled = TestClient(
        base_url="http://127.0.0.1:8787",
        headers={"Origin": "http://127.0.0.1:8787"},
        app=create_app(_settings(tmp_path, enabled=True, debug=False)),
    )
    debug = TestClient(
        base_url="http://127.0.0.1:8787",
        headers={"Origin": "http://127.0.0.1:8787"},
        app=create_app(_settings(tmp_path, enabled=False, debug=True)),
    )

    assert normal.get("/dev/coding-agents").status_code == 404
    assert "/dev/coding-agents" not in normal.get("/dashboard").text
    assert enabled.get("/dev/coding-agents").status_code == 200
    assert "/dev/coding-agents" in enabled.get("/dashboard").text
    assert debug.get("/dev/coding-agents").status_code == 200


def test_opening_page_does_not_launch_process_or_job(tmp_path, monkeypatch):
    def forbidden_run(*args, **kwargs):
        raise AssertionError("GET /dev/coding-agents must not launch subprocesses")

    monkeypatch.setattr(agents.subprocess, "run", forbidden_run)
    client, fake = _client_with_fake_service(tmp_path)

    response = client.get("/dev/coding-agents")

    assert response.status_code == 200
    assert "Read-only analysis and proposal jobs" in response.text
    assert fake.launches == []
    assert fake.refreshed == 0


def test_launch_requires_confirmation_and_rejects_arbitrary_fields(tmp_path):
    client, fake = _client_with_fake_service(tmp_path)

    refused = client.post(
        "/dev/coding-agents/launch",
        data={"repository_id": "repo1", "provider": "codex", "goal": "Analyze."},
    )
    assert "Explicit confirmation is required" in refused.text
    assert fake.launches == []

    unsafe = client.post(
        "/dev/coding-agents/launch",
        data={
            "repository_id": "repo1",
            "provider": "codex",
            "goal": "Analyze.",
            "launch_confirm": "read-only-analysis",
            "command": "git push",
            "cwd": "C:\\",
            "env": "TOKEN=x",
            "executable": "powershell",
        },
    )
    assert "bounded Coding Agents form fields" in unsafe.text
    assert fake.launches == []

    launched = client.post(
        "/dev/coding-agents/launch",
        data={
            "repository_id": "repo1",
            "provider": "claude",
            "goal": "Analyze only.",
            "model": "claude-fable-5",
            "effort": "high",
            "max_turns": "3",
            "timeout_seconds": "120",
            "launch_confirm": "read-only-analysis",
        },
        follow_redirects=False,
    )
    assert launched.status_code == 303
    assert len(fake.launches) == 1
    request = fake.launches[0]
    assert request.provider == "claude"
    assert request.model == "claude-fable-5"
    assert request.effort == "high"
    assert request.max_turns == 3
    assert request.timeout_seconds == 120


def test_status_refresh_and_stop_require_bounded_forms(tmp_path):
    client, fake = _client_with_fake_service(tmp_path)
    fake.launch(
        LaunchRequest(repository_id="repo1", provider="codex", goal="Existing job")
    )

    rejected_status = client.post(
        "/dev/coding-agents/status", data={"command": "claude auth status"}
    )
    assert "only its confirmation field" in rejected_status.text
    assert fake.refreshed == 0

    refreshed = client.post(
        "/dev/coding-agents/status",
        data={"confirm_status": "refresh-provider-status"},
        follow_redirects=False,
    )
    assert refreshed.status_code == 303
    assert fake.refreshed == 1

    rejected_stop = client.post(f"/dev/coding-agents/{'a' * 32}/stop")
    assert "Explicit stop confirmation" in rejected_stop.text
    assert fake.cancelled == []

    stopped = client.post(
        f"/dev/coding-agents/{'a' * 32}/stop",
        data={"confirm_stop": "stop-running-job"},
        follow_redirects=False,
    )
    assert stopped.status_code == 303
    assert fake.cancelled == ["a" * 32]
