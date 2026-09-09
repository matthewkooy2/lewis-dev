from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from app.config import Settings
import app.dev.coding_agents as agents
from app.dev.coding_agents import (
    AgentInvocation,
    CodingAgentError,
    CodingAgentService,
    CodingAgentStore,
    LaunchRequest,
    ProcessResult,
    SafeProcessRunner,
)


def _settings(
    tmp_path: Path, *, repos=(), codex_path=None, claude_path=None
) -> Settings:
    documents = tmp_path / "documents"
    documents.mkdir(exist_ok=True)
    return Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=documents,
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
        deepseek_code_model="deepseek-code-test",
        coding_agents_web_ui=True,
        coding_agent_repositories=tuple(repos),
        codex_cli_path=codex_path,
        claude_cli_path=claude_path,
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
    root = tmp_path / "repo"
    root.mkdir()
    (root / "README.md").write_text("hello\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _repo_option(settings: Settings):
    options = agents.configured_repository_options(settings)
    assert len(options) == 1
    return options[0]


class StubRunner:
    def __init__(self, lines: list[str], result: ProcessResult | None = None):
        self.lines = lines
        self.result = result or ProcessResult(returncode=0, duration_seconds=0.01)
        self.invocations: list[AgentInvocation] = []

    def run(
        self, invocation, *, on_stdout_line, on_stderr_line=None, cancel_event=None
    ):
        self.invocations.append(invocation)
        for line in self.lines:
            try:
                on_stdout_line(line)
            except Exception as exc:
                return ProcessResult(
                    returncode=None,
                    duration_seconds=0.01,
                    callback_error=str(exc),
                )
        return self.result


def _wait_job(store: CodingAgentStore, job_id: str):
    for _ in range(100):
        job = store.get_job(job_id)
        if job and job.state not in {"queued", "running"}:
            return job
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_codex_jsonl_normalization_and_invocation(tmp_path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    runner = StubRunner(
        [
            json.dumps({"type": "thread.started", "thread_id": "codex-thread-1"})
            + "\n",
            json.dumps({"type": "agent_message", "message": "Read the repo."}) + "\n",
            json.dumps({"type": "command.started", "command": ["rg", "TODO"]}) + "\n",
            json.dumps(
                {"type": "command.finished", "command": "rg TODO", "exit_code": 0}
            )
            + "\n",
            json.dumps({"type": "file.read", "path": "README.md"}) + "\n",
            json.dumps(
                {
                    "type": "usage",
                    "usage": {
                        "input_tokens": 10,
                        "cached_input_tokens": 3,
                        "output_tokens": 4,
                        "output_tokens_details": {"reasoning_tokens": 2},
                    },
                }
            )
            + "\n",
        ]
    )
    service = CodingAgentService(
        settings,
        runner=runner,  # type: ignore[arg-type]
        executable_resolver=lambda provider: Path("codex.exe"),
    )

    created = service.launch(
        LaunchRequest(
            repository_id=_repo_option(settings).repo_id,
            provider="codex",
            goal="Analyze only.",
            model="gpt-5.5",
        )
    )
    job = _wait_job(service.store, created.id)

    assert job.state == "completed"
    assert job.external_session_id == "codex-thread-1"
    assert job.input_tokens == 10
    assert job.cached_input_tokens == 3
    assert job.output_tokens == 4
    assert job.reasoning_tokens == 2
    assert job.token_usage_source == "reported"
    invocation = runner.invocations[0]
    assert invocation.args[:5] == (
        "codex.exe",
        "exec",
        "--json",
        "--sandbox",
        "read-only",
    )
    assert "--ask-for-approval" not in invocation.args
    assert "workspace-write" not in invocation.args
    assert "danger-full-access" not in invocation.args
    assert invocation.cwd == repo.resolve()
    event_types = [event.event_type for event in service.store.list_events(created.id)]
    assert {
        "job_started",
        "agent_message",
        "command_started",
        "command_finished",
        "file_observed",
        "usage_updated",
        "job_completed",
    } <= set(event_types)


def test_claude_stream_normalization_and_invocation(tmp_path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    runner = StubRunner(
        [
            json.dumps(
                {
                    "type": "system",
                    "subtype": "init",
                    "session_id": "00000000-0000-4000-8000-000000000001",
                    "model": "claude-fable-5",
                }
            )
            + "\n",
            json.dumps(
                {
                    "type": "assistant",
                    "session_id": "00000000-0000-4000-8000-000000000001",
                    "message": {
                        "content": [
                            {"type": "text", "text": "I can inspect this."},
                            {
                                "type": "tool_use",
                                "id": "toolu_1",
                                "name": "Read",
                                "input": {"file_path": "README.md"},
                            },
                        ],
                        "usage": {
                            "input_tokens": 5,
                            "cache_read_input_tokens": 7,
                            "output_tokens": 3,
                        },
                    },
                }
            )
            + "\n",
            json.dumps(
                {
                    "type": "result",
                    "subtype": "success",
                    "session_id": "00000000-0000-4000-8000-000000000001",
                    "duration_ms": 1200,
                    "total_cost_usd": 0.01,
                    "result": "Done.",
                }
            )
            + "\n",
        ]
    )
    service = CodingAgentService(
        settings,
        runner=runner,  # type: ignore[arg-type]
        executable_resolver=lambda provider: Path("claude.exe"),
    )

    created = service.launch(
        LaunchRequest(
            repository_id=_repo_option(settings).repo_id,
            provider="claude",
            goal="Analyze only.",
            model="claude-fable-5",
            effort="high",
        )
    )
    job = _wait_job(service.store, created.id)

    assert job.state == "completed"
    assert job.external_session_id == "00000000-0000-4000-8000-000000000001"
    assert job.input_tokens == 5
    assert job.cached_input_tokens == 7
    assert job.output_tokens == 3
    assert job.estimated_cost_usd == 0.01
    invocation = runner.invocations[0]
    assert invocation.args[:4] == ("claude.exe", "-p", "--output-format", "stream-json")
    assert "--verbose" in invocation.args
    assert "plan" in invocation.args
    assert "MultiEdit" not in " ".join(invocation.args)
    assert "--dangerously-skip-permissions" not in invocation.args
    assert "--tools" in invocation.args
    assert "Read,Grep,Glob,LS" in invocation.args
    assert "--effort" in invocation.args


def test_malformed_output_fails_job_without_real_model_call(tmp_path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    service = CodingAgentService(
        settings,
        runner=StubRunner(["not json\n"]),  # type: ignore[arg-type]
        executable_resolver=lambda provider: Path("codex.exe"),
    )

    created = service.launch(
        LaunchRequest(
            repository_id=_repo_option(settings).repo_id,
            provider="codex",
            goal="Analyze.",
        )
    )
    job = _wait_job(service.store, created.id)

    assert job.state == "failed"
    assert job.failure_category == "malformed_output"
    assert "malformed JSON" in (job.last_error or "")


def test_usage_persistence_preserves_reported_values_only(tmp_path):
    store = CodingAgentStore(tmp_path / "usage.sqlite3")
    now = agents._utc_now()
    job = agents.CodingAgentJob(
        id="1" * 32,
        provider="codex",
        repository_root=str(tmp_path),
        repository_label="repo",
        goal="goal",
        model=None,
        effort=None,
        max_turns=1,
        timeout_seconds=60,
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
    store.create_job(job)
    store.update_usage(
        "1" * 32,
        agents.UsageValues(
            input_tokens=1, output_tokens=2, token_usage_source="reported"
        ),
    )

    saved = store.get_job("1" * 32)
    assert saved is not None
    assert saved.input_tokens == 1
    assert saved.output_tokens == 2
    assert saved.cached_input_tokens is None
    assert saved.estimated_cost_usd is None
    assert saved.token_usage_source == "reported"


def test_repository_allowlist_enforcement(tmp_path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    selected = _repo_option(settings)

    approved = agents.repository_by_id(settings, selected.repo_id)
    assert approved.root == repo.resolve()

    with pytest.raises(agents.RepositoryRefusedError):
        agents.repository_by_id(settings, "not-approved")

    for generated_name in (".lewis-dev", "runtime"):
        generated_nested = repo / generated_name / "fixtures" / "nested"
        generated_nested.mkdir(parents=True)
        (generated_nested / ".git").mkdir()
        assert (
            agents.repository_by_id(settings, selected.repo_id).root == repo.resolve()
        )

    nested = repo / "nested"
    nested.mkdir()
    (nested / ".git").mkdir()
    with pytest.raises(agents.RepositoryRefusedError, match="Nested Git"):
        agents.repository_by_id(settings, selected.repo_id)


def test_provider_status_uses_documented_fixed_commands(tmp_path):
    codex = tmp_path / ("codex.exe" if os.name == "nt" else "codex")
    claude = tmp_path / ("claude.exe" if os.name == "nt" else "claude")
    codex.write_text("", encoding="utf-8")
    claude.write_text("", encoding="utf-8")
    calls: list[list[str]] = []

    def status_runner(args, cwd):
        calls.append(list(args))
        command = list(args)
        if command[0] == str(codex) and command[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=b"codex-cli 0.test\n", stderr=b""
            )
        if command[0] == str(codex) and command[1:] == ["login", "status"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=b"Logged in using ChatGPT\n", stderr=b""
            )
        if command[0] == str(claude) and command[1:] == ["--version"]:
            return subprocess.CompletedProcess(
                command, 0, stdout=b"2.test (Claude Code)\n", stderr=b""
            )
        if command[0] == str(claude) and command[1:] == ["auth", "status", "--json"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=b'{"loggedIn": true, "authMethod": "claude.ai", "subscriptionType": "pro"}',
                stderr=b"",
            )
        raise AssertionError(command)

    settings = _settings(tmp_path, codex_path=codex, claude_path=claude)
    service = CodingAgentService(settings, status_runner=status_runner)

    statuses = service.refresh_provider_statuses()

    assert statuses["codex"].authenticated is True
    assert statuses["claude"].authenticated is True
    assert "usage" in statuses["codex"].usage_status
    assert "unavailable" in statuses["claude"].usage_status
    assert calls == [
        [str(codex.resolve()), "--version"],
        [str(codex.resolve()), "login", "status"],
        [str(claude.resolve()), "--version"],
        [str(claude.resolve()), "auth", "status", "--json"],
    ]


def test_safe_invocation_rejects_write_capable_modes(tmp_path):
    with pytest.raises(CodingAgentError):
        agents._assert_safe_invocation(
            "codex", ("codex", "exec", "--json", "--sandbox", "workspace-write")
        )
    with pytest.raises(CodingAgentError):
        agents._assert_safe_invocation(
            "codex", ("codex", "exec", "--json", "--sandbox", "danger-full-access")
        )
    with pytest.raises(CodingAgentError):
        agents._assert_safe_invocation(
            "claude",
            (
                "claude",
                "-p",
                "--output-format",
                "stream-json",
                "--permission-mode",
                "bypassPermissions",
                "--tools",
                "default",
            ),
        )
    with pytest.raises(CodingAgentError):
        agents._validated_launch_request(
            LaunchRequest(
                repository_id="x",
                provider="claude",
                goal="g",
                model="bad model; rm",
                effort="high",
            ),
            _settings(tmp_path),
        )


def test_process_timeout_cancellation_and_output_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(agents, "_assert_safe_invocation", lambda provider, args: None)
    script = tmp_path / "slow.py"
    script.write_text(
        'import sys, time\nprint(\'{\\"type\\":\\"agent_message\\",\\"message\\":\\"start\\"}\', flush=True)\ntime.sleep(5)\n',
        encoding="utf-8",
    )
    runner = SafeProcessRunner()
    invocation = AgentInvocation(
        provider="codex",
        args=(sys.executable, str(script)),
        cwd=tmp_path,
        stdin_text="",
        timeout_seconds=1,
    )
    lines: list[str] = []
    result = runner.run(invocation, on_stdout_line=lines.append)
    assert result.timed_out is True
    assert lines

    cancel_event = agents.threading.Event()
    script.write_text("import time\ntime.sleep(5)\n", encoding="utf-8")
    invocation = AgentInvocation(
        "codex", (sys.executable, str(script)), tmp_path, "", 10
    )

    def cancel_soon():
        time.sleep(0.2)
        cancel_event.set()

    agents.threading.Thread(target=cancel_soon).start()
    result = runner.run(
        invocation, on_stdout_line=lambda line: None, cancel_event=cancel_event
    )
    assert result.cancelled is True

    script.write_text("print('x' * 10000)\n", encoding="utf-8")
    invocation = AgentInvocation(
        "codex", (sys.executable, str(script)), tmp_path, "", 10, max_output_bytes=100
    )
    result = runner.run(invocation, on_stdout_line=lambda line: None)
    assert result.output_limited is True
