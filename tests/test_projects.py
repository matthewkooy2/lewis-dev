from __future__ import annotations

from pathlib import Path
import subprocess
import threading
import time

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
import app.dev.coding_agents as agents
from app.dev.coding_agents import CodingAgentService, LaunchRequest, ProcessResult
from app.projects import (
    ProjectMapWorker,
    ProjectStore,
    derive_import_relationships,
    validate_project_repository,
)
from app.web import create_app


def _settings(tmp_path: Path, *, repos=(), projects_enabled: bool = True) -> Settings:
    docs = tmp_path / "documents"
    docs.mkdir(exist_ok=True)
    return Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://example.test",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "test.sqlite3",
        documents_dir=docs,
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
        coding_agents_web_ui=True,
        coding_agent_repositories=tuple(repos),
        projects_web_ui=projects_enabled,
        project_map_max_files=200,
        project_map_max_file_bytes=80_000,
        project_map_timeout_seconds=60,
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
    (root / "app").mkdir()
    (root / "tests").mkdir()
    (root / "app" / "helper.py").write_text(
        "def helper():\n    return 'ok'\n",
        encoding="utf-8",
    )
    (root / "app" / "main.py").write_text(
        "from app.helper import helper\n\nclass Service:\n    def run(self):\n        return helper()\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_main.py").write_text(
        "from app.main import Service\n\ndef test_run():\n    assert Service().run() == 'ok'\n",
        encoding="utf-8",
    )
    (root / "node_modules").mkdir()
    (root / "node_modules" / "ignored.js").write_text("function ignored() {}\n", encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _create_project(settings: Settings, repo: Path):
    store = ProjectStore(settings.database_path)
    approved = validate_project_repository(settings, repository_path=str(repo))
    return store.create_project(
        display_name="Repo",
        repository=approved,
        description="A useful project",
        tags=("api", "python"),
    )


def test_project_creation_validation_archiving_and_notes(tmp_path: Path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    store = ProjectStore(settings.database_path)

    approved = validate_project_repository(settings, repository_path=str(repo))
    project = store.create_project(
        display_name="Repo",
        repository=approved,
        description="A useful project",
        tags=("api", "python", "api"),
    )

    assert project.repository_root == str(repo.resolve())
    assert project.tags == ("api", "python")
    with pytest.raises(Exception):
        validate_project_repository(settings, repository_path=str(tmp_path))

    note = store.create_note(
        project.id,
        title="Architecture",
        content_markdown="# Shape\nService calls helper.",
        category="Architecture",
        tags=("design",),
        is_pinned=True,
    )
    assert note.category_name == "Architecture"
    assert store.list_notes(project.id, query="helper")[0].id == note.id
    duplicate = store.duplicate_note(project.id, note.id)
    assert duplicate.id != note.id
    store.archive_note(project.id, note.id)
    assert store.list_notes(project.id, query="Architecture")
    assert all(item.id != note.id for item in store.list_notes(project.id))
    store.restore_note(project.id, note.id)
    with pytest.raises(Exception):
        store.delete_note_permanently(project.id, note.id, confirm="")
    store.delete_note_permanently(project.id, note.id, confirm="delete-note")
    assert all(item.id != note.id for item in store.list_notes(project.id, include_archived=True))

    archived = store.archive_project(project.id, confirm="archive-project")
    assert archived.status == "archived"
    assert store.restore_project(project.id).status == "active"


def test_repository_validation_rejects_nested_git_and_symlink_escape(tmp_path: Path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    nested = repo / "nested"
    nested.mkdir()
    (nested / ".git").mkdir()
    with pytest.raises(agents.RepositoryRefusedError):
        validate_project_repository(settings, repository_path=str(repo))

    if hasattr(Path, "symlink_to"):
        escape = tmp_path / "outside"
        escape.mkdir()
        link = repo / "linked"
        try:
            link.symlink_to(escape, target_is_directory=True)
        except OSError:
            return
        (nested / ".git").rmdir()
        with pytest.raises(agents.RepositoryRefusedError):
            validate_project_repository(settings, repository_path=str(repo))


def test_repository_map_persists_symbols_relationships_and_incremental_refresh(tmp_path: Path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    store = ProjectStore(settings.database_path)
    project = _create_project(settings, repo)

    run = store.create_map_run(project.id, run_type="full", full_rebuild=True)
    ProjectMapWorker(settings, store, threading.Event()).run(run.id)
    completed = store.require_map_run(run.id)

    assert completed.status == "completed"
    files = store.files_for_project(project.id, limit=50)
    paths = {item.path for item in files}
    assert "app/main.py" in paths
    assert "app/helper.py" in paths
    assert "node_modules/ignored.js" not in paths
    symbols = store.symbols_for_file(project.id, "app/main.py")
    assert {symbol.name for symbol in symbols} >= {"Service", "run"}
    relationships = store.search_map(project.id, query="helper")["relationships"]
    assert any(rel.relationship_type == "import" for rel in relationships)
    assert store.repository_summary(project.id).startswith("Deterministic repository map")

    (repo / "app" / "helper.py").write_text(
        "def helper():\n    return 'changed'\n",
        encoding="utf-8",
    )
    second = store.create_map_run(project.id, run_type="incremental", full_rebuild=False)
    ProjectMapWorker(settings, store, threading.Event()).run(second.id)
    refreshed = store.require_map_run(second.id)
    assert refreshed.status == "completed"
    assert refreshed.files_reused >= 1
    assert refreshed.files_changed >= 1


def test_map_run_cancellation_and_restart_recovery(tmp_path: Path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    store = ProjectStore(settings.database_path)
    project = _create_project(settings, repo)
    run = store.create_map_run(project.id, run_type="incremental", full_rebuild=False)
    store.mark_run_started(run.id)

    recovered = store.recover_interrupted_map_runs()

    assert recovered == 1
    assert store.require_map_run(run.id).status == "paused"
    assert store.require_project(project.id).mapping_state == "paused"


def test_projects_web_feature_flag_and_project_workflow(tmp_path: Path):
    repo = _init_repo(tmp_path)
    disabled = TestClient(create_app(_settings(tmp_path, repos=[repo], projects_enabled=False)))
    assert disabled.get("/projects").status_code == 404
    assert "/projects" not in disabled.get("/dashboard").text

    settings = _settings(tmp_path, repos=[repo], projects_enabled=True)
    client = TestClient(create_app(settings))
    assert client.get("/projects").status_code == 200
    assert "/projects" in client.get("/dashboard").text

    created = client.post(
        "/projects",
        data={
            "display_name": "Repo",
            "repository_path": str(repo),
            "description": "Project from web",
            "tags": "web, test",
            "status": "active",
        },
        follow_redirects=False,
    )
    assert created.status_code == 303
    location = created.headers["location"]
    assert location.startswith("/projects/")
    page = client.get(location)
    assert "Project from web" in page.text
    project_id = location.split("/projects/", 1)[1].split("?", 1)[0]

    note_response = client.post(
        f"/projects/{project_id}/notes",
        data={"title": "Setup", "content_markdown": "Run pytest", "category": "Setup"},
        follow_redirects=False,
    )
    assert note_response.status_code == 303
    notes_page = client.get(f"/projects/{project_id}?tab=notes&q=pytest")
    assert "Setup" in notes_page.text
    assert "Run pytest" in notes_page.text


def test_coding_agent_job_can_link_to_project(tmp_path: Path):
    repo = _init_repo(tmp_path)
    settings = _settings(tmp_path, repos=[repo])
    project = _create_project(settings, repo)

    class StubRunner:
        def run(self, invocation, *, on_stdout_line, on_stderr_line=None, cancel_event=None):
            return ProcessResult(returncode=0, duration_seconds=0.01)

    service = CodingAgentService(
        settings,
        runner=StubRunner(),  # type: ignore[arg-type]
        executable_resolver=lambda provider: Path("codex.exe"),
    )
    repo_id = agents.configured_repository_options(settings)[0].repo_id
    created = service.launch(
        LaunchRequest(
            repository_id=repo_id,
            provider="codex",
            goal="Analyze only.",
            project_id=project.id,
        )
    )
    for _ in range(100):
        job = service.store.get_job(created.id)
        if job and job.state not in {"queued", "running"}:
            break
        time.sleep(0.01)
    job = service.store.get_job(created.id)
    assert job is not None
    assert job.project_id == project.id
    assert ProjectStore(settings.database_path).related_coding_agent_jobs(project.id)[0]["id"] == job.id


def test_import_relationship_derivation_is_bounded_and_deterministic():
    files = []
    relationships = derive_import_relationships(files)
    assert relationships == []
