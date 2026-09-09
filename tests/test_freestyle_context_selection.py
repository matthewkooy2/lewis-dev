"""Integration tests for goal-aware freestyle context selection and goal fit.

These tests exercise the deterministic context planner through the real
proposal/edit-generation slices with fake DeepSeek clients only: a web/UI
goal must ground its model context in actual web files, sensitive paths must
stay excluded, artifacts must record the context plan, and a "valid but
useless" bundle must surface advisory goal-fit warnings without blocking the
deterministic applier.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from fastapi.testclient import TestClient

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_proposal as proposal
from app.config import Settings
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    apply_freestyle_edit_bundle,
    create_freestyle_job,
    generate_freestyle_edit_bundle,
    get_freestyle_handoff,
    get_freestyle_job,
    materialize_freestyle_job,
    propose_freestyle_job,
)
from app.dev.freestyle_context_plan import (
    GOAL_FIT_WARNING_CREATE_ONLY,
    GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT,
    PLAN_WARNING_NO_UI_FILES,
)
from app.dev.freestyle_handoff import render_freestyle_handoff_markdown
from app.dev.freestyle_policy import APPROVAL_STORE_CLASS_NAME
from app.dev.freestyle_proposal import DeepSeekProposalConfig
from app.web import create_app

_TETRIS_GOAL = (
    "Add a new tab under Overview called Shennanigans where I can play Tetris."
)

# Mirrors the real app/web_routes.py shape: it imports and uses the approval
# store for approval routes, but route wiring for ordinary pages happens in
# regions that never touch approval code. Assembled with the concatenated
# class name so this test file itself never contains the definition text.
_ROUTES_WITH_APPROVAL_REFERENCES = f"""from app.approvals import {APPROVAL_STORE_CLASS_NAME}
from app.approval_service import approve_approval
from app.web_pages import approvals_page, chat_page


def register_web_routes(app):
    @app.get("/chat")
    def chat(request):
        return _page("Chat", request, chat_page())

    @app.post("/approvals/{{approval_id}}/approve")
    def approve_item(request, approval_id):
        return approve_approval(_approval_store(request), approval_id)


def _approval_store(request):
    return {APPROVAL_STORE_CLASS_NAME}(request.app.state.settings.database_path)


def _page(title, request, content):
    return content
"""
_APPROVAL_STORE_DEFINITION = (
    f"class {APPROVAL_STORE_CLASS_NAME}:\n"
    "    def __init__(self, database_path):\n"
    "        self.database_path = database_path\n"
)

_WEB_FILES = {
    "app/web.py": "def create_app():\n    return 'app'\n",
    "app/web_pages.py": "def overview_page():\n    return '<nav>overview</nav>'\n",
    "app/web_components.py": "def nav_link(label):\n    return f'<a>{label}</a>'\n",
    "app/web_routes.py": "ROUTES = ['/overview']\n",
    "app/web_rules.py": "RULES = ['server rendered']\n",
    "app/static/lewis.css": ".shell { color: green; }\n",
    "app/static/chat.js": "console.log('chat');\n",
    "app/dev/freestyle_web.py": "def freestyle_page():\n    return 'freestyle'\n",
    "tests/test_web.py": "def test_web():\n    assert True\n",
    "tests/test_web_rules.py": "def test_rules():\n    assert True\n",
    "tests/test_freestyle_web.py": "def test_freestyle_web():\n    assert True\n",
}
_FREESTYLE_DECOY_FILES = {
    f"app/dev/freestyle_{name}.py": f"{name.upper()}_INTERNAL = 1\n"
    for name in (
        "coding",
        "proposal",
        "edit_bundle",
        "edit_generation",
        "checks",
        "review",
        "run",
        "maintenance",
        "handoff",
        "state",
    )
}


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    )
    return completed.stdout.decode().strip()


def _commit_repo(root: Path) -> Path:
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _init_web_repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    for relative, content in (
        {
            **_WEB_FILES,
            **_FREESTYLE_DECOY_FILES,
            "README.md": "Web project\n",
            "pyproject.toml": "[project]\nname = 'web-project'\n",
        }
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return _commit_repo(root)


def _init_plain_repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "app" / "dev").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "dev" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "tests" / "test_sample.py").write_text(
        "def test_value():\n    assert True\n", encoding="utf-8"
    )
    (root / "README.md").write_text("Sample project\n", encoding="utf-8")
    return _commit_repo(root)


def _config() -> DeepSeekProposalConfig:
    return DeepSeekProposalConfig(
        api_key="test-deepseek-key",
        model="deepseek-test-code",
        base_url=proposal.DEEPSEEK_API_BASE_URL,
    )


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            model="deepseek-test-code",
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            usage=SimpleNamespace(
                prompt_tokens=100, completion_tokens=50, total_tokens=150
            ),
        )


class _FakeDeepSeekClient:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def _proposal_content(likely_files: list[str]) -> str:
    return json.dumps(
        {
            "proposed_implementation_steps": [
                "Wire a Shennanigans entry into the overview navigation."
            ],
            "likely_files_to_edit": likely_files,
            "risks_unknowns": [],
            "suggested_allowlisted_checks": ["pytest"],
        }
    )


def _tetris_bundle_content(job_id: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": "2026-06-12T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "create_file",
                    "path": "templates/shennanigans.html",
                    "content": "<html><body>tetris</body></html>\n",
                }
            ],
        }
    )


def _wired_bundle_content(job_id: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": "2026-06-12T12:00:00+00:00",
            "operation_count": 2,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/web_pages.py",
                    "old_text": "<nav>overview</nav>",
                    "new_text": "<nav>overview shennanigans</nav>",
                },
                {
                    "op": "create_file",
                    "path": "app/static/tetris.js",
                    "content": "console.log('tetris');\n",
                },
            ],
        }
    )


def _materialized_job(root: Path, goal: str = _TETRIS_GOAL):
    job = create_freestyle_job(goal, developer_approved=True, project_root=root)
    return materialize_freestyle_job(job.job_id, project_root=root)


def _proposed_job(
    root: Path,
    goal: str = _TETRIS_GOAL,
    likely_files: list[str] | None = None,
):
    job = _materialized_job(root, goal)
    return propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeDeepSeekClient(
            _proposal_content(likely_files or ["app/web_pages.py"])
        ),
    )


def _read_state_artifact(root: Path, reference: str) -> dict:
    path = root / freestyle.DEV_STATE_DIR_NAME / reference
    return json.loads(path.read_text(encoding="utf-8"))


def test_ui_goal_context_prioritizes_web_files_over_freestyle_internals(tmp_path):
    root = _init_web_repo(tmp_path)
    job = _materialized_job(root)
    worktree = Path(job.worktree_path)

    context = proposal.build_proposal_context(worktree, job.goal)

    included = list(context.inspected_paths)
    for web_path in _WEB_FILES:
        assert web_path in included, web_path
    for decoy in _FREESTYLE_DECOY_FILES:
        assert decoy not in included, decoy
    plan = context.context_plan
    assert plan["detected_goal_categories"] == ["web_ui"]
    assert plan["priority_candidate_count"] == len(_WEB_FILES)
    assert sorted(plan["included_priority_files"]) == sorted(_WEB_FILES)
    assert plan["warnings"] == []


def test_tests_keyword_goal_keeps_core_web_files_ahead_of_broad_test_files(tmp_path):
    """Regression for the real Tetris/Shenanigans run: a goal mentioning tests
    detected the broad tests/*.py pattern and the generic test files crowded
    every core web implementation file out of the bounded context."""
    root = tmp_path / "project"
    generic_tests = {
        f"tests/test_generic_{index:02d}.py": f"def test_g{index}():\n    assert True\n"
        for index in range(10)
    }
    for relative, content in (
        {
            **_WEB_FILES,
            **generic_tests,
            "README.md": "Web project\n",
            "pyproject.toml": "[project]\nname = 'web-project'\n",
        }
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _commit_repo(root)
    job = _materialized_job(
        root,
        goal=_TETRIS_GOAL
        + " Keep it local/client-side and add or update appropriate tests.",
    )

    context = proposal.build_proposal_context(Path(job.worktree_path), job.goal)

    included = list(context.inspected_paths)
    for web_path in _WEB_FILES:
        assert web_path in included, web_path
    implementation_paths = [
        path for path in _WEB_FILES if not path.startswith("tests/")
    ]
    last_implementation_index = max(
        included.index(path) for path in implementation_paths
    )
    included_test_indexes = [
        index for index, path in enumerate(included) if path.startswith("tests/")
    ]
    assert included_test_indexes
    assert all(index > last_implementation_index for index in included_test_indexes)
    plan = context.context_plan
    assert plan["detected_goal_categories"] == ["web_ui", "tests"]
    assert plan["core_priority_patterns"]
    assert plan["test_priority_patterns"]
    assert "core implementation" in plan["ordering_policy"]
    omitted_priority_paths = [
        record["path"] for record in plan["omitted_priority_files"]
    ]
    assert omitted_priority_paths
    assert all(path.startswith("tests/") for path in omitted_priority_paths)


def test_sensitive_files_stay_excluded_even_when_goal_asks_for_them(tmp_path):
    root = _init_web_repo(tmp_path)
    job = _materialized_job(
        root,
        goal=(
            "Restyle the page using the values in .env, secret-token.txt, and "
            "the approvalstore notes css."
        ),
    )
    worktree = Path(job.worktree_path)
    (worktree / ".env").write_text("ENV_SECRET_MARKER=bad\n", encoding="utf-8")
    (worktree / "secret-token.txt").write_text("TOKEN_MARKER\n", encoding="utf-8")
    (worktree / "approvalstore-notes.txt").write_text(
        "APPROVAL_MARKER\n", encoding="utf-8"
    )

    context = proposal.build_proposal_context(worktree, job.goal)

    included = set(context.inspected_paths)
    assert ".env" not in included
    assert "secret-token.txt" not in included
    assert "approvalstore-notes.txt" not in included
    serialized = json.dumps(context.files)
    for marker in ("ENV_SECRET_MARKER", "TOKEN_MARKER", "APPROVAL_MARKER"):
        assert marker not in serialized
    assert context.omission_counts.get("safety", 0) >= 1
    assert context.omission_counts.get("approval_store", 0) >= 1
    assert "web_ui" in context.context_plan["detected_goal_categories"]


def _init_approval_referencing_web_repo(tmp_path: Path) -> Path:
    """Web repo whose web_routes.py references (not defines) approval code."""
    root = tmp_path / "project"
    for relative, content in (
        {
            **_WEB_FILES,
            "app/web_routes.py": _ROUTES_WITH_APPROVAL_REFERENCES,
            "app/legacy_store.py": _APPROVAL_STORE_DEFINITION,
            "tests/test_web_shenanigans.py": (
                "def test_shenanigans_placeholder():\n    assert True\n"
            ),
            "README.md": "Web project\n",
            "pyproject.toml": "[project]\nname = 'web-project'\n",
        }
    ).items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    return _commit_repo(root)


_SHENANIGANS_ROUTE_GOAL = (
    "Wire the existing shenanigans_page into app/web_routes.py so GET "
    "/shenanigans returns the Shenanigans page using the same _page(...) "
    "wrapper pattern, and add a focused web route test."
)


def test_web_routes_referencing_approval_store_stays_context_eligible(tmp_path):
    """Regression for the Tetris refinement false positive: a routes file that
    merely imports and calls the approval store must stay inspectable for a
    web route task, while a file defining the store stays excluded."""
    root = _init_approval_referencing_web_repo(tmp_path)
    job = _materialized_job(root, goal=_SHENANIGANS_ROUTE_GOAL)

    context = proposal.build_proposal_context(Path(job.worktree_path), job.goal)

    included = list(context.inspected_paths)
    assert "app/web_routes.py" in included
    assert "app/web_routes.py" in context.context_plan["included_priority_files"]
    assert "app/legacy_store.py" not in included
    omitted = {record["path"]: record["reason"] for record in context.omitted_files}
    assert omitted["app/legacy_store.py"] == "approval_store_definition"
    assert context.omission_counts.get("approval_store", 0) == 0
    serialized = json.dumps(context.files)
    assert "_approval_store" in serialized
    assert "legacy_store" not in serialized


def test_proposal_accepts_likely_web_route_files_for_shenanigans_route_task(
    tmp_path,
):
    root = _init_approval_referencing_web_repo(tmp_path)
    job = _materialized_job(root, goal=_SHENANIGANS_ROUTE_GOAL)
    client = _FakeDeepSeekClient(
        _proposal_content(["app/web_routes.py", "tests/test_web_shenanigans.py"])
    )

    completed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.proposal_status == "proposed"
    artifact = _read_state_artifact(root, completed.proposal_artifact_path)
    assert artifact["likely_files_to_edit"] == [
        "app/web_routes.py",
        "tests/test_web_shenanigans.py",
    ]
    assert artifact["attempts"] == [{"attempt": 1, "status": "ok"}]
    assert artifact["suggested_allowlisted_checks"] == ["pytest-freestyle"]


@pytest.mark.parametrize(
    ("likely_file", "category"),
    [
        ("app/approvals.py", "approval_store"),
        ("app/legacy_store.py", "approval_store_definition"),
    ],
)
def test_proposal_still_rejects_approval_store_implementation_likely_files(
    tmp_path, likely_file, category
):
    root = _init_approval_referencing_web_repo(tmp_path)
    job = _materialized_job(root, goal=_SHENANIGANS_ROUTE_GOAL)
    client = _FakeDeepSeekClient(_proposal_content([likely_file]))

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert len(client.completions.calls) == 1
    failed = get_freestyle_job(job.job_id, project_root=root)
    failure = _read_state_artifact(root, failed.proposal_failure_artifact_path)
    attempt = failure["proposal_attempts"][0]
    assert attempt["status"] == "unsafe_likely_path"
    assert attempt["validator_category"] == category
    assert attempt["rejected_path_preview"] == f"[REDACTED_{category.upper()}]"
    assert attempt["retry_skipped_reason"] == "safety_related"


def test_proposal_artifact_records_context_plan_and_ui_prompt_guidance(tmp_path):
    root = _init_web_repo(tmp_path)
    job = _materialized_job(root)
    client = _FakeDeepSeekClient(_proposal_content(["app/web_pages.py"]))

    completed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    artifact = _read_state_artifact(root, completed.proposal_artifact_path)
    plan = artifact["context_plan"]
    assert plan["detected_goal_categories"] == ["web_ui"]
    assert "app/web_pages.py" in plan["high_priority_patterns"]
    assert "app/web_pages.py" in plan["included_priority_files"]
    assert plan["warnings"] == []
    call = client.completions.calls[0]
    system_prompt = call["messages"][0]["content"]
    assert "classified as web/UI work" in system_prompt
    assert "risks_unknowns" in system_prompt
    payload = json.loads(call["messages"][1]["content"])
    assert payload["goal_categories"] == ["web_ui"]


def test_generation_records_goal_fit_warnings_for_orphan_template_bundle(tmp_path):
    root = _init_web_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_tetris_bundle_content(job.job_id))

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_validation_status == "valid"
    assert completed.edit_bundle_goal_fit_warnings == (
        GOAL_FIT_WARNING_CREATE_ONLY,
        GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT,
    )
    system_prompt = client.completions.calls[0]["messages"][0]["content"]
    assert "classified as web/UI work" in system_prompt
    assert "wire them into existing navigation" in system_prompt
    artifact = _read_state_artifact(root, completed.edit_generation_artifact_path)
    assert artifact["goal_fit"]["goal_categories"] == ["web_ui"]
    assert artifact["goal_fit"]["warnings"] == [
        GOAL_FIT_WARNING_CREATE_ONLY,
        GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT,
    ]
    assert artifact["goal_fit"]["advisory_only"] is True
    assert artifact["context_plan"]["detected_goal_categories"] == ["web_ui"]
    assert artifact["context_plan"]["included_priority_files"]


def test_generation_with_wired_web_bundle_has_no_goal_fit_warnings(tmp_path):
    root = _init_web_repo(tmp_path)
    job = _proposed_job(root)
    client = _FakeDeepSeekClient(_wired_bundle_content(job.job_id))

    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.edit_generation_status == "generated"
    assert completed.edit_bundle_goal_fit_warnings == ()
    artifact = _read_state_artifact(root, completed.edit_generation_artifact_path)
    assert artifact["goal_fit"]["warnings"] == []


def test_goal_fit_warnings_surface_in_handoff_and_do_not_block_apply(tmp_path):
    root = _init_web_repo(tmp_path)
    job = _proposed_job(root)
    completed = generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeDeepSeekClient(_tetris_bundle_content(job.job_id)),
    )

    handoff = get_freestyle_handoff(job.job_id, project_root=root)
    assert handoff["goal_fit_warnings"] == [
        GOAL_FIT_WARNING_CREATE_ONLY,
        GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT,
    ]
    assert GOAL_FIT_WARNING_CREATE_ONLY in handoff["risks"]
    assert GOAL_FIT_WARNING_UNWIRED_ENTRY_POINT in handoff["risks"]
    steps = handoff["human_next_steps"]
    warning_step = next(
        index for index, step in enumerate(steps) if "goal-fit warnings" in step
    )
    apply_step = next(
        index for index, step in enumerate(steps) if step.startswith("Apply")
    )
    assert warning_step < apply_step
    markdown = render_freestyle_handoff_markdown(handoff)
    assert "## Goal-Fit Warnings (Advisory)" in markdown
    assert GOAL_FIT_WARNING_CREATE_ONLY in markdown

    bundle_path = root / freestyle.DEV_STATE_DIR_NAME / completed.edit_bundle_path
    applied = apply_freestyle_edit_bundle(
        job.job_id, str(bundle_path), project_root=root
    )
    assert applied.edit_apply_status == "applied"
    assert (Path(job.worktree_path) / "templates" / "shennanigans.html").is_file()


def test_ui_goal_whose_context_lost_every_web_file_fails_closed_before_deepseek(
    tmp_path,
):
    root = _init_web_repo(tmp_path)
    crowding_paths = [*_FREESTYLE_DECOY_FILES, "README.md", "pyproject.toml"]
    assert len(crowding_paths) >= proposal.MAX_CONTEXT_FILES
    job = _proposed_job(root, likely_files=crowding_paths)
    client = _FakeDeepSeekClient(_tetris_bundle_content(job.job_id))

    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    assert client.completions.calls == []
    failed = get_freestyle_job(job.job_id, project_root=root)
    assert failed.edit_generation_status == "error"
    assert failed.edit_bundle_path is None
    assert "classified as web/UI" in failed.edit_generation_errors[0]
    artifact = _read_state_artifact(root, failed.edit_generation_artifact_path)
    plan = artifact["context_plan"]
    assert plan["included_priority_files"] == []
    assert plan["priority_candidate_count"] == len(_WEB_FILES)
    assert plan["priority_files_omitted_due_to_budget"] is True
    handoff = get_freestyle_handoff(job.job_id, project_root=root)
    assert any(
        "omitted from edit-generation model context due to budget" in risk
        for risk in handoff["risks"]
    )


def test_ui_goal_in_repo_without_web_files_warns_but_still_proposes(tmp_path):
    root = _init_plain_repo(tmp_path)
    job = _materialized_job(root, goal="Add a new page tab to the dashboard UI.")
    client = _FakeDeepSeekClient(_proposal_content(["app/dev/sample.py"]))

    completed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.proposal_status == "proposed"
    artifact = _read_state_artifact(root, completed.proposal_artifact_path)
    plan = artifact["context_plan"]
    assert plan["detected_goal_categories"] == ["web_ui"]
    assert plan["priority_candidate_count"] == 0
    assert plan["warnings"] == [PLAN_WARNING_NO_UI_FILES]
    handoff = get_freestyle_handoff(job.job_id, project_root=root)
    assert any(PLAN_WARNING_NO_UI_FILES in risk for risk in handoff["risks"])


def test_non_ui_goal_gets_no_ui_guidance_and_no_goal_fit_warnings(tmp_path):
    root = _init_plain_repo(tmp_path)
    job = _materialized_job(root, goal="Update the sample value.")
    proposal_client = _FakeDeepSeekClient(_proposal_content(["app/dev/sample.py"]))
    proposed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=proposal_client,
    )
    bundle = json.dumps(
        {
            "schema_version": 1,
            "job_id": job.job_id,
            "created_at": "2026-06-12T12:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "replace_text",
                    "path": "app/dev/sample.py",
                    "old_text": "VALUE = 1",
                    "new_text": "VALUE = 2",
                }
            ],
        }
    )
    generation_client = _FakeDeepSeekClient(bundle)

    completed = generate_freestyle_edit_bundle(
        proposed.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=generation_client,
    )

    for client in (proposal_client, generation_client):
        system_prompt = client.completions.calls[0]["messages"][0]["content"]
        assert "classified as web/UI work" not in system_prompt
    assert completed.edit_bundle_goal_fit_warnings == ()
    proposal_artifact = _read_state_artifact(root, proposed.proposal_artifact_path)
    assert proposal_artifact["context_plan"]["detected_goal_categories"] == ["core_app"]
    handoff = get_freestyle_handoff(proposed.job_id, project_root=root)
    assert handoff["goal_fit_warnings"] == []


def _web_ui_settings(tmp_path: Path) -> Settings:
    documents = tmp_path / "documents"
    documents.mkdir(exist_ok=True)
    return Settings(
        deepseek_api_key=None,
        deepseek_model="test-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=True,
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


def test_freestyle_web_page_shows_goal_fit_warnings_before_apply_action(tmp_path):
    root = _init_web_repo(tmp_path)
    job = _proposed_job(root)
    generate_freestyle_edit_bundle(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeDeepSeekClient(_tetris_bundle_content(job.job_id)),
    )
    app = create_app(_web_ui_settings(tmp_path))
    app.state.freestyle_project_root = root
    client = TestClient(app)

    page = client.get(f"/dev/freestyle/{job.job_id}").text

    assert "goal-fit-warnings" in page
    assert "Goal-fit warnings" in page
    assert "only creates new files" in page
    assert page.index("goal-fit-warnings") < page.index("Apply Latest Generated Bundle")
