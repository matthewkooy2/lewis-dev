"""Tests for refining an existing freestyle isolated worktree in place."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_edit_bundle as edit_bundle
import app.dev.freestyle_maintenance as maintenance
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    get_freestyle_handoff,
    get_freestyle_job,
    materialize_freestyle_job,
    propose_freestyle_job,
    refine_freestyle_job,
    run_freestyle_job,
)
from app.dev.freestyle_policy import APPROVAL_STORE_CLASS_NAME
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
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".gitignore").write_text(".lewis-dev/\n", encoding="utf-8")
    (root / "app" / "web_routes.py").write_text("ROUTES = {}\n", encoding="utf-8")
    (root / "tests" / "test_routes.py").write_text(
        "def test_routes():\n    assert True\n", encoding="utf-8"
    )
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _config() -> DeepSeekProposalConfig:
    return DeepSeekProposalConfig(
        api_key="test-deepseek-key",
        model="deepseek-test-code",
        base_url=DEEPSEEK_API_BASE_URL,
    )


class _Completions:
    def __init__(self, contents: list[str]):
        self.contents = contents
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
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


class _Client:
    def __init__(self, contents: list[str]):
        self.completions = _Completions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def _proposal(*paths: str) -> str:
    return json.dumps(
        {
            "proposed_implementation_steps": ["Make the targeted isolated change."],
            "likely_files_to_edit": list(paths),
            "risks_unknowns": ["Keep existing working changes intact."],
            "suggested_allowlisted_checks": ["pytest"],
        }
    )


def _bundle(job_id: str, operations: list[dict]) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "job_id": job_id,
            "created_at": "2026-06-12T12:00:00+00:00",
            "operation_count": len(operations),
            "operations": operations,
        }
    )


def _lean_bundle(operations: list[dict]) -> str:
    """A lean model output: only schema_version and operations, no provenance."""
    return json.dumps({"schema_version": 1, "operations": operations})


def _set_metadata(root: Path, job_id: str, **values) -> None:
    path = root / freestyle.DEV_STATE_DIR_NAME / "jobs" / f"{job_id}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(values)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _applied_page_job(root: Path):
    job = create_freestyle_job(
        "Create a shenanigans page helper.",
        developer_approved=True,
        project_root=root,
    )
    initial = _Client(
        [
            _proposal("app/shenanigans.py", "app/web_routes.py"),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "create_file",
                        "path": "app/shenanigans.py",
                        "content": (
                            "def shenanigans_page():\n"
                            '    return "<canvas id=\\"tetris\\"></canvas>"\n'
                        ),
                    }
                ],
            ),
        ]
    )
    return run_freestyle_job(
        job.job_id,
        apply_latest_generated=True,
        confirm_apply=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=initial,
    )


def test_refine_sees_prior_applied_page_and_stops_before_apply(tmp_path):
    root = _init_repo(tmp_path)
    original_main = (root / "app" / "web_routes.py").read_text(encoding="utf-8")
    job = _applied_page_job(root)
    original_proposal = job.proposal_artifact_path
    original_bundle = job.edit_bundle_path
    worktree = Path(job.worktree_path)
    assert (worktree / "app" / "shenanigans.py").is_file()
    assert not (root / "app" / "shenanigans.py").exists()

    client = _Client(
        [
            _proposal("app/web_routes.py", "app/shenanigans.py"),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": "ROUTES = {}",
                        "new_text": 'ROUTES = {"/shenanigans": shenanigans_page}',
                    }
                ],
            ),
        ]
    )
    refined = refine_freestyle_job(
        job.job_id,
        "Wire /shenanigans to the existing shenanigans_page.",
        developer_approved=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    proposal_payload = json.loads(client.completions.calls[0]["messages"][1]["content"])
    files = {item["path"]: item["content"] for item in proposal_payload["safe_files"]}
    assert "app/shenanigans.py" in files
    assert "shenanigans_page" in files["app/shenanigans.py"]
    assert "current isolated worktree, not a fresh job" in proposal_payload["goal"]
    assert refined.latest_refinement_status == "generated"
    assert refined.edit_apply_status == "not_applied"
    assert (worktree / "app" / "web_routes.py").read_text(encoding="utf-8") == original_main
    assert (root / "app" / "web_routes.py").read_text(encoding="utf-8") == original_main
    assert refined.original_artifacts["proposal"] == original_proposal
    assert refined.original_artifacts["generated_bundle"] == original_bundle
    assert refined.original_stage_statuses["edit_apply"] == "applied"
    assert refined.proposal_artifact_path != original_proposal
    assert (root / freestyle.DEV_STATE_DIR_NAME / original_proposal).is_file()
    assert len(refined.refinement_history) == 1
    with pytest.raises(FreestyleJobRefusedError, match="explicit refinement"):
        propose_freestyle_job(job.job_id, project_root=root)
    assert (root / freestyle.DEV_STATE_DIR_NAME / original_proposal).is_file()

    handoff = get_freestyle_handoff(job.job_id, project_root=root)
    assert handoff["refinement"]["latest_goal"].startswith("Wire /shenanigans")
    assert handoff["refinement"]["original_artifacts"]["proposal"] == original_proposal
    assert handoff["artifacts"]["latest_refinement"]


def test_refinement_huge_goal_is_not_echoed_into_model_output_contract(tmp_path):
    """Regression for the Calendar refinement that hit finish_reason="length":
    a refinement carries a very large goal as input context, but the model
    output contract demands only schema_version and operations. The model
    returns a lean bundle without echoing the goal, and Lewis injects bounded
    trusted provenance after parsing so output budget is spent on operations."""
    root = _init_repo(tmp_path)
    job = _applied_page_job(root)
    huge_goal = "Wire /shenanigans into the routes. " + ("scope detail line. " * 400)
    client = _Client(
        [
            _proposal("app/web_routes.py", "app/shenanigans.py"),
            _lean_bundle(
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": "ROUTES = {}",
                        "new_text": 'ROUTES = {"/shenanigans": shenanigans_page}',
                    }
                ]
            ),
        ]
    )

    refined = refine_freestyle_job(
        job.job_id,
        huge_goal,
        developer_approved=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert refined.latest_refinement_status == "generated"
    assert refined.edit_bundle_validation_status == "valid"
    # The model only needed one call: it never had to echo the huge goal.
    assert len(client.completions.calls) == 2

    generation_call = client.completions.calls[1]
    system_prompt = generation_call["messages"][0]["content"]
    assert "exactly two root fields: schema_version and operations" in system_prompt
    user_payload = json.loads(generation_call["messages"][1]["content"])
    # The huge goal is supplied as input context but is not a required output.
    assert len(user_payload["goal"]) > edit_bundle.MAX_PROVENANCE_TEXT_CHARS
    schema = user_payload["exact_edit_bundle_schema"]
    assert schema["bundle"]["required_fields"] == ["schema_version", "operations"]
    assert "goal" in schema["bundle"]["host_injected_fields"]

    # The persisted bundle carries Lewis-injected, bounded provenance only.
    persisted = json.loads(
        (
            root / freestyle.DEV_STATE_DIR_NAME / refined.edit_bundle_path
        ).read_text(encoding="utf-8")
    )
    assert persisted["job_id"] == job.job_id
    assert persisted["operation_count"] == 1
    assert len(persisted["goal"]) <= edit_bundle.MAX_PROVENANCE_TEXT_CHARS
    assert "\n" not in persisted["goal"]


def test_refine_apply_requires_both_flags_and_uses_safe_applier(tmp_path):
    root = _init_repo(tmp_path)
    job = _applied_page_job(root)

    with pytest.raises(FreestyleJobRefusedError, match="requires both"):
        refine_freestyle_job(
            job.job_id,
            "Wire the route.",
            developer_approved=True,
            apply_latest_generated=True,
            project_root=root,
        )
    assert get_freestyle_job(job.job_id, project_root=root).refinement_count == 0

    client = _Client(
        [
            _proposal("app/web_routes.py", "app/shenanigans.py"),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": "ROUTES = {}",
                        "new_text": 'ROUTES = {"/shenanigans": shenanigans_page}',
                    }
                ],
            ),
        ]
    )
    refined = refine_freestyle_job(
        job.job_id,
        "Wire the route.",
        developer_approved=True,
        apply_latest_generated=True,
        confirm_apply=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert refined.edit_apply_status == "applied"
    assert refined.review_status == "reviewed"
    assert refined.latest_refinement_edit_artifact_path
    assert refined.latest_refinement_review_artifact_path
    assert '"/shenanigans"' in (
        Path(job.worktree_path) / "app" / "web_routes.py"
    ).read_text(encoding="utf-8")
    assert (root / "app" / "web_routes.py").read_text(encoding="utf-8") == "ROUTES = {}\n"


# Mirrors the real app/web_routes.py: approval-store imports and approval
# routes live in the same file as ordinary page routes. Assembled with the
# concatenated class name so this test file never contains the literal text.
_TETRIS_ROUTES = f'''from app.approvals import {APPROVAL_STORE_CLASS_NAME}
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
'''
_TETRIS_REFINEMENT_GOAL = (
    "Wire the existing shenanigans_page into app/web_routes.py so GET "
    "/shenanigans returns the Shenanigans page using the same _page(...) "
    "wrapper pattern as /dashboard, /usage, and /chat. Add or update a "
    "focused web route test for GET /shenanigans. Do not change "
    "app/static/tetris.js or the Tetris game logic unless required by the "
    "route test."
)
_CHAT_ROUTE_ANCHOR = '        return _page("Chat", request, chat_page())'
_WIRED_ROUTES_TEXT = (
    '        return _page("Chat", request, chat_page())\n'
    "\n"
    '    @app.get("/shenanigans")\n'
    "    def shenanigans(request):\n"
    '        return _page("Shenanigans", request, shenanigans_page())'
)


def _init_tetris_repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "app" / "static").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".gitignore").write_text(".lewis-dev/\n", encoding="utf-8")
    (root / "app" / "web_routes.py").write_text(_TETRIS_ROUTES, encoding="utf-8")
    (root / "app" / "web_pages.py").write_text(
        'def chat_page():\n    return "<p>chat</p>"\n', encoding="utf-8"
    )
    (root / "tests" / "test_routes.py").write_text(
        "def test_routes():\n    assert True\n", encoding="utf-8"
    )
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def test_tetris_refinement_proposes_route_wiring_and_stops_at_generated_bundle(
    tmp_path,
):
    """End-to-end regression for the real Tetris refinement failure: the
    approval-referencing routes file stays in context and proposable, one
    invalid-JSON proposal gets exactly one repair retry, the generic "pytest"
    suggestion normalizes instead of failing, and the refinement stops at the
    generated-bundle boundary without touching the worktree or checkout."""
    root = _init_tetris_repo(tmp_path)
    job = create_freestyle_job(
        "Add a new tab under Overview called Shenanigans where I can play "
        "Tetris. Keep it local/client-side and add appropriate tests.",
        developer_approved=True,
        project_root=root,
    )
    original_client = _Client(
        [
            _proposal("app/web_pages.py", "app/static/tetris.js"),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_pages.py",
                        "old_text": 'def chat_page():\n    return "<p>chat</p>"',
                        "new_text": (
                            'def chat_page():\n    return "<p>chat</p>"\n'
                            "\n\ndef shenanigans_page():\n"
                            "    return '<canvas id=\"tetris-canvas\"></canvas>'"
                        ),
                    },
                    {
                        "op": "create_file",
                        "path": "app/static/tetris.js",
                        "content": "console.log('tetris');\n",
                    },
                    {
                        "op": "create_file",
                        "path": "tests/test_web_shenanigans.py",
                        "content": "def test_shenanigans_page():\n    assert True\n",
                    },
                ],
            ),
        ]
    )
    applied = run_freestyle_job(
        job.job_id,
        apply_latest_generated=True,
        confirm_apply=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=original_client,
    )
    original_proposal = applied.proposal_artifact_path
    worktree = Path(applied.worktree_path)
    routes_before = (worktree / "app" / "web_routes.py").read_text(encoding="utf-8")

    refine_client = _Client(
        [
            "{ this is not valid json",
            _proposal("app/web_routes.py", "tests/test_web_shenanigans.py"),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": _CHAT_ROUTE_ANCHOR,
                        "new_text": _WIRED_ROUTES_TEXT,
                    },
                    {
                        "op": "replace_text",
                        "path": "tests/test_web_shenanigans.py",
                        "old_text": "def test_shenanigans_page():\n    assert True",
                        "new_text": (
                            "def test_shenanigans_route():\n"
                            '    assert "/shenanigans"'
                        ),
                    },
                ],
            ),
        ]
    )
    refined = refine_freestyle_job(
        job.job_id,
        _TETRIS_REFINEMENT_GOAL,
        developer_approved=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=refine_client,
    )

    assert len(refine_client.completions.calls) == 3
    proposal_payload = json.loads(
        refine_client.completions.calls[0]["messages"][1]["content"]
    )
    context_paths = {item["path"] for item in proposal_payload["safe_files"]}
    assert {
        "app/web_routes.py",
        "app/web_pages.py",
        "app/static/tetris.js",
        "tests/test_web_shenanigans.py",
    } <= context_paths
    contents = {
        item["path"]: item["content"] for item in proposal_payload["safe_files"]
    }
    assert "shenanigans_page" in contents["app/web_pages.py"]

    assert refined.latest_refinement_status == "generated"
    assert refined.proposal_status == "proposed"
    assert refined.edit_bundle_validation_status == "valid"
    assert refined.edit_apply_status == "not_applied"
    artifact_root = root / freestyle.DEV_STATE_DIR_NAME
    proposal_artifact = json.loads(
        (artifact_root / refined.proposal_artifact_path).read_text(encoding="utf-8")
    )
    assert proposal_artifact["likely_files_to_edit"] == [
        "app/web_routes.py",
        "tests/test_web_shenanigans.py",
    ]
    assert [item["status"] for item in proposal_artifact["attempts"]] == [
        "invalid_json",
        "ok",
    ]
    assert proposal_artifact["attempts"][1]["repair_retry"] is True
    assert proposal_artifact["suggested_allowlisted_checks"] == ["pytest-freestyle"]
    assert proposal_artifact["suggested_check_warnings"] == []

    assert (worktree / "app" / "web_routes.py").read_text(
        encoding="utf-8"
    ) == routes_before
    assert (root / "app" / "web_routes.py").read_text(
        encoding="utf-8"
    ) == _TETRIS_ROUTES
    assert refined.original_artifacts["proposal"] == original_proposal
    assert (artifact_root / original_proposal).is_file()
    assert refined.proposal_artifact_path != original_proposal


def test_tetris_refinement_recovers_from_stale_anchor_with_exact_snippets(tmp_path):
    """End-to-end regression for the third real Tetris refinement failure:
    DeepSeek anchored replace_text on a stale approximation of the truncated
    routes file and validation failed with "replace_text found no exact match
    in app/web_routes.py". The first generation attempt now triggers exactly
    one anchor-repair retry whose prompt carries fresh marker-free exact
    snippets from the current isolated worktree, and the corrected bundle
    wires the route plus its test."""
    root = _init_tetris_repo(tmp_path)
    job = create_freestyle_job(
        "Add a Shenanigans page route for the Tetris tab.",
        developer_approved=True,
        project_root=root,
    )
    materialize_freestyle_job(job.job_id, project_root=root)
    # Mirrors the real failure: the invented anchor drops one blank line
    # between the imports and the register function, so it never matches.
    stale_anchor = (
        "from app.web_pages import approvals_page, chat_page\n"
        "\n"
        "def register_web_routes(app):"
    )
    client = _Client(
        [
            _proposal("app/web_routes.py", "tests/test_routes.py"),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": stale_anchor,
                        "new_text": stale_anchor + "\n    pass",
                    }
                ],
            ),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": _CHAT_ROUTE_ANCHOR,
                        "new_text": _WIRED_ROUTES_TEXT,
                    },
                    {
                        "op": "replace_text",
                        "path": "tests/test_routes.py",
                        "old_text": "def test_routes():\n    assert True",
                        "new_text": (
                            "def test_shenanigans_route():\n"
                            '    assert "/shenanigans"'
                        ),
                    },
                ],
            ),
        ]
    )
    routes_before = (
        Path(job.worktree_path) / "app" / "web_routes.py"
    ).read_text(encoding="utf-8")

    refined = refine_freestyle_job(
        job.job_id,
        _TETRIS_REFINEMENT_GOAL,
        developer_approved=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert len(client.completions.calls) == 3
    assert refined.latest_refinement_status == "generated"
    assert refined.edit_generation_status == "generated"
    assert refined.edit_bundle_validation_status == "valid"
    assert refined.edit_apply_status == "not_applied"

    repair_messages = client.completions.calls[2]["messages"]
    assert [message["role"] for message in repair_messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    repair_payload = json.loads(repair_messages[-1]["content"])
    assert repair_payload["failed_path"] == "app/web_routes.py"
    assert "smaller exact anchor" in repair_payload["instruction"]
    fresh = repair_payload["fresh_exact_anchor_snippets"]
    assert any(
        snippet["label"] == "route:/chat"
        and _CHAT_ROUTE_ANCHOR in snippet["exact_text"]
        and snippet["exact_text"] in routes_before
        for snippet in fresh
    )
    assert all(
        "approve_approval" not in snippet["exact_text"]
        and APPROVAL_STORE_CLASS_NAME not in snippet["exact_text"]
        for snippet in fresh
    )

    artifact_root = root / freestyle.DEV_STATE_DIR_NAME
    generation_artifact = json.loads(
        (artifact_root / refined.edit_generation_artifact_path).read_text(
            encoding="utf-8"
        )
    )
    assert [
        record["status"] for record in generation_artifact["attempts"]
    ] == ["exact_match_error", "valid"]
    retry = generation_artifact["anchor_snippets"]["retry"]
    assert retry["used"] is True
    assert retry["failed_path"] == "app/web_routes.py"
    assert "route:/chat" in retry["fresh_snippet_labels"]
    bundle_payload = json.loads(
        (artifact_root / refined.edit_bundle_path).read_text(encoding="utf-8")
    )
    assert [
        operation["path"] for operation in bundle_payload["operations"]
    ] == ["app/web_routes.py", "tests/test_routes.py"]

    assert (Path(job.worktree_path) / "app" / "web_routes.py").read_text(
        encoding="utf-8"
    ) == routes_before
    assert (root / "app" / "web_routes.py").read_text(
        encoding="utf-8"
    ) == _TETRIS_ROUTES


def test_tetris_refinement_applies_two_replace_text_ops_on_one_file(tmp_path):
    """Regression for the live Tetris failure "Edit bundle contains more than
    one operation for the same path": the corrected bundle carries two
    non-overlapping replace_text ops on app/web_routes.py (one import wiring,
    one route wiring). It now validates, applies in the isolated worktree as a
    single folded write, leaves the approval code untouched, and never mutates
    the main checkout."""
    root = _init_tetris_repo(tmp_path)
    job = create_freestyle_job(
        "Add a Shenanigans page route for the Tetris tab.",
        developer_approved=True,
        project_root=root,
    )
    materialize_freestyle_job(job.job_id, project_root=root)
    import_anchor = "from app.web_pages import approvals_page, chat_page"
    wired_import = import_anchor + ", shenanigans_page"
    client = _Client(
        [
            _proposal("app/web_routes.py", "app/web_pages.py"),
            _bundle(
                job.job_id,
                [
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": import_anchor,
                        "new_text": wired_import,
                    },
                    {
                        "op": "replace_text",
                        "path": "app/web_routes.py",
                        "old_text": _CHAT_ROUTE_ANCHOR,
                        "new_text": _WIRED_ROUTES_TEXT,
                    },
                ],
            ),
        ]
    )

    refined = refine_freestyle_job(
        job.job_id,
        _TETRIS_REFINEMENT_GOAL,
        developer_approved=True,
        apply_latest_generated=True,
        confirm_apply=True,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert len(client.completions.calls) == 2
    assert refined.edit_bundle_validation_status == "valid"
    assert refined.edit_apply_status == "applied"
    assert refined.review_status == "reviewed"

    wired = (Path(job.worktree_path) / "app" / "web_routes.py").read_text(
        encoding="utf-8"
    )
    assert wired_import in wired
    assert '"/shenanigans"' in wired
    assert "def shenanigans(request):" in wired
    # Approval-referencing code in the same file is left untouched.
    assert "def _approval_store(request):" in wired
    assert "approve_approval(_approval_store(request), approval_id)" in wired
    # The main checkout never changes.
    assert (root / "app" / "web_routes.py").read_text(
        encoding="utf-8"
    ) == _TETRIS_ROUTES

    artifact = json.loads(
        (
            root
            / freestyle.DEV_STATE_DIR_NAME
            / refined.latest_refinement_edit_artifact_path
        ).read_text(encoding="utf-8")
    )
    assert artifact["touched_file_count"] == 1
    assert artifact["operation_count"] == 2
    entry = artifact["operations"][0]
    assert entry["path"] == "app/web_routes.py"
    assert entry["replacement_count"] == 2
    assert entry["descending_offset_application"] is True


def test_refine_refuses_unknown_unapproved_not_materialized_active_and_inconsistent(
    tmp_path,
):
    root = _init_repo(tmp_path)
    with pytest.raises(FreestyleJobRefusedError, match="was not found"):
        refine_freestyle_job(
            "0" * 32, "Refine it.", developer_approved=True, project_root=root
        )

    job = create_freestyle_job(
        "Create a page.", developer_approved=True, project_root=root
    )
    with pytest.raises(FreestyleJobRefusedError, match="materialized"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )

    freestyle.materialize_freestyle_job(job.job_id, project_root=root)
    _set_metadata(root, job.job_id, developer_approved=False)
    with pytest.raises(FreestyleJobRefusedError, match="persisted developer approval"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )

    _set_metadata(root, job.job_id, developer_approved=True, checks_status="running")
    with pytest.raises(FreestyleJobRefusedError, match="cannot be refined"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )

    _set_metadata(root, job.job_id, checks_status="not_run", edit_apply_status="inconsistent")
    with pytest.raises(FreestyleJobRefusedError, match="cannot be refined"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )


def test_refine_refuses_unsafe_missing_mismatched_and_wrong_registered_worktrees(
    tmp_path, monkeypatch
):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Create a page.", developer_approved=True, project_root=root
    )
    freestyle.materialize_freestyle_job(job.job_id, project_root=root)

    _set_metadata(root, job.job_id, worktree_path=str(tmp_path / "outside"))
    with pytest.raises(FreestyleJobRefusedError, match="does not match local dev state"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )

    _set_metadata(root, job.job_id, worktree_path=job.worktree_path, branch="wrong-branch")
    with pytest.raises(FreestyleJobRefusedError, match="branch does not match"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )

    _set_metadata(root, job.job_id, branch=job.branch)
    expected = root / freestyle.DEV_STATE_DIR_NAME / "worktrees" / job.job_id
    monkeypatch.setattr(
        maintenance, "_registered_worktrees", lambda unused_root: {expected: "wrong"}
    )
    with pytest.raises(FreestyleJobRefusedError, match="registered worktree branch"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )

    monkeypatch.undo()
    shutil.rmtree(job.worktree_path)
    with pytest.raises(FreestyleJobRefusedError, match="missing"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )


def test_refine_refuses_malformed_refinement_metadata_before_artifact_write(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Create a page.", developer_approved=True, project_root=root
    )
    freestyle.materialize_freestyle_job(job.job_id, project_root=root)
    _set_metadata(
        root,
        job.job_id,
        refinement_count=1,
        latest_refinement_id="../unsafe",
        refinement_history=[
            {
                "iteration": 1,
                "refinement_id": "../unsafe",
                "revision": 0,
            }
        ],
    )

    with pytest.raises(FreestyleJobRefusedError, match="metadata is inconsistent"):
        refine_freestyle_job(
            job.job_id, "Refine it.", developer_approved=True, project_root=root
        )
    assert not (root / freestyle.DEV_STATE_DIR_NAME / "refinements").exists()


def test_recovery_and_cleanup_remain_safe_for_refined_jobs(tmp_path):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Create a page.", developer_approved=True, project_root=root
    )
    freestyle.materialize_freestyle_job(job.job_id, project_root=root)
    refinement_id = "1" * 16
    artifact_dir = (
        root / freestyle.DEV_STATE_DIR_NAME / "refinements" / job.job_id
    )
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / f"refinement-{refinement_id}-0001.json"
    artifact.write_text("{}\n", encoding="utf-8")
    _set_metadata(
        root,
        job.job_id,
        refinement_count=1,
        latest_refinement_id=refinement_id,
        latest_refinement_goal="Refine it.",
        latest_refinement_status="running",
        latest_refinement_artifact_path=artifact.relative_to(
            root / freestyle.DEV_STATE_DIR_NAME
        ).as_posix(),
        refinement_history=[
            {
                "iteration": 1,
                "refinement_id": refinement_id,
                "refinement_goal": "Refine it.",
                "status": "running",
                "revision": 1,
            }
        ],
    )

    with pytest.raises(FreestyleJobRefusedError, match="active or inconsistent"):
        maintenance.cleanup_freestyle_job(
            job.job_id, confirm=True, project_root=root
        )
    recovered = maintenance.recover_freestyle_job(
        job.job_id, confirm=True, project_root=root
    )
    assert recovered.latest_refinement_status == "failed"
    assert recovered.refinement_history[-1]["status"] == "failed"
    assert artifact.is_file()
    assert Path(job.worktree_path).is_dir()

    result = maintenance.cleanup_freestyle_job(
        job.job_id, confirm=True, project_root=root
    )
    assert "removed_job_metadata" in result["actions_taken"]
    assert not artifact_dir.exists()


def test_refine_cli_requires_approval_and_dispatches_safe_defaults(
    tmp_path, monkeypatch, capsys
):
    root = _init_repo(tmp_path)
    job = create_freestyle_job(
        "Create a page.", developer_approved=True, project_root=root
    )
    calls: list[dict] = []

    def fake_refine(job_id, goal, **kwargs):
        calls.append({"job_id": job_id, "goal": goal, **kwargs})
        return job

    monkeypatch.setattr(freestyle, "refine_freestyle_job", fake_refine)
    assert freestyle.main(["refine", job.job_id, "--goal", "Wire route.", "--approve"]) == 0
    assert json.loads(capsys.readouterr().out)["job_id"] == job.job_id
    assert calls == [
        {
            "job_id": job.job_id,
            "goal": "Wire route.",
            "developer_approved": True,
            "apply_latest_generated": False,
            "confirm_apply": False,
            "run_checks": False,
            "check_ids": None,
        }
    ]
