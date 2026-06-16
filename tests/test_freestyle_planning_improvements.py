"""Tests for the deterministic planning, semantic critique, richer empty-
response diagnostics, and refinement-memory improvements to Freestyle.

These prove the concrete failures that motivated the work cannot recur:
an explicitly named implementation file is never displaced by tests, a
tests-only bundle for an implementation goal is flagged, a created-but-uncalled
helper is flagged, and an empty DeepSeek response produces a richer diagnostic
that still never stores prompts or secrets.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import app.dev.freestyle_coding as freestyle
import app.dev.freestyle_proposal as proposal
from app.dev.freestyle_bundle_critique import (
    CRITIQUE_FORBIDDEN_PATH_TOUCHED,
    CRITIQUE_HELPER_CREATED_NOT_CALLED,
    CRITIQUE_LIKELY_FILES_IGNORED,
    CRITIQUE_NAMED_TARGETS_NOT_TOUCHED,
    CRITIQUE_TESTS_ONLY_FOR_IMPLEMENTATION,
    CritiqueOperation,
    critique_bundle,
)
from app.dev.freestyle_coding import (
    FreestyleJobRefusedError,
    create_freestyle_job,
    generate_freestyle_edit_bundle,
    get_freestyle_job,
    materialize_freestyle_job,
    propose_freestyle_job,
)
from app.dev.freestyle_context_plan import (
    NAMED_TARGET_BUDGET_BLOCKER,
    extract_named_basenames,
    extract_named_paths,
    is_named_target,
    named_target_availability,
    plan_context,
    summarize_context_plan,
)
from app.dev.freestyle_intake import (
    build_intake_summary,
    classify_intent,
    extract_forbidden_paths,
    requires_implementation_change,
)
from app.dev.freestyle_proposal import DeepSeekProposalConfig, safe_response_envelope
from app.dev.freestyle_refinement import _memory_prompt_section
import app.dev.freestyle_bundle_critique as bundle_critique
import app.dev.freestyle_intake as intake


# --------------------------------------------------------------------------- #
# Named-path extraction and context-plan prioritisation (pure planner).
# --------------------------------------------------------------------------- #


def test_extract_named_paths_finds_repo_paths_and_ignores_prose():
    text = (
        "Refactor app/dev/freestyle_proposal.py and update tests/test_x.py, but "
        "do not download from https://example.com/page.html or touch version 1.2."
    )
    named = extract_named_paths(text)
    assert "app/dev/freestyle_proposal.py" in named
    assert "tests/test_x.py" in named
    # A bare version token is not a path; a URL host is not a repo file.
    assert all(not path.startswith("http") for path in named)


def test_plan_context_records_named_targets_from_goal_and_title():
    plan = plan_context(
        "Modify app/dev/freestyle_proposal.py to add a planning step.",
        title="Touch app/web_pages.py too",
    )
    assert "app/dev/freestyle_proposal.py" in plan.named_target_paths
    assert is_named_target("app/dev/freestyle_proposal.py", plan)
    assert not is_named_target("tests/test_other.py", plan)


def test_process_terms_do_not_make_calendar_refinement_internal_or_named():
    goal = (
        "Repair the Calendar UI wiring and /calendar layout. Use repo-map "
        "context from the previous Freestyle proposal failure artifact and "
        "review warnings, but do not touch Freestyle internals. The prose also "
        "mentions the code proposal engine as process background."
    )

    plan = plan_context(goal, title="Calendar refinement")

    assert "web_ui" in plan.goal_categories
    assert "freestyle_internal" not in plan.goal_categories
    assert "app/code_proposal_engine.py" not in plan.named_target_paths
    assert not is_named_target("app/code_proposal_engine.py", plan)


def test_concrete_basenames_are_explicit_but_words_are_not():
    vague = plan_context("The proposal engine wording appears in the review.")
    explicit = plan_context("Inspect code_proposal_engine.py for this change.")

    assert "code_proposal_engine.py" not in vague.named_target_basenames
    assert "code_proposal_engine.py" in extract_named_basenames(
        "Inspect code_proposal_engine.py."
    )
    assert is_named_target("app/code_proposal_engine.py", explicit)


def test_plan_context_accepts_extra_named_paths_from_a_refinement():
    plan = plan_context(
        "Improve robustness.",
        extra_named_paths=("app/dev/freestyle_proposal.py",),
    )
    assert is_named_target("app/dev/freestyle_proposal.py", plan)


def test_named_target_availability_classifies_each_named_path():
    plan = plan_context("Update app/impl.py and app/big.py and app/secret.py.")
    result = named_target_availability(
        plan,
        included_paths=["app/impl.py"],
        omitted_records=[
            {"path": "app/big.py", "reason": "total_context_limit"},
            {"path": "app/secret.py", "reason": "sensitive_content"},
        ],
        candidate_paths=["app/impl.py", "app/big.py"],
    )
    assert result["included"] == ["app/impl.py"]
    assert [record["path"] for record in result["omitted_due_to_budget"]] == [
        "app/big.py"
    ]
    assert result["excluded_by_policy"] == ["app/secret.py"]
    assert result["has_budget_blocker"] is True


def test_summary_warns_when_named_target_omitted_for_budget():
    plan = plan_context("Update app/impl.py to add a feature.")
    summary = summarize_context_plan(
        plan,
        included_paths=[],
        omitted_records=[{"path": "app/impl.py", "reason": "file_count_limit"}],
        priority_candidate_count=0,
        candidate_paths=["app/impl.py"],
    )
    assert NAMED_TARGET_BUDGET_BLOCKER in summary["warnings"]
    assert summary["named_target_files"]["has_budget_blocker"] is True


def test_named_target_budget_omission_still_refuses_before_deepseek():
    from app.dev.freestyle_proposal import (
        ProposalContext,
        _require_goal_grounded_context,
    )

    context = ProposalContext(
        files=(),
        omitted_files=(),
        omission_counts={},
        included_char_count=0,
        scanned_file_count=0,
        context_plan={
            "detected_goal_categories": ["core_app"],
            "named_target_files": {
                "omitted_due_to_budget": [
                    {"path": "app/explicit.py", "reason": "total_context_limit"}
                ]
            },
        },
    )

    with pytest.raises(FreestyleJobRefusedError, match=NAMED_TARGET_BUDGET_BLOCKER):
        _require_goal_grounded_context(context)


def test_plan_recommended_paths_stay_advisory_not_named_budget_blockers():
    from app.dev.freestyle_proposal import (
        ProposalContext,
        _plan_recommendation_section,
        _require_goal_grounded_context,
    )

    section = _plan_recommendation_section(
        ["app/dev/freestyle_ui_review.py"],
        included_paths=[],
        candidate_paths=["app/dev/freestyle_ui_review.py"],
    )
    context = ProposalContext(
        files=(),
        omitted_files=(
            {"path": "app/dev/freestyle_ui_review.py", "reason": "file_count_limit"},
        ),
        omission_counts={},
        included_char_count=0,
        scanned_file_count=1,
        context_plan={
            "detected_goal_categories": ["freestyle_internal"],
            "named_target_files": {"omitted_due_to_budget": []},
            "implementation_plan": section,
        },
    )

    assert section["advisory_only"] is True
    assert section["omitted_recommended_files_in_budget"] == [
        "app/dev/freestyle_ui_review.py"
    ]
    _require_goal_grounded_context(context)


# --------------------------------------------------------------------------- #
# Intent classification and intake.
# --------------------------------------------------------------------------- #


def test_classify_intent_and_requires_implementation_change():
    impl = classify_intent("Implement a caching layer in app/cache.py.")
    assert "implementation" in impl
    assert requires_implementation_change(impl)

    tests_only = classify_intent("Add tests for the existing cache behaviour.")
    assert tests_only == ("tests_only",)
    assert not requires_implementation_change(tests_only)


def test_extract_forbidden_paths_from_negated_clauses():
    forbidden = extract_forbidden_paths(
        "Update app/web_routes.py but do not modify app/approvals_view.py."
    )
    assert forbidden == ("app/approvals_view.py",)


def test_build_intake_summary_carries_refinement_memory():
    summary = build_intake_summary(
        "Implement a value helper in app/impl.py.",
        refinement_goal="Now also wire app/wiring.py; do not touch tests/test_skip.py.",
        previous_bundle_touched_files=["tests/test_only.py"],
        previous_proposal_likely_files=["app/impl.py"],
        rejection_text="The previous bundle only added a test.",
    )
    assert "app/impl.py" in summary["named_implementation_files"]
    assert "app/wiring.py" in summary["named_target_paths"]
    assert "tests/test_skip.py" in summary["forbidden_paths"]
    assert summary["previous_bundle_touched_files"] == ["tests/test_only.py"]
    assert summary["rejection_text_present"] is True
    assert summary["requires_implementation_change"] is True


# --------------------------------------------------------------------------- #
# Semantic bundle critique (pure).
# --------------------------------------------------------------------------- #


def test_critique_flags_tests_only_bundle_for_implementation_goal():
    warnings = critique_bundle(
        requires_implementation=True,
        named_implementation_files=("app/impl.py",),
        proposal_likely_files=("app/impl.py",),
        forbidden_paths=(),
        operations=(
            CritiqueOperation(
                op="create_file",
                path="tests/test_impl.py",
                new_content="def test_x():\n    assert True\n",
            ),
        ),
    )
    assert CRITIQUE_TESTS_ONLY_FOR_IMPLEMENTATION in warnings
    assert any(
        CRITIQUE_NAMED_TARGETS_NOT_TOUCHED.split("{")[0] in warning
        for warning in warnings
    )


def test_critique_flags_helper_created_but_never_called():
    warnings = critique_bundle(
        requires_implementation=True,
        named_implementation_files=(),
        proposal_likely_files=(),
        forbidden_paths=(),
        operations=(
            CritiqueOperation(
                op="create_file",
                path="app/helper.py",
                new_content="def compute(value):\n    return value + 1\n",
            ),
            CritiqueOperation(
                op="replace_file",
                path="app/main.py",
                new_content="from app.helper import compute\n\nX = 1\n",
            ),
        ),
    )
    assert any(
        CRITIQUE_HELPER_CREATED_NOT_CALLED.split("{")[0] in warning
        for warning in warnings
    )


def test_critique_clean_for_wired_helper_and_no_false_positive():
    warnings = critique_bundle(
        requires_implementation=True,
        named_implementation_files=("app/main.py",),
        proposal_likely_files=("app/main.py",),
        forbidden_paths=(),
        operations=(
            CritiqueOperation(
                op="create_file",
                path="app/helper.py",
                new_content="def compute(value):\n    return value + 1\n",
            ),
            CritiqueOperation(
                op="replace_file",
                path="app/main.py",
                new_content="from app.helper import compute\n\nX = compute(1)\n",
            ),
        ),
    )
    assert warnings == ()


def test_critique_flags_proposal_likely_files_ignored():
    warnings = critique_bundle(
        requires_implementation=True,
        named_implementation_files=(),
        proposal_likely_files=("app/expected.py",),
        forbidden_paths=(),
        operations=(
            CritiqueOperation(
                op="replace_file",
                path="app/unrelated.py",
                new_content="VALUE = 2\n",
            ),
        ),
    )
    assert any(
        CRITIQUE_LIKELY_FILES_IGNORED.split("{")[0] in warning for warning in warnings
    )


def test_critique_flags_refinement_forbidden_path_touched():
    warnings = critique_bundle(
        requires_implementation=True,
        named_implementation_files=(),
        proposal_likely_files=(),
        forbidden_paths=("app/locked.py",),
        operations=(
            CritiqueOperation(
                op="replace_file",
                path="app/locked.py",
                new_content="VALUE = 9\n",
            ),
        ),
    )
    assert any(
        CRITIQUE_FORBIDDEN_PATH_TOUCHED.split("{")[0] in warning for warning in warnings
    )


# --------------------------------------------------------------------------- #
# Safe response envelope (pure, secret-free).
# --------------------------------------------------------------------------- #


def test_safe_response_envelope_is_bounded_and_secret_free():
    response = SimpleNamespace(
        model="deepseek-test-code",
        id="resp-123",
        object="chat.completion",
        created=1700000000,
        choices=[
            SimpleNamespace(
                finish_reason="length",
                message=SimpleNamespace(content=""),
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=4096, completion_tokens=0, total_tokens=4096
        ),
    )
    envelope = safe_response_envelope(response)
    assert envelope["response_model"] == "deepseek-test-code"
    assert envelope["finish_reason"] == "length"
    assert envelope["content_length"] == 0
    assert envelope["usage"]["prompt_tokens"] == 4096
    # No message body or other unbounded content is captured.
    assert "content" not in envelope
    assert "message" not in envelope


# --------------------------------------------------------------------------- #
# Refinement memory and new-module purity.
# --------------------------------------------------------------------------- #


def test_memory_prompt_section_tells_refinement_not_to_repeat():
    section = _memory_prompt_section(
        {
            "previous_bundle_touched_files": ["tests/test_only.py"],
            "previous_proposal_likely_files": ["app/impl.py"],
            "previous_warnings": [CRITIQUE_TESTS_ONLY_FOR_IMPLEMENTATION],
        }
    )
    assert "tests/test_only.py" in section
    assert "app/impl.py" in section
    assert "Do not repeat" in section
    assert _memory_prompt_section({}) == ""


@pytest.mark.parametrize("module", [intake, bundle_critique])
def test_new_planning_modules_are_pure(module):
    source = Path(module.__file__).read_text(encoding="utf-8").casefold()
    for forbidden in (
        "subprocess",
        "create_deepseek_client",
        "deepseek_client",
        "os.environ",
        "os.getenv",
        "import os",
        "open(",
        "requests",
        "httpx",
        "sqlite",
    ):
        assert forbidden not in source, forbidden


# --------------------------------------------------------------------------- #
# Integration: real slices with fake DeepSeek clients.
# --------------------------------------------------------------------------- #


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    )


def _commit_repo(root: Path) -> Path:
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
            id="resp-1",
            object="chat.completion",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.content),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=120, completion_tokens=0, total_tokens=120
            ),
        )


class _FakeClient:
    def __init__(self, content: str):
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def _read_state_artifact(root: Path, reference: str) -> dict:
    path = root / freestyle.DEV_STATE_DIR_NAME / reference
    return json.loads(path.read_text(encoding="utf-8"))


def _init_named_target_repo(tmp_path: Path) -> Path:
    root = tmp_path / "project"
    (root / "app" / "dev").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "dev" / "freestyle_proposal.py").write_text(
        "DEMO_PROPOSAL = 1\n\n\ndef build():\n    return DEMO_PROPOSAL\n",
        encoding="utf-8",
    )
    for index in range(20):
        (root / "tests" / f"test_decoy_{index}.py").write_text(
            f"def test_decoy_{index}():\n    assert True\n", encoding="utf-8"
        )
    (root / "README.md").write_text("Demo project\n", encoding="utf-8")
    return _commit_repo(root)


def _materialized(root: Path, goal: str):
    job = create_freestyle_job(goal, developer_approved=True, project_root=root)
    return materialize_freestyle_job(job.job_id, project_root=root)


def test_named_implementation_file_outranks_tests_in_context(tmp_path):
    root = _init_named_target_repo(tmp_path)
    goal = "Add a deterministic planning step to app/dev/freestyle_proposal.py."
    job = _materialized(root, goal)

    context = proposal.build_proposal_context(Path(job.worktree_path), goal)
    inspected = list(context.inspected_paths)

    assert "app/dev/freestyle_proposal.py" in inspected
    named_index = inspected.index("app/dev/freestyle_proposal.py")
    test_indexes = [
        index for index, path in enumerate(inspected) if path.startswith("tests/")
    ]
    # The explicitly named implementation file is selected before any test file.
    assert all(named_index < test_index for test_index in test_indexes)
    named_files = context.context_plan["named_target_files"]
    assert "app/dev/freestyle_proposal.py" in named_files["included"]


def test_tests_only_bundle_for_implementation_goal_is_flagged(tmp_path):
    root = tmp_path / "project"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "README.md").write_text("Sample\n", encoding="utf-8")
    _commit_repo(root)

    goal = "Implement a new doubling helper in app/sample.py."
    job = _materialized(root, goal)
    proposed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeClient(
            json.dumps(
                {
                    "proposed_implementation_steps": ["Add a helper to app/sample.py."],
                    "likely_files_to_edit": ["app/sample.py"],
                    "risks_unknowns": [],
                    "suggested_allowlisted_checks": ["pytest"],
                }
            )
        ),
    )
    bundle = json.dumps(
        {
            "schema_version": 1,
            "job_id": proposed.job_id,
            "created_at": "2026-06-15T00:00:00+00:00",
            "operation_count": 1,
            "operations": [
                {
                    "op": "create_file",
                    "path": "tests/test_double.py",
                    "content": "def test_double():\n    assert True\n",
                }
            ],
        }
    )
    completed = generate_freestyle_edit_bundle(
        proposed.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeClient(bundle),
    )

    assert (
        CRITIQUE_TESTS_ONLY_FOR_IMPLEMENTATION
        in completed.edit_bundle_semantic_warnings
    )
    artifact = _read_state_artifact(root, completed.edit_generation_artifact_path)
    assert artifact["semantic_critique"]["advisory_only"] is True
    assert CRITIQUE_TESTS_ONLY_FOR_IMPLEMENTATION in artifact["semantic_critique"][
        "warnings"
    ]


def test_empty_proposal_response_records_rich_secret_free_diagnostics(tmp_path):
    root = tmp_path / "project"
    (root / "app").mkdir(parents=True)
    (root / "app" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "README.md").write_text("Sample\n", encoding="utf-8")
    _commit_repo(root)

    job = _materialized(root, "Implement a helper in app/sample.py.")
    client = _FakeClient("")
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        propose_freestyle_job(
            job.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=client,
        )

    failed = get_freestyle_job(job.job_id, project_root=root)
    failure = _read_state_artifact(root, failed.proposal_failure_artifact_path)
    attempt = failure["proposal_attempts"][0]
    assert attempt["status"] == "empty_response"
    diagnostics = attempt["response_diagnostics"]
    assert diagnostics["response_model"] == "deepseek-test-code"
    assert diagnostics["content_length"] == 0
    assert diagnostics["usage"]["prompt_tokens"] == 120
    serialized = json.dumps(failure)
    assert "test-deepseek-key" not in serialized
    # The raw prompt payload (goal + file contents marshalled for the model) is
    # never stored in the diagnostic.
    assert "safe_files" not in serialized


def test_empty_edit_bundle_response_records_rich_diagnostics(tmp_path):
    root = tmp_path / "project"
    (root / "app").mkdir(parents=True)
    (root / "app" / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    (root / "README.md").write_text("Sample\n", encoding="utf-8")
    _commit_repo(root)

    job = _materialized(root, "Implement a helper in app/sample.py.")
    proposed = propose_freestyle_job(
        job.job_id,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_FakeClient(
            json.dumps(
                {
                    "proposed_implementation_steps": ["Add a helper."],
                    "likely_files_to_edit": ["app/sample.py"],
                    "risks_unknowns": [],
                    "suggested_allowlisted_checks": ["pytest"],
                }
            )
        ),
    )
    with pytest.raises(FreestyleJobRefusedError, match="error was recorded"):
        generate_freestyle_edit_bundle(
            proposed.job_id,
            project_root=root,
            deepseek_config=_config(),
            deepseek_client=_FakeClient(""),
        )

    failed = get_freestyle_job(proposed.job_id, project_root=root)
    artifact = _read_state_artifact(root, failed.edit_generation_artifact_path)
    diagnostics = artifact["response_diagnostics"]
    assert diagnostics["empty_response"] is True
    assert diagnostics["content_length"] == 0
    assert diagnostics["attempt_count"] == 2
    assert diagnostics["repair_retry_used"] is True
    assert isinstance(diagnostics["prompt_token_estimate"], int)
    assert "test-deepseek-key" not in json.dumps(artifact)
