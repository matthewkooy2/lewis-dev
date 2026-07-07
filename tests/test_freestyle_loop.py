from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import app.dev.freestyle_coding as freestyle
from app.dev.freestyle_coding import create_freestyle_job, run_freestyle_loop
from app.dev.freestyle_loop import (
    FreestyleLoopLimits,
    get_freestyle_loop_state,
    select_loop_context,
)
from app.dev.freestyle_provider import LOCAL_OPENAI_PROVIDER, ProviderProposalConfig


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        shell=False,
    )
    return completed.stdout.decode().strip()


def _init_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    root = tmp_path / "project"
    root.mkdir()
    (root / ".gitignore").write_text(".lewis-dev/\n", encoding="utf-8")
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _config(*, local_schema: bool = False, text_mode: bool = False):
    if local_schema or text_mode:
        return ProviderProposalConfig(
            api_key="",
            model="qwen2.5-coder-7b-instruct",
            base_url="http://127.0.0.1:1234/v1",
            provider=LOCAL_OPENAI_PROVIDER,
            local_only=True,
            supports_json_response_format=local_schema,
        )
    return ProviderProposalConfig(
        api_key="test-key",
        model="deepseek-test-code",
        base_url="https://api.deepseek.com",
    )


class _SequenceCompletions:
    def __init__(self, contents: list[str]):
        self.contents = contents
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        return SimpleNamespace(
            model="test-model",
            choices=[
                SimpleNamespace(
                    finish_reason="stop",
                    message=SimpleNamespace(content=self.contents[index]),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
            ),
        )


class _SequenceClient:
    def __init__(self, contents: list[str]):
        self.completions = _SequenceCompletions(contents)
        self.chat = SimpleNamespace(completions=self.completions)


def _proposal(*paths: str) -> str:
    return json.dumps(
        {
            "proposed_implementation_steps": ["Make the requested code change."],
            "likely_files_to_edit": list(paths),
            "risks_unknowns": [],
            "suggested_allowlisted_checks": ["pytest"],
        }
    )


def _critic(approved: bool = True) -> str:
    return json.dumps(
        {
            "approved": approved,
            "blocking_issues": [] if approved else ["Need a narrower plan."],
            "missing_context": [],
            "assumptions": [],
            "revised_focus": [],
            "next_action": "Proceed." if approved else "Revise the plan.",
        }
    )


def _final(blocking: list[str] | None = None) -> str:
    issues = blocking or []
    return json.dumps(
        {
            "goal_satisfied": not issues,
            "blocking_issues": issues,
            "non_blocking_concerns": [],
            "tests_considered": ["pytest-freestyle"],
            "approval_recommendation": (
                "needs_human_changes" if issues else "present_for_human_approval"
            ),
            "summary": "Reviewed the isolated diff.",
        }
    )


def _bundle(path: str, old: str, new: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "operations": [
                {"op": "replace_text", "path": path, "old_text": old, "new_text": new}
            ],
        }
    )


def _source_snapshot(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): path.read_text(encoding="utf-8")
        for path in root.rglob("*")
        if path.is_file()
        and ".git" not in path.relative_to(root).parts
        and freestyle.DEV_STATE_DIR_NAME not in path.relative_to(root).parts
    }


def test_loop_reaches_human_approval_after_checks_and_persists_state(tmp_path):
    root = _init_repo(
        tmp_path,
        {
            "sample.py": "VALUE = 1\n",
            "tests/test_freestyle_sample.py": (
                "from sample import VALUE\n\n"
                "def test_freestyle_value():\n"
                "    assert VALUE == 2\n"
            ),
        },
    )
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    before = _source_snapshot(root)
    client = _SequenceClient(
        [
            _proposal("sample.py", "tests/test_freestyle_sample.py"),
            _critic(),
            _bundle("sample.py", "VALUE = 1", "VALUE = 2"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        check_ids=["pytest-freestyle"],
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    assert completed.latest_loop_stop_reason == "ready_for_human_approval"
    assert _source_snapshot(root) == before
    assert (Path(completed.worktree_path) / "sample.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 2\n"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    stages = [stage["stage"] for stage in state["stages"]]
    assert "planner" in stages
    assert "plan_critic" in stages
    assert "implementer" in stages
    assert "run_checks" in stages
    assert "final_reviewer" in stages
    assert state["human_approval_required"] is True
    assert "No commit, merge, push" in state["integration_statement"]
    assert state["metrics"]["model_calls_by_stage"]["planner"] == 1
    assert state["metrics"]["final_review_result"] == "clear"


def test_loop_uses_json_schema_mode_for_local_structured_stages(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    client = _SequenceClient(
        [
            _proposal("sample.py"),
            _critic(),
            _bundle("sample.py", "VALUE = 1", "VALUE = 2"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(local_schema=True),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    assert len(client.completions.calls) == 4
    assert all(
        call["response_format"]["type"] == "json_schema"
        for call in client.completions.calls
    )
    assert all(
        "schema" in call["response_format"]["json_schema"]
        for call in client.completions.calls
    )
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["provider"]["structured_output_mode"] == "json_schema"


def test_text_mode_malformed_loop_stage_gets_one_format_repair(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    client = _SequenceClient(
        [
            _proposal("sample.py"),
            "not json",
            _critic(),
            _bundle("sample.py", "VALUE = 1", "VALUE = 2"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(text_mode=True),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    assert len(client.completions.calls) == 5
    assert all("response_format" not in call for call in client.completions.calls)
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    critic = next(stage for stage in state["stages"] if stage["stage"] == "plan_critic")
    assert [attempt["status"] for attempt in critic["attempts"]] == [
        "format_error",
        "valid",
    ]
    assert "not json" in critic["attempts"][0]["raw_model_output"]


def test_plan_critic_prose_missing_context_is_advisory_when_file_included(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    critic = json.dumps(
        {
            "approved": False,
            "blocking_issues": ["Review sample.py before editing."],
            "missing_context": ["Implementation details in sample.py are not provided."],
            "assumptions": [],
            "revised_focus": ["Use the included sample.py source."],
            "next_action": "Review sample.py.",
        }
    )
    client = _SequenceClient(
        [
            _proposal("sample.py"),
            critic,
            _bundle("sample.py", "VALUE = 1", "VALUE = 2"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
        limits=FreestyleLoopLimits(max_planning_passes=1),
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    decisions = [item for item in state["attempts"] if item.get("stage") == "planner"]
    assert decisions[-1]["decision"] == "proceed_after_critic_warnings"


def test_loop_diagnoses_and_repairs_failing_tests(tmp_path):
    root = _init_repo(
        tmp_path,
        {
            "mathlib.py": "def multiply(a, b):\n    return a + b\n",
            "tests/test_freestyle_mathlib.py": (
                "from mathlib import multiply\n\n"
                "def test_freestyle_multiply():\n"
                "    assert multiply(2, 3) == 6\n"
            ),
        },
    )
    job = create_freestyle_job(
        "Make multiply(a, b) return the product.",
        developer_approved=True,
        project_root=root,
    )
    diagnosis = json.dumps(
        {
            "summary": "multiply still returns the wrong arithmetic result.",
            "root_cause": "The patch changed the operator but not to multiplication.",
            "failure_kind": "test_failure",
            "failed_command": "pytest-freestyle",
            "affected_files": ["mathlib.py"],
            "relevant_tests": ["tests/test_freestyle_mathlib.py"],
            "evidence": ["Expected 6 for multiply(2, 3)."],
            "repair_strategy": "Replace the operator with * only.",
            "confidence": "high",
        }
    )
    client = _SequenceClient(
        [
            _proposal("mathlib.py", "tests/test_freestyle_mathlib.py"),
            _critic(),
            _bundle("mathlib.py", "return a + b", "return a - b"),
            diagnosis,
            _bundle("mathlib.py", "return a - b", "return a * b"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        check_ids=["pytest-freestyle"],
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    assert (Path(completed.worktree_path) / "mathlib.py").read_text(
        encoding="utf-8"
    ) == "def multiply(a, b):\n    return a * b\n"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["diagnoses"][0]["failure_kind"] == "test_failure"
    assert state["repairs"][0]["cycle"] == 1
    assert state["metrics"]["model_calls_by_stage"]["diagnostician"] == 1
    assert state["metrics"]["model_calls_by_stage"]["repairer"] == 1


def test_safety_failure_stops_without_diagnosis_or_repair(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Unsafely write credentials to .env.",
        developer_approved=True,
        project_root=root,
    )
    unsafe_bundle = json.dumps(
        {
            "schema_version": 1,
            "operations": [
                {"op": "create_file", "path": ".env", "content": "TOKEN=bad\n"}
            ],
        }
    )
    client = _SequenceClient([_proposal("sample.py"), _critic(), unsafe_bundle])

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "safety_stop"
    assert completed.latest_loop_stop_reason == "safety_stop"
    assert len(client.completions.calls) == 3
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["diagnoses"] == []
    assert state["repairs"] == []


def test_repeated_test_failure_stops_instead_of_looping(tmp_path):
    root = _init_repo(
        tmp_path,
        {
            "mathlib.py": "def multiply(a, b):\n    return a + b\n",
            "README.md": "mathlib\n",
            "tests/test_freestyle_mathlib.py": (
                "from mathlib import multiply\n\n"
                "def test_freestyle_multiply():\n"
                "    assert multiply(2, 3) == 6\n"
            ),
        },
    )
    job = create_freestyle_job(
        "Make multiply(a, b) return the product.",
        developer_approved=True,
        project_root=root,
    )
    diagnosis = json.dumps(
        {
            "summary": "The math implementation is still wrong.",
            "root_cause": "The implementation was not repaired.",
            "failure_kind": "test_failure",
            "failed_command": "pytest-freestyle",
            "affected_files": ["mathlib.py"],
            "relevant_tests": ["tests/test_freestyle_mathlib.py"],
            "evidence": ["The same assertion still fails."],
            "repair_strategy": "Change mathlib.py, not documentation.",
            "confidence": "high",
        }
    )
    client = _SequenceClient(
        [
            _proposal("mathlib.py", "tests/test_freestyle_mathlib.py"),
            _critic(),
            _bundle("mathlib.py", "return a + b", "return a - b"),
            diagnosis,
            _bundle("README.md", "mathlib", "mathlib\n"),
            diagnosis,
            _bundle("README.md", "mathlib", "mathlib2"),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        check_ids=["pytest-freestyle"],
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "stopped"
    assert completed.latest_loop_stop_reason == "repeated_failure"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["metrics"]["repeated_failure_detected"] is True
    # The first repeated failure escalated to one full-rewrite attempt; only
    # the repeat after that escalation stopped the loop.
    decisions = [item.get("decision") for item in state["attempts"]]
    assert "escalate_after_repeated_failure" in decisions
    assert "strategy_escalation_full_rewrite" in decisions


def test_model_call_limit_stops_before_more_model_work(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    client = _SequenceClient([_proposal("sample.py")])

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
        limits=FreestyleLoopLimits(max_model_calls=1),
    )

    assert completed.latest_loop_status == "exhausted"
    assert completed.latest_loop_stop_reason == "model_call_limit"
    assert len(client.completions.calls) == 1


def test_wall_clock_limit_stops_loop(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    client = _SequenceClient([_proposal("sample.py")])

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
        limits=FreestyleLoopLimits(max_wall_clock_seconds=0),
    )

    assert completed.latest_loop_status == "exhausted"
    assert completed.latest_loop_stop_reason == "time_limit"


def test_trace_assisted_context_selection_uses_trace_paths(tmp_path):
    root = tmp_path / "trace-project"
    (root / "app").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "app" / "calc.py").write_text(
        "def add(a, b):\n"
        "    return a + b\n\n"
        "def total(values):\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result = add(result, value)\n"
        "    return result\n",
        encoding="utf-8",
    )
    (root / "tests" / "test_calc.py").write_text(
        "from app.calc import total\n\n"
        "def test_total():\n"
        "    assert total([1, 2]) == 3\n",
        encoding="utf-8",
    )

    context = select_loop_context(root, "Fix `total` so it handles empty lists.")

    assert context["trace_signals"][0]["symbol"] == "total"
    included = {item["path"] for item in context["files"]}
    assert "app/calc.py" in included


def _create_existing_bundle(path: str, content: str) -> str:
    return json.dumps(
        {
            "schema_version": 1,
            "operations": [{"op": "create_file", "path": path, "content": content}],
        }
    )


def test_loop_recovers_from_recoverable_bundle_failure(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    client = _SequenceClient(
        [
            _proposal("sample.py"),
            _critic(),
            _create_existing_bundle("sample.py", "VALUE = 2\n"),
            _bundle("sample.py", "VALUE = 1", "VALUE = 2"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    recoveries = state["generation_recoveries"]
    assert len(recoveries) == 1
    assert recoveries[0]["category"] == "create_existing_file"
    assert recoveries[0]["failed_path"] == "sample.py"
    assert state["metrics"]["generation_recovery_count"] == 1
    assert state["metrics"]["model_calls_by_category"]["recovery"] == 1
    failed_stage = next(
        record
        for record in state["stages"]
        if record["stage"] == "implementer" and record["status"] == "failed"
    )
    assert failed_stage["failure_category"] == "create_existing_file"
    recovery_payload = json.loads(
        client.completions.calls[3]["messages"][1]["content"]
    )
    failure_context = recovery_payload["stage_context"]["previous_generation_failure"]
    assert failure_context["category"] == "create_existing_file"
    assert len(failure_context["current_file_sha256"]) == 64
    assert "never use create_file" in failure_context["instruction"]


def test_stale_anchor_exhaustion_stops_with_precise_reason(tmp_path):
    root = _init_repo(tmp_path, {"sample.py": "VALUE = 1\n"})
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    client = _SequenceClient(
        [
            _proposal("sample.py"),
            _critic(),
            _bundle("sample.py", "VALUE = 999", "VALUE = 2"),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "stopped"
    assert completed.latest_loop_stop_reason == "stale_anchor"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["stop_reason"] == "stale_anchor"
    assert "stale-anchor" in state["next_action"]
    # One in-generation anchor retry plus the full loop-level recovery budget
    # (two regenerations, each with its own anchor retry): recovery stayed
    # bounded and the loop stopped with the precise reason.
    assert state["metrics"]["generation_recovery_count"] == 2
    assert len(client.completions.calls) == 8


def test_final_review_budget_is_reserved_for_final_review_only(tmp_path):
    files = {"sample.py": "VALUE = 1\n"}
    sequence = [
        _proposal("sample.py"),
        _critic(),
        _bundle("sample.py", "VALUE = 1", "VALUE = 2"),
        _final(),
    ]

    (tmp_path / "enough").mkdir()
    root = _init_repo(tmp_path / "enough", files)
    job = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root,
    )
    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=_SequenceClient(list(sequence)),
        limits=FreestyleLoopLimits(max_model_calls=4, reserved_final_review_calls=1),
    )
    assert completed.latest_loop_status == "ready_for_human_approval"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["metrics"]["model_calls_total"] == 4
    assert state["metrics"]["model_calls_by_category"]["final_review"] == 1

    (tmp_path / "tight").mkdir()
    root_tight = _init_repo(tmp_path / "tight", files)
    job_tight = create_freestyle_job(
        "Change sample.py so VALUE equals 2.",
        developer_approved=True,
        project_root=root_tight,
    )
    completed_tight = run_freestyle_loop(
        job_tight.job_id,
        run_checks=False,
        project_root=root_tight,
        deepseek_config=_config(),
        deepseek_client=_SequenceClient(list(sequence)),
        limits=FreestyleLoopLimits(max_model_calls=3, reserved_final_review_calls=1),
    )
    assert completed_tight.latest_loop_status == "exhausted"
    assert completed_tight.latest_loop_stop_reason == "model_call_limit"


def test_context_stop_happens_after_single_critic_pass(tmp_path):
    root = _init_repo(tmp_path, {"normalizer.py": "def f(x):\n    return x\n"})
    job = create_freestyle_job(
        "Only modify missing_required_context.py; stop if it is unavailable.",
        developer_approved=True,
        project_root=root,
    )
    critic = json.dumps(
        {
            "approved": False,
            "blocking_issues": ["The named file is not in context."],
            "missing_context": ["missing_required_context.py"],
            "assumptions": [],
            "revised_focus": [],
            "next_action": "Provide missing_required_context.py.",
        }
    )
    client = _SequenceClient([_proposal("missing_required_context.py"), critic])

    completed = run_freestyle_loop(
        job.job_id,
        run_checks=False,
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_stop_reason == "required_context_unavailable"
    # The stop happens after one planner and one critic call; no further
    # planning or generation is spent on unavailable context.
    assert len(client.completions.calls) == 2


def test_repair_prompt_carries_fresh_evidence_and_revisions_advance(tmp_path):
    root = _init_repo(
        tmp_path,
        {
            "mathlib.py": "def multiply(a, b):\n    return a + b\n",
            "tests/test_freestyle_mathlib.py": (
                "from mathlib import multiply\n\n"
                "def test_freestyle_multiply():\n"
                "    assert multiply(2, 3) == 6\n"
            ),
        },
    )
    job = create_freestyle_job(
        "Make multiply(a, b) return the product.",
        developer_approved=True,
        project_root=root,
    )
    diagnosis = json.dumps(
        {
            "summary": "multiply still returns the wrong arithmetic result.",
            "root_cause": "The patch changed the operator but not to multiplication.",
            "failure_kind": "test_failure",
            "failed_command": "pytest-freestyle",
            "affected_files": ["mathlib.py"],
            "relevant_tests": ["tests/test_freestyle_mathlib.py"],
            "evidence": ["Expected 6 for multiply(2, 3)."],
            "repair_strategy": "Replace the operator with * only.",
            "confidence": "high",
        }
    )
    client = _SequenceClient(
        [
            _proposal("mathlib.py", "tests/test_freestyle_mathlib.py"),
            _critic(),
            _bundle("mathlib.py", "return a + b", "return a - b"),
            diagnosis,
            _bundle("mathlib.py", "return a - b", "return a * b"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        check_ids=["pytest-freestyle"],
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    repair_payload = json.loads(client.completions.calls[4]["messages"][1]["content"])
    stage_context = repair_payload["stage_context"]
    assert "test_freestyle_multiply" in (
        stage_context["latest_failed_check_output"]["stdout_tail"]
        + stage_context["latest_failed_check_output"]["stderr_tail"]
    )
    affected = stage_context["current_source_of_affected_files"]
    assert affected[0]["path"] == "mathlib.py"
    assert "return a - b" in affected[0]["current_content"]
    assert affected[0]["complete"] is True
    assert len(affected[0]["sha256"]) == 64
    assert "return a + b" in stage_context["accumulated_diff"]
    assert stage_context["worktree_revision"] == 1
    assert any(
        "no longer exist" in constraint
        for constraint in stage_context["repair_constraints"]
    )

    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["worktree_revision"] == 2
    assert state["file_revisions"]["mathlib.py"]["revision"] == 2
    assert state["relevant_context"]["captured_at_revision"] == 2
    assert state["relevant_context"]["captured_by_stage"] == "repair_1_apply"
    assert state["repairs"][0]["result"] == "tests_passed"
    assert state["diagnoses"][0]["cycle"] == 1
    diagnostician_payload = json.loads(
        client.completions.calls[3]["messages"][1]["content"]
    )
    assert diagnostician_payload["previous_diagnoses"] == []
    assert diagnostician_payload["worktree_revision"] == 1


def test_repeated_equivalent_diagnosis_stops_the_loop(tmp_path):
    root = _init_repo(
        tmp_path,
        {
            "mathlib.py": "def multiply(a, b):\n    return a + b\n",
            "tests/test_freestyle_mathlib.py": (
                "from mathlib import multiply\n\n"
                "def test_freestyle_multiply():\n"
                "    assert multiply(2, 3) == 6\n"
            ),
        },
    )
    job = create_freestyle_job(
        "Make multiply(a, b) return the product.",
        developer_approved=True,
        project_root=root,
    )
    diagnosis = json.dumps(
        {
            "summary": "multiply returns the wrong arithmetic result.",
            "root_cause": "The operator is wrong.",
            "failure_kind": "test_failure",
            "failed_command": "pytest-freestyle",
            "affected_files": ["mathlib.py"],
            "relevant_tests": ["tests/test_freestyle_mathlib.py"],
            "evidence": ["Expected 6 for multiply(2, 3)."],
            "repair_strategy": "Fix the operator.",
            "confidence": "high",
        }
    )
    client = _SequenceClient(
        [
            _proposal("mathlib.py", "tests/test_freestyle_mathlib.py"),
            _critic(),
            _bundle("mathlib.py", "return a + b", "return a - b"),
            diagnosis,
            _bundle("mathlib.py", "return a - b", "return a // b"),
            diagnosis,
            _bundle("mathlib.py", "return a // b", "return a % b"),
            diagnosis,
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        check_ids=["pytest-freestyle"],
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "stopped"
    assert completed.latest_loop_stop_reason == "repeated_diagnosis"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    assert state["metrics"]["repeated_failure_detected"] is True
    # The first repeat escalated to a full rewrite instead of stopping; only
    # the second repeat (after the escalated repair also failed) stops.
    escalations = [
        item
        for item in state["attempts"]
        if item.get("decision") == "strategy_escalation_full_rewrite"
    ]
    assert len(escalations) == 1
    assert [repair["strategy"] for repair in state["repairs"]] == [
        "minimal",
        "full_rewrite",
    ]
    assert state["repairs"][-1]["result"] == "tests_failed"
    escalated_payload = json.loads(
        client.completions.calls[6]["messages"][1]["content"]
    )
    escalated_context = escalated_payload["stage_context"]
    assert escalated_context["repair_strategy_escalation"] == "full_rewrite"
    assert any(
        "Rewrite the affected function" in constraint
        for constraint in escalated_context["repair_constraints"]
    )
    reference_entries = [
        entry
        for entry in escalated_context["current_source_of_affected_files"]
        if entry.get("reference_only")
    ]
    assert any(
        entry["path"] == "tests/test_freestyle_mathlib.py"
        for entry in reference_entries
    )


def test_failed_repair_generation_abandons_cycle_and_next_cycle_recovers(tmp_path):
    root = _init_repo(
        tmp_path,
        {
            "mathlib.py": "def multiply(a, b):\n    return a + b\n",
            "tests/test_freestyle_mathlib.py": (
                "from mathlib import multiply\n\n"
                "def test_freestyle_multiply():\n"
                "    assert multiply(2, 3) == 6\n"
            ),
        },
    )
    job = create_freestyle_job(
        "Make multiply(a, b) return the product.",
        developer_approved=True,
        project_root=root,
    )
    diagnosis = json.dumps(
        {
            "summary": "multiply still returns the wrong arithmetic result.",
            "root_cause": "The patch changed the operator but not to multiplication.",
            "failure_kind": "test_failure",
            "failed_command": "pytest-freestyle",
            "affected_files": ["mathlib.py"],
            "relevant_tests": ["tests/test_freestyle_mathlib.py"],
            "evidence": ["Expected 6 for multiply(2, 3)."],
            "repair_strategy": "Replace the operator with * only.",
            "confidence": "high",
        }
    )
    stale = _bundle("mathlib.py", "return a * c", "return a * b")
    client = _SequenceClient(
        [
            _proposal("mathlib.py", "tests/test_freestyle_mathlib.py"),
            _critic(),
            _bundle("mathlib.py", "return a + b", "return a - b"),
            diagnosis,
            # Cycle 1: the initial attempt plus both budgeted recovery
            # regenerations keep anchoring on stale text (each attempt also
            # gets one in-generation anchor retry).
            stale,
            stale,
            stale,
            stale,
            stale,
            stale,
            diagnosis,
            _bundle("mathlib.py", "return a - b", "return a * b"),
            _final(),
        ]
    )

    completed = run_freestyle_loop(
        job.job_id,
        check_ids=["pytest-freestyle"],
        project_root=root,
        deepseek_config=_config(),
        deepseek_client=client,
    )

    assert completed.latest_loop_status == "ready_for_human_approval"
    state = get_freestyle_loop_state(completed.job_id, project_root=root)
    abandoned = [
        item
        for item in state["attempts"]
        if item.get("decision") == "cycle_abandoned_after_generation_failure"
    ]
    assert len(abandoned) == 1
    assert abandoned[0]["cycle"] == 1
    assert abandoned[0]["stop_reason"] == "stale_anchor"
    assert state["repairs"][0]["cycle"] == 2
    assert state["repairs"][0]["result"] == "tests_passed"
    assert state["metrics"]["generation_recovery_count"] == 2
