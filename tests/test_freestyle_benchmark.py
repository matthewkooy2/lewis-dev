from __future__ import annotations

from scripts import freestyle_local_benchmark as bench


def test_benchmark_outcome_distinguishes_failed_tests_from_success():
    result = {
        "expect_rejection": False,
        "job": {
            "proposal_status": "proposed",
            "edit_generation_status": "generated",
            "edit_bundle_validation_status": "valid",
            "edit_apply_status": "applied",
            "checks_status": "failed",
        },
        "loop": {"status": "stopped", "stop_reason": "repeated_failure"},
    }

    milestones = bench._milestones(result)
    result["milestones"] = milestones

    assert bench.OUTCOME_TESTS_FAILED in milestones
    assert bench._classify_outcome(result) == bench.OUTCOME_TESTS_FAILED
    assert "completed" not in milestones


def test_benchmark_outcome_marks_expected_safety_rejection():
    result = {
        "expect_rejection": True,
        "job": {"checks_status": "not_run"},
        "loop": {"status": "safety_stop", "stop_reason": "safety_stop"},
    }

    milestones = bench._milestones(result)
    result["milestones"] = milestones

    assert bench.OUTCOME_SAFETY_STOP in milestones
    assert bench._classify_outcome(result) == bench.OUTCOME_REJECTED_AS_EXPECTED


def test_repair_cycle_exhaustion_reports_tests_failed_not_budget_exhaustion():
    result = {
        "expect_rejection": False,
        "job": {
            "proposal_status": "proposed",
            "edit_generation_status": "generated",
            "edit_bundle_validation_status": "valid",
            "edit_apply_status": "applied",
            "checks_status": "failed",
        },
        "loop": {"status": "exhausted", "stop_reason": "repair_cycles_exhausted"},
    }

    result["milestones"] = bench._milestones(result)

    assert bench.OUTCOME_LOOP_EXHAUSTED not in result["milestones"]
    assert bench._classify_outcome(result) == bench.OUTCOME_TESTS_FAILED


def test_model_call_budget_stop_still_reports_loop_exhausted():
    result = {
        "expect_rejection": False,
        "job": {
            "proposal_status": "proposed",
            "edit_generation_status": "generated",
            "edit_bundle_validation_status": "valid",
            "edit_apply_status": "applied",
            "checks_status": "failed",
        },
        "loop": {"status": "exhausted", "stop_reason": "model_call_limit"},
    }

    result["milestones"] = bench._milestones(result)

    assert bench._classify_outcome(result) == bench.OUTCOME_LOOP_EXHAUSTED


def test_per_case_statistics_aggregate_repeated_runs():
    report = {
        "cases": [
            {
                "case_id": "simple-one-file",
                "outcome": bench.OUTCOME_TESTS_PASSED,
                "outcome_expected": True,
                "duration_seconds": 10.0,
                "loop": {
                    "stop_reason": "ready_for_human_approval",
                    "model_calls_total": 4,
                    "generation_recovery_count": 0,
                    "recovery_calls_total": 0,
                },
            },
            {
                "case_id": "simple-one-file",
                "outcome": bench.OUTCOME_TESTS_FAILED,
                "outcome_expected": False,
                "duration_seconds": 30.0,
                "loop": {
                    "stop_reason": "repair_cycles_exhausted",
                    "model_calls_total": 12,
                    "generation_recovery_count": 1,
                    "recovery_calls_total": 2,
                },
            },
        ]
    }

    statistics = bench._per_case_statistics(report)
    stats = statistics["simple-one-file"]

    assert stats["runs"] == 2
    assert stats["outcomes"] == {
        bench.OUTCOME_TESTS_PASSED: 1,
        bench.OUTCOME_TESTS_FAILED: 1,
    }
    assert stats["expected_outcome_count"] == 1
    assert stats["expected_outcome_rate"] == 0.5
    assert stats["mean_duration_seconds"] == 20.0
    assert stats["mean_model_calls"] == 8.0
    assert stats["generation_recoveries_total"] == 1
    assert stats["recovery_calls_total"] == 2
    assert stats["stop_reasons"] == {
        "ready_for_human_approval": 1,
        "repair_cycles_exhausted": 1,
    }


def test_benchmark_defaults_to_json_schema_mode_explicitly(monkeypatch, capsys):
    monkeypatch.setenv("LEWIS_FREESTYLE_PROVIDER", "local-openai")
    monkeypatch.delenv("LEWIS_LOCAL_JSON_MODE", raising=False)

    config = bench._resolve_benchmark_config(None, None)

    assert config.structured_output_mode == "json_schema"
    assert config.supports_json_response_format is True
    out = capsys.readouterr().out
    assert "json_schema" in out
    assert "benchmark default" in out


def test_benchmark_text_mode_is_explicit_and_warns(monkeypatch, capsys):
    monkeypatch.setenv("LEWIS_FREESTYLE_PROVIDER", "local-openai")
    monkeypatch.delenv("LEWIS_LOCAL_JSON_MODE", raising=False)

    config = bench._resolve_benchmark_config("text", 7)

    assert config.structured_output_mode == "text"
    assert config.supports_json_response_format is False
    assert config.sampling_seed == 7
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "seed: 7" in out


def test_every_case_declares_expected_outcomes():
    for case in bench.CASES:
        assert case["expected_outcomes"], case["case_id"]
