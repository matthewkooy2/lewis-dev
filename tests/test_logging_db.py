import json
import sqlite3

from app.logging_db import RunLogger


def test_recent_usage_summary_totals_usage_fields(tmp_path):
    logger = RunLogger(tmp_path / "test.sqlite3")
    logger.log_run(
        user_input="hello",
        model="test-model",
        tool_calls=[],
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        prompt_cache_hit_tokens=75,
        prompt_cache_miss_tokens=25,
        api_calls=2,
        estimated_cost=0.0123,
    )

    summary = logger.recent_usage_summary()

    assert summary.input_tokens == 100
    assert summary.output_tokens == 20
    assert summary.prompt_cache_hit_tokens == 75
    assert summary.prompt_cache_miss_tokens == 25
    assert summary.estimated_cost == 0.0123
    assert summary.api_calls == 2
    assert summary.run_count == 1


def test_memory_tool_call_log_redacts_proposed_content(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    logger = RunLogger(db_path)
    proposed_content = "Blake prefers concise implementation summaries."

    logger.log_run(
        user_input=f"remember that {proposed_content}",
        model="test-model",
        tool_calls=[
            {
                "name": "propose_memory",
                "arguments": json.dumps(
                    {
                        "memory_type": "preference",
                        "content": proposed_content,
                        "source": "chat",
                    }
                ),
                "result": {
                    "ok": True,
                    "approval_id": 1,
                    "status": "pending",
                    "message": "Created pending memory approval #1.",
                },
            }
        ],
    )

    with sqlite3.connect(db_path) as conn:
        stored = conn.execute("SELECT tool_calls FROM runs").fetchone()[0]

    assert proposed_content not in stored
    calls = json.loads(stored)
    assert calls[0]["name"] == "propose_memory"
    arguments = json.loads(calls[0]["arguments"])
    assert arguments["content"] == "[redacted]"
    assert arguments["memory_type"] == "preference"
