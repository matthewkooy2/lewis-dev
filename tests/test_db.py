import json
import sqlite3

from app.approvals import ApprovalStore
from app.dashboard import get_dashboard_summary, list_history_items
from app.db import initialize_database
from app.logging_db import RunLogger


def _table_names(db_path):
    with sqlite3.connect(db_path) as conn:
        return {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }


def _column_names(db_path, table_name):
    with sqlite3.connect(db_path) as conn:
        return [row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")]


def _migration_rows(db_path):
    with sqlite3.connect(db_path) as conn:
        return conn.execute(
            "SELECT version, name FROM schema_migrations ORDER BY version"
        ).fetchall()


def test_initialize_empty_database_creates_core_schema(tmp_path):
    db_path = tmp_path / "test.sqlite3"

    initialize_database(db_path)

    assert {
        "schema_migrations",
        "approvals",
        "runs",
        "memories",
        "memory_events",
        "coding_agent_repositories",
        "coding_agent_jobs",
        "coding_agent_events",
        "projects",
        "project_tags",
        "project_note_categories",
        "project_notes",
        "project_note_tags",
        "project_map_runs",
        "project_map_stages",
        "project_indexed_files",
        "project_indexed_symbols",
        "project_relationships",
        "project_summaries",
        "project_diagrams",
        "project_activity",
    } <= _table_names(db_path)
    assert _migration_rows(db_path) == [
        (1, "baseline_schema"),
        (2, "memory_schema"),
        (3, "coding_agents_schema"),
        (4, "projects_schema"),
    ]
    assert _column_names(db_path, "approvals") == [
        "id",
        "action_type",
        "title",
        "payload",
        "status",
        "created_at",
        "updated_at",
    ]
    assert _column_names(db_path, "runs") == [
        "id",
        "timestamp",
        "user_input",
        "model",
        "tool_calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "estimated_cost",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "reasoning_tokens",
        "api_calls",
    ]
    assert _column_names(db_path, "memories") == [
        "id",
        "approval_id",
        "memory_type",
        "content",
        "source",
        "tags_json",
        "status",
        "created_at",
        "updated_at",
        "archived_at",
        "archive_reason",
    ]
    assert _column_names(db_path, "memory_events") == [
        "id",
        "memory_id",
        "approval_id",
        "event_type",
        "actor",
        "details_json",
        "created_at",
    ]
    assert _column_names(db_path, "coding_agent_jobs") == [
        "id",
        "provider",
        "repository_root",
        "repository_label",
        "goal",
        "model",
        "effort",
        "max_turns",
        "timeout_seconds",
        "external_session_id",
        "state",
        "created_at",
        "started_at",
        "completed_at",
        "updated_at",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_tokens",
        "token_usage_source",
        "estimated_cost_usd",
        "cost_source",
        "duration_seconds",
        "failure_category",
        "cancellation_requested",
        "cancellation_requested_at",
        "last_error",
        "project_id",
    ]


def test_initialize_populated_database_preserves_approval_payload_json(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    payload = {
        "to": "Alex",
        "body": "Hello",
        "nested": {"keep": ["exact", "json", 1]},
    }
    raw_payload = json.dumps(payload, sort_keys=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE approvals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_type TEXT NOT NULL,
                title TEXT NOT NULL,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO approvals
                (action_type, title, payload, status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                "propose_email_draft",
                "Email Alex",
                raw_payload,
                "pending",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )

    initialize_database(db_path)

    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT payload FROM approvals WHERE id = 1").fetchone()
        count = conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]
    assert count == 1
    assert row[0] == raw_payload
    assert ApprovalStore(db_path).get(1).payload == payload


def test_initialize_old_style_runs_table_preserves_rows_and_adds_columns(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_input TEXT NOT NULL,
                model TEXT NOT NULL,
                tool_calls TEXT NOT NULL,
                input_tokens INTEGER NOT NULL DEFAULT 0,
                output_tokens INTEGER NOT NULL DEFAULT 0,
                total_tokens INTEGER NOT NULL DEFAULT 0,
                estimated_cost REAL NOT NULL DEFAULT 0.0
            )
            """
        )
        conn.execute(
            """
            INSERT INTO runs (
                timestamp, user_input, model, tool_calls, input_tokens,
                output_tokens, total_tokens, estimated_cost
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "2026-01-01T00:00:00+00:00",
                "hello",
                "test-model",
                "[]",
                10,
                5,
                15,
                0.01,
            ),
        )

    initialize_database(db_path)

    columns = _column_names(db_path, "runs")
    assert "prompt_cache_hit_tokens" in columns
    assert "prompt_cache_miss_tokens" in columns
    assert "reasoning_tokens" in columns
    assert "api_calls" in columns
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT user_input, input_tokens, output_tokens, total_tokens,
                   prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                   reasoning_tokens, api_calls, estimated_cost
            FROM runs WHERE id = 1
            """
        ).fetchone()
    assert row == ("hello", 10, 5, 15, 0, 0, 0, 0, 0.01)
    assert RunLogger(db_path).recent_usage_summary().run_count == 1


def test_initialize_twice_does_not_duplicate_migrations_or_alter_counts(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    store = ApprovalStore(db_path)
    item = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    logger = RunLogger(db_path)
    logger.log_run(
        user_input="hello",
        model="test-model",
        tool_calls=[],
        input_tokens=1,
        output_tokens=2,
        total_tokens=3,
    )

    initialize_database(db_path)
    initialize_database(db_path)

    assert _migration_rows(db_path) == [
        (1, "baseline_schema"),
        (2, "memory_schema"),
        (3, "coding_agents_schema"),
        (4, "projects_schema"),
    ]
    with sqlite3.connect(db_path) as conn:
        approval_count = conn.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]
        run_count = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert approval_count == 1
    assert run_count == 1
    assert ApprovalStore(db_path).get(item.id).payload == item.payload


def test_stores_and_dashboard_history_usage_work_after_initialization(tmp_path):
    db_path = tmp_path / "test.sqlite3"
    initialize_database(db_path)
    store = ApprovalStore(db_path)
    logger = RunLogger(db_path)
    pending_task = store.create(
        action_type="propose_task",
        title="Submit form",
        payload={"title": "Submit form", "description": "Submit it."},
    )
    completed_task = store.create(
        action_type="propose_task",
        title="Call dentist",
        payload={"title": "Call dentist", "description": "Schedule a cleaning."},
    )
    store.update_status(completed_task.id, "completed")
    logger.log_run(
        user_input="usage check",
        model="test-model",
        tool_calls=[],
        input_tokens=100,
        output_tokens=25,
        total_tokens=125,
        prompt_cache_hit_tokens=10,
        prompt_cache_miss_tokens=90,
        reasoning_tokens=5,
        api_calls=1,
        estimated_cost=0.0125,
    )

    dashboard = get_dashboard_summary(store, logger)
    history = list_history_items(store)
    usage = logger.recent_usage_summary()

    assert store.get(pending_task.id).status == "pending"
    assert dashboard["counts"]["pending_tasks"] == 1
    assert dashboard["counts"]["completed_tasks"] == 1
    assert dashboard["usage_summary"]["run_count"] == 1
    assert history[0]["approval_id"] == completed_task.id
    assert usage.input_tokens == 100
    assert usage.output_tokens == 25
    assert usage.prompt_cache_hit_tokens == 10
    assert usage.prompt_cache_miss_tokens == 90
    assert usage.reasoning_tokens == 5
    assert usage.api_calls == 1
