from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import DB_PATH
from app.db import connect_database, initialize_database


@dataclass(frozen=True)
class UsageSummary:
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost: float = 0.0
    api_calls: int = 0
    run_count: int = 0


@dataclass(frozen=True)
class UsageRun:
    id: int
    timestamp: str
    user_input: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_cost: float = 0.0
    api_calls: int = 0


class RunLogger:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        initialize_database(self.db_path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = connect_database(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_run(
        self,
        user_input: str,
        model: str,
        tool_calls: list[dict[str, Any]],
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        prompt_cache_hit_tokens: int = 0,
        prompt_cache_miss_tokens: int = 0,
        reasoning_tokens: int = 0,
        api_calls: int = 0,
        estimated_cost: float = 0.0,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    timestamp, user_input, model, tool_calls,
                    input_tokens, output_tokens, total_tokens,
                    prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                    reasoning_tokens, api_calls, estimated_cost
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    user_input,
                    model,
                    json.dumps(redact_tool_call_log(tool_calls), sort_keys=True),
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens,
                    reasoning_tokens,
                    api_calls,
                    estimated_cost,
                ),
            )

    def daily_estimated_cost(self) -> float:
        today = datetime.now(timezone.utc).date().isoformat()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(estimated_cost), 0) AS total FROM runs WHERE timestamp LIKE ?",
                (f"{today}%",),
            ).fetchone()
        return float(row["total"]) if row else 0.0

    def recent_usage_summary(self, limit: int = 50) -> UsageSummary:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT input_tokens, output_tokens, total_tokens,
                       prompt_cache_hit_tokens, prompt_cache_miss_tokens,
                       reasoning_tokens, estimated_cost, api_calls
                FROM runs
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        input_tokens = output_tokens = cache_hit = cache_miss = reasoning = 0
        api_calls = 0
        estimated_cost = 0.0
        for row in rows:
            input_tokens += int(row["input_tokens"] or 0)
            output_tokens += int(row["output_tokens"] or 0)
            cache_hit += int(row["prompt_cache_hit_tokens"] or 0)
            cache_miss += int(row["prompt_cache_miss_tokens"] or 0)
            reasoning += int(row["reasoning_tokens"] or 0)
            estimated_cost += float(row["estimated_cost"] or 0.0)
            row_api_calls = int(row["api_calls"] or 0)
            if row_api_calls == 0 and int(row["total_tokens"] or 0) > 0:
                row_api_calls = 1
            api_calls += row_api_calls

        return UsageSummary(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            prompt_cache_hit_tokens=cache_hit,
            prompt_cache_miss_tokens=cache_miss,
            reasoning_tokens=reasoning,
            estimated_cost=round(estimated_cost, 8),
            api_calls=api_calls,
            run_count=len(rows),
        )

    def recent_runs(self, limit: int = 50) -> list[UsageRun]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, timestamp, user_input, model, input_tokens,
                       output_tokens, total_tokens, prompt_cache_hit_tokens,
                       prompt_cache_miss_tokens, reasoning_tokens,
                       estimated_cost, api_calls
                FROM runs
                ORDER BY timestamp DESC, id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        runs: list[UsageRun] = []
        for row in rows:
            api_calls = int(row["api_calls"] or 0)
            if api_calls == 0 and int(row["total_tokens"] or 0) > 0:
                api_calls = 1
            runs.append(
                UsageRun(
                    id=int(row["id"]),
                    timestamp=str(row["timestamp"]),
                    user_input=str(row["user_input"]),
                    model=str(row["model"]),
                    input_tokens=int(row["input_tokens"] or 0),
                    output_tokens=int(row["output_tokens"] or 0),
                    prompt_cache_hit_tokens=int(row["prompt_cache_hit_tokens"] or 0),
                    prompt_cache_miss_tokens=int(row["prompt_cache_miss_tokens"] or 0),
                    reasoning_tokens=int(row["reasoning_tokens"] or 0),
                    estimated_cost=float(row["estimated_cost"] or 0.0),
                    api_calls=api_calls,
                )
            )
        return runs


def redact_tool_call_log(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    redacted: list[dict[str, Any]] = []
    for record in tool_calls:
        if record.get("name") != "propose_memory":
            redacted.append(record)
            continue
        clean = dict(record)
        clean["arguments"] = _redact_memory_arguments(clean.get("arguments"))
        redacted.append(clean)
    return redacted


def _redact_memory_arguments(arguments: Any) -> Any:
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return "[redacted memory proposal arguments]"
        return json.dumps(_redact_memory_argument_object(parsed), sort_keys=True)
    if isinstance(arguments, dict):
        return _redact_memory_argument_object(arguments)
    return arguments


def _redact_memory_argument_object(arguments: Any) -> Any:
    if not isinstance(arguments, dict):
        return arguments
    redacted = dict(arguments)
    if "content" in redacted:
        redacted["content"] = "[redacted]"
    return redacted
