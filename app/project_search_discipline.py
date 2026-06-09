from __future__ import annotations

import json
import re
from typing import Any


PROJECT_SEARCH_TOOL = "search_project_files"
MAX_PROJECT_SEARCHES_PER_TURN = 2
_CODE_PROPOSAL_PATTERNS = (
    re.compile(r"\bcode proposal\b"),
    re.compile(
        r"\b(?:add|build|change|create|fix|implement|modify|refactor|remove|"
        r"replace|update)\b.{0,120}\b(?:app|backend|behavior|bug|chat|code|"
        r"feature|frontend|lewis|ui)\b"
    ),
    re.compile(
        r"\b(?:make|create|plan)\b.{0,80}\bproposal\b.{0,120}"
        r"\b(?:file discovery|find relevant files|project search|relevant files)\b"
    ),
)


def project_search_query(arguments: object) -> str:
    parsed = arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return ""
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("query", "")).strip()


def should_suppress_project_search(
    query: str,
    tool_records: list[dict[str, Any]],
) -> str | None:
    searches = project_search_records(tool_records)
    if len(searches) >= MAX_PROJECT_SEARCHES_PER_TURN:
        return (
            f"Project search cap reached ({MAX_PROJECT_SEARCHES_PER_TURN}); "
            "answer from the evidence already gathered."
        )
    if any(
        near_duplicate_project_search(
            query,
            project_search_query(record.get("arguments")),
        )
        for record in searches
    ):
        return (
            "A repeated or near-duplicate project search was suppressed; "
            "answer from the evidence already gathered."
        )
    return None


def project_search_records(
    tool_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        record for record in tool_records if record.get("name") == PROJECT_SEARCH_TOOL
    ]


def has_useful_project_search_results(tool_records: list[dict[str, Any]]) -> bool:
    return any(
        isinstance(record.get("result"), dict)
        and record["result"].get("ok") is True
        and bool(record["result"].get("results"))
        for record in project_search_records(tool_records)
    )


def should_finish_project_search_turn(
    user_input: str,
    tool_records: list[dict[str, Any]],
) -> bool:
    del user_input
    searches = project_search_records(tool_records)
    if not searches:
        return False
    if has_useful_project_search_results(tool_records):
        return False
    return len(searches) >= MAX_PROJECT_SEARCHES_PER_TURN


def project_search_answer(tool_records: list[dict[str, Any]]) -> str:
    results: list[dict[str, Any]] = []
    suppressed = False
    for record in project_search_records(tool_records):
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        suppressed = suppressed or result.get("suppressed") is True
        for item in result.get("results", []):
            if isinstance(item, dict) and item not in results:
                results.append(item)

    if not results:
        suffix = (
            " A repeated or near-duplicate search was suppressed." if suppressed else ""
        )
        return (
            "Project search did not find strong matching evidence after the "
            f"bounded search pass.{suffix}"
        )

    lines = ["Project search found this relevant evidence:"]
    for item in results[:5]:
        path = str(item.get("path", "unknown path"))
        line_number = item.get("line_number")
        location = f"{path}:{line_number}" if line_number else path
        snippet = str(item.get("snippet", "")).strip()
        lines.append(f"- {location}: {snippet}" if snippet else f"- {location}")
    return "\n".join(lines)


def project_search_synthesized_answer(
    user_input: str,
    tool_records: list[dict[str, Any]],
) -> str:
    paths: list[str] = []
    snippets: list[str] = []
    for record in project_search_records(tool_records):
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        for item in result.get("results", []):
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            snippet = str(item.get("snippet", "")).strip()
            if path and path not in paths:
                paths.append(path)
            if snippet and snippet != "path match" and snippet not in snippets:
                snippets.append(snippet)

    if not paths:
        return project_search_answer(tool_records)

    subject = " ".join(user_input.strip().rstrip(".").split())
    path_summary = ", ".join(paths[:3])
    answer = (
        f"Based on the bounded read-only project search, {subject.lower()} "
        f"appears to be handled in {path_summary}."
    )
    if snippets:
        answer += " The matching code indicates: " + "; ".join(snippets[:3]) + "."
    return answer


def is_code_proposal_request(user_input: str) -> bool:
    normalized = " ".join(user_input.lower().split())
    return any(pattern.search(normalized) for pattern in _CODE_PROPOSAL_PATTERNS)


def near_duplicate_project_search(left: str, right: str) -> bool:
    left_terms = _query_terms(left)
    right_terms = _query_terms(right)
    if not left_terms or not right_terms:
        return False
    if left_terms == right_terms:
        return True
    overlap = len(left_terms & right_terms)
    union = len(left_terms | right_terms)
    return overlap / union >= 0.75


def _query_terms(query: str) -> frozenset[str]:
    return frozenset(re.findall(r"[a-z0-9_]+", query.lower()))
