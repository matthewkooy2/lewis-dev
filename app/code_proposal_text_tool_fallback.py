from __future__ import annotations

import re
from typing import Any

from app.project_search_discipline import near_duplicate_project_search
from app.search_first_code_proposal import (
    extract_requested_goal,
    grounded_proposed_approach,
)


MAX_TEXT_SEARCH_QUERIES = 2
SEARCH_PROJECT_FILES_BLOCK_PATTERN = re.compile(
    r"<search_project_files>\s*<query>(?P<query>.*?)</query>\s*</search_project_files>",
    re.IGNORECASE | re.DOTALL,
)


def extract_project_search_queries(text: str) -> list[str]:
    queries: list[str] = []
    for match in SEARCH_PROJECT_FILES_BLOCK_PATTERN.finditer(text):
        query = " ".join(match.group("query").split())
        if not query or any(
            near_duplicate_project_search(query, existing) for existing in queries
        ):
            continue
        queries.append(query)
        if len(queries) >= MAX_TEXT_SEARCH_QUERIES:
            break
    return queries


def code_proposal_args_from_search_results(
    user_input: str,
    search_records: list[dict[str, Any]],
) -> dict[str, Any]:
    relevant_files = _relevant_files(search_records)
    requested_goal = extract_requested_goal(user_input)
    evidence_note = (
        f"Project search found likely relevant files: {', '.join(relevant_files)}."
        if relevant_files
        else "Project search did not find a clear matching file, so keep the proposal review-only and verify the target surface during review."
    )
    return {
        "requested_goal": requested_goal,
        "target_surface": "command center web UI",
        "relevant_files": relevant_files,
        "proposed_approach": (
            f"{grounded_proposed_approach(requested_goal, relevant_files)} "
            f"{evidence_note} No files are edited, patched, or executed by Lewis."
        ),
        "safety_notes": [
            "Pending code proposal only.",
            "Project search is read-only.",
            "No files are modified and no commands are run.",
        ],
        "suggested_test_commands": [
            ".\\.venv\\Scripts\\python.exe -m ruff check .",
            ".\\.venv\\Scripts\\python.exe -m pytest tests/test_web.py",
        ],
    }


def _relevant_files(search_records: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    seen: set[str] = set()
    for record in search_records:
        result = record.get("result")
        if not isinstance(result, dict):
            continue
        results = result.get("results")
        if not isinstance(results, list):
            continue
        for item in results:
            if not isinstance(item, dict):
                continue
            path = str(item.get("path", "")).strip()
            if not path or path in seen:
                continue
            files.append(path)
            seen.add(path)
            if len(files) >= 5:
                return files
    return files
