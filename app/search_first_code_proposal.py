from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.approvals import ApprovalItem, ApprovalStore
from app.code_proposal_patch_preview import (
    PATCH_PREVIEW_REVIEW_ONLY_NOTE,
    PATCH_PREVIEW_UNAVAILABLE_NOTE,
    generate_grounded_patch_preview,
)
from app.code_proposals import propose_code_change
from app.project_search import ProjectFileSearch, ProjectSearchResult


MAX_PROJECT_SEARCHES = 2
MAX_RELEVANT_FILES = 5
MORE_GROUNDING_MESSAGE = (
    "Project search did not find enough relevant existing files to create a grounded "
    "code proposal. More grounding is needed before a proposal can be saved."
)
_INTENT_PATTERN = re.compile(r"\b(?:make|create)\b.{0,100}\bcode proposal\b")
_SEARCH_FIRST_PATTERN = re.compile(
    r"\b(?:use|using)\b.{0,50}\bproject search\b.{0,30}\bfirst\b"
)
_PROPOSAL_PREFIX_PATTERN = re.compile(
    r"^(?:make|create)\s+(?:a\s+)?code proposal(?:\s+(?:to|that))?\s+",
    re.IGNORECASE,
)
_META_COMMAND_PREFIX_PATTERN = re.compile(
    r"^(?:this time\s+)?i(?:\s*want\s*you\s*to)?\s*(?:create|make)\s+"
    r"(?:a\s+)?(?:code\s+)?proposal\s+(?:involving|to|that)(?:\s+|$)",
    re.IGNORECASE,
)
_SEARCH_FIRST_SUFFIX_PATTERN = re.compile(
    r",?\s+(?:use|using)\s+project search first\.?$",
    re.IGNORECASE,
)

_VERB_NORMALIZATIONS = {
    "add": "Add",
    "adding": "Add",
    "adds": "Add",
    "adjust": "Adjust",
    "adjusting": "Adjust",
    "adjusts": "Adjust",
    "animate": "Animate",
    "animating": "Animate",
    "animates": "Animate",
    "build": "Build",
    "building": "Build",
    "builds": "Build",
    "change": "Change",
    "changing": "Change",
    "changes": "Change",
    "convert": "Convert",
    "converting": "Convert",
    "converts": "Convert",
    "create": "Create",
    "creating": "Create",
    "creates": "Create",
    "fix": "Fix",
    "fixing": "Fix",
    "fixes": "Fix",
    "implement": "Implement",
    "implementing": "Implement",
    "implements": "Implement",
    "make": "Make",
    "making": "Make",
    "makes": "Make",
    "modify": "Modify",
    "modifying": "Modify",
    "modifies": "Modify",
    "refactor": "Refactor",
    "refactoring": "Refactor",
    "refactors": "Refactor",
    "remove": "Remove",
    "removing": "Remove",
    "removes": "Remove",
    "replace": "Replace",
    "replacing": "Replace",
    "replaces": "Replace",
    "transform": "Transform",
    "transforming": "Transform",
    "transforms": "Transform",
    "update": "Update",
    "updating": "Update",
    "updates": "Update",
}
_COMMAND_CENTER_UI_PATH_HINTS = (
    "chat.js",
    "lewis.css",
    "web_components.py",
    "web_pages.py",
    "web_routes.py",
    "web.py",
)
_COMMAND_CENTER_UI_TERMS = ("chat", "command center", "command_center", "message")
_CHAT_UI_REQUEST_TERMS = (
    "animation",
    "animated",
    "character",
    "chat",
    "color",
    "css",
    "style",
    "text",
    "thinking",
    "ui",
)
_IRRELEVANT_PATH_PARTS = ("readme", "code_proposal")
_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "code",
        "create",
        "first",
        "make",
        "proposal",
        "project",
        "search",
        "the",
        "to",
        "use",
        "using",
    }
)


@dataclass(frozen=True)
class SearchFirstCodeProposalResult:
    approval: ApprovalItem | None
    tool_records: list[dict[str, Any]]
    message: str


def is_search_first_code_proposal_request(user_input: str) -> bool:
    normalized = " ".join(user_input.lower().split())
    return bool(
        _INTENT_PATTERN.search(normalized) and _SEARCH_FIRST_PATTERN.search(normalized)
    )


def create_search_first_code_proposal(
    *,
    user_input: str,
    search: ProjectFileSearch,
    store: ApprovalStore,
    max_searches: int = MAX_PROJECT_SEARCHES,
) -> SearchFirstCodeProposalResult:
    search_records: list[dict[str, Any]] = []
    search_results: list[ProjectSearchResult] = []
    for query in project_search_queries(user_input)[: max(1, min(max_searches, 2))]:
        results = search.search(query)
        search_results.extend(results)
        search_records.append(
            {
                "name": "search_project_files",
                "arguments": {"query": query},
                "result": {
                    "ok": True,
                    "results": [_serialize_search_result(item) for item in results],
                },
            }
        )
    relevant_files = _relevant_files(search_results)
    if not relevant_files:
        return SearchFirstCodeProposalResult(
            approval=None,
            tool_records=search_records,
            message=MORE_GROUNDING_MESSAGE,
        )

    requested_goal = extract_requested_goal(user_input)
    evidence = f"Project search found these relevant existing files: {', '.join(relevant_files)}."
    proposed_approach = grounded_proposed_approach(requested_goal, relevant_files)
    grounded_preview = generate_grounded_patch_preview(
        requested_goal=requested_goal,
        relevant_files=relevant_files,
        search=search,
    )
    preview_note = (
        PATCH_PREVIEW_REVIEW_ONLY_NOTE
        if grounded_preview is not None
        else PATCH_PREVIEW_UNAVAILABLE_NOTE
    )
    approval = propose_code_change(
        store=store,
        requested_goal=requested_goal,
        target_surface="command center web UI",
        relevant_files=relevant_files,
        proposed_approach=(f"{proposed_approach} {evidence} {preview_note}"),
        patch_preview=(
            grounded_preview.patch_preview if grounded_preview is not None else None
        ),
        safety_notes=[
            "Pending code proposal only.",
            "Project search is read-only.",
            "Use only verified project search results during review.",
            preview_note,
        ],
        suggested_test_commands=[],
        project_root=search.project_root,
        align_relevant_files=False,
    )
    persisted = store.get(approval.id)
    if persisted is None or persisted.status != "pending":
        raise RuntimeError("The code proposal approval was not persisted as pending.")

    proposal_result = {
        "ok": True,
        "approval_id": persisted.id,
        "status": persisted.status,
        "action_type": persisted.action_type,
        "message": (
            f"Saved pending code proposal #{persisted.id}: "
            f"{persisted.title.rstrip(' .')}. "
            "Review it at /approvals. This is planning only; no project files "
            "were modified and no commands were run."
        ),
    }
    message = str(proposal_result["message"])
    return SearchFirstCodeProposalResult(
        approval=persisted,
        tool_records=[
            *search_records,
            {
                "name": "propose_code_change",
                "arguments": {
                    "requested_goal": requested_goal,
                    "target_surface": "command center web UI",
                    "relevant_files": relevant_files,
                    "proposed_approach": proposed_approach,
                },
                "result": proposal_result,
            },
        ],
        message=message,
    )


def project_search_queries(user_input: str) -> list[str]:
    normalized = user_input.lower()
    if (
        "chat" in normalized
        and any(term in normalized for term in _CHAT_UI_REQUEST_TERMS)
    ) or any(
        term in normalized for term in ("thinking", "animation", "animated character")
    ):
        return [
            "chat.js lewis.css thinking chat bubble animation color",
            "web_pages web_components chat_bubble chat-log chat-shell",
        ]
    if "command center" in normalized and any(
        term in normalized for term in ("chat", "web ui", "ui")
    ):
        return [
            "chat command_center command center",
            "web_components lewis.css chat.js chat input message rendering",
        ]

    words = re.findall(r"[a-z0-9_./-]+", user_input.lower())
    meaningful = [word for word in words if word not in _QUERY_STOP_WORDS]
    primary = " ".join(meaningful[:6]).strip()
    queries = [primary] if primary else ["app"]
    if "command center" in user_input.lower() and "chat" in words:
        queries.append("command center chat")
    elif len(meaningful) > 3:
        queries.append(" ".join(meaningful[-3:]))
    return list(dict.fromkeys(queries))[:MAX_PROJECT_SEARCHES]


def extract_requested_goal(user_input: str) -> str:
    goal = " ".join(user_input.split()).strip()
    goal = _PROPOSAL_PREFIX_PATTERN.sub("", goal, count=1)
    goal = _META_COMMAND_PREFIX_PATTERN.sub("", goal, count=1)
    goal = _SEARCH_FIRST_SUFFIX_PATTERN.sub("", goal, count=1).strip(" ,.")
    goal = _normalize_goal_wording(goal)
    if not goal:
        return "Create the requested code change."
    words = goal.split()
    if words and words[0].lower() in _VERB_NORMALIZATIONS:
        words[0] = _VERB_NORMALIZATIONS[words[0].lower()]
        goal = " ".join(words)
    goal = f"{goal[0].upper()}{goal[1:]}"
    return f"{goal.rstrip(' .!?')}."


def grounded_proposed_approach(
    requested_goal: str,
    relevant_files: list[str],
) -> str:
    files = set(relevant_files)
    normalized_goal = requested_goal.lower()
    clauses: list[str] = []

    if any(term in normalized_goal for term in ("thinking", "animation", "animated")):
        if "app/static/chat.js" in files:
            clauses.append(
                "Use `app/static/chat.js` and its existing thinking-bubble lifecycle "
                "to show and remove the requested Lewis thinking state while a chat "
                "request is pending."
            )
        if "app/static/lewis.css" in files:
            clauses.append(
                "Extend `app/static/lewis.css` around the existing `.bubble.thinking` "
                "and `thinking-dot` styles to define the visual animation."
            )
        markup_files = _present_files(
            relevant_files,
            ("app/web_pages.py", "app/web_components.py"),
        )
        if markup_files:
            clauses.append(
                f"Keep any required thinking-state markup scoped to {_file_list(markup_files)}."
            )
    elif any(term in normalized_goal for term in ("color", "blue", "black", "text")):
        if "app/static/lewis.css" in files:
            clauses.append(
                "Update `app/static/lewis.css` to scope the requested blue text color "
                "to the Command Center chat instead of changing the global ink color."
            )
        markup_files = _present_files(
            relevant_files,
            ("app/web_pages.py", "app/web_components.py"),
        )
        if markup_files:
            clauses.append(
                f"Use the existing chat markup in {_file_list(markup_files)} to keep "
                "the selector limited to chat content."
            )

    if not clauses:
        clauses.append(
            f"Review {_file_list(relevant_files)} and plan the implementation of "
            f"{requested_goal.rstrip('.').lower()} against the existing code."
        )

    clauses.append(
        "Keep the implementation scoped to these project-search-backed areas and "
        "verify the behavior through the existing review flow."
    )
    return " ".join(clauses)


def _normalize_goal_wording(goal: str) -> str:
    normalized = re.sub(r"\beverytime\b", "every time", goal, flags=re.IGNORECASE)
    normalized = re.sub(
        r"\bevery time\s+that\s+you(?:\s+are|\s*['’]?re)\s+[\"']?thinking[\"']?",
        "every time Lewis is thinking",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\byou(?:\s+are|\s*['’]?re)\s+[\"']?thinking[\"']?",
        "Lewis is thinking",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r",\s*(?=every time\b)",
        " ",
        normalized,
        flags=re.IGNORECASE,
    )
    return " ".join(normalized.split())


def _present_files(relevant_files: list[str], candidates: tuple[str, ...]) -> list[str]:
    return [path for path in candidates if path in relevant_files]


def _file_list(files: list[str]) -> str:
    formatted = [f"`{path}`" for path in files]
    if not formatted:
        return "the grounded project files"
    if len(formatted) == 1:
        return formatted[0]
    return f"{', '.join(formatted[:-1])} and {formatted[-1]}"


def _relevant_files(results: list[ProjectSearchResult]) -> list[str]:
    ranked: dict[str, int] = {}
    for result in results:
        path = result.path.lower()
        if any(part in path for part in _IRRELEVANT_PATH_PARTS):
            continue
        score = 0
        if any(path.endswith(hint) for hint in _COMMAND_CENTER_UI_PATH_HINTS):
            score += 3
        if "/static/" in path:
            score += 2
        evidence = f"{path} {result.snippet.lower()}"
        score += sum(term in evidence for term in _COMMAND_CENTER_UI_TERMS)
        if score >= 3:
            ranked[result.path] = max(score, ranked.get(result.path, 0))
    return [
        path
        for path, _score in sorted(
            ranked.items(),
            key=lambda item: (-item[1], item[0]),
        )[:MAX_RELEVANT_FILES]
    ]


def _serialize_search_result(result: ProjectSearchResult) -> dict[str, Any]:
    return {
        "path": result.path,
        "line_number": result.line_number,
        "snippet": result.snippet,
    }
