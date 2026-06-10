from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any

from app.approvals import ApprovalItem, ApprovalStore
from app.code_proposal_previews import (
    extract_patch_preview_paths,
    validate_patch_preview,
)
from app.code_proposal_patch_preview import generate_grounded_patch_preview
from app.code_proposals import propose_code_change
from app.project_search import ProjectFileSearch, ProjectSearchResult
from app.project_search_discipline import is_code_proposal_request


MAX_PROJECT_SEARCHES = 2
_NO_FILES_LEAD_IN = (
    "Project search did not find enough relevant existing file content to create "
    "a grounded code proposal."
)
_NO_STRATEGY_LEAD_IN = (
    "Lewis found candidate project files, but no conservative proposal strategy "
    "matched the full requested behavior."
)
_NO_GROUNDED_DIFF_LEAD_IN = (
    "Lewis found candidate project files and a compatible strategy, but no "
    "grounded patch could be safely generated."
)
_DESTRUCTIVE_LEAD_IN = (
    "Lewis found candidate project files, but the requested behavior appears "
    "destructive or storage-sensitive and needs explicit confirmation plus "
    "verified storage semantics before a safe proposal strategy can create a patch."
)
_DESTRUCTIVE_NO_FILES_LEAD_IN = (
    "Project search did not find matching candidate project files, and the "
    "requested behavior appears destructive or storage-sensitive. Explicit "
    "confirmation plus verified storage semantics would be required before "
    "a safe proposal strategy can be attempted."
)
_UNSUPPORTED_THINKING_CONTENT_LEAD_IN = (
    "Lewis found candidate project files, but the request asks for ASCII art, "
    "a person/avatar/character figure, replacement of the current thinking "
    "animation, or a thinking-bubble content/body change. No conservative "
    "strategy exists for ASCII/person/replacement thinking-bubble content "
    "behavior. The current thinking-animation strategy only safely adds a "
    "bounded CSS waiting animation to the existing thinking bubble; it cannot "
    "draw figures, replace the existing animation, or change bubble content."
)
PATCH_PREVIEW_REVIEW_ONLY_NOTE = (
    "The grounded patch preview is inert review-only text; Lewis does not apply it."
)
_COLOR_KIND = "chat_text_color"
_THINKING_ANIMATION_KIND = "thinking_animation"
_SETTING_LIMIT_KIND = "setting_limit"
_GENERIC_FEATURE_KIND = "generic_feature"
_COLOR_QUERY = "chat text color lewis.css bubble"
_THINKING_QUERY = "thinking animation chat.js lewis.css bubble"
_SECONDARY_QUERY = "web_pages web_components chat-log bubble thinking"
_MAX_RELEVANT_FILES = 5
_SETTING_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "code",
        "create",
        "make",
        "proposal",
        "the",
        "to",
    }
)
_SETTING_LIMIT_EXTRA_TERMS = ("max", "cap", "configuration", "settings")
_TOOL_LIMIT_EXTRA_TERMS = (
    "call",
    "calls",
    "tool_call",
    "tool_calls",
    "tool_call_limit",
    "effective_tool_call_limit",
    "max_tool_calls_per_request",
    "LEWIS_MAX_TOOL_CALLS",
)
_NON_EDITABLE_EVIDENCE_PARTS = (
    "code_proposal",
    "docs/",
    "readme",
    "tests/",
)
_GENERIC_EXCLUDED_PATH_PARTS = ("docs/", "readme", "tests/")
_GENERIC_SOURCE_SUFFIXES = (".css", ".html", ".js", ".py")
_GENERIC_QUERY_STOP_WORDS = frozenset(
    {
        "a",
        "after",
        "all",
        "an",
        "and",
        "are",
        "at",
        "be",
        "been",
        "by",
        "can",
        "code",
        "do",
        "does",
        "for",
        "from",
        "happening",
        "has",
        "have",
        "help",
        "i",
        "in",
        "is",
        "it",
        "just",
        "like",
        "make",
        "new",
        "not",
        "now",
        "of",
        "on",
        "or",
        "proposal",
        "so",
        "stops",
        "that",
        "the",
        "then",
        "this",
        "to",
        "when",
        "will",
        "with",
        "would",
        "you",
        "your",
    }
)
_GENERIC_UI_SURFACE_TERMS = frozenset(
    {"button", "history", "page", "screen", "tab", "ui"}
)
_GENERIC_UI_PATHS = frozenset(
    {
        "app/web_components.py",
        "app/web_pages.py",
        "app/web_routes.py",
    }
)
_DESTRUCTIVE_GOAL_TERMS = ("clear", "delete", "erase", "purge", "remove", "reset")
_ANIMATION_MODIFIER_TERMS = frozenset(
    {
        "bouncier",
        "bouncy",
        "calmer",
        "calm",
        "faster",
        "fast",
        "gentler",
        "gentle",
        "less distracting",
        "less intense",
        "less noticeable",
        "more intense",
        "more noticeable",
        "slower",
        "slow",
        "softer",
        "soft",
        "smoother",
        "smooth",
        "subtler",
        "subtle",
    }
)
_ASCII_ART_PATTERN = re.compile(r"\bascii\b")
_CHARACTER_FIGURE_PATTERN = re.compile(
    r"\b(?:person|persons|people|avatar|avatars|character|characters"
    r"|figure|figures|mascot|mascots)\b"
)
_ANIMATION_REPLACEMENT_TRIGGER_TERMS = ("replace", "replacing", "swap", "instead of")
_ANIMATION_REPLACEMENT_TARGET_TERMS = ("animation", "animated", "animate", "thinking")
_BUBBLE_CONTENT_CHANGE_TERMS = ("content", "body", "inside")
_SETTING_LIMIT_ENV_INT_PATTERN = re.compile(
    r"(?P<prefix>_env_int\(\s*(?P<quote>['\"])(?P<env_name>[A-Z0-9_]+)"
    r"(?P=quote)\s*,\s*)(?P<value>\d+)(?P<suffix>\s*\))"
)
_CONVERSATIONAL_LEADIN_PATTERN = re.compile(
    r"^(?:and\s+then|then|also|next(?:ly)?|separately|this\s+time)[,\s]+",
    re.IGNORECASE,
)
_REQUEST_FRAMING_PATTERN = re.compile(
    r"^(?:i\s+(?:would\s+like\s+(?:for\s+)?|want\s+)you\s+to|can\s+you|could\s+you|would\s+you|please)[,\s]+",
    re.IGNORECASE,
)
_PROPOSAL_FRAMING_PATTERN = re.compile(
    r"^(?:make|create)\s+(?:a\s+)?(?:separate\s+|another\s+)?(?:code\s+)?proposal"
    r"(?:\s+(?:to|that|involving))?\s*",
    re.IGNORECASE,
)
_SEARCH_SUFFIX_PATTERN = re.compile(
    r",?\s+(?:use|using)\s+project search first\.?$",
    re.IGNORECASE,
)
_CHAT_COLOR_BLOCK_PATTERN = re.compile(
    r"(?P<selector>[^{}]*\.chat-log\s+\.bubble[^{}]*)"
    r"\{(?P<body>[^{}]*)\}",
    re.IGNORECASE,
)
_THINKING_BLOCK_PATTERN = re.compile(
    r"(?P<selector>[^{}]*\.bubble\.thinking[^{}]*)"
    r"\{(?P<body>[^{}]*)\}",
    re.IGNORECASE,
)
_COLOR_TEXT_CONCEPT = "color/text styling"
_ANIMATION_CONCEPT = "animation"
_WHILE_THINKING_CONCEPT = "while-thinking/waiting state"
_COMPLETION_CONCEPT = "after-response/completion-ready state"
_STATUS_INDICATOR_CONCEPT = "status indicator"
_LIMIT_INCREASE_CONCEPT = "limit increase"
_STATIC_LABEL_CONCEPT = "static label/caption"
_REMOVAL_POLARITY_CONCEPT = "removal/clear/undo/reset polarity"
_ANIMATION_MODIFIER_CONCEPT = "animation modifier/qualitative tuning"
_ASCII_ART_CONCEPT = "ASCII art content"
_CHARACTER_FIGURE_CONCEPT = "person/avatar/character figure"
_ANIMATION_REPLACEMENT_CONCEPT = "replacement of the current animation"
_BUBBLE_CONTENT_CONCEPT = "thinking-bubble content/body change"
_UNSUPPORTED_THINKING_CONTENT_CONCEPTS = (
    _ASCII_ART_CONCEPT,
    _CHARACTER_FIGURE_CONCEPT,
    _ANIMATION_REPLACEMENT_CONCEPT,
    _BUBBLE_CONTENT_CONCEPT,
)
_CHAT_CONTENT_SURFACE = "chat content/message/bubble"
_DASHBOARD_NAV_SURFACE = "dashboard/sidebar/navigation link or icon"
_STATUS_INDICATOR_SURFACE = "status indicator"
_CONFIGURATION_SURFACE = "configuration/backend"
_GENERAL_UI_SURFACE = "general UI"


def _strip_optional(text: str, pattern: re.Pattern[str]) -> str:
    return pattern.sub("", text, count=1)


@dataclass(frozen=True)
class CodeProposalEngineResult:
    approval: ApprovalItem | None
    tool_records: list[dict[str, Any]]
    message: str


@dataclass(frozen=True)
class _GroundedChange:
    path: str
    original_text: str
    updated_text: str | None
    patch_preview: str


@dataclass(frozen=True)
class _GroundingStatus:
    evidence: dict[str, str]
    candidate_paths: list[str]
    change: _GroundedChange | None
    reason: str


@dataclass(frozen=True)
class _GoalProfile:
    concepts: frozenset[str]
    surfaces: frozenset[str]


@dataclass(frozen=True)
class _StrategySpec:
    kind: str
    name: str
    required_concepts: frozenset[str]
    supported_concepts: frozenset[str]
    supported_surfaces: frozenset[str]
    disqualifying_concepts: frozenset[str] = frozenset()
    disqualifying_surfaces: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _StrategySelection:
    kind: str
    approval_compatible: bool
    reason: str


_STRATEGY_SPECS = (
    _StrategySpec(
        kind=_SETTING_LIMIT_KIND,
        name="backend setting-limit strategy",
        required_concepts=frozenset({_LIMIT_INCREASE_CONCEPT}),
        supported_concepts=frozenset({_LIMIT_INCREASE_CONCEPT}),
        supported_surfaces=frozenset({_CONFIGURATION_SURFACE}),
        disqualifying_surfaces=frozenset(
            {_GENERAL_UI_SURFACE, _DASHBOARD_NAV_SURFACE, _STATUS_INDICATOR_SURFACE}
        ),
    ),
    _StrategySpec(
        kind=_THINKING_ANIMATION_KIND,
        name="chat thinking-animation strategy",
        required_concepts=frozenset({_ANIMATION_CONCEPT, _WHILE_THINKING_CONCEPT}),
        supported_concepts=frozenset({_ANIMATION_CONCEPT, _WHILE_THINKING_CONCEPT}),
        supported_surfaces=frozenset({_CHAT_CONTENT_SURFACE, _GENERAL_UI_SURFACE}),
        disqualifying_concepts=frozenset(
            {
                _COMPLETION_CONCEPT,
                _STATUS_INDICATOR_CONCEPT,
                _REMOVAL_POLARITY_CONCEPT,
                _ANIMATION_MODIFIER_CONCEPT,
                *_UNSUPPORTED_THINKING_CONTENT_CONCEPTS,
            }
        ),
        disqualifying_surfaces=frozenset(
            {_DASHBOARD_NAV_SURFACE, _STATUS_INDICATOR_SURFACE}
        ),
    ),
    _StrategySpec(
        kind=_COLOR_KIND,
        name="static chat-text-color strategy",
        required_concepts=frozenset({_COLOR_TEXT_CONCEPT}),
        supported_concepts=frozenset({_COLOR_TEXT_CONCEPT}),
        supported_surfaces=frozenset({_CHAT_CONTENT_SURFACE, _GENERAL_UI_SURFACE}),
        disqualifying_concepts=frozenset(
            {
                _WHILE_THINKING_CONCEPT,
                _COMPLETION_CONCEPT,
                _STATUS_INDICATOR_CONCEPT,
                _REMOVAL_POLARITY_CONCEPT,
            }
        ),
        disqualifying_surfaces=frozenset(
            {_DASHBOARD_NAV_SURFACE, _STATUS_INDICATOR_SURFACE}
        ),
    ),
    _StrategySpec(
        kind=_GENERIC_FEATURE_KIND,
        name="generic static-label strategy",
        required_concepts=frozenset({_STATIC_LABEL_CONCEPT}),
        supported_concepts=frozenset({_STATIC_LABEL_CONCEPT}),
        supported_surfaces=frozenset({_CHAT_CONTENT_SURFACE, _GENERAL_UI_SURFACE}),
        disqualifying_concepts=frozenset(
            {
                _ANIMATION_CONCEPT,
                _WHILE_THINKING_CONCEPT,
                _COMPLETION_CONCEPT,
                _STATUS_INDICATOR_CONCEPT,
                _LIMIT_INCREASE_CONCEPT,
            }
        ),
        disqualifying_surfaces=frozenset({_STATUS_INDICATOR_SURFACE}),
    ),
)


def create_code_proposal_if_supported(
    *,
    user_input: str,
    search: ProjectFileSearch,
    store: ApprovalStore,
    max_tool_calls: int,
) -> CodeProposalEngineResult | None:
    """Create one grounded pending proposal for supported replacement-engine intents."""
    requested_goal = clean_requested_goal(user_input)
    selection = _select_strategy(user_input, requested_goal)
    if selection is None:
        return None
    if max_tool_calls < 2:
        return _tool_limit_result(max_tool_calls, [])

    proposal_kind = selection.kind
    search_records: list[dict[str, Any]] = []
    search_results: list[ProjectSearchResult] = []
    evidence: dict[str, str] = {}
    candidate_paths: list[str] = []
    grounded_change: _GroundedChange | None = None
    failure_reason = "no project searches were available for this proposal."
    queries = _queries_for_kind(proposal_kind, requested_goal=requested_goal)

    for query in queries[: min(MAX_PROJECT_SEARCHES, max_tool_calls)]:
        results = search.search(query)
        search_results.extend(results)
        search_records.append(_search_record(query, results))
        status = _grounded_status(
            proposal_kind,
            requested_goal=requested_goal,
            results=search_results,
            search=search,
            compatibility_reason=(
                "" if selection.approval_compatible else selection.reason
            ),
        )
        evidence = status.evidence
        candidate_paths = status.candidate_paths
        grounded_change = status.change
        failure_reason = status.reason
        if grounded_change is not None:
            break

    if grounded_change is None:
        return CodeProposalEngineResult(
            approval=None,
            tool_records=search_records,
            message=_grounding_failure_message(
                requested_goal=requested_goal,
                search_records=search_records,
                candidate_paths=candidate_paths,
                reason=failure_reason,
            ),
        )
    if len(search_records) >= max_tool_calls:
        return _tool_limit_result(max_tool_calls, search_records)

    relevant_files = list(evidence)
    patch_preview = _validated_grounded_patch(
        grounded_change,
        relevant_files=relevant_files,
        project_root=search.project_root,
    )
    proposed_approach = _proposed_approach(
        proposal_kind,
        requested_goal=requested_goal,
        relevant_files=relevant_files,
        touched_path=grounded_change.path,
    )
    approval = propose_code_change(
        store=store,
        requested_goal=requested_goal,
        target_surface=_target_surface_for_kind(proposal_kind),
        relevant_files=relevant_files,
        proposed_approach=proposed_approach,
        patch_preview=patch_preview,
        safety_notes=[
            "Pending code proposal only.",
            "Project search and project file reads are safe and read-only.",
            PATCH_PREVIEW_REVIEW_ONLY_NOTE,
            "No project files are modified and no commands are run.",
        ],
        suggested_test_commands=[],
        extra_payload={
            "applicator_contract_version": 1,
            "selected_strategy": proposal_kind,
            "touched_paths": [grounded_change.path],
            "patch_preview_kind": "unified diff",
            "patch_is_inert": True,
            "commands_run": [record["name"] for record in search_records],
            "files_modified": [],
        },
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
            f"{persisted.title.rstrip(' .')}. Review it at /approvals. "
            "This is planning only; no project files were modified and no commands "
            "were run."
        ),
    }
    return CodeProposalEngineResult(
        approval=persisted,
        tool_records=[
            *search_records,
            {
                "name": "propose_code_change",
                "arguments": {
                    "requested_goal": requested_goal,
                    "target_surface": _target_surface_for_kind(proposal_kind),
                    "relevant_files": relevant_files,
                    "proposed_approach": proposed_approach,
                    "patch_preview": patch_preview,
                },
                "result": proposal_result,
            },
        ],
        message=str(proposal_result["message"]),
    )


def clean_requested_goal(user_input: str) -> str:
    goal = " ".join(user_input.split()).strip()
    goal = _strip_optional(goal, _CONVERSATIONAL_LEADIN_PATTERN)
    goal = _strip_optional(goal, _REQUEST_FRAMING_PATTERN)
    goal = _strip_optional(goal, _PROPOSAL_FRAMING_PATTERN)
    goal = _SEARCH_SUFFIX_PATTERN.sub("", goal, count=1).strip(" ,.")
    goal = re.sub(r"\beverytime\b", "every time", goal, flags=re.IGNORECASE)
    goal = re.sub(
        r"\bevery time\s+that\s+you(?:\s+are|\s*[''\u2019]?re)\s+[\x22\x27]?thinking[\x22\x27]?",
        "every time Lewis is thinking",
        goal,
        flags=re.IGNORECASE,
    )
    goal = re.sub(
        r"\byou(?:\s+are|\s*[''\u2019]?re)\s+[\x22\x27]?thinking[\x22\x27]?",
        "Lewis is thinking",
        goal,
        flags=re.IGNORECASE,
    )
    goal = re.sub(r",\s*(?=every time\b)", " ", goal, flags=re.IGNORECASE)
    goal = " ".join(goal.split())
    if not goal:
        return "Create the requested code change."

    verb, _, remainder = goal.partition(" ")
    normalized_verb = {
        "adding": "Add",
        "adds": "Add",
        "makes": "Make",
    }.get(verb.lower(), verb.capitalize())
    goal = f"{normalized_verb} {remainder}".strip()
    return f"{goal.rstrip(' .!?')}."


def _select_strategy(user_input: str, requested_goal: str) -> _StrategySelection | None:
    normalized = " ".join(user_input.lower().split())
    if "proposal" not in normalized:
        return None
    profile = _goal_profile(requested_goal)
    known_strategy_match = any(
        spec.kind != _GENERIC_FEATURE_KIND
        and spec.required_concepts <= profile.concepts
        for spec in _STRATEGY_SPECS
    )
    if not is_code_proposal_request(user_input) and not known_strategy_match:
        return None

    considered: list[str] = []
    for spec in _STRATEGY_SPECS:
        if not _strategy_is_candidate(spec, requested_goal, profile):
            continue
        incompatibility = _strategy_incompatibility(spec, profile)
        if spec.required_concepts <= profile.concepts and not incompatibility:
            return _StrategySelection(
                kind=spec.kind,
                approval_compatible=True,
                reason="",
            )
        considered.append(f"{spec.name}: {incompatibility}")

    reason = (
        "No conservative strategy matched the full cleaned goal's material concepts "
        "and target surface."
    )
    if considered:
        reason = f"{reason} Strategies considered: {'; '.join(considered)}."
    else:
        reason = (
            f"{reason} Strategies considered: none of the available approval "
            "strategies declared support for all material concepts."
        )
    return _StrategySelection(
        kind=_GENERIC_FEATURE_KIND,
        approval_compatible=not considered,
        reason=reason,
    )


def _goal_profile(requested_goal: str) -> _GoalProfile:
    normalized = requested_goal.lower()
    concepts: set[str] = set()
    surfaces: set[str] = set()

    if any(
        term in normalized
        for term in (
            "clear the",
            "remove the",
            "undo the",
            "unset the",
            "stop making",
            "revert the",
            "reset the",
        )
    ):
        concepts.add(_REMOVAL_POLARITY_CONCEPT)

    completion_state = any(
        phrase in normalized
        for phrase in (
            "after lewis is done thinking",
            "after lewis is done",
            "after response",
            "after the response",
            "done thinking",
            "finished thinking",
            "produced your response",
            "response is produced",
            "completion-ready",
            "completion ready",
        )
    )
    if completion_state:
        concepts.add(_COMPLETION_CONCEPT)
    if any(term in normalized for term in ("animation", "animated", "animate")):
        concepts.add(_ANIMATION_CONCEPT)
    if (
        any(term in normalized for term in ("thinking", "waiting"))
        and not completion_state
    ):
        concepts.add(_WHILE_THINKING_CONCEPT)
    if any(term in normalized for term in ("color", "colour", "blue", "black")) and any(
        term in normalized for term in ("text", "message", "caption")
    ):
        concepts.add(_COLOR_TEXT_CONCEPT)
    if any(
        phrase in normalized
        for phrase in ("status indicator", "status dot", "status badge", "indicator")
    ):
        concepts.add(_STATUS_INDICATOR_CONCEPT)
    if _is_setting_limit_goal(normalized):
        concepts.add(_LIMIT_INCREASE_CONCEPT)
    if any(term in normalized for term in ("label", "caption")):
        concepts.add(_STATIC_LABEL_CONCEPT)
    if "badge" in normalized and _STATUS_INDICATOR_CONCEPT not in concepts:
        concepts.add(_STATIC_LABEL_CONCEPT)
    if _ANIMATION_CONCEPT in concepts and any(
        term in normalized for term in _ANIMATION_MODIFIER_TERMS
    ):
        concepts.add(_ANIMATION_MODIFIER_CONCEPT)
    if _ASCII_ART_PATTERN.search(normalized):
        concepts.add(_ASCII_ART_CONCEPT)
    if _CHARACTER_FIGURE_PATTERN.search(normalized):
        concepts.add(_CHARACTER_FIGURE_CONCEPT)
    if any(
        term in normalized for term in _ANIMATION_REPLACEMENT_TRIGGER_TERMS
    ) and any(term in normalized for term in _ANIMATION_REPLACEMENT_TARGET_TERMS):
        concepts.add(_ANIMATION_REPLACEMENT_CONCEPT)
    if "bubble" in normalized and any(
        term in normalized for term in _BUBBLE_CONTENT_CHANGE_TERMS
    ):
        concepts.add(_BUBBLE_CONTENT_CONCEPT)

    dashboard_terms = ("dashboard", "sidebar", "navigation", "nav ", "left side")
    if any(term in normalized for term in dashboard_terms) or "chat icon" in normalized:
        surfaces.add(_DASHBOARD_NAV_SURFACE)
    if _STATUS_INDICATOR_CONCEPT in concepts:
        surfaces.add(_STATUS_INDICATOR_SURFACE)
    if (
        "chat" in normalized
        and any(term in normalized for term in ("text", "message", "bubble", "content"))
        and "chat icon" not in normalized
    ):
        surfaces.add(_CHAT_CONTENT_SURFACE)
    if _LIMIT_INCREASE_CONCEPT in concepts or any(
        term in normalized for term in ("configuration", "backend", "setting")
    ):
        surfaces.add(_CONFIGURATION_SURFACE)
    if any(
        term in normalized
        for term in ("ui", "interface", "dashboard", "sidebar", "navigation", "icon")
    ):
        surfaces.add(_GENERAL_UI_SURFACE)
    return _GoalProfile(concepts=frozenset(concepts), surfaces=frozenset(surfaces))


def _strategy_is_candidate(
    spec: _StrategySpec, requested_goal: str, profile: _GoalProfile
) -> bool:
    if spec.required_concepts <= profile.concepts:
        return True
    normalized = requested_goal.lower()
    if spec.kind == _THINKING_ANIMATION_KIND:
        return any(term in normalized for term in ("thinking", "waiting", "animation"))
    if spec.kind == _COLOR_KIND:
        return any(term in normalized for term in ("color", "colour", "blue", "black"))
    if spec.kind == _SETTING_LIMIT_KIND:
        return any(term in normalized for term in ("limit", "cap", "maximum"))
    return any(term in normalized for term in ("label", "badge", "caption"))


def _strategy_incompatibility(spec: _StrategySpec, profile: _GoalProfile) -> str:
    missing_concepts = spec.required_concepts - profile.concepts
    blocked_concepts = profile.concepts & spec.disqualifying_concepts
    blocked_surfaces = profile.surfaces & spec.disqualifying_surfaces
    unsupported_concepts = profile.concepts - spec.supported_concepts
    unsupported_surfaces = profile.surfaces - spec.supported_surfaces
    if not any(
        (blocked_concepts, blocked_surfaces, unsupported_concepts, unsupported_surfaces)
    ):
        return ""

    details: list[str] = []
    if missing_concepts:
        details.append(
            "requires material concepts " + ", ".join(sorted(missing_concepts))
        )
    if blocked_concepts or unsupported_concepts:
        details.append(
            "does not support material concepts "
            + ", ".join(sorted(blocked_concepts | unsupported_concepts))
        )
    if blocked_surfaces or unsupported_surfaces:
        details.append(
            "supports only target surfaces "
            + ", ".join(sorted(spec.supported_surfaces))
            + "; does not support target surfaces "
            + ", ".join(sorted(blocked_surfaces | unsupported_surfaces))
        )
    return " and ".join(details)


def _queries_for_kind(proposal_kind: str, *, requested_goal: str) -> tuple[str, ...]:
    if proposal_kind == _COLOR_KIND:
        return (_COLOR_QUERY, _SECONDARY_QUERY)
    if proposal_kind == _THINKING_ANIMATION_KIND:
        return (_THINKING_QUERY, _SECONDARY_QUERY)
    if proposal_kind == _SETTING_LIMIT_KIND:
        return tuple(_setting_limit_queries(requested_goal))
    return tuple(_generic_feature_queries(requested_goal))


def _generic_feature_queries(requested_goal: str) -> list[str]:
    normalized = requested_goal.lower()
    words = [
        word
        for word in re.findall(r"[a-z0-9_]+", normalized)
        if word not in _GENERIC_QUERY_STOP_WORDS
    ]
    goal_terms: list[str] = []
    for word in words:
        _append_unique_term(goal_terms, word)

    for left, right in zip(words, words[1:]):
        if left != right:
            _append_unique_term(goal_terms, f"{left}_{right}")

    surface_terms: list[str] = []
    if "history tab" in normalized:
        for term in (
            "history_tab",
            "clear_history",
            "history_page",
            "history_filters_html",
            "list_history_items",
            "run_history",
            "chat_history",
            "command_center_history",
        ):
            _append_unique_term(surface_terms, term)

    if set(words) & _GENERIC_UI_SURFACE_TERMS:
        for term in ("web_pages", "web_components", "web_routes", "button", "route"):
            _append_unique_term(surface_terms, term)

    if (
        "relevant files" in normalized
        or "file discovery" in normalized
        or "project search" in normalized
    ):
        for term in (
            "relevant_files",
            "file_discovery",
            "project_search",
            "search_project_files",
            "code_proposal_engine",
            "search_first_code_proposal",
        ):
            _append_unique_term(surface_terms, term)

    phrase_terms = [
        phrase
        for phrase in (
            "history tab",
            "command center",
            "relevant files",
            "project search",
        )
        if phrase in normalized
    ]
    primary = " ".join([*phrase_terms, *goal_terms[:8]]).strip()
    secondary = " ".join([*phrase_terms, *surface_terms[:12]]).strip()
    if not secondary:
        identifier_terms = [term for term in goal_terms if "_" in term]
        secondary = " ".join(identifier_terms[:8] or goal_terms[-6:]).strip()
    return list(dict.fromkeys(query for query in (primary, secondary) if query))


def _setting_limit_queries(requested_goal: str) -> list[str]:
    terms = _expanded_setting_terms(requested_goal)
    primary = " ".join(terms[:8])
    secondary_terms = [term for term in terms if "_" in term or term.isupper()]
    secondary = " ".join(secondary_terms[:8])
    return list(dict.fromkeys(query for query in (primary, secondary) if query))


def _expanded_setting_terms(requested_goal: str) -> list[str]:
    words = [
        word
        for word in re.findall(r"[A-Za-z0-9_]+", requested_goal)
        if word.lower() not in _SETTING_QUERY_STOP_WORDS
    ]
    terms: list[str] = []
    for word in words:
        _append_unique_term(terms, word.lower())
    if "limit" in terms:
        for term in _SETTING_LIMIT_EXTRA_TERMS:
            _append_unique_term(terms, term)
    if "tool" in terms and ("use" in terms or "call" in terms or "calls" in terms):
        for term in _TOOL_LIMIT_EXTRA_TERMS:
            _append_unique_term(terms, term)
    return terms or ["configuration", "settings"]


def _append_unique_term(terms: list[str], term: str) -> None:
    if term and term not in terms:
        terms.append(term)


def _search_record(query: str, results: list[ProjectSearchResult]) -> dict[str, Any]:
    return {
        "name": "search_project_files",
        "arguments": {"query": query},
        "result": {
            "ok": True,
            "results": [
                {
                    "path": item.path,
                    "line_number": item.line_number,
                    "snippet": item.snippet,
                }
                for item in results
            ],
        },
    }


def _evidence_files(
    proposal_kind: str,
    requested_goal: str,
    results: list[ProjectSearchResult],
    search: ProjectFileSearch,
) -> dict[str, str]:
    if proposal_kind == _SETTING_LIMIT_KIND:
        return _setting_limit_evidence(requested_goal, results, search)
    if proposal_kind == _GENERIC_FEATURE_KIND:
        return _generic_feature_evidence(requested_goal, results, search)

    result_paths = {result.path for result in results}
    preferred_paths = (
        ("app/static/lewis.css",)
        if proposal_kind == _COLOR_KIND
        else ("app/static/chat.js", "app/static/lewis.css")
    )
    evidence: dict[str, str] = {}
    for path in preferred_paths:
        if path not in result_paths:
            continue
        content = search.read_text(path)
        if content is None or not _content_supports_kind(proposal_kind, path, content):
            continue
        evidence[path] = content
    return evidence


def _generic_feature_evidence(
    requested_goal: str,
    results: list[ProjectSearchResult],
    search: ProjectFileSearch,
) -> dict[str, str]:
    terms = _generic_goal_terms(requested_goal)
    explicitly_targets_tests = any(
        term in requested_goal.lower()
        for term in ("test", "tests", "documentation", "docs")
    )
    targets_file_discovery = any(
        term in requested_goal.lower()
        for term in ("file discovery", "project search", "relevant files")
    )
    scores: dict[str, int] = {}
    contents: dict[str, str] = {}
    for result in results:
        path = result.path
        normalized_path = path.lower()
        if not normalized_path.endswith(_GENERIC_SOURCE_SUFFIXES):
            continue
        if not explicitly_targets_tests and any(
            part in normalized_path for part in _GENERIC_EXCLUDED_PATH_PARTS
        ):
            continue
        if not targets_file_discovery and "code_proposal" in normalized_path:
            continue
        content = search.read_text(path)
        if content is None:
            continue
        score = _generic_candidate_score(path, content, terms)
        if score <= 0:
            continue
        scores[path] = max(score, scores.get(path, 0))
        contents[path] = content

    ranked_paths = [
        path
        for path, _score in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:_MAX_RELEVANT_FILES]
    ]
    return {path: contents[path] for path in ranked_paths}


def _generic_goal_terms(requested_goal: str) -> list[str]:
    terms = [
        word
        for word in re.findall(r"[a-z0-9_]+", requested_goal.lower())
        if word not in _GENERIC_QUERY_STOP_WORDS
    ]
    if "history tab" in requested_goal.lower():
        terms.extend(
            [
                "history_page",
                "history_filters_html",
                "list_history_items",
            ]
        )
    if "relevant files" in requested_goal.lower():
        terms.extend(["relevant_files", "project_search", "search_project_files"])
    return list(dict.fromkeys(terms))


def _generic_candidate_score(path: str, content: str, terms: list[str]) -> int:
    normalized_path = path.lower()
    searchable = f"{normalized_path}\n{content.lower()}"
    searchable_tokens = set(re.findall(r"[a-z0-9_]+", searchable))
    matching_terms = {
        term
        for term in terms
        if term in searchable_tokens or ("_" in term and term in searchable)
    }
    if not matching_terms:
        return 0
    score = len(matching_terms) * 2
    if len(matching_terms) >= 2:
        score += len(matching_terms) * 2
    if normalized_path in _GENERIC_UI_PATHS and len(matching_terms) >= 2:
        score += 4
    if normalized_path.startswith("app/static/") and len(matching_terms) >= 2:
        score += 2
    if normalized_path.startswith("app/"):
        score += 1
    return score


def _grounded_status(
    proposal_kind: str,
    *,
    requested_goal: str,
    results: list[ProjectSearchResult],
    search: ProjectFileSearch,
    compatibility_reason: str,
) -> _GroundingStatus:
    evidence = _evidence_files(proposal_kind, requested_goal, results, search)
    candidate_paths = list(evidence) or _diagnostic_candidate_paths(results)
    if not evidence:
        reason = (
            "project search results were empty or were rejected because they did "
            "not provide matching editable source evidence after excluding "
            "README, documentation, and test/manual noise."
        )
        if compatibility_reason:
            reason = f"{compatibility_reason} {reason}"
        return _GroundingStatus(
            evidence=evidence,
            candidate_paths=candidate_paths,
            change=None,
            reason=reason,
        )
    if compatibility_reason:
        return _GroundingStatus(
            evidence=evidence,
            candidate_paths=candidate_paths,
            change=None,
            reason=compatibility_reason,
        )
    change = _generate_grounded_change(
        proposal_kind,
        requested_goal=requested_goal,
        evidence=evidence,
        search=search,
    )
    if change is None:
        return _GroundingStatus(
            evidence=evidence,
            candidate_paths=candidate_paths,
            change=None,
            reason=_no_grounded_diff_reason(proposal_kind, requested_goal),
        )
    return _GroundingStatus(
        evidence=evidence,
        candidate_paths=candidate_paths,
        change=change,
        reason="",
    )


def _diagnostic_candidate_paths(results: list[ProjectSearchResult]) -> list[str]:
    paths: list[str] = []
    for result in results:
        if result.path not in paths:
            paths.append(result.path)
        if len(paths) >= _MAX_RELEVANT_FILES:
            break
    return paths


def _content_supports_kind(proposal_kind: str, path: str, content: str) -> bool:
    normalized = content.lower()
    if proposal_kind == _COLOR_KIND:
        return path.endswith(".css") and "bubble" in normalized
    if path.endswith(".css"):
        return ".bubble.thinking" in normalized
    return path.endswith(".js") and "thinking" in normalized


def _setting_limit_evidence(
    requested_goal: str,
    results: list[ProjectSearchResult],
    search: ProjectFileSearch,
) -> dict[str, str]:
    expanded_terms = _expanded_setting_terms(requested_goal)
    candidate_scores: dict[str, int] = {}
    for result in results:
        path = result.path
        if _non_editable_evidence_path(path):
            continue
        content = search.read_text(path)
        if content is None:
            continue
        score = _setting_limit_score(path, content, expanded_terms)
        if score <= 0:
            continue
        candidate_scores[path] = max(score, candidate_scores.get(path, 0))

    evidence: dict[str, str] = {}
    for path, _score in sorted(
        candidate_scores.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        content = search.read_text(path)
        if content is not None:
            evidence[path] = content
    return evidence


def _non_editable_evidence_path(path: str) -> bool:
    normalized = path.lower()
    return any(part in normalized for part in _NON_EDITABLE_EVIDENCE_PARTS)


def _setting_limit_score(path: str, content: str, terms: list[str]) -> int:
    searchable = f"{path} {content}".lower()
    score = sum(1 for term in terms if term.lower() in searchable)
    if "config" in path.lower() or "settings" in searchable:
        score += 2
    if _SETTING_LIMIT_ENV_INT_PATTERN.search(content):
        score += 3
    return score


def _generate_grounded_change(
    proposal_kind: str,
    *,
    requested_goal: str,
    evidence: dict[str, str],
    search: ProjectFileSearch,
) -> _GroundedChange | None:
    if proposal_kind == _SETTING_LIMIT_KIND:
        return _setting_limit_change(requested_goal, evidence)
    if proposal_kind == _GENERIC_FEATURE_KIND:
        preview = generate_grounded_patch_preview(
            requested_goal=requested_goal,
            relevant_files=list(evidence),
            search=search,
        )
        if preview is None:
            return None
        return _GroundedChange(
            path=preview.path,
            original_text=evidence[preview.path],
            updated_text=None,
            patch_preview=preview.patch_preview,
        )

    path = "app/static/lewis.css"
    original = evidence.get(path)
    if original is None:
        return None
    if proposal_kind == _COLOR_KIND:
        updated = _chat_text_blue(original)
    else:
        updated = _thinking_animation(original)
    if updated == original:
        return None
    return _GroundedChange(
        path=path,
        original_text=original,
        updated_text=updated,
        patch_preview=_unified_diff(path, original, updated),
    )


def _setting_limit_change(
    requested_goal: str,
    evidence: dict[str, str],
) -> _GroundedChange | None:
    expanded_terms = _expanded_setting_terms(requested_goal)
    for path, original in evidence.items():
        updated = _increase_integer_setting_default(original, expanded_terms)
        if updated == original:
            continue
        return _GroundedChange(
            path=path,
            original_text=original,
            updated_text=updated,
            patch_preview=_unified_diff(path, original, updated),
        )
    return None


def _increase_integer_setting_default(content: str, terms: list[str]) -> str:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        if changed:
            return match.group(0)
        env_name = match.group("env_name")
        if not _env_name_matches_terms(env_name, terms):
            return match.group(0)
        current_value = int(match.group("value"))
        next_value = _increased_limit_value(current_value)
        changed = True
        return f"{match.group('prefix')}{next_value}{match.group('suffix')}"

    return _SETTING_LIMIT_ENV_INT_PATTERN.sub(replace, content)


def _env_name_matches_terms(env_name: str, terms: list[str]) -> bool:
    term_set = {term.lower() for term in terms}
    env_tokens = {token.lower() for token in env_name.split("_") if token}
    if not {"max", "limit"} & env_tokens:
        return False
    return len(env_tokens & term_set) >= 2


def _increased_limit_value(current_value: int) -> int:
    return current_value + max(1, (current_value + 1) // 2)


def _chat_text_blue(content: str) -> str:
    match = _CHAT_COLOR_BLOCK_PATTERN.search(content)
    if match is not None:
        body = match.group("body")
        updated_body, count = re.subn(
            r"color\s*:\s*var\(--ink\)\s*;",
            "color: #2563eb;",
            body,
            count=1,
            flags=re.IGNORECASE,
        )
        if count:
            return f"{content[: match.start('body')]}{updated_body}{content[match.end('body') :]}"
        if "color:" not in body.lower():
            updated_body = _add_css_declaration(body, "color: #2563eb;")
            return f"{content[: match.start('body')]}{updated_body}{content[match.end('body') :]}"

    suffix = "" if content.endswith("\n") else "\n"
    return f"{content}{suffix}\n.chat-log .bubble:not(.thinking) {{\n  color: #2563eb;\n}}\n"


def _thinking_animation(content: str) -> str:
    match = _THINKING_BLOCK_PATTERN.search(content)
    if match is None:
        return content
    body = match.group("body")
    updated_body = _add_css_declaration(
        body,
        "animation: lewis-thinking-pulse 1.2s ease-in-out infinite;",
    )
    updated = (
        f"{content[: match.start('body')]}{updated_body}{content[match.end('body') :]}"
    )
    reduced_block = (
        "\n@media (prefers-reduced-motion: reduce) {\n"
        "  .bubble.thinking { animation: none; }\n"
        "}\n"
    )
    if "@keyframes lewis-thinking-pulse" in updated:
        return updated + reduced_block
    suffix = "" if updated.endswith("\n") else "\n"
    return (
        f"{updated}{suffix}\n"
        "@keyframes lewis-thinking-pulse {\n"
        "  0%, 100% { transform: translateY(0); opacity: .7; }\n"
        "  50% { transform: translateY(-2px); opacity: 1; }\n"
        "}\n" + reduced_block
    )


def _add_css_declaration(body: str, declaration: str) -> str:
    if declaration.split(":", 1)[0].lower() in body.lower():
        return body
    if "\n" not in body:
        separator = " " if body.rstrip() else ""
        return f"{body.rstrip()}{separator}{declaration} "
    nonempty_lines = [line for line in body.splitlines() if line.strip()]
    indent_match = re.match(r"\s*", nonempty_lines[0]) if nonempty_lines else None
    indent = indent_match.group(0) if indent_match else "  "
    closing_indent = body.rsplit("\n", 1)[-1]
    return f"{body.rstrip()}\n{indent}{declaration}\n{closing_indent}"


def _unified_diff(path: str, original: str, updated: str) -> str:
    body = "\n".join(
        difflib.unified_diff(
            original.splitlines(),
            updated.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    return f"diff --git a/{path} b/{path}\n{body}"


def _validated_grounded_patch(
    change: _GroundedChange,
    *,
    relevant_files: list[str],
    project_root: Any,
) -> str:
    preview = validate_patch_preview(change.patch_preview, project_root)
    if not preview.startswith("diff --git "):
        raise ValueError("Grounded patch preview must start with a git diff header.")
    if not all(marker in preview for marker in ("\n--- ", "\n+++ ", "\n@@ ")):
        raise ValueError("Grounded patch preview is missing unified diff markers.")
    touched_paths = set(extract_patch_preview_paths(preview))
    if touched_paths != {change.path} or not touched_paths <= set(relevant_files):
        raise ValueError(
            "Grounded patch preview touches a file outside selected evidence."
        )
    if change.updated_text is not None and change.original_text == change.updated_text:
        raise ValueError("Grounded patch preview must describe a real change.")
    if change.updated_text is None and not _patch_has_changed_lines(preview):
        raise ValueError("Grounded patch preview must describe a real change.")
    return preview


def _patch_has_changed_lines(preview: str) -> bool:
    lines = preview.splitlines()
    has_removal = any(
        line.startswith("-") and not line.startswith("---") for line in lines
    )
    has_addition = any(
        line.startswith("+") and not line.startswith("+++") for line in lines
    )
    return has_removal and has_addition


def _proposed_approach(
    proposal_kind: str,
    *,
    requested_goal: str,
    relevant_files: list[str],
    touched_path: str,
) -> str:
    evidence = ", ".join(f"`{path}`" for path in relevant_files)
    if proposal_kind == _COLOR_KIND:
        implementation = (
            "Update `app/static/lewis.css` with a chat-scoped blue text rule instead "
            "of changing the global ink color."
        )
    elif proposal_kind == _THINKING_ANIMATION_KIND:
        implementation = (
            "Use the existing thinking-bubble lifecycle in `app/static/chat.js` and "
            "extend the `.bubble.thinking` styles in `app/static/lewis.css` with a "
            "bounded waiting animation."
        )
    elif proposal_kind == _SETTING_LIMIT_KIND:
        implementation = (
            f"Update the existing environment-backed integer limit default in "
            f"`{touched_path}` for {requested_goal.rstrip('.').lower()}, leaving "
            "runtime `.env` contents untouched."
        )
    elif proposal_kind == _GENERIC_FEATURE_KIND:
        implementation = (
            f"Use the existing behavior in `{touched_path}` to implement "
            f"{requested_goal.rstrip('.').lower()} with the conservative grounded "
            "change shown in the patch preview."
        )
    else:
        implementation = f"Review the grounded evidence and plan {requested_goal.rstrip('.').lower()}."
    return (
        f"{implementation} "
        f"Safe read-only project evidence selected: {evidence}. "
        f"{PATCH_PREVIEW_REVIEW_ONLY_NOTE}"
    )


def _target_surface_for_kind(proposal_kind: str) -> str:
    if proposal_kind == _SETTING_LIMIT_KIND:
        return "backend/service logic"
    if proposal_kind == _GENERIC_FEATURE_KIND:
        return "unspecified"
    return "command center web UI"


def _is_setting_limit_goal(cleaned_goal: str) -> bool:
    return any(
        term in cleaned_goal for term in ("increase", "raise", "bump", "expand")
    ) and any(term in cleaned_goal for term in ("limit", "cap", "maximum"))


def _no_grounded_diff_reason(proposal_kind: str, requested_goal: str) -> str:
    if proposal_kind == _SETTING_LIMIT_KIND:
        return (
            "candidate files were found, but none contained a safe matching integer "
            "environment default that could be increased."
        )
    if proposal_kind == _GENERIC_FEATURE_KIND:
        if any(term in requested_goal.lower() for term in _DESTRUCTIVE_GOAL_TERMS):
            return (
                "evidence-backed source candidates were found, but the requested "
                "behavior is potentially destructive and needs explicit confirmation "
                "plus verified storage semantics. The engine has no conservative "
                "grounded patch strategy for that coordinated change, and saving an "
                "approval without a patch_preview is prohibited."
            )
        return (
            "evidence-backed source candidates were found, but the engine has no "
            "conservative grounded patch strategy for this feature. Saving an "
            "approval without a patch_preview is prohibited."
        )
    return (
        "candidate files were found, but their existing content did not contain a "
        "safe editable block for this proposal."
    )


def _failure_lead_in(
    candidate_paths: list[str], reason: str, requested_goal: str
) -> str:
    if not candidate_paths:
        if any(term in requested_goal.lower() for term in _DESTRUCTIVE_GOAL_TERMS):
            return _DESTRUCTIVE_NO_FILES_LEAD_IN
        return _NO_FILES_LEAD_IN
    lowered = reason.lower()
    if "destructive" in lowered or "storage-sensitive" in lowered:
        return _DESTRUCTIVE_LEAD_IN
    if (
        lowered.startswith(
            "does not support material concepts removal/clear/undo/reset polarity"
        )
        or "removal/clear/undo/reset polarity" in lowered
    ):
        return (
            "Lewis found candidate project files and enough material evidence for "
            "the target surface, but the request asks to remove/clear/reset "
            "existing styling or animation. The supported static and "
            "thinking-animation strategies only safely add/set when grounded. "
            "An explicit removal/revert strategy is not currently available."
        )
    if any(
        concept.lower() in lowered
        for concept in _UNSUPPORTED_THINKING_CONTENT_CONCEPTS
    ):
        return _UNSUPPORTED_THINKING_CONTENT_LEAD_IN
    if any(term in requested_goal.lower() for term in _DESTRUCTIVE_GOAL_TERMS):
        return _DESTRUCTIVE_LEAD_IN
    if _ANIMATION_MODIFIER_CONCEPT.lower() in lowered:
        return (
            "Lewis found candidate project files, but the request asks for "
            "qualitative animation modification (calmer, slower, subtler, "
            "less distracting, etc.). The current thinking-animation strategy "
            "only safely adds a bounded waiting animation. No conservative "
            "modifier-specific strategy is currently available for qualitative "
            "animation tuning."
        )
    if "strategy" in lowered:
        return _NO_STRATEGY_LEAD_IN
    return _NO_GROUNDED_DIFF_LEAD_IN


def _grounding_failure_message(
    *,
    requested_goal: str,
    search_records: list[dict[str, Any]],
    candidate_paths: list[str],
    reason: str,
) -> str:
    queries = [
        str(record.get("arguments", {}).get("query", "")).strip()
        for record in search_records
        if isinstance(record.get("arguments"), dict)
    ]
    queries = [query for query in queries if query]
    candidates = ", ".join(candidate_paths[:5]) if candidate_paths else "none"
    candidate_state = "yes" if candidate_paths else "no"
    tried = "; ".join(queries) if queries else "none"
    lead_in = _failure_lead_in(candidate_paths, reason, requested_goal)
    return (
        f"{lead_in}\n"
        f"Cleaned goal: {requested_goal}\n"
        f"Search terms tried: {tried}\n"
        f"Candidate files found: {candidate_state} ({candidates}).\n"
        f"Reason no grounded diff could be created: {reason}\n"
        "No pending code proposal was saved."
    )


def _tool_limit_result(
    limit: int, completed_records: list[dict[str, Any]]
) -> CodeProposalEngineResult:
    message = (
        f"Tool call limit reached ({limit}). I stopped before making more tool calls."
    )
    if completed_records:
        names = [str(record["name"]) for record in completed_records]
        summary = ", ".join(dict.fromkeys(names))
        message = f"{message} Completed tool calls this turn: {summary}."
    return CodeProposalEngineResult(
        approval=None,
        tool_records=[*completed_records, {"name": "tool_limit", "result": message}],
        message=message,
    )
