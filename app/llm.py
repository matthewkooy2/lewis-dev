from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

from openai import BadRequestError, OpenAI

from app.budget import BudgetManager
from app.code_proposal_text_tool_fallback import (
    code_proposal_args_from_search_results,
    extract_project_search_queries,
)
from app.code_proposal_tool_strategy import code_proposal_tool_strategy
from app.code_proposal_patch_preview import (
    PATCH_PREVIEW_REVIEW_ONLY_NOTE,
    PATCH_PREVIEW_UNAVAILABLE_NOTE,
    generate_grounded_patch_preview,
)
from app.config import MISSING_API_KEY_MESSAGE, Settings
from app.deepseek_messages import assistant_tool_call_api_message
from app.logging_db import RunLogger
from app.dashboard import format_dashboard_count_lines
from app.memory_retrieval import build_active_memory_context
from app.model_profiles import profile_after_tool_call, profile_for_user_input
from app.project_search_discipline import (
    has_useful_project_search_results,
    is_direct_code_change_request,
    project_search_answer,
    project_search_query,
    project_search_synthesized_answer,
    should_finish_project_search_turn,
    should_suppress_project_search,
)
from app.search_first_code_proposal import (
    create_search_first_code_proposal,
    is_search_first_code_proposal_request,
)
from app.tool_call_limits import effective_tool_call_limit
from app.text_tool_calls import extract_text_tool_calls, strip_tool_call_markup
from app.tools import ToolRegistry


def build_system_prompt(settings: Settings) -> str:
    owner = settings.user_name if settings.user_name_configured else "the user"
    signoff_instruction = (
        f"When drafting email, you may sign with {settings.user_name} if appropriate."
        if settings.user_name_configured
        else (
            "When drafting email, do not assume a personal signoff name. "
            "Use a neutral signoff like 'Best regards,' with no name beneath it."
        )
    )
    project_inspection_instruction = (
        "Developer/code workspace mode is enabled. When the user asks to inspect "
        "Lewis project source files for code-planning work, you may use "
        "list_project_files, read_project_file, and search_project. These tools "
        "are read-only, project-relative, bounded, text-only, and separate from "
        "document search. Do not generate patches, edit files, apply patches, run "
        "shell commands, run tests, use browser automation, inspect runtime data, "
        "or execute code proposals."
        if settings.developer_code_workspace
        else (
            "Developer/code workspace mode is disabled. Do not inspect Lewis "
            "repository files for code-planning work; create planning-only code "
            "proposals instead."
        )
    )
    return f"""You are Lewis, {owner}'s supervised personal assistant.
Be helpful, accurate, concise.
You may use read-only tools.
You may not take real-world actions directly.
Any real-world action must be proposed for approval.
Only call tools present in the provided tool schema. Never print or narrate
tool-call markup as assistant text.

Never actually send emails, create calendar events, delete files, edit files,
submit forms, make purchases, or message people. For those actions, call the
appropriate proposal tool and explain that the item is pending approval.
When the user asks you to create a task, reminder, todo, or follow-up item,
call propose_task. Explain that it is only a pending local task proposal, not a
real reminder. Approval marks the task proposal as approved; it does not mean
the task is done or completed. Approval does not create alerts, notifications,
calendar events, emails, or external tasks.
When the user asks you to schedule, put something on the calendar, block off
time, set up a meeting, add a calendar item, or plan an event, call
propose_calendar_event. Explain that it is only a pending local calendar
proposal. Approval only marks the proposal approved; it does not create an
Outlook, Google Calendar, OS calendar, reminder, notification, email, invite,
or external event, and it does not notify attendees.
Only when the user explicitly asks you to remember or save something for memory
with wording like "remember that", "remember this", "save this preference", or
"you should remember", you may call propose_memory. Do not infer memories from
ordinary conversation, repeated facts, preferences, or useful context. A memory
proposal is only a pending approval item; it is not an active memory until the
user approves it through the local approvals flow. After calling propose_memory,
say it is pending approval and approval is required. Never say you remembered it
or saved it as an active memory.
When asked about local notes, documents, PDFs, or project files, use
search_documents. That tool only searches the allowlisted data/documents/
folder and accepts a query string only, never a path. Summarize returned
results for the user. Do not invent document contents. If the tool reports no
documents or no clear match, say that plainly.
{project_inspection_instruction}
When the user asks to add, change, update, fix, or plan code or UI behavior in
Lewis, call propose_code_change even if they do not use the phrase "code
proposal". This creates a local planning proposal only. Do not
read arbitrary files, expose sensitive/runtime files, run shell commands,
generate patches, edit files, or create migrations. Make the proposal clear
about the intended target surface when relevant: command center web UI,
CLI/terminal UI, backend/service logic, or database/migrations. If the user asks
for a UI change from command center chat, assume the command center web UI unless
they explicitly say CLI, terminal, or console. Do not reinterpret web/UI requests
as terminal ASCII features. For .env, secrets, credentials, tokens, database
files, or runtime data, do not inspect or expose contents or secret values;
prefer .env.example and environment-variable based runtime configuration.
If the user asks for a code proposal using project search first, call
search_project_files once or twice with targeted queries, then call
propose_code_change. Do not keep searching after you have enough safe file
context for a planning proposal.
When the user asks what tasks they have, call list_tasks and answer only from
the returned local data. When the user asks what calendar events or meetings
you have proposed, call list_calendar_proposals and answer only from the
returned local data. When the user asks what approvals are pending, call
list_pending_approvals and answer only from the returned local data. When the
user asks for a dashboard, call get_dashboard_summary and answer only from the
returned local data. Do not invent tasks, calendar proposals, approvals,
counts, or statuses.
If the user asks you to execute an approved item, explain that real execution is
not available yet. You may direct them to
`.\\.venv\\Scripts\\python.exe -m app.main execute-approved <approval_id>`, but
make clear this is only a local dry-run execution preview and no external
system is touched.
If a transient memory context block is provided, it contains only approved
active local memories for the current chat turn. Do not claim to remember or
rely on archived memories, rejected memory proposals, pending memory proposals,
or anything not included in that transient block.
{signoff_instruction}"""


@dataclass
class AssistantRun:
    answer: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0
    reasoning_tokens: int = 0
    api_calls: int = 0
    estimated_cost: float = 0.0
    memory_context_count: int = 0


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int | None = None
    prompt_cache_miss_tokens: int | None = None
    reasoning_tokens: int | None = None


def parse_token_usage(usage: object | None) -> TokenUsage:
    if usage is None:
        return TokenUsage()

    reasoning_tokens = _usage_optional_int(usage, "reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = _usage_optional_int(
            _usage_get(usage, "completion_tokens_details"), "reasoning_tokens"
        )

    return TokenUsage(
        prompt_tokens=_usage_int(usage, "prompt_tokens"),
        completion_tokens=_usage_int(usage, "completion_tokens"),
        total_tokens=_usage_int(usage, "total_tokens"),
        prompt_cache_hit_tokens=_usage_optional_int(usage, "prompt_cache_hit_tokens"),
        prompt_cache_miss_tokens=_usage_optional_int(usage, "prompt_cache_miss_tokens"),
        reasoning_tokens=reasoning_tokens,
    )


def _usage_get(source: object | None, key: str) -> object | None:
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _usage_optional_int(source: object | None, key: str) -> int | None:
    value = _usage_get(source, key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usage_int(source: object | None, key: str) -> int:
    return _usage_optional_int(source, key) or 0


_EXPLICIT_CODE_PROPOSAL_PATTERN = re.compile(
    r"\b(?:make|create|plan|propose|give me|i want)\b.{0,50}\bcode proposal\b",
    re.IGNORECASE,
)

_PROPOSAL_FRAMING_PATTERN = re.compile(
    r"\b(?:make|create)\b.{0,40}\b(?:a\s+)?proposal\b",
    re.IGNORECASE,
)

_BEHAVIOR_CHANGE_SIGNAL = re.compile(
    r"\b(?:add|build|change|create|fix|implement|modify|refactor|remove|"
    r"replace|update|animate|adjust|convert|transform)\b"
    r".{0,150}"
    r"\b(?:app|backend|behavior|bug|chat|code|feature|frontend|lewis|ui|"
    r"animation|animated|character|thinking|text|color|style|css|"
    r"display|render|button|input|output|screen|page|html|component|widget)\b",
    re.IGNORECASE,
)


def _has_explicit_code_proposal_phrase(user_input: str) -> bool:
    normalized = " ".join(user_input.lower().split())
    if "project search" in normalized:
        return False
    if bool(_EXPLICIT_CODE_PROPOSAL_PATTERN.search(normalized)):
        return True
    return bool(
        _PROPOSAL_FRAMING_PATTERN.search(normalized)
        and _BEHAVIOR_CHANGE_SIGNAL.search(normalized)
    )


class LewisAgent:
    def __init__(
        self,
        settings: Settings,
        tools: ToolRegistry,
        budget: BudgetManager,
        logger: RunLogger,
    ):
        self.settings = settings
        self.tools = tools
        self.budget = budget
        self.logger = logger
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt(settings)}
        ]
        self.latest_approval_id: int | None = None
        self.latest_email_draft_approval_id: int | None = None
        self.latest_task_approval_id: int | None = None
        self.latest_calendar_approval_id: int | None = None
        self.latest_memory_approval_id: int | None = None
        self.client = (
            OpenAI(
                api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url
            )
            if settings.has_api_key
            else None
        )

    def run(self, user_input: str) -> AssistantRun:
        if not self.settings.has_api_key or self.client is None:
            raise RuntimeError(MISSING_API_KEY_MESSAGE)

        search_first_code_proposal = self._maybe_run_search_first_code_proposal(
            user_input
        )
        if search_first_code_proposal is not None:
            return search_first_code_proposal

        local_tool_run = self._maybe_run_local_read_only_tool(user_input)
        if local_tool_run is not None:
            return local_tool_run

        if self._is_execute_approved_request(user_input):
            answer = self._execute_approved_safety_message(user_input)
            self._append_conversation_turn(user_input, answer)
            self.logger.log_run(
                user_input=user_input,
                model=self.settings.deepseek_model,
                tool_calls=[],
            )
            return AssistantRun(answer=answer)

        if self._is_latest_email_followup(user_input):
            answer = self._latest_email_safety_message()
            self._append_conversation_turn(user_input, answer)
            self.logger.log_run(
                user_input=user_input,
                model=self.settings.deepseek_model,
                tool_calls=[],
            )
            return AssistantRun(answer=answer)

        daily_cost = self.logger.daily_estimated_cost()
        spend_decision = self.budget.can_spend(daily_cost)
        if not spend_decision.allowed:
            return AssistantRun(answer=spend_decision.reason)

        self.messages.append({"role": "user", "content": user_input})
        messages, memory_context_count = self._messages_with_transient_memory_context(
            user_input
        )
        model_profile = profile_for_user_input(self.settings, user_input)
        tool_call_limit = effective_tool_call_limit(self.settings, model_profile)
        tool_records: list[dict[str, Any]] = []
        input_tokens = output_tokens = total_tokens = 0
        prompt_cache_hit_tokens = prompt_cache_miss_tokens = reasoning_tokens = 0
        api_calls = 0
        tools_enabled = bool(self.tools.schemas())

        while True:
            tool_strategy = code_proposal_tool_strategy(user_input, tool_records)
            tool_schemas = self.tools.schemas(tool_strategy.allowed_tool_names)
            if not self._may_expose_list_local_files(user_input):
                tool_schemas = [
                    schema
                    for schema in tool_schemas
                    if schema["function"]["name"] != "list_local_files"
                ]
            tool_choice: str | dict[str, Any] = "auto"
            if tool_strategy.required_tool_name is not None:
                tool_choice = "required"
            try:
                if tools_enabled and tool_schemas:
                    response = self.client.chat.completions.create(
                        model=model_profile.model,
                        messages=messages,
                        tools=tool_schemas,
                        tool_choice=tool_choice,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=model_profile.model,
                        messages=messages,
                    )
            except BadRequestError as exc:
                if not tools_enabled:
                    raise
                tools_enabled = False
                tool_records.append(
                    {
                        "name": "tool_calling_unavailable",
                        "result": (
                            "The model endpoint rejected tool calling, so Lewis retried "
                            "without tools for this request."
                        ),
                        "error": str(exc),
                    }
                )
                response = self.client.chat.completions.create(
                    model=model_profile.model,
                    messages=messages,
                )
            api_calls += 1
            token_usage = parse_token_usage(getattr(response, "usage", None))
            input_tokens += token_usage.prompt_tokens
            output_tokens += token_usage.completion_tokens
            total_tokens += token_usage.total_tokens
            prompt_cache_hit_tokens += token_usage.prompt_cache_hit_tokens or 0
            prompt_cache_miss_tokens += token_usage.prompt_cache_miss_tokens or 0
            reasoning_tokens += token_usage.reasoning_tokens or 0

            message = response.choices[0].message
            raw_tool_calls = list(message.tool_calls or [])
            tool_calls = self._registered_structured_tool_calls(
                raw_tool_calls,
                tool_schemas,
            )
            blocked_tool_call = len(tool_calls) != len(raw_tool_calls)
            if not raw_tool_calls:
                text_tool_calls = self._registered_text_tool_calls(
                    message.content or "",
                    tool_schemas,
                )
                if "DSML" in (message.content or ""):
                    blocked_tool_call = len(text_tool_calls) != len(
                        extract_text_tool_calls(message.content or "")
                    )
                if text_tool_calls:
                    tool_calls = text_tool_calls
            if not tool_calls:
                fallback = self._maybe_complete_text_search_first_code_proposal(
                    user_input=user_input,
                    content=message.content or "",
                    tool_records=tool_records,
                    tool_call_limit=tool_call_limit,
                )
                if fallback is not None:
                    answer = fallback
                    assistant_message = {"role": "assistant", "content": answer}
                    messages.append(assistant_message)
                    self.messages.append(assistant_message)
                    estimated_cost = self.budget.estimate_cost(
                        input_tokens, output_tokens
                    )
                    run = AssistantRun(
                        answer=answer,
                        tool_calls=tool_records,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                        reasoning_tokens=reasoning_tokens,
                        api_calls=api_calls,
                        estimated_cost=estimated_cost,
                        memory_context_count=memory_context_count,
                    )
                    self.logger.log_run(
                        user_input=user_input,
                        model=model_profile.model,
                        tool_calls=tool_records,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                        reasoning_tokens=reasoning_tokens,
                        api_calls=api_calls,
                        estimated_cost=estimated_cost,
                    )
                    self._print_debug_token_summary(run)
                    return run

                content = message.content or ""
                if (
                    blocked_tool_call or extract_text_tool_calls(content)
                ) and has_useful_project_search_results(tool_records):
                    visible_answer = project_search_synthesized_answer(
                        user_input,
                        tool_records,
                    )
                elif blocked_tool_call:
                    visible_answer = "I could not use that unavailable tool safely."
                else:
                    visible_answer = self._visible_assistant_content(content)
                answer = self._apply_proposal_safety(visible_answer, tool_records)
                assistant_message = {"role": "assistant", "content": answer}
                messages.append(assistant_message)
                self.messages.append(assistant_message)
                estimated_cost = self.budget.estimate_cost(input_tokens, output_tokens)
                run = AssistantRun(
                    answer=answer,
                    tool_calls=tool_records,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                    reasoning_tokens=reasoning_tokens,
                    api_calls=api_calls,
                    estimated_cost=estimated_cost,
                    memory_context_count=memory_context_count,
                )
                self.logger.log_run(
                    user_input=user_input,
                    model=model_profile.model,
                    tool_calls=tool_records,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                    reasoning_tokens=reasoning_tokens,
                    api_calls=api_calls,
                    estimated_cost=estimated_cost,
                )
                self._print_debug_token_summary(run)
                return run

            batch_decision = self.budget.can_make_tool_calls(
                len(tool_records),
                len(tool_calls),
                limit=tool_call_limit,
            )
            if not batch_decision.allowed:
                stop_text = self._tool_limit_stop_text(
                    batch_decision.reason,
                    tool_records,
                )
                estimated_cost = self.budget.estimate_cost(input_tokens, output_tokens)
                tool_records.append({"name": "tool_limit", "result": stop_text})
                self.logger.log_run(
                    user_input=user_input,
                    model=model_profile.model,
                    tool_calls=tool_records,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                    reasoning_tokens=reasoning_tokens,
                    api_calls=api_calls,
                    estimated_cost=estimated_cost,
                )
                assistant_message = {"role": "assistant", "content": stop_text}
                messages.append(assistant_message)
                self.messages.append(assistant_message)
                run = AssistantRun(
                    answer=stop_text,
                    tool_calls=tool_records,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                    reasoning_tokens=reasoning_tokens,
                    api_calls=api_calls,
                    estimated_cost=estimated_cost,
                    memory_context_count=memory_context_count,
                )
                self._print_debug_token_summary(run)
                return run

            assistant_message = assistant_tool_call_api_message(message, tool_calls)
            messages.append(assistant_message)
            self.messages.append(assistant_message)
            for call in tool_calls:
                decision = self.budget.can_make_tool_call(
                    len(tool_records),
                    limit=tool_call_limit,
                )
                if not decision.allowed:
                    stop_text = self._tool_limit_stop_text(
                        decision.reason,
                        tool_records,
                    )
                    estimated_cost = self.budget.estimate_cost(
                        input_tokens, output_tokens
                    )
                    tool_records.append({"name": "tool_limit", "result": stop_text})
                    self.logger.log_run(
                        user_input=user_input,
                        model=model_profile.model,
                        tool_calls=tool_records,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                        reasoning_tokens=reasoning_tokens,
                        api_calls=api_calls,
                        estimated_cost=estimated_cost,
                    )
                    assistant_message = {"role": "assistant", "content": stop_text}
                    messages.append(assistant_message)
                    self.messages.append(assistant_message)
                    run = AssistantRun(
                        answer=stop_text,
                        tool_calls=tool_records,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                        prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                        reasoning_tokens=reasoning_tokens,
                        api_calls=api_calls,
                        estimated_cost=estimated_cost,
                        memory_context_count=memory_context_count,
                    )
                    self._print_debug_token_summary(run)
                    return run

                function = call.function
                if (
                    function.name == "propose_memory"
                    and not self._is_explicit_memory_request(user_input)
                ):
                    result = {
                        "ok": False,
                        "error": (
                            "Memory proposals require an explicit user request "
                            "to remember or save something."
                        ),
                    }
                else:
                    suppression_reason = None
                    if function.name == "search_project_files":
                        suppression_reason = should_suppress_project_search(
                            project_search_query(function.arguments),
                            tool_records,
                        )
                    if suppression_reason is not None:
                        result = {
                            "ok": False,
                            "suppressed": True,
                            "error": suppression_reason,
                            "results": [],
                        }
                    else:
                        result = self.tools.execute(function.name, function.arguments)
                model_profile = profile_after_tool_call(
                    self.settings,
                    model_profile,
                    function.name,
                )
                tool_call_limit = effective_tool_call_limit(
                    self.settings,
                    model_profile,
                )
                record = {
                    "name": function.name,
                    "arguments": function.arguments,
                    "result": result,
                }
                tool_records.append(record)
                self._track_approval(function.name, result)
                tool_message = {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result, sort_keys=True),
                }
                messages.append(tool_message)
                self.messages.append(tool_message)

            if should_finish_project_search_turn(user_input, tool_records):
                answer = project_search_answer(tool_records)
                assistant_message = {"role": "assistant", "content": answer}
                messages.append(assistant_message)
                self.messages.append(assistant_message)
                estimated_cost = self.budget.estimate_cost(input_tokens, output_tokens)
                run = AssistantRun(
                    answer=answer,
                    tool_calls=tool_records,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                    reasoning_tokens=reasoning_tokens,
                    api_calls=api_calls,
                    estimated_cost=estimated_cost,
                    memory_context_count=memory_context_count,
                )
                self.logger.log_run(
                    user_input=user_input,
                    model=model_profile.model,
                    tool_calls=tool_records,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    prompt_cache_hit_tokens=prompt_cache_hit_tokens,
                    prompt_cache_miss_tokens=prompt_cache_miss_tokens,
                    reasoning_tokens=reasoning_tokens,
                    api_calls=api_calls,
                    estimated_cost=estimated_cost,
                )
                self._print_debug_token_summary(run)
                return run

    def _maybe_run_search_first_code_proposal(
        self, user_input: str
    ) -> AssistantRun | None:
        if (
            is_search_first_code_proposal_request(user_input)
            or is_direct_code_change_request(user_input)
            or _has_explicit_code_proposal_phrase(user_input)
        ):
            return self._run_deterministic_code_proposal_path(user_input)

        return None

    def _run_deterministic_code_proposal_path(self, user_input: str) -> AssistantRun:
        model_profile = profile_for_user_input(self.settings, user_input)
        tool_call_limit = effective_tool_call_limit(self.settings, model_profile)
        if tool_call_limit < 2:
            answer = self._tool_limit_stop_text(
                f"Tool call limit reached ({tool_call_limit}).",
                [],
            )
            tool_records = [{"name": "tool_limit", "result": answer}]
        else:
            result = create_search_first_code_proposal(
                user_input=user_input,
                search=self.tools.project_file_search,
                store=self.tools.approval_store,
                max_searches=min(2, tool_call_limit - 1),
            )
            if result.approval is not None:
                proposal_result = result.tool_records[-1]["result"]
                self._track_approval("propose_code_change", proposal_result)
            answer = result.message
            tool_records = result.tool_records

        self._append_conversation_turn(user_input, answer)
        self.logger.log_run(
            user_input=user_input,
            model=model_profile.model,
            tool_calls=tool_records,
        )
        return AssistantRun(answer=answer, tool_calls=tool_records)

    def _maybe_complete_text_search_first_code_proposal(
        self,
        *,
        user_input: str,
        content: str,
        tool_records: list[dict[str, Any]],
        tool_call_limit: int,
    ) -> str | None:
        strategy = code_proposal_tool_strategy(user_input, tool_records)
        if strategy.required_tool_name != "search_project_files":
            return None
        queries = extract_project_search_queries(content)
        if not queries:
            return None

        search_records: list[dict[str, Any]] = []
        for query in queries:
            decision = self.budget.can_make_tool_call(
                len(tool_records),
                limit=tool_call_limit,
            )
            if not decision.allowed:
                stop_text = self._tool_limit_stop_text(decision.reason, tool_records)
                tool_records.append({"name": "tool_limit", "result": stop_text})
                return stop_text
            suppression_reason = should_suppress_project_search(query, tool_records)
            if suppression_reason is not None:
                result = {
                    "ok": False,
                    "suppressed": True,
                    "error": suppression_reason,
                    "results": [],
                }
            else:
                result = self.tools.execute("search_project_files", {"query": query})
            record = {
                "name": "search_project_files",
                "arguments": {"query": query},
                "result": result,
            }
            tool_records.append(record)
            search_records.append(record)
            if isinstance(result, dict) and result.get("results"):
                break

        if not has_useful_project_search_results(tool_records):
            return project_search_answer(tool_records)

        decision = self.budget.can_make_tool_call(
            len(tool_records),
            limit=tool_call_limit,
        )
        if not decision.allowed:
            stop_text = self._tool_limit_stop_text(decision.reason, tool_records)
            tool_records.append({"name": "tool_limit", "result": stop_text})
            return stop_text

        proposal_args = code_proposal_args_from_search_results(
            user_input,
            search_records,
        )
        grounded_preview = generate_grounded_patch_preview(
            requested_goal=user_input,
            relevant_files=list(proposal_args["relevant_files"]),
            search=self.tools.project_file_search,
        )
        preview_note = (
            PATCH_PREVIEW_REVIEW_ONLY_NOTE
            if grounded_preview is not None
            else PATCH_PREVIEW_UNAVAILABLE_NOTE
        )
        proposal_args["proposed_approach"] = (
            f"{proposal_args['proposed_approach']} {preview_note}"
        )
        proposal_args["safety_notes"] = [
            *proposal_args["safety_notes"],
            preview_note,
        ]
        if grounded_preview is not None:
            proposal_args["patch_preview"] = grounded_preview.patch_preview
        result = self.tools.execute("propose_code_change", proposal_args)
        tool_records.append(
            {
                "name": "propose_code_change",
                "arguments": proposal_args,
                "result": result,
            }
        )
        self._track_approval("propose_code_change", result)
        if isinstance(result, dict) and result.get("ok") is True:
            return str(result.get("message", "Saved pending code proposal."))
        return "I could not save the code proposal."

    def _registered_structured_tool_calls(
        self,
        tool_calls: list[Any],
        tool_schemas: list[dict[str, Any]],
    ) -> list[Any]:
        allowed_names = self._allowed_registered_tool_names(tool_schemas)
        return [
            call
            for call in tool_calls
            if getattr(getattr(call, "function", None), "name", None) in allowed_names
        ]

    def _registered_text_tool_calls(
        self,
        content: str,
        tool_schemas: list[dict[str, Any]],
    ) -> list[Any]:
        if "DSML" not in content:
            return []
        allowed_names = self._allowed_registered_tool_names(tool_schemas)
        tool_calls: list[Any] = []
        for index, call in enumerate(extract_text_tool_calls(content), start=1):
            if call.name not in allowed_names:
                continue
            tool_calls.append(
                SimpleNamespace(
                    id=f"text_tool_call_{index}",
                    function=SimpleNamespace(
                        name=call.name,
                        arguments=json.dumps(call.arguments, sort_keys=True),
                    ),
                )
            )
        return tool_calls

    def _allowed_registered_tool_names(
        self,
        tool_schemas: list[dict[str, Any]],
    ) -> frozenset[str]:
        exposed_names = frozenset(
            str(schema["function"]["name"]) for schema in tool_schemas
        )
        return self.tools.registered_tool_names() & exposed_names

    @staticmethod
    def _visible_assistant_content(content: str) -> str:
        visible = strip_tool_call_markup(content)
        visible = re.sub(
            r"\bread_file\b",
            "an unavailable file tool",
            visible,
            flags=re.IGNORECASE,
        ).strip()
        if visible:
            return visible
        if extract_text_tool_calls(content):
            return "I could not use that unavailable tool safely."
        return visible

    @staticmethod
    def _tool_limit_stop_text(reason: str, tool_records: list[dict[str, Any]]) -> str:
        stop_text = f"{reason} I stopped before making more tool calls."
        completed = [
            str(record.get("name"))
            for record in tool_records
            if record.get("name") != "tool_limit"
        ]
        if not completed:
            return stop_text
        counts = {name: completed.count(name) for name in dict.fromkeys(completed)}
        summary = ", ".join(
            f"{name} x{count}" if count > 1 else name for name, count in counts.items()
        )
        return f"{stop_text} Completed tool calls this turn: {summary}."

    def _maybe_run_local_read_only_tool(self, user_input: str) -> AssistantRun | None:
        tool_name = self._local_read_only_tool_for(user_input)
        if tool_name is None:
            return None
        result = self.tools.execute(tool_name, {})
        answer = self._format_local_read_only_answer(tool_name, result)
        self._append_conversation_turn(user_input, answer)
        tool_records = [{"name": tool_name, "arguments": {}, "result": result}]
        self.logger.log_run(
            user_input=user_input,
            model=self.settings.deepseek_model,
            tool_calls=tool_records,
        )
        return AssistantRun(answer=answer, tool_calls=tool_records)

    @staticmethod
    def _may_expose_list_local_files(user_input: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", user_input.lower())
        normalized = " ".join(normalized.split())
        return "list local files" in normalized or "show local files" in normalized

    @staticmethod
    def _local_read_only_tool_for(user_input: str) -> str | None:
        normalized = re.sub(r"[^a-z0-9\s]", " ", user_input.lower())
        normalized = " ".join(normalized.split())
        slash_commands = {
            "/dashboard": "get_dashboard_summary",
            "/command-center": "get_dashboard_summary",
            "/tasks": "list_tasks",
            "/todos": "list_tasks",
            "/calendar": "list_calendar_proposals",
            "/approvals": "list_pending_approvals",
        }
        command = slash_commands.get(user_input.strip().lower())
        if command is not None:
            return command

        exact_commands = {
            "dashboard": "get_dashboard_summary",
            "command center": "get_dashboard_summary",
            "show dashboard": "get_dashboard_summary",
            "give me a dashboard": "get_dashboard_summary",
            "what tasks do i have": "list_tasks",
            "list tasks": "list_tasks",
            "show tasks": "list_tasks",
            "open tasks": "list_tasks",
            "current tasks": "list_tasks",
            "what todos do i have": "list_tasks",
            "list todos": "list_tasks",
            "show todos": "list_tasks",
            "what calendar proposals are pending": "list_calendar_proposals",
            "what meetings are pending approval": "list_calendar_proposals",
            "list calendar proposals": "list_calendar_proposals",
            "show calendar proposals": "list_calendar_proposals",
            "list meetings": "list_calendar_proposals",
            "show meetings": "list_calendar_proposals",
            "what approvals are pending": "list_pending_approvals",
            "pending approvals": "list_pending_approvals",
            "list pending approvals": "list_pending_approvals",
            "show pending approvals": "list_pending_approvals",
        }
        return exact_commands.get(normalized)

    @staticmethod
    def _format_local_read_only_answer(tool_name: str, result: dict[str, Any]) -> str:
        if result.get("ok") is not True:
            return f"I could not read the local data: {result.get('error', 'unknown error')}"
        if tool_name == "list_tasks":
            tasks = result.get("tasks", [])
            if not tasks:
                return "I found no local task proposals."
            lines = ["Here are your local task proposals:"]
            for task in tasks:
                lines.append(
                    f"#{task['approval_id']} [{task['status']}] {task['title']} - "
                    f"{task['description']}"
                )
            return "\n".join(lines)
        if tool_name == "list_calendar_proposals":
            proposals = result.get("calendar_proposals", [])
            if not proposals:
                return "I found no local calendar proposals."
            lines = ["Here are your local calendar proposals:"]
            for proposal in proposals:
                attendees = ", ".join(proposal.get("attendees", []))
                lines.append(
                    f"#{proposal['approval_id']} [{proposal['status']}] "
                    f"{proposal['title']} - {proposal['date_text']} "
                    f"{proposal['start_time_text']}-{proposal['end_time_text']} "
                    f"at {proposal['location']} ({attendees})"
                )
            return "\n".join(lines)
        if tool_name == "list_pending_approvals":
            approvals = result.get("approvals", [])
            if not approvals:
                return "I found no pending local approvals."
            lines = ["Pending local approvals:"]
            for item in approvals:
                lines.append(
                    f"#{item['approval_id']} {item['action_type']}: {item['title']}"
                )
            return "\n".join(lines)
        if tool_name == "get_dashboard_summary":
            dashboard = result.get("dashboard", {})
            counts = dashboard.get("counts", {})
            return "\n".join(
                ["Local dashboard:", *format_dashboard_count_lines(counts)]
            )
        return "I read the local data."

    def _append_conversation_turn(self, user_input: str, answer: str) -> None:
        self.messages.append({"role": "user", "content": user_input})
        self.messages.append({"role": "assistant", "content": answer})

    def _messages_with_transient_memory_context(
        self, user_input: str
    ) -> tuple[list[dict[str, Any]], int]:
        memory_context = build_active_memory_context(
            user_input,
            db_path=self.settings.database_path,
        )
        if not memory_context.prompt:
            return list(self.messages), 0
        return (
            [
                self.messages[0],
                {"role": "system", "content": memory_context.prompt},
                *self.messages[1:],
            ],
            memory_context.memory_count,
        )

    def _print_debug_token_summary(self, run: AssistantRun) -> None:
        if not self.settings.debug:
            return
        parts = [
            f"calls={run.api_calls}",
            f"prompt={run.input_tokens}",
            f"completion={run.output_tokens}",
            f"total={run.total_tokens}",
        ]
        if run.prompt_cache_hit_tokens:
            parts.append(f"cache_hit={run.prompt_cache_hit_tokens}")
        if run.prompt_cache_miss_tokens:
            parts.append(f"cache_miss={run.prompt_cache_miss_tokens}")
        if run.reasoning_tokens:
            parts.append(f"reasoning={run.reasoning_tokens}")
        if run.estimated_cost:
            parts.append(f"est_cost=${run.estimated_cost:.6f}")
        print("[debug] token usage: " + ", ".join(parts))

    def _track_approval(self, tool_name: str, result: dict[str, Any]) -> None:
        approval_id = result.get("approval_id")
        if result.get("ok") is not True or not isinstance(approval_id, int):
            return
        self.latest_approval_id = approval_id
        if tool_name == "propose_email_draft":
            self.latest_email_draft_approval_id = approval_id
        if tool_name == "propose_task":
            self.latest_task_approval_id = approval_id
        if tool_name == "propose_calendar_event":
            self.latest_calendar_approval_id = approval_id
        if tool_name == "propose_memory":
            self.latest_memory_approval_id = approval_id

    def _apply_proposal_safety(
        self, answer: str, tool_records: list[dict[str, Any]]
    ) -> str:
        answer = self._apply_memory_proposal_safety(answer, tool_records)
        answer = self._apply_calendar_proposal_safety(answer, tool_records)
        task_approval_id = self._latest_successful_task_approval_id(tool_records)
        if task_approval_id is None:
            return answer

        task_safety = (
            f"Saved as pending task approval item #{task_approval_id}. "
            "This is not a real reminder yet. Approval marks the task proposal "
            "as approved; it does not mark the task done or completed. No alerts, "
            "notifications, calendar events, emails, or external tasks are created."
        )
        if self._has_misleading_task_proposal_claim(answer):
            return task_safety
        if self._has_task_safety(answer):
            return answer
        return f"{answer}\n\n{task_safety}" if answer else task_safety

    def _apply_memory_proposal_safety(
        self, answer: str, tool_records: list[dict[str, Any]]
    ) -> str:
        approval_id = self._latest_successful_memory_approval_id(tool_records)
        if approval_id is None:
            return answer

        memory_safety = (
            f"Created pending memory approval #{approval_id}. Approval is required "
            "before this becomes an active memory."
        )
        if self._has_misleading_memory_proposal_claim(answer):
            return memory_safety
        if self._has_memory_safety(answer):
            return answer
        return f"{answer}\n\n{memory_safety}" if answer else memory_safety

    def _apply_calendar_proposal_safety(
        self, answer: str, tool_records: list[dict[str, Any]]
    ) -> str:
        approval_id = self._latest_successful_calendar_approval_id(tool_records)
        if approval_id is None:
            return answer

        calendar_safety = (
            f"Saved as pending local calendar proposal #{approval_id}. "
            "This is not a real calendar event. Approval only marks the proposal "
            "approved; it does not create an Outlook, Google Calendar, or OS "
            "calendar event, send invites, notify attendees, send email, or create "
            "reminders."
        )
        if self._has_misleading_calendar_proposal_claim(answer):
            return calendar_safety
        if self._has_calendar_safety(answer):
            return answer
        return f"{answer}\n\n{calendar_safety}" if answer else calendar_safety

    @staticmethod
    def _latest_successful_task_approval_id(
        tool_records: list[dict[str, Any]],
    ) -> int | None:
        for record in reversed(tool_records):
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            approval_id = (
                result.get("approval_id") if isinstance(result, dict) else None
            )
            if (
                record.get("name") == "propose_task"
                and result.get("ok") is True
                and isinstance(approval_id, int)
            ):
                return approval_id
        return None

    @staticmethod
    def _latest_successful_calendar_approval_id(
        tool_records: list[dict[str, Any]],
    ) -> int | None:
        for record in reversed(tool_records):
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            approval_id = result.get("approval_id")
            if (
                record.get("name") == "propose_calendar_event"
                and result.get("ok") is True
                and isinstance(approval_id, int)
            ):
                return approval_id
        return None

    @staticmethod
    def _latest_successful_memory_approval_id(
        tool_records: list[dict[str, Any]],
    ) -> int | None:
        for record in reversed(tool_records):
            result = record.get("result")
            if not isinstance(result, dict):
                continue
            approval_id = result.get("approval_id")
            if (
                record.get("name") == "propose_memory"
                and result.get("ok") is True
                and isinstance(approval_id, int)
            ):
                return approval_id
        return None

    @staticmethod
    def _has_misleading_task_proposal_claim(answer: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", answer.lower())
        normalized = " ".join(normalized.split())
        patterns = (
            r"\bi(?: will| ll) notify you\b",
            r"\bwill notify you\b",
            r"\bi(?: will| ll) remind you\b",
            r"\bwill remind you\b",
            r"\bi(?: will| ll) alert you\b",
            r"\bwill alert you\b",
            r"\breminder (?:is|has been) set\b",
            r"\bnotification (?:is|has been) set\b",
            r"\bmark (?:it|the task|this task)? ?as done\b",
            r"\bmarked (?:it|the task|this task)? ?as done\b",
            r"\bdone on my end\b",
            r"\btask (?:is|has been|will be) (?:done|completed)\b",
            r"\bcomplete(?:d)? (?:it|the task|this task)\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _has_misleading_memory_proposal_claim(answer: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", answer.lower())
        normalized = " ".join(normalized.split())
        patterns = (
            r"\bi(?: have| ve| will| ll)? ?remember(?:ed)?\b",
            r"\bi(?: have| ve)? ?saved (?:that|this|it).{0,40}\bmemory\b",
            r"\bsaved (?:that|this|it).{0,40}\bactive memory\b",
            r"\bit(?: is| s| has been)? ?remembered\b",
            r"\bmemory (?:is|has been) saved\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _has_memory_safety(answer: str) -> bool:
        normalized = answer.lower()
        return all(
            phrase in normalized
            for phrase in ("pending memory approval", "approval is required")
        )

    @staticmethod
    def _has_task_safety(answer: str) -> bool:
        normalized = answer.lower()
        return any(
            phrase in normalized
            for phrase in (
                "not a real reminder",
                "pending task proposal",
                "pending task approval",
                "creates no alerts",
                "no external tasks",
            )
        )

    @staticmethod
    def _has_misleading_calendar_proposal_claim(answer: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", answer.lower())
        normalized = " ".join(normalized.split())
        patterns = (
            r"\b(?:i(?: ve| have)|it(?: is| s)|event(?: is| has been)) scheduled\b",
            r"\b(?:created|added|put) (?:it|this|the event|the meeting).{0,40}\bcalendar\b",
            r"\b(?:calendar event|meeting|event) (?:is|has been) created\b",
            r"\b(?:sent|send) (?:an? )?(?:invite|invitation|email)\b",
            r"\b(?:i(?: will| ll)|will) notify (?:them|attendees|everyone)\b",
            r"\battendees (?:have been|were|will be) notified\b",
            r"\breminder (?:is|has been) set\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _has_calendar_safety(answer: str) -> bool:
        normalized = answer.lower()
        return any(
            phrase in normalized
            for phrase in (
                "pending local calendar proposal",
                "not a real calendar event",
                "does not create an outlook",
                "does not notify attendees",
                "send invites",
            )
        )

    def _is_latest_email_followup(self, user_input: str) -> bool:
        if self.latest_email_draft_approval_id is None:
            return False
        normalized = re.sub(r"[^a-z0-9\s]", " ", user_input.lower())
        normalized = " ".join(normalized.split())
        patterns = (
            r"\bsend (that|the)? ?(email|draft|it)\b",
            r"\blooks good\b.*\bsend (it|that|the draft|the email)\b",
            r"\bapprove (it|that|the draft|that draft|the email|that email)\b",
            r"\bthat looks good\b.*\b(approve|send)\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    def _latest_email_safety_message(self) -> str:
        approval_id = self.latest_email_draft_approval_id
        return (
            "I can't send emails directly yet. "
            f"That draft is saved as approval item #{approval_id}. "
            "You can review it with `python -m app.main approvals` and mark it "
            f"approved with `python -m app.main approve {approval_id}`. "
            "Approval currently only marks status; it does not send externally."
        )

    @staticmethod
    def _is_execute_approved_request(user_input: str) -> bool:
        normalized = re.sub(r"[^a-z0-9#\s]", " ", user_input.lower())
        normalized = " ".join(normalized.split())
        return bool(
            re.search(r"\b(execute|run|perform|carry out)\b", normalized)
            and re.search(r"\b(approved|approval|item)\b", normalized)
        )

    @staticmethod
    def _is_explicit_memory_request(user_input: str) -> bool:
        normalized = re.sub(r"[^a-z0-9\s]", " ", user_input.lower())
        normalized = " ".join(normalized.split())
        patterns = (
            r"\b(?:please )?remember (?:that|this|the following)\b",
            r"\b(?:you should|can you|could you|please) remember\b",
            r"\bsave (?:this|that|the following|my) "
            r"(?:preference|fact|memory|context|detail)\b",
            r"\bsave (?:this|that|it) (?:to|in|as) memory\b",
            r"\bstore (?:this|that|it) (?:to|in|as) memory\b",
            r"\bkeep (?:this|that|it) in memory\b",
        )
        return any(re.search(pattern, normalized) for pattern in patterns)

    @staticmethod
    def _execute_approved_safety_message(user_input: str) -> str:
        match = re.search(r"#?\b(\d+)\b", user_input)
        approval_id = match.group(1) if match else "<approval_id>"
        return (
            "Real execution is not available yet. Lewis can only do a local "
            "dry-run execution preview for approved items, and that preview does "
            "not send email, create calendar events, create reminders, or touch "
            "external systems.\n\n"
            "To preview what a future integration would do, run:\n"
            f"`.\\.venv\\Scripts\\python.exe -m app.main execute-approved {approval_id}`"
        )
