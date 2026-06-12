from __future__ import annotations

import html
from typing import Any

from app.code_proposal_previews import patch_preview_warning
from app.web_rules import (
    HISTORY_FILTER_LABELS,
    ActionButton,
    timestamp_display,
    visible_actions,
)
from app.memory_store import Memory, MemoryEvent


def page_html(title: str, active_path: str, content: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lewis - {escape(title)}</title>
  <link rel="stylesheet" href="/static/lewis.css">
</head>
<body>
  <aside>
    <div class="brand"><span class="brand-mark">L</span><span>Lewis</span></div>
    <nav>
      {nav_group(active_path, "Overview", (("/dashboard", "Dashboard"), ("/usage", "Usage")))}
      {nav_group(active_path, "Review", (("/approvals", "Approvals"), ("/tasks", "Tasks"), ("/calendar", "Calendar"), ("/history", "History")))}
      {nav_group(active_path, "Knowledge", (("/memories", "Memories"), ("/documents", "Document Search")))}
      {nav_group(active_path, "Assistant", (("/chat", "Chat"),))}
    </nav>
    <div class="aside-foot"><span class="dot"></span>Local-only &middot; 127.0.0.1</div>
  </aside>
  <main>{content}</main>
</body>
</html>"""


def nav_group(
    active_path: str, label: str, links: tuple[tuple[str, str], ...]
) -> str:
    items = "".join(nav_link(active_path, path, text) for path, text in links)
    return (
        "<div class='nav-group'>"
        f"<span class='nav-label'>{escape(label)}</span>"
        f"{items}"
        "</div>"
    )


def nav_link(active_path: str, path: str, label: str) -> str:
    active = " class='active'" if active_path == path else ""
    return f"<a{active} href='{path}'>{escape(label)}</a>"


def page_header(title: str, description: str, action_html: str = "") -> str:
    action = f"<div class='toolbar-actions'>{action_html}</div>" if action_html else ""
    return (
        "<section class='toolbar'>"
        "<div>"
        f"<h1>{escape(title)}</h1>"
        f"<p>{escape(description)}</p>"
        "</div>"
        f"{action}"
        "</section>"
    )


def metric(label: str, value: object) -> str:
    return f"<div class='metric'><span>{escape(label)}</span><strong>{escape(value)}</strong></div>"


def usage_summary_html(usage: dict[str, Any] | None) -> str:
    if not usage or not usage.get("run_count"):
        return "<p class='muted'>No usage logs yet.</p>"
    rows = [
        ("runs", usage.get("run_count", 0)),
        ("prompt/input tokens", usage.get("input_tokens", 0)),
        ("output tokens", usage.get("output_tokens", 0)),
        ("cache-hit tokens", usage.get("prompt_cache_hit_tokens", 0)),
        ("cache-miss tokens", usage.get("prompt_cache_miss_tokens", 0)),
        ("reasoning tokens", usage.get("reasoning_tokens", 0)),
        ("estimated cost", f"${float(usage.get('estimated_cost', 0.0)):.6f}"),
        ("API calls", usage.get("api_calls", 0)),
    ]
    return (
        "<div class='fields'>"
        + "".join(field(label, value) for label, value in rows)
        + "</div>"
    )


def usage_runs_html(runs: list[Any]) -> str:
    if not runs:
        return "<section class='band'><h2>Recent Runs</h2><p class='muted'>No usage logs yet.</p></section>"
    rows = []
    for run in runs:
        rows.append(
            "<tr>"
            f"<td>{escape(run.timestamp)}</td>"
            f"<td>{escape(run.user_input)}</td>"
            f"<td>{escape(run.input_tokens)}</td>"
            f"<td>{escape(run.output_tokens)}</td>"
            f"<td>{escape(run.prompt_cache_hit_tokens)}</td>"
            f"<td>{escape(run.prompt_cache_miss_tokens)}</td>"
            f"<td>{escape(run.reasoning_tokens)}</td>"
            f"<td>${run.estimated_cost:.6f}</td>"
            f"<td>{escape(run.api_calls)}</td>"
            "</tr>"
        )
    return (
        "<section class='band'><h2>Recent Runs</h2><div class='table-scroll'><table>"
        "<thead><tr><th>Time</th><th>Input</th><th>Prompt/Input</th>"
        "<th>Output</th><th>Cache Hit</th><th>Cache Miss</th>"
        "<th>Reasoning</th><th>Estimated Cost</th><th>API Calls</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div></section>"
    )


def approval_cards_html(items: list[dict[str, Any]], context: str) -> str:
    if not items:
        return empty_state_html("No approval items found.")
    cards = []
    for item in items:
        note = status_note_html(item)
        timestamp = timestamp_display(item, context)
        cards.append(
            "<article class='card'>"
            "<div class='card-head'>"
            f"<h3>#{escape(item['approval_id'])} {escape(item['title'])}</h3>"
            "<div class='meta'>"
            f"{status_badge(item['status'])}"
            f"{type_badge(item['action_type'])}"
            f"<span class='meta-time'>{escape(timestamp.meta_label)} {escape(timestamp.value)}</span>"
            "</div>"
            "</div>"
            f"{payload_html(item['action_type'], item['payload'])}"
            f"{note}"
            "<details class='card-details'>"
            "<summary>Record details</summary>"
            "<div class='fields'>"
            f"{field('id', item['approval_id'])}"
            f"{field('status', item['status'])}"
            f"{field('type', item['action_type'])}"
            f"{field('summary', item['title'])}"
            f"{field(timestamp.field_label, timestamp.value)}"
            "</div>"
            "</details>"
            f"{item_actions_html(item, context)}"
            "</article>"
        )
    return "<section class='grid'>" + "".join(cards) + "</section>"


def item_actions_html(item: dict[str, Any], context: str) -> str:
    actions = [form_button(action) for action in visible_actions(item, context)]
    if not actions:
        return ""
    return "<div class='actions'>" + "".join(actions) + "</div>"


def task_cards_html(tasks: list[dict[str, Any]]) -> str:
    if not tasks:
        return empty_state_html("No local task proposals found.")
    cards = []
    for task in tasks:
        item = {
            "approval_id": task["approval_id"],
            "action_type": "propose_task",
            "status": task["status"],
        }
        fields = [
            ("title", task["title"]),
            ("description", task["description"]),
            ("due date text", task["due_date_text"]),
            ("priority", task["priority"]),
            ("status", task["status"]),
            ("created", task["created_at"]),
        ]
        cards.append(
            "<article class='card'>"
            f"<h3>#{escape(task['approval_id'])} {escape(task['title'])}</h3>"
            f"<div class='meta'>{status_badge(task['status'])}</div>"
            f"<div class='fields'>{''.join(field(label, value) for label, value in fields)}</div>"
            f"{item_actions_html(item, 'tasks')}"
            "</article>"
        )
    return "<section class='grid'>" + "".join(cards) + "</section>"


def calendar_cards_html(proposals: list[dict[str, Any]]) -> str:
    if not proposals:
        return empty_state_html("No local calendar proposals found.")
    cards = []
    for proposal in proposals:
        item = {
            "approval_id": proposal["approval_id"],
            "action_type": "propose_calendar_event",
            "status": proposal["status"],
        }
        fields = [
            ("title", proposal["title"]),
            ("description", proposal["description"]),
            ("date", proposal["date_text"]),
            ("start time", proposal["start_time_text"]),
            ("end time", proposal["end_time_text"]),
            ("location", proposal["location"]),
            ("attendees", ", ".join(proposal["attendees"])),
            ("status", proposal["status"]),
            ("created", proposal["created_at"]),
        ]
        cards.append(
            "<article class='card'>"
            f"<h3>#{escape(proposal['approval_id'])} {escape(proposal['title'])}</h3>"
            f"<div class='meta'>{status_badge(proposal['status'])}</div>"
            f"<div class='fields'>{''.join(field(label, value) for label, value in fields)}</div>"
            f"{item_actions_html(item, 'calendar')}"
            "</article>"
        )
    return "<section class='grid'>" + "".join(cards) + "</section>"


def memory_cards_html(memories: list[Memory]) -> str:
    if not memories:
        return empty_state_html("No active memories found.")
    cards = []
    for memory in memories:
        tags_value = escape(", ".join(memory.tags))
        tags_input = (
            f"<input name='tags' value='{tags_value}' placeholder='style, project'>"
        )
        content_textarea = (
            f"<textarea name='content' rows='5'>{escape(memory.content)}</textarea>"
        )
        reason_input = (
            "<input name='reason' placeholder='Optional context for reviewers'>"
        )
        archive_reason_input = "<input name='reason' placeholder='Optional'>"
        cards.append(
            "<article class='card memory-card'>"
            "<div class='memory-card-header'>"
            "<div>"
            f"<h3>Memory #{escape(memory.id)}</h3>"
            f"<div class='meta'>{status_badge(memory.status)}{type_badge(memory.memory_type)}</div>"
            "</div>"
            f"<span class='memory-source'>source: {escape(memory.source)}</span>"
            "</div>"
            f"<div class='memory-content'>{escape(memory.content)}</div>"
            f"{tag_list_html(memory.tags)}"
            "<div class='memory-details'>"
            f"{compact_detail('Approval', f'#{memory.approval_id}')}"
            f"{compact_detail('Created', memory.created_at)}"
            "</div>"
            "<details class='memory-update'>"
            "<summary>Propose Replacement</summary>"
            "<form class='memory-update-form' method='post' "
            f"action='/memories/{escape(memory.id)}/propose-update'>"
            "<div class='form-grid'>"
            f"{labeled_control('Memory type', memory_type_select(memory.memory_type))}"
            f"{labeled_control('Tags', tags_input)}"
            "</div>"
            f"{labeled_control('Replacement memory', content_textarea)}"
            f"{labeled_control('Reason', reason_input)}"
            "<div class='form-actions'><button type='submit'>propose replacement</button></div>"
            "</form>"
            "</details>"
            "<form class='archive-form' method='post' "
            f"action='/memories/{escape(memory.id)}/archive'>"
            f"{labeled_control('Archive reason', archive_reason_input)}"
            "<div class='form-actions'><button class='warn' type='submit'>archive</button></div>"
            "</form>"
            "</article>"
        )
    return "<section class='grid'>" + "".join(cards) + "</section>"


def memory_type_select(selected: str) -> str:
    options = []
    for value in ("fact", "preference", "project_context"):
        selected_attr = " selected" if value == selected else ""
        options.append(
            f"<option value='{escape(value)}'{selected_attr}>{escape(value)}</option>"
        )
    return "<select name='memory_type'>" + "".join(options) + "</select>"


def memory_events_html(events: list[MemoryEvent]) -> str:
    if not events:
        return "<p class='muted'>No memory audit events found.</p>"
    rows = []
    for event in events:
        rows.append(
            "<tr>"
            f"<td>{escape(event.created_at)}</td>"
            f"<td>{escape(event.event_type)}</td>"
            f"<td>{escape(event.memory_id if event.memory_id is not None else '')}</td>"
            f"<td>{escape(event.approval_id if event.approval_id is not None else '')}</td>"
            f"<td>{escape(event.actor)}</td>"
            f"<td>{escape(event.details)}</td>"
            "</tr>"
        )
    return (
        "<div class='table-scroll'><table class='audit-table'>"
        "<thead><tr><th>Time</th><th>Event</th><th>Memory</th>"
        "<th>Approval</th><th>Actor</th><th>Details</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def history_filters_html(selected: str) -> str:
    return (
        "<section class='filters'>"
        + "".join(
            f"<a class='filter{' active' if key == selected else ''}' href='/history?type={key}'>{escape(label)}</a>"
            for key, label in HISTORY_FILTER_LABELS
        )
        + "</section>"
    )


def status_note_html(item: dict[str, Any]) -> str:
    if item["status"] != "execution_previewed":
        return ""
    return (
        "<p class='muted'>Execution was previewed locally; no external action was "
        "taken.</p>"
    )


def payload_html(action_type: str, payload: dict[str, Any]) -> str:
    if action_type == "propose_memory":
        return memory_proposal_html(payload)
    if action_type == "propose_memory_update":
        return memory_update_proposal_html(payload)
    if action_type == "propose_code_change":
        return code_proposal_html(payload)
    if action_type == "propose_email_draft":
        fields = [
            ("to", payload_value(payload, "to", "recipient", "recipients")),
            ("subject", payload_value(payload, "subject")),
            ("body", payload_value(payload, "body", "message", "draft")),
        ]
    elif action_type == "propose_task":
        fields = [
            ("title", payload.get("title")),
            ("description", payload.get("description")),
            ("due date text", payload.get("due_date_text")),
            ("priority", payload.get("priority")),
        ]
    elif action_type == "propose_calendar_event":
        attendees = payload.get("attendees") or []
        attendees_text = (
            ", ".join(str(item) for item in attendees)
            if isinstance(attendees, list)
            else attendees
        )
        fields = [
            ("title", payload.get("title")),
            ("description", payload.get("description")),
            ("date", payload.get("date_text")),
            ("start time", payload.get("start_time_text")),
            ("end time", payload.get("end_time_text")),
            ("location", payload.get("location")),
            ("attendees", attendees_text),
        ]
    else:
        fields = sorted(payload.items())
    return (
        "<div class='fields'>"
        + "".join(field(label, value) for label, value in fields)
        + "</div>"
    )


def code_proposal_html(payload: dict[str, Any]) -> str:
    relevant_files = payload.get("relevant_files", [])
    if not isinstance(relevant_files, list):
        relevant_files = []
    safety_notes = payload.get("safety_notes", [])
    if not isinstance(safety_notes, list):
        safety_notes = []
    test_commands = payload.get("suggested_test_commands", [])
    if not isinstance(test_commands, list):
        test_commands = []

    fields = [
        ("requested goal", payload.get("requested_goal")),
        ("target surface", payload.get("target_surface")),
        ("relevant files", ", ".join(str(path) for path in relevant_files)),
        ("proposed approach", payload.get("proposed_approach")),
        ("safety notes", "; ".join(str(note) for note in safety_notes)),
        (
            "suggested test commands",
            "; ".join(str(command) for command in test_commands),
        ),
    ]
    patch_preview = payload.get("patch_preview")
    if patch_preview:
        fields.append(
            (
                "patch preview implementation ready",
                yes_no(patch_preview_implementation_ready(payload)),
            )
        )
    warning = patch_preview_display_warning(payload)
    warning_html = f"<p class='warning'>{escape(warning)}</p>" if warning else ""
    preview_html = (
        "<details class='payload-details' open>"
        "<summary>Patch Preview</summary>"
        f"{warning_html}"
        f"{patch_preview_path_classifications_html(payload.get('patch_preview_path_classifications'))}"
        f"<pre class='patch-preview'>{escape(patch_preview)}</pre>"
        "</details>"
        if patch_preview
        else ""
    )
    return (
        "<section class='approval-payload code-review'>"
        "<div class='fields'>"
        + "".join(field(label, value) for label, value in fields)
        + "</div>"
        + preview_html
        + "</section>"
    )


def patch_preview_path_classifications_html(value: object) -> str:
    if not isinstance(value, list):
        return ""
    items = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        classification = item.get("classification")
        if path and classification:
            items.append(f"{path} ({classification})")
    if not items:
        return ""
    return (
        "<div class='fields'>"
        + field("patch preview files", "; ".join(items))
        + "</div>"
    )


def patch_preview_implementation_ready(payload: dict[str, Any]) -> bool:
    value = payload.get("patch_preview_implementation_ready")
    if isinstance(value, bool):
        return value
    return not bool(patch_preview_display_warning(payload))


def patch_preview_display_warning(payload: dict[str, Any]) -> str:
    explicit_warning = payload.get("patch_preview_warning")
    if explicit_warning:
        return str(explicit_warning)
    classifications = payload.get("patch_preview_path_classifications")
    if isinstance(classifications, list):
        items = [item for item in classifications if isinstance(item, dict)]
        return patch_preview_warning(items)
    return ""


def yes_no(value: bool) -> str:
    return "yes" if value else "no"


def memory_proposal_html(payload: dict[str, Any]) -> str:
    tags = payload.get("tags_json", [])
    return (
        "<section class='approval-payload memory-review'>"
        "<div class='memory-review-panel'>"
        "<h4>Proposed Memory</h4>"
        f"<div class='memory-content'>{escape(payload.get('content', ''))}</div>"
        f"{memory_payload_meta(payload.get('memory_type'), payload.get('source'), tags, None)}"
        "</div>"
        "</section>"
    )


def memory_update_proposal_html(payload: dict[str, Any]) -> str:
    original_tags = payload.get("original_tags_json", [])
    replacement_tags = payload.get("replacement_tags_json", [])
    memory_id = f"#{payload.get('memory_id', '')}"
    return (
        "<section class='approval-payload memory-review'>"
        "<div class='memory-review-grid'>"
        "<div class='memory-review-panel original'>"
        "<h4>Current Memory</h4>"
        f"<div class='memory-content'>{escape(payload.get('original_content', ''))}</div>"
        f"{memory_payload_meta(payload.get('original_memory_type'), payload.get('original_source'), original_tags, None)}"
        "</div>"
        "<div class='memory-review-panel replacement'>"
        "<h4>Proposed Replacement</h4>"
        f"<div class='memory-content'>{escape(payload.get('replacement_content', ''))}</div>"
        f"{memory_payload_meta(payload.get('replacement_memory_type'), payload.get('replacement_source'), replacement_tags, payload.get('reason'))}"
        "</div>"
        "</div>"
        f"{compact_detail('Memory', memory_id, 'memory-update-id')}"
        "</section>"
    )


def memory_payload_meta(
    memory_type: object,
    source: object,
    tags: object,
    reason: object,
) -> str:
    tag_values = tags if isinstance(tags, list) else []
    details = [
        compact_detail("Type", memory_type),
        compact_detail("Source", source),
    ]
    if tag_values:
        details.append(
            compact_detail("Tags", ", ".join(str(tag) for tag in tag_values))
        )
    if reason:
        details.append(compact_detail("Reason", reason))
    return "<div class='memory-details'>" + "".join(details) + "</div>"


def tag_list_html(tags: list[str]) -> str:
    if not tags:
        return ""
    return (
        "<div class='tag-list'>"
        + "".join(f"<span>{escape(tag)}</span>" for tag in tags)
        + "</div>"
    )


def compact_detail(label: object, value: object, class_name: str = "") -> str:
    value_text = "" if value is None else str(value)
    extra_class = f" {escape(class_name)}" if class_name else ""
    return (
        f"<div class='compact-detail{extra_class}'>"
        f"<span>{escape(label)}</span>"
        f"<strong>{escape(value_text)}</strong>"
        "</div>"
    )


def labeled_control(label: str, control_html: str) -> str:
    return f"<label class='control'><span>{escape(label)}</span>{control_html}</label>"


def field(label: object, value: object) -> str:
    value_text = "" if value is None else str(value)
    return (
        "<div class='field'>"
        f"<b>{escape(label)}</b>"
        f"<span>{escape(value_text)}</span>"
        "</div>"
    )


def form_button(action: ActionButton) -> str:
    class_attr = f" class='{action.class_name}'" if action.class_name else ""
    return f"<form method='post' action='{action.path}'><button{class_attr} type='submit'>{escape(action.label)}</button></form>"


def chat_bubble(role: str, content: str, memory_context_count: int = 0) -> str:
    role_class = "user" if role == "user" else "assistant"
    role_label = "You" if role_class == "user" else "Lewis"
    note = (
        chat_memory_context_note(memory_context_count)
        if role_class == "assistant"
        else ""
    )
    return (
        f"<div class='bubble {role_class}'>"
        f"<span class='bubble-role'>{role_label}</span>"
        f"{note}"
        f"<span class='bubble-text'>{escape(content)}</span>"
        "</div>"
    )


def chat_memory_context_note(memory_context_count: int) -> str:
    if memory_context_count <= 0:
        return ""
    label = "memory" if memory_context_count == 1 else "memories"
    return (
        "<div class='memory-context-note'>"
        f"Using {escape(memory_context_count)} approved {label} for context."
        "</div>"
    )


def empty_state_html(message: str) -> str:
    return f"<section class='empty-state'><p>{escape(message)}</p></section>"


def status_badge(status: object) -> str:
    value = str(status)
    class_name = "status-" + value.replace("_", "-")
    return f"<span class='badge {escape(class_name)}'>{escape(value)}</span>"


def type_badge(action_type: object) -> str:
    return f"<span class='badge badge-soft'>{escape(action_type)}</span>"


def flash(message: str) -> str:
    return f"<div class='flash' role='status'>{escape(message)}</div>" if message else ""


def payload_value(payload: dict[str, object], *keys: str) -> object | None:
    for key in keys:
        value = payload.get(key)
        if value:
            return value
    for value in payload.values():
        if isinstance(value, dict):
            nested = payload_value(value, *keys)
            if nested:
                return nested
    return None


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)
