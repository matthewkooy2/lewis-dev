# Architecture and tradeoffs

Lewis is a single-process local application. The CLI and server-rendered FastAPI
UI share Python services; SQLite stores assistant records and Projects data.
Freestyle stores job artifacts on disk beside disposable Git worktrees.

## Code tracing

`app/trace/service.py` builds an index through `workspace.py` and
`project_index.py`. The workspace filters paths and bounds reads. Python uses
`ast`; JavaScript/TypeScript uses a custom lexer and structural analyzer.
`engine.py` resolves symbols and traverses relationships; `review.py` adds
heuristic findings and `render.py` produces source-cited reports.

The common model separates language parsing from graph traversal. Relationship
labels distinguish confirmed, inferred, ambiguous, dynamic, and unresolved
results. These are static observations, not proof of runtime behavior. A custom
JS/TS parser keeps dependencies small but creates maintenance and language-coverage
costs; it is deliberately not described as a full compiler frontend.

## Supervised coding

The Freestyle entry points live in `app/dev/freestyle_coding.py`; the bounded
orchestration loop is in `freestyle_loop.py`.

```text
approved goal → filtered snapshot + worktree → plan and critique
             → structured edit bundle → exact-content validation
             → apply in worktree → approved checks → review → human handoff
```

Failures are recorded, and bounded repairs may retry selected stages. Human
handoff is a workflow result, not permission to merge. Exact-match anchors and
hashes reject stale edits. Tests verify that the source checkout remains intact.

A worktree provides change separation, not OS isolation. Pytest runs repository
code as the current user. Output artifact limits do not cap all subprocess
memory use. Run against trusted repositories and review generated tests.

The job model and loop currently carry too many stage-specific fields and
cross-module private imports. Simplification should reduce stored/derived state
and duplicated orchestration while preserving recovery and existing job files.
Splitting large files without reducing those dependencies would not solve this.

## Optional surfaces

- `app/llm.py` handles personal-assistant chat and bounded tool calls. Its narrow,
  deterministic proposal engine produces review-only diffs for selected intents.
  The unreachable legacy proposal fallback has been removed.
- `app/projects.py` stores repository maps and notes. It reuses symbol extraction
  but also maintains separate relationship/indexing logic from the tracer.
- `app/dev/coding_agents.py` wraps installed provider CLIs with fixed arguments,
  restricted tools, event normalization, cancellation, and timeouts. Its
  restrictions depend on the installed provider; no live-provider compatibility
  benchmark is implied by the scripted tests.
- `app/web.py` installs loopback host and same-origin write checks. Shared chat
  turns run in a worker thread under a per-app lock. The app is single-user;
  request cancellation does not interrupt an already running model call.

## Deliberate limits

SQLite, server-rendered HTML, and keyword document search fit local usage. No
queue service, distributed database, frontend framework, or vector database is
required for the demonstrated workflow. Live model evaluations, a unified
repository index, and a smaller Freestyle state contract remain future work.
