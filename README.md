# Lewis

Lewis is a terminal-based MVP for a supervised personal assistant. It uses
DeepSeek through the OpenAI Python SDK, searches local documents, and saves any
real-world action as a pending approval item instead of executing it.

## Setup

Install [Git](https://git-scm.com/downloads) and uv 0.12.11 using the
[official versioned installer](https://docs.astral.sh/uv/getting-started/installation/).

macOS/Linux:

```sh
curl -LsSf https://astral.sh/uv/0.12.11/install.sh | sh
```

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/0.12.11/install.ps1 | iex"
```

Reopen your terminal after installation so `uv` is on `PATH`. From the cloned
repository, run these commands on any platform:

```sh
uv python install
uv sync --locked
```

The repository pins Python 3.11.15 in `.python-version`, uv in `pyproject.toml`,
and direct/transitive dependencies in `uv.lock`. `uv sync --locked` creates
`.venv` with the locked development dependencies and refuses an outdated lockfile.
No environment activation is needed with `uv run --locked`.

Try the local dashboard without an API key:

```sh
uv run --locked python -m app.main web
```

Open [the local dashboard](http://127.0.0.1:8787). Chat requires a configured
model; dashboard pages, tests, document search, and the static tracer do not.

For chat, copy `.env.example` to `.env` (`cp .env.example .env` on macOS/Linux,
or `Copy-Item .env.example .env` in PowerShell) and set `DEEPSEEK_API_KEY`.
The default model is `deepseek-v4-flash`; change `DEEPSEEK_MODEL` as needed.
`.env` and the local SQLite database are ignored by Git.

### Existing pip-based environments

`requirements.txt` is a generated, hash-pinned export of the same lockfile,
including development tools. If you prefer pip, use Python 3.11.15:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install --require-hashes -r requirements.txt
```

On Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements.txt
```

With pip, replace `uv run --locked python` below with `.venv/bin/python`
(macOS/Linux) or `.\.venv\Scripts\python.exe` (Windows).
Use a fresh virtual environment when changing dependency sets; pip does not
remove unrelated packages already installed there.

## Run

Start chat:

```sh
uv run --locked python -m app.main chat
```

Start the local web dashboard:

```sh
uv run --locked python -m app.main web
```

Open:

```text
http://127.0.0.1:8787
```

The web dashboard is a local browser command center for dashboard counts,
approvals, task proposals, calendar proposals, recent usage logs, secure local
document search, and a simple Lewis chat page. It binds only to `127.0.0.1`, so
it is not exposed to your LAN. It does not print API keys, display `.env`
contents, add login, or add any new external integrations.

List approval items:

```sh
uv run --locked python -m app.main approvals
```

Approve or reject an item:

```sh
uv run --locked python -m app.main approve <approval_id>
uv run --locked python -m app.main reject <approval_id>
```

Preview execution for an approved item:

```sh
uv run --locked python -m app.main execute-approved <approval_id>
```

Summarize recent API usage:

```sh
uv run --locked python -m app.main usage
```

Search local documents directly:

```sh
uv run --locked python -m app.main search-docs "your query here"
```

## Development checks

Run the same checks as CI; no `.env`, API key, or running model is required:

```sh
uv sync --locked
uv run --locked python -m ruff check .
uv run --locked python -m ruff format --check .
uv run --locked python -m pytest -q -ra
uv run --locked python -m app.eval_runner
uv run --locked python -m app.main --help
```

The eval runner contains four deterministic component smoke checks. It does
not measure live model accuracy. The full test suite includes real temporary
Git repositories and subprocess checks; run it against trusted source code.
PowerShell and Windows-junction checks skip when their platform is unavailable.

[CI](.github/workflows/ci.yml) runs on Linux, macOS, and Windows for pushes
and pull requests, and checks that the pip export matches the lockfile.
Use `uv run --locked python -m ruff format .` to apply formatting.
See [CONTRIBUTING.md](CONTRIBUTING.md) for dependency updates and validation.

## Safety Model

Lewis may use read-only tools such as document search, file listing, and current
time lookup. It must not directly send emails, create calendar events, delete or
edit files, submit forms, make purchases, or message people.

Developer/code workspace repository inspection is off by default. Set
`LEWIS_DEVELOPER_CODE_WORKSPACE=true` only for local development if you want
Lewis to list, read, and keyword-search safe text files for code-planning
work. This mode is read-only, separate from document search, bounded,
project-relative, secret-redacted, and blocks hidden, env, runtime, database,
binary, cache, virtualenv, and out-of-workspace paths.

By default, developer code tools (chat tools and the CLI) analyze the Lewis
repository itself. To point them at a different project, set
`LEWIS_DEVELOPER_WORKSPACE_ROOT=/path/to/project`; the path must exist and be
a directory, or Lewis reports a clear error instead of silently falling back
to its own repository. `LEWIS_DEVELOPER_CODE_WORKSPACE` only gates whether
the chat tools are registered — it does not choose which repository they
analyze, and disabling it does not change `LEWIS_DEVELOPER_WORKSPACE_ROOT`.

Developer mode also enables the `trace_symbol` chat tool, and the CLI always
provides `python -m app.main trace <symbol> [--mode quick|deep|review]
[--file PATH] [--json]`: a static, read-only trace of one Python or
JavaScript/TypeScript symbol showing its definition, callers, callees, side
effects, tests, error handling, and (in review mode) deterministic findings
with file:line citations. It never executes project code and labels every
relationship as confirmed, inferred, framework-registered, ambiguous,
external, dynamic, or unresolved. See `docs/code-tracing.md` for usage,
supported languages, bounds, and static-analysis limitations.

Any action with external consequences is saved to the SQLite approval queue as
`pending`. Approving an item only marks it as approved. It does not execute the
action.

Approved items can be passed to `execute-approved`, but this currently performs
only local dry-run execution. The command changes the item to
`execution_previewed` and prints what a future integration would do. It does not
send email, create calendar events, create reminders, create files, notify
anyone, or touch external systems.

The execution layer is intentionally local-only preparation for future Outlook
email draft and calendar handlers. No Outlook, Gmail, Google Calendar, browser
automation, or other external integration is included yet.

## Local Web Dashboard

Run:

```sh
uv run --locked python -m app.main web
```

Then open `http://127.0.0.1:8787`.

The dashboard uses one shared, in-memory conversation across browser tabs.
Model calls run in a worker thread, and complete chat turns are serialized.
Other pages remain available during a model call; the chat history shows each
turn once its reply is ready. Cancelling an HTTP request does not stop a model
call that has already started: that turn finishes and is recorded before the
next turn can use the agent.

The dashboard can:

- Show local counts for pending drafts, tasks, calendar proposals, approved
  items, completed tasks, rejected items, and recent items.
- Review approval items and mark them approved or rejected.
- Preview approved items with local dry-run execution only.
- List and complete or reopen local task proposals.
- List local calendar proposals.
- Show recent usage logs, token counts, estimated cost, and API calls.
- Search only the allowlisted `data/documents/` folder.
- Provide a simple browser chat page using the existing Lewis behavior when an
  API key is configured.

The dashboard cannot:

- Expose Lewis to the LAN or bind outside `127.0.0.1`.
- Send email, create drafts in Outlook or Gmail, create calendar events, send
  invites, notify attendees, create reminders, or execute real external actions.
- Search Desktop, Downloads, Documents, OneDrive, AppData, `.ssh`, `C:\`, or
  arbitrary paths.
- Display API keys or `.env` contents.
- Add browser automation, login, React build tooling, Outlook, Gmail, or Google
  Calendar.

## Local Documents

Put documents you want Lewis to search in `data/documents/`. Supported files are
`.txt`, `.md`, and `.pdf`. The folder is intentionally manual: you choose what to
place there, and Lewis only reads from that allowlisted location.

Lewis cannot search your whole computer, user profile, Desktop, Downloads,
Documents, OneDrive, AppData, browser folders, `.ssh`, or any directory outside
the project. It also skips hidden files and folders, `.env` files, `.git`,
`__pycache__`, system files, key files, credential files, and anything that looks
like secrets.

This limit is deliberate. It keeps local search read-only, predictable, and less
likely to expose private files by accident. Lewis resolves every searched file
and verifies it is still under `data/documents/` before reading it. User-provided
paths are not accepted by the document search tool.

Direct search:

```sh
uv run --locked python -m app.main search-docs "your query here"
```

Chat examples:

```text
Search my documents for the onboarding notes.
What do my PDFs say about Project Alpha?
Look in my local notes for budget review details.
```

Lewis uses a simple keyword search with snippets and match counts. PDF
extraction uses `pypdf`; if a PDF cannot be parsed, it is skipped. This first
version does not use a vector database or semantic search.

## Budgeting and Logs

Lewis logs each run to `data/lewis.sqlite3`, including user input, model, tool
calls, token usage, and estimated cost. Configure budgets and pricing in `.env`:

- `LEWIS_MAX_TOOL_CALLS`
- `LEWIS_MAX_DAILY_COST_USD`
- `DEEPSEEK_PRICE_PER_1K_INPUT`
- `DEEPSEEK_PRICE_PER_1K_OUTPUT`

Set `LEWIS_DEBUG=true` to print a concise token summary after each model
request. `LEWIS_DEBUG` also enables the Freestyle web UI (see below).

## Freestyle Web UI

The Freestyle command center (`/dev/freestyle`, linked from the dashboard's
"Developer" navigation group) is a developer-only local UI for inspecting and
lightly operating approved, isolated-worktree coding jobs. It is disabled by
default. Enable it with either flag in `.env`:

- `LEWIS_FREESTYLE_UI=true` &mdash; enables only the Freestyle routes and nav
  link. Prefer this if you do not also want `LEWIS_DEBUG`'s unrelated token
  logging.
- `LEWIS_DEBUG=true` &mdash; enables Freestyle plus the broader debug output
  above.

Every mutating action in the UI (creating a job, applying a generated bundle,
running checks, refining, cleaning up, recovering stale state) still requires
its own explicit form confirmation; enabling the UI only makes the page
reachable, it does not skip any confirmation. See `LEWIS_FREESTYLE_PROVIDER`
and the surrounding `LEWIS_LOCAL_*` variables in `.env.example` to choose
which model (remote DeepSeek or a local OpenAI-compatible endpoint such as
LM Studio) backs Freestyle job generation; that is independent of whether the
UI itself is visible.

## Coding Agents Web UI

The Coding Agents command center (`/dev/coding-agents`, linked from the
Developer navigation group) is disabled by default. Enable it with either flag
in `.env`:

- `LEWIS_CODING_AGENTS_UI=true` - enables only the Coding Agents routes and nav
  link.
- `LEWIS_DEBUG=true` - enables Coding Agents plus the broader debug output.

Phase 1 is intentionally read-only: "Read-only analysis and proposal jobs. File
modifications and Git integration are disabled." Jobs may run only inside roots
listed in `LEWIS_CODING_AGENT_REPOSITORIES`, separated with the OS path
separator (`;` on Windows). Lewis resolves and validates the selected root at
launch, requires it to be the Git top level, rejects nested repositories and
symlink escapes, and never auto-discovers repositories or accepts browser-supplied
paths.

Provider CLIs use their existing local authentication. Lewis never stores or
passes provider secrets, environment variables from the browser, arbitrary shell
commands, arbitrary argv, executable paths, package-manager commands, Git write
operations, commits, pushes, merges, branch changes, or worktree creation.
Opening the page does not launch a provider process; provider authentication
status is refreshed only through an explicit status-refresh action.

The fixed Phase 1 invocations are:

```text
codex exec --json --sandbox read-only --color never --cd <approved-repo> [-m <model>] -
claude -p --output-format stream-json --verbose --permission-mode plan --tools Read,Grep,Glob,LS --disallowedTools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,TodoWrite --safe-mode --disable-slash-commands --no-chrome [--model <model>] [--effort <level>]
```

Optional local configuration:

- `LEWIS_CODEX_CLI_PATH` and `LEWIS_CLAUDE_CLI_PATH` can pin executable
  locations. Leave them empty for safe PATH lookup.
- `LEWIS_CODING_AGENT_DEFAULT_TIMEOUT_SECONDS` and
  `LEWIS_CODING_AGENT_MAX_TIMEOUT_SECONDS` set server-side runtime bounds.

## Projects Workspace

The Projects workspace (`/projects`) is disabled by default. Enable it with:

```text
LEWIS_PROJECTS_UI=true
```

Projects store project metadata, notes, repository map runs, indexed files,
symbols, relationships, summaries, diagrams, and activity in the Lewis SQLite
database. They do not write notes or generated map data into source
repositories. Opening `/projects` only reads the registry; repository mapping
starts only from an explicit refresh action.

Project repositories must still be approved. By default Projects reuse
`LEWIS_CODING_AGENT_REPOSITORIES`; set `LEWIS_PROJECT_REPOSITORIES` to use a
Projects-specific allowlist. Paths are canonicalized, must be Git top-level
repositories, and are checked for symlink escapes and unexpected nested Git
repositories. Mapping is read-only and uses fixed Git metadata commands plus
static parsing only. It does not install packages, run repository code, modify
files, commit, push, pull, merge, or switch branches.

Useful optional limits:

- `LEWIS_PROJECT_MAP_MAX_FILES`
- `LEWIS_PROJECT_MAP_MAX_FILE_BYTES`
- `LEWIS_PROJECT_MAP_TIMEOUT_SECONDS`
- `LEWIS_PROJECT_MAP_MAX_MODEL_CALLS`
- `LEWIS_PROJECT_MAP_MODEL_SUMMARIES`
- `LEWIS_PROJECT_MAP_IGNORED_DIRECTORIES`

After changing these settings on the Windows home PC, restart Lewis with
`scripts\windows\request-lewis-restart.ps1` or stop and start the local web
process.

## Current Limitations

- No browser automation.
- No Gmail, calendar, messaging, purchase, or file-edit execution.
- Approvals only change status; they do not trigger external actions.
- `execute-approved` is dry-run/local preview only; it does not execute real
  actions.
- The web dashboard is local-only and has no login yet.
- Document search is a simple local keyword search, not a persistent vector index.
- Document search only reads allowlisted files under `data/documents/`.
