# Lewis

Lewis is a terminal-based MVP for a supervised personal assistant. It uses
DeepSeek through the OpenAI Python SDK, searches local documents, and saves any
real-world action as a pending approval item instead of executing it.

## Setup

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and set `DEEPSEEK_API_KEY`. The default model is
`deepseek-v4-flash`, but you can change `DEEPSEEK_MODEL`.

## Run

Start chat:

```powershell
.\.venv\Scripts\python.exe -m app.main chat
```

Start the local web dashboard:

```powershell
.\.venv\Scripts\python.exe -m app.main web
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

```powershell
.\.venv\Scripts\python.exe -m app.main approvals
```

Approve or reject an item:

```powershell
.\.venv\Scripts\python.exe -m app.main approve <approval_id>
.\.venv\Scripts\python.exe -m app.main reject <approval_id>
```

Preview execution for an approved item:

```powershell
.\.venv\Scripts\python.exe -m app.main execute-approved <approval_id>
```

Summarize recent API usage:

```powershell
.\.venv\Scripts\python.exe -m app.main usage
```

Search local documents directly:

```powershell
.\.venv\Scripts\python.exe -m app.main search-docs "your query here"
```

Run evals:

```powershell
.\.venv\Scripts\python.exe -m app.eval_runner
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run lint:

```powershell
.\.venv\Scripts\python.exe -m ruff check .
```

Format code:

```powershell
.\.venv\Scripts\python.exe -m ruff format .
```

## Safety Model

Lewis may use read-only tools such as document search, file listing, and current
time lookup. It must not directly send emails, create calendar events, delete or
edit files, submit forms, make purchases, or message people.

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

```powershell
.\.venv\Scripts\python.exe -m app.main web
```

Then open `http://127.0.0.1:8787`.

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

```powershell
.\.venv\Scripts\python.exe -m app.main search-docs "your query here"
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
request.

## Current Limitations

- No browser automation.
- No Gmail, calendar, messaging, purchase, or file-edit execution.
- Approvals only change status; they do not trigger external actions.
- `execute-approved` is dry-run/local preview only; it does not execute real
  actions.
- The web dashboard is local-only and has no login yet.
- Document search is a simple local keyword search, not a persistent vector index.
- Document search only reads allowlisted files under `data/documents/`.
