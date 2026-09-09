# Lewis

A local coding-workflow prototype built with Python, FastAPI, and SQLite.
Lewis traces Python and JavaScript/TypeScript code, stages proposed edits in
separate Git worktrees, and records checks and review results for human handoff.

The focus is inspectable changes: source citations, explicit uncertainty,
exact-content edit validation, and an unchanged original checkout. This is a
single-user prototype, with deterministic integration tests rather than a claim
of production readiness or measured model accuracy.

## See it work

[Five-minute walkthrough](docs/showcase.md) · [Architecture](docs/architecture.md) ·
[Usage reference](docs/usage.md) · [Contributing](CONTRIBUTING.md)

- **Static tracing:** find callers, callees, side effects, and related tests,
  with source locations and confidence labels. Runs without a model or API key.
- **Supervised edits (Freestyle):** validate structured edits, run approved
  checks in a worktree, and stop for human review. Model responses can come
  from DeepSeek or a local OpenAI-compatible server.
- **Optional tools:** Projects provides repository maps and notes; Coding
  Agents wraps installed Codex/Claude CLIs for restricted analysis. The
  personal-assistant dashboard retains local approvals, memories, and search.

## Quick start

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

## Engineering checks

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q -ra
uv run --locked python -m app.eval_runner
```

[CI](.github/workflows/ci.yml) checks Linux, macOS, and Windows. PRs test the
combined branch; pushes to `main` and the active integration branch verify
landed changes. Tests use scripted model responses and real temporary Git
repositories. The eval runner contains component smoke checks, not model-quality
benchmarks. See [Contributing](CONTRIBUTING.md) for dependency and CI details.

## Boundaries and current limitations

- Run the dashboard on loopback, on a trusted local machine. It has host and
  browser-origin checks, but no user authentication or multi-user isolation.
- A Git worktree is **not a security sandbox**. Checks execute repository code
  with the current user's permissions. Only run them against trusted code.
- Static analysis is bounded and cannot resolve all dynamic behavior. Its
  JavaScript/TypeScript parser is purpose-built, not a full compiler frontend.
- The model workflow needs live evaluation before any reliability or productivity
  claims. The narrow deterministic chat proposal strategies remain experimental.
- Email/calendar actions are local proposals and previews; there are no email
  or calendar service integrations.

Detailed operating limits are in the [usage reference](docs/usage.md#safety-model)
and [tracing guide](docs/code-tracing.md). This repository does not currently
include an open-source license.
