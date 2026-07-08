# Coding Agents Phase 1

Coding Agents are developer-only, local web jobs for read-only analysis and proposals using either the locally authenticated Codex CLI or Claude Code CLI.

## Enablement

Set `LEWIS_CODING_AGENTS_UI=true` to enable `/dev/coding-agents`. `LEWIS_DEBUG=true` also enables it, but the dedicated flag is preferred when you only want this page.

Configure approved repositories explicitly:

```text
LEWIS_CODING_AGENT_REPOSITORIES=C:\path\to\repo1;C:\path\to\repo2
```

Lewis does not discover repositories automatically and the browser never supplies filesystem paths. Each selected repository is resolved at launch and must be the Git top level. Launch validation rejects non-Git directories, traversal by unknown ids, nested Git repositories, and symlinks that escape the approved root.

Optional executable pins:

```text
LEWIS_CODEX_CLI_PATH=C:\Users\Example\AppData\Local\Programs\OpenAI\Codex\bin\codex.exe
LEWIS_CLAUDE_CLI_PATH=C:\Users\Example\.local\bin\claude.exe
```

Leave these empty to use safe PATH lookup.

## Fixed Invocations

Codex jobs use:

```text
codex exec --json --sandbox read-only --color never --cd <approved-repo> [-m <model>] -
```

Claude jobs use:

```text
claude -p --output-format stream-json --verbose --permission-mode plan --tools Read,Grep,Glob,LS --disallowedTools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,TodoWrite --safe-mode --disable-slash-commands --no-chrome [--model <model>] [--effort <level>]
```

The prompt is sent on stdin. Browser input is limited to provider, approved repository id, goal, model, Claude effort, max turns, timeout, and explicit confirmation.

## Boundaries

Phase 1 is labeled in the UI as: "Read-only analysis and proposal jobs. File modifications and Git integration are disabled."

The server refuses write-capable modes and unsafe arguments. It does not accept shell commands, arbitrary argv, executable paths, working directories, environment variables, package-manager actions, Git commits, pushes, pulls, fetches, merges, rebases, branch switches, worktree creation, PR creation, Tailscale changes, or LM Studio changes from the browser.

Jobs run with a sanitized environment and existing provider CLI authentication. Lewis does not store credentials, hidden reasoning, full environment variables, or unbounded terminal output.

Provider account usage is shown only when a documented local interface exposes it reliably. In Phase 1, Codex account rate-limit/usage is marked unavailable pending a future app-server integration, and Claude subscription-window usage is marked unavailable rather than scraped from terminal UI text.
