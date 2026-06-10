# Stage 8A - Freestyle Coding Lane Design

## Goals and non-goals

The future freestyle coding lane gives a developer an explicitly invoked,
local-only way to ask DeepSeek to inspect a committed Lewis codebase, propose
and make narrowly mediated edits in an isolated worktree, run fixed allowlisted
checks, and produce a reviewable diff and risk report. It is intended for goals
that do not fit the existing known-strategy code proposal lane.

The lane must preserve human control. A human chooses whether to invoke it,
reviews every resulting change, and makes every merge decision. It must never
run from normal chat, mutate approvals or SQLite, use remote services other
than the DeepSeek API, or merge automatically.

Stage 8A is design-only. It does not implement the freestyle agent, add tools,
edit code, run checks, create branches or worktrees, or change runtime
behavior.

## Difference from known-strategy code proposals

The existing known-strategy proposal lane is a normal-chat feature that maps a
supported request to a conservative, grounded strategy and stores an inert
unified diff in a pending `propose_code_change` approval. Its validators and
applicators remain authoritative for that lane and must not be weakened,
bypassed, or broadened to accommodate freestyle work.

The freestyle lane is separate developer tooling for goals outside that fixed
strategy catalog. It is authorized only by direct developer CLI invocation,
works only in a dedicated sibling worktree, may eventually edit existing safe
text files through mediated tools, and produces a local report rather than an
approval. Freestyle results must never be represented as known-strategy
proposals or treated as approval-backed changes.

## Reuse of existing approval/applicator/validation ideas

The freestyle lane should reuse safety ideas, not the approval workflow or its
authority:

- Reuse conservative path validation, symlink refusal, size limits, changed-path
  verification, diff inspection, fixed subprocess arguments, timeouts, and
  fail-closed behavior.
- Reuse the principle that validated metadata is evidence from one validation
  pass, not a standing capability; revalidate before every read, write, check,
  and report operation.
- Preserve the existing DSML/XML/function-call markup suppression and private
  DeepSeek `reasoning_content` continuation handling.
- Keep known-strategy validators and applicators unchanged and independent.
- Do not create, update, complete, or otherwise mutate `ApprovalStore` records.
- Do not construct an `ApprovalStore` if doing so can initialize or mutate
  SQLite. The freestyle lane needs no approval database access.

## Developer-only entrypoint

The future entrypoint should be:

```text
python -m app.dev.freestyle_coding --goal "<goal>"
```

All freestyle modules must live only under `app/dev/`. Explicit developer CLI
invocation is the authorization boundary. Possession of a goal string,
approval, chat message, HTTP request, task, or model tool call is not
authorization.

Normal chat, runtime modules, web routes, tools, executors, and `app/main.py`
must not import or invoke freestyle modules. The freestyle entrypoint must not
be registered as a normal chat tool or exposed through an application route.

## Branch vs worktree recommendation

Use both a dedicated local branch and a sibling Git worktree created from the
committed `HEAD` of the active checkout. A branch alone would still expose the
active checkout to checkout operations and accidental edits; a sibling
worktree provides a separate filesystem boundary while the dedicated branch
provides a reviewable local history target.

Before creating the worktree, capture the active checkout's committed `HEAD`
without changing its branch or index. The active checkout may be dirty, but the
lane must never alter, stash, overwrite, clean, reset, include, or modify those
dirty files. The sibling worktree must be created from committed `HEAD`, so
uncommitted active-checkout content is excluded by construction.

The branch and sibling worktree must use host-generated, collision-checked
names and paths. The model must not choose Git arguments, branch names, or
worktree locations. All work remains local: no remotes, fetches, pushes, pull
requests, or automatic merge.

## DeepSeek API loop

The loop must use the DeepSeek API only. It must require both an available
DeepSeek API credential and an explicit non-empty `DEEPSEEK_CODE_MODEL`, and
fail closed before creating a worktree or calling the API if configuration is
missing or invalid.

The existing OpenAI-compatible SDK may be reused only as transport for the
DeepSeek endpoint. Do not add OpenAI, Anthropic, local-model, fallback-provider,
or broader provider abstractions. Do not use the application's `.env`-loading
settings path. Read DeepSeek credentials and configuration only from the
already-populated process environment, and never place secrets or API keys in
prompts, tool results, reports, exceptions, or logs.

The loop must have:

- Fixed, versioned system instructions that define the lane and its limits.
- Hard host-enforced maximums for API calls and tool calls.
- A host-owned registry of allowed tool names and schemas.
- Exact registered-tool filtering and refusal of every unregistered tool.
- Host validation of every tool argument before execution.
- A fail-closed stop on malformed calls, repeated refusals, limit exhaustion,
  API failure, or inconsistent state.
- `reasoning_content` retained only in private API continuation messages when
  required by DeepSeek, never displayed or persisted.
- Existing DSML/XML/function-call markup suppression applied to all displayed
  and persisted model text.

Tool results should be bounded and structured. The model must never receive a
general execution primitive or an authority-bearing object.

## Allowed model-facing tools

Only these narrow mediated tools are proposed for a later editing-capable
stage:

| Tool | Contract |
| --- | --- |
| `list_project_files` | List bounded safe paths inside the sibling worktree. |
| `search_project_files` | Search bounded safe text files with host-controlled limits. |
| `read_project_file` | Read a bounded existing safe text file. |
| `replace_text_in_file` | Replace one exact-match text occurrence in an existing safe text file, subject to path and size limits. |
| `inspect_worktree_diff` | Return a bounded, sanitized diff and changed-path summary. |
| `run_allowlisted_check` | Accept an enum only and run one fixed host-defined check. |

`replace_text_in_file` must be exact-match only. It must refuse zero matches,
multiple matches, stale expected text, oversized input or output, unsafe paths,
and any operation that would create, delete, rename, copy, empty, or change the
mode of a file.

The first implementation milestone must expose only
`list_project_files`, `search_project_files`, and `read_project_file`.

## Forbidden tools and actions

Explicitly forbidden:

- Arbitrary shell or unrestricted subprocess execution.
- Model-supplied command names, arguments, environment, working directories,
  scripts, pipes, redirects, or shell syntax.
- Normal chat or runtime execution.
- GitHub, remotes, fetch, pull, push, or pull-request creation.
- Browser automation, email, calendar, or other external actions.
- `.env`, credential, token, key, or secret reads or writes.
- SQLite mutation, migration, initialization, schema changes, or data changes.
- Automatic merge, rebase, cherry-pick, or changes to the active checkout.
- `ApprovalStore` redesign, creation, or mutation.
- Unrestricted file writes.
- Initial file creation, deletion, rename, copy, or mode changes.
- Symlinks, submodules, binary files, or oversized files.
- Hidden paths, `.git`, virtual environments, caches, `data/`, database files,
  or paths outside the sibling worktree.

## File path policy

Every model-supplied path must be treated as untrusted input and pass a
host-owned validator immediately before use. Accept only normalized,
project-relative, unambiguous safe text-file paths inside the sibling
worktree. Resolve and verify both the target and every parent component without
following symlinks outside the worktree.

Refuse absolute paths, parent traversal, alternate separators, hidden path
components, reserved names, streams, control characters, ambiguous Unicode
aliases, symlinks, submodules, binary files, oversized files, and paths outside
the worktree. Refuse `.git`, virtual environments, caches, generated runtime
data, `data/`, databases, `.env` files, and known secret-bearing paths.

Reads, searches, replacements, diffs, and changed-path verification must each
apply the policy independently. A path accepted by an earlier operation is not
a reusable capability.

## Secret, runtime-data, and SQLite policy

The process may read only the DeepSeek configuration already present in its
environment. Secret values must be used only to configure the SDK transport
and must never enter model messages, tools, audit records, reports, command
output, or exception text. Environment enumeration and `.env` loading are
forbidden.

Runtime data and SQLite are outside the lane. Refuse database files, journals,
WAL files, `data/`, runtime logs, caches, and similar mutable application state.
Do not initialize, connect to, query, migrate, or mutate SQLite. Do not create
or mutate `ApprovalStore`; freestyle work has no approval-side effects.

## Allowlisted checks

A later editing-capable stage may support `run_allowlisted_check` with an enum
mapping to exactly:

```text
ruff -> .\.venv\Scripts\python.exe -m ruff check .
pytest -> .\.venv\Scripts\python.exe -m pytest -q
```

The model supplies only the enum value. The host owns every executable,
argument, working directory, environment restriction, timeout, output limit,
and termination rule. Every subprocess must use a fixed argument array,
`shell=False`, a timeout, captured bounded output, and an allowlisted sibling
worktree working directory. No command text from the model, goal, repository,
or report may be executed.

Checks must run only after changed-path verification succeeds. Verify changed
paths again after each check because tests and linters can create files. Refuse
and stop if checks alter or create any non-allowlisted path. The read-only
first milestone runs no checks.

## Failure recovery

Fail closed on invalid configuration, API errors, malformed or unregistered
tool calls, exhausted limits, path refusal, stale exact-match replacement,
unexpected changed paths, subprocess timeout, failed checks, secret-redaction
failure, or inconsistent Git state.

Failure handling must never modify the active checkout or automatically merge,
stash, reset, clean, delete, or rewrite work. Stop model activity, capture a
redacted failure report, and leave the dedicated branch and sibling worktree
for human inspection unless a narrowly defined cleanup policy is later
approved. Reports should include safe manual cleanup guidance without
executing it.

## Diff and risk reporting

The final report must be generated from host-observed state, not solely from
model claims. It should include the goal, base commit, local branch, sibling
worktree path, bounded API/tool-call counts, tools used, checks requested and
results, changed paths, diff statistics, a sanitized bounded diff, refusals,
failures, and unresolved risks.

Before reporting success, verify that only allowed existing safe text files
changed and that there were no creations, deletions, renames, copies, mode
changes, symlinks, submodules, binary changes, secret/runtime-data access, or
unexpected check artifacts. Clearly distinguish model-provided risk analysis
from host-verified facts.

## Human review and merge workflow

The lane ends with a local branch, sibling worktree, and report. A human must
inspect the diff, evaluate risks and check results, decide whether further
manual testing is needed, and choose whether and how to commit or merge.

The lane must not merge, commit to the active branch, push, create a pull
request, mutate an approval, or mark work complete on the human's behalf.
Human rejection or abandonment leaves cleanup as an explicit manual action.

## Audit logging

Audit logs should be local, bounded, structured, and redacted. Record
timestamps, the sanitized goal, base commit, generated branch/worktree
identifiers, API/tool-call counts, registered tool names used, host validation
decisions, changed paths, check enum selections and outcomes, failures, and
final disposition.

Never persist prompts or tool results wholesale when they may contain source
content or secrets. Never persist `reasoning_content`, API keys, environment
values, authorization headers, raw SDK request/response objects, or
unsanitized subprocess output. Redaction failure is a fatal error that stops
the lane before writing the affected audit event.

## Preventing chat/runtime imports and shell execution

Keep all freestyle implementation under `app/dev/` and give it no registration
hook in normal chat, tools, web routes, executors, runtime startup, or
`app/main.py`. Add architecture tests before implementation that scan normal
runtime imports and prove they cannot import or invoke freestyle modules.

Centralize subprocess execution behind a small private host-controlled adapter
that accepts only internal enums or fixed Git operations, always uses fixed
argument arrays and `shell=False`, enforces timeouts and output bounds, and
refuses arbitrary arguments. Model-facing tools must never expose that adapter
or a generic Git/subprocess wrapper.

## Minimal first implementation milestone

The first implementation milestone must be read-only:

1. Validate DeepSeek-only process-environment configuration, including required
   `DEEPSEEK_CODE_MODEL`, and fail closed before side effects.
2. Create a dedicated local branch and sibling worktree from committed `HEAD`
   without touching dirty active-checkout files.
3. Run a bounded DeepSeek loop with fixed instructions and only
   `list_project_files`, `search_project_files`, and `read_project_file`.
4. Produce a redacted local audit record and report.

It must have no write tool, no checks, no approval mutation, no SQLite access,
no merge, no remotes, and no runtime/chat integration.

## Required tests before real editing

Before enabling `replace_text_in_file` or checks, require tests covering:

- DeepSeek-only enforcement and fail-closed configuration behavior.
- Required `DEEPSEEK_CODE_MODEL` and process-environment-only credential reads.
- Tool allowlisting, schema validation, and unregistered-tool refusal.
- API-call/tool-call limits and fail-closed loop termination.
- Private `reasoning_content` continuation handling without display or
  persistence.
- Existing DSML/XML/function-call markup suppression.
- Proof that normal chat, runtime, web, tools, executors, and `app/main.py`
  cannot import or invoke freestyle modules.
- Worktree isolation that preserves a dirty active checkout and excludes its
  uncommitted files.
- Path, traversal, hidden-path, symlink, secret, runtime-data, SQLite, binary,
  submodule, and oversized-file refusals.
- Exact-match write behavior, zero/multiple/stale-match refusal, and input and
  output size limits.
- Changed-path verification and diff-risk reporting, including refusal of
  creations, deletions, renames, copies, mode changes, and unexpected files.
- Fixed subprocess allowlists, enum-only check selection, timeouts, output
  bounds, and `shell=False`.
- No remotes, GitHub, push, pull request, automatic merge, or active-checkout
  mutation.
- No `ApprovalStore` or SQLite initialization, redesign, or mutation.
- Audit redaction, no private reasoning persistence, and failure recovery.

## Open questions

- May later stages create narrowly validated safe new text files, or should
  freestyle remain limited to editing existing files?
- Does `pytest -q` need stronger OS-level sandboxing before it can be exposed,
  given that repository tests can execute code and create files?
- What audit retention duration and manual sibling-worktree cleanup policy
  should apply?
- Would a future explicitly approved-proposal launch mode be useful, while
  remaining separate from normal chat and without granting approval records
  execution authority?

