# Stage 7A — Branch Applicator Design

Design-only local branch applicator spec. No implementation, no git operations, no
patch application, no subprocess execution.

---

## 1. Current Approval Storage Model

### Where approval records live

Approvals are rows in the SQLite database at `data/lewis.sqlite3` (`app/config.py:11`).

Schema (`app/db.py:76-88`, migration version 1):

```sql
CREATE TABLE approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    action_type     TEXT NOT NULL,
    title           TEXT NOT NULL,
    payload         TEXT NOT NULL,     -- JSON string
    status          TEXT NOT NULL DEFAULT 'pending',
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);
```

`ApprovalStore` in `app/approvals.py` wraps this table with `create()`, `list()`, `get()`,
and `update_status()` methods. Payloads are serialized as `json.dumps(payload,
sort_keys=True)` and deserialized with `json.loads()` in `_row_to_item()`.

### How approval type/status/payload are represented

- `action_type` — a plain string. Known values:
  - `"propose_code_change"` (`app/code_proposals.py:21`)
  - Memory action types: `"propose_memory"`, `"propose_memory_update"`
    (`app/memory_service.py`)
  - Task/calendar/email placeholder types
- `status` — one of the set defined in `app/approvals.py:15-21`:
  - `"pending"`, `"approved"`, `"rejected"`, `"completed"`, `"execution_previewed"`
- `payload` — a `dict[str, Any]` stored as JSON text. For code proposals the
  canonical payload shape is assembled in `app/code_proposals.py:76-100`.

  **Important: not all `propose_code_change` producers include the same fields.**
  See section 2 for which producers are applicator-eligible.

### How approve/reject currently behave

`app/approval_service.py:16-38`:

- `approve_approval()`:
  - For `"propose_memory"` / `"propose_memory_update"` action types: delegates to
    memory service, which creates/updates memory rows (has side effects beyond
    status).
  - For all other action types (including `"propose_code_change"`): calls
    `store.update_status(approval_id, "approved")`.
- `reject_approval()`:
  - For memory types: delegates to memory service (archives/rejects memory rows).
  - For all other types: calls `store.update_status(approval_id, "rejected")`.

### Whether approve/reject currently have side effects beyond status changes

- **For `"propose_code_change"`**: approve/reject has **no patch, command, memory,
  or external side effects**. However, `ApprovalStore.update_status()` (in
  `app/approvals.py:87-102`) updates **both** `status` and `updated_at` — it is
  not literally a status-column-only change. The method does **not** enforce a
  pending-only transition; it accepts any valid status for any row regardless of
  current state.
- The approve/reject routes (`/approvals/{id}/approve`, `/approvals/{id}/reject`
  in `app/web_routes.py:69-89`) return redirect messages confirming "No external
  action was executed."
- The `POST /approvals/{id}/execute-approved` route in `app/web_routes.py:91-103`
  calls `execute_approved()` from `app/executors.py` but this is a **preview**
  that returns a message without taking external action — it is explicitly inert.
- **Memory action types are the only approval types with side effects on
  approve/reject.**

Key consequence for the future applicator:

- Approving a `propose_code_change` approval is a status transition that also
  updates `updated_at`. The future applicator **must check current status
  itself** and should not assume transition history proves safety or prevents
  double-application.
- Memory approval types (`propose_memory`, `propose_memory_update`) are
  different and may mutate memory rows through separate dispatch logic in
  `app/approval_service.py` and `app/memory_service.py`.

---

## 2. Required Raw Payload Contract for Future Application

A future applicator must consume the raw `ApprovalItem.payload` dict from
`ApprovalStore.get(approval_id)`, not rendered approval-card HTML.

### Required approval type

```python
item.action_type == "propose_code_change"
```

### Required status

```python
item.status == "approved"
```

Refuse to apply a proposal that is still `"pending"`, `"rejected"`, `"completed"`,
or `"execution_previewed"`.

Note: `update_status()` does not reject transitions from non-pending statuses.
The applicator must enforce the approved check itself and must not assume that
the approved status implies any particular transition history.

### Which `propose_code_change` producers are applicator-eligible

Not all `propose_code_change` approvals are currently applicator-eligible.
Only proposals produced by the **replacement engine**
(`app/code_proposal_engine.py:410-432`) include the full set of structured
gating fields required for machine application:

| Producer | Code location | Includes applicator gating fields? |
|---|---|---|
| Replacement engine | `app/code_proposal_engine.py` | **Yes** — passes `extra_payload` with `selected_strategy`, `touched_paths`, `patch_preview_kind`, `patch_is_inert`, `commands_run`, `files_modified` |
| `create_search_first_code_proposal()` | `app/search_first_code_proposal.py` | **No** — calls `propose_code_change()` with no `extra_payload` argument |
| `ToolRegistry._code_proposal()` | `app/tools.py:604-614` | **No** — calls `propose_code_change()` with no `extra_payload` argument |
| `propose-code` CLI path | `app/main.py:558-570` | **No** — calls `propose_code_change()` with no `extra_payload` argument |

**Future applicator rule**: Refuse any approved `propose_code_change` payload
that does not satisfy the full applicator payload contract (all required gating
fields listed below must be present with correct types and values). Approvals
created by other producers may have a `patch_preview` field but lack the
machine-verifiable structured metadata required for safe application.

**Design recommendation for Stage 7B**: Either introduce an explicit applicator
payload contract/version field (e.g., `"applicator_contract_version": 1`) set
only by the replacement engine, or strictly gate on presence of all required
structured fields with exact type checks. Do not add partial-field leniency
unless there is an explicit, tested compatibility layer.

### Required fields in the raw payload

From `app/code_proposal_engine.py:422-432` and contract tests in
`test_code_proposal_replacement_contract.py`:

Gating fields (required for application — must be present with exact types/values):

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `patch_preview` | `str` | **yes** | Inert unified diff text starting with `diff --git ` |
| `patch_preview_kind` | `str` | **yes** | Must be exactly `"unified diff"` (space, not underscore) |
| `touched_paths` | `list[str]` | **yes** | Files the diff modifies; must be non-empty |
| `patch_is_inert` | `bool` | **yes** | Must be `True` |
| `patch_preview_implementation_ready` | `bool` | **yes** | Must be `True`; refuse if explicitly `False` or missing |

Audit / proof / reporting fields (useful metadata; must NOT be trusted for
safety decisions):

| Field | Type | Notes |
|-------|------|-------|
| `selected_strategy` | `str` | e.g. `"chat_text_color"`, `"thinking_animation"`, `"setting_limit"`. Producer assertion only. Must not be trusted directly for branch names without strict sanitization, or should not be used in branch names at all. |
| `commands_run` | `list[str]` | Read-only commands used during proposal creation. Producer assertion only. Must never be executed by the applicator. |
| `files_modified` | `list[str]` | Must be `[]` for inert proposals. Producer assertion only. |

Non-gating metadata (human-readable context, not verified):

| Field | Type | Notes |
|-------|------|-------|
| `requested_goal` | `str` | Human-readable goal text |
| `target_surface` | `str` | e.g. `"command center web UI"` |
| `relevant_files` | `list[str]` | Files read during proposal creation |
| `proposed_approach` | `str` | Human-readable approach description |
| `safety_notes` | `list[str]` | Safety guardrails |
| `suggested_test_commands` | `list[str]` | Carried for human review only; **must not be auto-executed** |
| `patch_preview_path_classifications` | `list[dict]` | Per-path classification from `app/code_proposal_file_structure.py` |
| `patch_preview_warning` | `str` | Warning if suspicious paths are present |

### Required `patch_preview_kind`

The exact persisted value from `app/code_proposal_engine.py:425` is:

```
"unified diff"
```

with a space, **not** `"unified_diff"` (underscore).

Refuse any other value including `"unified_diff"`, `"inline"`, `"context"`,
empty string, or missing. The future applicator must require the exact persisted
contract unless a compatibility layer is explicitly added and tested.

### Required `patch_is_inert`

Must be `True`. Refuse if `False`, missing, or any other falsy value.

Note: `patch_is_inert` is a producer assertion. It is useful metadata, but it
does not prove application safety. The applicator must independently validate
the patch regardless.

### Required touched paths

`touched_paths` must be a non-empty `list[str]`. The applicator must:

1. Extract paths from the raw `patch_preview` text using the existing
   `extract_patch_preview_paths()` function (`app/code_proposal_previews.py:72-78`).
2. Compare the extracted set against the `touched_paths` list from the payload.
3. **Refuse** if either set is empty.
4. **Refuse** if any extracted path is not present in `touched_paths` (a diff
   touching a file the payload didn't declare).
5. **Refuse** if `touched_paths` declares files not present in the extracted
   diff. The applicator must require exact equality between parsed diff paths
   and `touched_paths`; do not merely warn on extra declared paths. This
   prevents payloads that claim to touch files they do not actually touch.

Path comparison is case-sensitive and uses POSIX-style forward-slash relative
paths (the canonical format enforced by `validate_relative_project_path`).

---

## 3. Future Applicator Location

### Recommended module path

```
app/dev/apply_approved_proposal.py
```

Package: `app/dev/` (or `app/dev/__init__.py` plus the module).

Rationale:

- Keeps the applicator **outside normal chat runtime paths**. The chat loop
  lives in `app/llm.py` → `app/tools.py` → `app/code_proposal_engine.py`.
  Nothing in the chat path should import from `app/dev/`.
- `app/dev/` is a new package that signals developer-tooling-only.
- It does not interfere with `app/executors.py` (which handles the existing
  inert `execute_approved` preview path and should remain unchanged).
- It does not depend on `app/web_routes.py`, `app/web_pages.py`,
  `app/web_components.py`, `app/main.py`, or any UI layer.
- It imports from: `app/approvals.py` (for reading approval records),
  `app/code_proposal_previews.py`, `app/code_proposal_path_safety.py`,
  `app/code_proposal_file_structure.py`, `app/config.py`.

### Note on ApprovalStore read-only claims

Constructing `ApprovalStore` (in `app/approvals.py:36-38`) invokes
`initialize_database()`, which runs schema migrations if needed. Therefore
using `ApprovalStore.get()` is **not strictly read-only** in the strongest
sense — it may execute DDL on first access to a database file that hasn't been
migrated yet.

Future design options for Stage 7B:

- **Accept this as part of application tooling startup.** In normal operation
  the migrations are already applied (the chat runtime has initialized the DB
  by the time a user runs the applicator). The migration cost is a no-op on an
  already-migrated database.
- **Create a later no-migration / read-existing-DB-only path** if stricter
  behavior is desired. For example, a `connect_existing()` factory that skips
  `initialize_database()` and only opens the existing file.

Do not implement either option in Stage 7A.

A `__main__`-style entry point could be wired in `pyproject.toml` as a console
script later, but Stage 7A does not implement or wire it.

### Future CLI shape (placeholder only)

```
.\.venv\Scripts\python.exe -m app.dev.apply_approved_proposal 79
```

Where `79` is the approval ID. Do not implement in Stage 7A.

---

## 4. Patch Validation Rules

The applicator must validate the patch before touching any file. Validation
must be performed **before** any git operations, strictly in this order:

### Rule 1: Refuse missing patch preview

```python
patch_preview = payload.get("patch_preview")
if not isinstance(patch_preview, str) or not patch_preview.strip():
    raise Refusal("Approval payload has no patch_preview.")
```

### Rule 2: Refuse non-unified diffs

The `patch_preview` string must start with `"diff --git "` and contain the
standard unified diff markers `--- `, `+++ `, and `@@ ` within the first
reasonable line range. Additional checks:

- `patch_preview_kind` must be exactly `"unified diff"` (the exact persisted
  value from `app/code_proposal_engine.py:425`, with a space).
- No `Index:` lines (CVS-style diffs).
- No `*** ` lines (context diffs).
- No combined diffs (multi-file diff with `diff --combined` headers).
- No Git binary patch markers (`GIT binary patch`).

### Rule 3: Refuse absolute paths

Paths extracted from the diff headers (via `extract_patch_preview_paths()`)
must not be absolute. The existing `validate_relative_project_path()` in
`app/code_proposal_path_safety.py` rejects absolute POSIX and Windows paths.

### Rule 4: Refuse parent traversal

Paths must not contain `..` segments. Already enforced by
`validate_relative_project_path()` which rejects `""`, `"."`, and `".."` parts.

### Rule 5: Refuse files outside project root

Resolved path must stay within `ROOT_DIR`. Already enforced by
`validate_relative_project_path()` lines 62-68.

### Rule 6: Refuse `.env`, secrets, credentials, tokens

Enforced by `app/code_proposal_path_safety.py`:
- `BLOCKED_FILE_NAMES` includes `".env"`.
- `BLOCKED_SUFFIXES` includes `".key"`, `".pem"`, `".pfx"`.
- `SECRET_NAME_PATTERN` rejects paths containing `secret`, `credential`,
  `password`, `token`, `api_key`, `private_key`, `id_rsa`, etc.
- Dot-prefixed file and directory names are blocked.

### Rule 7: Refuse SQLite database files and runtime data

Enforced by existing safety:
- `BLOCKED_PATH_PARTS` includes `"data"` (which contains `lewis.sqlite3`).
- `BLOCKED_FILE_NAMES` includes `"lewis.sqlite3"`.
- `BLOCKED_SUFFIXES` includes `".sqlite"`, `".sqlite3"`, `".db"`.
- `BLOCKED_PATH_PARTS` includes `".venv"`, `"__pycache__"`, `".git"`,
  `".pytest_cache"`.

### Rule 8: Refuse binary patches

The `patch_preview` text must be decodable as UTF-8 and must not contain NUL
bytes (`\x00`). Any binary diff encoding (base85, quoted-printable in diff
headers) must be refused. Simple check: `"\x00" not in patch_preview`.
Also refuse Git binary patch markers (`GIT binary patch`).

### Rule 9: Refuse deletes unless explicitly supported later

If the diff contains a `deleted file mode` line or a hunk that removes all
content and the resulting file would be empty, refuse. File deletion is a
destructive operation that requires explicit opt-in. The applicator should
check for `"deleted file mode"` and `"--- a/..." / "+++ /dev/null"` patterns.

### Rule 10: Refuse rename diffs

If the diff contains `rename from` / `rename to` lines, refuse. Renames are
not currently supported.

### Rule 11: Refuse copy diffs

If the diff contains `copy from` / `copy to` lines, refuse. Copy diffs are
not currently supported.

### Rule 12: Refuse mode changes

If the diff contains a `new file mode`, `old mode` / `new mode`, or
`deleted file mode` line, refuse unless the mode is a regular file mode
(`100644`, `100755`). Specifically:

- Refuse `new file mode 120000` (symlink mode).
- Refuse any `mode` line with a value that is not `100644` or `100755`.
- Refuse `160000` (submodule / gitlink).

### Rule 13: Refuse symlink changes

The diff must not reference symlinks. Beyond the mode check (Rule 12), refuse
any diff where a parsed path resolves to an existing symlink file target at
application time. Also refuse any diff where a parent directory of a parsed
path is a symlink at application time.

### Rule 14: Refuse submodules

The diff must not contain `Subproject commit` lines or `gitlink` entries
(mode `160000`). Submodules are not supported.

### Rule 15: Refuse touched paths not exactly matching raw payload metadata

Cross-validate: the set of paths extracted from the diff via
`extract_patch_preview_paths()` must be **exactly equal** to the set declared
in `payload["touched_paths"]`. Refuse if:

- Any extracted path is missing from `touched_paths` (diff touches undeclared file).
- `touched_paths` declares files not present in the extracted diff (payload
  claims files it doesn't actually touch).

Do not "warn but not refuse" on extra declared paths. Exact equality is
required to prevent payloads from claiming to touch files they do not touch.

### Rule 16: Refuse if approval type/status is wrong

```python
if item.action_type != "propose_code_change":
    raise Refusal(f"Approval {approval_id} is not a code change proposal.")
if item.status != "approved":
    raise Refusal(f"Approval {approval_id} has status {item.status}, not approved.")
```

### Rule 17: Refuse if payload is missing applicator gating fields

Refuse if any of the required gating fields (`patch_preview_kind`, `touched_paths`,
`patch_is_inert`, `patch_preview_implementation_ready`) are missing, have wrong
types, or have wrong values. This inherently refuses non-replacement-engine
payloads that do not set `extra_payload`.

### Rule 18: Patch size, file count, and hunk count limits

The applicator must enforce:

- Maximum total patch size (suggested: 64 KiB).
- Maximum number of touched files per approval (suggested: 20).
- Maximum number of hunks per file (suggested: 20).
- Maximum number of total hunks across all files (suggested: 40).

These limits prevent overly large or complex patches from being applied in one
step.

### Rule 19: Bind patch to base commit

The applicator should record the current HEAD commit hash at application time.
If feasible, the approval payload should include or be matched against a
`patch_base_commit` digest so that a patch created against a stale tree is
refused. This prevents application of patches to incorrect base states.

### Rule 20: Verify after application that only declared paths changed

After `git apply`, verify with `git diff --name-only` (against the pre-apply
commit or staged state) that the set of changed files is exactly the set of
`touched_paths`. Refuse if any undeclared file was modified.

### Rule 21: Verify after tests that only declared paths changed

After running the allowlisted post-apply commands (ruff check, pytest), verify
again that only declared paths were modified. Account for known generated/cache
files (e.g., `__pycache__/`, `.ruff_cache/`, `.pytest_cache/`). Refuse if any
undeclared tracked file was modified.

### Rule 22: Define cleanup behavior

If application succeeds, leave the branch in place and report the branch name
for human review. Do not assume branch deletion is always harmless.

If application or validation fails:

- Check out the original branch.
- Delete the proposal branch (`git branch -D`).
- Report the failure.

If the user wants to keep the branch after failure for inspection, they must
opt in explicitly. The default is cleanup.

### Additional: Refuse if patch_preview_is_implementation_ready is explicitly false

If `patch_preview_implementation_ready` is `False` (explicitly, not just
missing), refuse with diagnostics pointing at `patch_preview_warning`.

---

## 5. Git/Worktree Rules for the Future

All git operations are local only. The applicator must never interact with
remotes.

### Rule A: Refuse dirty working tree

Before creating a branch, the applicator must detect **all** of:

- Tracked modifications (staged and unstaged).
- Untracked files.

A complete check requires:

```
git status --porcelain
```

and checking for any output. The check `git diff --quiet && git diff --cached --quiet`
is **insufficient** because it ignores untracked files.

The safest policy: refuse if `git status --porcelain` produces any output.
If a more permissive policy is desired (e.g., allowing untracked files outside
tracked directories), it must be explicitly documented and opt-in. The default
must be the safest policy: fully clean working tree.

### Rule B: Create deterministic safe local branch name

Branch naming pattern:

```
lewis/proposal-{approval_id}-{sanitized_slug}
```

Where `{sanitized_slug}` is a short sanitized identifier derived from the
proposal metadata (e.g., first 20 chars of `requested_goal`, lowercased,
non-alphanumeric replaced with `-`, trimmed).

**Do not use `selected_strategy` in branch names** without strict sanitization.
`selected_strategy` is a producer assertion field and could contain unexpected
characters. If used at all, it must be validated against a known allowlist of
strategy names before inclusion in a branch name.

Examples of acceptable branch names:
- `lewis/proposal-79-chat-text-color`
- `lewis/proposal-82-thinking-animation`
- `lewis/proposal-85-tool-use-limit`

The branch must be created from the current HEAD (`git checkout -b ...`).

If the branch already exists locally, refuse with a message. The user must
delete the old branch manually before re-applying.

### Rule C: Apply only on that branch with safe command execution

After creating and checking out the branch, apply the patch using **fixed
argument arrays** and **`shell=False`**. The `patch_preview` string must be
passed to `git apply` via **stdin**, never via shell redirection or pipes.

Implementation constraints:

```python
# Safe: fixed argument array, stdin feed, no shell
subprocess.run(
    ["git", "apply", "--verbose", "--check"],
    input=patch_preview,
    text=True,
    capture_output=True,
    check=False,
)
```

Prohibited:
- `shell=True`
- Shell redirection (`>`, `<`, `|`)
- Pipes
- Commands constructed from approval payload fields
- Commands constructed from user/chat text
- `git apply --reject` (produces `.rej` files and partial application, which
  breaks atomic safety)

The safe workflow is:

1. `git apply --check` with the patch on stdin to validate.
2. If check passes, `git apply` with the patch on stdin to apply atomically.
3. If `git apply` fails, abort by checking out the original branch and
   deleting the proposal branch (`git checkout - && git branch -D
   lewis/proposal-{id}-{slug}`).

### Rule D: Do not push

Never run `git push`. Never read or use `origin`, `upstream`, or any remote
configuration.

### Rule E: Do not create GitHub PRs

No GitHub CLI, no API calls, no hub integration.

### Rule F: Do not use remotes

Do not read `git remote`, `git ls-remote`, or any remote-tracking operation.

### Rule G: Do not read or expose tokens

Do not read `GITHUB_TOKEN`, `GH_TOKEN`, `GIT_TOKEN`, or any environment
variable containing `TOKEN`, `SECRET`, `KEY`, or `PASSWORD`.

---

## 6. Future Command/Test Rules

### Never run commands from the approval payload

The `suggested_test_commands` field in the payload is **carried for human
review only**. The applicator must not parse or execute it.

The `commands_run` field records what the proposal engine did during creation
(e.g., `"search_project_files"`). It is metadata only. The applicator must not
parse or execute it.

**Explicit prohibition**: `suggested_test_commands` and `commands_run` are
**never executable sources** for the applicator. The applicator must contain
its own hardcoded allowlist and must never derive commands from any payload
field, chat text, or user input.

### Only run fixed allowlisted commands

The applicator may automatically run **only** these commands after successful
patch application, in this order:

1. **Lint check:**
   ```
   .\.venv\Scripts\python.exe -m ruff check .
   ```

2. **Full test suite:**
   ```
   .\.venv\Scripts\python.exe -m pytest -q
   ```

No other commands are permitted. Specifically prohibited:
- Arbitrary shell commands
- Commands from user text or chat messages
- Commands from the approval payload (any field)
- `eval`, `exec`, `source`, or shell evaluation
- Package installers (`pip`, `npm`, etc.)
- `git commit`, `git merge`, or any git write operations beyond branch
  creation and cleanup

### Command execution environment

Commands must:
- Run from `ROOT_DIR` (`app/config.py:8`)
- Use `.\.venv\Scripts\python.exe` explicitly (per `.clinerules`)
- Use fixed argument arrays with `shell=False`
- Have a reasonable timeout (suggested: 120 seconds for ruff, 300 seconds for
  pytest)
- Have their stdout/stderr captured for the result report
- Not modify any files beyond the patch (ruff may auto-fix if configured, but
  the applicator should run `ruff check` not `ruff format` + `ruff check`
  unless explicit)

---

## 7. Future State/Reporting Model

### Proposed statuses

The applicator does not modify `ApprovalItem.status` directly (the user already
set it to `"approved"`). Instead, the applicator produces a **result report**
object. If the team later wants persistent tracking, these could become a new
column or separate table, but Stage 7A does NOT make schema changes.

Proposed **ephemeral** result statuses for the in-memory/printed report:

| Result status | Meaning |
|---------------|---------|
| `apply_succeeded_tests_passed` | Patch applied cleanly; ruff and pytest both passed |
| `apply_succeeded_tests_failed` | Patch applied cleanly; ruff or pytest failed |
| `apply_failed` | `git apply` rejected the patch; no files were modified |
| `apply_refused` | Pre-flight validation failed; no git operations performed |

These are **not persisted approval statuses**. They exist only in the runtime
`ApplyResult` object and on stdout. If any future stage wants persisted
execution reporting, it must be designed separately and may require SQLite
schema changes (do not add in Stage 7A or 7B unless explicitly approved).

### What should be recorded in a result report

A result report (printed to stdout and optionally returned as a structured
object) should include:

```python
@dataclass
class ApplyResult:
    approval_id: int
    requested_goal: str
    branch_name: str | None          # None if refused before branch creation
    result_status: str               # one of the four above
    touched_paths: list[str]
    patch_validation: dict[str, bool]  # per-rule results
    base_commit: str | None          # HEAD commit hash at application time
    git_apply_output: str | None
    ruff_output: str | None
    pytest_output: str | None
    pytest_exit_code: int | None
    errors: list[str]
    warnings: list[str]
    user_action_required: str | None  # e.g. "review and merge lewis/proposal-79/..."
```

### No SQLite schema/data changes in Stage 7A or Stage 7B

The result report is ephemeral (stdout + return value). No new columns, no new
tables, no migration, no `ApprovalStore` method changes.

---

## Summary of Safety Invariants Preserved

1. No files are modified outside a dedicated local git branch.
2. The working tree must be completely clean before starting (tracked
   modifications, staged changes, **and** untracked files all blocked by
   default).
3. The branch name is deterministic and namespaced under `lewis/`.
4. No remote operations, no pushes, no PRs, no tokens.
5. Only allowlisted commands run after application; commands are executed
   via fixed argument arrays with `shell=False`.
6. Patch validation runs before any git operation.
7. If validation or application fails, the applicator cleans up the branch.
8. The original branch is restored on failure.
9. Only replacement-engine code proposal approvals with the complete set of
   gating fields are applicator-eligible. Legacy/non-contract payloads are
   refused.
10. The applicator reads approval records from SQLite (or another approval
    store). It does not write to SQLite. Note: constructing `ApprovalStore`
    invokes `initialize_database()`, which is a startup cost but not a write
    on an already-migrated database.
11. `.env`, secrets, credentials, and runtime data paths are blocked by
    existing path-safety rules.
12. `git apply --reject` is prohibited. Patch application is atomic via
    stdin, never via shell redirection.
13. `patch_preview_kind` must match the exact persisted value `"unified diff"`
    (space, not underscore).
14. `commands_run` and `suggested_test_commands` are never executable sources.
15. `selected_strategy` is a producer assertion and must not be trusted for
    branch names without sanitization.
16. Renames, copies, mode changes, symlinks, submodules, binary patches,
    combined diffs, and file deletes are all refused unless explicitly
    supported later.
17. Exact equality is required between parsed diff paths and `touched_paths`.
18. Patch size, file count, and hunk count limits are enforced.
19. Post-application verification confirms only declared paths changed.

---

## Files Changed

- `docs/stage_7a_branch_applicator_design.md` — **updated** (Stage 7A.1 corrections)

## Tests Added/Updated

None. Stage 7A.1 is doc/contract correction only.

## Confirmation

No branch applicator, patch application, git command execution, subprocess
execution, GitHub integration, executor, runtime shell/code execution,
ApprovalStore redesign, SQLite schema change, or SQLite data migration was
added.

### Corrections applied in Stage 7A.1

1. **Eligible proposal payloads clarified**: Document now states that only
   replacement-engine proposals include the full set of gating fields; lists
   all three non-eligible producers with code locations; adds future refusal
   rule for non-contract payloads; recommends explicit contract version field
   for Stage 7B.

2. **`patch_preview_kind` corrected**: Changed from `"unified_diff"` to the
   exact persisted value `"unified diff"` (space) per
   `app/code_proposal_engine.py:425`. Clarified that compatibility with
   alternate spellings would need intentional, tested support.

3. **Gating vs audit fields distinguished**: Table now separates required
   application-gating fields (`patch_preview`, `patch_preview_kind`,
   `touched_paths`, `patch_is_inert`, `patch_preview_implementation_ready`)
   from audit/proof/reporting fields (`commands_run`, `files_modified`,
   `selected_strategy`). Documents that `commands_run`, `files_modified`, and
   `patch_is_inert` are producer assertions; applicator must independently
   validate. `selected_strategy` must not be trusted for branch names without
   strict sanitization.

4. **Approval side-effect wording corrected**: States explicitly that
   `update_status()` updates both `status` and `updated_at` — not literally
   status-column-only. Documents that `update_status()` does not enforce
   pending-only transitions and that the applicator must check current status
   itself. Clarifies memory approval types have separate dispatch logic.

5. **SQLite/read-only claims corrected**: Removes incorrect claim that the
   applicator never reads SQLite. Documents that constructing `ApprovalStore`
   invokes `initialize_database()` (DDL cost, not a write on already-migrated
   DB). Lists future design options (accept startup cost or create
   no-migration read-existing-DB-only path).

6. **Patch validation rules strengthened**: Added rules for rename diffs,
   copy diffs, all mode changes, symlink changes, submodules, combined diffs,
   Git binary patch markers, binary patches. Changed "warn but do not refuse"
   to "refuse" for extra declared paths (exact equality required). Added
   patch size/file count/hunk count limits. Added base commit binding.
   Added post-apply and post-test verification. Added cleanup behavior
   definition.

7. **Dirty-tree rule fixed**: Changed from `git diff --quiet && git diff --cached --quiet`
   (which ignores untracked files) to `git status --porcelain` for a complete
   check covering tracked modifications, staged changes, and untracked files.

8. **Unsafe command examples removed**: Removed `git apply --reject` example
   entirely. Specified `shell=False`, fixed argument arrays, stdin-only patch
   input. Added explicit prohibition on `suggested_test_commands` and
   `commands_run` as executable sources.

9. **Future reporting/status model clarified**: `ApplyResult` statuses are
   explicitly documented as ephemeral (not persisted approval statuses).
   States that any future persistent execution reporting would need separate
   design and may require schema changes. Removed `selected_strategy` from
   `ApplyResult` dataclass (audit field, not needed in result). Added
   `base_commit` field.