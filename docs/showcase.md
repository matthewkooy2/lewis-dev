# Five-minute walkthrough

Start from a fresh clone and complete the [README setup](../README.md#quick-start).
These commands run from the repository root with no API key or model server.
Keep `.env` absent for this walkthrough so local workspace overrides cannot
change the repository being demonstrated.

## 1. Trace actual application code

```sh
uv run --locked python -m app.main trace create_app --file app/web.py --mode deep
```

The report names the definition, callers, callees, and effects with file/line
citations. The tool analyzes source; it does not execute the repository it traces.

For structured output:

```sh
uv run --locked python -m app.main trace create_app --file app/web.py --mode deep --json
```

## 2. Demonstrate the worktree contract

```sh
uv run --locked pytest -q tests/test_freestyle_loop.py::test_loop_reaches_human_approval_after_checks_and_persists_state tests/test_freestyle_loop.py::test_safety_failure_stops_without_diagnosis_or_repair
```

These integration tests use **scripted model responses**, not live model
inference. They create real temporary Git repositories, stage a change in a
separate worktree, run checks, and verify the original source remains unchanged.
The second case verifies that a safety refusal stops the loop.

The worktree isolates changes to the checkout. Checks run with the current
user's permissions, and these fixtures exercise the workflow with scripted
responses.

## 3. Inspect the dashboard

```sh
uv run --locked python -m app.main web
```

Open <http://127.0.0.1:8787>. The dashboard works without a model; chat requires
one. Optional Freestyle and Projects screens are enabled through the explicit
flags in the [usage reference](usage.md). Do not use your personal database,
conversation history, or existing job artifacts in a public recording.
