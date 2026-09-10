# Contributing to Lewis

Follow the [README setup](README.md#setup) from the repository root. Development
uses Python 3.11.15 and uv 0.12.11. CI exercises that Python version on Linux,
macOS, and Windows; other Python versions are not currently in the CI matrix.
The interpreter and Python packages are pinned, while the hosted runner images
and their system tools still receive updates.

## Before submitting a change

Keep changes focused and describe the observable behavior they change.
Add regression tests for bugs or new behavior. Tests use synthetic data and
fake model responses, so API keys and live endpoints are unnecessary.

```sh
uv sync --locked
uv run --locked python -m ruff check .
uv run --locked python -m ruff format --check .
uv run --locked python -m pytest -q -ra
uv run --locked python -m app.eval_runner
uv run --locked python -m app.main --help
```

Run these commands from a fresh checkout to reproduce CI without local `.env`
configuration. Git must be on `PATH`: the integration tests create temporary
repositories and run real Git commands. Windows CI also runs the PowerShell
parser and junction tests. No external service secrets are configured in CI.

Pytest's default temporary directory is `runtime/pytest-temp/current`. Do not
run two suites concurrently with that same directory; pass distinct
`--basetemp` paths if parallel local runs are necessary. Runtime fixture
repositories are excluded from ordinary pytest and Ruff discovery.

For fixtures that compare exact edit anchors or SHA-256 hashes, write explicit
line endings (`Path.write_text(..., newline="\n")` or `Path.write_bytes(...)`).
Python otherwise translates newlines on Windows even when Git's `core.autocrlf`
is disabled. Use `read_bytes()` for byte-preservation assertions; `read_text()`
normalizes CRLF. Keep explicit LF and CRLF cases when testing edit preservation.

## Dependency changes

`pyproject.toml` defines runtime dependencies and the `dev` dependency group.
`uv.lock` records their resolved versions and hashes across platforms.
`requirements.txt` is the generated pip-compatible export, including `dev`.
Never edit the export by hand or generate it from `pip freeze`.

To add a dependency:

```sh
uv add PACKAGE
# For development tooling:
uv add --dev PACKAGE
```

To update an existing dependency intentionally:

```sh
uv lock --upgrade-package PACKAGE
```

Then regenerate the export, sync, and run the checks above:

```sh
uv export --locked --all-groups --no-emit-project --output-file requirements.txt
uv sync --locked
```

Commit `pyproject.toml`, `uv.lock`, and `requirements.txt` together when they
change. CI rejects a stale lockfile or a mismatched export. Normal setup uses
`--locked` so installing the project cannot silently select new package versions.

Ruff is pinned explicitly because formatter upgrades can change repository-wide
output. Keep any formatter upgrade and its formatting pass separate from
behavior changes. To update uv, change `tool.uv.required-version` in
`pyproject.toml`, the setup step in CI, and the documented installation version
together. For Python updates, change `.python-version`, the CI job label, and
the setup documentation, then validate all three platforms.

GitHub Actions are pinned to commit hashes, use read-only repository permissions,
and do not retain checkout credentials. Review action updates before changing
their hashes. Contributors should not need repository or model-provider secrets
to run the checks.
