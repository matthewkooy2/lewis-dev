# Code tracing and implementation review

Lewis can statically trace a named symbol (function, async function, class,
method, React component, hook) through the developer workspace and report
its definition, callers, callees, side effects, tests, error handling, and —
in review mode — deterministic findings. All of it is read-only static
analysis: the analyzed project is never executed, modified, or installed
into, and nothing touches the network or Git.

The application code does the mechanical discovery (parsing, indexing,
call-graph traversal) and hands the model a compact, structured,
evidence-backed report, so a small local model does not need to search the
repository or infer the call graph itself.

## Configuration

Two settings control developer code tools (`trace_symbol`,
`list_project_files`, `read_project_file`, `search_project`, and the CLI
`trace` command):

```
LEWIS_DEVELOPER_CODE_WORKSPACE=true          # feature gate for the chat tools
LEWIS_DEVELOPER_WORKSPACE_ROOT=/path/to/project  # optional: external repo to analyze
```

- `LEWIS_DEVELOPER_CODE_WORKSPACE` is a boolean permission gate. It only
  controls whether `trace_symbol`, `list_project_files`, `read_project_file`,
  and `search_project` are registered as chat tools; it does not select
  which repository they analyze.
- `LEWIS_DEVELOPER_WORKSPACE_ROOT` selects the repository. If unset, every
  developer code tool (chat tools and the CLI) analyzes Lewis itself. If
  set, it must be an existing directory; the configured path is used as-is
  and is validated at startup (chat) or at invocation (CLI). An invalid or
  missing root raises a clear error instead of silently falling back to the
  Lewis repository, so a misconfigured path never traces or exposes the
  wrong project.
- The CLI `trace` command honors `LEWIS_DEVELOPER_WORKSPACE_ROOT`
  independently of `LEWIS_DEVELOPER_CODE_WORKSPACE`, since it is invoked
  directly by a developer rather than autonomously by the chat model.

## Usage

### CLI

```
python -m app.main trace <symbol> [--file PATH] [--mode quick|deep|review] [--json]
```

Examples:

```
python -m app.main trace LewisAgent.run
python -m app.main trace app.llm.LewisAgent.run --mode deep
python -m app.main trace "app/tools.py::ToolRegistry.execute" --mode review
python -m app.main trace UserCard --file src/UserCard.tsx --json
```

`--json` prints the full structured `TraceResult` (schema
`lewis-trace-result/1`), which is also the shape intended for later reuse by
Freestyle context selection.

### Lewis chat tool

When developer/code workspace mode is enabled
(`LEWIS_DEVELOPER_CODE_WORKSPACE=1`), the chat model gets a `trace_symbol`
tool alongside the project-inspection tools. It is gated exactly like
`read_project_file`: absent in normal assistant mode. Representative
requests it serves:

- "Quick trace this function." / "Who calls this method?"
- "Deep trace `LewisAgent.run`." / "Trace this React component."
- "What side effects can this function cause?"
- "Find the tests for this method."
- "Review this implementation."

The tool returns a human-readable `report` plus a bounded structured `data`
payload (schema `lewis-trace-compact/1`).

### Symbol addressing

- Bare name: `run` (ambiguous names return a candidate list, never a guess)
- Qualified: `LewisAgent.run`
- Python dotted path: `app.llm.LewisAgent.run`
- File-scoped: `app/llm.py::LewisAgent.run`, or the `file`/`--file` argument

## Modes

- **quick** — definition (file, lines, signature, decorators, exports),
  direct callers, direct callees, direct side effects, and the unresolved or
  uncertain calls, plus a bounded line-numbered source snippet.
- **deep** — quick, plus bounded transitive call paths across files,
  transitive side effects, relevant tests (direct calls confirmed; indirect
  coverage labeled inferred), error-handling summaries along the graph,
  recursion/cycles, and explicit truncation notices when limits are hit.
- **review** — deep, plus deterministic findings across correctness, error
  handling, security, side effects, maintainability, consistency, and
  testing. Every finding cites `path:line`, carries a severity
  (`issue`/`risk`/`note`) and a confidence (`confirmed` = structurally
  certain from static analysis, `heuristic` = verify before acting), and
  confirmed findings include short evidence snippets.

## Supported languages

- **Python** (`.py`) — parsed with the stdlib `ast`: functions, async
  functions, classes/methods (including inherited methods and subclass
  overrides), imports/aliases/relative imports/star imports, `__all__`,
  decorators (route/fixture registration is labeled framework-registered),
  receiver typing from constructor calls, parameter and attribute
  annotations, factory return annotations, and `with ... as` bindings.
- **JavaScript / TypeScript / JSX / TSX** (`.js`, `.jsx`, `.mjs`, `.cjs`,
  `.ts`, `.tsx`) — a real lexer (comments, strings, template literals with
  `${}` islands, regex disambiguation, JSX elements) plus a bounded
  structural parser: functions, arrows, classes/methods, object-literal
  methods, ES module imports/exports/aliases/re-exports/`export *`,
  CommonJS `require`/`module.exports`/`exports.x`, dynamic `import()`
  (labeled dynamic), TypeScript interfaces/type aliases/enums (indexed as
  types, bodies excluded from call extraction), tsconfig/jsconfig `paths`
  and `baseUrl` aliases, React components (capitalized + returns JSX),
  hooks (`useX`), JSX usage as framework-registered callers, and
  `onClick={handler}` callback registrations.

More languages can be added by writing one analyzer that emits the shared
`ModuleIndex` shape (`app/trace/model.py`); traversal, rendering, safety,
and Lewis integration are language-independent.

## How relationships are labeled

Static analysis cannot fully resolve dynamic behavior, so every edge and
effect carries a label instead of a guess:

- `confirmed` — the target is statically certain (direct definition, import
  binding, `self.method` with no overrides).
- `inferred` — statically justified but not proven (receiver typed via
  constructor/annotation/factory return annotation, unique name match,
  star-import resolution).
- `framework-registered` — invoked by a framework, not by user code (route
  decorators, JSX rendering, event-handler props, fixtures).
- `ambiguous` — several candidates exist (e.g. subclass overrides, several
  same-named methods); candidates are listed.
- `external` — resolves outside the workspace (stdlib, node_modules, pip
  packages, builtins).
- `dynamic` — the target is computed at runtime (`getattr`, computed
  member calls, dynamic `import()`).
- `unresolved` — could not be established; the reason is stated (unknown
  receiver, name too common to attribute, missing module).

Side-effect categories: `filesystem`, `network`, `process`, `database`,
`environment`, `global_state`, `console_logging`, `dom_or_browser_storage`,
`time_or_randomness`, `concurrency`. Module-level effects are marked as
running at import time.

## Safety

Tracing reuses Lewis's shared path validator and adds trace-specific rules.
It stays inside the configured workspace root, and refuses: paths outside
the root, symlinks, hidden files/dirs, `.env*`, runtime data (`data/`,
databases), dependencies (`node_modules/`, venvs, `site-packages`),
build/generated output (`dist/`, `build/`, `coverage/`, `*.min.js`,
`*.d.ts`), caches, binaries, secret-named paths, and oversized files.
Source snippets in reports are secret-redacted. The tracer never writes,
never executes project code, never runs shell commands, never installs
packages, and never touches the network or Git remotes. The chat tool is
only registered in developer/code workspace mode.

## Limits (defaults, `TraceLimits`)

| Bound | Default |
|---|---|
| Files discovered | 400 |
| File size | 512 KB |
| Deep traversal depth | 4 |
| Deep nodes / edges | 60 / 240 |
| Callers / callees shown | 30 / 40 |
| Side effects / tests / findings | 40 / 20 / 40 |
| Snippet lines / report chars | 60 / 24,000 |

When any bound is reached the report says so in a `Truncation:` section and
returns the partial result rather than failing.

## Known static-analysis limitations

- Dynamic dispatch, monkeypatching, dependency injection, `getattr`,
  computed member access, and runtime registration cannot be proven; they
  are labeled `dynamic`/`ambiguous`/`unresolved`, and subclass overrides
  make `self.method()` calls `ambiguous` by design.
- Receiver typing follows constructors, annotations, and factory return
  annotations one level at a time; untyped values (e.g. destructured
  `useState` setters, `list`/`dict` receivers) stay `unresolved`.
- Very common method names (`get`, `execute`, `save`, …) are never
  attributed by name alone when the receiver is unknown.
- The JS/TS analyzer is a bounded structural parser, not a full grammar:
  exotic TS syntax (object-literal return-type annotations before bodies,
  generic function-type annotations in `.tsx`) can degrade a file to
  partial analysis — such files are flagged in `Warnings:` with
  `parse degraded`, and results from them may be incomplete.
- Python `is_statement`/await tracking does not extend to JS floating
  promises in all positions; review flags unawaited calls only when the
  callee is a workspace async definition.
- Tests are linked by resolved calls; parametrized or dynamically generated
  test invocations may be missed (the review says so when it reports
  missing tests).
- Review findings are deterministic checks, not proof of correctness;
  `heuristic` findings in particular are prompts for a human to verify.
