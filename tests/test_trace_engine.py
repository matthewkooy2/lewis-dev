import os
from pathlib import Path

from app.trace.model import TraceLimits
from app.trace.service import CodeTracer
from app.trace.workspace import TraceWorkspace


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _python_project(tmp_path: Path) -> Path:
    root = tmp_path / "pyproj"
    _write(
        root,
        "app/store.py",
        "import sqlite3\n"
        "\n"
        "class Store:\n"
        "    def save(self, item) -> None:\n"
        '        with sqlite3.connect("db") as conn:\n'
        '            conn.execute("INSERT INTO items VALUES (?)", (item,))\n'
        "\n"
        "    def load(self):\n"
        '        raise KeyError("missing")\n',
    )
    _write(
        root,
        "app/service.py",
        "from app.store import Store as BackingStore\n"
        "\n"
        "def build_store() -> BackingStore:\n"
        "    return BackingStore()\n"
        "\n"
        "class Service:\n"
        "    def __init__(self, store: BackingStore):\n"
        "        self.store = store\n"
        "\n"
        "    def record(self, item):\n"
        "        self.store.save(item)\n"
        "        return self.mirror(item)\n"
        "\n"
        "    def mirror(self, item):\n"
        "        return self.record(item)\n",
    )
    _write(
        root,
        "app/entry.py",
        "from app.service import Service, build_store\n"
        "\n"
        "def main():\n"
        "    service = Service(build_store())\n"
        '    service.record("x")\n'
        "    store = build_store()\n"
        '    store.save("y")\n',
    )
    _write(
        root,
        "tests/test_service.py",
        "from app.service import Service, build_store\n"
        "\n"
        "def test_record():\n"
        '    Service(build_store()).record("a")\n',
    )
    return root


def _js_project(tmp_path: Path) -> Path:
    root = tmp_path / "jsproj"
    _write(
        root,
        "tsconfig.json",
        "{\n"
        "  // comment allowed\n"
        '  "compilerOptions": {\n'
        '    "baseUrl": "src",\n'
        '    "paths": { "@lib/*": ["src/lib/*"], },\n'
        "  },\n"
        "}\n",
    )
    _write(
        root,
        "src/lib/api.ts",
        "export async function saveUser(user: unknown) {\n"
        '  const response = await fetch("/api", { method: "POST" });\n'
        "  if (!response.ok) {\n"
        '    throw new Error("save failed");\n'
        "  }\n"
        "  return response;\n"
        "}\n",
    )
    _write(
        root,
        "src/lib/index.ts",
        'export { saveUser as persistUser } from "./api";\n',
    )
    _write(
        root,
        "src/UserCard.tsx",
        'import { useState } from "react";\n'
        'import { persistUser } from "@lib/index";\n'
        "\n"
        "export function useSaver() {\n"
        "  const [busy, setBusy] = useState(false);\n"
        "  const save = async (user) => {\n"
        "    try {\n"
        "      await persistUser(user);\n"
        "    } catch (err) {\n"
        "    }\n"
        "  };\n"
        "  return save;\n"
        "}\n"
        "\n"
        "export default function UserCard({ user }) {\n"
        "  const save = useSaver();\n"
        "  return <button onClick={save}>{user.name}</button>;\n"
        "}\n",
    )
    _write(
        root,
        "src/UserCard.test.tsx",
        'import UserCard, { useSaver } from "./UserCard";\n'
        "\n"
        'test("renders", () => {\n'
        '  const el = <UserCard user={{ name: "Ada" }} />;\n'
        "});\n"
        "\n"
        'test("saves", async () => {\n'
        "  const save = useSaver();\n"
        '  await save({ name: "Ada" });\n'
        "});\n",
    )
    return root


# ---- workspace safety ---------------------------------------------------------


def test_workspace_blocks_unsafe_paths_and_symlinks(tmp_path):
    root = _python_project(tmp_path)
    _write(root, ".env", "SECRET=1")
    _write(root, "data/lewis.sqlite3", "not really sqlite")
    _write(root, "node_modules/pkg/index.js", "module.exports = 1;")
    _write(root, "dist/bundle.js", "x")
    _write(root, "app/secrets.py", "KEY = 'x'")
    _write(root, "app/big.py", "x = 1\n" * 100_000)
    _write(root, "app/vendor.min.js", "x")
    _write(root, "app/types.d.ts", "export type X = 1;")
    (root / "app" / "binary.py").write_bytes(b"print('x')\x00")
    outside = tmp_path / "outside.py"
    outside.write_text("escape = True\n", encoding="utf-8")
    if os.name != "nt":
        try:
            (root / "app" / "link.py").symlink_to(outside)
        except OSError:
            pass

    workspace = TraceWorkspace(root)
    listed = {record.path for record in workspace.list_source_files()}

    assert "app/store.py" in listed
    assert "tests/test_service.py" in listed
    assert not any(
        path.startswith((".", "data/", "node_modules/", "dist/")) for path in listed
    )
    assert "app/secrets.py" not in listed
    assert "app/big.py" not in listed
    assert "app/vendor.min.js" not in listed
    assert "app/types.d.ts" not in listed
    assert "app/binary.py" not in listed
    assert "app/link.py" not in listed

    assert workspace.read_source("../outside.py") is None
    assert workspace.read_source("/etc/passwd") is None
    assert workspace.read_source(".env") is None
    assert workspace.read_source("data/lewis.sqlite3") is None
    assert workspace.read_source("app/binary.py") is None
    assert workspace.read_source("app/link.py") is None
    assert workspace.read_source("app/store.py") is not None


def test_workspace_discovery_is_bounded(tmp_path):
    root = tmp_path / "many"
    for index in range(12):
        _write(root, f"src/module_{index:02}.py", "x = 1\n")
    workspace = TraceWorkspace(root, limits=TraceLimits(max_files=5))
    files = workspace.list_source_files()
    assert len(files) == 5
    assert workspace.discovery_truncated is True


def test_workspace_never_writes(tmp_path):
    root = _python_project(tmp_path)
    before = {
        path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()
    }
    tracer = CodeTracer(root)
    tracer.trace("Service.record", mode="review")
    after = {
        path: path.stat().st_mtime_ns for path in root.rglob("*") if path.is_file()
    }
    assert before == after


# ---- python engine --------------------------------------------------------------


def test_quick_trace_python_callers_callees_and_effects(tmp_path):
    tracer = CodeTracer(_python_project(tmp_path))
    result = tracer.trace("Store.save", mode="quick").result

    assert result.ok and result.target is not None
    assert result.target.path == "app/store.py"
    caller_ids = {(edge.caller_path, edge.caller_qualname) for edge in result.callers}
    assert ("app/service.py", "Service.record") in caller_ids
    assert ("app/entry.py", "main") in caller_ids
    effect_categories = {effect.category for effect in result.side_effects}
    assert "database" in effect_categories


def test_quick_trace_resolves_alias_and_return_annotation(tmp_path):
    tracer = CodeTracer(_python_project(tmp_path))
    result = tracer.trace("Store.save", mode="quick").result
    entry_edge = next(
        edge for edge in result.callers if edge.caller_path == "app/entry.py"
    )
    assert entry_edge.resolution == "inferred"
    assert "return annotation" in entry_edge.note


def test_trace_reports_candidates_for_ambiguous_symbol(tmp_path):
    root = _python_project(tmp_path)
    _write(root, "app/other.py", "def record(row):\n    return row\n")
    tracer = CodeTracer(root)
    result = tracer.trace("record", mode="quick").result

    assert result.ok is True
    assert result.target is None
    assert len(result.candidates) == 2
    report = tracer.trace("record", mode="quick").report
    assert "AMBIGUOUS" in report
    assert "app/other.py" in report

    picked = tracer.trace("record", mode="quick", file="app/other.py").result
    assert picked.target is not None
    assert picked.target.path == "app/other.py"

    qualified = tracer.trace("app/service.py::Service.record").result
    assert qualified.target is not None
    assert qualified.target.qualname == "Service.record"


def test_trace_unknown_symbol_suggests_close_names(tmp_path):
    tracer = CodeTracer(_python_project(tmp_path))
    outcome = tracer.trace("Service.recrod")
    assert outcome.result.ok is False
    assert "No definition found" in outcome.result.error
    assert "record" in outcome.result.error


def test_deep_trace_detects_cycles_and_transitive_effects(tmp_path):
    tracer = CodeTracer(_python_project(tmp_path))
    result = tracer.trace("Service.record", mode="deep").result

    assert result.cycles, "record<->mirror recursion should be reported"
    flattened = " ".join(" ".join(cycle) for cycle in result.cycles)
    assert "Service.record" in flattened and "Service.mirror" in flattened
    transitive = {effect.category for effect in result.transitive_side_effects}
    assert "database" in transitive
    assert any(link.via == "calls target" for link in result.tests)


def test_deep_trace_respects_depth_limit_with_truncation_notice(tmp_path):
    root = tmp_path / "deep"
    chain = (
        "\n".join(f"def f{i}():\n    return f{i + 1}()\n" for i in range(8))
        + "\ndef f8():\n    return 1\n"
    )
    _write(root, "chain.py", chain)
    tracer = CodeTracer(root, limits=TraceLimits(max_depth=2))
    result = tracer.trace("f0", mode="deep").result
    assert any("deeper than 2" in note for note in result.truncations)
    assert result.stats["deep_nodes"] <= 3


def test_dynamic_calls_are_labeled_not_guessed(tmp_path):
    root = tmp_path / "dyn"
    _write(
        root,
        "dyn.py",
        "class Runner:\n"
        "    def dispatch(self, name):\n"
        "        handler = getattr(self, name)\n"
        "        return handler()\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("Runner.dispatch", mode="quick").result
    resolutions = {edge.resolution for edge in result.callees}
    assert "dynamic" in resolutions
    assert all(
        edge.resolution != "confirmed" or edge.target_path for edge in result.callees
    )


def test_subclass_override_marks_dispatch_ambiguous(tmp_path):
    root = tmp_path / "poly"
    _write(
        root,
        "shapes.py",
        "class Base:\n"
        "    def area(self):\n"
        "        return 0\n"
        "\n"
        "    def describe(self):\n"
        "        return self.area()\n"
        "\n"
        "class Circle(Base):\n"
        "    def area(self):\n"
        "        return 3\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("Base.describe", mode="quick").result
    edge = next(edge for edge in result.callees if "area" in edge.callee_text)
    assert edge.resolution == "ambiguous"
    assert "override" in edge.note


def test_common_method_names_on_unknown_receivers_stay_unresolved(tmp_path):
    tracer = CodeTracer(_python_project(tmp_path))
    result = tracer.trace("Store.save", mode="quick").result
    conn_edge = next(
        edge for edge in result.callees if edge.callee_text == "conn.execute"
    )
    assert conn_edge.resolution == "external"
    assert "sqlite3" in conn_edge.note


# ---- js engine --------------------------------------------------------------------


def test_js_trace_resolves_reexports_and_tsconfig_aliases(tmp_path):
    tracer = CodeTracer(_js_project(tmp_path))
    result = tracer.trace("saveUser", mode="quick").result

    assert result.ok and result.target is not None
    assert result.target.path == "src/lib/api.ts"
    caller = next(iter(result.callers))
    assert caller.caller_path == "src/UserCard.tsx"
    assert caller.resolution == "confirmed"
    assert {effect.category for effect in result.side_effects} == {"network"}


def test_js_component_gets_framework_registered_jsx_caller(tmp_path):
    tracer = CodeTracer(_js_project(tmp_path))
    result = tracer.trace("UserCard", mode="quick").result
    jsx_callers = [
        edge for edge in result.callers if edge.resolution == "framework-registered"
    ]
    assert any(edge.caller_path == "src/UserCard.test.tsx" for edge in jsx_callers)
    assert result.target is not None and result.target.kind == "react_component"


def test_js_deep_trace_finds_tests_and_transitive_network(tmp_path):
    tracer = CodeTracer(_js_project(tmp_path))
    result = tracer.trace("useSaver", mode="deep").result
    assert any(link.test_path == "src/UserCard.test.tsx" for link in result.tests)
    assert {e.category for e in result.transitive_side_effects} == {"network"}


def test_parse_degraded_file_is_reported_honestly(tmp_path):
    root = _js_project(tmp_path)
    _write(root, "src/broken.js", "const t = `never closed\nfunction ghost() {}")
    tracer = CodeTracer(root)
    result = tracer.trace("saveUser", mode="quick").result
    assert any("degraded" in warning for warning in result.warnings)


# ---- review mode ---------------------------------------------------------------------


def test_review_flags_python_issues_with_lines(tmp_path):
    root = tmp_path / "reviewpy"
    _write(
        root,
        "app/logic.py",
        "import subprocess\n"
        "\n"
        "def get_data(items=[]):\n"
        "    try:\n"
        '        subprocess.run("ls", shell=True)\n'
        "    except:\n"
        "        pass\n"
        "    return items\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("get_data", mode="review").result
    messages = {(f.category, f.severity, f.line) for f in result.findings}
    texts = " ".join(f.message for f in result.findings)

    assert ("error_handling", "issue", 6) in messages  # bare except
    assert "Mutable default" in texts
    assert "shell=True" in texts or "Shell execution" in texts
    assert any(f.category == "testing" for f in result.findings)
    assert any(f.category == "side_effects" for f in result.findings)
    assert all(f.path == "app/logic.py" and f.line >= 1 for f in result.findings)
    assert any("Static review only" in warning for warning in result.warnings)


def test_review_confidence_separates_confirmed_from_heuristic(tmp_path):
    root = tmp_path / "reviewconf"
    _write(
        root,
        "app/mixed.py",
        "def helper(cursor, name):\n"
        "    cursor.execute(f\"SELECT * FROM users WHERE name = '{name}'\")\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("helper", mode="review").result
    sql = next(f for f in result.findings if "SQL" in f.message)
    assert sql.confidence == "heuristic"
    assert sql.category == "security"


def test_review_flags_js_swallowed_catch_and_loose_equality(tmp_path):
    root = _js_project(tmp_path)
    _write(
        root,
        "src/check.js",
        "export function check(user) {\n"
        "  if (user == null) {\n"
        "    try { risky(); } catch (e) {}\n"
        "  }\n"
        "  return user;\n"
        "}\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("check", mode="review").result
    categories = {f.category for f in result.findings}
    assert "error_handling" in categories
    swallowed = next(f for f in result.findings if "swallows" in f.message)
    assert swallowed.confidence == "confirmed"


def test_review_findings_survive_into_compact_dict(tmp_path):
    from app.trace.render import to_compact_dict

    root = tmp_path / "compact"
    _write(
        root, "app/x.py", "def f():\n    try:\n        g()\n    except:\n        pass\n"
    )
    tracer = CodeTracer(root)
    result = tracer.trace("f", mode="review").result
    data = to_compact_dict(result)
    assert data["mode"] == "review"
    assert data["findings"], "compact dict must carry findings"
    assert {"category", "severity", "confidence", "path", "line", "message"} <= set(
        data["findings"][0]
    )


# ---- limits / snippets ------------------------------------------------------------------


def test_snippets_are_redacted_and_bounded(tmp_path):
    root = tmp_path / "redact"
    body = "\n".join(f"    x{i} = {i}" for i in range(120))
    _write(
        root,
        "app/secretsish.py".replace("secretsish", "settings"),
        "def configure():\n"
        '    api_key = "sk-testtesttesttesttesttesttest"\n' + body + "\n",
    )
    tracer = CodeTracer(root, limits=TraceLimits(max_snippet_lines=10))
    result = tracer.trace("configure", mode="quick").result
    snippet = result.definition_snippet
    assert snippet is not None
    assert snippet.truncated is True
    assert snippet.end_line - snippet.start_line + 1 == 10
    assert "sk-testtesttesttesttesttesttest" not in snippet.text
    assert "[redacted]" in snippet.text


def test_caller_truncation_is_reported(tmp_path):
    root = tmp_path / "manycallers"
    _write(root, "core.py", "def hot():\n    return 1\n")
    callers = "from core import hot\n" + "\n".join(
        f"def caller_{i}():\n    return hot()\n" for i in range(12)
    )
    _write(root, "users.py", callers)
    tracer = CodeTracer(root, limits=TraceLimits(max_callers=5))
    result = tracer.trace("hot", mode="quick").result
    assert len(result.callers) == 5
    assert any("Callers truncated at 5" in note for note in result.truncations)


def test_index_rebuilds_when_files_change(tmp_path):
    root = tmp_path / "cache"
    _write(root, "m.py", "def one():\n    return 1\n")
    tracer = CodeTracer(root)
    assert tracer.trace("one").result.ok is True
    assert tracer.trace("two").result.ok is False
    _write(root, "m.py", "def one():\n    return 1\n\ndef two():\n    return 2\n")
    assert tracer.trace("two").result.ok is True


def test_result_to_dict_is_json_serializable(tmp_path):
    import json

    tracer = CodeTracer(_python_project(tmp_path))
    result = tracer.trace("Service.record", mode="deep").result
    payload = json.dumps(result.to_dict())
    assert "lewis-trace-result/1" in payload


def test_python_project_with_relative_imports_resolves(tmp_path):
    root = tmp_path / "relpkg"
    _write(root, "pkg/__init__.py", "")
    _write(root, "pkg/util.py", "def shared():\n    return 1\n")
    _write(
        root,
        "pkg/feature.py",
        "from .util import shared\n\ndef use():\n    return shared()\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("shared", mode="quick").result
    assert result.target is not None
    caller = next(iter(result.callers))
    assert caller.caller_path == "pkg/feature.py"
    assert caller.resolution == "confirmed"


def test_commonjs_require_calls_resolve_across_files(tmp_path):
    root = tmp_path / "cjs"
    _write(
        root,
        "lib/util.js",
        "function formatRow(row) {\n  return row;\n}\n"
        "module.exports = { formatRow };\n",
    )
    _write(
        root,
        "main.js",
        'const { formatRow } = require("./lib/util");\n'
        "function run() {\n  return formatRow(1);\n}\n",
    )
    tracer = CodeTracer(root)
    result = tracer.trace("formatRow", mode="quick").result
    assert result.target is not None
    assert result.target.path == "lib/util.js"
    caller = next(iter(result.callers))
    assert caller.caller_path == "main.js"
    assert caller.resolution == "confirmed"


def test_trace_rejects_unknown_mode(tmp_path):
    tracer = CodeTracer(_python_project(tmp_path))
    outcome = tracer.trace("Store.save", mode="everything")
    assert outcome.result.ok is False
    assert "Unknown mode" in outcome.result.error


def test_trace_rejects_unsafe_file_hints(tmp_path):
    tracer = CodeTracer(_python_project(tmp_path))
    for unsafe in ("../outside.py", "/etc/passwd", ".env", "data/lewis.sqlite3"):
        outcome = tracer.trace("Store.save", file=unsafe)
        assert outcome.result.ok is False
        assert "not an analyzable path" in outcome.result.error
