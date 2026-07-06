from app.trace.js_analysis import analyze_js_file
from app.trace.js_lexer import lex
from app.trace.model import (
    KIND_COMPONENT,
    KIND_HOOK,
    LANGUAGE_JAVASCRIPT,
    LANGUAGE_TYPESCRIPT,
)
from app.trace.python_analysis import analyze_python_file


# ---- Python analyzer --------------------------------------------------------


def test_python_definitions_imports_exports_and_signatures():
    source = (
        "from app.tools import ToolRegistry as Registry\n"
        "import subprocess\n"
        "\n"
        "__all__ = [\"Agent\"]\n"
        "\n"
        "MAX_SIZE = 10\n"
        "\n"
        "class Agent:\n"
        "    def __init__(self, registry: Registry):\n"
        "        self.registry = registry\n"
        "\n"
        "    async def run(self, text: str) -> str:\n"
        "        subprocess.run([\"echo\"])\n"
        "        return self.registry.execute(text)\n"
        "\n"
        "def _private_helper():\n"
        "    pass\n"
    )
    index = analyze_python_file("app/agent.py", source)

    assert index.parse_ok is True
    qualnames = {d.qualname: d for d in index.definitions}
    assert qualnames["Agent"].kind == "class"
    assert qualnames["Agent.run"].kind == "async_method"
    assert qualnames["Agent.run"].signature == "(self, text: str) -> str"
    assert qualnames["MAX_SIZE"].kind == "constant"
    imports = {b.local_name: b for b in index.imports}
    assert imports["Registry"].source_module == "app.tools"
    assert imports["Registry"].source_name == "ToolRegistry"
    assert [e.exported_name for e in index.exports] == ["Agent"]
    assert index.attribute_types["Agent.registry"] == "Registry"


def test_python_calls_carry_receiver_types_await_and_dynamic_flags():
    source = (
        "import asyncio\n"
        "from app.store import Store\n"
        "\n"
        "async def main(name):\n"
        "    store = Store()\n"
        "    store.save(name)\n"
        "    await asyncio.sleep(1)\n"
        "    handler = getattr(store, name)\n"
        "    handler()\n"
    )
    index = analyze_python_file("app/main2.py", source)
    calls = {c.callee_text: c for c in index.calls}

    assert calls["store.save"].receiver_type == "Store"
    assert calls["asyncio.sleep"].is_await is True
    assert calls["getattr"].is_dynamic is True
    assert calls["handler"].callee_text == "handler"


def test_python_side_effects_error_handling_and_globals():
    source = (
        "import os\n"
        "import subprocess\n"
        "\n"
        "STATE = []\n"
        "\n"
        "def worker(path):\n"
        "    global STATE\n"
        "    os.environ[\"MODE\"] = \"on\"\n"
        "    subprocess.run([\"ls\"])\n"
        "    try:\n"
        "        open(path).read()\n"
        "    except Exception:\n"
        "        pass\n"
        "    raise ValueError(\"bad\")\n"
    )
    index = analyze_python_file("app/worker.py", source)
    categories = {(e.category, e.qualname) for e in index.side_effects}

    assert ("global_state", "worker") in categories
    assert ("environment", "worker") in categories
    assert ("process", "worker") in categories
    assert ("filesystem", "worker") in categories
    info = next(e for e in index.error_handling if e.qualname == "worker")
    assert info.raises == ("ValueError",)
    assert info.swallowed_catch_lines != ()
    assert info.catches == ("except Exception",)


def test_python_module_level_effects_have_no_qualname():
    source = "import subprocess\nsubprocess.run(['ls'])\n"
    index = analyze_python_file("app/boot.py", source)
    effect = next(e for e in index.side_effects if e.category == "process")
    assert effect.qualname is None


def test_python_syntax_error_is_reported_not_raised():
    index = analyze_python_file("app/broken.py", "def broken(:\n")
    assert index.parse_ok is False
    assert "line" in index.parse_error


def test_python_test_path_detection():
    index = analyze_python_file("tests/test_agent.py", "def test_x():\n    pass\n")
    assert index.is_test is True
    assert analyze_python_file("app/agent.py", "x = 1\n").is_test is False


def test_python_relative_imports_recorded_with_level_dots():
    source = "from ..pkg import helper\nfrom . import sibling\n"
    index = analyze_python_file("app/sub/mod.py", source)
    modules = {b.source_module for b in index.imports}
    assert "..pkg" in modules
    assert "." in modules


# ---- JS lexer ----------------------------------------------------------------


def test_lexer_handles_template_literals_regex_and_comments():
    source = (
        "// line comment\n"
        "const re = /ab\\/c/g;\n"
        "const t = `value ${compute(1)} end`;\n"
        "/* block\n comment */\n"
        "const division = a / b;\n"
    )
    result = lex(source, jsx_enabled=True)
    assert result.ok is True
    types = [token.type for token in result.tokens]
    assert "regex" in types
    assert "template_expr_open" in types and "template_expr_close" in types
    values = [token.value for token in result.tokens if token.type == "id"]
    assert "compute" in values


def test_lexer_jsx_elements_and_expression_islands():
    source = "const el = <Card size={2}><Inner on={go}/>text</Card>;"
    result = lex(source, jsx_enabled=True)
    assert result.ok is True
    opens = [t.value for t in result.tokens if t.type == "jsx_open"]
    assert opens == ["Card", "Inner"]
    closes = [t.value for t in result.tokens if t.type == "jsx_close"]
    assert closes == ["Inner", "Card"]


def test_lexer_backtracks_non_jsx_angle_brackets():
    source = "const ok = a < b; foo(x < y, z > w);"
    result = lex(source, jsx_enabled=True)
    assert result.ok is True
    assert not [t for t in result.tokens if t.type == "jsx_open"]


def test_lexer_reports_unclosed_template():
    result = lex("const t = `oops", jsx_enabled=False)
    assert result.ok is False
    assert "template" in result.error


# ---- JS/TS analyzer -----------------------------------------------------------


def test_js_definitions_components_hooks_and_exports():
    source = (
        "import React, { useState } from \"react\";\n"
        "export function useCounter() {\n"
        "  const [n, setN] = useState(0);\n"
        "  return n;\n"
        "}\n"
        "export default function Panel() {\n"
        "  const n = useCounter();\n"
        "  return <div>{n}</div>;\n"
        "}\n"
        "export const helper = (x) => x + 1;\n"
    )
    index = analyze_js_file("src/Panel.jsx", source, LANGUAGE_JAVASCRIPT)
    kinds = {d.qualname: d.kind for d in index.definitions}
    assert kinds["useCounter"] == KIND_HOOK
    assert kinds["Panel"] == KIND_COMPONENT
    assert kinds["helper"] == "function"
    exports = {(e.exported_name, e.kind) for e in index.exports}
    assert ("default", "default") in exports
    assert ("useCounter", "named") in exports


def test_js_class_methods_object_methods_and_end_lines():
    source = (
        "class Store {\n"
        "  constructor() { this.items = []; }\n"
        "  async load(path) {\n"
        "    return fetch(path);\n"
        "  }\n"
        "}\n"
        "const service = {\n"
        "  create(name) { return name; },\n"
        "  remove: (id) => id,\n"
        "};\n"
    )
    index = analyze_js_file("src/store.js", source, LANGUAGE_JAVASCRIPT)
    defs = {d.qualname: d for d in index.definitions}
    assert defs["Store.load"].kind == "async_method"
    assert defs["Store"].end_line == 6
    assert defs["service.create"].parent == "service"
    assert defs["service.remove"].parent == "service"


def test_js_imports_requires_reexports_and_commonjs_exports():
    source = (
        "import def, { a as b, c } from \"./mod\";\n"
        "import * as ns from \"../other\";\n"
        "const fs = require(\"fs\");\n"
        "const { readFile } = require(\"node:fs\");\n"
        "export { x as y } from \"./impl\";\n"
        "export * from \"./wide\";\n"
        "module.exports = { thing, alias: local };\n"
        "exports.extra = extraLocal;\n"
    )
    index = analyze_js_file("src/mod2.js", source, LANGUAGE_JAVASCRIPT)
    imports = {(b.local_name, b.source_module, b.kind) for b in index.imports}
    assert ("def", "./mod", "default") in imports
    assert ("b", "./mod", "named") in imports
    assert ("ns", "../other", "namespace") in imports
    assert ("fs", "fs", "require") in imports
    assert ("readFile", "node:fs", "require") in imports
    exports = {(e.exported_name, e.kind) for e in index.exports}
    assert ("y", "named") in exports
    assert ("*", "star") in exports
    assert ("thing", "commonjs") in exports
    assert ("alias", "commonjs") in exports
    assert ("extra", "commonjs") in exports


def test_js_calls_track_await_new_chains_and_keyword_properties():
    source = (
        "async function go(api) {\n"
        "  await api.delete(1);\n"
        "  const client = new HttpClient();\n"
        "  fetch(\"/x\").then((r) => r.json());\n"
        "}\n"
    )
    index = analyze_js_file("src/go.ts", source, LANGUAGE_TYPESCRIPT)
    calls = {c.callee_text: c for c in index.calls}
    assert calls["api.delete"].is_await is True
    assert calls["HttpClient"].is_new is True
    assert "(…).then" in calls
    assert calls["fetch"].caller_qualname == "go"


def test_js_side_effects_and_error_handling():
    source = (
        "const fs = require(\"fs\");\n"
        "function work(path) {\n"
        "  fs.writeFileSync(path, \"data\");\n"
        "  localStorage.setItem(\"k\", \"v\");\n"
        "  console.log(\"done\");\n"
        "  try {\n"
        "    risky();\n"
        "  } catch (err) {\n"
        "  }\n"
        "  throw new CustomError(\"boom\");\n"
        "}\n"
    )
    index = analyze_js_file("src/work.js", source, LANGUAGE_JAVASCRIPT)
    categories = {(e.category, e.qualname) for e in index.side_effects}
    assert ("filesystem", "work") in categories
    assert ("dom_or_browser_storage", "work") in categories
    assert ("console_logging", "work") in categories
    info = next(e for e in index.error_handling if e.qualname == "work")
    assert "CustomError" in info.raises
    assert info.swallowed_catch_lines != ()


def test_js_jsx_component_usage_and_callback_references():
    source = (
        "import { Button } from \"./Button\";\n"
        "export default function Bar({ onSave }) {\n"
        "  const handle = () => onSave(1);\n"
        "  return <Button onClick={handle} label=\"go\" />;\n"
        "}\n"
    )
    index = analyze_js_file("src/Bar.jsx", source, LANGUAGE_JAVASCRIPT)
    jsx = [c for c in index.calls if c.jsx_component]
    assert [c.head for c in jsx] == ["Button"]
    callbacks = [c for c in index.calls if c.is_callback_reference]
    assert [c.head for c in callbacks] == ["handle"]


def test_ts_interfaces_types_do_not_create_fake_calls():
    source = (
        "export interface Props {\n"
        "  save(): void;\n"
        "  name: string;\n"
        "}\n"
        "export type Handler = (x: number) => void;\n"
        "export function real(): void {\n"
        "  actual();\n"
        "}\n"
    )
    index = analyze_js_file("src/types.ts", source, LANGUAGE_TYPESCRIPT)
    call_texts = {c.callee_text for c in index.calls}
    assert "actual" in call_texts
    assert "save" not in call_texts
    kinds = {d.qualname: d.kind for d in index.definitions}
    assert kinds["Props"] == "type"
    assert kinds["Handler"] == "type"


def test_js_test_path_detection():
    assert analyze_js_file(
        "src/__tests__/a.js", "test('x', () => {});", LANGUAGE_JAVASCRIPT
    ).is_test
    assert analyze_js_file(
        "src/App.test.tsx", "it('x', () => {});", LANGUAGE_TYPESCRIPT
    ).is_test
    assert not analyze_js_file("src/App.tsx", "x", LANGUAGE_TYPESCRIPT).is_test
