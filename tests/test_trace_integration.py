import json
from pathlib import Path

from app.approvals import ApprovalStore
from app.config import get_settings
from app.llm import build_system_prompt
from app.main import main
from app.rag import DocumentSearch
from app.tools import ToolRegistry
from app.trace.service import CodeTracer


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    _write(
        root,
        "app/calc.py",
        "def add(a, b):\n"
        "    return a + b\n"
        "\n"
        "def total(values):\n"
        "    result = 0\n"
        "    for value in values:\n"
        "        result = add(result, value)\n"
        "    return result\n",
    )
    _write(
        root,
        "tests/test_calc.py",
        "from app.calc import total\n"
        "\n"
        "def test_total():\n"
        "    assert total([1, 2]) == 3\n",
    )
    return root


def _tools(tmp_path: Path, *, developer: bool) -> ToolRegistry:
    return ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
        developer_code_workspace=developer,
        code_tracer=CodeTracer(_project(tmp_path)),
    )


def test_trace_tool_is_gated_from_normal_mode(tmp_path):
    tools = _tools(tmp_path, developer=False)
    names = {schema["function"]["name"] for schema in tools.schemas()}
    assert "trace_symbol" not in names
    assert tools.execute("trace_symbol", {"symbol": "total"}) == {
        "ok": False,
        "error": "Unknown tool: trace_symbol",
    }


def test_trace_tool_schema_and_quick_trace_in_developer_mode(tmp_path):
    tools = _tools(tmp_path, developer=True)
    schema = next(
        item["function"]
        for item in tools.schemas()
        if item["function"]["name"] == "trace_symbol"
    )
    assert schema["parameters"]["required"] == ["symbol"]
    assert set(schema["parameters"]["properties"]) == {"symbol", "file", "mode"}
    assert schema["parameters"]["additionalProperties"] is False

    result = tools.execute("trace_symbol", {"symbol": "total"})
    assert result["ok"] is True
    assert result["mode"] == "quick"
    assert "QUICK TRACE: total" in result["report"]
    assert "app/calc.py" in result["report"]
    assert result["data"]["target"]["qualname"] == "total"
    assert result["data"]["schema"] == "lewis-trace-compact/1"
    callee = result["data"]["callees"][0]
    assert callee["target"] == "add"
    assert callee["resolution"] == "confirmed"


def test_trace_tool_rejects_bad_arguments_and_unknown_symbols(tmp_path):
    tools = _tools(tmp_path, developer=True)
    assert tools.execute("trace_symbol", {})["ok"] is False
    assert (
        tools.execute("trace_symbol", {"symbol": "total", "write": "x"})["ok"] is False
    )
    missing = tools.execute("trace_symbol", {"symbol": "does_not_exist"})
    assert missing["ok"] is False
    assert "No definition found" in missing["error"]
    bad_mode = tools.execute("trace_symbol", {"symbol": "total", "mode": "explode"})
    assert bad_mode["ok"] is False
    assert "Unknown mode" in bad_mode["error"]


def test_trace_tool_review_mode_reports_findings(tmp_path):
    root = _project(tmp_path)
    _write(
        root,
        "app/risky.py",
        "def load(path):\n"
        "    try:\n"
        "        return open(path).read()\n"
        "    except:\n"
        "        pass\n",
    )
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        ApprovalStore(tmp_path / "test.sqlite3"),
        developer_code_workspace=True,
        code_tracer=CodeTracer(root),
    )
    result = tools.execute("trace_symbol", {"symbol": "load", "mode": "review"})
    assert result["ok"] is True
    assert "REVIEW" in result["report"]
    assert any(
        finding["category"] == "error_handling"
        for finding in result["data"]["findings"]
    )


def test_trace_tool_never_modifies_project_or_creates_approvals(tmp_path):
    root = _project(tmp_path)
    store = ApprovalStore(tmp_path / "test.sqlite3")
    tools = ToolRegistry(
        DocumentSearch(tmp_path / "documents"),
        store,
        developer_code_workspace=True,
        code_tracer=CodeTracer(root),
    )
    before = (root / "app" / "calc.py").read_text(encoding="utf-8")
    tools.execute("trace_symbol", {"symbol": "total", "mode": "review"})
    assert (root / "app" / "calc.py").read_text(encoding="utf-8") == before
    assert store.list() == []


def test_system_prompt_mentions_trace_only_in_developer_mode(monkeypatch):
    monkeypatch.setenv("LEWIS_DEVELOPER_CODE_WORKSPACE", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    prompt = build_system_prompt(get_settings())
    assert "trace_symbol" in prompt
    monkeypatch.setenv("LEWIS_DEVELOPER_CODE_WORKSPACE", "0")
    prompt = build_system_prompt(get_settings())
    assert "trace_symbol" not in prompt


# ---- CLI ----------------------------------------------------------------------


def test_cli_trace_prints_report(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    monkeypatch.setattr("app.main.ROOT_DIR", root)
    exit_code = main(["trace", "total"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "QUICK TRACE: total" in output
    assert "app/calc.py" in output
    assert "tests/test_calc.py" in output


def test_cli_trace_json_output_is_parseable(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    monkeypatch.setattr("app.main.ROOT_DIR", root)
    exit_code = main(["trace", "total", "--mode", "deep", "--json"])
    output = capsys.readouterr().out
    assert exit_code == 0
    payload = json.loads(output)
    assert payload["schema"] == "lewis-trace-result/1"
    assert payload["mode"] == "deep"
    assert payload["target"]["qualname"] == "total"


def test_cli_trace_unknown_symbol_exits_nonzero(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    monkeypatch.setattr("app.main.ROOT_DIR", root)
    exit_code = main(["trace", "nope_not_here"])
    output = capsys.readouterr().out
    assert exit_code == 1
    assert "TRACE ERROR" in output


def test_cli_trace_ambiguous_symbol_lists_candidates(tmp_path, monkeypatch, capsys):
    root = _project(tmp_path)
    _write(root, "app/more.py", "def total(x):\n    return x\n")
    monkeypatch.setattr("app.main.ROOT_DIR", root)
    exit_code = main(["trace", "total"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "AMBIGUOUS SYMBOL" in output
    assert "app/calc.py" in output and "app/more.py" in output

    exit_code = main(["trace", "total", "--file", "app/more.py"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "app/more.py" in output
