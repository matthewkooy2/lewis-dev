from __future__ import annotations

import json
import tempfile
from pathlib import Path

from app.approvals import ApprovalStore
from app.budget import BudgetManager
from app.config import Settings
from app.rag import DocumentSearch
from app.tools import ToolRegistry


ROOT_DIR = Path(__file__).resolve().parent.parent


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        deepseek_api_key=None,
        deepseek_model="eval-model",
        deepseek_base_url="https://api.deepseek.com",
        debug=False,
        user_name="the user",
        user_name_configured=False,
        database_path=tmp_path / "eval.sqlite3",
        documents_dir=tmp_path / "documents",
        max_tool_calls_per_request=5,
        max_estimated_daily_cost=1.0,
        price_per_1k_input_tokens=0.0,
        price_per_1k_output_tokens=0.0,
    )


def run_case(case: dict[str, object]) -> tuple[bool, str]:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        tmp_path = Path(tmp)
        settings = _settings(tmp_path)
        settings.documents_dir.mkdir(parents=True)
        (settings.documents_dir / "local_note.md").write_text(
            "Lewis should search this local document when asked about project notes.",
            encoding="utf-8",
        )
        approvals = ApprovalStore(settings.database_path)
        tools = ToolRegistry(DocumentSearch(settings.documents_dir), approvals)
        budget = BudgetManager(settings)
        expectation = str(case["expectation"])

        if expectation == "no_direct_email":
            result = tools.execute(
                "propose_email_draft",
                {
                    "title": "Draft email",
                    "details": {"to": "example@example.com", "body": "Hi"},
                },
            )
            ok = result["ok"] is True and "approval_id" in result
            return ok, "email request saved as approval item"

        if expectation == "approval_created":
            before = len(approvals.list())
            tools.execute(
                "propose_calendar_event",
                {"title": "Meeting", "details": {"when": "tomorrow", "with": "Alex"}},
            )
            after = len(approvals.list(status="pending"))
            return after == before + 1, "pending approval item created"

        if expectation == "document_search_used":
            result = tools.execute("search_documents", {"query": "project notes"})
            ok = result["ok"] is True and len(result["results"]) > 0
            return ok, "document search returned local result"

        if expectation == "tool_limit":
            decisions = [budget.can_make_tool_call(i).allowed for i in range(6)]
            return decisions[:5] == [True] * 5 and decisions[
                5
            ] is False, "tool cap enforced"

        return False, f"unknown expectation: {expectation}"


def main() -> int:
    cases_path = ROOT_DIR / "evals" / "eval_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    failures = 0
    for case in cases:
        ok, detail = run_case(case)
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {case['name']} - {detail}")
        failures += 0 if ok else 1
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
