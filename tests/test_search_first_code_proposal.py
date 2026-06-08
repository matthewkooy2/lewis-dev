from pathlib import Path

from app.approvals import ApprovalStore
import app.search_first_code_proposal as search_first_code_proposal
from app.code_proposal_patch_preview import (
    PATCH_PREVIEW_UNAVAILABLE_NOTE,
    GroundedPatchPreview,
)
from app.project_search import ProjectFileSearch
from app.search_first_code_proposal import (
    create_search_first_code_proposal,
    extract_requested_goal,
)


def test_grounded_proposal_adds_preview_from_real_safe_matching_context(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    target = project_root / "app" / "web_pages.py"
    target.write_text(
        "def chat_page():\n"
        "    content = [\n"
        "        \"<section class='chat-shell'>\",\n"
        "    ]\n",
        encoding="utf-8",
    )
    before = target.read_bytes()
    store = ApprovalStore(tmp_path / "test.sqlite3")

    result = create_search_first_code_proposal(
        user_input=(
            "Make a code proposal to add a small label to the command center chat UI, "
            "using project search first."
        ),
        search=ProjectFileSearch(project_root),
        store=store,
    )

    assert result.approval is not None
    assert result.approval.status == "pending"
    assert "app/web_pages.py" in result.approval.payload["relevant_files"]
    assert "app/web_pages.py" in result.approval.payload["proposed_approach"]
    assert result.approval.payload["proposed_approach"] != (
        "Create a pending review-only plan. No files are modified and no commands are run."
    )
    assert "app/web_pages.py" in result.approval.payload["patch_preview"]
    assert "Command Center Chat" in result.approval.payload["patch_preview"]
    assert target.read_bytes() == before


def test_grounded_proposal_omits_unconfident_preview_and_keeps_review_planning(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "web_components.py").write_text(
        "def command_center_chat():\n    return render_complex_chat()\n",
        encoding="utf-8",
    )
    store = ApprovalStore(tmp_path / "test.sqlite3")

    result = create_search_first_code_proposal(
        user_input=(
            "Make a code proposal to add a small label to the command center chat UI, "
            "using project search first."
        ),
        search=ProjectFileSearch(project_root),
        store=store,
    )

    assert result.approval is not None
    assert result.approval.status == "pending"
    assert "patch_preview" not in result.approval.payload
    assert (
        PATCH_PREVIEW_UNAVAILABLE_NOTE in result.approval.payload["proposed_approach"]
    )
    assert PATCH_PREVIEW_UNAVAILABLE_NOTE in result.approval.payload["safety_notes"]


def test_search_first_flow_calls_grounded_preview_generator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    (project_root / "app").mkdir(parents=True)
    (project_root / "app" / "web_components.py").write_text(
        "def command_center_chat():\n    return 'chat ui'\n",
        encoding="utf-8",
    )
    store = ApprovalStore(tmp_path / "test.sqlite3")
    calls: list[dict[str, object]] = []

    def fake_generate_grounded_patch_preview(
        *,
        requested_goal: str,
        relevant_files: list[str],
        search: ProjectFileSearch,
    ) -> GroundedPatchPreview:
        calls.append(
            {
                "requested_goal": requested_goal,
                "relevant_files": list(relevant_files),
                "search": search,
            }
        )
        return GroundedPatchPreview(
            path="app/web_components.py",
            patch_preview=(
                "diff --git a/app/web_components.py b/app/web_components.py\n"
                "--- a/app/web_components.py\n"
                "+++ b/app/web_components.py\n"
                "@@ -2,1 +2,1 @@\n"
                "-    return 'chat ui'\n"
                "+    return 'Chat: chat ui'"
            ),
        )

    monkeypatch.setattr(
        search_first_code_proposal,
        "generate_grounded_patch_preview",
        fake_generate_grounded_patch_preview,
    )
    search = ProjectFileSearch(project_root)

    result = create_search_first_code_proposal(
        user_input=(
            "Make a code proposal to add a small label to the command center chat UI, "
            "using project search first."
        ),
        search=search,
        store=store,
    )

    assert result.approval is not None
    assert calls == [
        {
            "requested_goal": "Add a small label to the command center chat UI.",
            "relevant_files": ["app/web_components.py"],
            "search": search,
        }
    ]
    assert result.approval.payload["patch_preview"].startswith(
        "diff --git a/app/web_components.py"
    )


class TestExtractRequestedGoal:
    def test_cleans_i_want_you_to_create_code_proposal_that(self):
        goal = extract_requested_goal(
            "I want you to create a code proposal that makes the text blue "
            "for this chat instead of black"
        )
        assert goal == "Make the text blue for this chat instead of black."

    def test_cleans_i_want_you_to_create_code_proposal_that_animation(self):
        goal = extract_requested_goal(
            "I want you to create a code proposal that adds a new animation "
            'in this chat when you are "thinking"'
        )
        assert goal == "Add a new animation in this chat when Lewis is thinking."

    def test_cleans_this_time_create_proposal_involving(self):
        goal = extract_requested_goal(
            "this time I want you to create a proposal involving adding an "
            "animated character thinking, everytime that you are thinking"
        )
        assert (
            goal == "Add an animated character thinking every time Lewis is thinking."
        )

    def test_cleans_make_code_proposal_to_prefix(self):
        goal = extract_requested_goal(
            "Make a code proposal to add a small label to the command center chat UI, "
            "using project search first."
        )
        assert goal == "Add a small label to the command center chat UI."

    def test_cleans_create_code_proposal_that_prefix(self):
        goal = extract_requested_goal(
            "create a code proposal that changes the background color"
        )
        assert goal == "Change the background color."

    def test_preserves_plain_goal(self):
        goal = extract_requested_goal("Make the chat text blue instead of black.")
        assert goal == "Make the chat text blue instead of black."

    def test_empty_after_cleaning_returns_default(self):
        goal = extract_requested_goal("I want you to create a code proposal that")
        assert goal == "Create the requested code change."

    def test_makes_to_make(self):
        goal = extract_requested_goal(
            "I want you to create a code proposal that makes the button bigger"
        )
        assert goal == "Make the button bigger."

    def test_collapses_doubled_terminal_punctuation(self):
        goal = extract_requested_goal(
            "I want you to create a code proposal that makes the button bigger.."
        )
        assert goal == "Make the button bigger."
