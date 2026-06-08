from pathlib import Path

from app.code_proposal_patch_preview import generate_grounded_patch_preview
from app.project_search import ProjectFileSearch


def test_grounded_patch_preview_uses_only_a_verified_relevant_file(
    tmp_path: Path,
) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "web_components.py").write_text(
        "def command_center_chat():\n    return 'chat ui'\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "invented.py").write_text(
        "def unused():\n    return 'chat ui'\n",
        encoding="utf-8",
    )

    preview = generate_grounded_patch_preview(
        requested_goal="Add a small label to the command center chat UI.",
        relevant_files=["app/web_components.py"],
        search=ProjectFileSearch(tmp_path),
    )

    assert preview is not None
    assert preview.path == "app/web_components.py"
    assert "app/web_components.py" in preview.patch_preview
    assert "app/invented.py" not in preview.patch_preview


def test_grounded_patch_preview_is_omitted_when_change_is_not_confident(
    tmp_path: Path,
) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "web_components.py").write_text(
        "def command_center_chat():\n    return render_complex_chat()\n",
        encoding="utf-8",
    )

    preview = generate_grounded_patch_preview(
        requested_goal="Add a small label to the command center chat UI.",
        relevant_files=["app/web_components.py"],
        search=ProjectFileSearch(tmp_path),
    )

    assert preview is None


def test_grounded_patch_preview_uses_matching_bounded_chat_page_context(
    tmp_path: Path,
) -> None:
    (tmp_path / "app").mkdir()
    target = tmp_path / "app" / "web_pages.py"
    target.write_text(
        "\n".join(
            ["unrelated = True"] * 100
            + [
                "def chat_page():",
                "    content = [",
                "        \"<section class='chat-shell'>\",",
                "    ]",
            ]
        ),
        encoding="utf-8",
    )
    before = target.read_bytes()

    preview = generate_grounded_patch_preview(
        requested_goal="Add a small label to the command center chat UI.",
        relevant_files=["app/web_pages.py", "app/invented.py", ".env"],
        search=ProjectFileSearch(tmp_path),
    )

    assert preview is not None
    assert preview.path == "app/web_pages.py"
    assert "app/web_pages.py" in preview.patch_preview
    assert "Command Center Chat" in preview.patch_preview
    assert "app/invented.py" not in preview.patch_preview
    assert ".env" not in preview.patch_preview
    assert target.read_bytes() == before
