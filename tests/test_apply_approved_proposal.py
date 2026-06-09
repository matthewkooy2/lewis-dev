"""Stage 7C — Tests for the apply-only local branch applicator.

All git operations run inside temporary repositories under tmp_path. The
real Lewis repository state is never touched. No tests run pytest/ruff via
the applicator, and no payload commands are ever executed.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

import app.dev.apply_approved_proposal as applicator
from app.approvals import ApprovalStore
from app.dev.apply_approved_proposal import (
    ApplyProposalResult,
    ApplyRefusedError,
    apply_approved_proposal,
    format_report,
)

CSS_REL_PATH = "app/static/lewis.css"
CSS_BEFORE = ".chat-log .bubble {\n    color: var(--ink);\n}\n"
PATCH = (
    "diff --git a/app/static/lewis.css b/app/static/lewis.css\n"
    "--- a/app/static/lewis.css\n"
    "+++ b/app/static/lewis.css\n"
    "@@ -1,3 +1,3 @@\n"
    " .chat-log .bubble {\n"
    "-    color: var(--ink);\n"
    "+    color: blue;\n"
    " }\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=True,
    )
    return completed.stdout.decode("utf-8", errors="replace").strip()


def _init_repo(tmp_path: Path, *, css_content: str = CSS_BEFORE) -> Path:
    root = tmp_path / "repo"
    (root / "app" / "static").mkdir(parents=True)
    (root / "app" / "static" / "lewis.css").write_bytes(css_content.encode("utf-8"))
    (root / "README.md").write_bytes(b"readme\n")
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "core.autocrlf", "false")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "init")
    return root


def _payload(**overrides) -> dict:
    payload = {
        "applicator_contract_version": 1,
        "requested_goal": "Make the chat text blue.",
        "target_surface": "command center web UI",
        "relevant_files": [CSS_REL_PATH],
        "proposed_approach": "Change the CSS color property.",
        "patch_preview": PATCH,
        "patch_preview_implementation_ready": True,
        "patch_preview_kind": "unified diff",
        "patch_is_inert": True,
        "touched_paths": [CSS_REL_PATH],
        "selected_strategy": "chat_text_color",
        "commands_run": [],
        "files_modified": [],
        "safety_notes": ["Pending code proposal only."],
        "suggested_test_commands": [],
    }
    payload.update(overrides)
    return payload


def _store(tmp_path: Path) -> ApprovalStore:
    return ApprovalStore(tmp_path / "approvals.sqlite3")


def _approved(store: ApprovalStore, payload: dict) -> int:
    item = store.create("propose_code_change", "Test proposal", payload)
    store.update_status(item.id, "approved")
    return item.id


def _make_symlink(link: Path, target: Path, *, is_dir: bool) -> None:
    try:
        os.symlink(target, link, target_is_directory=is_dir)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is not supported in this environment")


# ---------------------------------------------------------------------------
# Approval envelope and validator refusals
# ---------------------------------------------------------------------------


def test_refuses_non_approved_approval(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    item = store.create("propose_code_change", "Test proposal", _payload())
    with pytest.raises(ApplyRefusedError, match="expected 'approved'"):
        apply_approved_proposal(item.id, store=store, project_root=root)


def test_refuses_missing_approval(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    with pytest.raises(ApplyRefusedError, match="not found"):
        apply_approved_proposal(9999, store=store, project_root=root)


def test_refuses_invalid_validator_payload(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload(patch_preview_kind="unified_diff"))
    with pytest.raises(ApplyRefusedError, match="Validation failed"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


# ---------------------------------------------------------------------------
# Dirty working tree refusals (tracked, staged, untracked)
# ---------------------------------------------------------------------------


def test_refuses_dirty_tree_with_tracked_change(tmp_path):
    root = _init_repo(tmp_path)
    (root / "README.md").write_bytes(b"readme modified\n")
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    with pytest.raises(ApplyRefusedError, match="not clean"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


def test_refuses_dirty_tree_with_staged_change(tmp_path):
    root = _init_repo(tmp_path)
    (root / "README.md").write_bytes(b"readme staged\n")
    _git(root, "add", "README.md")
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    with pytest.raises(ApplyRefusedError, match="not clean"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


def test_refuses_dirty_tree_with_untracked_file(tmp_path):
    root = _init_repo(tmp_path)
    (root / "untracked.txt").write_bytes(b"new\n")
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    with pytest.raises(ApplyRefusedError, match="not clean"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


# ---------------------------------------------------------------------------
# Branch rules
# ---------------------------------------------------------------------------


def test_refuses_branch_name_collision(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    _git(root, "branch", f"lewis/proposal-{approval_id}")
    with pytest.raises(ApplyRefusedError, match="already exists"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


def test_branch_name_ignores_selected_strategy(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(
        store, _payload(selected_strategy="evil; rm -rf / #strategy")
    )
    result = apply_approved_proposal(approval_id, store=store, project_root=root)
    assert result.branch_name == f"lewis/proposal-{approval_id}"


# ---------------------------------------------------------------------------
# Filesystem preflight refusals
# ---------------------------------------------------------------------------


def test_refuses_symlink_parent_path(tmp_path):
    root = _init_repo(tmp_path)
    real_dir = root / "static_real"
    real_dir.mkdir()
    (real_dir / "lewis.css").write_bytes(CSS_BEFORE.encode("utf-8"))
    shutil.rmtree(root / "app" / "static")
    _make_symlink(root / "app" / "static", real_dir, is_dir=True)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    with pytest.raises(ApplyRefusedError, match="symlink"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


def test_refuses_symlink_target_path(tmp_path):
    root = _init_repo(tmp_path)
    real_file = root / "other.css"
    real_file.write_bytes(CSS_BEFORE.encode("utf-8"))
    (root / "app" / "static" / "lewis.css").unlink()
    _make_symlink(root / "app" / "static" / "lewis.css", real_file, is_dir=False)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    with pytest.raises(ApplyRefusedError, match="symlink"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


def test_refuses_symlink_parent_component_without_real_symlinks(tmp_path, monkeypatch):
    # Real symlink creation needs privileges Windows may not grant, so this
    # exercises the same preflight refusal by faking Path.is_symlink.
    (tmp_path / "app" / "static").mkdir(parents=True)
    (tmp_path / "app" / "static" / "lewis.css").write_bytes(b"body {}\n")
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "static")
    with pytest.raises(ApplyRefusedError, match="symlink"):
        applicator._filesystem_preflight((CSS_REL_PATH,), tmp_path)


def test_refuses_symlink_target_component_without_real_symlinks(tmp_path, monkeypatch):
    (tmp_path / "app" / "static").mkdir(parents=True)
    (tmp_path / "app" / "static" / "lewis.css").write_bytes(b"body {}\n")
    monkeypatch.setattr(Path, "is_symlink", lambda self: self.name == "lewis.css")
    with pytest.raises(ApplyRefusedError, match="symlink"):
        applicator._filesystem_preflight((CSS_REL_PATH,), tmp_path)


def test_refuses_missing_existing_target_file(tmp_path):
    root = _init_repo(tmp_path)
    missing_path = "app/static/missing.css"
    patch = PATCH.replace(CSS_REL_PATH, missing_path)
    store = _store(tmp_path)
    approval_id = _approved(
        store, _payload(patch_preview=patch, touched_paths=[missing_path])
    )
    with pytest.raises(ApplyRefusedError, match="does not exist"):
        apply_approved_proposal(approval_id, store=store, project_root=root)
    assert _git(root, "branch", "--list", f"lewis/proposal-{approval_id}") == ""


def test_refuses_forbidden_runtime_path_in_preflight(tmp_path):
    # Defense in depth: the preflight re-checks forbidden paths against the
    # actual project root even though the validator already refuses them.
    with pytest.raises(ApplyRefusedError, match="blocked runtime path"):
        applicator._filesystem_preflight(("data/lewis.css",), tmp_path)


# ---------------------------------------------------------------------------
# Base-commit binding
# ---------------------------------------------------------------------------


def test_reports_no_base_binding_when_metadata_absent(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    result = apply_approved_proposal(approval_id, store=store, project_root=root)
    assert "binding not enforced" in result.base_binding


def test_binds_to_matching_base_commit(tmp_path):
    root = _init_repo(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    store = _store(tmp_path)
    approval_id = _approved(store, _payload(patch_base_commit=head))
    result = apply_approved_proposal(approval_id, store=store, project_root=root)
    assert head in result.base_binding
    assert result.base_commit == head


def test_refuses_mismatched_base_commit(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload(patch_base_commit="0" * 40))
    with pytest.raises(ApplyRefusedError, match="different base"):
        apply_approved_proposal(approval_id, store=store, project_root=root)
    assert _git(root, "branch", "--list", f"lewis/proposal-{approval_id}") == ""


# ---------------------------------------------------------------------------
# Successful application
# ---------------------------------------------------------------------------


def test_applies_valid_single_file_patch_on_new_branch(tmp_path):
    root = _init_repo(tmp_path)
    head = _git(root, "rev-parse", "HEAD")
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())

    result = apply_approved_proposal(approval_id, store=store, project_root=root)

    branch = f"lewis/proposal-{approval_id}"
    assert result.branch_name == branch
    assert result.original_branch == "main"
    assert result.base_commit == head
    assert result.touched_paths == (CSS_REL_PATH,)
    assert result.tests_run == "none"
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == branch
    content = (root / CSS_REL_PATH).read_text(encoding="utf-8")
    assert "color: blue;" in content
    report = format_report(result)
    assert branch in report
    assert "tests run:          none" in report


def test_verifies_only_declared_path_changed(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    result = apply_approved_proposal(approval_id, store=store, project_root=root)
    assert result.changed_files == (CSS_REL_PATH,)
    assert (root / "README.md").read_bytes() == b"readme\n"


def test_refuses_when_post_apply_includes_undeclared_file(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())

    original_run_git = applicator._run_git

    def sneaky_run_git(args, run_root, *, input_bytes=None):
        result = original_run_git(args, run_root, input_bytes=input_bytes)
        if tuple(args) == ("git", "apply"):
            (root / "sneaky.txt").write_bytes(b"oops\n")
        return result

    monkeypatch.setattr(applicator, "_run_git", sneaky_run_git)
    with pytest.raises(ApplyRefusedError, match="Post-apply verification failed"):
        apply_approved_proposal(approval_id, store=store, project_root=root)
    # Branch and working tree are left in place for inspection, not deleted.
    branch = f"lewis/proposal-{approval_id}"
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == branch


def test_refuses_when_touched_file_left_empty_after_apply(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())

    original_run_git = applicator._run_git

    def emptying_run_git(args, run_root, *, input_bytes=None):
        result = original_run_git(args, run_root, input_bytes=input_bytes)
        if tuple(args) == ("git", "apply"):
            (root / CSS_REL_PATH).write_bytes(b"")
        return result

    monkeypatch.setattr(applicator, "_run_git", emptying_run_git)
    with pytest.raises(ApplyRefusedError, match="effectively deleted"):
        apply_approved_proposal(approval_id, store=store, project_root=root)


def test_apply_check_failure_returns_to_original_branch(tmp_path):
    # Repo content does not match the patch context, so git apply --check
    # fails after branch creation; the tree is untouched, we return to the
    # original branch, and the proposal branch is left for manual cleanup.
    root = _init_repo(tmp_path, css_content=".chat-log .bubble {\n    color: red;\n}\n")
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    with pytest.raises(ApplyRefusedError, match="git apply --check failed"):
        apply_approved_proposal(approval_id, store=store, project_root=root)
    assert _git(root, "rev-parse", "--abbrev-ref", "HEAD") == "main"
    branch = f"lewis/proposal-{approval_id}"
    assert branch in _git(root, "branch", "--list", branch)


# ---------------------------------------------------------------------------
# No tests run, no payload commands executed, no approval mutation
# ---------------------------------------------------------------------------


def test_runs_only_git_commands_and_never_payload_commands(tmp_path, monkeypatch):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(
        store,
        _payload(
            commands_run=["echo HACKED-RUN"],
            suggested_test_commands=["echo HACKED-SUGGESTED", "pytest -q"],
        ),
    )

    recorded: list[tuple[list[str], dict]] = []
    real_run = subprocess.run

    def recording_run(args, **kwargs):
        recorded.append((list(args), kwargs))
        return real_run(args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording_run)
    apply_approved_proposal(approval_id, store=store, project_root=root)

    assert recorded, "expected git invocations to be recorded"
    for args, kwargs in recorded:
        assert args[0] == "git"
        assert kwargs.get("shell") in (None, False)
        joined = " ".join(args)
        assert "echo" not in joined
        assert "HACKED" not in joined
        assert "pytest" not in joined
        assert "ruff" not in joined


def test_does_not_mutate_approval_status(tmp_path):
    root = _init_repo(tmp_path)
    store = _store(tmp_path)
    approval_id = _approved(store, _payload())
    before = store.get(approval_id)
    apply_approved_proposal(approval_id, store=store, project_root=root)
    after = store.get(approval_id)
    assert after.status == "approved"
    assert after.payload == before.payload
    assert after.updated_at == before.updated_at


# ---------------------------------------------------------------------------
# Runtime isolation
# ---------------------------------------------------------------------------


def test_no_chat_runtime_imports_in_applicator_module():
    """The applicator must not import chat/session/runtime modules."""
    source = applicator.__file__
    with open(source, encoding="utf-8") as f:
        content = f.read()
    forbidden_imports = [
        "from app.llm",
        "from app.tools",
        "from app.main",
        "from app.web_routes",
        "from app.web_pages",
        "from app.web_components",
        "from app.dashboard",
        "from app.executors",
    ]
    for forbidden in forbidden_imports:
        assert forbidden not in content, f"Applicator imports {forbidden}"


def test_no_runtime_module_imports_dev_applicator():
    """No module outside app/dev may import the dev applicator package."""
    app_dir = Path(applicator.__file__).resolve().parents[1]
    for module_path in app_dir.rglob("*.py"):
        relative = module_path.relative_to(app_dir)
        if relative.parts[0] == "dev":
            continue
        content = module_path.read_text(encoding="utf-8")
        assert "app.dev" not in content, f"{relative} references app.dev"
        assert "apply_approved_proposal" not in content, (
            f"{relative} references the dev applicator"
        )


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def test_cli_prints_refusal_and_returns_one(monkeypatch, capsys):
    def refusing(approval_id):
        raise ApplyRefusedError("nope")

    monkeypatch.setattr(applicator, "apply_approved_proposal", refusing)
    assert applicator.main(["5"]) == 1
    assert "REFUSED: nope" in capsys.readouterr().out


def test_cli_prints_report_and_returns_zero(monkeypatch, capsys):
    result = ApplyProposalResult(
        approval_id=7,
        branch_name="lewis/proposal-7",
        original_branch="main",
        base_commit="a" * 40,
        base_binding="no base-commit metadata in payload; binding not enforced",
        touched_paths=(CSS_REL_PATH,),
        validation_result="passed (revalidated immediately before apply)",
        apply_result="applied cleanly (git apply --check, then git apply)",
        changed_files=(CSS_REL_PATH,),
        tests_run="none",
        next_steps=("git diff",),
    )
    monkeypatch.setattr(applicator, "apply_approved_proposal", lambda i: result)
    assert applicator.main(["7"]) == 0
    out = capsys.readouterr().out
    assert "lewis/proposal-7" in out
    assert "tests run:          none" in out
