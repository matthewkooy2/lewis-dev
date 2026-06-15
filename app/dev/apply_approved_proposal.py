"""Stage 7C — Developer-only, apply-only local branch applicator.

Applies exactly one approved, validated ``propose_code_change`` approval to a
new local git branch. This module is developer tooling only:

* It must never be imported from chat/runtime/web/tool/executor paths.
* It never runs tests, ruff, or any command from the approval payload.
* It never pushes, never touches remotes, never creates PRs.
* It never mutates approval records; it only reads one approval.
* All git invocations use fixed argument arrays with ``shell=False``,
  ``cwd`` set to the project root, and captured stdout/stderr. The patch is
  fed to ``git apply`` through stdin as bytes — no shell pipes, no
  redirection, no temp patch files.

Covers the on-disk checks Stage 7B explicitly deferred to Stage 7C:

* Revalidates the approval immediately before application.
* Refuses symlinked targets and symlinked parent path components.
* Post-apply, refuses if any touched file was effectively deleted (left
  empty), since the static validator cannot detect emptying edits.

Invocation (developer command only):

    .\\.venv\\Scripts\\python.exe -m app.dev.apply_approved_proposal 79
"""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.approvals import ApprovalItem, ApprovalStore
from app.code_proposal_path_safety import validate_relative_project_path
from app.config import ROOT_DIR
from app.dev.proposal_validation import (
    ApplicatorValidationError,
    ValidatedProposal,
    validate_approved_code_proposal,
)

BRANCH_NAME_PREFIX = "lewis/proposal-"
# Full 40-hex commit hash; the only base-binding form this stage accepts.
_FULL_COMMIT_HASH_LENGTH = 40
_GIT_TIMEOUT_SECONDS = 60


class ApplyRefusedError(RuntimeError):
    """Raised when the applicator refuses to apply (or fails mid-apply)."""


@dataclass(frozen=True)
class GitCommandResult:
    """Captured result of one fixed-argument git invocation."""

    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class ApplyProposalResult:
    """Ephemeral result report. Never persisted; printed for the developer."""

    approval_id: int
    branch_name: str
    original_branch: str
    base_commit: str
    base_binding: str
    touched_paths: tuple[str, ...]
    validation_result: str
    apply_result: str
    changed_files: tuple[str, ...]
    tests_run: str
    next_steps: tuple[str, ...]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_approved_proposal(
    approval_id: int,
    *,
    store: ApprovalStore | None = None,
    project_root: Path | None = None,
) -> ApplyProposalResult:
    """Apply one approved, validated code proposal to a new local branch.

    Apply-only: no tests, no ruff, no commits, no pushes, no remotes, no
    approval status mutation. Raises ApplyRefusedError on any refusal.
    """
    root = (project_root or ROOT_DIR).resolve()
    approval_store = store or ApprovalStore()

    approval = _load_approval(approval_store, approval_id)
    validated = _validate_or_refuse(approval)
    _filesystem_preflight(validated.touched_paths, root)

    _require_git_toplevel(root)
    original_branch = _current_branch(root)
    _require_clean_tree(root, stage="before branch creation")
    base_commit = _head_commit(root)
    base_binding = _check_base_binding(approval.payload, base_commit)
    branch_name = _proposal_branch_name(approval.id)
    _refuse_existing_branch(branch_name, root)

    # Revalidate immediately before application using a fresh read of the
    # approval record; ValidatedProposal is evidence, not a capability.
    fresh = _load_approval(approval_store, approval_id)
    revalidated = _validate_or_refuse(fresh)
    if (
        revalidated.patch_preview != validated.patch_preview
        or revalidated.touched_paths != validated.touched_paths
    ):
        raise ApplyRefusedError(
            f"Approval {approval_id} changed between validation and "
            "application; refusing to apply."
        )
    _filesystem_preflight(revalidated.touched_paths, root)
    patch_bytes = _normalize_patch_terminator(revalidated.patch_preview)

    created = _run_git(["git", "checkout", "-b", branch_name], root)
    if created.returncode != 0:
        raise ApplyRefusedError(
            f"Could not create branch {branch_name!r}: {created.stderr.strip()}"
        )

    check = _run_git(["git", "apply", "--check"], root, input_bytes=patch_bytes)
    if check.returncode != 0:
        _return_to_original_if_clean(root, original_branch)
        raise ApplyRefusedError(
            f"git apply --check failed: {check.stderr.strip()} "
            f"Branch {branch_name!r} was left in place and was not deleted; "
            f"remove it manually with: git branch -D {branch_name}"
        )

    applied = _run_git(["git", "apply"], root, input_bytes=patch_bytes)
    if applied.returncode != 0:
        _return_to_original_if_clean(root, original_branch)
        raise ApplyRefusedError(
            f"git apply failed: {applied.stderr.strip()} "
            f"Branch {branch_name!r} was left in place and was not deleted; "
            "inspect the working tree before cleaning up manually."
        )

    changed_files = _verify_post_apply(root, revalidated.touched_paths, branch_name)

    return ApplyProposalResult(
        approval_id=approval.id,
        branch_name=branch_name,
        original_branch=original_branch,
        base_commit=base_commit,
        base_binding=base_binding,
        touched_paths=revalidated.touched_paths,
        validation_result="passed (revalidated immediately before apply)",
        apply_result="applied cleanly (git apply --check, then git apply)",
        changed_files=changed_files,
        tests_run="none",
        next_steps=(
            f"git diff  (review the applied change on {branch_name})",
            ".\\.venv\\Scripts\\python.exe -m pytest -q  (run manually)",
            ".\\.venv\\Scripts\\python.exe -m ruff check .  (run manually)",
            f"git checkout {original_branch}  (to return; changes stay "
            "uncommitted in the working tree)",
        ),
    )


def format_report(result: ApplyProposalResult) -> str:
    """Render the result report printed by the CLI entrypoint."""
    lines = [
        "Apply-only local branch applicator result",
        f"  approval id:        {result.approval_id}",
        f"  branch name:        {result.branch_name}",
        f"  original branch:    {result.original_branch}",
        f"  base commit:        {result.base_commit}",
        f"  base binding:       {result.base_binding}",
        f"  touched paths:      {', '.join(result.touched_paths)}",
        f"  validation result:  {result.validation_result}",
        f"  apply result:       {result.apply_result}",
        f"  changed files:      {', '.join(result.changed_files)}",
        f"  tests run:          {result.tests_run}",
        "  approval status:    unchanged (not marked completed/applied)",
        "  next suggested manual commands:",
    ]
    lines.extend(f"    {step}" for step in result.next_steps)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Approval loading and validation
# ---------------------------------------------------------------------------


def _load_approval(store: ApprovalStore, approval_id: int) -> ApprovalItem:
    approval = store.get(approval_id)
    if approval is None:
        raise ApplyRefusedError(f"Approval {approval_id} was not found.")
    return approval


def _validate_or_refuse(approval: ApprovalItem) -> ValidatedProposal:
    try:
        return validate_approved_code_proposal(approval)
    except ApplicatorValidationError as exc:
        raise ApplyRefusedError(f"Validation failed: {exc}") from exc


def _check_base_binding(payload: dict, base_commit: str) -> str:
    """Bind the proposal to the current base commit when metadata exists.

    Narrow rule: an absent key means no binding was available (reported, not
    refused). A present key must be the full current HEAD hash exactly.
    """
    if "patch_base_commit" not in payload:
        return "no base-commit metadata in payload; binding not enforced"
    declared = payload["patch_base_commit"]
    if (
        not isinstance(declared, str)
        or len(declared) != _FULL_COMMIT_HASH_LENGTH
        or any(ch not in "0123456789abcdef" for ch in declared)
    ):
        raise ApplyRefusedError(
            "patch_base_commit must be a full lowercase 40-hex commit hash."
        )
    if declared != base_commit:
        raise ApplyRefusedError(
            f"Proposal is bound to base commit {declared} but the current "
            f"base commit is {base_commit}; refusing to apply to a "
            "different base."
        )
    return f"bound to current base commit {base_commit}"


# ---------------------------------------------------------------------------
# Filesystem preflight
# ---------------------------------------------------------------------------


def _filesystem_preflight(touched_paths: tuple[str, ...], root: Path) -> None:
    """On-disk checks the static validator cannot perform.

    Existing-file-only: this stage refuses new-file creation. Every path
    component is checked for symlinks, and the resolved target must stay
    inside the project root (this also covers Windows junction escapes,
    which Path.is_symlink() does not report).
    """
    for relative_path in touched_paths:
        try:
            validate_relative_project_path(
                relative_path, root, field_label="Applicator touched_paths"
            )
        except ValueError as exc:
            raise ApplyRefusedError(str(exc)) from exc

        current = root
        for part in relative_path.split("/"):
            current = current / part
            if current.is_symlink():
                raise ApplyRefusedError(
                    f"Touched path {relative_path!r} has a symlink at "
                    f"{current.name!r}; refusing to apply."
                )
        if not current.exists():
            raise ApplyRefusedError(
                f"Touched path {relative_path!r} does not exist; new-file "
                "creation is not supported in this stage."
            )
        if not current.is_file():
            raise ApplyRefusedError(
                f"Touched path {relative_path!r} is not a regular file."
            )
        resolved = current.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ApplyRefusedError(
                f"Touched path {relative_path!r} resolves outside the project root."
            ) from exc


def _normalize_patch_terminator(patch_preview: str) -> bytes:
    """Encode the validated patch for git apply stdin, newline-terminated.

    git apply treats a patch whose last line lacks a trailing newline as
    "corrupt patch at <stdin>:N". Append exactly one "\\n" when missing;
    the patch contents are never otherwise rewritten.
    """
    if not patch_preview.endswith("\n"):
        patch_preview += "\n"
    return patch_preview.encode("utf-8")


# ---------------------------------------------------------------------------
# Git helpers (fixed argument arrays, shell=False, stdin-only patch input)
# ---------------------------------------------------------------------------


def _run_git(
    args: list[str], root: Path, *, input_bytes: bytes | None = None
) -> GitCommandResult:
    """Run one fixed git command. The only subprocess gateway in this module."""
    if not args or args[0] != "git" or not all(isinstance(a, str) for a in args):
        raise ApplyRefusedError("Internal error: only fixed git commands may run.")
    completed = subprocess.run(
        args,
        cwd=str(root),
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=False,
        check=False,
        timeout=_GIT_TIMEOUT_SECONDS,
    )
    return GitCommandResult(
        args=tuple(args),
        returncode=completed.returncode,
        stdout=completed.stdout.decode("utf-8", errors="replace"),
        stderr=completed.stderr.decode("utf-8", errors="replace"),
    )


def _require_git_toplevel(root: Path) -> None:
    result = _run_git(["git", "rev-parse", "--show-toplevel"], root)
    if result.returncode != 0:
        raise ApplyRefusedError(
            f"Project root {root} is not a git work tree: {result.stderr.strip()}"
        )
    toplevel = Path(result.stdout.strip()).resolve()
    if toplevel != root:
        raise ApplyRefusedError(
            f"Project root {root} is not the git toplevel ({toplevel})."
        )


def _current_branch(root: Path) -> str:
    result = _run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], root)
    if result.returncode != 0:
        raise ApplyRefusedError(
            f"Could not determine the current branch: {result.stderr.strip()}"
        )
    branch = result.stdout.strip()
    if branch == "HEAD":
        raise ApplyRefusedError(
            "HEAD is detached; check out a named branch before applying."
        )
    return branch


def _require_clean_tree(root: Path, *, stage: str) -> None:
    """Refuse any dirty state: tracked, staged, and untracked alike.

    ``git status --porcelain`` reports unstaged modifications, staged
    changes, and untracked files in one pass; any output at all is a refusal.
    """
    entries = _status_entries(root)
    if entries:
        listing = ", ".join(entries[:10])
        raise ApplyRefusedError(
            f"Working tree is not clean {stage} "
            f"(tracked/staged/untracked changes present): {listing}"
        )


def _status_entries(root: Path) -> list[str]:
    result = _run_git(["git", "status", "--porcelain"], root)
    if result.returncode != 0:
        raise ApplyRefusedError(f"git status failed: {result.stderr.strip()}")
    return [line for line in result.stdout.splitlines() if line.strip()]


def _head_commit(root: Path) -> str:
    result = _run_git(["git", "rev-parse", "HEAD"], root)
    if result.returncode != 0:
        raise ApplyRefusedError(
            f"Could not read the base commit: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _proposal_branch_name(approval_id: int) -> str:
    """Deterministic branch name from the integer approval id only.

    ``selected_strategy`` and other payload text are never used, so no
    payload field can influence the branch name.
    """
    if not isinstance(approval_id, int) or approval_id <= 0:
        raise ApplyRefusedError("Approval id must be a positive integer.")
    return f"{BRANCH_NAME_PREFIX}{approval_id}"


def _refuse_existing_branch(branch_name: str, root: Path) -> None:
    result = _run_git(
        ["git", "rev-parse", "--verify", "--quiet", f"refs/heads/{branch_name}"],
        root,
    )
    if result.returncode == 0:
        raise ApplyRefusedError(
            f"Branch {branch_name!r} already exists; delete it manually "
            "before re-applying."
        )


def _return_to_original_if_clean(root: Path, original_branch: str) -> None:
    """After a failed apply, go back to the original branch only when safe.

    Only checks out the original branch if the working tree is still clean,
    so a partial application can never be dragged onto the original branch.
    The proposal branch is never deleted automatically.
    """
    try:
        if _status_entries(root):
            return
        _run_git(["git", "checkout", original_branch], root)
    except ApplyRefusedError:
        return


# ---------------------------------------------------------------------------
# Post-apply verification
# ---------------------------------------------------------------------------


def _verify_post_apply(
    root: Path, touched_paths: tuple[str, ...], branch_name: str
) -> tuple[str, ...]:
    """Require the post-apply change set to equal the declared paths exactly.

    Every status entry must be an unstaged modification (`` M``) of a
    declared path; anything staged, untracked, renamed, or undeclared is a
    refusal. Also refuses touched files left empty — the effective-deletion
    case Stage 7B deferred here. On refusal the branch and working tree are
    left as-is for manual inspection; nothing is deleted automatically.
    """
    entries = _status_entries(root)
    changed: list[str] = []
    problems: list[str] = []
    for entry in entries:
        code, path_part = entry[:2], entry[3:]
        if code != " M" or path_part.startswith('"') or " -> " in path_part:
            problems.append(f"unexpected status entry {entry!r}")
            continue
        changed.append(path_part)
    if sorted(changed) != sorted(touched_paths):
        undeclared = sorted(set(changed) - set(touched_paths))
        missing = sorted(set(touched_paths) - set(changed))
        if undeclared:
            problems.append(f"undeclared files changed: {undeclared}")
        if missing:
            problems.append(f"declared files unchanged: {missing}")
    for relative_path in touched_paths:
        target = root / relative_path
        if not target.is_file() or target.stat().st_size == 0:
            problems.append(
                f"touched file {relative_path!r} was effectively deleted "
                "(missing or empty after apply)"
            )
    if problems:
        raise ApplyRefusedError(
            "Post-apply verification failed: "
            + "; ".join(problems)
            + f". Branch {branch_name!r} and its working tree were left "
            "in place for manual inspection; nothing was deleted."
        )
    return tuple(changed)


# ---------------------------------------------------------------------------
# Developer CLI entrypoint
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one approved, validated code proposal to a new local "
            "git branch. Apply-only: no tests, no push, no remotes, no PRs."
        )
    )
    parser.add_argument("approval_id", type=int, help="Approval record id")
    args = parser.parse_args(argv)
    try:
        result = apply_approved_proposal(args.approval_id)
    except ApplyRefusedError as exc:
        print(f"REFUSED: {exc}")
        return 1
    print(format_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
