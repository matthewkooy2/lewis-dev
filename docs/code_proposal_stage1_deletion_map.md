# Code Proposal Replacement: Stage 1 Deletion and Bypass Map

Historical migration plan. September 2026: the unreachable legacy chat fallback
and `app/search_first_code_proposal.py` have been retired. The replacement
engine and existing end-to-end proposal contracts remain. Other entries below
are historical targets, not evidence that every migration has been completed.

Stage 1 adds replacement-contract tests and this map. It adds no production
implementation and deletes no production files.

## Replacement Contract

The Stage 2 engine must be the single code-proposal route. From a user request,
it must use safe read-only project context, create one pending approval, and
store a real unified diff that passes `git apply --check` against the unchanged
project tree. The diff is inert review text only and must never be applied.

Every Stage 2 production module or function must remove or bypass an old item
listed below. A new parallel proposal route is not acceptable.

## Delete or Bypass After Replacement Tests Pass

| Old item | Exact functions or branches replaced | Deletion criteria |
| --- | --- | --- |
| `app/code_proposal_text_tool_fallback.py` | `extract_project_search_queries`, `code_proposal_args_from_search_results`, `_relevant_files` | Delete the file after `LewisAgent` no longer completes proposals from model-emitted text search blocks, replacement-contract tests pass, and raw markup suppression tests still pass through `app/text_tool_calls.py`. |
| `app/code_proposal_tool_strategy.py` | `ToolStrategy`, `code_proposal_tool_strategy`, `_is_search_first_code_proposal`, `PROJECT_SEARCH_TOOL`, `CODE_PROPOSAL_TOOL` | Delete the file after the LLM loop no longer selects a proposal-specific search/propose sequence and all non-proposal project-search tests still pass. |
| `app/search_first_code_proposal.py` | `SearchFirstCodeProposalResult`, `is_search_first_code_proposal_request`, `create_search_first_code_proposal`, `project_search_queries`, `extract_requested_goal`, `grounded_proposed_approach`, and all private helpers | Delete the file after all direct-change and explicit-proposal prompts enter the replacement engine, create one pending approval, and pass the real inert unified-diff contract without importing this module. |
| `app/code_proposal_patch_preview.py` | `generate_grounded_patch_preview`, `_preview_for_excerpt`, `_python_label_preview`, `_with_chat_label`, `_unified_single_line_replacement`, `_unified_single_line_addition`, `_is_simple_ui_label_request`; retire `GroundedPatchPreview` and preview-note constants when unused | Delete the placeholder generator after the replacement engine generates applicable diffs for ordinary changes, safely omits proposals it cannot ground, and no production import remains. |
| `app/llm.py` proposal routing | `_has_explicit_code_proposal_phrase`; `LewisAgent._maybe_run_search_first_code_proposal`; `LewisAgent._run_deterministic_code_proposal_path`; `LewisAgent._maybe_complete_text_search_first_code_proposal`; the early proposal branch, `code_proposal_tool_strategy` branch, fallback branch, and imports from the four files above | Replace these branches with one call into the Stage 2 engine. Delete them after normal chat, project search, tool limits, raw markup suppression, DeepSeek `reasoning_content`, and memory tests still pass. Keep generic approval tracking and non-code proposal safety behavior. |
| `app/code_proposals.py` duplicated normalization and payload assembly | `CodeProposal`, `propose_code_change`, `validate_code_proposal`, `_validated_relevant_files`, `_required_text`, `_optional_text`, `_normalized_text_list`, `_reject_patch_content`, `_redact_secret_values`, target-surface inference helpers, contextual-safety helpers, `_proposal_title` | Remove or reduce this module to a single persistence boundary after the replacement engine owns one canonical validated payload and all callers migrate. Delete it entirely when `app/tools.py`, `app/main.py`, tests, and rendering no longer import it. Path safety must continue to delegate to `validate_relative_project_path`. |
| `app/tools.py` code-proposal tool path | `ToolRegistry._add_code_proposal_tool`, `ToolRegistry._code_proposal`, and the `propose_code_change` import | Remove or bypass this proposal path after Stage 2 owns proposal creation. Keep `ToolRegistry` and safe read-only project search. |
| `app/main.py` direct code-proposal creation | The `propose_code_change` import and the `main()` `propose-code` branch's direct payload construction | Route the CLI command through the same Stage 2 boundary or remove the command. The old direct call can be deleted after CLI tests cover the replacement behavior. |

## Preserve

- `ApprovalStore`, `ApprovalItem`, `app/approval_service.py`, approval statuses,
  and SQLite rows/data.
- `app/code_proposal_path_safety.py` and safe read-only access in
  `app/project_search.py`.
- Raw markup suppression in `app/text_tool_calls.py`.
- DeepSeek `reasoning_content` handling in `app/deepseek_messages.py`.
- Memory behavior and memory tests.
- Web/CLI approval rendering infrastructure. Payload readers may be adapted to
  the canonical Stage 2 payload, but the approval review surface remains.
- Patch path safety/classification behavior from `app/code_proposal_previews.py`
  and `app/code_proposal_file_structure.py` until equivalent replacement tests
  prove the same protections.

## Stage 2 Gate

Before deleting an old item:

1. The failing replacement-contract cases in
   `tests/test_code_proposal_replacement_contract.py` must pass.
2. Focused proposal, path-safety, approval, web, CLI, memory, raw-markup, and
   DeepSeek reasoning tests must pass.
3. A repository search must show no production imports or calls to the deleted
   item.
4. The replacement must create one pending approval and must not edit project
   files, execute commands, or mutate existing SQLite data.
