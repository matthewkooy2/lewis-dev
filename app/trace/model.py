"""Shared normalized data model for code tracing.

Language analyzers (Python, JavaScript/TypeScript) each produce the same
per-file ``ModuleIndex`` shape, so the traversal engine, renderer, and Lewis
integration never contain language-specific logic. Everything is a plain
dataclass convertible to JSON-safe dicts for reuse by other subsystems.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


TRACE_MODE_QUICK = "quick"
TRACE_MODE_DEEP = "deep"
TRACE_MODE_REVIEW = "review"
TRACE_MODES = (TRACE_MODE_QUICK, TRACE_MODE_DEEP, TRACE_MODE_REVIEW)

# How a relationship or fact was established. Static analysis cannot resolve
# every dynamic behavior, so results always carry one of these labels instead
# of guessing.
RESOLUTION_CONFIRMED = "confirmed"
RESOLUTION_INFERRED = "inferred"
RESOLUTION_FRAMEWORK = "framework-registered"
RESOLUTION_AMBIGUOUS = "ambiguous"
RESOLUTION_EXTERNAL = "external"
RESOLUTION_DYNAMIC = "dynamic"
RESOLUTION_UNRESOLVED = "unresolved"
RESOLUTIONS = (
    RESOLUTION_CONFIRMED,
    RESOLUTION_INFERRED,
    RESOLUTION_FRAMEWORK,
    RESOLUTION_AMBIGUOUS,
    RESOLUTION_EXTERNAL,
    RESOLUTION_DYNAMIC,
    RESOLUTION_UNRESOLVED,
)

LANGUAGE_PYTHON = "python"
LANGUAGE_JAVASCRIPT = "javascript"
LANGUAGE_TYPESCRIPT = "typescript"

KIND_FUNCTION = "function"
KIND_ASYNC_FUNCTION = "async_function"
KIND_METHOD = "method"
KIND_ASYNC_METHOD = "async_method"
KIND_CLASS = "class"
KIND_COMPONENT = "react_component"
KIND_HOOK = "react_hook"
KIND_CONSTANT = "constant"
KIND_VARIABLE = "variable"
KIND_TYPE = "type"

EFFECT_FILESYSTEM = "filesystem"
EFFECT_NETWORK = "network"
EFFECT_PROCESS = "process"
EFFECT_DATABASE = "database"
EFFECT_ENVIRONMENT = "environment"
EFFECT_GLOBAL_STATE = "global_state"
EFFECT_CONSOLE_LOGGING = "console_logging"
EFFECT_DOM_STORAGE = "dom_or_browser_storage"
EFFECT_TIME_RANDOM = "time_or_randomness"
EFFECT_CONCURRENCY = "concurrency"
EFFECT_IMPORT_TIME = "import_time"


@dataclass(frozen=True)
class SymbolDefinition:
    """One named definition discovered in a workspace source file."""

    path: str
    qualname: str
    name: str
    kind: str
    language: str
    line: int
    end_line: int
    signature: str = ""
    parent: str | None = None
    decorators: tuple[str, ...] = ()
    exported: bool = False
    export_names: tuple[str, ...] = ()
    doc: str = ""

    @property
    def symbol_id(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass(frozen=True)
class ImportBinding:
    """A name made available in a module by an import statement."""

    local_name: str
    source_module: str
    source_name: str | None
    kind: str  # module | named | default | namespace | star | side_effect | require | dynamic
    line: int


@dataclass(frozen=True)
class ExportBinding:
    """A name a module makes available to importers."""

    exported_name: str
    local_name: str | None
    source_module: str | None
    source_name: str | None
    kind: str  # named | default | star | commonjs | all_list | implicit
    line: int


@dataclass(frozen=True)
class RawCall:
    """One call site as written, before project-level resolution."""

    caller_qualname: str | None
    callee_text: str
    head: str
    attr_chain: tuple[str, ...]
    line: int
    is_new: bool = False
    is_await: bool = False
    is_statement: bool = False
    is_dynamic: bool = False
    jsx_component: bool = False
    argument_names: tuple[str, ...] = ()
    receiver_type: str | None = None
    is_callback_reference: bool = False


@dataclass(frozen=True)
class SideEffect:
    """A concrete side-effect signal observed at a specific line."""

    category: str
    detail: str
    path: str
    line: int
    qualname: str | None
    resolution: str = RESOLUTION_CONFIRMED


@dataclass(frozen=True)
class ErrorHandling:
    """Raise/throw and catch behavior for one function scope."""

    qualname: str
    raises: tuple[str, ...] = ()
    raise_lines: tuple[int, ...] = ()
    catches: tuple[str, ...] = ()
    catch_lines: tuple[int, ...] = ()
    swallowed_catch_lines: tuple[int, ...] = ()
    bare_catch_lines: tuple[int, ...] = ()


@dataclass(frozen=True)
class ModuleIndex:
    """Normalized per-file analysis result shared by all languages."""

    path: str
    language: str
    line_count: int
    definitions: tuple[SymbolDefinition, ...] = ()
    imports: tuple[ImportBinding, ...] = ()
    exports: tuple[ExportBinding, ...] = ()
    calls: tuple[RawCall, ...] = ()
    side_effects: tuple[SideEffect, ...] = ()
    error_handling: tuple[ErrorHandling, ...] = ()
    class_bases: dict[str, tuple[str, ...]] = field(default_factory=dict)
    attribute_types: dict[str, str] = field(default_factory=dict)
    is_test: bool = False
    parse_ok: bool = True
    parse_error: str = ""


@dataclass(frozen=True)
class CallEdge:
    """A resolved (or honestly unresolved) call relationship."""

    caller_path: str
    caller_qualname: str | None
    line: int
    callee_text: str
    target_path: str | None
    target_qualname: str | None
    resolution: str
    note: str = ""
    depth: int = 0


@dataclass(frozen=True)
class TestLink:
    """A test that exercises or references the traced symbol."""

    test_path: str
    test_name: str
    line: int
    via: str
    resolution: str


@dataclass(frozen=True)
class Snippet:
    """A bounded, secret-redacted source excerpt."""

    path: str
    start_line: int
    end_line: int
    text: str
    truncated: bool = False


@dataclass(frozen=True)
class ReviewFinding:
    """One review observation with explicit confidence."""

    category: str  # correctness | error_handling | security | side_effects |
    #                maintainability | consistency | testing
    severity: str  # issue | risk | note
    confidence: str  # confirmed | heuristic
    path: str
    line: int
    message: str


@dataclass(frozen=True)
class TraceLimits:
    """Bounds applied to discovery, traversal, and output."""

    max_files: int = 400
    max_file_bytes: int = 512 * 1024
    max_depth: int = 4
    max_nodes: int = 60
    max_edges: int = 240
    max_callers: int = 30
    max_callees: int = 40
    max_side_effects: int = 40
    max_tests: int = 20
    max_candidates: int = 10
    max_snippet_lines: int = 60
    max_findings: int = 40
    max_report_chars: int = 24_000


@dataclass
class TraceResult:
    """Complete outcome of one trace or review request."""

    mode: str
    ok: bool
    target: SymbolDefinition | None = None
    candidates: tuple[SymbolDefinition, ...] = ()
    definition_snippet: Snippet | None = None
    callers: tuple[CallEdge, ...] = ()
    callees: tuple[CallEdge, ...] = ()
    call_paths: tuple[tuple[str, ...], ...] = ()
    side_effects: tuple[SideEffect, ...] = ()
    transitive_side_effects: tuple[SideEffect, ...] = ()
    tests: tuple[TestLink, ...] = ()
    error_handling: tuple[ErrorHandling, ...] = ()
    cycles: tuple[tuple[str, ...], ...] = ()
    unresolved_calls: tuple[CallEdge, ...] = ()
    findings: tuple[ReviewFinding, ...] = ()
    evidence: tuple[Snippet, ...] = ()
    warnings: tuple[str, ...] = ()
    truncations: tuple[str, ...] = ()
    stats: dict[str, int] = field(default_factory=dict)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe dict for CLI --json output and future Freestyle reuse."""
        payload = asdict(self)
        payload["schema"] = "lewis-trace-result/1"
        return payload
