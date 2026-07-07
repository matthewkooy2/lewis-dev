"""Structural analysis of JavaScript/TypeScript token streams.

Consumes the token stream from ``js_lexer`` and produces the shared
``ModuleIndex`` shape: definitions (functions, arrows, classes, methods,
object methods, React components/hooks, TS types), ES module and CommonJS
import/export bindings, call sites, side-effect signals, and per-function
error handling.

This is a bounded structural parser, not a full ECMAScript grammar: scope is
tracked through brace/paren bookkeeping and declaration lookahead. Constructs
it cannot attribute precisely degrade to honest labels (dynamic/unresolved)
rather than guesses, and files whose lexing fails are marked ``parse_ok
False``. No analyzed code is ever executed.
"""

from __future__ import annotations

from pathlib import PurePosixPath

from app.trace.js_lexer import Token, lex
from app.trace.model import (
    EFFECT_CONCURRENCY,
    EFFECT_CONSOLE_LOGGING,
    EFFECT_DATABASE,
    EFFECT_DOM_STORAGE,
    EFFECT_ENVIRONMENT,
    EFFECT_FILESYSTEM,
    EFFECT_GLOBAL_STATE,
    EFFECT_NETWORK,
    EFFECT_PROCESS,
    EFFECT_TIME_RANDOM,
    ErrorHandling,
    ExportBinding,
    ImportBinding,
    KIND_ASYNC_FUNCTION,
    KIND_ASYNC_METHOD,
    KIND_CLASS,
    KIND_COMPONENT,
    KIND_FUNCTION,
    KIND_HOOK,
    KIND_METHOD,
    KIND_TYPE,
    LANGUAGE_TYPESCRIPT,
    ModuleIndex,
    RawCall,
    RESOLUTION_CONFIRMED,
    RESOLUTION_INFERRED,
    SideEffect,
    SymbolDefinition,
)


MAX_LOOKAHEAD = 240
MAX_SIGNATURE_CHARS = 160
MAX_ARGUMENT_NAMES = 6
MAX_EFFECTS_PER_FILE = 200

MODIFIER_TOKENS = {
    "static",
    "async",
    "get",
    "set",
    "public",
    "private",
    "protected",
    "readonly",
    "override",
    "abstract",
    "declare",
}

# Reserved words that can never be a bare call head (`if (...)`, `return (...)`).
# Contextual keywords such as get/set/from/type are legal identifiers and are
# accepted; `this`/`super` are legal call heads.
NON_CALL_HEAD_KEYWORDS = {
    "async", "break", "case", "catch", "class", "const", "continue",
    "debugger", "default", "delete", "do", "else", "export", "extends",
    "finally", "for", "function", "if", "import", "in", "instanceof", "let",
    "new", "of", "return", "switch", "throw", "try", "typeof", "var", "void",
    "while", "with", "yield", "await",
}

# Import specifier roots mapped to the side-effect category their calls imply.
EFFECT_IMPORT_ROOTS = {
    "fs": EFFECT_FILESYSTEM,
    "fs/promises": EFFECT_FILESYSTEM,
    "child_process": EFFECT_PROCESS,
    "http": EFFECT_NETWORK,
    "https": EFFECT_NETWORK,
    "net": EFFECT_NETWORK,
    "dgram": EFFECT_NETWORK,
    "tls": EFFECT_NETWORK,
    "axios": EFFECT_NETWORK,
    "node-fetch": EFFECT_NETWORK,
    "undici": EFFECT_NETWORK,
    "ws": EFFECT_NETWORK,
    "pg": EFFECT_DATABASE,
    "mysql": EFFECT_DATABASE,
    "mysql2": EFFECT_DATABASE,
    "mongoose": EFFECT_DATABASE,
    "mongodb": EFFECT_DATABASE,
    "knex": EFFECT_DATABASE,
    "redis": EFFECT_DATABASE,
    "ioredis": EFFECT_DATABASE,
    "sqlite3": EFFECT_DATABASE,
    "better-sqlite3": EFFECT_DATABASE,
}

DOM_MUTATION_METHODS = {
    "write",
    "appendChild",
    "removeChild",
    "insertBefore",
    "replaceChild",
    "replaceChildren",
    "setAttribute",
    "removeAttribute",
    "insertAdjacentHTML",
    "focus",
    "blur",
}
CLASSLIST_METHODS = {"add", "remove", "toggle", "replace"}
STORAGE_OBJECTS = {"localStorage", "sessionStorage"}
TIMER_HEADS = {"setTimeout", "setInterval", "requestAnimationFrame", "queueMicrotask"}
REACT_WRAPPERS = {"memo", "forwardRef", "useCallback"}


def is_js_test_path(path: str) -> bool:
    lowered = path.lower()
    name = PurePosixPath(lowered).name
    return (
        "__tests__/" in lowered
        or lowered.startswith("tests/")
        or "/tests/" in lowered
        or ".test." in name
        or ".spec." in name
    )


def analyze_js_file(path: str, text: str, language: str) -> ModuleIndex:
    suffix = PurePosixPath(path).suffix.lower()
    jsx_enabled = suffix != ".ts"
    lexed = lex(text, jsx_enabled=jsx_enabled)
    line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    analyzer = _Analyzer(path, language, lexed.tokens)
    analyzer.run()
    return ModuleIndex(
        path=path,
        language=language,
        line_count=line_count,
        definitions=tuple(analyzer.finalized_definitions()),
        imports=tuple(analyzer.imports),
        exports=tuple(analyzer.exports),
        calls=tuple(analyzer.calls),
        side_effects=tuple(analyzer.side_effects[:MAX_EFFECTS_PER_FILE]),
        error_handling=tuple(analyzer.finalized_error_handling()),
        class_bases=dict(analyzer.class_bases),
        attribute_types={},
        is_test=is_js_test_path(path),
        parse_ok=lexed.ok,
        parse_error=lexed.error,
    )


class _Frame:
    def __init__(
        self,
        kind: str,
        qualname: str | None,
        *,
        def_index: int | None = None,
        expr_arrow: bool = False,
        snapshot: tuple[int, int, int] | None = None,
    ):
        self.kind = kind  # module | function | class | object
        self.qualname = qualname
        self.def_index = def_index
        self.expr_arrow = expr_arrow
        self.snapshot = snapshot or (0, 0, 0)
        self.awaiting_body_brace = False
        self.contains_jsx = False
        self.local_types: dict[str, str] = {}
        # Parameter/destructured bindings visible in this scope, mapped to the
        # line of their local default implementation (None when no callable
        # default is declared).
        self.local_bindings: dict[str, int | None] = {}
        self.object_name: str | None = None


class _Brace:
    def __init__(self, label: str, pops_frame: bool):
        self.label = label
        self.pops_frame = pops_frame
        self.expect_member = label in {"class_body", "object", "type_body"}


class _Pending:
    """A definition header waiting for its body brace."""

    def __init__(self, kind: str, def_index: int | None, paren_depth: int):
        self.kind = kind  # function | class | object_value | type_body
        self.def_index = def_index
        self.paren_depth = paren_depth
        self.object_name: str | None = None
        self.param_bindings: dict[str, int | None] = {}


class _JsErrorInfo:
    def __init__(self) -> None:
        self.raises: list[str] = []
        self.raise_lines: list[int] = []
        self.catches: list[str] = []
        self.catch_lines: list[int] = []
        self.swallowed: list[int] = []


class _Analyzer:
    def __init__(self, path: str, language: str, tokens: list[Token]):
        self.path = path
        self.language = language
        self.tokens = tokens
        self.i = 0
        self.paren_depth = 0
        self.special_depth = 0  # open jsx/template expression islands
        self.frames: list[_Frame] = [_Frame("module", None)]
        self.braces: list[_Brace] = []
        self.pending_scopes: list[_Pending] = []
        self.non_call_parens: set[int] = set()
        self.arrow_defs: dict[int, int] = {}  # token index of "=>" -> def index
        self.arrow_params: dict[int, dict[str, int | None]] = {}
        # Token ranges of parameter lists whose calls/effects belong to the
        # function being defined (defaults run inside its scope), keyed as
        # (start_index, end_index, qualname).
        self._param_scopes: list[tuple[int, int, str]] = []
        self.definitions: list[dict] = []
        self.imports: list[ImportBinding] = []
        self.exports: list[ExportBinding] = []
        self.calls: list[RawCall] = []
        self.side_effects: list[SideEffect] = []
        self.class_bases: dict[str, tuple[str, ...]] = {}
        self.module_let_names: set[str] = set()
        self.error_infos: dict[str, _JsErrorInfo] = {}
        self._effect_import_categories: dict[str, str] = {}
        self._export_next_definition: list[str] | None = None
        self._pending_decorators: list[str] = []
        self._pending_jsx_handler_attr = False
        self._env_read_noted: set[str | None] = set()

    # ---- driving ----------------------------------------------------------

    def run(self) -> None:
        while self.i < len(self.tokens):
            token = self.tokens[self.i]
            if self._in_type_body() and token.type == "punct":
                self._handle_structural_punct(token)
                self.i += 1
                continue
            if self._in_type_body():
                self.i += 1
                continue
            handler = getattr(self, f"_on_{token.type}", None)
            if handler is not None:
                handler(token)
            self.i += 1

    def finalized_definitions(self) -> list[SymbolDefinition]:
        exported_names = {
            binding.local_name
            for binding in self.exports
            if binding.local_name is not None
        }
        export_aliases: dict[str, list[str]] = {}
        for binding in self.exports:
            if binding.local_name:
                export_aliases.setdefault(binding.local_name, []).append(
                    binding.exported_name
                )
        results: list[SymbolDefinition] = []
        for record in self.definitions:
            kind = record["kind"]
            name = record["name"]
            if (
                kind in {KIND_FUNCTION, KIND_ASYNC_FUNCTION}
                and record.get("contains_jsx")
                and name[:1].isupper()
            ):
                kind = KIND_COMPONENT
            elif (
                kind in {KIND_FUNCTION, KIND_ASYNC_FUNCTION}
                and len(name) > 3
                and name.startswith("use")
                and name[3].isupper()
            ):
                kind = KIND_HOOK
            exported = record["exported"] or (
                record["parent"] is None and name in exported_names
            )
            results.append(
                SymbolDefinition(
                    path=self.path,
                    qualname=record["qualname"],
                    name=name,
                    kind=kind,
                    language=self.language,
                    line=record["line"],
                    end_line=record.get("end_line") or record["line"],
                    signature=record.get("signature", ""),
                    parent=record["parent"],
                    decorators=tuple(record.get("decorators", ())),
                    exported=exported,
                    export_names=tuple(export_aliases.get(name, ())),
                )
            )
        return results

    def finalized_error_handling(self) -> list[ErrorHandling]:
        return [
            ErrorHandling(
                qualname=qualname,
                raises=tuple(info.raises),
                raise_lines=tuple(info.raise_lines),
                catches=tuple(info.catches),
                catch_lines=tuple(info.catch_lines),
                swallowed_catch_lines=tuple(info.swallowed),
            )
            for qualname, info in self.error_infos.items()
        ]

    # ---- token handlers ----------------------------------------------------

    def _on_keyword(self, token: Token) -> None:
        value = token.value
        top_brace = self.braces[-1] if self.braces else None
        if top_brace is not None and top_brace.expect_member:
            if top_brace.label == "class_body":
                self._handle_class_member(token)
                return
            if top_brace.label == "object" and value not in {"const", "let", "var"}:
                self._handle_object_member(token)
                return
        if value == "import":
            self._handle_import(token)
            return
        if value == "export":
            self._handle_export(token)
            return
        if value == "function":
            self._handle_function_keyword(token, name_hint=None)
            return
        if value == "class":
            self._handle_class_keyword(token)
            return
        if value in {"const", "let", "var"}:
            self._handle_declaration(token)
            return
        if value in {"interface", "enum", "namespace"} and self.language == LANGUAGE_TYPESCRIPT:
            self._handle_type_declaration(token)
            return
        if value == "type" and self.language == LANGUAGE_TYPESCRIPT:
            if self._peek_type(1) == "id" and self._peek_value_is(2, "="):
                self._handle_type_alias(token)
                return
        if value == "throw":
            self._handle_throw(token)
            return
        if value == "catch":
            self._handle_catch(token)
            return
        if value in {"this", "super"}:
            self._maybe_member_or_call(token)
            return

    def _on_id(self, token: Token) -> None:
        top_brace = self.braces[-1] if self.braces else None
        if top_brace is not None and top_brace.expect_member:
            if top_brace.label == "class_body":
                self._handle_class_member(token)
                return
            if top_brace.label == "object":
                self._handle_object_member(token)
                return
        if token.value in {"module", "exports"}:
            self._maybe_commonjs_export(token)
        self._maybe_member_or_call(token)

    def _maybe_commonjs_export(self, token: Token) -> None:
        j = self.i
        if token.value == "module":
            if not (
                self._peek_value_is(1, ".")
                and self.i + 2 < len(self.tokens)
                and self.tokens[self.i + 2].value == "exports"
            ):
                return
            j = self.i + 2
        after = j + 1
        if after >= len(self.tokens):
            return
        if self.tokens[after].type == "punct" and self.tokens[after].value == ".":
            name_index = after + 1
            if (
                name_index + 1 < len(self.tokens)
                and self.tokens[name_index].type in {"id", "keyword"}
                and self.tokens[name_index + 1].type == "punct"
                and self.tokens[name_index + 1].value == "="
            ):
                value_index = name_index + 2
                local = None
                if (
                    value_index < len(self.tokens)
                    and self.tokens[value_index].type == "id"
                ):
                    local = self.tokens[value_index].value
                self.exports.append(
                    ExportBinding(
                        self.tokens[name_index].value,
                        local,
                        None,
                        None,
                        "commonjs",
                        token.line,
                    )
                )
            return
        if (
            token.value == "module"
            and self.tokens[after].type == "punct"
            and self.tokens[after].value == "="
        ):
            value_index = after + 1
            if value_index >= len(self.tokens):
                return
            value = self.tokens[value_index]
            if value.type == "id":
                self.exports.append(
                    ExportBinding("default", value.value, None, None, "commonjs", token.line)
                )
                return
            if value.type == "punct" and value.value == "{":
                self._scan_commonjs_object_exports(value_index, token.line)

    def _scan_commonjs_object_exports(self, open_index: int, line: int) -> None:
        depth = 0
        k = open_index
        end = min(len(self.tokens), open_index + 400)
        while k < end:
            item = self.tokens[k]
            if item.type == "punct":
                if item.value == "{":
                    depth += 1
                elif item.value == "}":
                    depth -= 1
                    if depth == 0:
                        return
            elif depth == 1 and item.type in {"id", "keyword"}:
                next_token = self.tokens[k + 1] if k + 1 < len(self.tokens) else None
                if next_token is not None and next_token.type == "punct":
                    if next_token.value in {",", "}", "("}:
                        self.exports.append(
                            ExportBinding(
                                item.value, item.value, None, None, "commonjs", line
                            )
                        )
                    elif next_token.value == ":":
                        value_token = (
                            self.tokens[k + 2] if k + 2 < len(self.tokens) else None
                        )
                        local = (
                            value_token.value
                            if value_token is not None and value_token.type == "id"
                            else None
                        )
                        self.exports.append(
                            ExportBinding(
                                item.value, local, None, None, "commonjs", line
                            )
                        )
                        k += 2
            k += 1

    def _on_punct(self, token: Token) -> None:
        value = token.value
        if value == "=>":
            self._handle_arrow(token)
            return
        if value == "@":
            self._capture_decorator()
            return
        if value == "#":
            return
        if value == "(":
            if self.i in self.non_call_parens:
                self.paren_depth += 1
                return
            self._maybe_record_call(token)
            self.paren_depth += 1
            return
        self._handle_structural_punct(token)

    def _on_jsx_open(self, token: Token) -> None:
        name = token.value
        for frame in reversed(self.frames):
            if frame.kind == "function":
                frame.contains_jsx = True
                break
        if name and name[0].isupper():
            parts = name.split(".")
            self.calls.append(
                RawCall(
                    caller_qualname=self._caller_qualname(),
                    callee_text=f"<{name}>",
                    head=parts[0],
                    attr_chain=tuple(parts[1:]),
                    line=token.line,
                    jsx_component=True,
                )
            )
        self._pending_jsx_handler_attr = False

    def _on_jsx_attr(self, token: Token) -> None:
        self._pending_jsx_handler_attr = (
            token.value.startswith("on") and len(token.value) > 2
        )

    def _on_jsx_expr_open(self, token: Token) -> None:
        self.special_depth += 1
        if self._pending_jsx_handler_attr and self.i + 2 < len(self.tokens):
            name_token = self.tokens[self.i + 1]
            close_token = self.tokens[self.i + 2]
            if name_token.type == "id" and close_token.type == "jsx_expr_close":
                is_local, local_default_line = self._lookup_local_binding(
                    name_token.value
                )
                self.calls.append(
                    RawCall(
                        caller_qualname=self._caller_qualname(),
                        callee_text=name_token.value,
                        head=name_token.value,
                        attr_chain=(),
                        line=token.line,
                        is_callback_reference=True,
                        is_local_binding=is_local,
                        local_default_line=local_default_line,
                    )
                )
        self._pending_jsx_handler_attr = False

    def _on_jsx_expr_close(self, token: Token) -> None:
        self.special_depth = max(0, self.special_depth - 1)
        self._close_expression_arrows_on_special()

    def _on_template_expr_open(self, token: Token) -> None:
        self.special_depth += 1

    def _on_template_expr_close(self, token: Token) -> None:
        self.special_depth = max(0, self.special_depth - 1)
        self._close_expression_arrows_on_special()

    def _on_jsx_close(self, token: Token) -> None:
        self._pending_jsx_handler_attr = False

    # ---- structural punctuation ---------------------------------------------

    def _handle_structural_punct(self, token: Token) -> None:
        value = token.value
        if value == "{":
            self._open_brace(token)
            return
        if value == "}":
            self._close_brace(token)
            return
        if value == "(":
            self.paren_depth += 1
            return
        if value == ")":
            self.paren_depth = max(0, self.paren_depth - 1)
            self._close_expression_arrows(token)
            return
        if value == "[":
            return
        if value == "]":
            return
        if value in {",", ";"}:
            self._close_expression_arrows(token)
            if value == ";":
                self._mark_member_expected()
            return
        if value == "=":
            self._maybe_global_state_write(token)
            self._maybe_html_sink_write(token)
            return

    def _open_brace(self, token: Token) -> None:
        pending = None
        if self.pending_scopes and self.pending_scopes[-1].paren_depth == self.paren_depth:
            pending = self.pending_scopes.pop()
        if pending is not None and pending.kind == "function":
            frame = _Frame("function", self._def_qualname(pending.def_index),
                          def_index=pending.def_index)
            frame.local_bindings.update(pending.param_bindings)
            self.frames.append(frame)
            self.braces.append(_Brace("function_body", pops_frame=True))
            return
        if pending is not None and pending.kind == "class":
            frame = _Frame("class", self._def_qualname(pending.def_index),
                          def_index=pending.def_index)
            self.frames.append(frame)
            self.braces.append(_Brace("class_body", pops_frame=True))
            return
        if pending is not None and pending.kind == "type_body":
            self.braces.append(_Brace("type_body", pops_frame=False))
            return
        if pending is not None and pending.kind == "object_value":
            frame = _Frame("object", None)
            frame.object_name = pending.object_name
            self.frames.append(frame)
            self.braces.append(_Brace("object", pops_frame=True))
            return
        top = self.frames[-1]
        if top.kind == "function" and top.awaiting_body_brace:
            top.awaiting_body_brace = False
            self.braces.append(_Brace("function_body", pops_frame=True))
            return
        previous = self._previous_significant_index(self.i)
        prev_token = self.tokens[previous] if previous is not None else None
        if prev_token is not None and prev_token.type == "punct" and prev_token.value in {
            "=", "(", ",", "[", ":", "??", "||", "&&", "=>",
        }:
            if prev_token.value == "=>":
                self.braces.append(_Brace("block", pops_frame=False))
                return
            frame = _Frame("object", None)
            self.frames.append(frame)
            self.braces.append(_Brace("object", pops_frame=True))
            return
        if prev_token is not None and prev_token.type == "keyword" and prev_token.value == "return":
            frame = _Frame("object", None)
            self.frames.append(frame)
            self.braces.append(_Brace("object", pops_frame=True))
            return
        if prev_token is not None and prev_token.type == "keyword" and prev_token.value in {
            "const", "let", "var",
        }:
            self.braces.append(_Brace("pattern", pops_frame=False))
            return
        self.braces.append(_Brace("block", pops_frame=False))

    def _close_brace(self, token: Token) -> None:
        if not self.braces:
            return
        entry = self.braces.pop()
        if entry.pops_frame and len(self.frames) > 1:
            self._pop_frame(token.line)
        self._close_expression_arrows(token)
        self._mark_member_expected()

    def _pop_frame(self, end_line: int) -> None:
        frame = self.frames.pop()
        if frame.def_index is not None:
            record = self.definitions[frame.def_index]
            record["end_line"] = end_line
            if frame.contains_jsx:
                record["contains_jsx"] = True
        if frame.contains_jsx:
            # JSX inside a nested scope also marks the enclosing function
            # (components often render through inner closures).
            for outer in reversed(self.frames):
                if outer.kind == "function":
                    outer.contains_jsx = True
                    break

    def _mark_member_expected(self) -> None:
        if self.braces and self.braces[-1].label in {"class_body", "object"}:
            self.braces[-1].expect_member = True

    def _close_expression_arrows(self, token: Token) -> None:
        while len(self.frames) > 1:
            frame = self.frames[-1]
            if not frame.expr_arrow:
                return
            start_paren, start_brace, start_special = frame.snapshot
            close = False
            if token.value == ")" and self.paren_depth < start_paren:
                close = True
            elif token.value in {",", ";"} and (
                self.paren_depth <= start_paren
                and len(self.braces) <= start_brace
                and self.special_depth <= start_special
            ):
                close = True
            elif token.value == "}" and len(self.braces) < start_brace:
                close = True
            if not close:
                return
            self._pop_frame(token.line)

    def _close_expression_arrows_on_special(self) -> None:
        while len(self.frames) > 1:
            frame = self.frames[-1]
            if not frame.expr_arrow or self.special_depth >= frame.snapshot[2]:
                return
            self._pop_frame(self.tokens[self.i].line if self.i < len(self.tokens) else 0)

    # ---- imports / exports ---------------------------------------------------

    def _handle_import(self, token: Token) -> None:
        if self._peek_value_is(1, "("):
            source = "<computed>"
            if self._peek_type(2) == "string":
                source = self.tokens[self.i + 2].value
            self.imports.append(
                ImportBinding(
                    local_name="import()",
                    source_module=source,
                    source_name=None,
                    kind="dynamic",
                    line=token.line,
                )
            )
            self.calls.append(
                RawCall(
                    caller_qualname=self._caller_qualname(),
                    callee_text=f"import({source!r})",
                    head="import",
                    attr_chain=(),
                    line=token.line,
                    is_dynamic=True,
                )
            )
            return
        j = self.i + 1
        end = min(len(self.tokens), self.i + MAX_LOOKAHEAD)
        if j < end and self.tokens[j].type == "string":
            self.imports.append(
                ImportBinding(
                    local_name="",
                    source_module=self.tokens[j].value,
                    source_name=None,
                    kind="side_effect",
                    line=token.line,
                )
            )
            self.i = j
            return
        if j < end and self.tokens[j].type == "keyword" and self.tokens[j].value == "type":
            j += 1  # TS `import type { ... }` — bindings are types only.
        default_name: str | None = None
        namespace_name: str | None = None
        named: list[tuple[str, str]] = []
        while j < end:
            item = self.tokens[j]
            if item.type == "string":
                source = item.value
                self._record_es_imports(
                    source, default_name, namespace_name, named, token.line
                )
                self.i = j
                return
            if item.type == "id" and default_name is None and not named and namespace_name is None:
                default_name = item.value
            elif item.type == "punct" and item.value == "*":
                if j + 2 < end and self.tokens[j + 1].value == "as":
                    namespace_name = self.tokens[j + 2].value
                    j += 2
            elif item.type == "punct" and item.value == "{":
                j += 1
                while j < end and not (
                    self.tokens[j].type == "punct" and self.tokens[j].value == "}"
                ):
                    spec = self.tokens[j]
                    if spec.type in {"id", "keyword"} and spec.value not in {"as", "type"}:
                        original = spec.value
                        local = original
                        if (
                            j + 2 < end
                            and self.tokens[j + 1].value == "as"
                            and self.tokens[j + 2].type in {"id", "keyword"}
                        ):
                            local = self.tokens[j + 2].value
                            j += 2
                        named.append((original, local))
                    j += 1
            elif item.type == "punct" and item.value == ";":
                break
            j += 1
        self.i = min(j, len(self.tokens) - 1) if self.tokens else 0

    def _record_es_imports(
        self,
        source: str,
        default_name: str | None,
        namespace_name: str | None,
        named: list[tuple[str, str]],
        line: int,
    ) -> None:
        category = EFFECT_IMPORT_ROOTS.get(_bare_root(source))
        if default_name:
            self.imports.append(
                ImportBinding(default_name, source, "default", "default", line)
            )
            if category:
                self._effect_import_categories[default_name] = category
        if namespace_name:
            self.imports.append(
                ImportBinding(namespace_name, source, None, "namespace", line)
            )
            if category:
                self._effect_import_categories[namespace_name] = category
        for original, local in named:
            self.imports.append(
                ImportBinding(local, source, original, "named", line)
            )
            if category:
                self._effect_import_categories[local] = category
        if not default_name and not namespace_name and not named:
            self.imports.append(ImportBinding("", source, None, "side_effect", line))

    def _handle_export(self, token: Token) -> None:
        next_type = self._peek_type(1)
        next_value = self.tokens[self.i + 1].value if self.i + 1 < len(self.tokens) else ""
        if next_type == "keyword" and next_value == "default":
            after = self.tokens[self.i + 2] if self.i + 2 < len(self.tokens) else None
            if after is not None and after.type == "keyword" and after.value in {
                "function", "class", "async",
            }:
                self._export_next_definition = ["default"]
                self.i += 1
                return
            if after is not None and after.type == "id":
                self.exports.append(
                    ExportBinding("default", after.value, None, None, "default", token.line)
                )
                self.i += 2
                return
            self.exports.append(
                ExportBinding("default", None, None, None, "default", token.line)
            )
            self.i += 1
            return
        if next_type == "punct" and next_value == "{":
            self._handle_export_clause(token)
            return
        if next_type == "punct" and next_value == "*":
            source = None
            alias = None
            j = self.i + 2
            end = min(len(self.tokens), self.i + MAX_LOOKAHEAD)
            while j < end:
                item = self.tokens[j]
                if item.value == "as" and j + 1 < end:
                    alias = self.tokens[j + 1].value
                    j += 1
                elif item.type == "string":
                    source = item.value
                    break
                elif item.value == ";":
                    break
                j += 1
            self.exports.append(
                ExportBinding(alias or "*", None, source, "*", "star", token.line)
            )
            self.i = j
            return
        if next_type == "keyword" and next_value in {
            "function", "class", "const", "let", "var", "async",
            "interface", "type", "enum",
        }:
            self._export_next_definition = ["named"]
            return

    def _handle_export_clause(self, token: Token) -> None:
        j = self.i + 2
        end = min(len(self.tokens), self.i + MAX_LOOKAHEAD)
        specifiers: list[tuple[str, str]] = []
        while j < end and not (
            self.tokens[j].type == "punct" and self.tokens[j].value == "}"
        ):
            spec = self.tokens[j]
            if spec.type in {"id", "keyword"} and spec.value not in {"as", "type"}:
                local = spec.value
                exported = local
                if (
                    j + 2 < end
                    and self.tokens[j + 1].value == "as"
                    and self.tokens[j + 2].type in {"id", "keyword", "string"}
                ):
                    exported = self.tokens[j + 2].value
                    j += 2
                specifiers.append((local, exported))
            j += 1
        source = None
        if (
            j + 2 < len(self.tokens)
            and self.tokens[j + 1].value == "from"
            and self.tokens[j + 2].type == "string"
        ):
            source = self.tokens[j + 2].value
            j += 2
        for local, exported in specifiers:
            if source is None:
                self.exports.append(
                    ExportBinding(exported, local, None, None, "named", token.line)
                )
            else:
                self.exports.append(
                    ExportBinding(exported, None, source, local, "named", token.line)
                )
        self.i = j

    def _consume_export_flag(self) -> tuple[bool, list[str]]:
        flags = self._export_next_definition
        self._export_next_definition = None
        if not flags:
            return False, []
        if "default" in flags:
            return True, ["default"]
        return True, []

    # ---- declarations ----------------------------------------------------------

    def _handle_declaration(self, token: Token) -> None:
        exported, export_names = self._consume_export_flag()
        keyword = token.value
        j = self.i + 1
        if j >= len(self.tokens):
            return
        first = self.tokens[j]
        if first.type == "punct" and first.value in {"{", "["}:
            self._maybe_destructured_require(token, exported)
            return
        if first.type != "id":
            return
        name = first.value
        at_module_level = self.frames[-1].kind == "module"
        if at_module_level and keyword in {"let", "var"}:
            self.module_let_names.add(name)
        # Find `=` for this declarator (skip TS type annotation).
        k = j + 1
        end = min(len(self.tokens), j + MAX_LOOKAHEAD)
        depth = 0
        while k < end:
            item = self.tokens[k]
            if item.type == "punct":
                if item.value in {"(", "["}:
                    depth += 1
                elif item.value in {")", "]"}:
                    depth -= 1
                elif item.value == "=" and depth == 0:
                    break
                elif item.value in {";", ","} and depth == 0:
                    return
                elif item.value == "{" and depth == 0:
                    return
            k += 1
        if k >= end or self.tokens[k].value != "=":
            return
        value_index = k + 1
        self._bind_declaration_value(
            name, value_index, token.line, exported, export_names
        )

    def _bind_declaration_value(
        self,
        name: str,
        value_index: int,
        line: int,
        exported: bool,
        export_names: list[str],
    ) -> None:
        if value_index >= len(self.tokens):
            return
        value = self.tokens[value_index]
        if value.type == "id" and value.value == "require":
            source = ""
            if (
                value_index + 2 < len(self.tokens)
                and self.tokens[value_index + 1].value == "("
                and self.tokens[value_index + 2].type == "string"
            ):
                source = self.tokens[value_index + 2].value
            self.imports.append(
                ImportBinding(name, source or "<computed>", None, "require", line)
            )
            category = EFFECT_IMPORT_ROOTS.get(_bare_root(source))
            if category:
                self._effect_import_categories[name] = category
            return
        if value.type == "keyword" and value.value == "new":
            constructed = self._chain_text_forward(value_index + 1)
            if constructed:
                self.frames[-1].local_types[name] = constructed
            return
        offset = 0
        is_async = False
        if value.type == "keyword" and value.value == "async":
            is_async = True
            offset = 1
            value = (
                self.tokens[value_index + 1]
                if value_index + 1 < len(self.tokens)
                else value
            )
        if value.type == "keyword" and value.value == "function":
            def_index = self._add_definition(
                name,
                KIND_ASYNC_FUNCTION if is_async else KIND_FUNCTION,
                line,
                parent=self._container_name(),
                exported=exported,
                export_names=export_names,
                signature=self._signature_after(value_index + offset + 1),
            )
            self._register_function_header(value_index + offset, def_index)
            return
        arrow_index = self._find_arrow(value_index + offset)
        if arrow_index is not None:
            def_index = self._add_definition(
                name,
                KIND_ASYNC_FUNCTION if is_async else KIND_FUNCTION,
                line,
                parent=self._container_name(),
                exported=exported,
                export_names=export_names,
                signature=self._signature_between(value_index + offset, arrow_index),
            )
            self._note_arrow(arrow_index, def_index, value_index + offset)
            return
        if value.type == "id" and value.value in REACT_WRAPPERS:
            inner = value_index + offset + 2  # wrapper ( ...
            arrow_index = self._find_arrow(inner)
            if arrow_index is not None:
                def_index = self._add_definition(
                    name,
                    KIND_FUNCTION,
                    line,
                    parent=self._container_name(),
                    exported=exported,
                    export_names=export_names,
                    decorators=(value.value,),
                )
                self._note_arrow(arrow_index, def_index, inner)
            return
        if value.type == "punct" and value.value == "{":
            pending = _Pending("object_value", None, self.paren_depth)
            pending.object_name = name
            self.pending_scopes.append(pending)
            if exported:
                self.exports.append(
                    ExportBinding(name, name, None, None, "named", line)
                )
            return
        if exported:
            self.exports.append(ExportBinding(name, name, None, None, "named", line))

    def _maybe_destructured_require(self, token: Token, exported: bool) -> None:
        j = self.i + 1
        end = min(len(self.tokens), self.i + MAX_LOOKAHEAD)
        names: list[str] = []
        depth = 0
        require_source: str | None = None
        while j < end:
            item = self.tokens[j]
            if item.type == "punct":
                if item.value in {"{", "["}:
                    depth += 1
                elif item.value in {"}", "]"}:
                    depth -= 1
                    if depth == 0:
                        # Look for `= require("...")`.
                        if (
                            j + 3 < end
                            and self.tokens[j + 1].value == "="
                            and self.tokens[j + 2].type == "id"
                            and self.tokens[j + 2].value == "require"
                            and self.tokens[j + 3].value == "("
                        ):
                            if (
                                j + 4 < end
                                and self.tokens[j + 4].type == "string"
                            ):
                                require_source = self.tokens[j + 4].value
                        break
                elif item.value == ";":
                    break
            elif item.type == "id" and depth >= 1:
                names.append(item.value)
            j += 1
        if require_source is None:
            # Plain destructuring (`const { cb = (x) => ... } = options`):
            # inside a function these are local bindings, and calling one is
            # an injected/dynamic callback, not a module-level reference.
            for frame in reversed(self.frames):
                if frame.kind == "function":
                    frame.local_bindings.update(
                        self._collect_binding_names(self.i + 1)
                    )
                    break
                if frame.kind == "module":
                    break
            return
        category = EFFECT_IMPORT_ROOTS.get(_bare_root(require_source))
        for name in names[:40]:
            self.imports.append(
                ImportBinding(name, require_source, name, "require", token.line)
            )
            if category:
                self._effect_import_categories[name] = category
        self.i = j

    # ---- functions / classes -----------------------------------------------

    def _handle_function_keyword(self, token: Token, name_hint: str | None) -> None:
        exported, export_names = self._consume_export_flag()
        previous = self._previous_significant_index(self.i)
        is_async = (
            previous is not None
            and self.tokens[previous].type == "keyword"
            and self.tokens[previous].value == "async"
        )
        j = self.i + 1
        if j < len(self.tokens) and self.tokens[j].type == "punct" and self.tokens[j].value == "*":
            j += 1
        name = name_hint
        if j < len(self.tokens) and self.tokens[j].type == "id":
            name = self.tokens[j].value
            j += 1
        if name is None:
            if "default" in export_names:
                name = "default"
            else:
                return  # anonymous function expression: no definition
        def_index = self._add_definition(
            name,
            KIND_ASYNC_FUNCTION if is_async else KIND_FUNCTION,
            token.line,
            parent=self._container_name(),
            exported=exported,
            export_names=export_names,
            signature=self._signature_after(j - 1),
        )
        self._register_function_header(self.i, def_index)

    def _register_function_header(self, header_index: int, def_index: int) -> None:
        end = min(len(self.tokens), header_index + MAX_LOOKAHEAD)
        pending = _Pending("function", def_index, self.paren_depth)
        for j in range(header_index, end):
            item = self.tokens[j]
            if item.type == "punct" and item.value == "(":
                self.non_call_parens.add(j)
                pending.param_bindings = self._collect_binding_names(j)
                self._note_param_scope(j, def_index)
                break
        self.pending_scopes.append(pending)

    def _note_arrow(self, arrow_index: int, def_index: int, params_start: int) -> None:
        """Register an arrow definition together with its parameter bindings."""
        self.arrow_defs[arrow_index] = def_index
        self.arrow_params[arrow_index] = self._collect_binding_names(params_start)
        self._param_scopes.append(
            (params_start, arrow_index, self.definitions[def_index]["qualname"])
        )

    def _note_param_scope(self, open_index: int, def_index: int) -> None:
        """Attribute calls inside a parameter list to the defined function."""
        depth = 0
        end = min(len(self.tokens), open_index + MAX_LOOKAHEAD)
        close = open_index
        for j in range(open_index, end):
            item = self.tokens[j]
            if item.type == "punct":
                if item.value == "(":
                    depth += 1
                elif item.value == ")":
                    depth -= 1
                    if depth == 0:
                        close = j
                        break
        if close > open_index:
            self._param_scopes.append(
                (open_index, close, self.definitions[def_index]["qualname"])
            )

    def _handle_class_keyword(self, token: Token) -> None:
        exported, export_names = self._consume_export_flag()
        j = self.i + 1
        name = "default"
        if j < len(self.tokens) and self.tokens[j].type == "id":
            name = self.tokens[j].value
            j += 1
        bases: list[str] = []
        end = min(len(self.tokens), self.i + MAX_LOOKAHEAD)
        while j < end:
            item = self.tokens[j]
            if item.type == "keyword" and item.value == "extends":
                base = self._chain_text_forward(j + 1)
                if base:
                    bases.append(base)
            if item.type == "punct" and item.value == "{":
                break
            j += 1
        def_index = self._add_definition(
            name,
            KIND_CLASS,
            token.line,
            parent=self._container_name(),
            exported=exported,
            export_names=export_names,
            signature=f"class {name}" + (f" extends {bases[0]}" if bases else ""),
            decorators=tuple(self._pending_decorators),
        )
        self._pending_decorators = []
        qualname = self.definitions[def_index]["qualname"]
        self.class_bases[qualname] = tuple(bases)
        self.pending_scopes.append(_Pending("class", def_index, self.paren_depth))

    def _handle_class_member(self, token: Token) -> None:
        brace = self.braces[-1]
        j = self.i
        end = min(len(self.tokens), self.i + 24)
        is_async = False
        modifiers_seen = 0
        while j < end:
            item = self.tokens[j]
            if item.type in {"id", "keyword"} and item.value in MODIFIER_TOKENS:
                next_item = self.tokens[j + 1] if j + 1 < len(self.tokens) else None
                if next_item is not None and next_item.type == "punct" and next_item.value == "(":
                    break  # modifier word is actually the method name
                if item.value == "async":
                    is_async = True
                j += 1
                modifiers_seen += 1
                continue
            if item.type == "punct" and item.value in {"*", "#"}:
                j += 1
                continue
            break
        if j >= len(self.tokens):
            return
        name_token = self.tokens[j]
        if name_token.type not in {"id", "keyword", "string", "number"}:
            return
        name = name_token.value
        after = self.tokens[j + 1] if j + 1 < len(self.tokens) else None
        brace.expect_member = False
        class_frame = self.frames[-1]
        parent = class_frame.qualname or ""
        if after is not None and after.type == "punct" and after.value == "(":
            def_index = self._add_definition(
                name,
                KIND_ASYNC_METHOD if is_async else KIND_METHOD,
                name_token.line,
                parent=parent,
                signature=self._signature_after(j),
                decorators=tuple(self._pending_decorators),
            )
            self._pending_decorators = []
            self.non_call_parens.add(j + 1)
            pending = _Pending("function", def_index, self.paren_depth)
            pending.param_bindings = self._collect_binding_names(j + 1)
            self._note_param_scope(j + 1, def_index)
            self.pending_scopes.append(pending)
            self.i = j
            return
        if after is not None and after.type == "punct" and after.value == "=":
            arrow_index = self._find_arrow(j + 2)
            if arrow_index is not None:
                def_index = self._add_definition(
                    name,
                    KIND_ASYNC_METHOD if is_async else KIND_METHOD,
                    name_token.line,
                    parent=parent,
                    decorators=tuple(self._pending_decorators),
                )
                self._pending_decorators = []
                self._note_arrow(arrow_index, def_index, j + 2)
                self.i = j + 1
                return
        self.i = j if modifiers_seen else self.i

    def _handle_object_member(self, token: Token) -> None:
        brace = self.braces[-1]
        j = self.i
        is_async = False
        while (
            self.tokens[j].value in {"async", "get", "set"}
            and j + 1 < len(self.tokens)
            and self.tokens[j + 1].type in {"id", "keyword", "string"}
        ):
            if self.tokens[j].value == "async":
                is_async = True
            j += 1
        name_token = self.tokens[j]
        if name_token.type not in {"id", "keyword", "string"}:
            return
        name = name_token.value
        after = self.tokens[j + 1] if j + 1 < len(self.tokens) else None
        container = self._object_container_name()
        if after is not None and after.type == "punct" and after.value == "(":
            brace.expect_member = False
            def_index = self._add_definition(
                name,
                KIND_ASYNC_METHOD if is_async else KIND_METHOD,
                name_token.line,
                parent=container,
                signature=self._signature_after(j),
            )
            self.non_call_parens.add(j + 1)
            pending = _Pending("function", def_index, self.paren_depth)
            pending.param_bindings = self._collect_binding_names(j + 1)
            self._note_param_scope(j + 1, def_index)
            self.pending_scopes.append(pending)
            self.i = j
            return
        if after is not None and after.type == "punct" and after.value == ":":
            brace.expect_member = False
            value_index = j + 2
            value = (
                self.tokens[value_index] if value_index < len(self.tokens) else None
            )
            if value is None:
                return
            if value.type == "keyword" and value.value in {"async", "function"}:
                inner_async = value.value == "async"
                fn_index = value_index + (1 if inner_async else 0)
                if (
                    fn_index < len(self.tokens)
                    and self.tokens[fn_index].type == "keyword"
                    and self.tokens[fn_index].value == "function"
                ):
                    def_index = self._add_definition(
                        name,
                        KIND_ASYNC_METHOD if inner_async else KIND_METHOD,
                        name_token.line,
                        parent=container,
                    )
                    self._register_function_header(fn_index, def_index)
                    self.i = j + 1
                    return
            arrow_index = self._find_arrow(value_index)
            if arrow_index is not None:
                def_index = self._add_definition(
                    name,
                    KIND_ASYNC_METHOD if is_async else KIND_METHOD,
                    name_token.line,
                    parent=container,
                )
                self._note_arrow(arrow_index, def_index, value_index)
                self.i = j + 1
                return
            if value.type == "punct" and value.value == "{":
                pending = _Pending("object_value", None, self.paren_depth)
                pending.object_name = (
                    f"{container}.{name}" if container else name
                )
                self.pending_scopes.append(pending)
                self.i = j + 1
                return
            self.i = j + 1
            return
        # Shorthand property or spread: nothing to record.
        brace.expect_member = False

    def _handle_arrow(self, token: Token) -> None:
        def_index = self.arrow_defs.pop(self.i, None)
        params = self.arrow_params.pop(self.i, None)
        next_token = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
        if def_index is None:
            return  # anonymous arrow: calls attribute to the enclosing scope
        qualname = self._def_qualname(def_index)
        if next_token is not None and next_token.type == "punct" and next_token.value == "{":
            frame = _Frame("function", qualname, def_index=def_index)
            frame.awaiting_body_brace = True
            frame.local_bindings.update(params or {})
            self.frames.append(frame)
            return
        frame = _Frame(
            "function",
            qualname,
            def_index=def_index,
            expr_arrow=True,
            snapshot=(self.paren_depth, len(self.braces), self.special_depth),
        )
        frame.local_bindings.update(params or {})
        self.frames.append(frame)

    # ---- TS type declarations -------------------------------------------------

    def _handle_type_declaration(self, token: Token) -> None:
        exported, export_names = self._consume_export_flag()
        if token.value == "namespace":
            return
        j = self.i + 1
        if j >= len(self.tokens) or self.tokens[j].type != "id":
            return
        name = self.tokens[j].value
        self._add_definition(
            name,
            KIND_TYPE,
            token.line,
            parent=self._container_name(),
            exported=exported,
            export_names=export_names,
            signature=f"{token.value} {name}",
        )
        self.pending_scopes.append(_Pending("type_body", None, self.paren_depth))
        self.i = j

    def _handle_type_alias(self, token: Token) -> None:
        exported, export_names = self._consume_export_flag()
        name = self.tokens[self.i + 1].value
        self._add_definition(
            name,
            KIND_TYPE,
            token.line,
            parent=self._container_name(),
            exported=exported,
            export_names=export_names,
            signature=f"type {name}",
        )
        # Skip the alias body up to a plausible statement end.
        j = self.i + 3
        end = min(len(self.tokens), self.i + 2000)
        depth = 0
        while j < end:
            item = self.tokens[j]
            if item.type == "punct":
                if item.value in {"{", "(", "["}:
                    depth += 1
                elif item.value in {"}", ")", "]"}:
                    depth -= 1
                    if depth < 0:
                        break
                elif item.value == ";" and depth == 0:
                    break
            elif item.type == "keyword" and depth == 0 and item.value in {
                "const", "let", "var", "function", "class", "export",
                "import", "interface", "type", "enum", "return",
            }:
                j -= 1
                break
            j += 1
        self.i = min(j, len(self.tokens) - 1)

    # ---- throw / catch --------------------------------------------------------

    def _handle_throw(self, token: Token) -> None:
        info = self._error_info()
        label = "throw"
        j = self.i + 1
        if j < len(self.tokens):
            item = self.tokens[j]
            if item.type == "keyword" and item.value == "new" and j + 1 < len(self.tokens):
                label = self._chain_text_forward(j + 1) or "throw new"
            elif item.type == "id":
                label = item.value
        info.raises.append(label)
        info.raise_lines.append(token.line)

    def _handle_catch(self, token: Token) -> None:
        info = self._error_info()
        j = self.i + 1
        end = min(len(self.tokens), self.i + 30)
        param = ""
        while j < end:
            item = self.tokens[j]
            if item.type == "punct" and item.value == "{":
                break
            if item.type == "id" and not param:
                param = item.value
            j += 1
        label = f"catch ({param})" if param else "catch"
        info.catches.append(label)
        info.catch_lines.append(token.line)
        if (
            j + 1 < len(self.tokens)
            and self.tokens[j + 1].type == "punct"
            and self.tokens[j + 1].value == "}"
        ):
            info.swallowed.append(token.line)

    def _error_info(self) -> _JsErrorInfo:
        qualname = self._caller_qualname() or "<module>"
        return self.error_infos.setdefault(qualname, _JsErrorInfo())

    # ---- calls and effects ------------------------------------------------------

    def _maybe_member_or_call(self, token: Token) -> None:
        next_token = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
        if token.type == "id" and token.value == "process":
            self._maybe_process_env(token)
        if (
            next_token is not None
            and next_token.type == "punct"
            and next_token.value == "="
            and token.type == "id"
        ):
            return  # handled by the `=` effect checks

    def _maybe_process_env(self, token: Token) -> None:
        if (
            self._peek_value_is(1, ".")
            and self.i + 2 < len(self.tokens)
            and self.tokens[self.i + 2].value == "env"
        ):
            qualname = self._caller_qualname()
            is_write = False
            j = self.i + 3
            if j < len(self.tokens) and self.tokens[j].value == ".":
                j += 2
            if j < len(self.tokens) and self.tokens[j].type == "punct" and self.tokens[j].value == "=":
                is_write = True
            if is_write:
                self._add_effect(
                    EFFECT_ENVIRONMENT, "process.env write", token.line, qualname
                )
            elif qualname not in self._env_read_noted:
                self._env_read_noted.add(qualname)
                self._add_effect(
                    EFFECT_ENVIRONMENT, "process.env (read)", token.line, qualname
                )

    def _maybe_record_call(self, token: Token) -> None:
        parts, head_index, dynamic, anchored = self._chain_before(self.i)
        if not parts and not dynamic:
            return
        caller = self._caller_qualname()
        is_new = False
        is_await = False
        if head_index is not None:
            before = self._previous_significant_index(head_index)
            if before is not None:
                prior = self.tokens[before]
                if prior.type == "keyword" and prior.value == "new":
                    is_new = True
                    before2 = self._previous_significant_index(before)
                    if before2 is not None and self.tokens[before2].value == "await":
                        is_await = True
                elif prior.type == "keyword" and prior.value == "await":
                    is_await = True
        if anchored:
            # Method chained onto a call/subscript result: receiver unknown.
            head = ""
            chain = tuple(parts)
            callee_text = "(…)." + ".".join(parts)
        else:
            head = parts[0] if parts else ""
            chain = tuple(parts[1:])
            callee_text = ".".join(parts) if parts else "(dynamic)()"
        receiver_type = None
        if head:
            for frame in reversed(self.frames):
                if head in frame.local_types:
                    receiver_type = frame.local_types[head]
                    break
        is_local, local_default_line = (
            self._lookup_local_binding(head) if head else (False, None)
        )
        next_token = (
            self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
        )
        has_arguments = not (
            next_token is not None
            and next_token.type == "punct"
            and next_token.value == ")"
        )
        call = RawCall(
            caller_qualname=caller,
            callee_text=callee_text,
            head=head,
            attr_chain=chain,
            line=token.line,
            is_new=is_new,
            is_await=is_await,
            is_dynamic=dynamic,
            argument_names=self._argument_names(self.i),
            receiver_type=receiver_type,
            has_arguments=has_arguments,
            is_local_binding=is_local,
            local_default_line=local_default_line,
            callback_args=self._object_literal_callback_args(self.i),
        )
        self.calls.append(call)
        self._classify_js_effect(call)
        if chain and chain[-1] == "catch":
            self._error_info().catches.append(".catch()")
            self._error_info().catch_lines.append(token.line)

    def _chain_before(
        self, paren_index: int
    ) -> tuple[list[str], int | None, bool, bool]:
        """Callee chain ending at ``(``: (parts, head index, dynamic, anchored)."""
        j = paren_index - 1
        if j < 0:
            return [], None, False, False
        token = self.tokens[j]
        if token.type == "punct" and token.value == ">":
            balanced = self._skip_generic_backwards(j)
            if balanced is not None:
                j = balanced - 1
                if j < 0:
                    return [], None, False, False
                token = self.tokens[j]
        if token.type == "punct" and token.value in {")", "]"}:
            return [], None, True, False
        parts: list[str] = []
        head_index: int | None = None
        expecting_name = True
        anchored = False
        while j >= 0 and len(parts) < 12:
            token = self.tokens[j]
            if expecting_name:
                if token.type == "punct" and token.value == "!":
                    j -= 1
                    continue
                # Property position accepts any word, including keywords
                # (`api.delete`, `axios.get`); the head is validated below.
                if token.type in {"id", "keyword"}:
                    name = token.value
                    if (
                        j - 1 >= 0
                        and self.tokens[j - 1].type == "punct"
                        and self.tokens[j - 1].value == "#"
                    ):
                        name = "#" + name
                        j -= 1
                    parts.append(name)
                    head_index = j
                    expecting_name = False
                    j -= 1
                    continue
                if parts and token.type == "punct" and token.value in {")", "]"}:
                    anchored = True
                    head_index = None
                break
            if token.type == "punct" and token.value in {".", "?."}:
                expecting_name = True
                j -= 1
                continue
            break
        parts.reverse()
        if not parts:
            return [], None, False, False
        if not anchored:
            head_token = self.tokens[head_index] if head_index is not None else None
            if (
                head_token is not None
                and head_token.type == "keyword"
                and head_token.value in NON_CALL_HEAD_KEYWORDS
            ):
                return [], None, False, False
        return parts, head_index, False, anchored

    def _skip_generic_backwards(self, close_index: int) -> int | None:
        depth = 0
        j = close_index
        lower = max(0, close_index - 40)
        while j >= lower:
            value = self.tokens[j].value if self.tokens[j].type == "punct" else ""
            if value == ">":
                depth += 1
            elif value == "<":
                depth -= 1
                if depth == 0:
                    return j
            j -= 1
        return None

    def _argument_names(self, paren_index: int) -> tuple[str, ...]:
        names: list[str] = []
        depth = 0
        j = paren_index
        end = min(len(self.tokens), paren_index + 80)
        while j < end:
            token = self.tokens[j]
            if token.type == "punct":
                if token.value == "(":
                    depth += 1
                elif token.value == ")":
                    depth -= 1
                    if depth == 0:
                        break
            if token.type == "id" and depth == 1:
                prev_token = self.tokens[j - 1]
                next_token = self.tokens[j + 1] if j + 1 < len(self.tokens) else None
                prev_ok = prev_token.type == "punct" and prev_token.value in {"(", ","}
                next_ok = next_token is not None and next_token.type == "punct" and next_token.value in {",", ")"}
                if prev_ok and next_ok:
                    names.append(token.value)
                    if len(names) >= MAX_ARGUMENT_NAMES:
                        break
            j += 1
        return tuple(names)

    def _object_literal_callback_args(self, paren_index: int) -> tuple[tuple[str, str], ...]:
        """Identifier-valued properties of object literals passed as arguments.

        Supports the common options-object pattern: for
        ``f(rows, { getDueDate, isEligiblePayment: shouldShowRow })`` this
        yields (("getDueDate", "getDueDate"), ("isEligiblePayment",
        "shouldShowRow")). Non-identifier values are skipped.
        """
        pairs: list[tuple[str, str]] = []
        depth = 0
        object_level: int | None = None
        j = paren_index
        end = min(len(self.tokens), paren_index + MAX_LOOKAHEAD)
        while j < end and len(pairs) < 8:
            token = self.tokens[j]
            if token.type == "punct":
                if token.value in {"(", "[", "{"}:
                    if token.value == "{" and depth == 1 and object_level is None:
                        object_level = depth + 1
                    depth += 1
                elif token.value in {")", "]", "}"}:
                    depth -= 1
                    if token.value == "}" and object_level is not None and depth < object_level:
                        object_level = None
                    if depth == 0:
                        break
            elif (
                token.type in {"id", "keyword"}
                and object_level is not None
                and depth == object_level
            ):
                next_token = self.tokens[j + 1] if j + 1 < len(self.tokens) else None
                next_value = (
                    next_token.value
                    if next_token is not None and next_token.type == "punct"
                    else ""
                )
                if next_value in {",", "}"} and token.type == "id":
                    pairs.append((token.value, token.value))  # shorthand
                    j += 1
                    continue
                if next_value == ":":
                    value_token = (
                        self.tokens[j + 2] if j + 2 < len(self.tokens) else None
                    )
                    after_value = (
                        self.tokens[j + 3] if j + 3 < len(self.tokens) else None
                    )
                    if (
                        value_token is not None
                        and value_token.type == "id"
                        and after_value is not None
                        and after_value.type == "punct"
                        and after_value.value in {",", "}"}
                    ):
                        pairs.append((token.value, value_token.value))
                        j += 3
                        continue
                    # Complex value: skip to the property end at this level.
                    j = self._skip_to_binding_end(j + 2)
                    continue
            j += 1
        return tuple(pairs)

    def _classify_js_effect(self, call: RawCall) -> None:
        head = call.head
        chain = call.attr_chain
        line = call.line
        qualname = call.caller_qualname
        if not head:
            return
        category = self._effect_import_categories.get(head)
        if category is not None:
            self._add_effect(category, call.callee_text, line, qualname)
            return
        if head == "fetch" and not chain:
            self._add_effect(EFFECT_NETWORK, "fetch()", line, qualname)
            return
        if head in TIMER_HEADS and not chain:
            self._add_effect(EFFECT_CONCURRENCY, f"{head}()", line, qualname)
            return
        if head in {"alert", "confirm", "prompt"} and not chain:
            self._add_effect(EFFECT_DOM_STORAGE, f"{head}()", line, qualname)
            return
        if call.is_new and head in {"XMLHttpRequest", "WebSocket", "EventSource"}:
            self._add_effect(EFFECT_NETWORK, f"new {head}()", line, qualname)
            return
        if call.is_new and head == "Worker":
            self._add_effect(EFFECT_CONCURRENCY, "new Worker()", line, qualname)
            return
        if call.is_new and head == "Date" and not chain:
            # Only zero-argument `new Date()` reads the clock; `new Date(value)`
            # and `new Date(y, m, d)` construct deterministically from inputs.
            if not call.has_arguments:
                self._add_effect(EFFECT_TIME_RANDOM, "new Date()", line, qualname)
            return
        if head == "console" and chain:
            self._add_effect(EFFECT_CONSOLE_LOGGING, call.callee_text, line, qualname)
            return
        if head == "Math" and chain[:1] == ("random",):
            self._add_effect(EFFECT_TIME_RANDOM, "Math.random()", line, qualname)
            return
        if head == "Date" and chain[:1] == ("now",):
            self._add_effect(EFFECT_TIME_RANDOM, "Date.now()", line, qualname)
            return
        if head == "crypto" and chain and chain[0] in {"getRandomValues", "randomUUID"}:
            self._add_effect(EFFECT_TIME_RANDOM, call.callee_text, line, qualname)
            return
        if head == "process" and chain[:1] in {("exit",), ("kill",), ("abort",)}:
            self._add_effect(EFFECT_PROCESS, call.callee_text, line, qualname)
            return
        if head in STORAGE_OBJECTS:
            self._add_effect(EFFECT_DOM_STORAGE, call.callee_text, line, qualname)
            return
        if head == "history" and chain[:1] in {("pushState",), ("replaceState",)}:
            self._add_effect(EFFECT_DOM_STORAGE, call.callee_text, line, qualname)
            return
        if head == "navigator" and chain[:1] == ("sendBeacon",):
            self._add_effect(EFFECT_NETWORK, call.callee_text, line, qualname)
            return
        if head in {"document", "window"}:
            last = chain[-1] if chain else ""
            if last in DOM_MUTATION_METHODS:
                self._add_effect(EFFECT_DOM_STORAGE, call.callee_text, line, qualname)
            elif last == "addEventListener":
                self._add_effect(
                    EFFECT_DOM_STORAGE,
                    f"{call.callee_text} (listener registration)",
                    line,
                    qualname,
                )
            return
        last = chain[-1] if chain else ""
        if last in DOM_MUTATION_METHODS and len(chain) >= 1:
            self._add_effect(
                EFFECT_DOM_STORAGE,
                f"{call.callee_text} (method-name match)",
                line,
                qualname,
                resolution=RESOLUTION_INFERRED,
            )
            return
        if len(chain) >= 2 and chain[-2] == "classList" and chain[-1] in CLASSLIST_METHODS:
            self._add_effect(
                EFFECT_DOM_STORAGE,
                call.callee_text,
                line,
                qualname,
                resolution=RESOLUTION_INFERRED,
            )
            return
        if head == "addEventListener" and not chain:
            self._add_effect(
                EFFECT_DOM_STORAGE, "addEventListener()", line, qualname
            )

    def _maybe_global_state_write(self, token: Token) -> None:
        previous = self._previous_significant_index(self.i)
        if previous is None:
            return
        target = self.tokens[previous]
        if target.type != "id":
            return
        if target.value not in self.module_let_names:
            return
        if self._caller_qualname() is None:
            return
        before = self._previous_significant_index(previous)
        if before is not None and self.tokens[before].type == "punct" and self.tokens[before].value in {".", "?."}:
            return
        next_token = self.tokens[self.i + 1] if self.i + 1 < len(self.tokens) else None
        if next_token is not None and next_token.type == "punct" and next_token.value == "=":
            return  # `==` handled as comparison
        self._add_effect(
            EFFECT_GLOBAL_STATE,
            f"assigns module-level `{target.value}`",
            token.line,
            self._caller_qualname(),
            resolution=RESOLUTION_INFERRED,
        )

    def _maybe_html_sink_write(self, token: Token) -> None:
        previous = self._previous_significant_index(self.i)
        if previous is None:
            return
        target = self.tokens[previous]
        if target.type == "id" and target.value in {"innerHTML", "outerHTML"}:
            before = self._previous_significant_index(previous)
            if before is not None and self.tokens[before].type == "punct" and self.tokens[before].value in {".", "?."}:
                self._add_effect(
                    EFFECT_DOM_STORAGE,
                    f"{target.value} assignment",
                    token.line,
                    self._caller_qualname(),
                )

    def _add_effect(
        self,
        category: str,
        detail: str,
        line: int,
        qualname: str | None,
        *,
        resolution: str = RESOLUTION_CONFIRMED,
    ) -> None:
        if len(self.side_effects) >= MAX_EFFECTS_PER_FILE:
            return
        self.side_effects.append(
            SideEffect(
                category=category,
                detail=detail,
                path=self.path,
                line=line,
                qualname=qualname,
                resolution=resolution,
            )
        )

    # ---- shared helpers -----------------------------------------------------

    def _add_definition(
        self,
        name: str,
        kind: str,
        line: int,
        *,
        parent: str | None = None,
        exported: bool = False,
        export_names: list[str] | tuple[str, ...] = (),
        signature: str = "",
        decorators: tuple[str, ...] = (),
    ) -> int:
        qualname = f"{parent}.{name}" if parent else name
        record = {
            "name": name,
            "qualname": qualname,
            "kind": kind,
            "line": line,
            "end_line": None,
            "parent": parent,
            "exported": exported or bool(export_names),
            "export_names": list(export_names),
            "signature": signature[:MAX_SIGNATURE_CHARS],
            "decorators": list(decorators),
            "contains_jsx": False,
        }
        self.definitions.append(record)
        index = len(self.definitions) - 1
        if "default" in export_names:
            self.exports.append(
                ExportBinding("default", name, None, None, "default", line)
            )
        elif exported and parent is None:
            self.exports.append(
                ExportBinding(name, name, None, None, "named", line)
            )
        return index

    def _def_qualname(self, def_index: int | None) -> str | None:
        if def_index is None:
            return None
        return str(self.definitions[def_index]["qualname"])

    def _container_name(self) -> str | None:
        for frame in reversed(self.frames):
            if frame.kind == "function" and frame.qualname:
                return frame.qualname
            if frame.kind == "class" and frame.qualname:
                return frame.qualname
        return None

    def _object_container_name(self) -> str | None:
        for frame in reversed(self.frames):
            if frame.kind == "object":
                return frame.object_name
            if frame.kind in {"function", "class"}:
                return frame.qualname
        return None

    def _caller_qualname(self) -> str | None:
        # Defaults in a parameter list execute inside the function being
        # defined, so calls/effects there belong to it, not the outer scope.
        while self._param_scopes and self._param_scopes[0][1] < self.i:
            self._param_scopes.pop(0)
        for start, end_index, qualname in reversed(self._param_scopes):
            if start <= self.i <= end_index:
                return qualname
        for frame in reversed(self.frames):
            if frame.kind == "function":
                return frame.qualname
        return None

    def _find_arrow(self, start: int) -> int | None:
        """Token index of the `=>` belonging to a maybe-arrow at ``start``."""
        if start >= len(self.tokens):
            return None
        j = start
        first = self.tokens[j]
        if first.type == "keyword" and first.value == "async":
            j += 1
            if j >= len(self.tokens):
                return None
            first = self.tokens[j]
        if first.type == "id":
            next_token = self.tokens[j + 1] if j + 1 < len(self.tokens) else None
            if next_token is not None and next_token.type == "punct" and next_token.value == "=>":
                return j + 1
            return None
        if first.type == "punct" and first.value == "(":
            depth = 0
            end = min(len(self.tokens), start + MAX_LOOKAHEAD)
            while j < end:
                item = self.tokens[j]
                if item.type == "punct":
                    if item.value == "(":
                        depth += 1
                    elif item.value == ")":
                        depth -= 1
                        if depth == 0:
                            k = j + 1
                            # Skip a TS return-type annotation `: Type`.
                            if k < end and self.tokens[k].type == "punct" and self.tokens[k].value == ":":
                                k += 1
                                type_depth = 0
                                while k < end:
                                    entry = self.tokens[k]
                                    if entry.type == "punct":
                                        if entry.value in {"(", "[", "{", "<"}:
                                            type_depth += 1
                                        elif entry.value in {")", "]", "}", ">"}:
                                            type_depth -= 1
                                            if type_depth < 0:
                                                return None
                                        elif entry.value == "=>" and type_depth == 0:
                                            return k
                                        elif entry.value in {";", ","} and type_depth == 0:
                                            return None
                                    k += 1
                                return None
                            if k < end and self.tokens[k].type == "punct" and self.tokens[k].value == "=>":
                                return k
                            return None
                j += 1
        return None

    def _collect_binding_names(self, start: int) -> dict[str, int | None]:
        """Names bound by a parameter list or destructuring pattern.

        Maps each binding to the line of its default value when that default
        is a function/arrow (a local default implementation), else None.
        Identifiers inside default expressions, object-pattern keys with an
        alias, and TS type annotations are not bound.
        """
        bindings: dict[str, int | None] = {}
        if start >= len(self.tokens):
            return bindings
        first = self.tokens[start]
        if first.type == "keyword" and first.value == "async":
            start += 1
            if start >= len(self.tokens):
                return bindings
            first = self.tokens[start]
        if first.type == "id":
            bindings[first.value] = None  # single-parameter arrow
            return bindings
        if not (first.type == "punct" and first.value in {"(", "{", "["}):
            return bindings
        stack: list[str] = []
        j = start
        end = min(len(self.tokens), start + MAX_LOOKAHEAD)
        while j < end and len(bindings) < 24:
            item = self.tokens[j]
            if item.type == "punct":
                if item.value in {"(", "{", "["}:
                    stack.append(item.value)
                elif item.value in {")", "}", "]"}:
                    if stack:
                        stack.pop()
                    if not stack:
                        break
                j += 1
                continue
            if item.type in {"id", "keyword"} and stack:
                next_token = self.tokens[j + 1] if j + 1 < len(self.tokens) else None
                next_value = (
                    next_token.value
                    if next_token is not None and next_token.type == "punct"
                    else ""
                )
                if next_value == ":":
                    if stack[-1] == "{":
                        # Object-pattern key: the alias/pattern after `:` binds.
                        j += 2
                        continue
                    # TS parameter annotation: bind the name, skip the type.
                    if item.type == "id":
                        bindings[item.value] = None
                    j = self._skip_to_binding_end(j + 2)
                    continue
                if next_value == "=":
                    if item.type == "id":
                        bindings[item.value] = self._default_impl_line(j + 2)
                    j = self._skip_to_binding_end(j + 2)
                    continue
                if next_value in {",", ")", "}", "]"}:
                    if item.type == "id":
                        bindings[item.value] = None
                    j += 1
                    continue
            j += 1
        return bindings

    def _skip_to_binding_end(self, start: int) -> int:
        """Index of the `,` or closing bracket ending the current binding."""
        j = start
        end = min(len(self.tokens), start + MAX_LOOKAHEAD)
        level = 0
        while j < end:
            item = self.tokens[j]
            if item.type == "punct":
                if item.value in {"(", "{", "["}:
                    level += 1
                elif item.value in {")", "}", "]"}:
                    if level == 0:
                        return j  # caller pops this closer
                    level -= 1
                elif item.value == "," and level == 0:
                    return j
            j += 1
        return j

    def _default_impl_line(self, value_index: int) -> int | None:
        """Line of a binding default when it is a function/arrow literal."""
        if value_index >= len(self.tokens):
            return None
        token = self.tokens[value_index]
        inner = value_index
        if token.type == "keyword" and token.value == "async":
            inner += 1
            if inner >= len(self.tokens):
                return None
        if self.tokens[inner].type == "keyword" and self.tokens[inner].value == "function":
            return token.line
        if self._find_arrow(value_index) is not None:
            return token.line
        return None

    def _lookup_local_binding(self, name: str) -> tuple[bool, int | None]:
        """Whether ``name`` is a parameter/destructured binding in scope."""
        for frame in reversed(self.frames):
            if name in frame.local_bindings:
                return True, frame.local_bindings[name]
            if frame.kind == "module":
                break
        return False, None

    def _signature_after(self, name_index: int) -> str:
        """Bounded raw parameter text after a function/method name token."""
        j = name_index + 1
        end = min(len(self.tokens), name_index + MAX_LOOKAHEAD)
        while j < end and not (
            self.tokens[j].type == "punct" and self.tokens[j].value == "("
        ):
            j += 1
        if j >= end:
            return ""
        return self._signature_between(j, None)

    def _signature_between(self, open_index: int, stop_index: int | None) -> str:
        j = open_index
        end = min(len(self.tokens), open_index + MAX_LOOKAHEAD)
        if stop_index is not None:
            end = min(end, stop_index)
        depth = 0
        pieces: list[str] = []
        while j < end:
            item = self.tokens[j]
            if item.type == "punct":
                if item.value == "(":
                    depth += 1
                elif item.value == ")":
                    depth -= 1
                    if depth == 0:
                        pieces.append(")")
                        break
            if item.type == "string":
                pieces.append(f'"{item.value}"')
            else:
                pieces.append(item.value)
            j += 1
        text = " ".join(pieces)
        text = text.replace("( ", "(").replace(" )", ")").replace(" ,", ",")
        text = text.replace(" :", ":").replace(" .", ".").replace(". ", ".")
        return text[:MAX_SIGNATURE_CHARS]

    def _chain_text_forward(self, start: int) -> str:
        parts: list[str] = []
        j = start
        end = min(len(self.tokens), start + 12)
        expecting = True
        while j < end:
            item = self.tokens[j]
            if expecting and item.type in {"id", "keyword"}:
                parts.append(item.value)
                expecting = False
                j += 1
                continue
            if not expecting and item.type == "punct" and item.value in {".", "?."}:
                expecting = True
                j += 1
                continue
            break
        return ".".join(parts)

    def _previous_significant_index(self, index: int) -> int | None:
        j = index - 1
        while j >= 0:
            token = self.tokens[j]
            if token.type in {"jsx_open", "jsx_close", "jsx_attr"}:
                return None
            return j
        return None

    def _peek_type(self, offset: int) -> str:
        index = self.i + offset
        return self.tokens[index].type if index < len(self.tokens) else ""

    def _peek_value_is(self, offset: int, value: str) -> bool:
        index = self.i + offset
        return index < len(self.tokens) and self.tokens[index].value == value

    def _in_type_body(self) -> bool:
        return bool(self.braces) and self.braces[-1].label == "type_body"


def _bare_root(source: str) -> str:
    cleaned = source[5:] if source.startswith("node:") else source
    if cleaned in EFFECT_IMPORT_ROOTS:
        return cleaned
    return cleaned.split("/")[0]
