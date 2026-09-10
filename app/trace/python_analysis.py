"""Python source analysis for code tracing, built on the stdlib ``ast``.

Produces the shared ``ModuleIndex`` shape: definitions with qualified names,
import/export bindings, call sites with receiver-type hints, side-effect
signals, and per-function error handling. Pure string/AST work only — this
module never opens files, executes code, or imports analyzed modules.
"""

from __future__ import annotations

import ast
from pathlib import PurePosixPath

from app.trace.model import (
    EFFECT_CONCURRENCY,
    EFFECT_CONSOLE_LOGGING,
    EFFECT_DATABASE,
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
    KIND_CONSTANT,
    KIND_FUNCTION,
    KIND_METHOD,
    LANGUAGE_PYTHON,
    ModuleIndex,
    RawCall,
    RESOLUTION_CONFIRMED,
    RESOLUTION_INFERRED,
    SideEffect,
    SymbolDefinition,
)


MAX_SIGNATURE_CHARS = 200
MAX_DOC_CHARS = 160
MAX_CALLEE_TEXT_CHARS = 120
MAX_ARGUMENT_NAMES = 6

# Root modules classified by the side effect their calls perform.
EFFECT_MODULE_ROOTS = {
    "shutil": EFFECT_FILESYSTEM,
    "tempfile": EFFECT_FILESYSTEM,
    "subprocess": EFFECT_PROCESS,
    "multiprocessing": EFFECT_PROCESS,
    "socket": EFFECT_NETWORK,
    "requests": EFFECT_NETWORK,
    "httpx": EFFECT_NETWORK,
    "urllib": EFFECT_NETWORK,
    "aiohttp": EFFECT_NETWORK,
    "http": EFFECT_NETWORK,
    "ftplib": EFFECT_NETWORK,
    "smtplib": EFFECT_NETWORK,
    "websockets": EFFECT_NETWORK,
    "openai": EFFECT_NETWORK,
    "sqlite3": EFFECT_DATABASE,
    "psycopg2": EFFECT_DATABASE,
    "pymysql": EFFECT_DATABASE,
    "sqlalchemy": EFFECT_DATABASE,
    "redis": EFFECT_DATABASE,
    "pymongo": EFFECT_DATABASE,
    "logging": EFFECT_CONSOLE_LOGGING,
    "warnings": EFFECT_CONSOLE_LOGGING,
    "random": EFFECT_TIME_RANDOM,
    "secrets": EFFECT_TIME_RANDOM,
    "threading": EFFECT_CONCURRENCY,
    "concurrent": EFFECT_CONCURRENCY,
}

# os.<function> classifications; the os module mixes several effect kinds.
OS_FUNCTION_EFFECTS = {
    "remove": EFFECT_FILESYSTEM,
    "unlink": EFFECT_FILESYSTEM,
    "rename": EFFECT_FILESYSTEM,
    "renames": EFFECT_FILESYSTEM,
    "replace": EFFECT_FILESYSTEM,
    "makedirs": EFFECT_FILESYSTEM,
    "mkdir": EFFECT_FILESYSTEM,
    "rmdir": EFFECT_FILESYSTEM,
    "removedirs": EFFECT_FILESYSTEM,
    "chmod": EFFECT_FILESYSTEM,
    "chown": EFFECT_FILESYSTEM,
    "symlink": EFFECT_FILESYSTEM,
    "link": EFFECT_FILESYSTEM,
    "truncate": EFFECT_FILESYSTEM,
    "system": EFFECT_PROCESS,
    "popen": EFFECT_PROCESS,
    "kill": EFFECT_PROCESS,
    "fork": EFFECT_PROCESS,
    "execv": EFFECT_PROCESS,
    "execve": EFFECT_PROCESS,
    "execvp": EFFECT_PROCESS,
    "spawnv": EFFECT_PROCESS,
    "putenv": EFFECT_ENVIRONMENT,
    "unsetenv": EFFECT_ENVIRONMENT,
    "getenv": EFFECT_ENVIRONMENT,
}

TIME_FUNCTION_EFFECTS = {"sleep", "time", "monotonic", "perf_counter"}
ASYNCIO_CONCURRENCY_FUNCTIONS = {
    "create_task",
    "ensure_future",
    "gather",
    "run",
    "to_thread",
}
DATETIME_NOW_FUNCTIONS = {"now", "utcnow", "today"}

# Method names that imply an effect even when the receiver type is unknown.
# These are reported as inferred, never confirmed.
FILESYSTEM_METHOD_NAMES = {
    "write_text",
    "write_bytes",
    "read_text",
    "read_bytes",
    "unlink",
    "mkdir",
    "rmdir",
    "touch",
    "symlink_to",
    "hardlink_to",
}
DATABASE_METHOD_NAMES = {
    "execute",
    "executemany",
    "executescript",
    "commit",
    "rollback",
}
CONTAINER_MUTATORS = {
    "append",
    "add",
    "update",
    "extend",
    "insert",
    "remove",
    "pop",
    "popitem",
    "clear",
    "setdefault",
}


def is_python_test_path(path: str) -> bool:
    lowered = path.lower()
    name = PurePosixPath(lowered).name
    return (
        lowered.startswith("tests/")
        or "/tests/" in lowered
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def analyze_python_file(path: str, text: str) -> ModuleIndex:
    line_count = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError) as exc:
        message = exc.msg if isinstance(exc, SyntaxError) else str(exc)
        return ModuleIndex(
            path=path,
            language=LANGUAGE_PYTHON,
            line_count=line_count,
            is_test=is_python_test_path(path),
            parse_ok=False,
            parse_error=f"line {getattr(exc, 'lineno', '?')}: {message}",
        )

    collector = _Collector(path)
    collector.walk_module(tree)
    return ModuleIndex(
        path=path,
        language=LANGUAGE_PYTHON,
        line_count=line_count,
        definitions=tuple(collector.definitions),
        imports=tuple(collector.imports),
        exports=_export_bindings(collector),
        calls=tuple(collector.calls),
        side_effects=tuple(collector.side_effects),
        error_handling=tuple(collector.error_handling),
        class_bases=dict(collector.class_bases),
        attribute_types=dict(collector.attribute_types),
        is_test=is_python_test_path(path),
    )


def _export_bindings(collector: _Collector) -> tuple[ExportBinding, ...]:
    if collector.dunder_all_names is not None:
        return tuple(
            ExportBinding(
                exported_name=name,
                local_name=name,
                source_module=None,
                source_name=None,
                kind="all_list",
                line=line,
            )
            for name, line in collector.dunder_all_names
        )
    implicit: list[ExportBinding] = []
    for definition in collector.definitions:
        if definition.parent is None and not definition.name.startswith("_"):
            implicit.append(
                ExportBinding(
                    exported_name=definition.name,
                    local_name=definition.name,
                    source_module=None,
                    source_name=None,
                    kind="implicit",
                    line=definition.line,
                )
            )
    return tuple(implicit)


class _Scope:
    def __init__(
        self,
        qualname: str | None,
        kind: str,
        class_name: str | None,
        local_types: dict[str, str],
    ):
        self.qualname = qualname
        self.kind = kind  # module | class | function
        self.class_name = class_name
        self.local_types = local_types


class _Collector:
    def __init__(self, path: str):
        self.path = path
        self.definitions: list[SymbolDefinition] = []
        self.imports: list[ImportBinding] = []
        self.calls: list[RawCall] = []
        self.side_effects: list[SideEffect] = []
        self.error_handling: list[ErrorHandling] = []
        self.class_bases: dict[str, tuple[str, ...]] = {}
        self.attribute_types: dict[str, str] = {}
        self.dunder_all_names: list[tuple[str, int]] | None = None
        self.module_level_names: set[str] = set()
        self._import_roots: dict[str, str] = {}
        self._from_import_sources: dict[str, tuple[str, str]] = {}

    # ---- walking ---------------------------------------------------------

    def walk_module(self, tree: ast.Module) -> None:
        scope = _Scope(None, "module", None, {})
        for node in tree.body:
            self._collect_module_level_name(node)
        for node in tree.body:
            self._visit(node, scope, None)

    def _collect_module_level_name(self, node: ast.stmt) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.module_level_names.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.module_level_names.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            self.module_level_names.add(node.target.id)

    def _visit(
        self, node: ast.AST, scope: _Scope, error_info: _ErrorInfo | None
    ) -> None:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._handle_import(node)
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._handle_function(node, scope)
            return
        if isinstance(node, ast.ClassDef):
            self._handle_class(node, scope)
            return
        if isinstance(node, ast.Assign):
            self._handle_assign(node, scope)
            return
        if isinstance(node, ast.AnnAssign):
            self._handle_ann_assign(node, scope)
            return
        if isinstance(node, ast.Global):
            self._note_global(node, scope)
            return
        if isinstance(node, ast.Expr):
            root = node.value if isinstance(node.value, ast.Call) else None
            self._scan_expressions(node.value, scope, statement_root=root)
            return
        if isinstance(node, ast.Raise):
            if error_info is not None:
                error_info.add_raise(node)
            self._scan_expressions(node, scope)
            return
        if isinstance(node, ast.Try):
            if error_info is not None:
                error_info.add_try(node)
            for handler in node.handlers:
                if handler.type is not None:
                    self._scan_expressions(handler.type, scope)
                for child in handler.body:
                    self._visit(child, scope, error_info)
            for child in node.body + node.orelse + node.finalbody:
                self._visit(child, scope, error_info)
            return
        if isinstance(node, (ast.AugAssign, ast.Delete)):
            self._note_environ_write(node, scope)
        if isinstance(node, (ast.With, ast.AsyncWith)):
            self._collect_with_aliases(node, scope)
        self._scan_expressions(node, scope)
        for child in _child_statements(node):
            self._visit(child, scope, error_info)

    def _collect_with_aliases(
        self, node: ast.With | ast.AsyncWith, scope: _Scope
    ) -> None:
        """`with make_thing() as x:` types x like an assignment would."""
        for item in node.items:
            if not isinstance(item.optional_vars, ast.Name):
                continue
            value = item.context_expr
            hint = _constructor_name(value)
            if hint is None and isinstance(value, ast.Call):
                head, chain, dynamic = _callee_parts(value.func)
                if head and not dynamic:
                    hint = "~call:" + (".".join([head, *chain]) if chain else head)
            if hint:
                scope.local_types[item.optional_vars.id] = hint

    def _scan_expressions(
        self,
        node: ast.AST,
        scope: _Scope,
        *,
        statement_root: ast.Call | None = None,
    ) -> None:
        for expr in _iter_expressions(node):
            if isinstance(expr, ast.Call):
                self._record_call(expr, scope, is_statement=expr is statement_root)

    # ---- imports ---------------------------------------------------------

    def _handle_import(self, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                self.imports.append(
                    ImportBinding(
                        local_name=local,
                        source_module=alias.name,
                        source_name=None,
                        kind="module",
                        line=node.lineno,
                    )
                )
                self._import_roots[local] = alias.name.split(".")[0]
            return
        module = "." * node.level + (node.module or "")
        for alias in node.names:
            if alias.name == "*":
                self.imports.append(
                    ImportBinding(
                        local_name="*",
                        source_module=module,
                        source_name="*",
                        kind="star",
                        line=node.lineno,
                    )
                )
                continue
            local = alias.asname or alias.name
            self.imports.append(
                ImportBinding(
                    local_name=local,
                    source_module=module,
                    source_name=alias.name,
                    kind="named",
                    line=node.lineno,
                )
            )
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root:
                self._import_roots[local] = root
                self._from_import_sources[local] = (root, alias.name)

    # ---- definitions -----------------------------------------------------

    def _handle_function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, scope: _Scope
    ) -> None:
        qualname = f"{scope.qualname}.{node.name}" if scope.qualname else node.name
        is_method = scope.kind == "class"
        is_async = isinstance(node, ast.AsyncFunctionDef)
        if is_method:
            kind = KIND_ASYNC_METHOD if is_async else KIND_METHOD
        else:
            kind = KIND_ASYNC_FUNCTION if is_async else KIND_FUNCTION
        decorators = tuple(
            _bounded(_safe_unparse(decorator), 80) for decorator in node.decorator_list
        )
        self.definitions.append(
            SymbolDefinition(
                path=self.path,
                qualname=qualname,
                name=node.name,
                kind=kind,
                language=LANGUAGE_PYTHON,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=_function_signature(node),
                parent=scope.qualname,
                decorators=decorators,
                doc=_first_line(ast.get_docstring(node) or ""),
            )
        )

        # Default argument values evaluate at definition time, in the
        # enclosing scope.
        for default in list(node.args.defaults) + [
            item for item in node.args.kw_defaults if item is not None
        ]:
            self._scan_expressions(default, scope)

        local_types: dict[str, str] = {}
        annotated = dict(_annotated_args(node))
        local_types.update(annotated)
        class_name = scope.class_name if is_method else None
        if is_method and node.args.args and class_name:
            first = node.args.args[0].arg
            if first in {"self", "cls"}:
                local_types[first] = class_name.rsplit(".", 1)[-1]
        function_scope = _Scope(qualname, "function", scope.class_name, local_types)
        if is_method and class_name:
            self._collect_attribute_types(node, class_name, annotated)
        error_info = _ErrorInfo()
        for child in node.body:
            self._visit(child, function_scope, error_info)
        self.error_handling.append(error_info.build(qualname))

    def _handle_class(self, node: ast.ClassDef, scope: _Scope) -> None:
        qualname = f"{scope.qualname}.{node.name}" if scope.qualname else node.name
        bases = tuple(_bounded(_safe_unparse(base), 80) for base in node.bases)
        self.class_bases[qualname] = bases
        signature = (
            f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"
        )
        self.definitions.append(
            SymbolDefinition(
                path=self.path,
                qualname=qualname,
                name=node.name,
                kind=KIND_CLASS,
                language=LANGUAGE_PYTHON,
                line=node.lineno,
                end_line=node.end_lineno or node.lineno,
                signature=_bounded(signature, MAX_SIGNATURE_CHARS),
                parent=scope.qualname,
                decorators=tuple(
                    _bounded(_safe_unparse(item), 80) for item in node.decorator_list
                ),
                doc=_first_line(ast.get_docstring(node) or ""),
            )
        )
        class_scope = _Scope(qualname, "class", qualname, {})
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                annotation = _bounded(_safe_unparse(child.annotation), 80)
                self.attribute_types.setdefault(
                    f"{qualname}.{child.target.id}", annotation
                )
            self._visit(child, class_scope, None)

    def _handle_assign(self, node: ast.Assign, scope: _Scope) -> None:
        if scope.kind == "module":
            self._maybe_record_dunder_all(node)
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id.isupper():
                    self.definitions.append(
                        SymbolDefinition(
                            path=self.path,
                            qualname=target.id,
                            name=target.id,
                            kind=KIND_CONSTANT,
                            language=LANGUAGE_PYTHON,
                            line=node.lineno,
                            end_line=node.end_lineno or node.lineno,
                            signature="",
                        )
                    )
        value = node.value
        if isinstance(value, ast.Await):
            value = value.value
        constructed = _constructor_name(value)
        if constructed is None and isinstance(value, ast.Call):
            # Lowercase factory call: keep the callee so the engine can
            # follow its return annotation (`agent = build_agent(...)`).
            head, chain, dynamic = _callee_parts(value.func)
            if head and not dynamic:
                callee = ".".join([head, *chain]) if chain else head
                constructed = f"~call:{callee}"
        if constructed:
            for target in node.targets:
                if isinstance(target, ast.Name):
                    scope.local_types[target.id] = constructed
        self._note_environ_write(node, scope)
        root = node.value if isinstance(node.value, ast.Call) else None
        self._scan_expressions(node.value, scope, statement_root=root)

    def _handle_ann_assign(self, node: ast.AnnAssign, scope: _Scope) -> None:
        annotation = _bounded(_safe_unparse(node.annotation), 80)
        if isinstance(node.target, ast.Name):
            scope.local_types[node.target.id] = annotation
        if node.value is not None:
            self._scan_expressions(node.value, scope)

    def _collect_attribute_types(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        class_name: str,
        annotated_params: dict[str, str],
    ) -> None:
        for statement in ast.walk(node):
            if isinstance(statement, ast.Assign):
                value = statement.value
                if isinstance(value, ast.Await):
                    value = value.value
                value_type = _constructor_name(value)
                if value_type is None and isinstance(value, ast.Name):
                    value_type = annotated_params.get(value.id)
                if value_type is None and isinstance(value, ast.Call):
                    head, chain, dynamic = _callee_parts(value.func)
                    if head and not dynamic:
                        callee = ".".join([head, *chain]) if chain else head
                        value_type = f"~call:{callee}"
                if value_type is None:
                    continue
                for target in statement.targets:
                    attr = _self_attribute_name(target)
                    if attr:
                        self.attribute_types.setdefault(
                            f"{class_name}.{attr}", value_type
                        )
            elif isinstance(statement, ast.AnnAssign):
                attr = _self_attribute_name(statement.target)
                if attr:
                    self.attribute_types.setdefault(
                        f"{class_name}.{attr}",
                        _bounded(_safe_unparse(statement.annotation), 80),
                    )

    def _maybe_record_dunder_all(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, (ast.List, ast.Tuple)):
                    names: list[tuple[str, int]] = []
                    for element in node.value.elts:
                        if isinstance(element, ast.Constant) and isinstance(
                            element.value, str
                        ):
                            names.append((element.value, node.lineno))
                    self.dunder_all_names = names

    # ---- side-effect notes -------------------------------------------------

    def _note_global(self, node: ast.Global, scope: _Scope) -> None:
        if scope.qualname is None:
            return
        for name in node.names:
            self.side_effects.append(
                SideEffect(
                    category=EFFECT_GLOBAL_STATE,
                    detail=f"global {name}",
                    path=self.path,
                    line=node.lineno,
                    qualname=scope.qualname,
                    resolution=RESOLUTION_CONFIRMED,
                )
            )

    def _note_environ_write(self, node: ast.stmt, scope: _Scope) -> None:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        elif isinstance(node, ast.Delete):
            targets = list(node.targets)
        for target in targets:
            if isinstance(target, ast.Subscript) and _is_os_environ(target.value):
                self.side_effects.append(
                    SideEffect(
                        category=EFFECT_ENVIRONMENT,
                        detail="os.environ write",
                        path=self.path,
                        line=node.lineno,
                        qualname=scope.qualname,
                        resolution=RESOLUTION_CONFIRMED,
                    )
                )

    # ---- calls and effects -------------------------------------------------

    def _record_call(
        self, node: ast.Call, scope: _Scope, *, is_statement: bool
    ) -> None:
        head, chain, dynamic = _callee_parts(node.func)
        callee_text = _bounded(_safe_unparse(node.func), MAX_CALLEE_TEXT_CHARS)
        argument_names = tuple(
            arg.id for arg in node.args if isinstance(arg, ast.Name)
        )[:MAX_ARGUMENT_NAMES]
        receiver_type = scope.local_types.get(head) if head else None
        if dynamic and chain:
            # `Service(...).record(...)`: the chain is anchored on a call whose
            # result type may be statically knowable.
            anchor_hint = _call_anchor_hint(node.func)
            if anchor_hint is not None:
                receiver_type = anchor_hint
                head = ""
                dynamic = False
        if head == "getattr":
            dynamic = True
        call = RawCall(
            caller_qualname=scope.qualname,
            callee_text=callee_text,
            head=head or "",
            attr_chain=chain,
            line=node.lineno,
            is_await=bool(getattr(node, "_lewis_awaited", False)),
            is_statement=is_statement,
            is_dynamic=dynamic,
            argument_names=argument_names,
            receiver_type=receiver_type,
        )
        self.calls.append(call)
        self._classify_effect(call, scope)

    def _classify_effect(self, call: RawCall, scope: _Scope) -> None:
        effect = self._effect_for_call(call)
        if effect is None:
            return
        category, detail, resolution = effect
        self.side_effects.append(
            SideEffect(
                category=category,
                detail=detail,
                path=self.path,
                line=call.line,
                qualname=scope.qualname,
                resolution=resolution,
            )
        )

    def _effect_for_call(self, call: RawCall) -> tuple[str, str, str] | None:
        head = call.head
        chain = call.attr_chain
        text = call.callee_text
        if not head:
            return None
        if head == "open" and not chain:
            return EFFECT_FILESYSTEM, "open()", RESOLUTION_CONFIRMED
        if head == "print" and not chain:
            return EFFECT_CONSOLE_LOGGING, "print()", RESOLUTION_CONFIRMED
        if head == "input" and not chain:
            return EFFECT_CONSOLE_LOGGING, "input()", RESOLUTION_CONFIRMED

        if head in self._from_import_sources:
            source_root, source_name = self._from_import_sources[head]
            effect = _effect_for_dotted(source_root, (source_name, *chain), text)
            if effect is not None:
                return effect
        root = self._import_roots.get(head)
        if root is not None:
            effect = _effect_for_dotted(root, chain, text)
            if effect is not None:
                return effect

        last = chain[-1] if chain else ""
        if last in FILESYSTEM_METHOD_NAMES:
            return EFFECT_FILESYSTEM, f"{text} (method-name match)", RESOLUTION_INFERRED
        if last in DATABASE_METHOD_NAMES:
            return EFFECT_DATABASE, f"{text} (method-name match)", RESOLUTION_INFERRED
        if (
            len(chain) == 1
            and head in self.module_level_names
            and chain[0] in CONTAINER_MUTATORS
        ):
            return (
                EFFECT_GLOBAL_STATE,
                f"mutates module-level {head}.{chain[0]}()",
                RESOLUTION_INFERRED,
            )
        return None


class _ErrorInfo:
    def __init__(self) -> None:
        self.raises: list[str] = []
        self.raise_lines: list[int] = []
        self.catches: list[str] = []
        self.catch_lines: list[int] = []
        self.swallowed: list[int] = []
        self.bare: list[int] = []

    def add_raise(self, node: ast.Raise) -> None:
        if node.exc is None:
            label = "re-raise"
        elif isinstance(node.exc, ast.Call):
            label = _bounded(_safe_unparse(node.exc.func), 60)
        else:
            label = _bounded(_safe_unparse(node.exc), 60)
        self.raises.append(label)
        self.raise_lines.append(node.lineno)

    def add_try(self, node: ast.Try) -> None:
        for handler in node.handlers:
            if handler.type is None:
                label = "except:"
                self.bare.append(handler.lineno)
            else:
                label = f"except {_bounded(_safe_unparse(handler.type), 60)}"
            self.catches.append(label)
            self.catch_lines.append(handler.lineno)
            if _handler_swallows(handler):
                self.swallowed.append(handler.lineno)

    def build(self, qualname: str) -> ErrorHandling:
        return ErrorHandling(
            qualname=qualname,
            raises=tuple(self.raises),
            raise_lines=tuple(self.raise_lines),
            catches=tuple(self.catches),
            catch_lines=tuple(self.catch_lines),
            swallowed_catch_lines=tuple(self.swallowed),
            bare_catch_lines=tuple(self.bare),
        )


# ---- helpers --------------------------------------------------------------


def _effect_for_dotted(
    root: str, chain: tuple[str, ...], text: str
) -> tuple[str, str, str] | None:
    if root == "os":
        if chain and chain[0] in OS_FUNCTION_EFFECTS:
            category = OS_FUNCTION_EFFECTS[chain[0]]
            detail = f"{text} (read)" if chain[0] == "getenv" else text
            return category, detail, RESOLUTION_CONFIRMED
        if chain[:2] in {("environ", "setdefault"), ("environ", "update")}:
            return EFFECT_ENVIRONMENT, text, RESOLUTION_CONFIRMED
        if chain[:2] == ("environ", "get"):
            return EFFECT_ENVIRONMENT, f"{text} (read)", RESOLUTION_CONFIRMED
        return None
    if root == "sys":
        if chain[:1] == ("exit",):
            return EFFECT_PROCESS, text, RESOLUTION_CONFIRMED
        if chain[:2] in {("stdout", "write"), ("stderr", "write")}:
            return EFFECT_CONSOLE_LOGGING, text, RESOLUTION_CONFIRMED
        return None
    if root == "time":
        if chain and chain[0] in TIME_FUNCTION_EFFECTS:
            return EFFECT_TIME_RANDOM, text, RESOLUTION_CONFIRMED
        return None
    if root == "datetime":
        if chain and chain[-1] in DATETIME_NOW_FUNCTIONS:
            return EFFECT_TIME_RANDOM, text, RESOLUTION_CONFIRMED
        return None
    if root == "asyncio":
        if chain and chain[0] in ASYNCIO_CONCURRENCY_FUNCTIONS:
            return EFFECT_CONCURRENCY, text, RESOLUTION_CONFIRMED
        return None
    if root == "uuid":
        if chain and chain[0].startswith("uuid"):
            return EFFECT_TIME_RANDOM, text, RESOLUTION_CONFIRMED
        return None
    category = EFFECT_MODULE_ROOTS.get(root)
    if category is not None:
        return category, text, RESOLUTION_CONFIRMED
    return None


def _handler_swallows(handler: ast.ExceptHandler) -> bool:
    meaningful = [
        node
        for node in handler.body
        if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant))
    ]
    if not meaningful:
        return True
    return all(
        isinstance(node, (ast.Pass, ast.Continue, ast.Break)) for node in meaningful
    )


def _callee_parts(func: ast.expr) -> tuple[str | None, tuple[str, ...], bool]:
    chain: list[str] = []
    node = func
    while isinstance(node, ast.Attribute):
        chain.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return node.id, tuple(reversed(chain)), False
    if isinstance(node, ast.Call):
        inner_head, _, _ = _callee_parts(node.func)
        return inner_head, tuple(reversed(chain)), True
    return None, tuple(reversed(chain)), True


def _iter_expressions(node: ast.AST):
    stack = [node]
    while stack:
        current = stack.pop()
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(current, ast.Await) and isinstance(current.value, ast.Call):
            current.value._lewis_awaited = True  # type: ignore[attr-defined]
        if isinstance(current, ast.Call):
            yield current
        for child in ast.iter_child_nodes(current):
            if isinstance(child, ast.stmt):
                continue
            stack.append(child)


def _child_statements(node: ast.AST) -> list[ast.stmt]:
    return [
        child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt)
    ]


def _annotated_args(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[tuple[str, str]]:
    annotated: list[tuple[str, str]] = []
    all_args = (
        list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs)
    )
    for arg in all_args:
        if arg.annotation is not None:
            annotated.append((arg.arg, _bounded(_safe_unparse(arg.annotation), 80)))
    return annotated


def _constructor_name(value: ast.expr) -> str | None:
    if not isinstance(value, ast.Call):
        return None
    head, chain, dynamic = _callee_parts(value.func)
    if dynamic or head is None:
        return None
    name = ".".join([head, *chain]) if chain else head
    last = chain[-1] if chain else head
    if last and last[0].isupper():
        return name
    return None


def _call_anchor_hint(func: ast.expr) -> str | None:
    """Type hint for a call chain anchored on another call's result."""
    node = func
    while isinstance(node, ast.Attribute):
        node = node.value
    if not isinstance(node, ast.Call):
        return None
    constructed = _constructor_name(node)
    if constructed is not None:
        return constructed
    head, chain, dynamic = _callee_parts(node.func)
    if head and not dynamic:
        return "~call:" + (".".join([head, *chain]) if chain else head)
    return None


def _self_attribute_name(target: ast.expr) -> str | None:
    if (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
    ):
        return target.attr
    return None


def _is_os_environ(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    try:
        args = ast.unparse(node.args)
    except Exception:
        args = "..."
    returns = ""
    if node.returns is not None:
        returns = f" -> {_safe_unparse(node.returns)}"
    return _bounded(f"({args}){returns}", MAX_SIGNATURE_CHARS)


def _first_line(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return _bounded(stripped.splitlines()[0], MAX_DOC_CHARS)


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
