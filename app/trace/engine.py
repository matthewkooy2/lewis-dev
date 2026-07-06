"""Trace traversal: target resolution, call resolution, quick and deep modes.

All language-specific knowledge lives in the analyzers and the project
index; this module only walks the normalized shapes. Every produced edge
carries an explicit resolution label — static analysis cannot prove dynamic
dispatch, so uncertain relationships are labeled, never guessed.
"""

from __future__ import annotations

import difflib
from dataclasses import dataclass

from app.secret_redaction import redact_secret_values
from app.trace.model import (
    CallEdge,
    KIND_CLASS,
    KIND_COMPONENT,
    KIND_HOOK,
    LANGUAGE_PYTHON,
    ModuleIndex,
    RawCall,
    RESOLUTION_AMBIGUOUS,
    RESOLUTION_CONFIRMED,
    RESOLUTION_DYNAMIC,
    RESOLUTION_EXTERNAL,
    RESOLUTION_FRAMEWORK,
    RESOLUTION_INFERRED,
    RESOLUTION_UNRESOLVED,
    SideEffect,
    Snippet,
    SymbolDefinition,
    TestLink,
    TraceLimits,
    TraceResult,
)
from app.trace.project_index import ProjectIndex


PYTHON_BUILTINS = frozenset(
    {
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "classmethod",
        "dict", "dir", "divmod", "enumerate", "filter", "float", "format",
        "frozenset", "getattr", "hasattr", "hash", "hex", "id", "int",
        "isinstance", "issubclass", "iter", "len", "list", "map", "max", "min",
        "next", "object", "oct", "ord", "pow", "property", "range", "repr",
        "reversed", "round", "set", "setattr", "slice", "sorted",
        "staticmethod", "str", "sum", "super", "tuple", "type", "vars", "zip",
        "print", "input", "open", "Exception", "ValueError", "TypeError",
        "KeyError", "IndexError", "RuntimeError", "OSError", "IOError",
        "AttributeError", "NotImplementedError", "StopIteration",
        "FileNotFoundError", "PermissionError", "KeyboardInterrupt",
        "ZeroDivisionError", "ArithmeticError", "LookupError", "NameError",
        "UnicodeDecodeError", "UnicodeEncodeError", "ConnectionError",
        "TimeoutError", "InterruptedError", "AssertionError", "BaseException",
    }
)

JS_GLOBALS = frozenset(
    {
        "Array", "ArrayBuffer", "BigInt", "Boolean", "DataView", "Date",
        "Error", "EvalError", "Function", "Infinity", "Intl", "JSON", "Map",
        "Math", "NaN", "Number", "Object", "Promise", "Proxy", "RangeError",
        "ReferenceError", "Reflect", "RegExp", "Set", "String", "Symbol",
        "SyntaxError", "TypeError", "URIError", "URL", "URLSearchParams",
        "WeakMap", "WeakSet", "decodeURI", "decodeURIComponent", "encodeURI",
        "encodeURIComponent", "eval", "isFinite", "isNaN", "parseFloat",
        "parseInt", "structuredClone", "fetch", "console", "setTimeout",
        "setInterval", "clearTimeout", "clearInterval", "requestAnimationFrame",
        "queueMicrotask", "alert", "confirm", "prompt", "document", "window",
        "navigator", "location", "history", "localStorage", "sessionStorage",
        "crypto", "require", "String", "Buffer", "process", "globalThis",
    }
)

FRAMEWORK_DECORATOR_MARKERS = (
    ".get(", ".post(", ".put(", ".patch(", ".delete(", ".head(", ".options(",
    ".route(", ".websocket(", ".middleware(", ".exception_handler(",
    ".on_event(", "fixture", ".command(", ".task(",
)

CALLABLE_KINDS = frozenset(
    {
        "function", "async_function", "method", "async_method",
        KIND_COMPONENT, KIND_HOOK, KIND_CLASS,
    }
)

# Method names too generic to attribute by name alone when the receiver is
# unknown (they collide with stdlib/DB-API/collection methods constantly).
COMMON_METHOD_NAMES = frozenset(
    {
        "add", "append", "clear", "close", "commit", "connect", "copy",
        "count", "create", "delete", "execute", "extend", "find", "format",
        "get", "insert", "items", "join", "keys", "list", "load", "open",
        "pop", "push", "put", "read", "remove", "rollback", "run", "save",
        "search", "send", "set", "sort", "split", "start", "stop", "strip",
        "update", "values", "write",
    }
)


@dataclass(frozen=True)
class ResolvedCallee:
    kind: str  # project | external | dynamic | unresolved | ambiguous
    definition: SymbolDefinition | None
    resolution: str
    note: str = ""
    candidates: tuple[SymbolDefinition, ...] = ()


@dataclass(frozen=True)
class TargetLookup:
    target: SymbolDefinition | None
    candidates: tuple[SymbolDefinition, ...]
    error: str = ""
    suggestions: tuple[str, ...] = ()


class TraceEngine:
    def __init__(self, index: ProjectIndex, limits: TraceLimits | None = None):
        self.index = index
        self.limits = limits or TraceLimits()
        self._resolution_cache: dict[tuple[str, int, str], ResolvedCallee] = {}

    # ---- target resolution ---------------------------------------------------

    def resolve_target(self, symbol: str, file_hint: str | None = None) -> TargetLookup:
        spec = (symbol or "").strip()
        if not spec:
            return TargetLookup(None, (), error="A symbol name is required.")
        file_path = None
        if "::" in spec:
            file_part, _, spec = spec.partition("::")
            file_path = file_part.strip()
            spec = spec.strip()
        if file_hint:
            file_path = file_hint.strip()
        if file_path is not None:
            validated = self.index.workspace.safe_relative_path(file_path)
            if validated is None:
                return TargetLookup(
                    None,
                    (),
                    error=(
                        f"File {file_path!r} is not an analyzable path inside "
                        "the workspace."
                    ),
                )
            if validated not in self.index.modules:
                return TargetLookup(
                    None, (), error=f"File {validated!r} was not analyzed."
                )
            file_path = validated

        candidates = self._collect_candidates(spec, file_path)
        if not candidates:
            suggestions = tuple(
                difflib.get_close_matches(
                    spec.rsplit(".", 1)[-1],
                    list(self.index.definitions_by_name),
                    n=5,
                    cutoff=0.6,
                )
            )
            return TargetLookup(
                None,
                (),
                error=f"No definition found for {symbol!r}.",
                suggestions=suggestions,
            )
        if len(candidates) == 1:
            return TargetLookup(candidates[0], tuple(candidates))
        return TargetLookup(None, tuple(candidates[: self.limits.max_candidates]))

    def _collect_candidates(
        self, spec: str, file_path: str | None
    ) -> list[SymbolDefinition]:
        scoped = (
            [self.index.modules[file_path]] if file_path else self.index.modules.values()
        )
        exact: list[SymbolDefinition] = []
        by_name: list[SymbolDefinition] = []
        by_suffix: list[SymbolDefinition] = []
        for module in scoped:
            for definition in module.definitions:
                if definition.qualname == spec:
                    exact.append(definition)
                elif definition.name == spec:
                    by_name.append(definition)
                elif definition.qualname.endswith("." + spec):
                    by_suffix.append(definition)
        if "." not in spec:
            # For a bare name, a module-level def whose qualname equals the
            # name is no more exact than a method with that name — surface
            # all of them as candidates rather than silently picking one.
            merged = _dedupe(exact + by_name)
            if merged:
                return merged
            return _dedupe(by_suffix)
        if exact:
            return _dedupe(exact)
        if file_path is None:
            dotted = self._resolve_python_dotted(spec)
            if dotted:
                return _dedupe(dotted)
        if by_name:
            return _dedupe(by_name)
        return _dedupe(by_suffix)

    def _resolve_python_dotted(self, spec: str) -> list[SymbolDefinition]:
        parts = spec.split(".")
        for split_at in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:split_at])
            path = self.index.python_modules.get(module_name)
            if path is None:
                continue
            qualname = ".".join(parts[split_at:])
            definition = self.index.definition_at(path, qualname)
            if definition is not None:
                return [definition]
        return []

    # ---- call resolution -----------------------------------------------------

    def resolve_call(self, module: ModuleIndex, call: RawCall) -> ResolvedCallee:
        key = (module.path, call.line, call.callee_text)
        cached = self._resolution_cache.get(key)
        if cached is not None:
            return cached
        resolved = self._resolve_call_uncached(module, call)
        self._resolution_cache[key] = resolved
        return resolved

    def _resolve_call_uncached(
        self, module: ModuleIndex, call: RawCall
    ) -> ResolvedCallee:
        if call.is_dynamic:
            return ResolvedCallee(
                "dynamic", None, RESOLUTION_DYNAMIC,
                note="call target is computed at runtime",
            )
        if call.jsx_component:
            return self._resolve_jsx_component(module, call)
        if call.is_callback_reference:
            resolved = self._resolve_name(module, call.head, call.caller_qualname)
            if resolved is not None:
                return ResolvedCallee(
                    "project", resolved, RESOLUTION_FRAMEWORK,
                    note="registered as an event handler/callback",
                )
            return ResolvedCallee(
                "unresolved", None, RESOLUTION_UNRESOLVED,
                note="handler reference not found in workspace",
            )
        receiver = call.receiver_type
        if (call.head in {"self", "cls", "this"} or receiver) and call.attr_chain:
            resolved = self._resolve_receiver_call(module, call, receiver)
            if resolved is not None:
                return resolved
        if not call.head:
            return self._resolve_by_method_name(call, receiver_note="expression")

        if not call.attr_chain:
            direct = self._resolve_name(module, call.head, call.caller_qualname)
            if direct is not None:
                note = "constructor call" if direct.kind == KIND_CLASS else ""
                return ResolvedCallee("project", direct, RESOLUTION_CONFIRMED, note)
        else:
            scoped = self._resolve_name(module, call.head, call.caller_qualname)
            if scoped is not None and scoped.kind == KIND_CLASS:
                method = self.index.resolve_method(scoped, call.attr_chain[0])
                if method is not None and len(call.attr_chain) == 1:
                    return ResolvedCallee(
                        "project", method, RESOLUTION_CONFIRMED,
                        note=f"class-level call on {scoped.name}",
                    )

        from_import = self._resolve_via_imports(module, call)
        if from_import is not None:
            return from_import

        if self._is_builtin(module, call.head):
            return ResolvedCallee(
                "external", None, RESOLUTION_EXTERNAL,
                note=(
                    "python builtin"
                    if module.language == LANGUAGE_PYTHON
                    else "JS global/builtin"
                ),
            )
        return self._fallback_name_match(module, call)

    def _resolve_jsx_component(
        self, module: ModuleIndex, call: RawCall
    ) -> ResolvedCallee:
        resolved = self._resolve_name(module, call.head, call.caller_qualname)
        if resolved is not None and not call.attr_chain:
            return ResolvedCallee(
                "project", resolved, RESOLUTION_FRAMEWORK,
                note="rendered as JSX; React invokes the component",
            )
        via_import = self._resolve_via_imports(module, call)
        if via_import is not None and via_import.kind == "project":
            return ResolvedCallee(
                "project", via_import.definition, RESOLUTION_FRAMEWORK,
                note="rendered as JSX; React invokes the component",
            )
        if via_import is not None and via_import.kind == "external":
            return ResolvedCallee(
                "external", None, RESOLUTION_EXTERNAL,
                note="component imported from an external package",
            )
        return ResolvedCallee(
            "unresolved", None, RESOLUTION_UNRESOLVED,
            note="JSX component not found in workspace",
        )

    def _class_from_type_hint(
        self,
        module: ModuleIndex,
        hint: str,
        caller_qualname: str | None,
    ) -> tuple[SymbolDefinition | None, str]:
        """Class definition for a type hint, following `~call:` factory hints."""
        if hint.startswith("~call:"):
            callee = hint[len("~call:"):]
            factory = self._resolve_dotted_name(module, callee, caller_qualname)
            if factory is None:
                return None, ""
            if factory.kind == KIND_CLASS:
                return factory, f"instance created by {factory.name}(...)"
            if "->" not in factory.signature:
                return None, ""
            return_type = factory.signature.rsplit("->", 1)[-1].strip()
            factory_module = self.index.module_for(factory.path) or module
            resolved = self.index.resolve_type_text(factory_module, return_type)
            if resolved is None:
                resolved = self.index.resolve_type_text(module, return_type)
            return resolved, f"via return annotation of {factory.name}()"
        return self.index.resolve_type_text(module, hint), ""

    def _factory_is_external(self, module: ModuleIndex, hint: str) -> str | None:
        """External module name when a `~call:` factory is an external import."""
        if not hint.startswith("~call:"):
            return None
        head = hint[len("~call:"):].split(".")[0]
        for binding in module.imports:
            if binding.kind == "star" or binding.local_name != head:
                continue
            if self.index.resolve_import_binding(module, binding, None) == "external":
                return binding.source_module
            return None
        return None

    def _resolve_dotted_name(
        self, module: ModuleIndex, dotted: str, caller_qualname: str | None
    ) -> SymbolDefinition | None:
        parts = dotted.split(".")
        head_def = self._resolve_name(module, parts[0], caller_qualname)
        if head_def is not None:
            current = head_def
            for part in parts[1:]:
                nested = self.index.definition_at(
                    current.path, f"{current.qualname}.{part}"
                )
                if nested is None and current.kind == KIND_CLASS:
                    nested = self.index.resolve_method(current, part)
                if nested is None:
                    return None
                current = nested
            return current
        for binding in module.imports:
            if binding.kind == "star" or binding.local_name != parts[0]:
                continue
            resolved = self.index.resolve_import_binding(
                module, binding, parts[1] if len(parts) > 1 else None
            )
            if resolved in ("external", (None, None)):
                return None
            path, qualname = resolved  # type: ignore[misc]
            if not qualname:
                return None
            return self.index.definition_at(path, qualname)
        return None

    def _resolve_receiver_call(
        self, module: ModuleIndex, call: RawCall, receiver: str | None
    ) -> ResolvedCallee | None:
        if not call.attr_chain:
            return None
        class_def: SymbolDefinition | None = None
        receiver_note = ""
        if call.head in {"self", "cls", "this"}:
            class_def = self._enclosing_class(module, call.caller_qualname)
        if class_def is None and receiver:
            external_source = self._factory_is_external(module, receiver)
            if external_source is not None:
                return ResolvedCallee(
                    "external", None, RESOLUTION_EXTERNAL,
                    note=(
                        f"method on an object created by external "
                        f"`{external_source}`"
                    ),
                )
            class_def, receiver_note = self._class_from_type_hint(
                module, receiver, call.caller_qualname
            )
        if class_def is None or class_def.kind != KIND_CLASS:
            return None
        hop_inferred = call.head not in {"self", "cls", "this"}
        current = class_def
        for attribute in call.attr_chain[:-1]:
            attr_module = self.index.module_for(current.path)
            if attr_module is None:
                return None
            attr_type = attr_module.attribute_types.get(
                f"{current.qualname}.{attribute}"
            )
            if not attr_type:
                return ResolvedCallee(
                    "unresolved", None, RESOLUTION_UNRESOLVED,
                    note=(
                        f"attribute `{attribute}` on {current.name} has no "
                        "statically known type"
                    ),
                )
            next_def, _ = self._class_from_type_hint(
                attr_module, attr_type, call.caller_qualname
            )
            if next_def is None or next_def.kind != KIND_CLASS:
                return ResolvedCallee(
                    "unresolved", None, RESOLUTION_UNRESOLVED,
                    note=f"receiver type `{attr_type}` not found in workspace",
                )
            current = next_def
            hop_inferred = True
        method_name = call.attr_chain[-1]
        method = self.index.resolve_method(current, method_name)
        if method is None:
            return ResolvedCallee(
                "unresolved", None, RESOLUTION_UNRESOLVED,
                note=f"no method `{method_name}` found on {current.name} or its bases",
            )
        overrides = self.index.subclass_overrides(current, method_name)
        if overrides:
            candidates = tuple([method, *overrides][: self.limits.max_candidates])
            return ResolvedCallee(
                "ambiguous", method, RESOLUTION_AMBIGUOUS,
                note=(
                    f"dynamic dispatch: {len(overrides)} subclass override(s) exist"
                ),
                candidates=candidates,
            )
        inherited = method.path != current.path or not method.qualname.startswith(
            current.qualname + "."
        )
        notes = [receiver_note]
        if inherited:
            notes.append(f"inherited via {current.name} base classes")
        resolution = (
            RESOLUTION_INFERRED if (hop_inferred or inherited) else RESOLUTION_CONFIRMED
        )
        return ResolvedCallee(
            "project", method, resolution, "; ".join(note for note in notes if note)
        )

    def _resolve_via_imports(
        self, module: ModuleIndex, call: RawCall
    ) -> ResolvedCallee | None:
        binding = None
        for candidate in module.imports:
            if candidate.kind == "star":
                continue
            if candidate.local_name == call.head:
                binding = candidate
                break
        if binding is None:
            return self._resolve_via_star_imports(module, call)
        name_in_module = call.attr_chain[0] if call.attr_chain else None
        resolved = self.index.resolve_import_binding(module, binding, name_in_module)
        if resolved == "external":
            return ResolvedCallee(
                "external", None, RESOLUTION_EXTERNAL,
                note=f"from external module `{binding.source_module}`",
            )
        if resolved == (None, None):
            return ResolvedCallee(
                "unresolved", None, RESOLUTION_UNRESOLVED,
                note=(
                    f"`{name_in_module or binding.local_name}` not found in "
                    f"module `{binding.source_module}`"
                ),
            )
        path, qualname = resolved  # type: ignore[misc]
        remainder = call.attr_chain[1:] if call.attr_chain else ()
        if qualname == "" and remainder:
            base = self.index.module_level_definition(path, remainder[0])
            if base is not None:
                qualname = base.qualname
                remainder = remainder[1:]
        definition = (
            self.index.definition_at(path, qualname) if qualname else None
        )
        for part in remainder:
            if definition is None:
                break
            nested = self.index.definition_at(
                path, f"{definition.qualname}.{part}"
            )
            if nested is None and definition.kind == KIND_CLASS:
                nested = self.index.resolve_method(definition, part)
            definition = nested
            if definition is not None:
                path = definition.path
        if definition is None:
            if qualname == "" and not remainder:
                return ResolvedCallee(
                    "unresolved", None, RESOLUTION_UNRESOLVED,
                    note=f"module `{binding.source_module}` is called directly",
                )
            return ResolvedCallee(
                "unresolved", None, RESOLUTION_UNRESOLVED,
                note=(
                    f"could not resolve `{call.callee_text}` inside "
                    f"`{binding.source_module}`"
                ),
            )
        note = "via import"
        if binding.local_name != (binding.source_name or binding.local_name):
            note = f"via import alias `{binding.local_name}`"
        return ResolvedCallee("project", definition, RESOLUTION_CONFIRMED, note)

    def _resolve_via_star_imports(
        self, module: ModuleIndex, call: RawCall
    ) -> ResolvedCallee | None:
        if module.language != LANGUAGE_PYTHON or call.attr_chain:
            return None
        for binding in module.imports:
            if binding.kind != "star":
                continue
            target_module = self.index.resolve_python_module(
                module.path, binding.source_module
            )
            if target_module is None:
                continue
            resolved = self.index.resolve_python_symbol(target_module, call.head)
            if resolved is not None:
                path, qualname = resolved
                definition = self.index.definition_at(path, qualname)
                if definition is not None:
                    return ResolvedCallee(
                        "project", definition, RESOLUTION_INFERRED,
                        note=f"via star import from `{binding.source_module}`",
                    )
        return None

    def _resolve_by_method_name(
        self, call: RawCall, receiver_note: str
    ) -> ResolvedCallee:
        name = call.attr_chain[-1] if call.attr_chain else call.head
        if name in COMMON_METHOD_NAMES:
            return ResolvedCallee(
                "unresolved", None, RESOLUTION_UNRESOLVED,
                note=(
                    f"receiver unknown and `{name}` is too common to attribute "
                    "by name"
                ),
            )
        candidates = tuple(
            definition
            for definition in self.index.definitions_by_name.get(name, ())
            if definition.kind in {"method", "async_method"}
        )
        if len(candidates) == 1:
            return ResolvedCallee(
                "project", candidates[0], RESOLUTION_INFERRED,
                note=f"matched by method name only (receiver is an {receiver_note})",
            )
        if len(candidates) > 1:
            return ResolvedCallee(
                "ambiguous", None, RESOLUTION_AMBIGUOUS,
                note=f"{len(candidates)} methods named `{name}` in workspace",
                candidates=candidates[: self.limits.max_candidates],
            )
        return ResolvedCallee(
            "unresolved", None, RESOLUTION_UNRESOLVED,
            note=f"no workspace method named `{name}`",
        )

    def _fallback_name_match(
        self, module: ModuleIndex, call: RawCall
    ) -> ResolvedCallee:
        name = call.attr_chain[-1] if call.attr_chain else call.head
        if call.attr_chain and name in COMMON_METHOD_NAMES:
            return ResolvedCallee(
                "unresolved", None, RESOLUTION_UNRESOLVED,
                note=(
                    f"receiver unknown and `{name}` is too common to attribute "
                    "by name"
                ),
            )
        candidates = tuple(
            definition
            for definition in self.index.definitions_by_name.get(name, ())
            if definition.kind in CALLABLE_KINDS
        )
        if call.attr_chain:
            if len(candidates) == 1:
                return ResolvedCallee(
                    "project", candidates[0], RESOLUTION_INFERRED,
                    note="matched by method name only (receiver unknown)",
                )
            if len(candidates) > 1:
                return ResolvedCallee(
                    "ambiguous", None, RESOLUTION_AMBIGUOUS,
                    note=f"{len(candidates)} workspace definitions named `{name}`",
                    candidates=candidates[: self.limits.max_candidates],
                )
            return ResolvedCallee(
                "unresolved", None, RESOLUTION_UNRESOLVED,
                note="receiver unknown; no matching workspace definition",
            )
        if len(candidates) == 1:
            return ResolvedCallee(
                "project", candidates[0], RESOLUTION_INFERRED,
                note="name match; the name is not statically in scope here",
            )
        if len(candidates) > 1:
            return ResolvedCallee(
                "ambiguous", None, RESOLUTION_AMBIGUOUS,
                note=f"{len(candidates)} workspace definitions named `{name}`",
                candidates=candidates[: self.limits.max_candidates],
            )
        return ResolvedCallee(
            "unresolved", None, RESOLUTION_UNRESOLVED,
            note="not defined, imported, or known as a builtin here",
        )

    # ---- shared name lookup -----------------------------------------------------

    def _resolve_name(
        self, module: ModuleIndex, name: str, caller_qualname: str | None
    ) -> SymbolDefinition | None:
        if caller_qualname:
            prefix_parts = caller_qualname.split(".")
            for cut in range(len(prefix_parts), 0, -1):
                prefix = ".".join(prefix_parts[:cut])
                nested = self.index.definition_at(module.path, f"{prefix}.{name}")
                if nested is not None:
                    return nested
        return self.index.module_level_definition(module.path, name)

    def _enclosing_class(
        self, module: ModuleIndex, caller_qualname: str | None
    ) -> SymbolDefinition | None:
        if not caller_qualname:
            return None
        parts = caller_qualname.split(".")
        for cut in range(len(parts) - 1, 0, -1):
            candidate = self.index.definition_at(module.path, ".".join(parts[:cut]))
            if candidate is not None and candidate.kind == KIND_CLASS:
                return candidate
        return None

    def _is_builtin(self, module: ModuleIndex, head: str) -> bool:
        if module.language == LANGUAGE_PYTHON:
            return head in PYTHON_BUILTINS
        return head in JS_GLOBALS

    # ---- quick / deep traces ------------------------------------------------------

    def quick_trace(self, target: SymbolDefinition) -> TraceResult:
        result = self._base_trace("quick", target)
        return result

    def deep_trace(self, target: SymbolDefinition) -> TraceResult:
        result = self._base_trace("deep", target)
        self._extend_deep(result, target)
        return result

    def _base_trace(self, mode: str, target: SymbolDefinition) -> TraceResult:
        limits = self.limits
        module = self.index.module_for(target.path)
        result = TraceResult(mode=mode, ok=True, target=target)
        truncations: list[str] = []
        warnings = list(self.index.warnings[:5])
        if module is None:
            result.ok = False
            result.error = f"Module {target.path} missing from index."
            return result
        if not module.parse_ok:
            warnings.append(
                f"{target.path} was only partially analyzed ({module.parse_error})."
            )

        scope = self._scope_qualnames(target)
        callees: list[CallEdge] = []
        unresolved: list[CallEdge] = []
        callees_truncated = False
        project_call_lines: set[tuple[int, str | None, str]] = set()
        for call in module.calls:
            if call.caller_qualname not in scope:
                continue
            resolved = self.resolve_call(module, call)
            edge = _edge_from_call(target.path, call, resolved, depth=1)
            if resolved.kind == "project":
                project_call_lines.add(
                    (call.line, call.caller_qualname, call.callee_text)
                )
            if resolved.kind in {"dynamic", "unresolved", "ambiguous"}:
                if len(unresolved) < limits.max_callees:
                    unresolved.append(edge)
            if len(callees) < limits.max_callees:
                callees.append(edge)
            else:
                callees_truncated = True
        if callees_truncated:
            truncations.append(f"Callees truncated at {limits.max_callees}.")

        callers, tests = self._find_callers_and_tests(target)
        if len(callers) > limits.max_callers:
            kept = callers[: limits.max_callers]
            dropped = callers[limits.max_callers :]
            ambiguous_dropped = sum(
                1 for edge in dropped if edge.resolution == RESOLUTION_AMBIGUOUS
            )
            detail = (
                f" ({ambiguous_dropped} of the omitted are name-match-only)"
                if ambiguous_dropped
                else ""
            )
            truncations.append(
                f"Callers truncated at {limits.max_callers}; "
                f"{len(dropped)} omitted{detail}."
            )
            callers = kept

        effects = [
            effect
            for effect in module.side_effects
            if effect.qualname in scope
            and not _superseded_inferred_effect(effect, project_call_lines)
        ]
        if len(effects) > limits.max_side_effects:
            truncations.append(
                f"Side effects truncated at {limits.max_side_effects}."
            )
            effects = effects[: limits.max_side_effects]

        error_handling = tuple(
            info for info in module.error_handling if info.qualname in scope
        )
        framework_note = _framework_registration_note(target)
        if framework_note:
            callers = [
                CallEdge(
                    caller_path=target.path,
                    caller_qualname=None,
                    line=target.line,
                    callee_text=framework_note,
                    target_path=target.path,
                    target_qualname=target.qualname,
                    resolution=RESOLUTION_FRAMEWORK,
                    note=framework_note,
                ),
                *callers,
            ]

        result.callers = tuple(callers)
        result.callees = tuple(callees)
        result.unresolved_calls = tuple(unresolved[: limits.max_callees])
        result.side_effects = tuple(effects)
        result.error_handling = error_handling
        result.tests = tuple(tests[: limits.max_tests])
        result.definition_snippet = self._snippet(target)
        result.warnings = tuple(warnings)
        result.truncations = tuple(truncations)
        result.stats = {
            "files_indexed": len(self.index.modules),
            "definitions_indexed": len(self.index.definitions),
            "direct_callers": len(result.callers),
            "direct_callees": len(result.callees),
        }
        return result

    def _extend_deep(self, result: TraceResult, target: SymbolDefinition) -> None:
        limits = self.limits
        truncations = list(result.truncations)
        visited: dict[str, SymbolDefinition] = {target.symbol_id: target}
        edges: list[CallEdge] = []
        parents: dict[str, str] = {}
        frontier: list[tuple[SymbolDefinition, int]] = [(target, 0)]
        depth_truncated = False
        node_truncated = False
        while frontier:
            current, depth = frontier.pop(0)
            if depth >= limits.max_depth:
                depth_truncated = True
                continue
            module = self.index.module_for(current.path)
            if module is None:
                continue
            scope = self._scope_qualnames(current)
            for call in module.calls:
                if call.caller_qualname not in scope:
                    continue
                resolved = self.resolve_call(module, call)
                if resolved.kind != "project" or resolved.definition is None:
                    continue
                callee = resolved.definition
                if callee.symbol_id == current.symbol_id and depth > 0:
                    continue
                if len(edges) >= limits.max_edges:
                    truncations.append(
                        f"Deep traversal stopped at {limits.max_edges} edges."
                    )
                    frontier.clear()
                    break
                edges.append(
                    _edge_from_call(current.path, call, resolved, depth=depth + 1)
                )
                if callee.symbol_id not in visited:
                    if len(visited) >= limits.max_nodes:
                        node_truncated = True
                        continue
                    visited[callee.symbol_id] = callee
                    parents[callee.symbol_id] = current.symbol_id
                    frontier.append((callee, depth + 1))
        if depth_truncated:
            truncations.append(
                f"Call paths deeper than {limits.max_depth} levels were not followed."
            )
        if node_truncated:
            truncations.append(
                f"Deep traversal stopped adding functions at {limits.max_nodes} nodes."
            )

        transitive_effects: list[SideEffect] = []
        for symbol_id, definition in visited.items():
            if symbol_id == target.symbol_id:
                continue
            module = self.index.module_for(definition.path)
            if module is None:
                continue
            scope = self._scope_qualnames(definition)
            for effect in module.side_effects:
                if effect.qualname in scope:
                    transitive_effects.append(effect)
                    if len(transitive_effects) >= limits.max_side_effects:
                        truncations.append(
                            "Transitive side effects truncated at "
                            f"{limits.max_side_effects}."
                        )
                        break
            if len(transitive_effects) >= limits.max_side_effects:
                break

        deep_error_handling = list(result.error_handling)
        seen_error_scopes = {info.qualname for info in deep_error_handling}
        for symbol_id, definition in visited.items():
            if symbol_id == target.symbol_id:
                continue
            module = self.index.module_for(definition.path)
            if module is None:
                continue
            for info in module.error_handling:
                if info.qualname == definition.qualname and info.qualname not in seen_error_scopes:
                    if info.raises or info.catches:
                        deep_error_handling.append(info)
                        seen_error_scopes.add(info.qualname)

        result.call_paths = _representative_paths(target, visited, parents)
        result.cycles = _find_cycles(target, edges, visited)
        result.transitive_side_effects = tuple(transitive_effects)
        result.error_handling = tuple(deep_error_handling[:40])
        result.truncations = tuple(truncations)
        result.stats.update(
            {
                "deep_nodes": len(visited),
                "deep_edges": len(edges),
            }
        )
        # Tests that exercise transitive nodes are added as inferred links.
        existing = {(link.test_path, link.line) for link in result.tests}
        indirect: list[TestLink] = []
        for symbol_id, definition in visited.items():
            if symbol_id == target.symbol_id:
                continue
            if len(result.tests) + len(indirect) >= self.limits.max_tests:
                break
            for link in self._tests_for(definition, via_prefix="indirectly covers"):
                if (link.test_path, link.line) not in existing:
                    indirect.append(link)
                    existing.add((link.test_path, link.line))
                    if len(result.tests) + len(indirect) >= self.limits.max_tests:
                        break
        result.tests = tuple([*result.tests, *indirect][: self.limits.max_tests])

    # ---- callers / tests ---------------------------------------------------------

    def _find_callers_and_tests(
        self, target: SymbolDefinition
    ) -> tuple[list[CallEdge], list[TestLink]]:
        callers: list[CallEdge] = []
        tests: list[TestLink] = []
        for path, module in self.index.modules.items():
            for call in module.calls:
                resolved = self.resolve_call(module, call)
                matched = (
                    resolved.definition is not None
                    and resolved.definition.symbol_id == target.symbol_id
                )
                if not matched and resolved.candidates:
                    matched = any(
                        candidate.symbol_id == target.symbol_id
                        for candidate in resolved.candidates
                    )
                    if matched:
                        resolved = ResolvedCallee(
                            "ambiguous",
                            None,
                            RESOLUTION_AMBIGUOUS,
                            note=resolved.note,
                            candidates=resolved.candidates,
                        )
                if not matched:
                    continue
                edge = CallEdge(
                    caller_path=path,
                    caller_qualname=call.caller_qualname,
                    line=call.line,
                    callee_text=call.callee_text,
                    target_path=target.path,
                    target_qualname=target.qualname,
                    resolution=resolved.resolution,
                    note=resolved.note,
                )
                callers.append(edge)
                if module.is_test:
                    tests.append(
                        TestLink(
                            test_path=path,
                            test_name=call.caller_qualname or "<module>",
                            line=call.line,
                            via="calls target",
                            resolution=resolved.resolution,
                        )
                    )
        rank = {
            RESOLUTION_CONFIRMED: 0,
            RESOLUTION_INFERRED: 1,
            RESOLUTION_FRAMEWORK: 2,
            RESOLUTION_AMBIGUOUS: 3,
        }
        callers.sort(
            key=lambda edge: (
                rank.get(edge.resolution, 4),
                edge.caller_path,
                edge.line,
            )
        )
        return callers, tests

    def _tests_for(
        self, definition: SymbolDefinition, via_prefix: str
    ) -> list[TestLink]:
        links: list[TestLink] = []
        for path, module in self.index.modules.items():
            if not module.is_test:
                continue
            for call in module.calls:
                resolved = self.resolve_call(module, call)
                if (
                    resolved.definition is not None
                    and resolved.definition.symbol_id == definition.symbol_id
                ):
                    links.append(
                        TestLink(
                            test_path=path,
                            test_name=call.caller_qualname or "<module>",
                            line=call.line,
                            via=f"{via_prefix} {definition.qualname}",
                            resolution=RESOLUTION_INFERRED,
                        )
                    )
                    if len(links) >= 5:
                        return links
        return links

    # ---- misc ---------------------------------------------------------------------

    def _scope_qualnames(self, target: SymbolDefinition) -> set[str | None]:
        module = self.index.module_for(target.path)
        scope: set[str | None] = {target.qualname}
        if module is None:
            return scope
        prefix = target.qualname + "."
        for definition in module.definitions:
            if definition.qualname.startswith(prefix):
                scope.add(definition.qualname)
        return scope

    def _snippet(self, target: SymbolDefinition) -> Snippet | None:
        text = self.index.workspace.read_source(target.path)
        if text is None:
            return None
        lines = text.splitlines()
        start = max(1, target.line)
        end = min(len(lines), max(target.end_line, target.line))
        truncated = False
        if end - start + 1 > self.limits.max_snippet_lines:
            end = start + self.limits.max_snippet_lines - 1
            truncated = True
        body = "\n".join(lines[start - 1 : end])
        return Snippet(
            path=target.path,
            start_line=start,
            end_line=end,
            text=redact_secret_values(body),
            truncated=truncated,
        )


# ---- module-level helpers ---------------------------------------------------


def _dedupe(definitions: list[SymbolDefinition]) -> list[SymbolDefinition]:
    seen: set[str] = set()
    unique: list[SymbolDefinition] = []
    for definition in definitions:
        if definition.symbol_id in seen:
            continue
        seen.add(definition.symbol_id)
        unique.append(definition)
    unique.sort(key=lambda item: (item.path, item.line))
    return unique


def _edge_from_call(
    caller_path: str, call: RawCall, resolved: ResolvedCallee, depth: int
) -> CallEdge:
    definition = resolved.definition
    note = resolved.note
    if resolved.candidates and resolved.kind == "ambiguous":
        preview = ", ".join(
            f"{candidate.path}:{candidate.line}" for candidate in resolved.candidates[:3]
        )
        suffix = "…" if len(resolved.candidates) > 3 else ""
        note = f"{note} [candidates: {preview}{suffix}]" if note else preview
    return CallEdge(
        caller_path=caller_path,
        caller_qualname=call.caller_qualname,
        line=call.line,
        callee_text=call.callee_text,
        target_path=definition.path if definition else None,
        target_qualname=definition.qualname if definition else None,
        resolution=resolved.resolution,
        note=note,
        depth=depth,
    )


def _superseded_inferred_effect(
    effect: SideEffect, project_call_lines: set[tuple[int, str | None, str]]
) -> bool:
    """Drop a method-name-match effect when its own call resolved to project
    code (the effect then surfaces transitively through that callee)."""
    if effect.resolution != RESOLUTION_INFERRED:
        return False
    if "(method-name match)" not in effect.detail:
        return False
    callee_text = effect.detail.split(" (method-name match)")[0]
    return (effect.line, effect.qualname, callee_text) in project_call_lines


def _framework_registration_note(target: SymbolDefinition) -> str:
    for decorator in target.decorators:
        lowered = decorator.lower()
        for marker in FRAMEWORK_DECORATOR_MARKERS:
            if marker.rstrip("(") in lowered:
                return f"framework-registered via decorator @{decorator}"
    if target.kind == KIND_COMPONENT:
        return ""
    return ""


def _representative_paths(
    target: SymbolDefinition,
    visited: dict[str, SymbolDefinition],
    parents: dict[str, str],
) -> tuple[tuple[str, ...], ...]:
    children: set[str] = set(parents.values())
    leaves = [symbol_id for symbol_id in visited if symbol_id not in children]
    paths: list[tuple[str, ...]] = []
    for leaf in leaves:
        if leaf == target.symbol_id:
            continue
        chain: list[str] = []
        cursor: str | None = leaf
        for _ in range(24):
            if cursor is None:
                break
            chain.append(cursor)
            cursor = parents.get(cursor)
        chain.reverse()
        if chain and chain[0] != target.symbol_id:
            chain.insert(0, target.symbol_id)
        paths.append(tuple(chain))
        if len(paths) >= 12:
            break
    paths.sort(key=len, reverse=True)
    return tuple(paths[:8])


def _find_cycles(
    target: SymbolDefinition,
    edges: list[CallEdge],
    visited: dict[str, SymbolDefinition],
) -> tuple[tuple[str, ...], ...]:
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        if edge.target_path is None or edge.target_qualname is None:
            continue
        source_id = None
        for symbol_id, definition in visited.items():
            if (
                definition.path == edge.caller_path
                and definition.qualname == edge.caller_qualname
            ):
                source_id = symbol_id
                break
        if source_id is None:
            # The caller may be a nested scope of a visited definition.
            for symbol_id, definition in visited.items():
                if definition.path == edge.caller_path and (
                    edge.caller_qualname or ""
                ).startswith(definition.qualname):
                    source_id = symbol_id
                    break
        if source_id is None:
            continue
        adjacency.setdefault(source_id, set()).add(
            f"{edge.target_path}::{edge.target_qualname}"
        )

    cycles: list[tuple[str, ...]] = []
    stack: list[str] = []
    on_stack: set[str] = set()
    explored: set[str] = set()

    def depth_first(node: str) -> None:
        if len(cycles) >= 5:
            return
        stack.append(node)
        on_stack.add(node)
        for neighbor in sorted(adjacency.get(node, ())):
            if neighbor in on_stack:
                start = stack.index(neighbor)
                cycles.append(tuple([*stack[start:], neighbor]))
                if len(cycles) >= 5:
                    break
            elif neighbor not in explored and neighbor in visited:
                depth_first(neighbor)
        stack.pop()
        on_stack.discard(node)
        explored.add(node)

    depth_first(target.symbol_id)
    return tuple(cycles)
