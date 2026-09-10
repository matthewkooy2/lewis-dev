"""Workspace-wide symbol index and import resolution for code tracing.

Builds per-file ``ModuleIndex`` records through the safe workspace, then
answers the cross-file questions the engine needs: which file a module
specifier points to, which definition an imported name refers to (following
aliases and bounded re-export chains), which class provides a method
(following bounded inheritance), and which definitions share a name.

Python resolution covers absolute and relative imports, packages
(``__init__.py``), aliases, and star imports. JavaScript/TypeScript
resolution covers relative specifiers with extension/index probing,
CommonJS ``require``, and ``tsconfig.json``/``jsconfig.json`` ``paths``
aliases. Bare specifiers are external. Nothing here executes analyzed code.
"""

from __future__ import annotations

import json
import re
from pathlib import PurePosixPath

from app.trace.js_analysis import analyze_js_file
from app.trace.model import (
    ImportBinding,
    KIND_CLASS,
    LANGUAGE_PYTHON,
    ModuleIndex,
    SymbolDefinition,
)
from app.trace.python_analysis import analyze_python_file
from app.trace.workspace import TraceWorkspace


MAX_REEXPORT_DEPTH = 6
MAX_BASE_CLASS_DEPTH = 6
JS_EXTENSION_ORDER = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

_TYPE_TEXT_CLEANUP = re.compile(r"\b(?:Optional|Final|ClassVar)\[(.*)\]$")
_FIRST_DOTTED_NAME = re.compile(r"[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*")


class ProjectIndex:
    """Analyzed modules plus cross-file resolution helpers."""

    def __init__(self, workspace: TraceWorkspace):
        self.workspace = workspace
        self.modules: dict[str, ModuleIndex] = {}
        self.definitions: dict[tuple[str, str], SymbolDefinition] = {}
        self.definitions_by_name: dict[str, list[SymbolDefinition]] = {}
        self.python_modules: dict[str, str] = {}
        self.js_alias_patterns: list[tuple[str, str]] = []
        self.js_base_url: str = ""
        self.warnings: list[str] = []
        self._file_set: set[str] = set()

    # ---- construction ------------------------------------------------------

    @classmethod
    def build(cls, workspace: TraceWorkspace) -> ProjectIndex:
        index = cls(workspace)
        files = workspace.list_source_files()
        if workspace.discovery_truncated:
            index.warnings.append(
                f"File discovery stopped at {workspace.limits.max_files} files; "
                "some sources were not analyzed."
            )
        index._file_set = {record.path for record in files}
        index._load_js_aliases()
        for record in files:
            text = workspace.read_source(record.path)
            if text is None:
                continue
            if record.language == LANGUAGE_PYTHON:
                module = analyze_python_file(record.path, text)
            else:
                module = analyze_js_file(record.path, text, record.language)
            index._register(module)
        return index

    def _register(self, module: ModuleIndex) -> None:
        self.modules[module.path] = module
        if not module.parse_ok:
            self.warnings.append(
                f"{module.path}: analysis degraded ({module.parse_error})."
            )
        if module.language == LANGUAGE_PYTHON:
            dotted = _python_module_name(module.path)
            if dotted:
                self.python_modules.setdefault(dotted, module.path)
        for definition in module.definitions:
            self.definitions[(module.path, definition.qualname)] = definition
            self.definitions_by_name.setdefault(definition.name, []).append(definition)

    def _load_js_aliases(self) -> None:
        for name in ("tsconfig.json", "jsconfig.json"):
            raw = self._read_config_text(name)
            if raw is None:
                continue
            data = _parse_tolerant_json(raw)
            if not isinstance(data, dict):
                continue
            options = data.get("compilerOptions")
            if not isinstance(options, dict):
                continue
            base_url = options.get("baseUrl")
            if isinstance(base_url, str):
                self.js_base_url = base_url.strip("./") or ""
            paths = options.get("paths")
            if isinstance(paths, dict):
                for pattern, targets in paths.items():
                    if not isinstance(targets, list) or not targets:
                        continue
                    target = targets[0]
                    if isinstance(pattern, str) and isinstance(target, str):
                        self.js_alias_patterns.append((pattern, target))
            return

    def _read_config_text(self, name: str) -> str | None:
        # tsconfig/jsconfig are JSON, outside the analyzable-suffix set, so
        # read them directly with the same containment/size checks.
        candidate = self.workspace.resolved_root / name
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.workspace.resolved_root)
            if candidate.is_symlink() or not resolved.is_file():
                return None
            if resolved.stat().st_size > self.workspace.limits.max_file_bytes:
                return None
            data = resolved.read_bytes()
        except (OSError, ValueError):
            return None
        if b"\x00" in data:
            return None
        try:
            return data.decode("utf-8")
        except UnicodeDecodeError:
            return None

    # ---- lookups -------------------------------------------------------------

    def module_for(self, path: str) -> ModuleIndex | None:
        return self.modules.get(path)

    def definition_at(self, path: str, qualname: str) -> SymbolDefinition | None:
        return self.definitions.get((path, qualname))

    def module_level_definition(self, path: str, name: str) -> SymbolDefinition | None:
        definition = self.definitions.get((path, name))
        if definition is not None and definition.parent is None:
            return definition
        return None

    # ---- Python module resolution ---------------------------------------------

    def resolve_python_module(self, from_path: str, source_module: str) -> str | None:
        """Workspace file for a Python import specifier; None when external."""
        if not source_module:
            return None
        if source_module.startswith("."):
            level = len(source_module) - len(source_module.lstrip("."))
            remainder = source_module[level:]
            package_parts = list(PurePosixPath(from_path).parent.parts)
            if PurePosixPath(from_path).name == "__init__.py":
                package_parts = package_parts
            climb = level - 1
            if climb > len(package_parts):
                return None
            base_parts = package_parts[: len(package_parts) - climb]
            dotted = ".".join(
                part for part in [*base_parts, *remainder.split(".")] if part
            )
        else:
            dotted = source_module
        if dotted in self.python_modules:
            return self.python_modules[dotted]
        return None

    def resolve_python_symbol(
        self, module_path: str, name: str, depth: int = 0
    ) -> tuple[str, str] | None:
        """(path, qualname) for ``name`` exported by a Python module."""
        if depth > MAX_REEXPORT_DEPTH:
            return None
        module = self.modules.get(module_path)
        if module is None:
            return None
        definition = self.module_level_definition(module_path, name)
        if definition is not None:
            return module_path, definition.qualname
        for binding in module.imports:
            if binding.kind == "named" and binding.local_name == name:
                target_module = self.resolve_python_module(
                    module_path, binding.source_module
                )
                if target_module is not None and binding.source_name:
                    resolved = self.resolve_python_symbol(
                        target_module, binding.source_name, depth + 1
                    )
                    if resolved is not None:
                        return resolved
            if binding.kind == "star":
                target_module = self.resolve_python_module(
                    module_path, binding.source_module
                )
                if target_module is not None:
                    resolved = self.resolve_python_symbol(
                        target_module, name, depth + 1
                    )
                    if resolved is not None:
                        return resolved
        submodule = self.resolve_python_module(module_path, name)
        if submodule is not None:
            return submodule, ""
        return None

    # ---- JS module resolution ----------------------------------------------

    def resolve_js_module(self, from_path: str, specifier: str) -> str | None:
        """Workspace file for a JS/TS import specifier; None when external."""
        if not specifier or specifier == "<computed>":
            return None
        cleaned = specifier.split("?")[0].split("#")[0]
        if cleaned.startswith("."):
            base = PurePosixPath(from_path).parent
            combined = _normalize_relative(base, cleaned)
            if combined is None:
                return None
            return self._probe_js_file(combined)
        for pattern, target in self.js_alias_patterns:
            mapped = _apply_alias(pattern, target, cleaned)
            if mapped is not None:
                probed = self._probe_js_file(mapped)
                if probed is not None:
                    return probed
        if self.js_base_url:
            probed = self._probe_js_file(f"{self.js_base_url}/{cleaned}")
            if probed is not None:
                return probed
        return None

    def _probe_js_file(self, base: str) -> str | None:
        base = base.lstrip("/")
        if not base:
            return None
        candidates = [base] if PurePosixPath(base).suffix else []
        if not candidates:
            candidates = [f"{base}{ext}" for ext in JS_EXTENSION_ORDER]
            candidates += [f"{base}/index{ext}" for ext in JS_EXTENSION_ORDER]
        else:
            # Explicit ".js" specifiers in TS projects compile from ".ts".
            suffix = PurePosixPath(base).suffix
            if suffix == ".js":
                stem = base[: -len(suffix)]
                candidates += [f"{stem}.ts", f"{stem}.tsx"]
        for candidate in candidates:
            if candidate in self._file_set:
                return candidate
        return None

    def resolve_js_symbol(
        self, module_path: str, exported_name: str, depth: int = 0
    ) -> tuple[str, str] | None:
        """(path, qualname) for a name exported by a JS/TS module."""
        if depth > MAX_REEXPORT_DEPTH:
            return None
        module = self.modules.get(module_path)
        if module is None:
            return None
        star_sources: list[str] = []
        for binding in module.exports:
            if binding.kind == "star":
                if binding.source_module and binding.exported_name in {
                    "*",
                    exported_name,
                }:
                    if (
                        binding.exported_name == exported_name
                        and binding.exported_name != "*"
                    ):
                        # `export * as ns from "m"`: the namespace itself.
                        target = self.resolve_js_module(
                            module_path, binding.source_module
                        )
                        if target is not None:
                            return target, ""
                    star_sources.append(binding.source_module)
                continue
            if binding.exported_name != exported_name:
                continue
            if binding.source_module:
                target = self.resolve_js_module(module_path, binding.source_module)
                if target is None:
                    return None
                source_name = binding.source_name or exported_name
                return self.resolve_js_symbol(target, source_name, depth + 1)
            if binding.local_name:
                definition = self.module_level_definition(
                    module_path, binding.local_name
                )
                if definition is not None:
                    return module_path, definition.qualname
                container = self.definitions.get((module_path, binding.local_name))
                if container is not None:
                    return module_path, container.qualname
            return module_path, binding.local_name or exported_name
        for source in star_sources:
            target = self.resolve_js_module(module_path, source)
            if target is not None:
                resolved = self.resolve_js_symbol(target, exported_name, depth + 1)
                if resolved is not None:
                    return resolved
        definition = self.module_level_definition(module_path, exported_name)
        if definition is not None and definition.exported:
            return module_path, definition.qualname
        return None

    # ---- shared binding resolution ---------------------------------------------

    def resolve_import_binding(
        self, module: ModuleIndex, binding: ImportBinding, name_in_module: str | None
    ) -> tuple[str, str] | tuple[None, None] | str:
        """Resolve one import binding.

        Returns (path, qualname) inside the workspace, the string
        ``"external"`` for out-of-workspace modules, or (None, None) when the
        module resolves but the name cannot be found.
        """
        if module.language == LANGUAGE_PYTHON:
            target_module = self.resolve_python_module(
                module.path, binding.source_module
            )
            if target_module is None:
                return "external"
            if binding.kind == "module":
                if name_in_module is None:
                    return target_module, ""
                resolved = self.resolve_python_symbol(target_module, name_in_module)
                return resolved if resolved is not None else (None, None)
            wanted = binding.source_name or binding.local_name
            resolved = self.resolve_python_symbol(target_module, wanted)
            if resolved is None:
                return (None, None)
            if name_in_module:
                # Attribute access below an imported symbol (Class.method).
                path, qualname = resolved
                nested = self.definitions.get(
                    (
                        path,
                        f"{qualname}.{name_in_module}" if qualname else name_in_module,
                    )
                )
                if nested is not None:
                    return path, nested.qualname
            return resolved
        target_module = self.resolve_js_module(module.path, binding.source_module)
        if target_module is None:
            return "external"
        if (
            binding.kind in {"namespace", "require", "module"}
            and binding.source_name is None
        ):
            if name_in_module is None:
                return target_module, ""
            resolved = self.resolve_js_symbol(target_module, name_in_module)
            return resolved if resolved is not None else (None, None)
        if binding.kind == "default":
            resolved = self.resolve_js_symbol(target_module, "default")
        else:
            wanted = binding.source_name or binding.local_name
            resolved = self.resolve_js_symbol(target_module, wanted)
        if resolved is None:
            return (None, None)
        if name_in_module:
            path, qualname = resolved
            nested = self.definitions.get((path, f"{qualname}.{name_in_module}"))
            if nested is not None:
                return path, nested.qualname
        return resolved

    # ---- classes ---------------------------------------------------------------

    def resolve_type_text(
        self, module: ModuleIndex, type_text: str
    ) -> SymbolDefinition | None:
        """Best-effort class definition for annotation/constructor text."""
        if not type_text:
            return None
        text = type_text.strip()
        match = _TYPE_TEXT_CLEANUP.match(text)
        if match:
            text = match.group(1)
        text = text.split("|")[0].strip()
        name_match = _FIRST_DOTTED_NAME.search(text)
        if name_match is None:
            return None
        dotted = name_match.group(0)
        parts = dotted.split(".")
        local = self.definitions.get((module.path, dotted))
        if local is not None:
            return local
        head, rest = parts[0], parts[1:]
        definition = self.module_level_definition(module.path, head)
        if definition is not None and not rest:
            return definition
        for binding in module.imports:
            if binding.local_name != head or binding.kind == "star":
                continue
            resolved = self.resolve_import_binding(
                module, binding, rest[0] if rest else None
            )
            if resolved == "external" or resolved == (None, None):
                return None
            path, qualname = resolved  # type: ignore[misc]
            if qualname:
                found = self.definitions.get((path, qualname))
                if found is not None:
                    return found
            elif rest:
                target = self.modules.get(path)
                if target is not None:
                    nested = self.module_level_definition(path, rest[0])
                    if nested is not None:
                        return nested
        return None

    def resolve_method(
        self,
        class_def: SymbolDefinition,
        method_name: str,
        depth: int = 0,
    ) -> SymbolDefinition | None:
        """Method on a class or, bounded, on its base classes."""
        if depth > MAX_BASE_CLASS_DEPTH:
            return None
        direct = self.definitions.get(
            (class_def.path, f"{class_def.qualname}.{method_name}")
        )
        if direct is not None:
            return direct
        module = self.modules.get(class_def.path)
        if module is None:
            return None
        for base_text in module.class_bases.get(class_def.qualname, ()):
            base_def = self.resolve_type_text(module, base_text)
            if base_def is not None and base_def.kind == KIND_CLASS:
                found = self.resolve_method(base_def, method_name, depth + 1)
                if found is not None:
                    return found
        return None

    def subclass_overrides(
        self, class_def: SymbolDefinition, method_name: str
    ) -> list[SymbolDefinition]:
        """Definitions overriding ``method_name`` in direct subclasses."""
        overrides: list[SymbolDefinition] = []
        for path, module in self.modules.items():
            for qualname, bases in module.class_bases.items():
                for base_text in bases:
                    base_def = self.resolve_type_text(module, base_text)
                    if (
                        base_def is not None
                        and base_def.path == class_def.path
                        and base_def.qualname == class_def.qualname
                    ):
                        override = self.definitions.get(
                            (path, f"{qualname}.{method_name}")
                        )
                        if override is not None:
                            overrides.append(override)
        return overrides


# ---- helpers ----------------------------------------------------------------


def _python_module_name(path: str) -> str:
    posix = PurePosixPath(path)
    if posix.suffix != ".py":
        return ""
    parts = list(posix.parts)
    parts[-1] = posix.stem
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(part for part in parts if part)


def _normalize_relative(base: PurePosixPath, specifier: str) -> str | None:
    parts = list(base.parts)
    for segment in specifier.split("/"):
        if segment in {"", "."}:
            continue
        if segment == "..":
            if not parts:
                return None
            parts.pop()
            continue
        parts.append(segment)
    return "/".join(parts)


def _apply_alias(pattern: str, target: str, specifier: str) -> str | None:
    if pattern.endswith("/*") and target.endswith("/*"):
        prefix = pattern[:-1]
        if specifier.startswith(prefix):
            return target[:-1] + specifier[len(prefix) :]
        return None
    if pattern == specifier:
        return target
    return None


def _parse_tolerant_json(raw: str) -> object | None:
    """Parse tsconfig-style JSON that may contain comments/trailing commas."""
    without_comments = re.sub(r"//[^\n]*", "", raw)
    without_comments = re.sub(r"/\*.*?\*/", "", without_comments, flags=re.DOTALL)
    without_trailing = re.sub(r",\s*([}\]])", r"\1", without_comments)
    try:
        return json.loads(without_trailing)
    except json.JSONDecodeError:
        return None
