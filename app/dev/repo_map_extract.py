"""Pure, deterministic static extractors for the persistent repo map.

Given a repo-relative path and already-safety-cleared file text, this module
derives the structural facts the repo map records: a per-file descriptor and
responsibility summary, Python/JS/CSS/test symbols, web routes, CSS selectors
and layout rules, JS DOM lookups and emitted classes, test relationships,
retrieval chunks, and the relationship edges that connect them.

Everything here is pure string and AST work: this module performs no file,
process, model, or network I/O, opens no files, and imports nothing from the
rest of the application. Safety classification (approval-sensitive files,
secret content, excluded paths) is decided by the caller before any text is
handed in; an approval-sensitive file is only ever described as metadata and
its symbols, selectors, and chunks are never extracted.
"""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
from pathlib import PurePosixPath
import re

SCHEMA_VERSION = 1

MAX_SYMBOLS_PER_FILE = 400
MAX_SELECTORS_PER_FILE = 600
MAX_JS_TARGETS_PER_FILE = 400
MAX_CHUNKS_PER_FILE = 300
MAX_IMPORTS_PER_FILE = 200
MAX_CHUNK_LINES = 200
MAX_SUMMARY_CHARS = 200

# Roles a safe source file can play in the repo. ``approval_sensitive_metadata``
# is the one role whose files are described but never parsed for symbols,
# selectors, or chunks, so the map never makes them easier to edit.
ROLE_TEST = "test"
ROLE_WEB_ROUTE = "web_route"
ROLE_WEB_PAGE = "web_page"
ROLE_STATIC_CSS = "static_css"
ROLE_STATIC_JS = "static_js"
ROLE_FREESTYLE_MACHINERY = "freestyle_machinery"
ROLE_APPROVAL_SENSITIVE = "approval_sensitive_metadata"
ROLE_NORMAL_SOURCE = "normal_source"

_LANGUAGE_BY_SUFFIX = {
    "py": "python",
    "js": "javascript",
    "mjs": "javascript",
    "jsx": "javascript",
    "ts": "typescript",
    "tsx": "typescript",
    "css": "css",
    "html": "html",
    "htm": "html",
    "md": "markdown",
    "rst": "restructuredtext",
    "json": "json",
    "toml": "toml",
    "yaml": "yaml",
    "yml": "yaml",
    "cfg": "ini",
    "ini": "ini",
    "txt": "text",
    "sql": "sql",
    "csv": "csv",
}

# HTML tags whose presence in a Python string marks a page-builder module that
# emits markup, so it is inspected as a rendered web page rather than plain
# source. Kept conservative to avoid classifying ordinary prose as markup.
_PYTHON_MARKUP_SIGNAL = re.compile(
    r"</?(?:html|head|body|div|span|ul|ol|li|table|section|nav|main|header|"
    r"footer|button|form|input|a|h[1-6]|p)\b",
    re.IGNORECASE,
)

# FastAPI/Flask-style route decorators: ``@app.get("/calendar")`` and friends.
_ROUTE_DECORATOR = re.compile(
    r"@(?P<obj>[A-Za-z_][\w.]*)\.(?P<method>get|post|put|patch|delete|head|"
    r"options|route|websocket)\s*\(\s*(?P<quote>['\"])(?P<path>[^'\"]+)(?P=quote)",
)

# Static asset references in markup/strings: ``/static/lewis.css``,
# ``app/static/calendar.js`` and similar.
_ASSET_REFERENCE = re.compile(
    r"""['"(=\s](?P<ref>/?(?:[\w./-]*?)static/[\w./-]+\.(?:css|js|mjs))""",
    re.IGNORECASE,
)

# CSS selector blocks: capture the selector list preceding each ``{ ... }``.
_CSS_BLOCK = re.compile(r"(?P<sel>[^{}]+)\{(?P<body>[^{}]*)\}", re.DOTALL)
_CSS_CLASS = re.compile(r"\.(-?[_a-zA-Z][\w-]*)")
_CSS_ID = re.compile(r"#(-?[_a-zA-Z][\w-]*)")
_CSS_LAYOUT_DECLS = (
    "display:grid",
    "display:flex",
    "display:inline-grid",
    "display:inline-flex",
    "grid-template-columns",
    "grid-template-rows",
    "grid-template-areas",
    "flex-direction",
    "flex-wrap",
    "grid-auto-flow",
    "place-items",
    "justify-content",
    "align-items",
)

# JS DOM lookups that must resolve against rendered markup (a missing target is
# a real defect) versus runtime-managed classes (applied via the DOM API, never
# a missing target but still a class the page presents).
_JS_LOOKUP_ID = re.compile(r"getElementById\(\s*['\"]([\w-]+)['\"]")
_JS_LOOKUP_CLASS = re.compile(r"getElementsByClassName\(\s*['\"]([\w-]+)['\"]")
_JS_QUERY = re.compile(r"querySelector(?:All)?\(\s*['\"]([^'\"]+)['\"]")
_JS_MANAGED_CLASS = re.compile(
    r"classList\.(?:add|remove|toggle|replace)\(\s*['\"]([\w-]+)['\"]"
)
_JS_CLASSNAME_ASSIGN = re.compile(r"\.className\s*=\s*['\"]([^'\"]+)['\"]")
_JS_FUNCTION = re.compile(
    r"(?:^|\n)\s*(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(",
)
_JS_ADD_LISTENER = re.compile(
    r"\.addEventListener\(\s*['\"]([\w-]+)['\"]",
)
_SIMPLE_SELECTOR_TOKEN = re.compile(r"[.#]([\w-]+)")


@dataclass(frozen=True)
class FileDescriptor:
    path: str
    content_hash: str
    language: str
    role: str
    safety_classification: str
    size_bytes: int
    line_count: int
    imports: tuple[str, ...]
    responsibility_summary: str

    def to_record(self) -> dict[str, object]:
        return {
            "path": self.path,
            "content_hash": self.content_hash,
            "language": self.language,
            "role": self.role,
            "safety_classification": self.safety_classification,
            "size_bytes": self.size_bytes,
            "line_count": self.line_count,
            "imports": list(self.imports),
            "responsibility_summary": self.responsibility_summary,
        }


@dataclass(frozen=True)
class ExtractedFile:
    """Every record derived from one safe file, ready to persist."""

    descriptor: FileDescriptor
    symbols: tuple[dict[str, object], ...] = ()
    routes: tuple[dict[str, object], ...] = ()
    css_selectors: tuple[dict[str, object], ...] = ()
    js_assets: tuple[dict[str, object], ...] = ()
    tests: tuple[dict[str, object], ...] = ()
    chunks: tuple[dict[str, object], ...] = ()
    relationships: tuple[dict[str, object], ...] = field(default=())


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def language_for_path(path: str) -> str:
    suffix = PurePosixPath(path).suffix.lstrip(".").casefold()
    return _LANGUAGE_BY_SUFFIX.get(suffix, "other")


def is_static_asset_path(path: str) -> bool:
    lowered = path.casefold()
    return "static/" in lowered and lowered.rsplit(".", 1)[-1] in {
        "css",
        "js",
        "mjs",
    }


def is_test_path(path: str) -> bool:
    lowered = path.casefold()
    name = PurePosixPath(lowered).name
    return (
        lowered.startswith("tests/")
        or "/tests/" in lowered
        or name.startswith("test_")
        or name.endswith("_test.py")
        or name == "conftest.py"
    )


def classify_role(path: str, text: str, *, approval_sensitive: bool = False) -> str:
    """Classify one safe file into a fixed repo-map role.

    ``approval_sensitive`` is decided by the caller (it owns the approval
    policy predicates); when set, the file is recorded only as metadata.
    """
    if approval_sensitive:
        return ROLE_APPROVAL_SENSITIVE
    lowered = path.casefold()
    suffix = lowered.rsplit(".", 1)[-1] if "." in lowered else ""
    if is_test_path(path):
        return ROLE_TEST
    if "app/dev/freestyle" in lowered or lowered.startswith("app/dev/repo_map"):
        return ROLE_FREESTYLE_MACHINERY
    if suffix == "css":
        return ROLE_STATIC_CSS
    if suffix in {"js", "mjs", "jsx"}:
        return ROLE_STATIC_JS
    if suffix == "py":
        if _ROUTE_DECORATOR.search(text):
            return ROLE_WEB_ROUTE
        if _PYTHON_MARKUP_SIGNAL.search(text):
            return ROLE_WEB_PAGE
    if suffix in {"html", "htm"}:
        return ROLE_WEB_PAGE
    return ROLE_NORMAL_SOURCE


def extract_file(
    path: str,
    text: str,
    *,
    safety_classification: str = "safe",
    approval_sensitive: bool = False,
) -> ExtractedFile:
    """Derive every repo-map record for one already-safety-cleared file.

    An approval-sensitive file yields only its descriptor; its body is never
    parsed for symbols, selectors, assets, or chunks.
    """
    role = classify_role(path, text, approval_sensitive=approval_sensitive)
    language = language_for_path(path)
    imports = () if approval_sensitive else _imports(path, text, language)
    descriptor = FileDescriptor(
        path=path,
        content_hash=content_hash(text),
        language=language,
        role=role,
        safety_classification=(
            "approval_sensitive" if approval_sensitive else safety_classification
        ),
        size_bytes=len(text.encode("utf-8")),
        line_count=text.count("\n") + (1 if text and not text.endswith("\n") else 0),
        imports=imports,
        responsibility_summary=_responsibility_summary(path, role, text, language),
    )
    if approval_sensitive:
        return ExtractedFile(descriptor=descriptor)

    symbols = _symbols(path, text, language)
    routes = _routes(path, text) if language == "python" else ()
    selectors = _css_selectors(path, text) if language == "css" else ()
    js = _js_asset(path, text) if language in {"javascript", "typescript"} else ()
    tests = _tests(path, text) if role == ROLE_TEST and language == "python" else ()
    chunks = _chunks(path, text, language, role, symbols, selectors, js)
    relationships = _relationships(path, role, language, text, routes, js)
    return ExtractedFile(
        descriptor=descriptor,
        symbols=tuple(symbols[:MAX_SYMBOLS_PER_FILE]),
        routes=tuple(routes),
        css_selectors=tuple(selectors[:MAX_SELECTORS_PER_FILE]),
        js_assets=tuple(js),
        tests=tuple(tests),
        chunks=tuple(chunks[:MAX_CHUNKS_PER_FILE]),
        relationships=tuple(relationships),
    )


def _imports(path: str, text: str, language: str) -> tuple[str, ...]:
    if language == "python":
        return _python_imports(text)
    if language in {"javascript", "typescript"}:
        return _js_imports(text)
    return ()


def _python_imports(text: str) -> tuple[str, ...]:
    tree = _safe_python_ast(text)
    if tree is None:
        return ()
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    return _bounded_unique(found, MAX_IMPORTS_PER_FILE)


def _js_imports(text: str) -> tuple[str, ...]:
    found = re.findall(r"""import\s+[^'\";]*from\s+['"]([^'"]+)['"]""", text)
    found.extend(re.findall(r"""require\(\s*['"]([^'"]+)['"]\s*\)""", text))
    return _bounded_unique(found, MAX_IMPORTS_PER_FILE)


def _symbols(path: str, text: str, language: str) -> list[dict[str, object]]:
    if language == "python":
        return _python_symbols(path, text)
    if language in {"javascript", "typescript"}:
        return _js_symbols(path, text)
    if language == "css":
        return [
            {
                "path": path,
                "name": str(selector["selector"]),
                "kind": "css_selector",
                "line": selector["line"],
            }
            for selector in _css_selectors(path, text)[:MAX_SYMBOLS_PER_FILE]
        ]
    return []


def _python_symbols(path: str, text: str) -> list[dict[str, object]]:
    tree = _safe_python_ast(text)
    if tree is None:
        return []
    symbols: list[dict[str, object]] = []

    def record(name: str, kind: str, node: ast.AST, parent: str | None) -> None:
        symbols.append(
            {
                "path": path,
                "name": name,
                "kind": kind,
                "line": getattr(node, "lineno", 0),
                "end_line": getattr(node, "end_lineno", getattr(node, "lineno", 0)),
                "parent": parent,
            }
        )

    def visit(node: ast.AST, parent: str | None) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                kind = "method" if parent else "function"
                name = child.name
                if name.startswith("test_") and parent is None:
                    kind = "test_function"
                elif _has_fixture_decorator(child):
                    kind = "fixture"
                elif _has_route_decorator(child):
                    kind = "route_handler"
                record(name, kind, child, parent)
                visit(child, name)
            elif isinstance(child, ast.ClassDef):
                record(child.name, "class", child, parent)
                visit(child, child.name)
            elif isinstance(child, ast.Assign) and parent is None:
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        record(target.id, "constant", child, None)

    visit(tree, None)
    return symbols[:MAX_SYMBOLS_PER_FILE]


def _has_fixture_decorator(node: ast.AST) -> bool:
    for decorator in getattr(node, "decorator_list", []):
        if "fixture" in _decorator_text(decorator):
            return True
    return False


def _has_route_decorator(node: ast.AST) -> bool:
    for decorator in getattr(node, "decorator_list", []):
        text = _decorator_text(decorator)
        if re.search(r"\.(get|post|put|patch|delete|route|websocket)$", text):
            return True
    return False


def _decorator_text(node: ast.AST) -> str:
    target = node.func if isinstance(node, ast.Call) else node
    parts: list[str] = []
    while isinstance(target, ast.Attribute):
        parts.append(target.attr)
        target = target.value
    if isinstance(target, ast.Name):
        parts.append(target.id)
    return ".".join(reversed(parts))


def _js_symbols(path: str, text: str) -> list[dict[str, object]]:
    symbols: list[dict[str, object]] = []
    for match in _JS_FUNCTION.finditer(text):
        symbols.append(
            {
                "path": path,
                "name": match.group(1),
                "kind": "js_function",
                "line": text.count("\n", 0, match.start()) + 1,
            }
        )
    for match in re.finditer(
        r"(?:^|\n)\s*(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*"
        r"(?:async\s*)?\([^)]*\)\s*=>",
        text,
    ):
        symbols.append(
            {
                "path": path,
                "name": match.group(1),
                "kind": "js_function",
                "line": text.count("\n", 0, match.start()) + 1,
            }
        )
    return symbols[:MAX_SYMBOLS_PER_FILE]


def _routes(path: str, text: str) -> list[dict[str, object]]:
    lines = text.splitlines()
    routes: list[dict[str, object]] = []
    for match in _ROUTE_DECORATOR.finditer(text):
        line = text.count("\n", 0, match.start()) + 1
        method = match.group("method").upper()
        if method in {"ROUTE", "WEBSOCKET"}:
            method = "GET" if method == "ROUTE" else "WEBSOCKET"
        handler, end_line = _route_handler_after(lines, line)
        block = "\n".join(lines[line - 1 : (end_line or line)])
        routes.append(
            {
                "path": path,
                "route": match.group("path"),
                "method": method,
                "handler": handler,
                "line": line,
                "end_line": end_line or line,
                "referenced_assets": list(_referenced_assets(block)),
            }
        )
    return routes


def _route_handler_after(lines: Sequence[str], start_line: int) -> tuple[str | None, int]:
    for offset in range(start_line, min(start_line + 6, len(lines)) + 1):
        candidate = lines[offset - 1] if offset - 1 < len(lines) else ""
        match = re.match(r"\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\(", candidate)
        if match:
            return match.group(1), offset
    return None, start_line


def _css_selectors(path: str, text: str) -> list[dict[str, object]]:
    selectors: list[dict[str, object]] = []
    stripped = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    for match in _CSS_BLOCK.finditer(stripped):
        selector_text = match.group("sel").strip()
        if not selector_text or selector_text.startswith("@"):
            continue
        body = match.group("body")
        normalized_body = re.sub(r"\s+", "", body).casefold()
        layout = [decl for decl in _CSS_LAYOUT_DECLS if decl in normalized_body]
        line = text.count("\n", 0, match.start()) + 1
        for selector in re.split(r",", selector_text):
            selector = selector.strip()
            if not selector:
                continue
            selectors.append(
                {
                    "path": path,
                    "selector": selector,
                    "classes": list(dict.fromkeys(_CSS_CLASS.findall(selector)))[:32],
                    "ids": list(dict.fromkeys(_CSS_ID.findall(selector)))[:32],
                    "layout": layout,
                    "line": line,
                }
            )
            if len(selectors) >= MAX_SELECTORS_PER_FILE:
                return selectors
    return selectors


def _js_asset(path: str, text: str) -> list[dict[str, object]]:
    lookup_ids = _bounded_unique(_JS_LOOKUP_ID.findall(text), MAX_JS_TARGETS_PER_FILE)
    lookup_classes = list(
        dict.fromkeys(_JS_LOOKUP_CLASS.findall(text))
    )[:MAX_JS_TARGETS_PER_FILE]
    query_tokens: list[str] = []
    for selector in _JS_QUERY.findall(text):
        query_tokens.extend(_SIMPLE_SELECTOR_TOKEN.findall(selector))
    lookup_classes = list(dict.fromkeys([*lookup_classes, *query_tokens]))[
        :MAX_JS_TARGETS_PER_FILE
    ]
    managed = list(dict.fromkeys(_JS_MANAGED_CLASS.findall(text)))
    for assigned in _JS_CLASSNAME_ASSIGN.findall(text):
        managed.extend(assigned.split())
    managed = list(dict.fromkeys(managed))[:MAX_JS_TARGETS_PER_FILE]
    events = _bounded_unique(_JS_ADD_LISTENER.findall(text), MAX_JS_TARGETS_PER_FILE)
    if not (lookup_ids or lookup_classes or managed or events):
        return []
    return [
        {
            "path": path,
            "lookup_ids": list(lookup_ids),
            "lookup_classes": lookup_classes,
            "managed_classes": managed,
            "event_types": list(events),
        }
    ]


def _tests(path: str, text: str) -> list[dict[str, object]]:
    tree = _safe_python_ast(text)
    if tree is None:
        return []
    test_functions: list[str] = []
    fixtures: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("test_"):
                test_functions.append(node.name)
            elif _has_fixture_decorator(node):
                fixtures.append(node.name)
    routes = list(
        dict.fromkeys(match.group("path") for match in _ROUTE_DECORATOR.finditer(text))
    )
    routes.extend(re.findall(r"""\.(?:get|post)\(\s*['"](/[\w./-]*)['"]""", text))
    referenced_modules = [
        module for module in _python_imports(text) if module.startswith("app")
    ]
    if not (test_functions or fixtures):
        return []
    return [
        {
            "path": path,
            "test_functions": test_functions[:MAX_SYMBOLS_PER_FILE],
            "fixtures": fixtures[:MAX_SYMBOLS_PER_FILE],
            "referenced_routes": list(dict.fromkeys(routes))[:MAX_SYMBOLS_PER_FILE],
            "referenced_modules": list(dict.fromkeys(referenced_modules))[
                :MAX_IMPORTS_PER_FILE
            ],
        }
    ]


def _chunks(
    path: str,
    text: str,
    language: str,
    role: str,
    symbols: Sequence[dict[str, object]],
    selectors: Sequence[dict[str, object]],
    js: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    lines = text.splitlines()
    chunks: list[dict[str, object]] = []

    def add(start: int, end: int, kind: str, descriptor: str) -> None:
        start = max(1, start)
        end = min(len(lines), max(start, end))
        body = "\n".join(lines[start - 1 : end])
        if not body.strip():
            return
        chunks.append(
            {
                "path": path,
                "start_line": start,
                "end_line": end,
                "kind": kind,
                "hash": content_hash(body),
                "descriptor": descriptor[:MAX_SUMMARY_CHARS],
                "char_count": len(body),
            }
        )

    if language == "python":
        for symbol in symbols:
            if symbol["kind"] in {"class", "function", "test_function", "route_handler"}:
                start = int(symbol.get("line", 0) or 0)
                end = int(symbol.get("end_line", start) or start)
                kind = "test" if symbol["kind"] == "test_function" else (
                    "route" if symbol["kind"] == "route_handler" else str(symbol["kind"])
                )
                add(start, min(end, start + MAX_CHUNK_LINES), kind, f"{symbol['kind']} {symbol['name']}")
    elif language == "css":
        for selector in selectors:
            line = int(selector.get("line", 0) or 0)
            add(line, line, "css_block", f"selector {selector['selector']}")
    elif language in {"javascript", "typescript"}:
        for symbol in symbols:
            line = int(symbol.get("line", 0) or 0)
            add(line, line + MAX_CHUNK_LINES, "js_function", f"js {symbol['name']}")
    if not chunks:
        add(1, min(len(lines), MAX_CHUNK_LINES), "generic", f"{role} {PurePosixPath(path).name}")
    return chunks


def _relationships(
    path: str,
    role: str,
    language: str,
    text: str,
    routes: Sequence[dict[str, object]],
    js: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for route in routes:
        if route.get("handler"):
            edges.append(
                {
                    "kind": "route_to_handler",
                    "source": f"{path}:{route['route']}",
                    "target": f"{path}:{route['handler']}",
                }
            )
        for asset in route.get("referenced_assets", []):
            edges.append(
                {
                    "kind": "page_to_asset",
                    "source": f"{path}:{route['route']}",
                    "target": str(asset),
                }
            )
    if role in {ROLE_WEB_PAGE, ROLE_WEB_ROUTE}:
        for asset in _referenced_assets(text):
            edges.append(
                {"kind": "page_to_asset", "source": path, "target": str(asset)}
            )
    for record in js:
        for class_name in [
            *record.get("lookup_classes", []),
            *record.get("managed_classes", []),
        ]:
            edges.append(
                {"kind": "js_to_class", "source": path, "target": f".{class_name}"}
            )
        for dom_id in record.get("lookup_ids", []):
            edges.append(
                {"kind": "js_to_dom_id", "source": path, "target": f"#{dom_id}"}
            )
    return edges


def _referenced_assets(text: str) -> tuple[str, ...]:
    refs = [
        _normalize_asset_ref(match.group("ref"))
        for match in _ASSET_REFERENCE.finditer(text)
    ]
    return _bounded_unique([ref for ref in refs if ref], MAX_JS_TARGETS_PER_FILE)


def _normalize_asset_ref(ref: str) -> str:
    ref = ref.strip()
    index = ref.casefold().find("static/")
    if index == -1:
        return ref
    return ref[index:]


def _responsibility_summary(path: str, role: str, text: str, language: str) -> str:
    name = PurePosixPath(path).name
    if role == ROLE_APPROVAL_SENSITIVE:
        return (
            "Approval-sensitive source recorded as metadata only; excluded from "
            "model context, symbols, and chunks."
        )
    if role == ROLE_TEST:
        subject = _test_subject(text)
        base = f"Tests {subject}." if subject else f"Test module {name}."
        return base[:MAX_SUMMARY_CHARS]
    if role == ROLE_WEB_ROUTE:
        routes = [match.group("path") for match in _ROUTE_DECORATOR.finditer(text)]
        joined = ", ".join(dict.fromkeys(routes))[:120]
        return (f"Defines web route handlers ({joined})." if joined
                else "Defines web route handlers.")[:MAX_SUMMARY_CHARS]
    if role == ROLE_WEB_PAGE:
        return f"Builds rendered web page markup ({name})."[:MAX_SUMMARY_CHARS]
    if role == ROLE_STATIC_CSS:
        return (
            f"Static stylesheet defining UI and layout selectors ({name})."
        )[:MAX_SUMMARY_CHARS]
    if role == ROLE_STATIC_JS:
        return (
            f"Static client script managing DOM targets and behavior ({name})."
        )[:MAX_SUMMARY_CHARS]
    if role == ROLE_FREESTYLE_MACHINERY:
        return f"Freestyle pipeline machinery ({name})."[:MAX_SUMMARY_CHARS]
    summary = _module_docstring_summary(text, language)
    if summary:
        return summary[:MAX_SUMMARY_CHARS]
    return f"{language.capitalize()} source module ({name})."[:MAX_SUMMARY_CHARS]


def _test_subject(text: str) -> str | None:
    for module in _python_imports(text):
        if module.startswith("app."):
            return module
    return None


def _module_docstring_summary(text: str, language: str) -> str | None:
    if language != "python":
        return None
    tree = _safe_python_ast(text)
    if tree is None:
        return None
    doc = ast.get_docstring(tree)
    if not doc:
        return None
    first = doc.strip().splitlines()[0].strip()
    return first or None


def _safe_python_ast(text: str) -> ast.Module | None:
    try:
        return ast.parse(text)
    except (SyntaxError, ValueError):
        return None


def _bounded_unique(values: Sequence[str], limit: int) -> tuple[str, ...]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not value:
            continue
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
        if len(ordered) >= limit:
            break
    return tuple(ordered)
