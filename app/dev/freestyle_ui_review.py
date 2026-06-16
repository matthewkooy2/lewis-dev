"""Advisory, deterministic UI-quality heuristics for freestyle web review.

Pure text analysis. Given the web-relevant files a freestyle job changed
(rendered HTML, page-builder Python, CSS, and static JS) plus any best-effort
locally rendered pages, this module flags obvious rendered-layout and
selector-consistency failures so a refinement can repair them. It performs no
file, process, model, or network I/O and imports no other application module,
so the review slice can depend on it safely.

Every finding is advisory. Nothing here blocks edit-bundle validation, edit
application, review, or approval. The goal is to make a broken calendar-style
layout visible to the next refinement, not to gate any pipeline stage.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
import re
from typing import Any
from urllib.parse import unquote, urlsplit

# Stable warning codes. They are prefixes of the human-readable warning strings
# so the handoff, web UI, and refinement memory can surface and group them, and
# so regression tests can assert a specific disconnect was detected.
UI_WARNING_UNSTYLED_SELECTORS = "UI_REVIEW_UNSTYLED_SELECTORS"
UI_WARNING_ORPHAN_CSS_SELECTORS = "UI_REVIEW_ORPHAN_CSS_SELECTORS"
UI_WARNING_GRID_CONTAINER_WITHOUT_LAYOUT = "UI_REVIEW_GRID_CONTAINER_WITHOUT_LAYOUT"
UI_WARNING_JS_MISSING_TARGETS = "UI_REVIEW_JS_MISSING_TARGETS"
UI_WARNING_BROKEN_CALENDAR_LAYOUT = "UI_REVIEW_BROKEN_CALENDAR_LAYOUT"
UI_WARNING_MISSING_JS_ASSET = "UI_REVIEW_MISSING_JS_ASSET"
UI_WARNING_UNREFERENCED_STATIC_ASSET = "UI_REVIEW_UNREFERENCED_STATIC_ASSET"

# Bounds so the review artifact, handoff, and refinement prompt stay small.
MAX_SAMPLE = 8
MAX_WARNINGS = 24
MAX_SUMMARY_ITEMS = 40
MAX_SELECTOR_CHARS = 80

_MARKUP_SUFFIXES = frozenset({".html", ".htm", ".jinja", ".jinja2", ".j2", ".tmpl"})
_CSS_SUFFIXES = frozenset({".css"})
_JS_SUFFIXES = frozenset({".js", ".mjs"})

KIND_HTML = "html"
KIND_CSS = "css"
KIND_JS = "js"
KIND_PYTHON_MARKUP = "python_markup"

# A ``.py`` file is treated as markup-bearing only when its text contains real
# HTML structure, so a freestyle page-builder module (which emits HTML strings)
# is inspected while ordinary Python is ignored.
_PYTHON_MARKUP_SIGNAL = re.compile(
    r"<(?:div|section|span|ul|ol|li|table|tr|td|th|script|link|article|nav"
    r"|header|main|form|button|h[1-6])\b",
    re.IGNORECASE,
)

_CLASS_ATTR_PATTERN = re.compile(r"""class\s*=\s*["']([^"']*)["']""", re.IGNORECASE)
_ID_ATTR_PATTERN = re.compile(r"""id\s*=\s*["']([^"']+)["']""", re.IGNORECASE)
_START_TAG_PATTERN = re.compile(
    r"""<(?P<tag>[A-Za-z][\w:-]*)(?P<attrs>[^<>]*)>""", re.IGNORECASE | re.DOTALL
)
_ATTR_PATTERN = re.compile(
    r"""([A-Za-z_:][\w:.-]*)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s"'=<>`]+)))?""",
    re.DOTALL,
)
_SCRIPT_SRC_PATTERN = re.compile(
    r"""<script\b[^>]*?\bsrc\s*=\s*["']([^"']+)["']""", re.IGNORECASE | re.DOTALL
)
_LINK_HREF_PATTERN = re.compile(
    r"""<link\b[^>]*?\bhref\s*=\s*["']([^"']+)["']""", re.IGNORECASE | re.DOTALL
)
_INLINE_SCRIPT_PATTERN = re.compile(
    r"<script\b(?![^>]*\bsrc\s*=)[^>]*>(.*?)</script>", re.IGNORECASE | re.DOTALL
)
_HTML_COMMENT_PATTERN = re.compile(r"<!--.*?-->", re.DOTALL)
_LIST_ITEM_PATTERN = re.compile(r"<li\b[^>]*>(.*?)</li>", re.IGNORECASE | re.DOTALL)
_TAG_PATTERN = re.compile(r"<[^>]+>")
_DAY_NUMBER_TEXT = re.compile(r"^\s*(?:[1-9]|[12][0-9]|3[01])\s*$")

_CSS_COMMENT_PATTERN = re.compile(r"/\*.*?\*/", re.DOTALL)
_CSS_RULE_PATTERN = re.compile(r"([^{}]+)\{([^{}]*)\}", re.DOTALL)
_CSS_SELECTOR_TOKEN = re.compile(r"([.#])([A-Za-z_][\w-]*)")
_CSS_LAYOUT_PATTERN = re.compile(
    r"display\s*:\s*(?:inline-)?(?:grid|flex)\b|grid-template", re.IGNORECASE
)

_GET_BY_ID_PATTERN = re.compile(
    r"""getElementById\(\s*["']([^"']+)["']""", re.IGNORECASE
)
_GET_BY_CLASS_PATTERN = re.compile(
    r"""getElementsByClassName\(\s*["']([^"']+)["']""", re.IGNORECASE
)
_QUERY_SELECTOR_PATTERN = re.compile(
    r"""querySelector(?:All)?\(\s*["']([^"']+)["']""", re.IGNORECASE
)
_SELECTOR_SIMPLE_TOKEN = re.compile(r"([.#])([A-Za-z_][\w-]*)")
_SELECTOR_NAME_PATTERN = re.compile(r"[A-Za-z_][\w-]*")
_JS_IDENTIFIER = r"[A-Za-z_$][\w$]*"
_JS_LITERAL_PATTERN = re.compile(
    r"""(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.DOTALL,
)
_JS_STRING_CONST_PATTERN = re.compile(
    rf"""(?:const|let|var)\s+(?P<name>{_JS_IDENTIFIER})\s*=\s*"""
    r"""(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.DOTALL,
)
_CLASSNAME_LITERAL_ASSIGN_PATTERN = re.compile(
    r"""\.className\s*=\s*(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.DOTALL,
)
_CLASSNAME_IDENT_ASSIGN_PATTERN = re.compile(
    rf"""\.className\s*=\s*(?P<name>{_JS_IDENTIFIER})\b"""
)
_ID_LITERAL_ASSIGN_PATTERN = re.compile(
    r"""\.id\s*=\s*(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.DOTALL,
)
_ID_IDENT_ASSIGN_PATTERN = re.compile(rf"""\.id\s*=\s*(?P<name>{_JS_IDENTIFIER})\b""")
_CLASSLIST_CALL_PATTERN = re.compile(
    r"""classList\.(?P<method>add|toggle|replace)\(\s*(?P<args>[^)]*)\)""",
    re.IGNORECASE | re.DOTALL,
)
_SET_CLASS_ATTR_PATTERN = re.compile(
    r"""setAttribute\(\s*["']class["']\s*,\s*(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_SET_ID_ATTR_PATTERN = re.compile(
    r"""setAttribute\(\s*["']id["']\s*,\s*(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_INNER_HTML_PATTERN = re.compile(
    r"""\.(?:innerHTML|outerHTML)\s*=\s*(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_INSERT_HTML_PATTERN = re.compile(
    r"""insertAdjacentHTML\(\s*(?P<where>["'`])(?:\\.|(?!(?P=where)).)*(?P=where)\s*,\s*(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.IGNORECASE | re.DOTALL,
)
_OBJECT_ASSIGN_PATTERN = re.compile(
    r"""Object\.assign\(\s*[^,]+,\s*\{(?P<body>[^{}]*)\}\s*\)""",
    re.DOTALL,
)
_OBJECT_DOM_PROP_PATTERN = re.compile(
    r"""(?P<prop>className|id)\s*:\s*(?P<quote>["'`])(?P<value>(?:\\.|(?!(?P=quote)).)*)(?P=quote)""",
    re.DOTALL,
)
_JSON_PARSE_ID_PATTERN = re.compile(
    r"""JSON\.parse\(\s*(?:document\.)?getElementById\(\s*["']([^"']+)["']\s*\)\.textContent""",
    re.IGNORECASE,
)

# Weekday labels: short and long forms. A real weekday header repeats most of
# them, so the broken-layout heuristic requires several distinct labels rather
# than a single incidental "Sun" or "Mon" in prose.
_WEEKDAY_PATTERN = re.compile(
    r"\b(mon(?:day)?|tue(?:s|sday)?|wed(?:nesday)?|thu(?:r|rs|rsday)?"
    r"|fri(?:day)?|sat(?:urday)?|sun(?:day)?)\b",
    re.IGNORECASE,
)
_WEEKDAY_CANONICAL = {
    "mon": "mon", "monday": "mon",
    "tue": "tue", "tues": "tue", "tuesday": "tue",
    "wed": "wed", "wednesday": "wed",
    "thu": "thu", "thur": "thu", "thurs": "thu", "thursday": "thu",
    "fri": "fri", "friday": "fri",
    "sat": "sat", "saturday": "sat",
    "sun": "sun", "sunday": "sun",
}
_MIN_DISTINCT_WEEKDAYS = 4
_MIN_LIST_ITEM_DAY_CELLS = 7

# Class names that denote a multi-cell layout container and are therefore
# expected to carry a CSS grid/flex rule. ``"grid"`` as a substring already
# implies layout intent; the rest name calendar surfaces specifically.
_GRID_CONTAINER_NAMES = frozenset(
    {
        "calendar",
        "calendar-days",
        "calendar-grid",
        "calendar-month",
        "calendar-week",
        "calendar-weeks",
        "weekdays",
        "weekday-row",
        "weekday-header",
        "month",
        "month-grid",
        "agenda",
        "agenda-grid",
        "days",
        "day-grid",
    }
)


@dataclass(frozen=True)
class UiSource:
    """One web-relevant file the job changed.

    ``text`` is the file's current content. ``added_text`` is the subset of
    lines the change introduced (used to focus warnings on what the job added);
    when ``None`` the whole file is treated as introduced, which keeps unit
    fixtures simple.
    """

    path: str
    kind: str
    text: str
    status: str = "modified"
    added_text: str | None = None
    # ``False`` for files supplied only as context (e.g. the existing
    # stylesheet), so they enlarge the effective corpus without being treated as
    # something this job introduced.
    changed: bool = True

    @property
    def introduced(self) -> str:
        if not self.changed:
            return ""
        return self.text if self.added_text is None else self.added_text


@dataclass(frozen=True)
class RenderedPage:
    """HTML captured from a locally rendered route (best-effort, additive)."""

    route: str
    html: str


@dataclass(frozen=True)
class ReferencedJsAsset:
    """Static JavaScript asset text loaded because rendered markup references it."""

    path: str
    src: str
    text: str


@dataclass(frozen=True)
class UiReview:
    warnings: tuple[str, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _MarkupFacts:
    classes: frozenset[str]
    ids: frozenset[str]
    data_container_classes: frozenset[str]
    data_container_ids: frozenset[str]
    script_srcs: tuple[str, ...]
    link_hrefs: tuple[str, ...]
    list_item_day_cells: int
    distinct_weekdays: int


@dataclass(frozen=True)
class _CssFacts:
    class_selectors: frozenset[str]
    id_selectors: frozenset[str]
    layout_class_names: frozenset[str]
    layout_id_names: frozenset[str]


@dataclass(frozen=True)
class _JsFacts:
    # Ids/classes the script *looks up* (getElementById, querySelector,
    # getElementsByClassName) and therefore expects the page to already render.
    ids: frozenset[str]
    lookup_classes: frozenset[str]
    # Ids/classes the script emits at runtime. These are not expected to exist
    # in the static markup, so they never clear a missing-target warning, but
    # they do count as present for CSS orphan detection because a rule may
    # legitimately style runtime DOM.
    emitted_ids: frozenset[str]
    emitted_classes: frozenset[str]
    data_ids: frozenset[str]


def classify_web_source(path: str, text: str) -> str | None:
    """Return the UI kind of one path/content pair, or ``None`` if not web UI."""
    suffix = _suffix(path)
    if suffix in _MARKUP_SUFFIXES:
        return KIND_HTML
    if suffix in _CSS_SUFFIXES:
        return KIND_CSS
    if suffix in _JS_SUFFIXES:
        return KIND_JS
    if suffix == ".py" and _PYTHON_MARKUP_SIGNAL.search(text):
        return KIND_PYTHON_MARKUP
    return None


def is_static_asset_path(path: str) -> bool:
    """Whether a repo-relative path lives under a ``static`` directory."""
    return "static" in path.replace("\\", "/").split("/")


def review_ui(
    sources: Sequence[UiSource],
    *,
    rendered_pages: Sequence[RenderedPage] = (),
    present_static_assets: frozenset[str] = frozenset(),
    referenced_js_assets: Sequence[ReferencedJsAsset] = (),
) -> UiReview:
    """Run every advisory UI heuristic over the changed web files.

    The "present" markup and CSS corpora are the union of every changed web
    file's full content plus any rendered pages, so a selector satisfied
    elsewhere is never flagged. The "introduced" corpus is what the change
    added, so warnings stay focused on this job rather than the whole codebase.
    Rendered pages only ever enlarge the present set, so they can resolve a
    disconnect but never invent one.
    """
    markup_sources = [s for s in sources if s.kind in (KIND_HTML, KIND_PYTHON_MARKUP)]
    css_sources = [s for s in sources if s.kind == KIND_CSS]
    js_sources = [s for s in sources if s.kind == KIND_JS]

    full_markup = [s.text for s in markup_sources] + [p.html for p in rendered_pages]
    introduced_markup = [s.introduced for s in markup_sources]
    full_css = "\n".join(s.text for s in css_sources)
    introduced_css = "\n".join(s.introduced for s in css_sources)
    # Inline <script> bodies count as JS too, so a template that wires missing
    # ids is inspected alongside standalone .js assets.
    inline_js = [
        body for text in full_markup for body in _INLINE_SCRIPT_PATTERN.findall(text)
    ]

    present = _merge_markup(_markup_facts(text) for text in full_markup)
    introduced = _merge_markup(_markup_facts(text) for text in introduced_markup)
    css_full = _css_facts(full_css)
    css_introduced = _css_facts(introduced_css)
    referenced_assets = _referenced_js_assets(
        js_sources, present.script_srcs, referenced_js_assets
    )
    inline_assets = tuple(
        ReferencedJsAsset(path=f"inline-script-{index}", src="inline", text=body)
        for index, body in enumerate(inline_js, start=1)
    )
    js_asset_facts = tuple(
        (asset, _js_facts(asset.text)) for asset in (*referenced_assets, *inline_assets)
    )
    js_full = _merge_js(facts for _asset, facts in js_asset_facts)

    has_changed_css = any(s.changed for s in css_sources)
    has_css_view = bool(full_css.strip())
    has_markup_view = bool(full_markup)

    warnings: list[str] = []
    summary: dict[str, Any] = {
        "web_source_count": len(sources),
        "markup_source_count": len(markup_sources),
        "css_source_count": len(css_sources),
        "js_source_count": len(js_sources),
        "rendered_page_count": len(rendered_pages),
        "present_classes": _sample(sorted(present.classes), MAX_SUMMARY_ITEMS),
        "present_ids": _sample(sorted(present.ids), MAX_SUMMARY_ITEMS),
        "server_rendered_selectors": {
            "classes": _sample(sorted(present.classes), MAX_SUMMARY_ITEMS),
            "ids": _sample(sorted(present.ids), MAX_SUMMARY_ITEMS),
        },
        "referenced_js_assets": _sample(
            [asset.path for asset in referenced_assets], MAX_SUMMARY_ITEMS
        ),
        "js_emitted_selectors_by_asset": _js_asset_selector_summary(
            js_asset_facts, emitted=True
        ),
        "js_target_selectors_by_asset": _js_asset_selector_summary(
            js_asset_facts, emitted=False
        ),
        "grid_layout_classes": _sample(
            sorted(css_full.layout_class_names), MAX_SUMMARY_ITEMS
        ),
    }

    ignored_data_classes = introduced.data_container_classes
    ignored_data_ids = introduced.data_container_ids | (introduced.ids & js_full.data_ids)
    summary["ignored_data_containers"] = _sample(
        [f".{name}" for name in sorted(ignored_data_classes)]
        + [f"#{name}" for name in sorted(ignored_data_ids)],
        MAX_SAMPLE,
    )

    warnings.extend(
        _check_unstyled_selectors(
            introduced,
            css_full,
            has_changed_css,
            summary,
            ignored_data_classes=ignored_data_classes,
            ignored_data_ids=ignored_data_ids,
        )
    )
    warnings.extend(
        _check_orphan_css(
            css_introduced,
            present.classes,
            present.ids,
            js_full.emitted_classes,
            js_full.emitted_ids,
            has_markup_view,
            summary,
        )
    )
    warnings.extend(
        _check_grid_containers(present, css_full, has_css_view, summary)
    )
    warnings.extend(
        _check_js_targets(js_full, present, has_markup_view, summary)
    )
    warnings.extend(
        _check_broken_calendar(present, css_full, has_css_view, summary)
    )
    warnings.extend(
        _check_assets(sources, present, present_static_assets, summary)
    )

    return UiReview(warnings=tuple(warnings[:MAX_WARNINGS]), summary=summary)


# --------------------------------------------------------------------------- #
# Individual heuristics.
# --------------------------------------------------------------------------- #


def _check_unstyled_selectors(
    introduced: _MarkupFacts,
    css: _CssFacts,
    has_changed_css: bool,
    summary: dict[str, Any],
    *,
    ignored_data_classes: frozenset[str] = frozenset(),
    ignored_data_ids: frozenset[str] = frozenset(),
) -> list[str]:
    """Class/id names the change introduced into the page that no CSS styles.

    Gated on the same job also changing CSS: a markup-only change usually reuses
    existing styles, so comparing it against the effective stylesheet would only
    add noise. When the job is doing styling work, a freshly introduced but
    unstyled hook is a real smell.
    """
    if not has_changed_css:
        return []
    unstyled_classes = sorted(
        introduced.classes - css.class_selectors - ignored_data_classes
    )
    unstyled_ids = sorted(introduced.ids - css.id_selectors - ignored_data_ids)
    summary["unstyled_classes"] = _sample(unstyled_classes, MAX_SAMPLE)
    summary["unstyled_ids"] = _sample(unstyled_ids, MAX_SAMPLE)
    selectors = [f".{name}" for name in unstyled_classes] + [
        f"#{name}" for name in unstyled_ids
    ]
    if not selectors:
        return []
    return [
        f"{UI_WARNING_UNSTYLED_SELECTORS}: the changed page uses "
        f"{_render_selectors(selectors)} but no CSS rule styles "
        f"{'them' if len(selectors) > 1 else 'it'}; the rendered layout may fall "
        "back to default flow."
    ]


def _check_orphan_css(
    introduced_css: _CssFacts,
    server_classes: frozenset[str],
    server_ids: frozenset[str],
    runtime_classes: frozenset[str],
    runtime_ids: frozenset[str],
    has_markup_view: bool,
    summary: dict[str, Any],
) -> list[str]:
    """CSS selectors the change added that match nothing in the rendered page.

    Skipped when there is no markup view at all (no changed markup and no
    rendered page), because then the present set is unknown and every selector
    would look orphaned.
    """
    if not has_markup_view:
        return []
    runtime_css_classes = sorted(introduced_css.class_selectors & runtime_classes)
    runtime_css_ids = sorted(introduced_css.id_selectors & runtime_ids)
    runtime_selectors = [f".{name}" for name in runtime_css_classes] + [
        f"#{name}" for name in runtime_css_ids
    ]
    summary["runtime_emitted_css_selectors"] = _sample(runtime_selectors, MAX_SAMPLE)
    orphan_classes = sorted(
        introduced_css.class_selectors - server_classes - runtime_classes
    )
    orphan_ids = sorted(introduced_css.id_selectors - server_ids - runtime_ids)
    summary["orphan_css_classes"] = _sample(orphan_classes, MAX_SAMPLE)
    summary["orphan_css_ids"] = _sample(orphan_ids, MAX_SAMPLE)
    selectors = [f".{name}" for name in orphan_classes] + [
        f"#{name}" for name in orphan_ids
    ]
    summary["true_orphan_css_selectors"] = _sample(selectors, MAX_SAMPLE)
    if not selectors:
        return []
    return [
        f"{UI_WARNING_ORPHAN_CSS_SELECTORS}: the changed CSS styles "
        f"{_render_selectors(selectors)} but no rendered element carries "
        f"{'those names' if len(selectors) > 1 else 'that name'}; the rule may "
        "target a class that was renamed or never emitted."
    ]


def _check_grid_containers(
    present: _MarkupFacts,
    css: _CssFacts,
    has_css_view: bool,
    summary: dict[str, Any],
) -> list[str]:
    """Layout-container class names with no grid/flex rule in the effective CSS."""
    if not has_css_view:
        return []
    expected = {name for name in present.classes if _is_grid_container_name(name)}
    laid_out = css.layout_class_names
    missing = sorted(expected - laid_out)
    summary["grid_containers_without_layout"] = _sample(missing, MAX_SAMPLE)
    if not missing:
        return []
    selectors = [f".{name}" for name in missing]
    return [
        f"{UI_WARNING_GRID_CONTAINER_WITHOUT_LAYOUT}: {_render_selectors(selectors)} "
        "name a calendar/grid container but no CSS rule gives them "
        "display:grid or display:flex, so their children will stack in normal "
        "document flow instead of a grid."
    ]


def _check_js_targets(
    js: _JsFacts,
    present: _MarkupFacts,
    has_markup_view: bool,
    summary: dict[str, Any],
) -> list[str]:
    """Element ids/classes referenced JS reads that the page never renders."""
    if not has_markup_view:
        return []
    missing_ids = sorted(js.ids - present.ids)
    missing_classes = sorted(js.lookup_classes - present.classes)
    summary["js_missing_ids"] = _sample(missing_ids, MAX_SAMPLE)
    summary["js_missing_classes"] = _sample(missing_classes, MAX_SAMPLE)
    selectors = [f"#{name}" for name in missing_ids] + [
        f".{name}" for name in missing_classes
    ]
    summary["missing_js_targets"] = _sample(selectors, MAX_SAMPLE)
    if not selectors:
        return []
    return [
        f"{UI_WARNING_JS_MISSING_TARGETS}: referenced JavaScript looks for "
        f"{_render_selectors(selectors)} but no rendered element provides "
        f"{'them' if len(selectors) > 1 else 'it'}; the script will get null "
        "and its behavior will silently break."
    ]


def _check_broken_calendar(
    present: _MarkupFacts,
    css: _CssFacts,
    has_css_view: bool,
    summary: dict[str, Any],
) -> list[str]:
    """Structural signals that a calendar rendered as text/list instead of a grid."""
    signals: list[str] = []
    has_calendar_grid = any(
        _is_calendarish_name(name) and name in css.layout_class_names
        for name in present.classes
    )
    if (
        has_css_view
        and present.distinct_weekdays >= _MIN_DISTINCT_WEEKDAYS
        and not has_calendar_grid
    ):
        signals.append(
            "weekday labels are rendered without any grid/flex container, so they "
            "will sit inline or stack instead of forming a calendar header row"
        )
    if present.list_item_day_cells >= _MIN_LIST_ITEM_DAY_CELLS:
        signals.append(
            f"{present.list_item_day_cells} day-number cells are rendered as plain "
            "<li> list items rather than grid cells, so the days will stack "
            "vertically as a list"
        )
    summary["broken_calendar_signals"] = signals
    summary["distinct_weekday_labels"] = present.distinct_weekdays
    summary["list_item_day_cells"] = present.list_item_day_cells
    if not signals:
        return []
    return [
        f"{UI_WARNING_BROKEN_CALENDAR_LAYOUT}: likely broken calendar layout — "
        + "; ".join(signals)
        + "."
    ]


def _check_assets(
    sources: Sequence[UiSource],
    present: _MarkupFacts,
    present_static_assets: frozenset[str],
    summary: dict[str, Any],
) -> list[str]:
    """Referenced-but-missing JS assets and changed-but-unreferenced static assets."""
    warnings: list[str] = []

    missing_assets = sorted(
        {
            src
            for src in present.script_srcs
            if static_js_reference_candidates(src)
            and not _reference_is_present(src, present_static_assets)
        }
    )
    summary["missing_js_assets"] = _sample(missing_assets, MAX_SAMPLE)
    if missing_assets:
        warnings.append(
            f"{UI_WARNING_MISSING_JS_ASSET}: the page references "
            f"{_render_selectors(missing_assets)} but no matching static asset is "
            "present in the worktree; the script tag will 404 at runtime."
        )

    # A static JS asset this job created or modified that no rendered page loads.
    # Stylesheets are intentionally excluded: they are linked once in the shared
    # page scaffold rather than per page, so flagging a changed .css would fire
    # constantly. The missing-asset half above still covers broken <script src>.
    changed_assets = sorted(
        {
            source.path
            for source in sources
            if source.status != "deleted"
            and source.kind == KIND_JS
            and is_static_asset_path(source.path)
        }
    )
    unreferenced = [
        path
        for path in changed_assets
        if not _source_path_is_referenced(path, present.script_srcs)
    ]
    summary["unreferenced_static_assets"] = _sample(unreferenced, MAX_SAMPLE)
    if unreferenced:
        warnings.append(
            f"{UI_WARNING_UNREFERENCED_STATIC_ASSET}: the changed static JavaScript "
            f"asset(s) {_render_selectors(unreferenced)} are not referenced by any "
            "rendered page; add a <script> tag or remove the asset."
        )
    return warnings


# --------------------------------------------------------------------------- #
# Extraction.
# --------------------------------------------------------------------------- #


def _markup_facts(text: str) -> _MarkupFacts:
    body = _HTML_COMMENT_PATTERN.sub(" ", text)
    classes: set[str] = set()
    ids: set[str] = set()
    data_container_classes: set[str] = set()
    data_container_ids: set[str] = set()
    for match in _START_TAG_PATTERN.finditer(body):
        tag = match.group("tag").casefold()
        attrs = _html_attrs(match.group("attrs"))
        class_names = {
            token
            for token in attrs.get("class", "").split()
            if _valid_name(token)
        }
        id_name = attrs.get("id", "")
        valid_id = id_name if _valid_name(id_name) else ""
        classes.update(class_names)
        if valid_id:
            ids.add(valid_id)
        is_payload = _is_payload_tag(tag, attrs)
        if valid_id and (is_payload or _is_data_container_name(valid_id)):
            data_container_ids.add(valid_id)
        for class_name in class_names:
            if is_payload or _is_data_container_name(class_name):
                data_container_classes.add(class_name)
    script_srcs = tuple(_SCRIPT_SRC_PATTERN.findall(body))
    link_hrefs = tuple(_LINK_HREF_PATTERN.findall(body))
    list_item_day_cells = sum(
        1
        for inner in _LIST_ITEM_PATTERN.findall(body)
        if _DAY_NUMBER_TEXT.match(_TAG_PATTERN.sub("", inner))
    )
    distinct_weekdays = len(
        {
            _WEEKDAY_CANONICAL[match.lower()]
            for match in _WEEKDAY_PATTERN.findall(body)
            if match.lower() in _WEEKDAY_CANONICAL
        }
    )
    return _MarkupFacts(
        classes=frozenset(classes),
        ids=frozenset(ids),
        data_container_classes=frozenset(data_container_classes),
        data_container_ids=frozenset(data_container_ids),
        script_srcs=script_srcs,
        link_hrefs=link_hrefs,
        list_item_day_cells=list_item_day_cells,
        distinct_weekdays=distinct_weekdays,
    )


def _merge_markup(items: Iterable[_MarkupFacts]) -> _MarkupFacts:
    classes: set[str] = set()
    ids: set[str] = set()
    data_container_classes: set[str] = set()
    data_container_ids: set[str] = set()
    script_srcs: list[str] = []
    link_hrefs: list[str] = []
    list_item_day_cells = 0
    distinct_weekdays = 0
    for facts in items:
        classes |= facts.classes
        ids |= facts.ids
        data_container_classes |= facts.data_container_classes
        data_container_ids |= facts.data_container_ids
        script_srcs.extend(facts.script_srcs)
        link_hrefs.extend(facts.link_hrefs)
        list_item_day_cells += facts.list_item_day_cells
        distinct_weekdays = max(distinct_weekdays, facts.distinct_weekdays)
    return _MarkupFacts(
        classes=frozenset(classes),
        ids=frozenset(ids),
        data_container_classes=frozenset(data_container_classes),
        data_container_ids=frozenset(data_container_ids),
        script_srcs=tuple(script_srcs),
        link_hrefs=tuple(link_hrefs),
        list_item_day_cells=list_item_day_cells,
        distinct_weekdays=distinct_weekdays,
    )


def _css_facts(css_text: str) -> _CssFacts:
    body = _CSS_COMMENT_PATTERN.sub(" ", css_text)
    class_selectors: set[str] = set()
    id_selectors: set[str] = set()
    layout_class_names: set[str] = set()
    layout_id_names: set[str] = set()
    for selector_group, declarations in _CSS_RULE_PATTERN.findall(body):
        is_layout = bool(_CSS_LAYOUT_PATTERN.search(declarations))
        for kind, name in _CSS_SELECTOR_TOKEN.findall(selector_group):
            if kind == ".":
                class_selectors.add(name)
                if is_layout:
                    layout_class_names.add(name)
            else:
                id_selectors.add(name)
                if is_layout:
                    layout_id_names.add(name)
    return _CssFacts(
        class_selectors=frozenset(class_selectors),
        id_selectors=frozenset(id_selectors),
        layout_class_names=frozenset(layout_class_names),
        layout_id_names=frozenset(layout_id_names),
    )


def _js_facts(js_text: str) -> _JsFacts:
    ids = {
        name
        for name in _GET_BY_ID_PATTERN.findall(js_text)
        if _valid_selector_name(name)
    }
    lookup_classes = {
        name
        for name in _GET_BY_CLASS_PATTERN.findall(js_text)
        if _valid_selector_name(name)
    }
    for selector in _QUERY_SELECTOR_PATTERN.findall(js_text):
        for kind, name in _SELECTOR_SIMPLE_TOKEN.findall(selector):
            if kind == "#":
                if _valid_selector_name(name):
                    ids.add(name)
            else:
                if _valid_selector_name(name):
                    lookup_classes.add(name)
    emitted_classes, emitted_ids = _js_emitted_selectors(js_text)
    data_ids = {
        name
        for name in _JSON_PARSE_ID_PATTERN.findall(js_text)
        if _valid_selector_name(name)
    }
    return _JsFacts(
        ids=frozenset(ids),
        lookup_classes=frozenset(lookup_classes),
        emitted_ids=frozenset(emitted_ids),
        emitted_classes=frozenset(emitted_classes),
        data_ids=frozenset(data_ids),
    )


def _merge_js(items: Iterable[_JsFacts]) -> _JsFacts:
    ids: set[str] = set()
    lookup_classes: set[str] = set()
    emitted_ids: set[str] = set()
    emitted_classes: set[str] = set()
    data_ids: set[str] = set()
    for facts in items:
        ids |= facts.ids
        lookup_classes |= facts.lookup_classes
        emitted_ids |= facts.emitted_ids
        emitted_classes |= facts.emitted_classes
        data_ids |= facts.data_ids
    return _JsFacts(
        ids=frozenset(ids),
        lookup_classes=frozenset(lookup_classes),
        emitted_ids=frozenset(emitted_ids),
        emitted_classes=frozenset(emitted_classes),
        data_ids=frozenset(data_ids),
    )


def _js_emitted_selectors(js_text: str) -> tuple[set[str], set[str]]:
    string_constants = _js_string_constants(js_text)
    classes: set[str] = set()
    ids: set[str] = set()

    for match in _CLASSNAME_LITERAL_ASSIGN_PATTERN.finditer(js_text):
        classes.update(_class_tokens(_js_literal_value(match)))
    for match in _CLASSNAME_IDENT_ASSIGN_PATTERN.finditer(js_text):
        classes.update(_class_tokens(string_constants.get(match.group("name"), "")))
    for match in _SET_CLASS_ATTR_PATTERN.finditer(js_text):
        classes.update(_class_tokens(_js_literal_value(match)))
    for match in _ID_LITERAL_ASSIGN_PATTERN.finditer(js_text):
        value = _selector_token(_js_literal_value(match))
        if value:
            ids.add(value)
    for match in _ID_IDENT_ASSIGN_PATTERN.finditer(js_text):
        value = _selector_token(string_constants.get(match.group("name"), ""))
        if value:
            ids.add(value)
    for match in _SET_ID_ATTR_PATTERN.finditer(js_text):
        value = _selector_token(_js_literal_value(match))
        if value:
            ids.add(value)
    for match in _CLASSLIST_CALL_PATTERN.finditer(js_text):
        values = [_js_literal_value(item) for item in _JS_LITERAL_PATTERN.finditer(match.group("args"))]
        if match.group("method").casefold() == "replace" and len(values) > 1:
            values = values[1:]
        for value in values:
            token = _selector_token(value)
            if token:
                classes.add(token)
    for match in _INNER_HTML_PATTERN.finditer(js_text):
        html = _js_literal_value(match)
        facts = _markup_facts(html)
        classes.update(facts.classes)
        ids.update(facts.ids)
    for match in _INSERT_HTML_PATTERN.finditer(js_text):
        html = _js_literal_value(match)
        facts = _markup_facts(html)
        classes.update(facts.classes)
        ids.update(facts.ids)
    for match in _OBJECT_ASSIGN_PATTERN.finditer(js_text):
        for prop in _OBJECT_DOM_PROP_PATTERN.finditer(match.group("body")):
            value = _js_literal_value(prop)
            if prop.group("prop") == "className":
                classes.update(_class_tokens(value))
            else:
                token = _selector_token(value)
                if token:
                    ids.add(token)
    return classes, ids


def _js_string_constants(js_text: str) -> dict[str, str]:
    constants: dict[str, str] = {}
    for match in _JS_STRING_CONST_PATTERN.finditer(js_text):
        value = _js_literal_value(match)
        if value and len(value) <= MAX_SELECTOR_CHARS * MAX_SAMPLE:
            constants[match.group("name")] = value
    return constants


def _js_literal_value(match: re.Match[str]) -> str:
    return _decode_js_literal(match.group("value"))


def _decode_js_literal(value: str) -> str:
    return (
        value.replace(r"\/", "/")
        .replace(r"\"", '"')
        .replace(r"\'", "'")
        .replace(r"\`", "`")
        .replace(r"\n", "\n")
        .replace(r"\t", "\t")
    )


def _class_tokens(value: str) -> set[str]:
    return {
        token
        for token in value.split()
        if _valid_selector_name(token)
    }


def _selector_token(value: str) -> str:
    stripped = value.strip()
    return stripped if _valid_selector_name(stripped) else ""


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #


def referenced_static_js_srcs(markup_texts: Iterable[str]) -> tuple[str, ...]:
    """Static JS refs found in markup, keeping only same-app static candidates."""
    seen: set[str] = set()
    refs: list[str] = []
    for text in markup_texts:
        for src in _markup_facts(text).script_srcs:
            if src in seen or not static_js_reference_candidates(src):
                continue
            seen.add(src)
            refs.append(src)
    return tuple(refs)


def static_js_reference_candidates(reference: str) -> tuple[str, ...]:
    """Repo-relative static JS paths a local script reference may resolve to."""
    path = _local_reference_path(reference)
    if path is None:
        return ()
    parts = _safe_static_parts(path)
    if not parts or _suffix(parts[-1]) not in _JS_SUFFIXES:
        return ()
    if len(parts) >= 3 and parts[0] == "app" and parts[1] == "static":
        return ("/".join(parts),)
    if len(parts) >= 2 and parts[0] == "static":
        tail = "/".join(parts[1:])
        return (f"app/static/{tail}", f"static/{tail}")
    return ()


def _referenced_js_assets(
    js_sources: Sequence[UiSource],
    script_srcs: Sequence[str],
    explicit_assets: Sequence[ReferencedJsAsset],
) -> tuple[ReferencedJsAsset, ...]:
    assets: list[ReferencedJsAsset] = []
    seen: set[str] = set()
    script_src_set = set(script_srcs)
    for asset in explicit_assets:
        if (
            asset.path in seen
            or asset.src not in script_src_set
            or asset.path not in static_js_reference_candidates(asset.src)
        ):
            continue
        seen.add(asset.path)
        assets.append(asset)
    for source in js_sources:
        src = _matching_reference_for_source(source.path, script_srcs)
        if src is None or source.path in seen:
            continue
        seen.add(source.path)
        assets.append(ReferencedJsAsset(path=source.path, src=src, text=source.text))
    return tuple(assets)


def _matching_reference_for_source(path: str, script_srcs: Sequence[str]) -> str | None:
    for src in script_srcs:
        if path in static_js_reference_candidates(src):
            return src
    return None


def _source_path_is_referenced(path: str, script_srcs: Sequence[str]) -> bool:
    return _matching_reference_for_source(path, script_srcs) is not None


def _reference_is_present(
    reference: str,
    present_static_assets: frozenset[str],
) -> bool:
    return bool(set(static_js_reference_candidates(reference)) & present_static_assets)


def _js_asset_selector_summary(
    js_asset_facts: Sequence[tuple[ReferencedJsAsset, _JsFacts]],
    *,
    emitted: bool,
) -> dict[str, list[str]]:
    details: dict[str, list[str]] = {}
    for asset, facts in js_asset_facts[:MAX_SAMPLE]:
        if emitted:
            selectors = [f".{name}" for name in sorted(facts.emitted_classes)] + [
                f"#{name}" for name in sorted(facts.emitted_ids)
            ]
        else:
            selectors = [f"#{name}" for name in sorted(facts.ids)] + [
                f".{name}" for name in sorted(facts.lookup_classes)
            ]
        if selectors:
            details[asset.path] = _sample(selectors, MAX_SAMPLE)
    return details


def _html_attrs(attr_text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for name, double, single, bare in _ATTR_PATTERN.findall(attr_text):
        value = double or single or bare or ""
        attrs[name.casefold()] = value
    return attrs


def _is_payload_tag(tag: str, attrs: dict[str, str]) -> bool:
    if "hidden" in attrs:
        return True
    if tag != "script":
        return False
    content_type = attrs.get("type", "").split(";", 1)[0].strip().casefold()
    return content_type == "application/json" or content_type.endswith("+json")


def _is_data_container_name(name: str) -> bool:
    lowered = name.casefold()
    pieces = [piece for piece in re.split(r"[-_\s]+", lowered) if piece]
    return bool({"data", "json", "payload"} & set(pieces))


def _local_reference_path(reference: str) -> str | None:
    raw = reference.strip()
    lowered = raw.casefold()
    if not raw or raw.startswith("//") or "\\" in raw:
        return None
    if lowered.startswith(("data:", "mailto:", "javascript:")):
        return None
    parsed = urlsplit(raw)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    if not path or path.startswith("//") or "\\" in path:
        return None
    return path[1:] if path.startswith("/") else path


def _safe_static_parts(path: str) -> tuple[str, ...]:
    parts = tuple(part for part in path.split("/") if part)
    if not parts or len(parts) > 64:
        return ()
    if any(part in {".", ".."} for part in parts):
        return ()
    return parts


def _is_grid_container_name(name: str) -> bool:
    lowered = name.casefold()
    return "grid" in lowered or lowered in _GRID_CONTAINER_NAMES


_CALENDARISH_TOKENS = ("calendar", "grid", "week", "month", "agenda", "day")


def _is_calendarish_name(name: str) -> bool:
    """Whether a class name plausibly names a calendar surface or its grid."""
    lowered = name.casefold()
    return any(token in lowered for token in _CALENDARISH_TOKENS)


def _valid_name(token: str) -> bool:
    return bool(token) and len(token) <= MAX_SELECTOR_CHARS and "{" not in token


def _valid_selector_name(token: str) -> bool:
    return bool(_SELECTOR_NAME_PATTERN.fullmatch(token)) and len(token) <= MAX_SELECTOR_CHARS


def _suffix(path: str) -> str:
    name = _basename(path)
    dot = name.rfind(".")
    return name[dot:].casefold() if dot > 0 else ""


def _basename(path: str) -> str:
    cleaned = path.replace("\\", "/").split("?", 1)[0].split("#", 1)[0]
    return cleaned.rstrip("/").rsplit("/", 1)[-1]


def _render_selectors(selectors: Sequence[str]) -> str:
    shown = [s[:MAX_SELECTOR_CHARS] for s in selectors[:MAX_SAMPLE]]
    rendered = ", ".join(shown)
    extra = len(selectors) - len(shown)
    if extra > 0:
        rendered += f", and {extra} more"
    return rendered


def _sample(values: Sequence[str], limit: int) -> list[str]:
    return [str(value)[:MAX_SELECTOR_CHARS] for value in values[:limit]]
