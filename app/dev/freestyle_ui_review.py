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
_CLASSLIST_PATTERN = re.compile(
    r"""classList\.\w+\(\s*["']([^"']+)["']""", re.IGNORECASE
)
_QUERY_SELECTOR_PATTERN = re.compile(
    r"""querySelector(?:All)?\(\s*["']([^"']+)["']""", re.IGNORECASE
)
_SELECTOR_SIMPLE_TOKEN = re.compile(r"([.#])([A-Za-z_][\w-]*)")

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
class UiReview:
    warnings: tuple[str, ...] = ()
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _MarkupFacts:
    classes: frozenset[str]
    ids: frozenset[str]
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
    # Classes the script *applies at runtime* (classList.add/remove/toggle).
    # These are not expected to exist in the static markup, so they never
    # trigger a missing-target warning, but they do count as present for CSS
    # orphan detection because a rule may legitimately style a runtime state.
    managed_classes: frozenset[str]


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
    full_js = [s.text for s in js_sources] + inline_js
    introduced_js = [s.introduced for s in js_sources] + [
        body for s in markup_sources for body in _INLINE_SCRIPT_PATTERN.findall(s.introduced)
    ]

    present = _merge_markup(_markup_facts(text) for text in full_markup)
    introduced = _merge_markup(_markup_facts(text) for text in introduced_markup)
    css_full = _css_facts(full_css)
    css_introduced = _css_facts(introduced_css)
    js_full = _merge_js(_js_facts(text) for text in full_js)
    js_introduced = _merge_js(_js_facts(text) for text in introduced_js)

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
        "grid_layout_classes": _sample(
            sorted(css_full.layout_class_names), MAX_SUMMARY_ITEMS
        ),
    }

    # Classes referenced as real element classes anywhere (markup, renders, or
    # JS) are "present" for orphan-CSS purposes, so a runtime state class a rule
    # legitimately styles is not mistaken for a dangling selector.
    present_css_classes = (
        present.classes | js_full.lookup_classes | js_full.managed_classes
    )

    warnings.extend(
        _check_unstyled_selectors(introduced, css_full, has_changed_css, summary)
    )
    warnings.extend(
        _check_orphan_css(
            css_introduced, present_css_classes, present.ids, has_markup_view, summary
        )
    )
    warnings.extend(
        _check_grid_containers(present, css_full, has_css_view, summary)
    )
    warnings.extend(
        _check_js_targets(js_introduced, present, has_markup_view, summary)
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
) -> list[str]:
    """Class/id names the change introduced into the page that no CSS styles.

    Gated on the same job also changing CSS: a markup-only change usually reuses
    existing styles, so comparing it against the effective stylesheet would only
    add noise. When the job is doing styling work, a freshly introduced but
    unstyled hook is a real smell.
    """
    if not has_changed_css:
        return []
    unstyled_classes = sorted(introduced.classes - css.class_selectors)
    unstyled_ids = sorted(introduced.ids - css.id_selectors)
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
    present_classes: frozenset[str],
    present_ids: frozenset[str],
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
    orphan_classes = sorted(introduced_css.class_selectors - present_classes)
    orphan_ids = sorted(introduced_css.id_selectors - present_ids)
    summary["orphan_css_classes"] = _sample(orphan_classes, MAX_SAMPLE)
    summary["orphan_css_ids"] = _sample(orphan_ids, MAX_SAMPLE)
    selectors = [f".{name}" for name in orphan_classes] + [
        f"#{name}" for name in orphan_ids
    ]
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
    """Element ids/classes the changed JS reads that the page never renders."""
    if not has_markup_view:
        return []
    missing_ids = sorted(js.ids - present.ids)
    missing_classes = sorted(js.lookup_classes - present.classes)
    summary["js_missing_ids"] = _sample(missing_ids, MAX_SAMPLE)
    summary["js_missing_classes"] = _sample(missing_classes, MAX_SAMPLE)
    selectors = [f"#{name}" for name in missing_ids] + [
        f".{name}" for name in missing_classes
    ]
    if not selectors:
        return []
    return [
        f"{UI_WARNING_JS_MISSING_TARGETS}: the changed JavaScript references "
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
    referenced_basenames = {
        _basename(ref) for ref in (*present.script_srcs, *present.link_hrefs)
    }
    present_basenames = {_basename(asset) for asset in present_static_assets}

    missing_assets = sorted(
        {
            src
            for src in present.script_srcs
            if _is_local_reference(src)
            and _suffix(src) in _JS_SUFFIXES
            and not _reference_is_present(src, present_static_assets, present_basenames)
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
        if _basename(path) not in referenced_basenames
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
    for group in _CLASS_ATTR_PATTERN.findall(body):
        classes.update(token for token in group.split() if _valid_name(token))
    ids = {token for token in _ID_ATTR_PATTERN.findall(body) if _valid_name(token)}
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
        script_srcs=script_srcs,
        link_hrefs=link_hrefs,
        list_item_day_cells=list_item_day_cells,
        distinct_weekdays=distinct_weekdays,
    )


def _merge_markup(items: Iterable[_MarkupFacts]) -> _MarkupFacts:
    classes: set[str] = set()
    ids: set[str] = set()
    script_srcs: list[str] = []
    link_hrefs: list[str] = []
    list_item_day_cells = 0
    distinct_weekdays = 0
    for facts in items:
        classes |= facts.classes
        ids |= facts.ids
        script_srcs.extend(facts.script_srcs)
        link_hrefs.extend(facts.link_hrefs)
        list_item_day_cells += facts.list_item_day_cells
        distinct_weekdays = max(distinct_weekdays, facts.distinct_weekdays)
    return _MarkupFacts(
        classes=frozenset(classes),
        ids=frozenset(ids),
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
    ids = {name for name in _GET_BY_ID_PATTERN.findall(js_text) if _valid_name(name)}
    lookup_classes = {
        name for name in _GET_BY_CLASS_PATTERN.findall(js_text) if _valid_name(name)
    }
    managed_classes = {
        name for name in _CLASSLIST_PATTERN.findall(js_text) if _valid_name(name)
    }
    for selector in _QUERY_SELECTOR_PATTERN.findall(js_text):
        for kind, name in _SELECTOR_SIMPLE_TOKEN.findall(selector):
            if kind == "#":
                ids.add(name)
            else:
                lookup_classes.add(name)
    return _JsFacts(
        ids=frozenset(ids),
        lookup_classes=frozenset(lookup_classes),
        managed_classes=frozenset(managed_classes),
    )


def _merge_js(items: Iterable[_JsFacts]) -> _JsFacts:
    ids: set[str] = set()
    lookup_classes: set[str] = set()
    managed_classes: set[str] = set()
    for facts in items:
        ids |= facts.ids
        lookup_classes |= facts.lookup_classes
        managed_classes |= facts.managed_classes
    return _JsFacts(
        ids=frozenset(ids),
        lookup_classes=frozenset(lookup_classes),
        managed_classes=frozenset(managed_classes),
    )


# --------------------------------------------------------------------------- #
# Small helpers.
# --------------------------------------------------------------------------- #


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


def _suffix(path: str) -> str:
    name = _basename(path)
    dot = name.rfind(".")
    return name[dot:].casefold() if dot > 0 else ""


def _basename(path: str) -> str:
    cleaned = path.replace("\\", "/").split("?", 1)[0].split("#", 1)[0]
    return cleaned.rstrip("/").rsplit("/", 1)[-1]


def _is_local_reference(reference: str) -> bool:
    lowered = reference.strip().casefold()
    if not lowered:
        return False
    return not lowered.startswith(("http://", "https://", "//", "data:", "mailto:"))


def _reference_is_present(
    reference: str,
    present_static_assets: frozenset[str],
    present_basenames: set[str],
) -> bool:
    cleaned = reference.replace("\\", "/").split("?", 1)[0].split("#", 1)[0]
    stripped = cleaned.lstrip("/")
    candidates = {stripped, f"app/{stripped}"}
    if stripped.startswith("static/"):
        candidates.add(f"app/{stripped}")
    if candidates & present_static_assets:
        return True
    return _basename(reference) in present_basenames


def _render_selectors(selectors: Sequence[str]) -> str:
    shown = [s[:MAX_SELECTOR_CHARS] for s in selectors[:MAX_SAMPLE]]
    rendered = ", ".join(shown)
    extra = len(selectors) - len(shown)
    if extra > 0:
        rendered += f", and {extra} more"
    return rendered


def _sample(values: Sequence[str], limit: int) -> list[str]:
    return [str(value)[:MAX_SELECTOR_CHARS] for value in values[:limit]]
