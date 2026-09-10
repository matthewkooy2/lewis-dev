"""Tests for the advisory web-UI quality heuristics used by freestyle review."""

from __future__ import annotations

from pathlib import Path

from app.dev.freestyle_ui_review import (
    ReferencedJsAsset,
    RenderedPage,
    UI_WARNING_BROKEN_CALENDAR_LAYOUT,
    UI_WARNING_GRID_CONTAINER_WITHOUT_LAYOUT,
    UI_WARNING_JS_MISSING_TARGETS,
    UI_WARNING_MISSING_JS_ASSET,
    UI_WARNING_ORPHAN_CSS_SELECTORS,
    UI_WARNING_UNREFERENCED_STATIC_ASSET,
    UI_WARNING_UNSTYLED_SELECTORS,
    UiSource,
    classify_web_source,
    review_ui,
    static_js_reference_candidates,
)


# --------------------------------------------------------------------------- #
# Fixtures: a deliberately broken calendar and a healthy one.
# --------------------------------------------------------------------------- #

# Days render as plain vertical <li> text, the weekday header has no grid
# container, the CSS targets ``.calandar`` (a typo that matches no element) and
# never gives the day containers a grid/flex layout, and the JS reaches for an
# ``#calendar-grid`` id that the markup never renders.
_BROKEN_CALENDAR_HTML = """
<section class="calendar">
  <div class="weekday-row">
    <span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span>
    <span>Fri</span><span>Sat</span><span>Sun</span>
  </div>
  <ul class="calendar-days">
    <li>1</li><li>2</li><li>3</li><li>4</li><li>5</li><li>6</li><li>7</li>
    <li>8</li><li>9</li><li>10</li><li>11</li><li>12</li><li>13</li><li>14</li>
  </ul>
</section>
<script src="/static/calendar.js"></script>
"""

_BROKEN_CALENDAR_CSS = """
.calandar { color: #222; }            /* typo: never matches .calendar */
.weekday-row { font-weight: 600; }
.calendar-days { list-style: none; padding: 0; }
"""

_BROKEN_CALENDAR_JS = """
const grid = document.getElementById('calendar-grid');
grid.classList.add('ready');
"""

_HEALTHY_CALENDAR_HTML = """
<section class="calendar-grid">
  <div class="weekday-row">
    <span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span>
    <span>Fri</span><span>Sat</span><span>Sun</span>
  </div>
  <div class="calendar-day">1</div>
  <div class="calendar-day">2</div>
  <div class="calendar-day">3</div>
</section>
<script src="/static/calendar.js"></script>
"""

_HEALTHY_CALENDAR_CSS = """
.calendar-grid { display: grid; grid-template-columns: repeat(7, 1fr); gap: 4px; }
.weekday-row { display: flex; justify-content: space-between; }
.calendar-day { padding: 8px; border: 1px solid #ccc; }
"""

_HEALTHY_CALENDAR_JS = """
const grid = document.querySelector('.calendar-grid');
if (grid) { grid.classList.add('ready'); }
"""


def _broken_sources() -> list[UiSource]:
    return [
        UiSource("app/web_pages.py", "python_markup", _BROKEN_CALENDAR_HTML),
        UiSource("app/static/lewis.css", "css", _BROKEN_CALENDAR_CSS),
        UiSource("app/static/calendar.js", "js", _BROKEN_CALENDAR_JS),
    ]


def _healthy_sources() -> list[UiSource]:
    return [
        UiSource("app/web_pages.py", "python_markup", _HEALTHY_CALENDAR_HTML),
        UiSource("app/static/lewis.css", "css", _HEALTHY_CALENDAR_CSS),
        UiSource("app/static/calendar.js", "js", _HEALTHY_CALENDAR_JS),
    ]


def _codes(warnings) -> set[str]:
    return {warning.split(":", 1)[0] for warning in warnings}


# --------------------------------------------------------------------------- #
# The headline broken/healthy contrast required by the slice.
# --------------------------------------------------------------------------- #


def test_broken_calendar_produces_clear_layout_and_selector_warnings():
    review = review_ui(
        _broken_sources(),
        present_static_assets=frozenset({"app/static/calendar.js"}),
    )
    codes = _codes(review.warnings)

    assert UI_WARNING_BROKEN_CALENDAR_LAYOUT in codes
    assert UI_WARNING_ORPHAN_CSS_SELECTORS in codes
    assert UI_WARNING_JS_MISSING_TARGETS in codes
    assert UI_WARNING_GRID_CONTAINER_WITHOUT_LAYOUT in codes
    assert UI_WARNING_UNSTYLED_SELECTORS in codes

    text = "\n".join(review.warnings)
    assert "calendar-grid" in text  # the id the JS cannot find
    assert "calandar" in text  # the orphan CSS typo is surfaced verbatim


def test_healthy_calendar_produces_no_warnings():
    review = review_ui(
        _healthy_sources(),
        present_static_assets=frozenset({"app/static/calendar.js"}),
    )
    assert review.warnings == ()


# --------------------------------------------------------------------------- #
# Each disconnect in isolation.
# --------------------------------------------------------------------------- #


def test_days_rendered_as_plain_list_items_are_flagged():
    html = (
        '<section class="calendar-grid">'
        '<div class="weekday-row"><span>Mon</span><span>Tue</span>'
        "<span>Wed</span><span>Thu</span><span>Fri</span></div>"
        "<ul>" + "".join(f"<li>{day}</li>" for day in range(1, 15)) + "</ul>"
        "</section>"
    )
    css = ".calendar-grid { display: grid; } .weekday-row { display: flex; }"
    review = review_ui(
        [UiSource("page.html", "html", html), UiSource("a.css", "css", css)]
    )
    broken = [
        w for w in review.warnings if w.startswith(UI_WARNING_BROKEN_CALENDAR_LAYOUT)
    ]
    assert broken
    assert "list items" in broken[0]


def test_weekday_labels_without_grid_container_are_flagged():
    html = (
        "<div class='week'>"
        "<span>Mon</span><span>Tue</span><span>Wed</span>"
        "<span>Thu</span><span>Fri</span><span>Sat</span><span>Sun</span>"
        "</div>"
    )
    css = ".week { color: #000; }"  # no grid/flex anywhere
    review = review_ui(
        [UiSource("p.html", "html", html), UiSource("a.css", "css", css)]
    )
    broken = [
        w for w in review.warnings if w.startswith(UI_WARNING_BROKEN_CALENDAR_LAYOUT)
    ]
    assert broken
    assert "without any grid/flex container" in broken[0]


def test_css_selector_typo_not_matching_markup_is_an_orphan():
    review = review_ui(
        [
            UiSource("p.html", "html", "<div class='calendar-grid'></div>"),
            UiSource(
                "a.css",
                "css",
                ".calendar-grid { display: grid; } .clander-grid { color: red; }",
            ),
        ]
    )
    orphan = [
        w for w in review.warnings if w.startswith(UI_WARNING_ORPHAN_CSS_SELECTORS)
    ]
    assert orphan
    assert ".clander-grid" in orphan[0]
    assert ".calendar-grid" not in orphan[0]


def test_js_referencing_missing_ids_and_classes_is_flagged():
    html = "<div id='present' class='shown'></div><script src='/static/s.js'></script>"
    js = (
        "document.getElementById('missing-panel');"
        "document.querySelector('.absent-row');"
        "document.getElementById('present');"
    )
    review = review_ui(
        [UiSource("p.html", "html", html), UiSource("app/static/s.js", "js", js)]
    )
    missing = [
        w for w in review.warnings if w.startswith(UI_WARNING_JS_MISSING_TARGETS)
    ]
    assert missing
    assert "#missing-panel" in missing[0]
    assert ".absent-row" in missing[0]
    assert "#present" not in missing[0]


def test_grid_container_without_layout_rule_is_flagged():
    review = review_ui(
        [
            UiSource("p.html", "html", "<div class='calendar-grid'></div>"),
            UiSource("a.css", "css", ".calendar-grid { color: navy; }"),
        ]
    )
    grid = [
        w
        for w in review.warnings
        if w.startswith(UI_WARNING_GRID_CONTAINER_WITHOUT_LAYOUT)
    ]
    assert grid
    assert ".calendar-grid" in grid[0]


def test_referenced_but_missing_js_asset_is_flagged():
    review = review_ui(
        [UiSource("p.html", "html", "<script src='/static/missing.js'></script>")],
        present_static_assets=frozenset({"app/static/other.js"}),
    )
    assert any(w.startswith(UI_WARNING_MISSING_JS_ASSET) for w in review.warnings)


def test_present_js_asset_is_not_flagged_as_missing():
    review = review_ui(
        [UiSource("p.html", "html", "<script src='/static/chat.js'></script>")],
        present_static_assets=frozenset({"app/static/chat.js"}),
    )
    assert not any(w.startswith(UI_WARNING_MISSING_JS_ASSET) for w in review.warnings)


def test_changed_unreferenced_js_asset_is_flagged():
    review = review_ui(
        [
            UiSource("app/static/lonely.js", "js", "console.log('hi');"),
            UiSource("p.html", "html", "<div class='x'></div>"),
        ]
    )
    assert any(
        w.startswith(UI_WARNING_UNREFERENCED_STATIC_ASSET) for w in review.warnings
    )


def test_external_script_reference_is_not_treated_as_missing_asset():
    review = review_ui(
        [
            UiSource(
                "p.html",
                "html",
                "<script src='https://cdn.example.com/lib.js'></script>",
            )
        ]
    )
    assert not any(w.startswith(UI_WARNING_MISSING_JS_ASSET) for w in review.warnings)


def test_referenced_js_emitted_classes_satisfy_css_selectors():
    html = "<script src='/static/calendar.js'></script>"
    css = ".cal-day { } .day-number { } .event-chip { }"
    js = """
const dayClass = "cal-day";
day.className = dayClass;
number.classList.add("day-number");
chip.setAttribute("class", "event-chip");
"""

    review = review_ui(
        [
            UiSource("p.html", "html", html),
            UiSource("app/static/lewis.css", "css", css),
        ],
        present_static_assets=frozenset({"app/static/calendar.js"}),
        referenced_js_assets=(
            ReferencedJsAsset("app/static/calendar.js", "/static/calendar.js", js),
        ),
    )

    assert not any(
        w.startswith(UI_WARNING_ORPHAN_CSS_SELECTORS) for w in review.warnings
    )
    runtime = set(review.summary["runtime_emitted_css_selectors"])
    assert {".cal-day", ".day-number", ".event-chip"} <= runtime
    emitted = set(
        review.summary["js_emitted_selectors_by_asset"]["app/static/calendar.js"]
    )
    assert {".cal-day", ".day-number", ".event-chip"} <= emitted


def test_unreferenced_js_does_not_suppress_orphan_css_warning():
    review = review_ui(
        [
            UiSource("p.html", "html", "<div></div>"),
            UiSource("app/static/lewis.css", "css", ".chat-message { color: red; }"),
            UiSource(
                "app/static/chat.js",
                "js",
                'message.classList.add("chat-message");',
            ),
        ]
    )

    orphan = [
        w for w in review.warnings if w.startswith(UI_WARNING_ORPHAN_CSS_SELECTORS)
    ]
    assert orphan
    assert ".chat-message" in orphan[0]


def test_referenced_js_missing_target_warns():
    review = review_ui(
        [
            UiSource("p.html", "html", "<script src='/static/calendar.js'></script>"),
            UiSource(
                "app/static/calendar.js",
                "js",
                'document.getElementById("cal-grid");',
            ),
        ],
        present_static_assets=frozenset({"app/static/calendar.js"}),
    )

    missing = [
        w for w in review.warnings if w.startswith(UI_WARNING_JS_MISSING_TARGETS)
    ]
    assert missing
    assert "#cal-grid" in missing[0]
    assert review.summary["missing_js_targets"] == ["#cal-grid"]


def test_referenced_js_existing_target_does_not_warn():
    review = review_ui(
        [
            UiSource(
                "p.html",
                "html",
                "<div id='cal-grid'></div><script src='/static/calendar.js'></script>",
            ),
            UiSource(
                "app/static/calendar.js",
                "js",
                'document.getElementById("cal-grid");',
            ),
        ],
        present_static_assets=frozenset({"app/static/calendar.js"}),
    )

    assert not any(w.startswith(UI_WARNING_JS_MISSING_TARGETS) for w in review.warnings)


def test_data_container_is_ignored_for_unstyled_selector_warning():
    html = (
        '<script type="application/json" id="calendar-data">{"events":[]}</script>'
        "<div id='styled'></div>"
    )
    review = review_ui(
        [
            UiSource("p.html", "html", html),
            UiSource("app/static/lewis.css", "css", "#styled { display: block; }"),
        ]
    )

    assert not any(w.startswith(UI_WARNING_UNSTYLED_SELECTORS) for w in review.warnings)
    assert "#calendar-data" in review.summary["ignored_data_containers"]


def test_visible_unstyled_container_still_warns():
    review = review_ui(
        [
            UiSource("p.html", "html", "<div id='cal-grid'></div>"),
            UiSource("app/static/lewis.css", "css", ".other { color: red; }"),
        ]
    )

    unstyled = [
        w for w in review.warnings if w.startswith(UI_WARNING_UNSTYLED_SELECTORS)
    ]
    assert unstyled
    assert "#cal-grid" in unstyled[0]


def test_truly_orphan_css_still_warns():
    review = review_ui(
        [
            UiSource("p.html", "html", "<div></div>"),
            UiSource("app/static/lewis.css", "css", ".never-emitted { color: red; }"),
            UiSource(
                "app/static/calendar.js",
                "js",
                'cell.className = "something-else";',
            ),
        ]
    )

    orphan = [
        w for w in review.warnings if w.startswith(UI_WARNING_ORPHAN_CSS_SELECTORS)
    ]
    assert orphan
    assert ".never-emitted" in orphan[0]
    assert review.summary["true_orphan_css_selectors"] == [".never-emitted"]


def test_safe_static_reference_candidates_reject_unsafe_script_urls():
    assert static_js_reference_candidates("/static/calendar.js?v=123#month") == (
        "app/static/calendar.js",
        "static/calendar.js",
    )
    assert static_js_reference_candidates("app/static/calendar.js") == (
        "app/static/calendar.js",
    )
    assert static_js_reference_candidates("https://example.com/x.js") == ()
    assert static_js_reference_candidates("//example.com/x.js") == ()
    assert static_js_reference_candidates("/static/../secret.js") == ()
    assert static_js_reference_candidates("/not-static/x.js") == ()
    assert static_js_reference_candidates("/static/calendar.css") == ()

    review = review_ui(
        [
            UiSource(
                "p.html",
                "html",
                "<script src='/static/calendar.js?v=123#month'></script>",
            )
        ],
        present_static_assets=frozenset({"app/static/calendar.js"}),
    )
    assert not any(w.startswith(UI_WARNING_MISSING_JS_ASSET) for w in review.warnings)


def test_calendar_style_js_rendered_selectors_are_not_reported_as_orphans():
    html = """
<script type="application/json" id="calendar-data">{"events":[]}</script>
<div id="calendar-root"></div>
<script src="/static/calendar.js"></script>
"""
    css = """
#calendar-root { display: block; }
.cal-day { border: 1px solid #ddd; }
.day-number { font-weight: 700; }
.event-chip { border-radius: 4px; }
.agenda-event { padding: 8px; }
.agenda-event-title { font-weight: 700; }
.agenda-event-time { color: #555; }
.agenda-event-location { color: #555; }
.agenda-event-description { color: #333; }
"""
    js = """
JSON.parse(document.getElementById("calendar-data").textContent);
document.getElementById("cal-grid");
day.className = "cal-day";
number.classList.add("day-number");
chip.setAttribute("class", "event-chip");
agenda.innerHTML = `
  <article class="agenda-event">
    <h3 class="agenda-event-title"></h3>
    <time class="agenda-event-time"></time>
    <p class="agenda-event-location"></p>
    <p class="agenda-event-description"></p>
  </article>
`;
"""

    review = review_ui(
        [
            UiSource("p.html", "html", html),
            UiSource("app/static/lewis.css", "css", css),
            UiSource("app/static/calendar.js", "js", js),
        ],
        present_static_assets=frozenset({"app/static/calendar.js"}),
    )

    text = "\n".join(review.warnings)
    assert UI_WARNING_ORPHAN_CSS_SELECTORS not in _codes(review.warnings)
    assert "#calendar-data" not in text
    assert "#calendar-data" in review.summary["ignored_data_containers"]
    assert any(w.startswith(UI_WARNING_JS_MISSING_TARGETS) for w in review.warnings)
    assert "#cal-grid" in text
    runtime = set(review.summary["runtime_emitted_css_selectors"])
    assert {
        ".cal-day",
        ".day-number",
        ".event-chip",
        ".agenda-event",
        ".agenda-event-title",
        ".agenda-event-time",
        ".agenda-event-location",
        ".agenda-event-description",
    } <= runtime


# --------------------------------------------------------------------------- #
# Noise control and additive rendering.
# --------------------------------------------------------------------------- #


def test_markup_only_change_does_not_run_styling_checks():
    # No CSS changed: an unstyled new class must not be flagged, and a grid
    # container cannot be judged without a stylesheet view.
    review = review_ui(
        [UiSource("p.html", "html", "<div class='calendar-grid'></div>")]
    )
    codes = _codes(review.warnings)
    assert UI_WARNING_UNSTYLED_SELECTORS not in codes
    assert UI_WARNING_GRID_CONTAINER_WITHOUT_LAYOUT not in codes


def test_added_text_focuses_warnings_on_the_change():
    # The full file styles a class the change did not introduce, but only the
    # newly added orphan selector should be reported.
    css_full = ".existing { color: black; } .added-orphan { color: red; }"
    css_added = ".added-orphan { color: red; }"
    review = review_ui(
        [
            UiSource("p.html", "html", "<div class='existing'></div>"),
            UiSource("a.css", "css", css_full, added_text=css_added),
        ]
    )
    orphan = [
        w for w in review.warnings if w.startswith(UI_WARNING_ORPHAN_CSS_SELECTORS)
    ]
    assert orphan
    assert ".added-orphan" in orphan[0]
    assert ".existing" not in orphan[0]


def test_rendered_page_resolves_an_apparent_orphan():
    # The changed CSS targets a class only the unchanged page renders. The
    # changed markup view does not contain it, so without a render the selector
    # looks orphaned; the rendered page must clear it.
    css = UiSource("a.css", "css", ".server-rendered { display: grid; }")
    markup = UiSource("p.html", "html", "<div class='unrelated'></div>")
    without_render = review_ui([css, markup])
    with_render = review_ui(
        [css, markup],
        rendered_pages=[
            RenderedPage("/dashboard", "<div class='server-rendered'></div>")
        ],
    )
    assert any(
        w.startswith(UI_WARNING_ORPHAN_CSS_SELECTORS) for w in without_render.warnings
    )
    assert not any(
        w.startswith(UI_WARNING_ORPHAN_CSS_SELECTORS) for w in with_render.warnings
    )


def test_supplemental_css_is_not_treated_as_introduced():
    # Existing stylesheet supplied as context (changed=False) provides the
    # styled set without making its selectors look like new orphans.
    review = review_ui(
        [
            UiSource("p.html", "html", "<div class='calendar-grid'></div>"),
            UiSource(
                "app/static/lewis.css",
                "css",
                ".calendar-grid { display: grid; } .navbar { color: #000; }",
                changed=False,
            ),
        ]
    )
    assert review.warnings == ()


def test_classify_web_source_recognizes_kinds():
    assert classify_web_source("a.css", "") == "css"
    assert classify_web_source("a.js", "") == "js"
    assert classify_web_source("page.html", "<div></div>") == "html"
    assert (
        classify_web_source("app/web_pages.py", "<section class='x'>")
        == "python_markup"
    )
    assert classify_web_source("app/db.py", "def connect(): pass") is None
    assert classify_web_source("notes.txt", "hello") is None


def test_warnings_are_bounded():
    classes = " ".join(f"c{i}" for i in range(200))
    html = f"<div class='{classes}'></div>"
    css = ".styled { color: red; }"
    review = review_ui(
        [UiSource("p.html", "html", html), UiSource("a.css", "css", css)]
    )
    assert len(review.warnings) <= 24
    for value in review.summary.values():
        if isinstance(value, list):
            assert len(value) <= 40


# --------------------------------------------------------------------------- #
# Purity: this module must stay local and side-effect free.
# --------------------------------------------------------------------------- #


def test_ui_review_module_is_pure():
    source = (
        Path(__import__("app.dev.freestyle_ui_review", fromlist=["__file__"]).__file__)
        .read_text(encoding="utf-8")
        .casefold()
    )
    for forbidden in (
        "subprocess",
        "import socket",
        "import os",
        "from app.",
        "import app",
        "requests",
        "httpx",
        "testclient",
        "open(",
        "deepseek",
    ):
        assert forbidden not in source, forbidden
