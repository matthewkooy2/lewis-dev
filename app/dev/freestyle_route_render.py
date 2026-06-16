"""Best-effort local route-render helper for freestyle web-UI review.

This renders a small allowlist of read-only page routes through the existing
in-process FastAPI/``TestClient`` pattern and returns their HTML so the UI
review can inspect what the application actually emits, not just the changed
source text. It never starts a network server (``TestClient`` drives the ASGI
app in-process), never requires Playwright or screenshots, and points the app
at a throwaway database so it never reads or mutates the real approval store.

Everything here is best-effort and fail-open: any import error, route error, or
server exception yields fewer (or no) rendered pages rather than raising, so a
review never fails because rendering was unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
import tempfile

from app.dev.freestyle_ui_review import RenderedPage

# Read-only, side-effect-free page routes worth rendering for layout inspection.
# Approval routes are deliberately excluded: this helper must not touch approval
# routes or the approval store, and the approve/reject/execute endpoints are
# POST-only mutations regardless.
SAFE_RENDER_ROUTES: tuple[str, ...] = (
    "/dashboard",
    "/calendar",
    "/tasks",
    "/history",
    "/usage",
    "/memories",
    "/documents",
    "/chat",
)

MAX_RENDER_ROUTES = 12
MAX_RENDERED_HTML_CHARS = 200_000


def render_local_pages(
    routes: Sequence[str] | None = None,
) -> tuple[RenderedPage, ...]:
    """Render the allowlisted routes in-process and return their HTML.

    Returns an empty tuple if FastAPI/``TestClient`` is unavailable or the app
    cannot be built; individual route failures are skipped. Never raises.
    """
    selected = _validated_routes(routes)
    if not selected:
        return ()
    try:
        from fastapi.testclient import TestClient

        from app.config import get_settings
        from app.web import create_app
    except Exception:
        return ()

    with tempfile.TemporaryDirectory(prefix="lewis-ui-render-") as tmp:
        # A throwaway database keeps the real approval store untouched; debug is
        # off so optional developer-only routes are not mounted.
        try:
            settings = replace(
                get_settings(),
                database_path=Path(tmp) / "render.sqlite3",
                debug=False,
            )
            app = create_app(settings)
        except Exception:
            return ()
        return _render_with_client(TestClient, app, selected)


def _render_with_client(test_client_cls, app, routes: tuple[str, ...]):
    pages: list[RenderedPage] = []
    try:
        client = test_client_cls(app, raise_server_exceptions=False)
    except Exception:
        return ()
    with suppress(Exception):
        with client:
            for route in routes:
                page = _render_one(client, route)
                if page is not None:
                    pages.append(page)
    return tuple(pages)


def _render_one(client, route: str) -> RenderedPage | None:
    try:
        response = client.get(route)
    except Exception:
        return None
    if response.status_code != 200:
        return None
    content_type = response.headers.get("content-type", "")
    if "html" not in content_type.casefold():
        return None
    try:
        html = response.text
    except Exception:
        return None
    return RenderedPage(route=route, html=html[:MAX_RENDERED_HTML_CHARS])


def _validated_routes(routes: Sequence[str] | None) -> tuple[str, ...]:
    candidates = SAFE_RENDER_ROUTES if routes is None else routes
    safe: list[str] = []
    for route in candidates:
        if (
            isinstance(route, str)
            and route in SAFE_RENDER_ROUTES
            and route not in safe
        ):
            safe.append(route)
    return tuple(safe[:MAX_RENDER_ROUTES])
