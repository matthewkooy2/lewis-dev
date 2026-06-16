"""Tests for the best-effort local route-render helper."""

from __future__ import annotations

from pathlib import Path

import app.dev.freestyle_route_render as route_render
from app.dev.freestyle_route_render import (
    SAFE_RENDER_ROUTES,
    _validated_routes,
    render_local_pages,
)
from app.dev.freestyle_ui_review import RenderedPage


def test_renders_safe_routes_as_html_without_a_server():
    pages = render_local_pages()
    # FastAPI/TestClient is a dependency here, so rendering should succeed.
    assert pages
    assert all(isinstance(page, RenderedPage) for page in pages)
    assert all(page.route in SAFE_RENDER_ROUTES for page in pages)
    rendered = {page.route for page in pages}
    assert "/dashboard" in rendered
    assert all(page.html.strip() for page in pages)


def test_approval_routes_are_never_rendered():
    assert "/approvals" not in SAFE_RENDER_ROUTES
    pages = render_local_pages()
    assert all("approval" not in page.route for page in pages)


def test_render_does_not_touch_the_real_database(tmp_path, monkeypatch):
    # Rendering must initialize only a throwaway database, never the configured
    # one. Point the configured path at a sentinel and confirm it is untouched.
    sentinel = tmp_path / "real.sqlite3"
    monkeypatch.setenv("LEWIS_DB_PATH", str(sentinel))
    pages = render_local_pages()
    assert pages
    assert not sentinel.exists()


def test_validated_routes_filters_to_the_allowlist():
    assert _validated_routes(["/dashboard", "/dashboard"]) == ("/dashboard",)
    assert _validated_routes(["/approvals", "/etc/passwd", 42]) == ()
    assert _validated_routes(None) == SAFE_RENDER_ROUTES


def test_render_fails_open_when_app_cannot_be_built(monkeypatch):
    def boom(_settings=None):
        raise RuntimeError("no app")

    monkeypatch.setattr(route_render, "render_local_pages", render_local_pages)
    # Simulate create_app blowing up by breaking get_settings inside the helper.
    import app.config as config

    monkeypatch.setattr(config, "get_settings", boom)
    assert render_local_pages() == ()


def test_helper_does_not_start_a_network_server_or_use_sockets():
    source = Path(route_render.__file__).read_text(encoding="utf-8").casefold()
    assert "uvicorn" not in source
    assert "import socket" not in source
    assert "run_web_server" not in source
