"""Browser-origin boundary for the unauthenticated, loopback-only dashboard."""

from urllib.parse import urlsplit

from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse
from starlette.types import ASGIApp, Receive, Scope, Send


def _origin(value: str) -> tuple[str, str | None, int] | None:
    try:
        url = urlsplit(value)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username:
            return None
        return (
            url.scheme,
            url.hostname,
            url.port or (443 if url.scheme == "https" else 80),
        )
    except ValueError:
        return None


class LocalOriginMiddleware:
    """Reject browser writes unless their Origin or Referer matches this server."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] not in {
            "GET",
            "HEAD",
            "OPTIONS",
        }:
            headers = Headers(scope=scope)
            target = _origin(f"{scope['scheme']}://{headers.get('host', '')}")
            source = _origin(headers.get("origin", headers.get("referer", "")))
            if (
                target is None
                or source != target
                or headers.get("sec-fetch-site") in {"cross-site", "same-site"}
            ):
                response = PlainTextResponse(
                    "A same-origin request is required.", status_code=403
                )
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)
