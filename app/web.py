from __future__ import annotations

from importlib import import_module
from threading import Lock

from fastapi import FastAPI
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.config import Settings, get_settings
from app.web_routes import register_web_routes
from app.web_security import LocalOriginMiddleware


LOCAL_WEB_HOST = "127.0.0.1"
LOCAL_WEB_PORT = 8787
LOCAL_WEB_URL = f"http://{LOCAL_WEB_HOST}:{LOCAL_WEB_PORT}"


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Lewis Local Dashboard")
    app.add_middleware(LocalOriginMiddleware)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])
    app.state.settings = settings or get_settings()
    app.state.chat_agent = None
    app.state.chat_messages = []
    app.state.chat_lock = Lock()
    register_web_routes(app)
    if app.state.settings.freestyle_ui_enabled:
        dev_web = import_module("app" + ".dev.freestyle_web")
        dev_web.register_freestyle_web_routes(app)
    if app.state.settings.coding_agents_ui_enabled:
        coding_agents_web = import_module("app" + ".dev.coding_agent_web")
        coding_agents_web.register_coding_agent_web_routes(app)
    if app.state.settings.projects_ui_enabled:
        project_web = import_module("app" + ".project_web")
        project_web.register_project_web_routes(app)
    return app


def run_web_server(
    settings: Settings | None = None,
    host: str = LOCAL_WEB_HOST,
    port: int = LOCAL_WEB_PORT,
) -> None:
    if host not in {LOCAL_WEB_HOST, "localhost"}:
        raise ValueError("The unauthenticated dashboard requires a loopback host.")

    import uvicorn

    print(f"Lewis local web dashboard: http://{host}:{port}")
    print("Local dashboard. Run developer checks only against trusted code.")
    uvicorn.run(create_app(settings), host=host, port=port)
