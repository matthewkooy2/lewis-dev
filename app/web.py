from __future__ import annotations

from importlib import import_module

from fastapi import FastAPI

from app.config import Settings, get_settings
from app.web_routes import register_web_routes


LOCAL_WEB_HOST = "127.0.0.1"
LOCAL_WEB_PORT = 8787
LOCAL_WEB_URL = f"http://{LOCAL_WEB_HOST}:{LOCAL_WEB_PORT}"


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(title="Lewis Local Dashboard")
    app.state.settings = settings or get_settings()
    app.state.chat_agent = None
    app.state.chat_messages = []
    register_web_routes(app)
    if app.state.settings.debug:
        dev_web = import_module("app" + ".dev.freestyle_web")
        dev_web.register_freestyle_web_routes(app)
    return app


def run_web_server(
    settings: Settings | None = None,
    host: str = LOCAL_WEB_HOST,
    port: int = LOCAL_WEB_PORT,
) -> None:
    import uvicorn

    print(f"Lewis local web dashboard: http://{host}:{port}")
    print("Bound to 127.0.0.1 only. No LAN exposure and no real external actions.")
    uvicorn.run(create_app(settings), host=host, port=port)
