from __future__ import annotations

import os
from urllib.parse import urlsplit

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse


LOCAL_ORIGINS = (
    "http://127.0.0.1:7788", "http://localhost:7788",
    "http://127.0.0.1:8002", "http://localhost:8002",
)


def allowed_browser_origins() -> list[str]:
    origins = list(LOCAL_ORIGINS)
    for value in os.getenv("BACKEND_ALLOWED_ORIGINS", "").split(","):
        value = value.strip().rstrip("/")
        if not value:
            continue
        parts = urlsplit(value)
        if (parts.scheme not in {"http", "https"} or not parts.hostname
                or parts.username or parts.password or parts.path or parts.query
                or parts.fragment or "*" in value):
            raise ValueError("BACKEND_ALLOWED_ORIGINS must contain explicit HTTP(S) origins")
        if value not in origins:
            origins.append(value)
    return origins


def configure_http_security(app: FastAPI, origins: list[str] | None = None) -> None:
    trusted = origins if origins is not None else allowed_browser_origins()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=trusted,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Accept"],
    )

    @app.middleware("http")
    async def reject_untrusted_browser_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        # CORS alone hides responses but does not stop simple cross-site writes.
        if (request.url.path == "/api" or request.url.path.startswith("/api/")) and origin is not None and origin not in trusted:
            return JSONResponse(status_code=403, content={"detail": "Browser origin is not allowed"})
        return await call_next(request)
