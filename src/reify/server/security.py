"""Shared HTTP server hardening helpers.

Reify's HTTP servers (`rest_api.py`, `openai_proxy.py`, `mcp_sse.py`) all
default to a single-user, loopback-only deployment. This module centralizes
two safety mechanisms that activate when an operator binds to a non-loopback
interface:

1.  **CORS middleware** with explicit allow-origins (no `*` by default).
2.  **Bearer-token authentication** middleware. If `REIFY_HTTP_TOKEN` is set,
    every non-health request must present a matching `Authorization: Bearer
    <token>` header.

The CLI is responsible for calling :func:`assert_safe_bind` before starting
uvicorn, which refuses to expose an unauthenticated server on a non-loopback
host.
"""

from __future__ import annotations

import os
import secrets
from collections.abc import Sequence

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})

# Endpoints that are intentionally unauthenticated (used by load balancers
# and uptime probes). Keep this list extremely short.
PUBLIC_PATHS = frozenset({"/health"})

DEFAULT_CORS_ORIGINS: tuple[str, ...] = (
    "http://127.0.0.1:5173",  # vite dev server (dashboard)
    "http://localhost:5173",
)


def _cors_origins_from_env() -> Sequence[str]:
    """Read REIFY_CORS_ORIGINS as a comma-separated list, or fall back."""
    raw = os.environ.get("REIFY_CORS_ORIGINS", "").strip()
    if not raw:
        return DEFAULT_CORS_ORIGINS
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


def _http_token_from_env() -> str | None:
    token = os.environ.get("REIFY_HTTP_TOKEN", "").strip()
    return token or None


class BearerTokenMiddleware(BaseHTTPMiddleware):
    """Reject requests whose Authorization header does not match the token.

    Activated only when `REIFY_HTTP_TOKEN` is set. Compares using a constant-
    time function to avoid timing oracles.
    """

    def __init__(self, app: FastAPI, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next):  # type: ignore[no-untyped-def]
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        scheme, _, presented = header.partition(" ")
        if scheme.lower() != "bearer" or not presented:
            return JSONResponse(
                {"detail": "Missing bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not secrets.compare_digest(presented, self._token):
            return JSONResponse(
                {"detail": "Invalid bearer token"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer"},
            )
        return await call_next(request)


def harden_app(app: FastAPI) -> None:
    """Attach CORS and (optional) bearer-token middleware to a FastAPI app.

    Idempotent: safe to call once per app at construction time.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(_cors_origins_from_env()),
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type"],
    )
    token = _http_token_from_env()
    if token:
        app.add_middleware(BearerTokenMiddleware, token=token)


def is_loopback(host: str) -> bool:
    return host in LOOPBACK_HOSTS


def assert_safe_bind(host: str) -> None:
    """Refuse to start a non-loopback server without an auth token.

    Called by the CLI right before uvicorn.run(). If the operator wants to
    expose the server on a public interface, they must opt in by setting
    REIFY_HTTP_TOKEN.
    """
    if is_loopback(host):
        return
    if _http_token_from_env():
        return
    raise SystemExit(
        f"Refusing to bind {host!r}: a non-loopback host requires a "
        "bearer token. Set REIFY_HTTP_TOKEN to a strong secret, or rebind "
        "to 127.0.0.1. See SECURITY.md for hardening guidance."
    )


__all__ = [
    "harden_app",
    "assert_safe_bind",
    "is_loopback",
    "BearerTokenMiddleware",
    "PUBLIC_PATHS",
]
