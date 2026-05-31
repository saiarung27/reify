"""Tests for the shared HTTP server hardening helpers."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reify.server.security import (
    assert_safe_bind,
    harden_app,
    is_loopback,
)


def _make_app() -> FastAPI:
    app = FastAPI()
    harden_app(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/private")
    async def private() -> dict[str, str]:
        return {"hello": "world"}

    return app


def test_is_loopback() -> None:
    assert is_loopback("127.0.0.1")
    assert is_loopback("localhost")
    assert is_loopback("::1")
    assert not is_loopback("0.0.0.0")
    assert not is_loopback("10.0.0.1")


def test_assert_safe_bind_accepts_loopback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REIFY_HTTP_TOKEN", raising=False)
    # Should not raise.
    assert_safe_bind("127.0.0.1")
    assert_safe_bind("localhost")


def test_assert_safe_bind_rejects_non_loopback_without_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REIFY_HTTP_TOKEN", raising=False)
    with pytest.raises(SystemExit) as exc:
        assert_safe_bind("0.0.0.0")
    assert "bearer token" in str(exc.value).lower()


def test_assert_safe_bind_allows_non_loopback_with_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REIFY_HTTP_TOKEN", "s3cret")
    # Should not raise.
    assert_safe_bind("0.0.0.0")


def test_harden_app_no_token_allows_unauth_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REIFY_HTTP_TOKEN", raising=False)
    client = TestClient(_make_app())
    assert client.get("/health").status_code == 200
    assert client.get("/private").status_code == 200


def test_harden_app_with_token_blocks_unauth_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REIFY_HTTP_TOKEN", "s3cret")
    client = TestClient(_make_app())

    # Health is intentionally public.
    assert client.get("/health").status_code == 200

    # Missing header.
    r = client.get("/private")
    assert r.status_code == 401
    assert r.headers["www-authenticate"] == "Bearer"

    # Wrong scheme.
    r = client.get("/private", headers={"Authorization": "Basic abc"})
    assert r.status_code == 401

    # Wrong token.
    r = client.get("/private", headers={"Authorization": "Bearer wrong"})
    assert r.status_code == 401

    # Correct token.
    r = client.get("/private", headers={"Authorization": "Bearer s3cret"})
    assert r.status_code == 200


def test_harden_app_cors_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REIFY_HTTP_TOKEN", raising=False)
    monkeypatch.setenv("REIFY_CORS_ORIGINS", "https://example.com")
    client = TestClient(_make_app())
    r = client.options(
        "/private",
        headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == "https://example.com"
