"""Tests for the v0.6 OpenAI-compatible HTTP proxy."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reify import Reify
from reify.server.openai_proxy import OpenAIProxy, OpenAIProxyConfig
from reify.storage import SQLiteStore


@pytest.fixture
def engine() -> Reify:
    store = SQLiteStore(":memory:")
    return Reify(storage=store)


@pytest.fixture
def proxy(engine: Reify) -> OpenAIProxy:
    config = OpenAIProxyConfig(
        upstream_spec="openai:gpt-4o",
        context_limit=3,
    )
    return OpenAIProxy(engine, config)


def test_proxy_health_endpoint(proxy: OpenAIProxy) -> None:
    client = TestClient(proxy.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_proxy_chat_completions_no_user_message(proxy: OpenAIProxy) -> None:
    client = TestClient(proxy.app)
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "gpt-4o",
            "messages": [{"role": "system", "content": "You are helpful."}],
        },
    )
    assert response.status_code == 400


def test_proxy_config_defaults() -> None:
    config = OpenAIProxyConfig()
    assert config.upstream_spec == "openai:gpt-4o"
    assert config.context_limit == 5
    assert config.context_budget_tokens == 2000


def test_proxy_config_custom() -> None:
    config = OpenAIProxyConfig(
        upstream_spec="openai:gpt-3.5-turbo",
        context_limit=10,
        context_budget_tokens=4000,
    )
    assert config.upstream_spec == "openai:gpt-3.5-turbo"
    assert config.context_limit == 10
    assert config.context_budget_tokens == 4000
