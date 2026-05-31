"""Tests for :mod:`reify.llm.anthropic`.

Mocks the underlying ``httpx.AsyncClient.post`` so the provider can be
exercised end-to-end without a real Anthropic API key.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import httpx
import pytest

from reify.core.errors import ConfigError
from reify.llm import AnthropicProvider, create_llm_provider
from reify.llm.anthropic import (
    _DEFAULT_MAX_TOKENS,
    _extract_text,
    _split_system_message,
)
from reify.llm.base import LLMRequest


# ---------------------------------------------------------------------------
# Pure-function helpers
# ---------------------------------------------------------------------------


def test_split_system_message_extracts_single_system() -> None:
    system, chat = _split_system_message(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello."},
        ]
    )
    assert system == "Be concise."
    assert chat == [{"role": "user", "content": "Hello."}]


def test_split_system_message_concatenates_multiple_systems() -> None:
    system, chat = _split_system_message(
        [
            {"role": "system", "content": "Be concise."},
            {"role": "system", "content": "Use markdown."},
            {"role": "user", "content": "Hello."},
        ]
    )
    assert system == "Be concise.\n\nUse markdown."
    assert chat == [{"role": "user", "content": "Hello."}]


def test_split_system_message_returns_none_when_no_system() -> None:
    system, chat = _split_system_message(
        [{"role": "user", "content": "Hi"}]
    )
    assert system is None
    assert chat == [{"role": "user", "content": "Hi"}]


def test_extract_text_concatenates_text_blocks() -> None:
    data = {
        "content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world."},
        ]
    }
    assert _extract_text(data) == "Hello world."


def test_extract_text_skips_non_text_blocks() -> None:
    data = {
        "content": [
            {"type": "tool_use", "id": "x"},
            {"type": "text", "text": "ok"},
        ]
    }
    assert _extract_text(data) == "ok"


def test_extract_text_raises_when_content_missing() -> None:
    with pytest.raises(ConfigError, match="missing 'content'"):
        _extract_text({})


def test_extract_text_raises_when_no_text_blocks() -> None:
    with pytest.raises(ConfigError, match="no text blocks"):
        _extract_text({"content": [{"type": "tool_use", "id": "x"}]})


# ---------------------------------------------------------------------------
# complete() with a mocked HTTP client
# ---------------------------------------------------------------------------


def _ok_response(*, content: str = "ok", model: str = "claude-3-5-sonnet-latest") -> httpx.Response:
    return httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
        json={
            "content": [{"type": "text", "text": content}],
            "model": model,
            "usage": {"input_tokens": 4, "output_tokens": 2},
        },
    )


@pytest.mark.asyncio
async def test_anthropic_complete_happy_path() -> None:
    provider = AnthropicProvider(api_key="sk-ant-fake")
    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _ok_response(content="Hello.", model="claude-3-5-sonnet-latest")
        resp = await provider.complete(
            LLMRequest(
                messages=[
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hi."},
                ],
                model="claude-3-5-sonnet-latest",
            )
        )

    assert resp.content == "Hello."
    assert resp.model == "claude-3-5-sonnet-latest"
    assert resp.input_tokens == 4
    assert resp.output_tokens == 2

    # Verify wire shape: the system prompt must be a top-level field, not
    # a chat message; max_tokens must be present even if the caller omitted
    # it; auth headers must be set correctly.
    args, kwargs = mock_post.call_args
    sent_payload = kwargs["json"]
    sent_headers = kwargs["headers"]
    assert sent_payload["system"] == "Be concise."
    assert sent_payload["messages"] == [{"role": "user", "content": "Hi."}]
    assert sent_payload["max_tokens"] == _DEFAULT_MAX_TOKENS
    assert sent_payload["model"] == "claude-3-5-sonnet-latest"
    assert sent_headers["x-api-key"] == "sk-ant-fake"
    assert sent_headers["anthropic-version"] == "2023-06-01"


@pytest.mark.asyncio
async def test_anthropic_complete_respects_caller_max_tokens() -> None:
    provider = AnthropicProvider(api_key="sk-ant-fake")
    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = _ok_response()
        await provider.complete(
            LLMRequest(
                messages=[{"role": "user", "content": "Hi."}],
                model="claude-3-5-sonnet-latest",
                max_tokens=42,
            )
        )

    sent_payload = mock_post.call_args.kwargs["json"]
    assert sent_payload["max_tokens"] == 42
    # No system block when the caller didn't supply one.
    assert "system" not in sent_payload


@pytest.mark.asyncio
async def test_anthropic_complete_raises_on_4xx() -> None:
    provider = AnthropicProvider(api_key="sk-ant-fake")
    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = httpx.Response(
            429,
            request=httpx.Request("POST", "https://api.anthropic.com/v1/messages"),
            json={"error": {"type": "rate_limit_error", "message": "slow down"}},
        )
        with pytest.raises(ConfigError, match="status 429"):
            await provider.complete(
                LLMRequest(
                    messages=[{"role": "user", "content": "Hi."}],
                    model="claude-3-5-sonnet-latest",
                )
            )


@pytest.mark.asyncio
async def test_anthropic_complete_rejects_json_schema() -> None:
    """v0.1 does not enforce server-side JSON schema; the provider must
    raise rather than silently return free-form text."""
    provider = AnthropicProvider(api_key="sk-ant-fake")
    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        with pytest.raises(ConfigError, match="json_schema"):
            await provider.complete(
                LLMRequest(
                    messages=[{"role": "user", "content": "Hi."}],
                    model="claude-3-5-sonnet-latest",
                    json_schema={"type": "object"},
                )
            )
        # We must not have hit the wire when refusing.
        mock_post.assert_not_called()


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


def test_create_llm_provider_anthropic_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        create_llm_provider("anthropic:claude-3-5-sonnet-latest")


def test_create_llm_provider_anthropic_returns_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake")
    provider = create_llm_provider("anthropic:claude-3-5-sonnet-latest")
    assert isinstance(provider, AnthropicProvider)
    assert provider.default_model == "claude-3-5-sonnet-latest"
    assert provider.name == "anthropic"
