"""LLM provider tests with mocked HTTP responses.

These tests use httpx mocking to test error paths that are difficult to trigger
with real API calls: rate limits, malformed responses, network timeouts, etc.
"""

from __future__ import annotations

import pytest
import httpx
from unittest.mock import AsyncMock, patch

from reify.llm.openai import OpenAIProvider
from reify.llm.ollama import OllamaProvider
from reify.llm.base import LLMRequest, LLMResponse
from reify.core.errors import ConfigError


@pytest.mark.asyncio
async def test_openai_rate_limit_error() -> None:
    """OpenAI provider should handle 429 rate limit errors."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return 429
    mock_response = httpx.Response(
        429,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"error": {"message": "Rate limit exceeded"}},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
        )

        with pytest.raises(ConfigError, match="status 429"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_openai_malformed_json_response() -> None:
    """OpenAI provider should handle malformed JSON responses."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return invalid JSON
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        text="invalid json {{{",
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
        )

        with pytest.raises(ConfigError, match="OpenAI provider error"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_openai_missing_choices_in_response() -> None:
    """OpenAI provider should handle responses missing 'choices' field."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return response without choices
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"data": "no choices field"},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
        )

        with pytest.raises(ConfigError, match="OpenAI provider error"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_openai_empty_choices_array() -> None:
    """OpenAI provider should handle empty choices array."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return empty choices
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"choices": []},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
        )

        with pytest.raises(ConfigError, match="OpenAI provider error"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_openai_500_internal_error() -> None:
    """OpenAI provider should handle 500 internal server errors."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return 500
    mock_response = httpx.Response(
        500,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"error": {"message": "Internal server error"}},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
        )

        with pytest.raises(ConfigError, match="status 500"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_openai_401_unauthorized() -> None:
    """OpenAI provider should handle 401 unauthorized errors."""
    provider = OpenAIProvider(api_key="invalid-key")

    # Mock httpx client to return 401
    mock_response = httpx.Response(
        401,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={"error": {"message": "Invalid API key"}},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
        )

        with pytest.raises(ConfigError, match="status 401"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_ollama_malformed_response() -> None:
    """Ollama provider should handle malformed responses."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    # Mock httpx client to return invalid JSON
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        text="invalid json",
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="llama3",
        )

        with pytest.raises(ConfigError, match="Ollama error"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_ollama_missing_response_field() -> None:
    """Ollama provider should handle responses missing 'response' field."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    # Mock httpx client to return response without 'response' field
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        json={"done": True},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="llama3",
        )

        # Should return empty content if response field is missing
        result = await provider.complete(request)
        assert result.content == ""


@pytest.mark.asyncio
async def test_ollama_500_error() -> None:
    """Ollama provider should handle 500 errors."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    # Mock httpx client to return 500
    mock_response = httpx.Response(
        500,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        text="Internal server error",
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="llama3",
        )

        with pytest.raises(ConfigError, match="status 500"):
            await provider.complete(request)


@pytest.mark.asyncio
async def test_openai_successful_response_with_tokens() -> None:
    """OpenAI provider should parse successful responses correctly."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return successful response
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        },
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
        )

        result = await provider.complete(request)
        assert result.content == "Test response"


@pytest.mark.asyncio
async def test_ollama_successful_response() -> None:
    """Ollama provider should parse successful responses correctly."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    # Mock httpx client to return successful response
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        json={
            "response": "Test response",
            "prompt_eval_count": 10,
            "eval_count": 20,
            "total_duration": 1000000,
        },
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="llama3",
        )

        result = await provider.complete(request)
        assert result.content == "Test response"
        assert result.input_tokens == 10
        assert result.output_tokens == 20
        assert result.latency_ms == 1.0


@pytest.mark.asyncio
async def test_openai_with_max_tokens() -> None:
    """OpenAI provider should include max_tokens in request."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return successful response
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        },
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            max_tokens=100,
        )

        await provider.complete(request)

        # Verify max_tokens was included in the request
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["max_tokens"] == 100


@pytest.mark.asyncio
async def test_openai_with_temperature() -> None:
    """OpenAI provider should include temperature in request."""
    provider = OpenAIProvider(api_key="fake-key")

    # Mock httpx client to return successful response
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
        json={
            "choices": [
                {
                    "message": {
                        "content": "Test response"
                    }
                }
            ]
        },
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            temperature=0.7,
        )

        await provider.complete(request)

        # Verify temperature was included in the request
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert payload["temperature"] == 0.7


@pytest.mark.asyncio
async def test_ollama_with_temperature() -> None:
    """Ollama provider should include temperature in request."""
    provider = OllamaProvider(base_url="http://localhost:11434")

    # Mock httpx client to return successful response
    mock_response = httpx.Response(
        200,
        request=httpx.Request("POST", "http://localhost:11434/api/generate"),
        json={"response": "Test response"},
    )

    with patch.object(provider._client, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_response

        request = LLMRequest(
            messages=[{"role": "user", "content": "test"}],
            model="llama3",
            temperature=0.5,
        )

        await provider.complete(request)

        # Verify temperature was included in the request
        call_args = mock_post.call_args
        payload = call_args.kwargs["json"]
        assert "options" in payload
        assert payload["options"]["temperature"] == 0.5
