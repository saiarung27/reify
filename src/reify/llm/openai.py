"""OpenAI-compatible LLM provider.

Works with OpenAI API and any OpenAI-compatible endpoint.
Supports any model available via the configured endpoint.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from reify.core.errors import ConfigError
from reify.llm.base import LLMProvider, LLMRequest, LLMResponse

log = logging.getLogger(__name__)


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible provider for remote model inference.

    Usage:
        provider = OpenAIProvider(
            api_key="sk-...",
            base_url="https://api.openai.com/v1",
            model="gpt-4"
        )
        response = await provider.complete(
            LLMRequest(messages=[...], model="gpt-4", temperature=0.1)
        )
    """

    name: str = "openai"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 60.0,
        verify_ssl: bool = True,
        default_model: str | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.default_model = default_model
        self._client = httpx.AsyncClient(timeout=timeout, verify=verify_ssl)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a completion against OpenAI API."""
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": request.model,
            "messages": list(request.messages),
            "temperature": request.temperature,
        }

        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens

        if request.json_schema is not None:
            # OpenAI supports structured output via response_format
            payload["response_format"] = {
                "type": "json_object",
                "schema": request.json_schema,
            }

        try:
            response = await self._post_chat(headers=headers, payload=payload)
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            return LLMResponse(content=content, model=request.model)

        except httpx.HTTPStatusError as e:
            log.error(f"OpenAI API error: {e.response.status_code} - {e.response.text}")
            raise ConfigError(f"OpenAI API request failed with status {e.response.status_code}: {e.response.text[:200]}")
        except Exception as e:
            log.error(f"OpenAI provider error: {e}")
            raise ConfigError(f"OpenAI provider error: {type(e).__name__}: {e}")

    async def _post_chat(self, *, headers: dict[str, str], payload: dict[str, Any]) -> httpx.Response:
        response = await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        if response.status_code != 400 or "max_tokens" not in payload:
            return response

        try:
            data = response.json()
        except Exception:
            return response

        error = data.get("error", {})
        message = str(error.get("message", ""))
        param = error.get("param")
        if param != "max_tokens" and "max_completion_tokens" not in message:
            return response

        retry_payload = dict(payload)
        retry_payload["max_completion_tokens"] = retry_payload.pop("max_tokens")
        return await self._client.post(
            f"{self._base_url}/chat/completions",
            headers=headers,
            json=retry_payload,
        )
