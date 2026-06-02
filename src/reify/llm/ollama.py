"""Ollama LLM provider.

Uses Ollama (https://ollama.com) for local model inference. Supports
any model available via Ollama (llama2, llama3, mistral, etc.).

This is the free/local option that aligns with FIRST_PRINCIPLES.md:
- Local-first: runs on user's machine
- Graceful degradation: works offline
- No API keys required
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from reify.core.errors import ConfigError
from reify.llm.base import LLMProvider, LLMRequest, LLMResponse

log = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama provider for local model inference.

    Requires Ollama to be running locally (default: http://localhost:11434).
    Install Ollama from https://ollama.com and run `ollama serve`.

    Usage:
        provider = OllamaProvider(base_url="http://localhost:11434")
        response = await provider.complete(
            LLMRequest(messages=[...], model="llama3", temperature=0.1)
        )
    """

    name: str = "ollama"

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: float = 180.0,
        default_model: str | None = None,
    ) -> None:
        # ARW classifications are background work; local inference on a
        # 7B-8B model can run 30-90s under load (model swap-in, parallel
        # downloads, CPU contention). 60s was too aggressive and surfaced
        # as ``ReadTimeout`` errors that were silently dropped by reflect().
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self.default_model = default_model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a completion against Ollama."""
        # Convert OpenAI-style messages to Ollama format
        # Ollama uses a simpler format with "prompt" and "system"
        system_prompt = ""
        user_prompt = ""

        for msg in request.messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system_prompt = content
            elif role in ("user", "assistant"):
                user_prompt += f"{role}: {content}\n"

        # Combine system and user prompts
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n{user_prompt}"
        else:
            full_prompt = user_prompt

        # Build Ollama API request
        payload: dict[str, Any] = {
            "model": request.model,
            "prompt": full_prompt,
            "stream": False,
        }

        if request.temperature is not None:
            payload["options"] = {"temperature": request.temperature}

        if request.max_tokens is not None:
            if "options" not in payload:
                payload["options"] = {}
            payload["options"]["num_predict"] = request.max_tokens

        try:
            response = await self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data.get("response", ""),
                model=request.model,
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
                latency_ms=data.get("total_duration", 0) / 1_000_000,
            )
        except httpx.HTTPStatusError as e:
            log.error(f"Ollama request failed: {e}")
            raise ConfigError(f"Ollama request failed with status {e.response.status_code}: {e.response.text[:200]}. Ensure Ollama is running at {self._base_url}") from e
        except Exception as e:
            log.error(f"Ollama error: {e}")
            raise ConfigError(f"Ollama error: {type(e).__name__}: {e}. Ensure Ollama is running at {self._base_url}") from e

    async def __aenter__(self) -> OllamaProvider:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._client.aclose()
