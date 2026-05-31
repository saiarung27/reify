"""OpenAI-compatible HTTP proxy for Reify.

This provides a drop-in OpenAI chat/completions endpoint that injects
retrieved context from Reify before forwarding to the upstream LLM provider.

Per FIRST_PRINCIPLES.md, this adapter is thin and transport-only - it does
not contain unique memory logic absent from the library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from reify import Reify
from reify.llm.base import LLMProvider, LLMRequest, LLMResponse, parse_provider_spec
from reify.llm.openai import OpenAIProvider

log = logging.getLogger(__name__)


@dataclass
class OpenAIProxyConfig:
    """Configuration for the OpenAI proxy."""

    upstream_spec: str = "openai:gpt-4o"
    """Upstream LLM provider spec (e.g., 'openai:gpt-4o')."""

    context_limit: int = 5
    """Maximum number of memories to inject as context."""

    context_budget_tokens: int = 2000
    """Maximum tokens for injected context."""


class ChatMessage(BaseModel):
    """OpenAI chat message format."""

    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """OpenAI chat/completions request format."""

    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    max_tokens: int | None = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    """OpenAI chat/completions response format."""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[dict[str, Any]]
    usage: dict[str, int]


class OpenAIProxy:
    """OpenAI-compatible HTTP proxy server."""

    def __init__(
        self,
        engine: Reify,
        config: OpenAIProxyConfig | None = None,
    ) -> None:
        self._engine = engine
        self._config = config or OpenAIProxyConfig()
        self._app = FastAPI(title="Reify OpenAI Proxy")
        from reify.server.security import harden_app
        harden_app(self._app)
        self._upstream: LLMProvider | None = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self._app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest) -> JSONResponse:
            return await self._handle_chat_completions(request)

        @self._app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

    async def _handle_chat_completions(self, request: ChatCompletionRequest) -> JSONResponse:
        """Handle a chat/completions request."""
        # Validate the request body before allocating the upstream client so a
        # malformed request returns 400 instead of failing on missing
        # provider credentials.
        last_user_message = None
        for msg in reversed(request.messages):
            if msg.role == "user":
                last_user_message = msg.content
                break

        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        if self._upstream is None:
            try:
                self._upstream = self._create_upstream()
            except ValueError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc

        # Retrieve context from Reify.
        response = await self._engine.recall(last_user_message, limit=self._config.context_limit)
        context_memories = [r.memory.content for r in response.results]

        # Inject context into the system message.
        messages = [msg.model_dump() for msg in request.messages]
        if context_memories:
            context_text = "\n".join(f"- {m}" for m in context_memories)
            context_prompt = f"Relevant context from memory:\n{context_text}\n\nUse this context to inform your response."
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = f"{messages[0]['content']}\n\n{context_prompt}"
            else:
                messages.insert(0, {"role": "system", "content": context_prompt})

        # Forward to upstream provider.
        llm_request = LLMRequest(
            messages=messages,
            model=request.model,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )
        llm_response = await self._upstream.complete(llm_request)

        # Format as OpenAI response.
        return JSONResponse(
            {
                "id": f"chatcmpl-{id(llm_response)}",
                "object": "chat.completion",
                "created": 0,
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": llm_response.content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": llm_response.input_tokens,
                    "completion_tokens": llm_response.output_tokens,
                    "total_tokens": llm_response.input_tokens + llm_response.output_tokens,
                },
            }
        )

    def _create_upstream(self) -> LLMProvider:
        """Create the upstream LLM provider."""
        import os

        provider_name, model = parse_provider_spec(self._config.upstream_spec)
        if provider_name == "openai":
            # Read API key from environment variable.
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Set it to use the OpenAI proxy."
                )
            return OpenAIProvider(
                api_key=api_key,
                default_model=model,
            )
        raise ValueError(f"Unsupported provider: {provider_name}")

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self._app


__all__ = [
    "OpenAIProxy",
    "OpenAIProxyConfig",
]
