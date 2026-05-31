"""Anthropic Messages API LLM provider.

Wraps `https://api.anthropic.com/v1/messages` as a Reify
:class:`~reify.llm.base.LLMProvider`. Mirrors the shape of
:class:`~reify.llm.openai.OpenAIProvider` so the engine can swap between
providers via the ``provider:model`` spec string with no other change.

Differences from the OpenAI provider that this module hides from callers:

* Auth uses ``x-api-key`` plus an explicit ``anthropic-version`` header.
* Anthropic puts the system prompt in a top-level ``system`` field, not in
  the ``messages`` array.
* ``max_tokens`` is *required* by Anthropic, so this provider supplies a
  conservative default when the caller leaves :attr:`LLMRequest.max_tokens`
  unset.
* The response shape is ``{"content": [{"type": "text", "text": "..."}]}``
  with input/output token counts under ``usage``.

Streaming, tool use, and structured-output enforcement are intentionally
out of scope for v0.1, matching the engine's batch-oriented write path.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from reify.core.errors import ConfigError
from reify.llm.base import LLMProvider, LLMRequest, LLMResponse

log = logging.getLogger(__name__)


# Pinned to the latest stable API version at the time of writing. Anthropic
# guarantees backwards compatibility within a date-versioned API line, so a
# pinned version is safer than tracking "latest" and being surprised by a
# breaking field rename.
_ANTHROPIC_API_VERSION = "2023-06-01"

# Anthropic requires ``max_tokens``. We pick a conservative default that
# fits all current Reify call sites (extraction, classification, summary)
# without truncating reconsolidation output. Callers that need more set
# :attr:`LLMRequest.max_tokens` explicitly.
_DEFAULT_MAX_TOKENS = 1024


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API provider for remote model inference.

    Usage::

        provider = AnthropicProvider(
            api_key="sk-ant-...",
            default_model="claude-3-5-sonnet-latest",
        )
        response = await provider.complete(
            LLMRequest(
                messages=[
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "Hello."},
                ],
                model="claude-3-5-sonnet-latest",
            )
        )
    """

    name: str = "anthropic"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com/v1",
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
        """Run a completion against the Anthropic Messages API.

        :raises ConfigError: on non-2xx responses or any other error from
            the Anthropic API. The engine treats any provider error as a
            best-effort signal to fall back to the deterministic path.
        """
        if request.json_schema is not None:
            # Anthropic supports JSON output via prompt engineering rather
            # than a server-enforced schema. Until we add a JSON-mode helper
            # that injects the schema into the system prompt and validates
            # the response, we refuse rather than silently return free-form
            # text (per the LLMProvider docstring contract).
            raise ConfigError(
                "AnthropicProvider does not yet support json_schema "
                "(server-enforced JSON mode). Use OpenAIProvider for "
                "structured output, or remove json_schema from the request."
            )

        system_prompt, chat_messages = _split_system_message(request.messages)

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": chat_messages,
            "max_tokens": request.max_tokens or _DEFAULT_MAX_TOKENS,
            "temperature": request.temperature,
        }
        if system_prompt:
            payload["system"] = system_prompt

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_API_VERSION,
            "content-type": "application/json",
        }

        try:
            response = await self._client.post(
                f"{self._base_url}/messages",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()

            data = response.json()
            content = _extract_text(data)
            usage = data.get("usage") or {}
            return LLMResponse(
                content=content,
                model=data.get("model") or request.model,
                input_tokens=int(usage.get("input_tokens", 0) or 0),
                output_tokens=int(usage.get("output_tokens", 0) or 0),
            )

        except httpx.HTTPStatusError as e:
            log.error(
                "Anthropic API error: %s - %s",
                e.response.status_code,
                e.response.text,
            )
            raise ConfigError(
                f"Anthropic API request failed with status "
                f"{e.response.status_code}: {e.response.text[:200]}"
            )
        except ConfigError:
            raise
        except Exception as e:
            log.error("Anthropic provider error: %s", e)
            raise ConfigError(f"Anthropic provider error: {type(e).__name__}: {e}")


def _split_system_message(
    messages: list[dict[str, str]] | tuple[dict[str, str], ...] | Any,
) -> tuple[str | None, list[dict[str, str]]]:
    """Pull system messages out of the OpenAI-style messages array.

    Anthropic accepts at most one ``system`` value at the top level; if the
    caller provided multiple system messages, we concatenate them in order
    so no instruction is silently dropped.
    """
    system_parts: list[str] = []
    chat: list[dict[str, str]] = []
    for msg in messages:
        role = msg.get("role")
        if role == "system":
            content = msg.get("content", "")
            if content:
                system_parts.append(str(content))
        else:
            chat.append({"role": str(role), "content": str(msg.get("content", ""))})
    system_prompt = "\n\n".join(system_parts) if system_parts else None
    return system_prompt, chat


def _extract_text(data: dict[str, Any]) -> str:
    """Concatenate every ``text`` block in an Anthropic response.

    Anthropic returns ``content`` as a list of blocks; in v0.1 we only
    surface text, so non-text blocks (tool_use, etc.) are skipped.
    """
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise ConfigError(
            "Anthropic response missing 'content' array: "
            f"{str(data)[:200]}"
        )
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)
    if not parts:
        raise ConfigError(
            "Anthropic response contained no text blocks: "
            f"{str(data)[:200]}"
        )
    return "".join(parts)


__all__ = ["AnthropicProvider"]
