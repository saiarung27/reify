"""LLM provider abstraction.

Every model call in Reify goes through this interface. Concrete providers are
loaded by spec strings of the form ``provider:model`` (e.g. ``openai:gpt-5``).

We deliberately keep the surface narrow. Streaming, function calling, and
tool use are *not* required of providers in v0; the engine's write path is
batch-oriented and does not benefit from streaming. Tool use lives at the
agent layer above Reify, not inside it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from reify.core.errors import ConfigError


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """A single request to a provider.

    The provider is responsible for serializing this into its own wire
    format. ``json_schema`` enables structured output where supported; if
    the provider cannot enforce a schema, it must raise rather than silently
    return free-form text.
    """

    messages: Sequence[dict[str, str]]
    """OpenAI-style messages: ``[{"role": "system", "content": "..."}]``."""

    model: str
    """Provider-specific model identifier (without the ``provider:`` prefix)."""

    temperature: float = 0.0
    max_tokens: int | None = None
    json_schema: dict[str, Any] | None = None
    """If set, the provider must return content matching this JSON Schema."""

    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """The result of an :class:`LLMRequest`."""

    content: str
    """Text response. If ``json_schema`` was set, this is a JSON string."""

    model: str
    """Echo of the model that produced the response."""

    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@runtime_checkable
class LLMProvider(Protocol):
    """Provider contract.

    Implementations should be lightweight to construct; expensive setup
    (HTTP clients, auth handshakes) is permitted but should be lazy where
    possible.
    """

    name: str
    """Stable provider name. Examples: ``openai``, ``anthropic``, ``ollama``."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Run a single completion. Must be safe to call concurrently."""


def parse_provider_spec(spec: str) -> tuple[str, str]:
    """Split a ``provider:model`` spec into its parts.

    >>> parse_provider_spec("openai:gpt-5")
    ('openai', 'gpt-5')

    Raises :class:`ConfigError` if the spec is malformed.
    """
    if ":" not in spec:
        raise ConfigError(
            f"Invalid LLM provider spec {spec!r}: expected 'provider:model' (e.g. 'openai:gpt-5')."
        )
    provider, _, model = spec.partition(":")
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ConfigError(f"Invalid LLM provider spec {spec!r}: empty provider or model.")
    return provider, model
