"""LLM provider interface and registry.

The interface lives in :mod:`reify.llm.base`. Concrete providers (OpenAI,
Anthropic, Ollama) live in their own modules and are loaded lazily so that
optional extras are not required at import time.
"""

import os

from reify.core.errors import ConfigError
from reify.llm.anthropic import AnthropicProvider
from reify.llm.base import LLMProvider, LLMRequest, LLMResponse, parse_provider_spec
from reify.llm.ollama import OllamaProvider
from reify.llm.openai import OpenAIProvider

__all__ = [
    "AnthropicProvider",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "OllamaProvider",
    "OpenAIProvider",
    "create_llm_provider",
    "parse_provider_spec",
]


def create_llm_provider(spec: str) -> LLMProvider:
    """Create an LLM provider from a spec string.
    
    Args:
        spec: Provider spec in format "provider:model" (e.g. "openai:gpt-4")
        
    Returns:
        Configured LLM provider instance
        
    Raises:
        ConfigError: If provider is not supported or configuration is invalid
    """
    provider_name, model = parse_provider_spec(spec)
    
    if provider_name == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ConfigError("OpenAI provider requires OPENAI_API_KEY environment variable. Set it with: export OPENAI_API_KEY=sk-...")

        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        # Custom OpenAI-compatible endpoints (vLLM, llama.cpp servers, internal
        # proxies) often present self-signed certs. Skip strict verification
        # only when the base URL is not the public OpenAI API.
        verify_ssl = "api.openai.com" not in base_url
        return OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            verify_ssl=verify_ssl,
            default_model=model,
        )

    elif provider_name == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        return OllamaProvider(base_url=base_url, default_model=model)

    elif provider_name == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ConfigError(
                "Anthropic provider requires ANTHROPIC_API_KEY environment "
                "variable. Set it with: export ANTHROPIC_API_KEY=sk-ant-..."
            )
        base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1")
        return AnthropicProvider(
            api_key=api_key,
            base_url=base_url,
            default_model=model,
        )

    else:
        raise ConfigError(
            f"Unsupported LLM provider: {provider_name}. "
            "Supported providers: openai, anthropic, ollama"
        )
