"""Tests for :mod:`reify.llm.base`."""

from __future__ import annotations

import pytest

from reify.core.errors import ConfigError
from reify.llm import parse_provider_spec


def test_parse_provider_spec_happy_path() -> None:
    assert parse_provider_spec("openai:gpt-5") == ("openai", "gpt-5")
    assert parse_provider_spec("ollama:llama3.2:3b") == ("ollama", "llama3.2:3b")
    assert parse_provider_spec("anthropic:claude-3-5-sonnet-latest") == (
        "anthropic",
        "claude-3-5-sonnet-latest",
    )


@pytest.mark.parametrize("bad", ["", ":", "openai:", ":gpt-5", "noseparator"])
def test_parse_provider_spec_rejects_malformed(bad: str) -> None:
    with pytest.raises(ConfigError):
        parse_provider_spec(bad)
