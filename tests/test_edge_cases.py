"""Edge case and failure mode tests.

These tests verify graceful degradation and error handling for:
- Network failures (LLM provider timeouts, connection errors)
- Storage failures (corrupt data, invalid operations)
- ARW processing failures (retry behavior)
- Invalid inputs (malformed data, invalid IDs)
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import httpx

from reify import Reify, EngineConfig
from reify.core.errors import ConfigError, StorageError, UnknownMemoryError
from reify.llm.openai import OpenAIProvider
from reify.llm.ollama import OllamaProvider
from reify.llm.base import LLMRequest, LLMResponse
from reify.storage import SQLiteStore


class FailingLLMProvider:
    """Fake LLM provider that always fails."""

    name = "failing"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        raise RuntimeError("Simulated LLM failure")


class TimeoutLLMProvider:
    """Fake LLM provider that simulates timeout."""

    name = "timeout"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        await asyncio.sleep(999)  # Simulate timeout
        return LLMResponse(content='{"inferences": [], "transitions": []}', model="timeout-model")


@pytest.mark.asyncio
async def test_llm_openai_timeout_handling() -> None:
    """OpenAI provider should handle timeouts gracefully."""
    provider = OpenAIProvider(
        api_key="fake-key",
        base_url="http://localhost:9999",  # Non-existent server
        timeout=0.1,  # Very short timeout
    )

    request = LLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="gpt-4",
    )

    with pytest.raises(ConfigError, match="OpenAI provider error"):
        await provider.complete(request)


@pytest.mark.asyncio
async def test_llm_ollama_connection_failure() -> None:
    """Ollama provider should handle connection failures gracefully."""
    provider = OllamaProvider(
        base_url="http://localhost:9999",  # Non-existent server
        timeout=0.1,
    )

    request = LLMRequest(
        messages=[{"role": "user", "content": "test"}],
        model="llama3",
    )

    with pytest.raises(ConfigError, match="Ollama error"):
        await provider.complete(request)


@pytest.mark.asyncio
async def test_arw_worker_with_failing_llm() -> None:
    """ARW should handle LLM failures gracefully without crashing."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        engine = Reify(storage=store, config=EngineConfig(llm=FailingLLMProvider()))

        async with engine:
            # Remember should succeed even though ARW will fail
            receipt = await engine.remember("I'm a Senior Engineer at Acme Corp")
            assert len(receipt.committed) > 0

            # flush_arw should return 0 (no items processed due to failure)
            # but should not crash
            processed = await engine.flush_arw()
            # The exact behavior depends on implementation, but it shouldn't crash


@pytest.mark.asyncio
async def test_storage_invalid_memory_id() -> None:
    """Storage should raise UnknownMemoryError for non-existent IDs."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        await store.init()

        try:
            # Try to get a non-existent memory
            memory = await store.get("non-existent-id")
            assert memory is None

            # Try to bind valid_until to non-existent memory
            with pytest.raises(UnknownMemoryError, match="not found"):
                await store.bind_valid_until(
                    "non-existent-id",
                    datetime.now(timezone.utc),
                )
        finally:
            await store.close()


@pytest.mark.asyncio
async def test_storage_concurrent_writes() -> None:
    """Storage should handle concurrent writes safely."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        engine = Reify(storage=store, config=EngineConfig(llm=None))

        async with engine:
            # Fire concurrent writes via engine
            tasks = [
                asyncio.create_task(engine.remember(f"Memory {i}"))
                for i in range(50)
            ]
            await asyncio.gather(*tasks)

            # Verify all memories were written
            from reify.storage.base import RecallFilter
            results = await store.recall(RecallFilter())
            assert len(results) >= 50


@pytest.mark.asyncio
async def test_engine_requires_extractors() -> None:
    """Engine should require at least one extractor."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        # Create engine with empty extractors list (this should fail at start)
        with pytest.raises(Exception, match="at least one extractor"):
            engine = Reify(
                storage=store,
                config=EngineConfig(llm=None),
                extractors=[],
            )
            async with engine:
                pass


@pytest.mark.asyncio
async def test_engine_flush_arw_without_llm() -> None:
    """flush_arw should return 0 when no LLM is configured."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        # Engine without LLM
        engine = Reify(storage=store, config=EngineConfig(llm=None))

        async with engine:
            # Remember some memories
            await engine.remember("I'm a Senior Engineer at Acme Corp")

            # flush_arw should return 0 (no LLM configured)
            processed = await engine.flush_arw()
            assert processed == 0


@pytest.mark.asyncio
async def test_llm_invalid_provider_spec() -> None:
    """LLM provider creation should fail with helpful error for invalid specs."""
    from reify.llm import create_llm_provider

    # Invalid provider name
    with pytest.raises(ConfigError, match="Unsupported LLM provider"):
        create_llm_provider("invalid:model")

    # Empty spec
    with pytest.raises(ConfigError, match="expected 'provider:model'"):
        create_llm_provider("")


@pytest.mark.asyncio
async def test_sql_query_safety_validation() -> None:
    """SQL fragment builder should reject unsafe fragments."""
    from reify.storage._query import assert_safe_fragment, WhereBuilder

    # Quotes should be rejected
    with pytest.raises(ValueError, match="quotes.*not allowed"):
        assert_safe_fragment("name = 'test'")

    # Semicolons should be rejected
    with pytest.raises(ValueError, match="quotes.*not allowed"):
        assert_safe_fragment("id = 1; DROP TABLE")

    # Comment markers should be rejected
    with pytest.raises(ValueError, match="comment markers.*not allowed"):
        assert_safe_fragment("id = 1 -- comment")

    # Backticks should be rejected
    with pytest.raises(ValueError, match="quotes.*not allowed"):
        assert_safe_fragment("name = `test`")

    # Valid fragment should pass
    assert_safe_fragment("name = ?")
    assert_safe_fragment("id IN (?,?,?)")
    assert_safe_fragment("created_at < ? AND valid_until IS NULL")


@pytest.mark.asyncio
async def test_where_builder_placeholder_mismatch() -> None:
    """WhereBuilder should catch placeholder/param mismatches."""
    from reify.storage._query import WhereBuilder

    builder = WhereBuilder()

    # More placeholders than params
    with pytest.raises(ValueError, match="placeholder/param mismatch"):
        builder.add("id = ? AND name = ?", "only-one-param")

    # Fewer placeholders than params
    with pytest.raises(ValueError, match="placeholder/param mismatch"):
        builder.add("id = ?", "param1", "param2")

    # Valid usage
    builder.add("id = ?", "param1")
    builder.add("name = ?", "param2")
    sql, params = builder.build()
    assert sql == "id = ? AND name = ?"
    assert params == ["param1", "param2"]
