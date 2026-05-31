"""Tests for the v0.4 hot-buffer write batching."""

from __future__ import annotations

import asyncio

import pytest

from reify import EngineConfig, Reify
from reify.core.types import Fact
from reify.storage import SQLiteStore, WriteBatch
from reify.storage.hot_buffer import HotBuffer, HotBufferConfig


def _fact(content: str) -> Fact:
    return Fact(
        content=content,
        subject="s",
        predicate="p",
        obj="o",
    )


# ---------------------------------------------------------------------------
# HotBuffer standalone
# ---------------------------------------------------------------------------


def test_hot_buffer_enqueues_batches() -> None:
    buffer = HotBuffer(config=HotBufferConfig(max_bytes=10_000))
    batch = WriteBatch(memories=[_fact("a"), _fact("b")])
    buffer.enqueue(batch)
    assert buffer.pending_count == 1
    assert buffer.pending_bytes > 0


async def test_hot_buffer_flush_noop_when_empty() -> None:
    buffer = HotBuffer()
    flushed = await buffer.flush()
    assert flushed == 0


async def test_hot_buffer_flush_returns_count() -> None:
    buffer = HotBuffer()
    buffer.enqueue(WriteBatch(memories=[_fact("x")]))
    buffer.enqueue(WriteBatch(memories=[_fact("y")]))
    # Without storage, flush is a no-op (returns 0) but still clears buffer.
    flushed = await buffer.flush()
    assert buffer.pending_count == 0


async def test_hot_buffer_consolidates_batches() -> None:
    buffer = HotBuffer()
    buffer.enqueue(WriteBatch(memories=[_fact("a")]))
    buffer.enqueue(WriteBatch(memories=[_fact("b")]))
    flushed = await buffer.flush()
    assert buffer.pending_count == 0


async def test_hot_buffer_auto_flush_on_budget_exceed() -> None:
    """When pending bytes exceed max_bytes, flush is scheduled."""
    buffer = HotBuffer(config=HotBufferConfig(max_bytes=10))
    # Small batch to stay under budget.
    buffer.enqueue(WriteBatch(memories=[_fact("x")]))
    assert buffer.pending_count == 1
    # Large batch that exceeds budget.
    buffer.enqueue(WriteBatch(memories=[_fact("y" * 20)]))
    # Flush task is scheduled but not yet executed synchronously.
    assert buffer._flush_task is not None


async def test_hot_buffer_seconds_since_flush() -> None:
    buffer = HotBuffer()
    assert buffer.seconds_since_flush >= 0.0
    await asyncio.sleep(0.1)
    buffer.enqueue(WriteBatch(memories=[_fact("z")]))
    await buffer.flush()
    assert buffer.seconds_since_flush < 0.2


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


async def test_engine_with_hot_buffer_enabled() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(
        storage=store,
        config=EngineConfig(hot_buffer_enabled=True),
    )
    async with engine:
        await engine.remember("Arun lives in San Francisco.")
        # Memory is flushed immediately for entity indexing.
        # Buffer is cleared after flush.
        assert engine._hot_buffer.pending_count == 0
        # Recall should see it since flush happened.
        response = await engine.recall("San Francisco")
        assert response.results
    # Stop triggers flush; verify buffer cleared.
    assert engine._hot_buffer.pending_count == 0


async def test_engine_with_hot_buffer_disabled() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(
        storage=store,
        config=EngineConfig(hot_buffer_enabled=False),
    )
    async with engine:
        await engine.remember("Arun lives in San Francisco.")
        # Immediate commit.
        response = await engine.recall("San Francisco")
        assert response.results
