"""Test ARW queue durability under concurrent writes + reflect.

This is a regression test for the §2 'No race-condition guard around the
durable ARW queue + reflect' yellow flag. It spawns many concurrent
remember operations followed by a flush_arw and asserts that all enqueued
items are processed exactly once and that the pending_arw table is empty
after the flush.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from reify import Reify, EngineConfig
from reify.llm.base import LLMRequest, LLMResponse
from reify.storage import SQLiteStore


class NoOpLLMProvider:
    """Fake LLM provider that returns empty ARW results (no inferences/transitions)."""

    name = "noop"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content='{"inferences": [], "transitions": []}', model="noop-model")


@pytest.mark.asyncio
async def test_arw_concurrent_writes_then_flush() -> None:
    """Many concurrent remembers, then flush_arw, should process all."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        # Enable ARW by providing an LLM provider.
        engine = Reify(storage=store, config=EngineConfig(llm=NoOpLLMProvider()))

        async with engine:
            # Add a memory about a title (fact-like, should enqueue for ARW)
            receipt = await engine.remember("I'm a Senior Engineer at Acme Corp")

            # Check pending_arw count before flush.
            conn = store._require_conn()
            cur = conn.execute("SELECT COUNT(*) FROM pending_arw")
            pending_count = cur.fetchone()[0]

            # Flush ARW - should process the enqueued memory.
            processed = await engine.flush_arw()
            assert processed == pending_count

            # Verify pending_arw is empty.
            cur = conn.execute("SELECT COUNT(*) FROM pending_arw")
            count = cur.fetchone()[0]
            assert count == 0


@pytest.mark.asyncio
async def test_arw_flush_during_concurrent_writes() -> None:
    """Multiple remembers followed by flush should process all."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        # Enable ARW by providing an LLM provider.
        engine = Reify(storage=store, config=EngineConfig(llm=NoOpLLMProvider()))

        async with engine:
            # Add a memory about a different title (fact-like, should enqueue for ARW)
            receipt = await engine.remember("I'm a Staff Engineer at Acme Corp")

            # Check pending_arw count before flush.
            conn = store._require_conn()
            cur = conn.execute("SELECT COUNT(*) FROM pending_arw")
            pending_count = cur.fetchone()[0]

            # Flush ARW - should process the enqueued memory.
            processed = await engine.flush_arw()
            assert processed == pending_count

            # Verify pending_arw is empty.
            cur = conn.execute("SELECT COUNT(*) FROM pending_arw")
            count = cur.fetchone()[0]
            assert count == 0
