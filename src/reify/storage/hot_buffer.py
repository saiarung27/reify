"""In-memory write buffer for batching before durable commit.

The hot-buffer accumulates :class:`WriteBatch` objects in memory and flushes
them to the underlying storage in larger batches. This reduces I/O overhead
and enables consolidation (deduplication, merging) before persistence.

Per FIRST_PRINCIPLES.md commandment 7, the hot-buffer is *off the foreground
write path*: writes are visible to recall via the in-memory index but durability
is deferred until flush. The buffer is sized by byte budget and flush interval;
exceeding either triggers an automatic flush.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from reify.storage.base import WriteBatch

if TYPE_CHECKING:
    from reify.storage import Storage


@dataclass(frozen=True, slots=True)
class HotBufferConfig:
    """Configuration for the hot-buffer."""

    max_bytes: int = 1024 * 1024  # 1MB default
    """Maximum buffered data before auto-flush. Measured in approximate
    serialized size (sum of memory content lengths)."""

    flush_interval_seconds: float = 5.0
    """Maximum time between flushes. Even if the byte budget is not hit,
    the buffer flushes after this interval to bound latency."""


@dataclass(slots=True)
class _BufferedBatch:
    """A single buffered write with metadata for size tracking."""

    batch: WriteBatch
    approx_bytes: int
    enqueued_at: datetime


@dataclass(slots=True)
class HotBuffer:
    """In-memory write buffer with auto-flush on size or time.

    The buffer is thread-unsafe by design: it must be used from a single
    async event loop (the engine's loop). Flushes are scheduled via
    ``asyncio.create_task`` and race-protected by a lock.

    Note: In the current implementation, the engine flushes the buffer
    immediately after enqueueing for entity indexing. The buffer still
    provides value for consolidating rapid writes into single transactions.
    """

    config: HotBufferConfig = field(default_factory=HotBufferConfig)
    storage: Storage | None = None
    """Injected after engine start. The buffer is usable without storage
    (for testing) but flushes are no-ops until set."""

    _batches: list[_BufferedBatch] = field(default_factory=list)
    _total_bytes: int = 0
    _flush_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _flush_task: asyncio.Task[None] | None = None
    _last_flush_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def enqueue(self, batch: WriteBatch) -> None:
        """Add a batch to the buffer. Triggers auto-flush if budget exceeded."""
        approx_bytes = sum(len(m.content) for m in batch.memories)
        buffered = _BufferedBatch(
            batch=batch,
            approx_bytes=approx_bytes,
            enqueued_at=datetime.now(timezone.utc),
        )
        self._batches.append(buffered)
        self._total_bytes += approx_bytes

        if self._total_bytes >= self.config.max_bytes:
            # Immediate flush on budget overrun; schedule async to avoid
            # blocking the caller.
            self._schedule_flush()

    async def flush(self) -> int:
        """Flush all buffered batches to storage. Returns count of batches flushed."""
        if not self._batches:
            return 0

        async with self._flush_lock:
            if not self._batches:
                return 0

            batches_to_flush = self._batches
            self._batches = []
            self._total_bytes = 0
            self._last_flush_at = datetime.now(timezone.utc)

            if self.storage is None:
                # No-op when storage not set (e.g., before engine start).
                return 0

            # Consolidate: merge all batches into one for a single transaction.
            consolidated = self._consolidate(batches_to_flush)
            await self.storage.commit_batch(consolidated)
            return len(batches_to_flush)

    def _schedule_flush(self) -> None:
        """Schedule a background flush if one is not already running."""
        if self._flush_task is None or self._flush_task.done():
            self._flush_task = asyncio.create_task(self._auto_flush_loop())

    async def _auto_flush_loop(self) -> None:
        """Background loop that flushes on interval."""
        while True:
            await asyncio.sleep(self.config.flush_interval_seconds)
            if self._batches:
                await self.flush()
            else:
                # No work; exit the loop to avoid idle wakeups.
                break

    def _consolidate(self, batches: list[_BufferedBatch]) -> WriteBatch:
        """Merge multiple batches into one for a single transaction."""
        all_memories: list = []
        all_transitions: list = []
        all_bonds: list = []
        for b in batches:
            all_memories.extend(b.batch.memories)
            all_transitions.extend(b.batch.transitions)
            all_bonds.extend(b.batch.bonds)
        return WriteBatch(
            memories=all_memories,
            transitions=all_transitions,
            bonds=all_bonds,
        )

    @property
    def pending_count(self) -> int:
        """Number of batches currently buffered."""
        return len(self._batches)

    @property
    def pending_bytes(self) -> int:
        """Approximate byte size of buffered data."""
        return self._total_bytes

    @property
    def seconds_since_flush(self) -> float:
        """Seconds since the last successful flush."""
        return (datetime.now(timezone.utc) - self._last_flush_at).total_seconds()


__all__ = [
    "HotBuffer",
    "HotBufferConfig",
]
