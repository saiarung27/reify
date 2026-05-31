"""Embedder protocol.

Embedders convert text into fixed-dimensional vectors that the storage
layer indexes for similarity search. They are deliberately decoupled from
storage so that a memory can be re-embedded with a newer model without
rewriting the underlying record.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Convert text to vectors.

    Implementations must be deterministic for a given model identifier; the
    same text under the same model must always yield the same vector (within
    floating-point tolerance).
    """

    model: str
    """The model identifier. Recorded alongside vectors so that a future
    re-embed sweep can identify stale entries."""

    dim: int
    """Output vector dimension."""

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Implementations should batch internally for throughput. The return
        order matches ``texts``.
        """
