"""Storage protocol.

Every storage backend implements this interface. The default backend is
SQLite + ``sqlite-vec``; Postgres + pgvector and Turso are planned for v1.

Design principles
-----------------
* **Atomic writes.** A single :meth:`Storage.commit_batch` call writes a
  set of memories, transitions, and identity bonds in one transaction.
  Either everything lands or nothing does.
* **No hidden network IO.** Embedding and LLM calls are the responsibility
  of upstream pipelines; storage only persists what is handed to it.
* **Bitemporal native.** Every memory carries ``valid_at`` /
  ``valid_until`` columns and the storage layer offers ``as_of`` and
  ``during`` query helpers natively.
* **Soft delete by default.** Hard delete is available for compliance use
  cases but is opt-in.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from reify.core.ids import MemoryId
from reify.core.types import (
    IdentityBond,
    Memory,
    Network,
    Transition,
)

# Avoid a circular import: retrieve depends on storage at module load.
# We import the entity types lazily inside method signatures via TYPE_CHECKING.
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from reify.retrieve.entities import Entity, EntityEdge, EntityMention


@dataclass(frozen=True, slots=True)
class WriteBatch:
    """A unit of work for :meth:`Storage.commit_batch`.

    Treat this as immutable. Build it up in calling code, then hand it to
    storage in a single call.
    """

    memories: Sequence[Memory] = ()
    transitions: Sequence[Transition] = ()
    bonds: Sequence[IdentityBond] = ()


@dataclass(frozen=True, slots=True)
class RecallFilter:
    """A typed filter for :meth:`Storage.recall`.

    All fields are optional and combine with conjunctive (AND) semantics.
    """

    networks: Sequence[Network] | None = None
    identity_ids: Sequence[str] | None = None
    tags_any: Sequence[str] | None = None
    valid_at: datetime | None = None
    """If set, only memories whose validity interval contains this instant
    are returned. This is the bitemporal ``as_of`` operator."""
    since: datetime | None = None
    """Lower bound on ``created_at``."""
    include_deleted: bool = False


@runtime_checkable
class Storage(Protocol):
    """The storage interface every backend must satisfy.

    Implementations should be cheap to construct and may keep a connection
    pool internally. They must be safe to use from a single async event loop;
    cross-thread / cross-process safety is the implementation's choice.
    """

    # -- lifecycle ----------------------------------------------------------

    async def init(self) -> None:
        """Create or migrate the underlying schema. Idempotent."""

    async def close(self) -> None:
        """Release any held resources."""

    # -- writes -------------------------------------------------------------

    async def commit_batch(self, batch: WriteBatch) -> None:
        """Atomically persist a :class:`WriteBatch`.

        Raises :class:`reify.core.errors.StorageError` on operational
        failure. Schema violations are caught earlier by Pydantic and surface
        as :class:`reify.core.errors.SchemaError`.
        """

    async def attach_embedding(
        self,
        memory_id: MemoryId,
        vector: Sequence[float],
        *,
        model: str,
    ) -> None:
        """Attach a vector embedding to a memory.

        Embedders are decoupled from writes so that the same record can be
        re-embedded with a new model without rewriting the memory.
        """

    async def soft_delete(self, memory_id: MemoryId, *, at: datetime | None = None) -> None:
        """Tombstone a memory. Excluded from default recall thereafter."""

    async def bind_valid_until(self, memory_id: MemoryId, valid_until: datetime) -> None:
        """Bind a memory's validity end time (bitemporal).

        Used by ARW when a contradicts/resolves relation is detected to
        bound the prior claim's validity window.
        """

    async def update_metadata(self, memory_id: MemoryId, metadata: dict[str, Any]) -> None:
        """Replace a memory's structured metadata blob."""

    async def soft_delete_matching(
        self,
        *,
        tag: str | None = None,
        before: datetime | None = None,
        identity_id: str | None = None,
        at: datetime | None = None,
    ) -> int:
        """Tombstone memories matching a coarse filter and return the count."""

    async def sweep_expired(self, *, now: datetime | None = None) -> int:
        """Tombstone memories whose retention policy has expired."""

    async def expired_memory_ids(self, *, now: datetime | None = None) -> list[MemoryId]:
        """Return ids of memories whose retention policy has expired."""

    # -- reads --------------------------------------------------------------

    async def get(self, memory_id: MemoryId) -> Memory | None:
        """Return a memory by id, or ``None`` if it does not exist."""

    async def recall(
        self,
        filter_: RecallFilter,
        *,
        limit: int = 50,
    ) -> list[Memory]:
        """Return memories matching ``filter_``, newest-first by default.

        This is the *non-vector* recall path used by lexical and graph
        retrieval stages. Vector retrieval lives on dedicated methods below.
        """

    async def vector_search(
        self,
        vector: Sequence[float],
        *,
        networks: Sequence[Network] | None = None,
        identity_ids: Sequence[str] | None = None,
        valid_at: datetime | None = None,
        limit: int = 20,
    ) -> list[tuple[Memory, float]]:
        """Approximate-nearest-neighbor search.

        Returns ``(memory, distance)`` pairs ordered by ascending distance.
        ``valid_at`` enables bitemporal filtering inside the index scan.
        """

    # -- transitions --------------------------------------------------------

    async def transitions_for(
        self,
        *,
        memory_ids: Iterable[MemoryId] | None = None,
        between: tuple[datetime, datetime] | None = None,
        identity_id: str | None = None,
    ) -> list[Transition]:
        """Return transitions matching the given filters.

        Powers the ``what changed?`` query family on the BTG.
        """

    # -- identity bonds -----------------------------------------------------

    async def bonds_for(
        self,
        identity_id: str,
        *,
        min_confidence: float = 0.0,
    ) -> list[IdentityBond]:
        """Return outgoing bonds from ``identity_id``."""

    # -- entity graph (v0.3) -----------------------------------------------

    async def upsert_entities(self, entities: "Sequence[Entity]") -> None:
        """Insert or update canonical entity rows. Idempotent on ``id``."""

    async def add_entity_mentions(self, mentions: "Sequence[EntityMention]") -> None:
        """Persist entity mentions for memories already in the store."""

    async def add_entity_edges(self, edges: "Sequence[EntityEdge]") -> None:
        """Insert or merge weighted entity edges (PK aggregates by ``relation``)."""

    async def entities_by_ids(self, ids: Sequence[str]) -> list[str]:
        """Return the subset of ``ids`` present in the entities table."""

    async def neighbors(
        self,
        entity_id: str,
        *,
        limit: int = 32,
    ) -> list[Any]:
        """Return outgoing edges from ``entity_id``, ordered by weight desc.

        Each element is the lightweight edge tuple expected by graph-walk
        (``to_entity``, ``relation``, ``weight``).
        """

    async def memories_mentioning(
        self,
        entity_ids: Sequence[str],
        *,
        identity_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[Memory, str, float]]:
        """Return ``(memory, entity_id, mention_confidence)`` triples."""
