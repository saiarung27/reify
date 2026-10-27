"""SQLite implementation of the :class:`reify.storage.base.Storage` protocol.

Design notes
------------
* **Plain SQL is truth.** The schema lives in ``_schema.sql`` so it can be
  read and modified without touching Python. Pydantic models in
  :mod:`reify.core.types` are a typed view over this schema.
* **One physical table for all five memory networks.** Common fields are
  columns; network-specific fields live in ``payload_json`` and are
  rehydrated by Pydantic on read. This is the v0 "two memory types"
  implementation: the type system has five networks, the storage layer
  has one table.
* **WAL journaling** for crash safety and concurrent readers.
* **Async surface, sync underneath.** SQLite is synchronous; we route
  each call through the store's single-worker executor (see
  :meth:`SQLiteStore._run_blocking`) and serialize writes with an
  ``asyncio.Lock`` so the public API can be awaited from anywhere without
  blocking the event loop, and the one connection is never used
  concurrently from two threads.
* **Vector search is brute-force cosine for v0.** sqlite-vec wiring lands
  when the volume justifies it. For the wedge demo the metadata-first
  recall path does not require vector search at all.
"""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import struct
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar, cast

_T = TypeVar("_T")

from pydantic import TypeAdapter, ValidationError

from reify.core.errors import SchemaError, StorageError, UnknownMemoryError
from reify.core.ids import MemoryId
from reify.core.types import (
    IdentityBond,
    Memory,
    Network,
    Transition,
)
from reify.storage.base import RecallFilter, Storage, WriteBatch


SCHEMA_VERSION = 2
"""Bumped whenever the SQL in ``_schema.sql`` changes in a breaking way."""

_MEMORY_ADAPTER: TypeAdapter[Memory] = TypeAdapter(Memory)
"""Round-trips rows back into the discriminated :data:`Memory` union."""

_COMMON_MEMORY_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "network",
        "content",
        "created_at",
        "valid_at",
        "valid_until",
        "confidence",
        "identity_id",
        "tags",
        "metadata",
        "provenance",
        "soft_deleted_at",
    }
)
"""Fields stored in dedicated columns (or, for ``provenance``, a separate
table). Everything *not* in this set is stashed in ``payload_json``."""


def _iso(dt: datetime | None) -> str | None:
    """Serialize a ``datetime`` to ISO-8601 string, or pass through ``None``."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _pack_vector(vec: Sequence[float]) -> bytes:
    """Pack a vector into a compact little-endian float32 blob."""
    return struct.pack(f"<{len(vec)}f", *vec)


def _unpack_vector(blob: bytes) -> list[float]:
    """Inverse of :func:`_pack_vector`."""
    count = len(blob) // 4
    return list(struct.unpack(f"<{count}f", blob))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity in ``[-1, 1]``. Returns 0 for zero-norm inputs."""
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


class SQLiteStore(Storage):
    """SQLite-backed storage. The default v0 storage backend.

    Use :meth:`init` once per process before any other call. Use
    :meth:`close` on shutdown. The store is safe to share across coroutines;
    writes are serialized via an internal lock so callers do not need to.

    Parameters
    ----------
    path:
        Filesystem path to the SQLite database. The file is created if it
        does not exist. Pass ``":memory:"`` for an in-memory store, which is
        what the test suite uses.
    """

    def __init__(self, path: Path | str) -> None:
        self._path = str(path)
        self._conn: sqlite3.Connection | None = None
        self._write_lock = asyncio.Lock()
        # Single-worker executor for ALL blocking SQLite work (see
        # _run_blocking). Created lazily; see close().
        self._executor: ThreadPoolExecutor | None = None

    # ------------------------------------------------------------------
    # Blocking-work executor: the store owns ONE sqlite3 connection with
    # check_same_thread=False. That flag permits cross-thread use but does
    # NOT make concurrent use safe: two coroutines awaiting storage at
    # once used to run their _do bodies on different default-executor
    # threads, and concurrent conn.execute calls raised
    # sqlite3.InterfaceError ("bad parameter or other API misuse"). Seen
    # live when profile generation ran parallel graph searches, silently
    # degrading personas to bare fallbacks. Routing every blocking call
    # through one single-worker executor means the connection is ever
    # touched by exactly one thread.
    # ------------------------------------------------------------------

    def _ensure_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="reify-store"
            )
        return self._executor

    async def _run_blocking(self, fn: Callable[[], _T]) -> _T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._ensure_executor(), fn)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def init(self) -> None:
        """Open the connection, apply the schema, record the version.

        Idempotent.
        """
        if self._conn is not None:
            return

        def _open() -> sqlite3.Connection:
            conn = sqlite3.connect(
                self._path,
                check_same_thread=False,
                isolation_level=None,  # autocommit; we manage transactions explicitly
            )
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            # User function: priority of an entity kind for upgrade-on-conflict.
            # Mirrors ``reify.retrieve.entities._KIND_PRIORITY`` so the SQL
            # path agrees with the Python path.
            _priority = {"topic": 1, "person": 2, "role": 3, "location": 4, "org": 5}
            conn.create_function(
                "_kind_priority",
                1,
                lambda k: _priority.get(k, 0),
                deterministic=True,
            )
            schema_sql = (
                resources.files("reify.storage").joinpath("_schema.sql").read_text("utf-8")
            )
            conn.executescript(schema_sql)
            # Record the schema version if this is a fresh database.
            conn.execute(
                "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (?, ?)",
                (SCHEMA_VERSION, _iso(datetime.now(timezone.utc))),
            )
            return conn

        self._conn = await self._run_blocking(_open)

    async def close(self) -> None:
        """Close the underlying connection. Idempotent."""
        conn, self._conn = self._conn, None
        if conn is not None:
            await self._run_blocking(conn.close)
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def commit_batch(self, batch: WriteBatch) -> None:
        """Atomically persist a :class:`WriteBatch`.

        Either every memory, transition, and bond in the batch lands or none
        do. A failure of any constraint surfaces as
        :class:`StorageError`.
        """
        conn = self._require_conn()

        def _do() -> None:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for mem in batch.memories:
                    self._insert_memory(conn, mem)
                for tr in batch.transitions:
                    self._insert_transition(conn, tr)
                for bond in batch.bonds:
                    self._upsert_bond(conn, bond)
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise StorageError(f"commit_batch failed: {exc}") from exc

        async with self._write_lock:
            await self._run_blocking(_do)

    async def attach_embedding(
        self,
        memory_id: MemoryId,
        vector: Sequence[float],
        *,
        model: str,
    ) -> None:
        """Attach (or replace) a vector embedding for a memory."""
        conn = self._require_conn()
        blob = _pack_vector(vector)
        now = _iso(datetime.now(timezone.utc))

        def _do() -> None:
            try:
                conn.execute(
                    """
                    INSERT INTO embeddings (memory_id, model, dim, vector, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(memory_id, model) DO UPDATE SET
                        dim        = excluded.dim,
                        vector     = excluded.vector,
                        created_at = excluded.created_at
                    """,
                    (memory_id, model, len(vector), blob, now),
                )
            except sqlite3.IntegrityError as exc:
                raise UnknownMemoryError(
                    f"cannot attach embedding to unknown memory {memory_id!r}"
                ) from exc

        async with self._write_lock:
            await self._run_blocking(_do)

    async def soft_delete(self, memory_id: MemoryId, *, at: datetime | None = None) -> None:
        """Tombstone a memory. Idempotent on subsequent calls."""
        conn = self._require_conn()
        when = _iso(at or datetime.now(timezone.utc))

        def _do() -> None:
            cur = conn.execute(
                "UPDATE memories SET soft_deleted_at = ? WHERE id = ? AND soft_deleted_at IS NULL",
                (when, memory_id),
            )
            if cur.rowcount == 0:
                # Either the memory doesn't exist or it's already tombstoned;
                # we treat both as success but verify existence to surface a
                # clear error in the unknown case.
                existing = conn.execute(
                    "SELECT 1 FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
                if existing is None:
                    raise UnknownMemoryError(memory_id)

        async with self._write_lock:
            await self._run_blocking(_do)

    async def bind_valid_until(self, memory_id: MemoryId, valid_until: datetime) -> None:
        """Bind a memory's validity end time (bitemporal)."""
        conn = self._require_conn()
        when = _iso(valid_until)

        def _do() -> None:
            cur = conn.execute(
                "UPDATE memories SET valid_until = ? WHERE id = ? AND valid_until IS NULL",
                (when, memory_id),
            )
            if cur.rowcount == 0:
                # Either the memory doesn't exist or valid_until is already set
                existing = conn.execute(
                    "SELECT 1 FROM memories WHERE id = ?", (memory_id,)
                ).fetchone()
                if existing is None:
                    raise UnknownMemoryError(f"Memory {memory_id} not found. Cannot bind valid_until to non-existent memory.")

        async with self._write_lock:
            await self._run_blocking(_do)

    async def update_metadata(self, memory_id: MemoryId, metadata: dict[str, Any]) -> None:
        """Replace a memory's metadata blob."""
        conn = self._require_conn()
        metadata_json = json.dumps(metadata, sort_keys=True)

        def _do() -> None:
            cur = conn.execute(
                "UPDATE memories SET metadata_json = ? WHERE id = ?",
                (metadata_json, memory_id),
            )
            if cur.rowcount == 0:
                raise UnknownMemoryError(memory_id)

        async with self._write_lock:
            await self._run_blocking(_do)

    async def soft_delete_matching(
        self,
        *,
        tag: str | None = None,
        before: datetime | None = None,
        identity_id: str | None = None,
        at: datetime | None = None,
    ) -> int:
        """Tombstone memories matching a filter and return the count."""
        conn = self._require_conn()
        when = _iso(at or datetime.now(timezone.utc))
        clauses = ["soft_deleted_at IS NULL"]
        params: list[Any] = []
        if tag is not None:
            clauses.append(
                "EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)"
            )
            params.append(tag)
        if before is not None:
            params.append(_iso(before))
            clauses.append("created_at < ?")
        if identity_id is not None:
            clauses.append("identity_id = ?")
            params.append(identity_id)
        where = " AND ".join(clauses)

        def _do() -> int:
            cur = conn.execute(
                f"UPDATE memories SET soft_deleted_at = ? WHERE {where}",
                [when, *params],
            )
            return int(cur.rowcount)

        async with self._write_lock:
            return await self._run_blocking(_do)

    async def sweep_expired(self, *, now: datetime | None = None) -> int:
        """Tombstone memories whose retention expiry is in the past."""
        conn = self._require_conn()
        when = _iso(now or datetime.now(timezone.utc))

        def _do() -> int:
            cur = conn.execute(
                """
                UPDATE memories
                SET soft_deleted_at = COALESCE(soft_deleted_at, ?)
                WHERE soft_deleted_at IS NULL
                  AND json_extract(metadata_json, '$.retention.expires_at') IS NOT NULL
                  AND json_extract(metadata_json, '$.retention.expires_at') < ?
                """,
                (when, when),
            )
            return int(cur.rowcount)

        async with self._write_lock:
            return await self._run_blocking(_do)

    async def expired_memory_ids(self, *, now: datetime | None = None) -> list[MemoryId]:
        """Return ids of memories whose retention expiry is in the past."""
        conn = self._require_conn()
        when = _iso(now or datetime.now(timezone.utc))

        def _do() -> list[MemoryId]:
            rows = conn.execute(
                """
                SELECT id
                FROM memories
                WHERE soft_deleted_at IS NULL
                  AND json_extract(metadata_json, '$.retention.expires_at') IS NOT NULL
                  AND json_extract(metadata_json, '$.retention.expires_at') < ?
                ORDER BY created_at ASC
                """,
                (when,),
            ).fetchall()
            return [cast(MemoryId, row["id"]) for row in rows]

        return await self._run_blocking(_do)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    async def get(self, memory_id: MemoryId) -> Memory | None:
        conn = self._require_conn()

        def _do() -> Memory | None:
            row = conn.execute(
                "SELECT * FROM memories WHERE id = ?", (memory_id,)
            ).fetchone()
            if row is None:
                return None
            provenance = self._load_provenance(conn, memory_id)
            return self._row_to_memory(row, provenance)

        return await self._run_blocking(_do)

    async def recall(self, filter_: RecallFilter, *, limit: int = 50) -> list[Memory]:
        """Metadata-first recall.

        Filters compose with conjunctive (AND) semantics, ordered by recency
        (newest ``created_at`` first). This is the fast structural recall
        path used by the metadata-first stages of the retrieval pipeline.
        Lexical (BM25), graph-walk, and vector stages live in dedicated
        retrieval modules.
        """
        conn = self._require_conn()
        clauses, params = self._build_recall_where(filter_)
        sql = f"SELECT * FROM memories WHERE {clauses} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        def _do() -> list[Memory]:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_memory(r, self._load_provenance(conn, r["id"])) for r in rows]

        return await self._run_blocking(_do)

    async def vector_search(
        self,
        vector: Sequence[float],
        *,
        networks: Sequence[Network] | None = None,
        identity_ids: Sequence[str] | None = None,
        valid_at: datetime | None = None,
        limit: int = 20,
    ) -> list[tuple[Memory, float]]:
        """Brute-force cosine search over stored embeddings.

        Adequate for v0 dataset sizes (tens of thousands of records). The
        sqlite-vec backed ANN index lands when the volume justifies it.
        Returns ``(memory, distance)`` pairs ordered by ascending distance,
        where ``distance = 1 - cosine_similarity``.
        """
        conn = self._require_conn()
        clauses, params = self._build_recall_where(
            RecallFilter(networks=networks, identity_ids=identity_ids, valid_at=valid_at)
        )

        def _do() -> list[tuple[Memory, float]]:
            sql = f"""
                SELECT m.*, e.vector AS _vec
                FROM memories m
                INNER JOIN embeddings e ON e.memory_id = m.id
                WHERE {clauses}
            """
            rows = conn.execute(sql, params).fetchall()
            scored: list[tuple[Memory, float]] = []
            for r in rows:
                v = _unpack_vector(r["_vec"])
                if len(v) != len(vector):
                    continue  # different embedder; skip rather than fail
                dist = 1.0 - _cosine(vector, v)
                mem = self._row_to_memory(r, self._load_provenance(conn, r["id"]))
                scored.append((mem, dist))
            scored.sort(key=lambda pair: pair[1])
            return scored[:limit]

        return await self._run_blocking(_do)

    # ------------------------------------------------------------------
    # Transitions
    # ------------------------------------------------------------------

    async def transitions_for(
        self,
        *,
        memory_ids: Iterable[MemoryId] | None = None,
        between: tuple[datetime, datetime] | None = None,
        identity_id: str | None = None,
    ) -> list[Transition]:
        conn = self._require_conn()
        clauses: list[str] = []
        params: list[Any] = []
        ids = list(memory_ids) if memory_ids is not None else None
        if ids:
            placeholders = ",".join("?" for _ in ids)
            clauses.append(f"(from_memory_id IN ({placeholders}) OR to_memory_id IN ({placeholders}))")
            params.extend(ids)
            params.extend(ids)
        if between is not None:
            start, end = between
            clauses.append("at >= ? AND at <= ?")
            params.extend([cast(Any, _iso(start)), cast(Any, _iso(end))])
        if identity_id is not None:
            clauses.append(
                """
                (
                    identity_id = ?
                    OR from_memory_id IN (
                        SELECT id FROM memories WHERE identity_id = ?
                    )
                    OR to_memory_id IN (
                        SELECT id FROM memories WHERE identity_id = ?
                    )
                )
                """
            )
            params.extend([identity_id, identity_id, identity_id])
        where = " AND ".join(clauses) if clauses else "1=1"
        sql = f"SELECT * FROM transitions WHERE {where} ORDER BY at DESC"

        def _do() -> list[Transition]:
            rows = conn.execute(sql, params).fetchall()
            return [self._row_to_transition(r) for r in rows]

        return await self._run_blocking(_do)

    # ------------------------------------------------------------------
    # Identity bonds
    # ------------------------------------------------------------------

    async def bonds_for(
        self,
        identity_id: str,
        *,
        min_confidence: float = 0.0,
    ) -> list[IdentityBond]:
        conn = self._require_conn()

        def _do() -> list[IdentityBond]:
            rows = conn.execute(
                """
                SELECT * FROM identity_bonds
                WHERE from_identity = ? AND confidence >= ?
                ORDER BY confidence DESC
                """,
                (identity_id, min_confidence),
            ).fetchall()
            return [
                IdentityBond(
                    from_identity=r["from_identity"],
                    to_identity=r["to_identity"],
                    confidence=r["confidence"],
                    rationale=r["rationale"],
                    formed_at=datetime.fromisoformat(r["formed_at"]),
                    last_reinforced_at=(
                        datetime.fromisoformat(r["last_reinforced_at"])
                        if r["last_reinforced_at"]
                        else None
                    ),
                )
                for r in rows
            ]

        return await self._run_blocking(_do)

    # ------------------------------------------------------------------
    # Entity graph (v0.3)
    # ------------------------------------------------------------------

    async def upsert_entities(self, entities: Sequence[Any]) -> None:
        """Insert or update canonical entity rows."""
        if not entities:
            return
        conn = self._require_conn()
        now = _iso(datetime.now(timezone.utc))

        def _do() -> None:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for ent in entities:
                    # Kind upgrade on conflict: priority order encoded as
                    # a CASE so a later "org" classification beats an
                    # earlier "person" guess for the same surface.
                    conn.execute(
                        """
                        INSERT INTO entities (id, label, kind, created_at, metadata_json)
                        VALUES (?, ?, ?, ?, '{}')
                        ON CONFLICT(id) DO UPDATE SET
                            label = excluded.label,
                            kind = CASE
                                WHEN _kind_priority(excluded.kind) > _kind_priority(kind)
                                    THEN excluded.kind
                                ELSE kind
                            END
                        """,
                        (ent.id, ent.label, ent.kind, now),
                    )
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise StorageError(f"upsert_entities failed: {exc}") from exc

        async with self._write_lock:
            await self._run_blocking(_do)

    async def add_entity_mentions(self, mentions: Sequence[Any]) -> None:
        """Persist entity mentions. Existing (memory, entity, span) rows are
        replaced so re-extraction of the same memory is idempotent."""
        if not mentions:
            return
        conn = self._require_conn()

        def _do() -> None:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for m in mentions:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO entity_mentions
                            (memory_id, entity_id, span_start, span_end,
                             surface, confidence, role)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            m.memory_id,
                            m.entity_id,
                            m.span_start,
                            m.span_end,
                            m.surface,
                            m.confidence,
                            m.role,
                        ),
                    )
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise StorageError(f"add_entity_mentions failed: {exc}") from exc

        async with self._write_lock:
            await self._run_blocking(_do)

    async def add_entity_edges(self, edges: Sequence[Any]) -> None:
        """Insert or merge weighted edges. Re-observed edges have their
        weight aggregated (max of existing vs new) and ``last_seen_at``
        bumped forward."""
        if not edges:
            return
        conn = self._require_conn()

        def _do() -> None:
            try:
                conn.execute("BEGIN IMMEDIATE")
                for e in edges:
                    conn.execute(
                        """
                        INSERT INTO entity_edges
                            (from_entity, to_entity, relation, weight, last_seen_at)
                        VALUES (?, ?, ?, ?, ?)
                        ON CONFLICT(from_entity, to_entity, relation) DO UPDATE SET
                            weight       = MAX(weight, excluded.weight),
                            last_seen_at = MAX(last_seen_at, excluded.last_seen_at)
                        """,
                        (
                            e.from_entity,
                            e.to_entity,
                            e.relation,
                            e.weight,
                            _iso(e.last_seen_at),
                        ),
                    )
                conn.execute("COMMIT")
            except sqlite3.Error as exc:
                conn.execute("ROLLBACK")
                raise StorageError(f"add_entity_edges failed: {exc}") from exc

        async with self._write_lock:
            await self._run_blocking(_do)

    async def entities_by_ids(self, ids: Sequence[str]) -> list[str]:
        if not ids:
            return []
        conn = self._require_conn()
        placeholders = ",".join("?" for _ in ids)

        def _do() -> list[str]:
            rows = conn.execute(
                f"SELECT id FROM entities WHERE id IN ({placeholders})",
                list(ids),
            ).fetchall()
            return [r["id"] for r in rows]

        return await self._run_blocking(_do)

    async def neighbors(self, entity_id: str, *, limit: int = 32) -> list[Any]:
        from reify.retrieve.graph import _Edge  # local to avoid circular at import time
        conn = self._require_conn()

        def _do() -> list[Any]:
            rows = conn.execute(
                """
                SELECT to_entity, relation, weight
                FROM entity_edges
                WHERE from_entity = ?
                ORDER BY weight DESC, last_seen_at DESC
                LIMIT ?
                """,
                (entity_id, limit),
            ).fetchall()
            return [
                _Edge(
                    to_entity=r["to_entity"],
                    relation=r["relation"],
                    weight=float(r["weight"]),
                )
                for r in rows
            ]

        return await self._run_blocking(_do)

    async def memories_mentioning(
        self,
        entity_ids: Sequence[str],
        *,
        identity_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[Memory, str, float]]:
        if not entity_ids:
            return []
        conn = self._require_conn()
        placeholders = ",".join("?" for _ in entity_ids)
        clauses: list[str] = [
            "m.soft_deleted_at IS NULL",
            f"em.entity_id IN ({placeholders})",
        ]
        params: list[Any] = list(entity_ids)
        if identity_id is not None:
            clauses.append("m.identity_id = ?")
            params.append(identity_id)
        if valid_at is not None:
            iso = _iso(valid_at)
            clauses.append("m.valid_at <= ? AND (m.valid_until IS NULL OR m.valid_until > ?)")
            params.extend([iso, iso])
        where = " AND ".join(clauses)
        sql = f"""
            SELECT m.*, em.entity_id AS _entity_id, em.confidence AS _conf
            FROM memories m
            INNER JOIN entity_mentions em ON em.memory_id = m.id
            WHERE {where}
            ORDER BY m.created_at DESC
            LIMIT ?
        """
        params.append(limit)

        def _do() -> list[tuple[Memory, str, float]]:
            rows = conn.execute(sql, params).fetchall()
            out: list[tuple[Memory, str, float]] = []
            for r in rows:
                mem = self._row_to_memory(r, self._load_provenance(conn, r["id"]))
                out.append((mem, r["_entity_id"], float(r["_conf"])))
            return out

        return await self._run_blocking(_do)

    async def mention_spans(
        self, entity_ids: Sequence[str]
    ) -> list[tuple[str, str, int, int]]:
        """Mention spans per entity, definitional (oldest) first.

        Returns (entity_id, memory_id, span_start, span_end) rows. The
        adapter uses the first span per entity to build a sentence-level
        node description from the memory that introduces the entity.
        """
        if not entity_ids:
            return []
        conn = self._require_conn()
        placeholders = ",".join("?" for _ in entity_ids)
        sql = f"""
            SELECT em.entity_id AS _entity_id, em.memory_id AS _memory_id,
                   em.span_start AS _start, em.span_end AS _end
            FROM entity_mentions em
            INNER JOIN memories m ON m.id = em.memory_id
            WHERE em.entity_id IN ({placeholders})
              AND m.soft_deleted_at IS NULL
            ORDER BY m.created_at ASC, em.span_start ASC
        """

        def _do() -> list[tuple[str, str, int, int]]:
            rows = conn.execute(sql, list(entity_ids)).fetchall()
            return [
                (r["_entity_id"], r["_memory_id"], int(r["_start"]), int(r["_end"]))
                for r in rows
            ]

        return await self._run_blocking(_do)

    async def list_entities(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        """All entities in the store, oldest first. Graph introspection."""
        conn = self._require_conn()

        def _do() -> list[dict[str, Any]]:
            rows = conn.execute(
                "SELECT id, label, kind FROM entities ORDER BY created_at LIMIT ?",
                (limit,),
            ).fetchall()
            return [
                {"id": r["id"], "label": r["label"], "kind": r["kind"]} for r in rows
            ]

        return await self._run_blocking(_do)

    async def list_edges(self, *, limit: int = 1000) -> list[dict[str, Any]]:
        """All entity edges, strongest first. Graph introspection."""
        conn = self._require_conn()

        def _do() -> list[dict[str, Any]]:
            rows = conn.execute(
                """
                SELECT from_entity, to_entity, relation, weight, last_seen_at
                FROM entity_edges
                ORDER BY weight DESC, last_seen_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [
                {
                    "from_entity": r["from_entity"],
                    "to_entity": r["to_entity"],
                    "relation": r["relation"],
                    "weight": float(r["weight"]),
                    "last_seen_at": r["last_seen_at"],
                }
                for r in rows
            ]

        return await self._run_blocking(_do)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _require_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise StorageError("SQLiteStore.init() must be awaited before use")
        return self._conn

    def _insert_memory(self, conn: sqlite3.Connection, m: Memory) -> None:
        dump = m.model_dump(mode="json")
        payload = {k: v for k, v in dump.items() if k not in _COMMON_MEMORY_FIELDS}
        conn.execute(
            """
            INSERT OR REPLACE INTO memories
                (id, network, content, created_at, valid_at, valid_until,
                 confidence, identity_id, tags_json, metadata_json,
                 payload_json, soft_deleted_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                m.id,
                m.network.value,
                m.content,
                _iso(m.created_at),
                _iso(m.valid_at),
                _iso(m.valid_until),
                m.confidence,
                m.identity_id,
                json.dumps(list(m.tags)),
                json.dumps(m.metadata),
                json.dumps(payload),
                _iso(m.soft_deleted_at),
            ),
        )
        # Provenance is append-only; clear and rewrite for an OR-REPLACE write.
        conn.execute("DELETE FROM provenance WHERE memory_id = ?", (m.id,))
        for p in m.provenance:
            conn.execute(
                """
                INSERT INTO provenance
                    (memory_id, source, session_id, actor, raw_event_id, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    m.id,
                    p.source,
                    p.session_id,
                    p.actor,
                    p.raw_event_id,
                    _iso(p.recorded_at),
                ),
            )

    def _insert_transition(self, conn: sqlite3.Connection, t: Transition) -> None:
        conn.execute(
            """
            INSERT OR REPLACE INTO transitions
                (id, kind, from_memory_id, to_memory_id, at, recorded_at,
                 confidence, summary, identity_id, evidence_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                t.id,
                t.kind.value,
                t.from_memory_id,
                t.to_memory_id,
                _iso(t.at),
                _iso(t.recorded_at),
                t.confidence,
                t.summary,
                t.identity_id,
                json.dumps(list(t.evidence)),
            ),
        )

    def _upsert_bond(self, conn: sqlite3.Connection, b: IdentityBond) -> None:
        conn.execute(
            """
            INSERT INTO identity_bonds
                (from_identity, to_identity, confidence, rationale,
                 formed_at, last_reinforced_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(from_identity, to_identity) DO UPDATE SET
                confidence         = excluded.confidence,
                rationale          = excluded.rationale,
                last_reinforced_at = excluded.last_reinforced_at
            """,
            (
                b.from_identity,
                b.to_identity,
                b.confidence,
                b.rationale,
                _iso(b.formed_at),
                _iso(b.last_reinforced_at),
            ),
        )

    def _load_provenance(self, conn: sqlite3.Connection, memory_id: str) -> list[dict[str, Any]]:
        rows = conn.execute(
            """
            SELECT source, session_id, actor, raw_event_id, recorded_at
            FROM provenance WHERE memory_id = ?
            ORDER BY rowid ASC
            """,
            (memory_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def _row_to_memory(self, row: sqlite3.Row, provenance: list[dict[str, Any]]) -> Memory:
        try:
            payload: dict[str, Any] = json.loads(row["payload_json"])
        except (TypeError, ValueError) as exc:
            raise StorageError(f"corrupt payload_json for memory {row['id']}: {exc}") from exc

        payload.update(
            {
                "id": row["id"],
                "network": row["network"],
                "content": row["content"],
                "created_at": row["created_at"],
                "valid_at": row["valid_at"],
                "valid_until": row["valid_until"],
                "confidence": row["confidence"],
                "identity_id": row["identity_id"],
                "tags": json.loads(row["tags_json"]),
                "metadata": json.loads(row["metadata_json"]),
                "provenance": provenance,
                "soft_deleted_at": row["soft_deleted_at"],
            }
        )
        try:
            return _MEMORY_ADAPTER.validate_python(payload)
        except ValidationError as exc:
            # A schema violation in storage is a bug, not a user error.
            raise SchemaError(f"row for memory {row['id']} failed validation: {exc}") from exc

    def _row_to_transition(self, row: sqlite3.Row) -> Transition:
        return Transition(
            id=row["id"],
            kind=row["kind"],
            from_memory_id=row["from_memory_id"],
            to_memory_id=row["to_memory_id"],
            at=datetime.fromisoformat(row["at"]),
            recorded_at=datetime.fromisoformat(row["recorded_at"]),
            confidence=row["confidence"],
            summary=row["summary"],
            identity_id=row["identity_id"],
            evidence=tuple(json.loads(row["evidence_json"])),
        )

    @staticmethod
    def _build_recall_where(f: RecallFilter) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if not f.include_deleted:
            clauses.append("soft_deleted_at IS NULL")
        if f.networks:
            placeholders = ",".join("?" for _ in f.networks)
            clauses.append(f"network IN ({placeholders})")
            params.extend(n.value for n in f.networks)
        if f.identity_ids:
            placeholders = ",".join("?" for _ in f.identity_ids)
            clauses.append(f"identity_id IN ({placeholders})")
            params.extend(f.identity_ids)
        if f.tags_any:
            # tags_json is a JSON array. EXISTS over json_each is the cheap
            # SQLite-native idiom.
            tag_clauses = []
            for tag in f.tags_any:
                tag_clauses.append(
                    "EXISTS (SELECT 1 FROM json_each(tags_json) WHERE value = ?)"
                )
                params.append(tag)
            clauses.append("(" + " OR ".join(tag_clauses) + ")")
        if f.valid_at is not None:
            clauses.append(
                "valid_at <= ? AND (valid_until IS NULL OR valid_until > ?)"
            )
            iso = _iso(f.valid_at)
            params.extend([iso, iso])
        if f.since is not None:
            clauses.append("created_at >= ?")
            params.append(_iso(f.since))
        return (" AND ".join(clauses) if clauses else "1=1"), params
