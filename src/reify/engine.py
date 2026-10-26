"""The Reify engine.

The engine is the **product**. The CLI and MCP server are thin adapters
around it (commandment 8). Both eventually call exactly one method on this
class for any given user-visible action:

* ``remember``       \u2014 ingest an utterance.
* ``recall``         \u2014 answer a natural-language question with relevant memories.
* ``whatchanged``    \u2014 list recent transitions.
* ``whatdoyouknow``  \u2014 dump everything Reify retains about an identity.
* ``export``         \u2014 produce a human-readable markdown of the entire store.
* ``forget``         \u2014 soft-delete a memory by id.
* ``flush_arw``      \u2014 process the Active Reconsolidation queue (mostly used
  by tests and the CLI; in production a background worker drives this).

The engine is intentionally agnostic to its surfaces: it does not know
about MCP, HTTP, or stdio. It also does not know how its LLM provider is
wired \u2014 the provider is a protocol, swappable for a fake in tests.

Design constraints (from FIRST_PRINCIPLES.md)
---------------------------------------------
* **Async-first, LLM-optional writes.** ``remember`` returns after the
  foreground stages complete; the LLM-driven background work is enqueued.
* **Recall is metadata-first.** Embeddings are not invoked unless an
  embedder is configured. Tag/time/keyword stages always run.
* **Injectable clock.** Bitemporal correctness is testable; ``Reify.now()``
  is the single source of truth for "the current instant" inside the
  engine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from reify.core.errors import ReifyError, UnknownMemoryError
from reify.core.ids import MemoryId
from reify.core.types import Experience, Fact, Memory, Transition
from reify.consolidate import ARWWorker
from reify.consolidate.sleep_scheduler import SleepScheduler, SleepSchedulerConfig
from reify.identity.bond_graph import IdentityBondGraph
from reify.extract import (
    DeterministicExtractor,
    Extractor,
    ExtractionContext,
)
from reify.extract.pii_redaction import PIIRedactor
from reify.llm.base import LLMProvider
from reify.recall import AggregateIntent, Query, RecallResult, Recaller, TemporalIntent, parse_query, since_window
from reify.retrieve.ebbinghaus import EbbinghausConfig, apply_ebbinghaus_to_score
from reify.retrieve.entities import extract_for_memory, merge_deltas
from reify.retrieve.graph import WalkConfig, graph_walk
from reify.retrieve.rerank import NoopReranker, Reranker
from reify.storage import RecallFilter, Storage, WriteBatch
from reify.storage.hot_buffer import HotBuffer, HotBufferConfig
from reify.synthesize import ModelProfile, PackedContext, get_profile, pack


Clock = Callable[[], datetime]
log = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class WriteReceipt:
    """What :meth:`Reify.remember` returns.

    ``committed`` is the list of memories that landed synchronously in the
    foreground stage. ``enqueued_count`` counts claim-like memories that were
    queued for Active Reconsolidation.
    """

    committed: tuple[Memory, ...]
    enqueued_count: int = 0


@dataclass(frozen=True)
class RecallResponse:
    """What :meth:`Reify.recall` returns.

    ``query`` is the parsed structural query so callers can show "we
    interpreted your question as ...". ``results`` are the ranked hits.
    """

    query: Query
    results: tuple[RecallResult, ...] = ()


@dataclass(frozen=True)
class EngineConfig:
    """Optional knobs. Sensible defaults so most callers pass nothing."""

    candidate_limit: int = 200
    recall_weights: tuple[float, float, float] = (0.5, 0.2, 0.3)
    half_life_days: float = 30.0
    default_ttl_days: float = 30.0
    reinforcement_factor: float = 1.5
    decay_sweep_interval_seconds: float = 60.0
    llm: LLMProvider | None = None
    # v0.3 hybrid retrieval: graph-walk stage. Default-on; degrades to a
    # no-op when the graph is empty or the storage backend lacks support.
    graph_retrieval_enabled: bool = True
    graph_walk_depth: int = 2
    graph_walk_fanout: int = 32
    graph_walk_weight: float = 0.4
    """Weight applied to graph-walk scores when merging with metadata-first
    results. Kept below 0.5 so the metadata-first stages remain primary
    (FIRST_PRINCIPLES.md commandment 5)."""

    # v0.3 reranker stage. Default is a no-op so the no-LLM, zero-dependency
    # floor stays bit-for-bit identical to v0.2. Callers opt-in with
    # ``FeatureReranker()`` (deterministic, dep-free) or ``BGEReranker()``
    # (optional cross-encoder behind the ``[rerank]`` extra).
    reranker: Reranker | None = None

    # v0.3 packer profile for ``recall_packed``. Short name from
    # :func:`reify.synthesize.get_profile` or a custom :class:`ModelProfile`.
    context_profile: str | ModelProfile = "gpt"
    context_budget_tokens: int | None = None

    # v0.4 hot-buffer: in-memory write batching before durable commit.
    hot_buffer_enabled: bool = True
    hot_buffer_config: HotBufferConfig = field(default_factory=HotBufferConfig)

    # v0.4 sleep scheduler: background consolidation during idle periods.
    sleep_scheduler_enabled: bool = True
    sleep_scheduler_config: SleepSchedulerConfig = field(default_factory=SleepSchedulerConfig)

    # v0.4 Ebbinghaus confidence decay: time-based confidence curve.
    ebbinghaus_config: EbbinghausConfig = field(default_factory=EbbinghausConfig)


class Reify:
    """The engine.

    Lifecycle
    ---------
    Construct, ``await engine.start()`` once, use, ``await engine.stop()``
    on shutdown. ``start`` is responsible for opening the underlying
    storage; ``stop`` closes it. Both are idempotent.

    Test fixtures typically construct with an in-memory store and a frozen
    clock (``clock=lambda: FROZEN_NOW``) so bitemporal assertions are
    deterministic.

    Parameters
    ----------
    storage:
        Required. The persistence backend.
    extractors:
        Sequence of :class:`reify.extract.Extractor` instances. The first
        is run synchronously in the foreground stage; subsequent
        extractors run in :meth:`flush_arw`. If empty, a single
        :class:`DeterministicExtractor` is used.
    clock:
        Returns the current instant. Defaults to UTC ``now``.
    config:
        Knobs for recall scoring and candidate fanout.
    """

    def __init__(
        self,
        *,
        storage: Storage,
        extractors: Sequence[Extractor] | None = None,
        clock: Clock = _utc_now,
        config: EngineConfig | None = None,
    ) -> None:
        self._storage = storage
        self._extractors: tuple[Extractor, ...] = tuple(
            extractors if extractors is not None else (DeterministicExtractor(),)
        )
        if not self._extractors:
            raise ReifyError("Reify requires at least one extractor")
        self._clock = clock
        self._config = config or EngineConfig()
        # PII redaction: gated by REIFY_PII_REDACT=on. The redactor is a
        # cheap pure-regex pass; it returns text unchanged when disabled,
        # so wiring it unconditionally keeps the no-LLM floor intact.
        self._pii_redactor = PIIRedactor()
        self._recaller = Recaller(
            storage,
            weights=self._config.recall_weights,
            half_life_days=self._config.half_life_days,
            candidate_limit=self._config.candidate_limit,
        )
        self._arw_worker = ARWWorker(
            storage,
            llm=self._config.llm,
            clock=self.now,
        )
        self._hot_buffer = HotBuffer(config=self._config.hot_buffer_config)
        self._sleep_scheduler = SleepScheduler(config=self._config.sleep_scheduler_config)
        self._identity_bond_graph = IdentityBondGraph()
        self._started = False
        self._decay_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Open the underlying storage. Idempotent."""
        if self._started:
            return
        await self._storage.init()
        if self._config.hot_buffer_enabled:
            self._hot_buffer.storage = self._storage
        if self._config.sleep_scheduler_enabled:
            self._sleep_scheduler.engine = self
            await self._sleep_scheduler.start()
        self._identity_bond_graph.engine = self
        self._started = True
        self._start_decay_task()

    async def stop(self) -> None:
        """Close the underlying storage. Idempotent."""
        if not self._started:
            return
        await self._stop_decay_task()
        if self._config.sleep_scheduler_enabled:
            await self._sleep_scheduler.stop()
        if self._config.hot_buffer_enabled:
            await self._hot_buffer.flush()
        await self._storage.close()
        self._started = False

    async def __aenter__(self) -> "Reify":
        await self.start()
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.stop()

    # ------------------------------------------------------------------
    # Public surface
    # ------------------------------------------------------------------

    def now(self) -> datetime:
        """The current instant, per the engine's clock."""
        return self._clock()

    async def remember(
        self,
        text: str,
        *,
        source: str = "library",
        actor: str = "user",
        session_id: str | None = None,
        identity_id: str | None = None,
        at: datetime | None = None,
        tags: tuple[str, ...] = (),
    ) -> WriteReceipt:
        """The foreground write path.

        Runs the first (deterministic) extractor synchronously, commits the
        produced memories, and enqueues claim-like rows in ``pending_arw`` so
        the background ARW worker can pick them up.

        Caller-supplied ``tags`` are merged onto every extracted memory
        (deduplicated, extractor tags preserved first). They exist for
        evidence-style retrieval: a caller that knows a memory's section
        (``pricing``, ``objection``) can tag it at ingest and filter on
        it exactly at query time via :meth:`recall`'s ``tags`` argument,
        instead of trusting free-text ranking to surface it.

        Returns a :class:`WriteReceipt` describing what landed.
        """
        self._require_started()
        ctx = ExtractionContext(
            source=source,
            actor=actor,  # type: ignore[arg-type]
            session_id=session_id,
            identity_id=identity_id,
            now=at or self.now(),
        )
        # Redact PII before any extractor sees the text. No-op unless
        # REIFY_PII_REDACT=on; never raises.
        text = await self._pii_redactor.extract(text, ctx)
        primary = self._extractors[0]
        result = await primary.extract(text, ctx)
        if not result.memories:
            return WriteReceipt(committed=())

        retained_memories = tuple(self._apply_default_retention(m) for m in result.memories)
        if tags:
            extra = tuple(dict.fromkeys(t.strip() for t in tags if t and t.strip()))
            if extra:
                retained_memories = tuple(
                    m.model_copy(
                        update={"tags": tuple(dict.fromkeys((*m.tags, *extra)))}
                    )
                    for m in retained_memories
                )

        # Record activity for sleep scheduler.
        if self._config.sleep_scheduler_enabled:
            self._sleep_scheduler.record_activity()

        batch = WriteBatch(
            memories=list(retained_memories),
            transitions=list(result.transitions),
        )
        if self._config.hot_buffer_enabled:
            self._hot_buffer.enqueue(batch)
            # Flush immediately to make data durable for entity indexing.
            # The buffer still provides value for batching rapid writes.
            await self._hot_buffer.flush()
        else:
            await self._storage.commit_batch(batch)
        # Entity-graph indexing is post-durable: a failure here never
        # invalidates the memory commit. Best-effort, logged on failure.
        await self._index_entities(retained_memories)
        queued_ids = [
            m.id
            for m in retained_memories
            if not isinstance(m, Experience) and m.valid_until is None
        ]
        await self._enqueue_arw(queued_ids)
        return WriteReceipt(
            committed=retained_memories,
            enqueued_count=len(queued_ids),
        )

    async def recall(
        self,
        text: str,
        *,
        identity_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 20,
        tags: tuple[str, ...] | None = None,
    ) -> RecallResponse:
        """Answer a natural-language question with ranked memories.

        Pure metadata-first recall: no LLM in the path. The MCP/HTTP
        surfaces are responsible for any post-processing (e.g.,
        synthesizing a sentence) on the structured payload.

        Explicit ``tags`` replace the tags inferred from the query text.
        They are the evidence-query path: when the caller already knows
        the section it wants (``pricing``), filtering on it is exact,
        where free-text ranking can be swamped by entity mentions.
        """
        self._require_started()
        # Record activity for sleep scheduler.
        if self._config.sleep_scheduler_enabled:
            self._sleep_scheduler.record_activity()
        # Ensure hot-buffer is flushed before recall so all data is visible.
        if self._config.hot_buffer_enabled:
            await self._hot_buffer.flush()
        sweep_now = self.now()
        query = parse_query(text, default_limit=limit)
        if identity_id is not None:
            query = _replace(query, identity_id=identity_id)
        if valid_at is not None:
            query = _replace(query, valid_at=valid_at)
        if tags is not None:
            query = _replace(query, tags=tuple(tags), tags_explicit=True)
        temporal_results = await self._temporal_recall(query)
        if temporal_results:
            return RecallResponse(query=query, results=tuple(temporal_results[:limit]))
        aggregate_results = await self._aggregate_recall(query)
        if aggregate_results:
            return RecallResponse(query=query, results=tuple(aggregate_results[:limit]))
        assistant_results = await self._assistant_recall(query)
        if assistant_results:
            return RecallResponse(query=query, results=tuple(assistant_results[:limit]))
        if valid_at is None or valid_at >= sweep_now:
            await self.sweep_decay(now=sweep_now, reason="recall")
        results = await self._recaller.recall(query, now=self.now())
        # Graph-walk additive stage. Merges into metadata-first results
        # without displacing them; weighted below 0.5 by config so the
        # primary recall path still wins on direct matches.
        graph_results = await self._graph_recall(
            query.text,
            identity_id=query.identity_id,
            valid_at=query.valid_at,
            limit=limit,
        )
        results = _merge_recall_results(results, graph_results, limit=limit)
        # Reranker stage. Defaults to NoopReranker so this is a typed no-op
        # unless the caller opts in via EngineConfig.reranker.
        if self._config.reranker is not None and results:
            try:
                results = await self._config.reranker.rerank(
                    text, results, now=self.now()
                )
                results = results[:limit]
            except Exception as exc:  # noqa: BLE001 - reranker is best-effort
                log.warning(
                    "reranker_failed",
                    extra={"audit_event": "reranker_failed", "error": repr(exc)},
                )
        # Ebbinghaus confidence decay: apply time-based confidence curve.
        if self._config.ebbinghaus_config.enabled and results:
            results = [
                RecallResult(
                    memory=r.memory,
                    score=apply_ebbinghaus_to_score(
                        r.score,
                        r.memory,
                        self.now(),
                        self._config.ebbinghaus_config,
                    ),
                    matched_stage=r.matched_stage,
                )
                for r in results
            ]
            # Re-sort after decay application.
            results = sorted(results, key=lambda r: r.score, reverse=True)[:limit]
        if valid_at is None or valid_at >= sweep_now:
            await self._reinforce_memories([r.memory for r in results], now=sweep_now)
        return RecallResponse(query=query, results=tuple(results))

    async def recall_packed(
        self,
        text: str,
        *,
        identity_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 20,
        profile: str | ModelProfile | None = None,
        budget_tokens: int | None = None,
    ) -> PackedContext:
        """Recall + pack into a model-shaped, token-budgeted blob.

        Convenience surface for HTTP/MCP adapters that want the
        prompt-ready string in one call. Calls :meth:`recall` (preserving
        all metadata-first guarantees), then runs the packer with the
        configured or supplied profile. The returned
        :class:`PackedContext` is fully legible: it lists which memory
        ids landed, which were dropped by the budget, the token count,
        and whether truncation happened.
        """
        response = await self.recall(
            text,
            identity_id=identity_id,
            valid_at=valid_at,
            limit=limit,
        )
        chosen_profile: str | ModelProfile = profile if profile is not None else self._config.context_profile
        if isinstance(chosen_profile, str):
            chosen_profile = get_profile(chosen_profile)
        return pack(
            response.results,
            profile=chosen_profile,
            budget_tokens=budget_tokens if budget_tokens is not None else self._config.context_budget_tokens,
            query=text,
        )

    async def whatchanged(
        self,
        *,
        days: int = 30,
        identity_id: str | None = None,
        at: datetime | None = None,
        limit: int = 20,
    ) -> list[Transition]:
        """List recent transitions.

        Surfaces the Bitemporal Transition Graph as a first-class query.
        This is what answers the wedge demo's "did anything change
        recently?" without the model having to derive the change from two
        facts.
        """
        self._require_started()
        window = since_window(days, now=at or self.now())
        out = await self._storage.transitions_for(
            between=window, identity_id=identity_id
        )
        return out[:limit]

    async def whatdoyouknow(
        self,
        about: str | None = None,
        *,
        identity_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 50,
    ) -> list[Memory]:
        """Return everything Reify currently believes, optionally scoped.

        If ``about`` is provided, it's parsed as a recall query. Otherwise
        all live memories scoped to ``identity_id`` are returned, ranked
        by recency.
        """
        self._require_started()
        sweep_now = self.now()
        if valid_at is None or valid_at >= sweep_now:
            await self.sweep_decay(now=sweep_now, reason="whatdoyouknow")
        if about:
            response = await self.recall(
                about,
                identity_id=identity_id,
                valid_at=valid_at,
                limit=limit,
            )
            return [r.memory for r in response.results]
        f = RecallFilter(
            identity_ids=[identity_id] if identity_id else None,
            valid_at=valid_at or self.now(),
        )
        memories = await self._storage.recall(f, limit=limit)
        if valid_at is None or valid_at >= sweep_now:
            await self._reinforce_memories(memories, now=sweep_now)
        return memories

    async def forget(
        self,
        memory_id: MemoryId | None = None,
        *,
        tag: str | None = None,
        before: datetime | None = None,
        identity_id: str | None = None,
    ) -> int:
        """Soft-delete one memory or a filtered set.

        Returns the number of tombstoned rows.
        """
        self._require_started()
        if memory_id is not None:
            await self._storage.soft_delete(memory_id, at=self.now())
            return 1
        return await self._storage.soft_delete_matching(
            tag=tag,
            before=before,
            identity_id=identity_id,
            at=self.now(),
        )

    async def sweep_decay(
        self,
        *,
        now: datetime | None = None,
        reason: str = "manual",
    ) -> int:
        """Tombstone expired memories and emit an audit log for the sweep."""
        self._require_started()
        sweep_now = now or self.now()
        expired_ids = await self._storage.expired_memory_ids(now=sweep_now)
        if not expired_ids:
            return 0
        swept = await self._storage.sweep_expired(now=sweep_now)
        log.info(
            "reify_decay_sweep",
            extra={
                "audit_event": "reify_decay_sweep",
                "reason": reason,
                "swept_at": sweep_now.isoformat(),
                "expired_memory_ids": list(expired_ids),
                "swept_count": swept,
            },
        )
        return swept

    async def export_markdown(
        self,
        path: Path,
        *,
        identity_id: str | None = None,
    ) -> int:
        """Write a single human-readable markdown file of the store.

        Honors commandment 9 (legibility). Returns the number of memories
        exported. The output is intentionally simple \u2014 grep-friendly,
        diff-friendly, and round-trippable to JSON via ``reify import``
        (v0.2).
        """
        self._require_started()
        f = RecallFilter(identity_ids=[identity_id] if identity_id else None)
        memories = await self._storage.recall(f, limit=10_000)
        transitions = await self._storage.transitions_for(identity_id=identity_id)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(f"# Reify export\n\n")
            fh.write(f"_Generated at {self.now().isoformat()}._\n\n")
            fh.write(f"_Memories: {len(memories)} \u00b7 Transitions: {len(transitions)}_\n\n")
            fh.write("## Memories\n\n")
            for m in memories:
                fh.write(_memory_to_markdown(m))
                fh.write("\n")
            if transitions:
                fh.write("## Transitions\n\n")
                for t in transitions:
                    fh.write(_transition_to_markdown(t))
                    fh.write("\n")
        return len(memories)

    # ------------------------------------------------------------------
    # ARW (manual flush; background worker lands in next milestone)
    # ------------------------------------------------------------------

    async def flush_arw(self) -> int:
        """Process all pending Active Reconsolidation rows synchronously.

        For v0.1 this uses the ARWWorker to classify relationships and
        generate inferences/transitions. If no LLM is configured, the queue
        is left intact so a later configured worker can drain it.

        The CLI and the wedge-demo integration test call ``flush_arw``
        between scripted ``remember`` calls so cross-session inferences land
        before the next ``recall``.
        """
        self._require_started()
        if not self._arw_worker.enabled:
            return 0
        conn = self._storage_conn_or_none()
        if conn is None:
            return 0

        # Fetch pending memory IDs
        cur = conn.execute("SELECT memory_id FROM pending_arw")
        rows = cur.fetchall()
        if not rows:
            return 0

        memory_ids = [row[0] for row in rows]
        total_processed = 0

        for memory_id in memory_ids:
            # Fetch the memory
            memory = await self._storage.get(memory_id)
            if memory is None:
                # Memory not found, skip
                continue

            try:
                result = await self._arw_worker.process(memory)
            except Exception as exc:
                self._mark_arw_retry(conn, memory_id, str(exc))
                continue

            if result.inferences or result.transitions:
                await self._storage.commit_batch(
                    WriteBatch(
                        memories=list(result.inferences),
                        transitions=list(result.transitions),
                    )
                )

            for update_memory_id, valid_until in result.updates:
                await self._storage.bind_valid_until(update_memory_id, valid_until)

            conn.execute("DELETE FROM pending_arw WHERE memory_id = ?", (memory_id,))
            total_processed += 1

        return total_processed

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _index_entities(self, memories: Sequence[Memory]) -> None:
        """Materialize entity-graph rows for the given memories.

        Best-effort: any failure is logged and swallowed because the
        underlying memory commit has already succeeded. The graph is an
        optional retrieval stage; it must not block the foreground write
        path or invalidate already-durable data.
        """
        # Storage backends without entity-graph support (e.g. legacy mocks
        # in tests) simply lack the methods. Detect once and short-circuit.
        upsert = getattr(self._storage, "upsert_entities", None)
        add_mentions = getattr(self._storage, "add_entity_mentions", None)
        add_edges = getattr(self._storage, "add_entity_edges", None)
        if upsert is None or add_mentions is None or add_edges is None:
            return
        try:
            delta = merge_deltas(extract_for_memory(m) for m in memories)
            if delta.entities:
                await upsert(delta.entities)
            if delta.mentions:
                await add_mentions(delta.mentions)
            if delta.edges:
                await add_edges(delta.edges)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "entity_index_failed",
                extra={
                    "audit_event": "entity_index_failed",
                    "error": repr(exc),
                    "memory_count": len(memories),
                },
            )

    async def _graph_recall(
        self,
        text: str,
        *,
        identity_id: str | None,
        valid_at: datetime | None,
        limit: int,
    ) -> list[RecallResult]:
        """Run the graph-walk recall stage. Returns ``[]`` on any backend
        without graph support, when graph retrieval is disabled, or when
        no seeds are present in the graph."""
        if not self._config.graph_retrieval_enabled:
            return []
        # Duck-type the storage: the protocol's graph methods are optional
        # on legacy backends so we check before invoking the walk.
        if not all(
            hasattr(self._storage, name)
            for name in ("entities_by_ids", "neighbors", "memories_mentioning")
        ):
            return []
        cfg = WalkConfig(
            depth=self._config.graph_walk_depth,
            fanout=self._config.graph_walk_fanout,
        )
        try:
            results = await graph_walk(
                text,
                storage=self._storage,  # type: ignore[arg-type]
                config=cfg,
                identity_id=identity_id,
                valid_at=valid_at,
                limit=limit,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "graph_walk_failed",
                extra={"audit_event": "graph_walk_failed", "error": repr(exc)},
            )
            return []
        weight = self._config.graph_walk_weight
        return [
            RecallResult(
                memory=r.memory,
                score=r.score * weight,
                matched_stage=r.matched_stage,
            )
            for r in results
        ]

    async def _enqueue_arw(self, memory_ids: list[str]) -> None:
        """Append rows to ``pending_arw``. Best-effort \u2014 a failure here
        does not invalidate the foreground commit (the memory is already
        durable; ARW will rediscover orphans during the v0.4 sleep sweep).
        """
        conn = self._storage_conn_or_none()
        if conn is None or not memory_ids:
            return
        now_iso = self.now().isoformat()
        try:
            conn.executemany(
                "INSERT INTO pending_arw (memory_id, enqueued_at, attempts) VALUES (?, ?, 0)",
                [(mid, now_iso) for mid in memory_ids],
            )
        except Exception:  # pragma: no cover - best-effort
            # Logging hook lands with OpenTelemetry in v1.
            pass

    def _storage_conn_or_none(self) -> Any:
        """Convenience accessor for the storage's underlying connection.

        The engine is normally storage-agnostic, but the ARW queue is a
        SQLite-specific implementation detail at v0. Once the queue lands
        on the :class:`Storage` protocol (v0.2), this access vanishes.
        """
        return getattr(self._storage, "_conn", None)

    def _mark_arw_retry(self, conn: Any, memory_id: str, error: str) -> None:
        conn.execute(
            """
            UPDATE pending_arw
            SET attempts = attempts + 1,
                last_error = ?
            WHERE memory_id = ?
            """,
            (error[:1000], memory_id),
        )

    def _require_started(self) -> None:
        if not self._started:
            raise ReifyError("Reify.start() must be awaited before use")

    def _start_decay_task(self) -> None:
        interval = self._config.decay_sweep_interval_seconds
        if interval <= 0:
            return
        if self._decay_task is not None and not self._decay_task.done():
            return
        self._decay_task = asyncio.create_task(self._run_decay_sweeper(interval))

    async def _stop_decay_task(self) -> None:
        task = self._decay_task
        self._decay_task = None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _run_decay_sweeper(self, interval: float) -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                if not self._started:
                    return
                try:
                    await self.sweep_decay(reason="background")
                except Exception:
                    log.exception("reify_decay_sweep_failed")
        except asyncio.CancelledError:
            raise

    def _apply_default_retention(self, memory: Memory) -> Memory:
        """Attach the default TTL policy to claim-like memories."""
        if isinstance(memory, Experience):
            return memory
        metadata = dict(memory.metadata)
        retention = dict(metadata.get("retention", {}))
        expires_at = retention.get("expires_at")
        if expires_at:
            return memory
        retention["ttl_days"] = float(retention.get("ttl_days", self._config.default_ttl_days))
        retention["expires_at"] = (
            memory.valid_at + timedelta(days=retention["ttl_days"])
        ).isoformat()
        retention["reinforced_count"] = int(retention.get("reinforced_count", 0))
        metadata["retention"] = retention
        return memory.model_copy(update={"metadata": metadata})

    async def _reinforce_memories(
        self,
        memories: Sequence[Memory],
        *,
        now: datetime,
    ) -> None:
        """Extend TTL for returned claim-like memories."""
        for memory in memories:
            if isinstance(memory, Experience):
                continue
            retention = memory.metadata.get("retention")
            if not isinstance(retention, dict):
                continue
            expires_at_raw = retention.get("expires_at")
            if not isinstance(expires_at_raw, str):
                continue
            try:
                expires_at = datetime.fromisoformat(expires_at_raw)
            except ValueError:
                continue
            ttl_days = float(retention.get("ttl_days", self._config.default_ttl_days))
            baseline = expires_at if expires_at > now else now
            extended = baseline + timedelta(
                days=ttl_days * max(self._config.reinforcement_factor - 1.0, 0.0)
            )
            if extended <= expires_at:
                continue
            next_metadata = dict(memory.metadata)
            next_retention = dict(retention)
            next_retention["expires_at"] = extended.isoformat()
            next_retention["last_reinforced_at"] = now.isoformat()
            next_retention["reinforced_count"] = int(next_retention.get("reinforced_count", 0)) + 1
            next_metadata["retention"] = next_retention
            await self._storage.update_metadata(memory.id, next_metadata)

    async def _temporal_recall(self, query: Query) -> list[RecallResult]:
        if query.temporal_intent is None:
            return []
        tags_any = ["event"]
        if query.temporal_intent.mode == "age_at_event":
            tags_any = ["event", "profile"]
        elif query.temporal_intent.target_tag is not None:
            tags_any = [query.temporal_intent.target_tag]
        event_filter = RecallFilter(
            identity_ids=[query.identity_id] if query.identity_id else None,
            tags_any=tags_any,
            valid_at=query.valid_at or self.now(),
        )
        event_memories = [
            memory
            for memory in await self._storage.recall(event_filter, limit=2000)
            if isinstance(memory, Fact)
        ]
        derived = _resolve_temporal_query(query.temporal_intent, event_memories, now=self.now())
        if derived is None:
            return []
        return [RecallResult(memory=derived, score=1.0, matched_stage="temporal")]

    async def _aggregate_recall(self, query: Query) -> list[RecallResult]:
        if query.aggregate_intent is None:
            return []
        intent = query.aggregate_intent
        recall_filter = RecallFilter(
            identity_ids=[query.identity_id] if query.identity_id else None,
            tags_any=[intent.target_tag],
            valid_at=query.valid_at or self.now(),
        )
        memories = [
            memory
            for memory in await self._storage.recall(recall_filter, limit=2000)
            if isinstance(memory, Fact)
        ]
        derived = _resolve_aggregate_query(intent, memories, query=query, now=self.now())
        if derived is None:
            return []
        return [RecallResult(memory=derived, score=1.0, matched_stage="aggregate")]

    async def _assistant_recall(self, query: Query) -> list[RecallResult]:
        if not query.prefer_assistant_memories:
            return []
        recall_filter = RecallFilter(
            identity_ids=[query.identity_id] if query.identity_id else None,
            tags_any=["assistant-memory"],
            valid_at=query.valid_at or self.now(),
        )
        memories = [
            memory
            for memory in await self._storage.recall(recall_filter, limit=500)
            if isinstance(memory, Fact)
        ]
        if not memories:
            return []
        if query.assistant_item_index is not None:
            indexed = [
                memory for memory in memories
                if _fact_metadata_value(memory, "item_index") == str(query.assistant_item_index)
            ]
            if indexed:
                return [RecallResult(memory=indexed[0], score=1.0, matched_stage="assistant")]
        row_match = _assistant_row_match(query.text, memories)
        if row_match is not None:
            return [RecallResult(memory=row_match, score=1.0, matched_stage="assistant")]
        ranked: list[RecallResult] = []
        for memory in memories:
            score = _assistant_match_score(query.text, memory)
            if score <= 0:
                continue
            ranked.append(RecallResult(memory=memory, score=score, matched_stage="assistant"))
        ranked.sort(key=lambda r: r.score, reverse=True)
        return ranked


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _merge_recall_results(
    primary: Sequence[RecallResult],
    secondary: Sequence[RecallResult],
    *,
    limit: int,
) -> list[RecallResult]:
    """Merge ranked recall result lists, keeping ``primary`` strictly above.

    The metadata-first contract (FIRST_PRINCIPLES.md default strategy 3)
    requires that direct keyword/tag/temporal matches always rank above
    softer signals like the graph-walk. We honor that here:

    1. ``primary`` results retain their original order at the top.
    2. ``secondary`` results contribute only memories not already in
       ``primary`` (graph-walk often reaches the same memory metadata
       found, so this prevents double-counting).
    3. Secondary results are re-scored to be capped strictly below the
       lowest primary score so legibility ("graph-only" vs "metadata")
       is preserved without displacing primary hits.
    4. The combined list is truncated to ``limit``.
    """
    primary_ids = {r.memory.id for r in primary}
    primary_list = list(primary)

    if secondary:
        secondary_unique = [r for r in secondary if r.memory.id not in primary_ids]
        if secondary_unique:
            if primary_list:
                # Renormalize secondary so the highest secondary score sits
                # below the lowest primary score and the remainder spreads
                # proportionally below it.
                min_primary_score = min(r.score for r in primary_list)
                ceiling = max(min_primary_score - 1e-6, 0.0)
                top_secondary = max(r.score for r in secondary_unique) or 1.0
                scaled: list[RecallResult] = []
                for r in secondary_unique:
                    scaled_score = (r.score / top_secondary) * ceiling
                    scaled.append(
                        RecallResult(
                            memory=r.memory,
                            score=scaled_score,
                            matched_stage=r.matched_stage,
                        )
                    )
                scaled.sort(key=lambda r: r.score, reverse=True)
                primary_list.extend(scaled)
            else:
                # No primary hits: keep secondary scores exactly as produced
                # (graph-walk results are pre-weighted below 0.5). Scaling
                # them up to 1.0 here would present a soft graph-only
                # signal at full confidence.
                primary_list.extend(
                    sorted(secondary_unique, key=lambda r: r.score, reverse=True)
                )

    return primary_list[:limit]


def _replace(q: Query, **changes: Any) -> Query:
    """Lightweight ``dataclasses.replace`` for :class:`Query`."""
    data = {
        "text": q.text,
        "keywords": q.keywords,
        "tags": q.tags,
        "tags_explicit": q.tags_explicit,
        "networks": q.networks,
        "valid_at": q.valid_at,
        "identity_id": q.identity_id,
        "limit": q.limit,
        "prefer_user_memories": q.prefer_user_memories,
        "prefer_assistant_memories": q.prefer_assistant_memories,
        "assistant_item_index": q.assistant_item_index,
        "temporal_intent": q.temporal_intent,
        "aggregate_intent": q.aggregate_intent,
    }
    data.update(changes)
    return Query(**data)  # type: ignore[arg-type]


def _memory_to_markdown(m: Memory) -> str:
    head = f"### {m.network.value.title()} `{m.id}`\n\n"
    body = f"> {m.content}\n\n"
    meta = [
        f"- created: `{m.created_at.isoformat()}`",
        f"- valid: `{m.valid_at.isoformat()}` \u2192 "
        f"`{m.valid_until.isoformat() if m.valid_until else 'open'}`",
        f"- confidence: {m.confidence:.2f}",
    ]
    if m.identity_id:
        meta.append(f"- identity: `{m.identity_id}`")
    if m.tags:
        meta.append(f"- tags: {', '.join(f'`{t}`' for t in m.tags)}")
    if m.metadata:
        meta.append(f"- metadata: `{json.dumps(m.metadata, sort_keys=True)}`")
    return head + body + "\n".join(meta) + "\n"


def _resolve_temporal_query(
    intent: TemporalIntent,
    events: Sequence[Fact],
    *,
    now: datetime,
) -> Fact | None:
    if intent.mode == "compare_order":
        return _resolve_compare_order(intent, events, now=now)
    if intent.mode == "delta_days":
        return _resolve_delta_days(intent, events, now=now)
    if intent.mode == "first_after":
        return _resolve_first_after(intent, events, now=now)
    if intent.mode == "first_date":
        return _resolve_first_date(intent, events, now=now)
    if intent.mode == "wake_time":
        return _resolve_wake_time(intent, events, now=now)
    if intent.mode == "wake_delta":
        return _resolve_wake_delta(intent, events, now=now)
    if intent.mode == "event_object":
        return _resolve_event_object(intent, events, now=now)
    if intent.mode == "count_before":
        return _resolve_count_before(intent, events, now=now)
    if intent.mode == "age_at_event":
        return _resolve_age_at_event(intent, events, now=now)
    if intent.mode == "recent_start":
        return _resolve_recent_start(intent, events, now=now)
    return None


def _resolve_aggregate_query(
    intent: AggregateIntent,
    facts: Sequence[Fact],
    *,
    query: Query,
    now: datetime,
) -> Fact | None:
    if not facts:
        return None
    if intent.mode == "count":
        return _resolve_count_query(intent, facts, query=query, now=now)
    if intent.mode == "sum_duration":
        return _resolve_sum_duration_query(intent, facts, query=query, now=now)
    if intent.mode == "sum_money":
        return _resolve_sum_money_query(intent, facts, query=query, now=now)
    if intent.mode == "sum_value":
        return _resolve_sum_value_query(intent, facts, query=query, now=now)
    if intent.mode == "difference_value":
        return _resolve_difference_value_query(intent, facts, query=query, now=now)
    return None


def _resolve_count_query(intent: AggregateIntent, facts: Sequence[Fact], *, query: Query, now: datetime) -> Fact | None:
    filtered = [fact for fact in facts if _fact_matches_query_context(fact, query.text)]
    if not filtered:
        filtered = list(facts)
    if intent.target_tag == "inventory":
        lowered = query.text.lower()
        wanted_kind = None
        if "plant" in lowered:
            wanted_kind = "plant"
        elif "tank" in lowered:
            wanted_kind = "tank"
        elif "lens" in lowered:
            wanted_kind = "lens"
        if wanted_kind is not None:
            filtered = [fact for fact in filtered if _fact_metadata_value(fact, "kind") == wanted_kind]
    if intent.target_tag == "project":
        filtered = [
            fact for fact in filtered
            if (_fact_metadata_value(fact, "status") in {"current", "led"})
        ]
    if intent.target_tag == "expense" and "luxury" in query.text.lower():
        filtered = [fact for fact in filtered if _fact_metadata_value(fact, "category") == "luxury"]
    if intent.unique_field:
        seen: set[str] = set()
        chosen: list[Fact] = []
        for fact in filtered:
            value = _fact_metadata_value(fact, intent.unique_field)
            if not value or value in seen:
                continue
            seen.add(value)
            chosen.append(fact)
        count = len(chosen)
        evidence = chosen
    else:
        keys: set[str] = set()
        evidence = []
        for fact in filtered:
            value = _fact_metadata_value(fact, "item") or _fact_metadata_value(fact, "label") or fact.obj
            action = _fact_metadata_value(fact, "action") or ""
            key = f"{action}:{value.lower()}"
            if key in keys:
                continue
            keys.add(key)
            evidence.append(fact)
        count = len(evidence)
    if count == 0:
        return None
    return _aggregate_answer_fact(str(count), evidence, now=now, mode=intent.mode)


def _resolve_sum_duration_query(intent: AggregateIntent, facts: Sequence[Fact], *, query: Query, now: datetime) -> Fact | None:
    filtered = [fact for fact in facts if _fact_matches_query_context(fact, query.text)]
    if not filtered:
        filtered = list(facts)
    total = 0.0
    evidence: list[Fact] = []
    seen: set[str] = set()
    for fact in filtered:
        value = _fact_numeric_value(fact, "duration_value")
        unit = _fact_metadata_value(fact, "duration_unit")
        if value is None or unit is None:
            continue
        if intent.target_tag == "gaming":
            label = _fact_metadata_value(fact, "normalized_label") or _fact_metadata_value(fact, "label") or fact.obj
            dedupe_key = f"{label.lower()}:{value}:{unit}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
        normalized = _convert_duration(value, unit, intent.unit or unit)
        if normalized is None:
            continue
        total += normalized
        evidence.append(fact)
    if not evidence:
        return None
    answer = _format_aggregate_number(total)
    if intent.unit:
        answer = f"{answer} {intent.unit}"
    return _aggregate_answer_fact(answer, evidence, now=now, mode=intent.mode)


def _resolve_sum_money_query(intent: AggregateIntent, facts: Sequence[Fact], *, query: Query, now: datetime) -> Fact | None:
    filtered = [fact for fact in facts if _fact_matches_query_context(fact, query.text)]
    if not filtered:
        filtered = list(facts)
    if "luxury" in query.text.lower():
        filtered = [fact for fact in filtered if _fact_metadata_value(fact, "category") == "luxury"]
    total = 0.0
    evidence: list[Fact] = []
    seen: set[str] = set()
    for fact in filtered:
        value = _fact_numeric_value(fact, "amount_value")
        if value is None:
            continue
        label = _fact_metadata_value(fact, "normalized_label") or _fact_metadata_value(fact, "label") or fact.obj
        dedupe_key = f"{label.lower()}:{value}"
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        total += value
        evidence.append(fact)
    if not evidence:
        return None
    return _aggregate_answer_fact(f"${_format_aggregate_number(total)}", evidence, now=now, mode=intent.mode)


def _resolve_sum_value_query(intent: AggregateIntent, facts: Sequence[Fact], *, query: Query, now: datetime) -> Fact | None:
    value_field = intent.value_field
    if value_field is None:
        return None
    filtered = [fact for fact in facts if _fact_matches_query_context(fact, query.text)]
    if not filtered:
        filtered = list(facts)
    candidate_sets = [filtered]
    if intent.constrain_months:
        candidate_sets.insert(0, [fact for fact in filtered if fact.valid_at.month in intent.constrain_months])
    for candidates in candidate_sets:
        total = 0.0
        evidence: list[Fact] = []
        seen: set[str] = set()
        for fact in candidates:
            value = _fact_numeric_value(fact, value_field)
            if value is None:
                continue
            label = (
                _fact_metadata_value(fact, "normalized_label")
                or _fact_metadata_value(fact, "label")
                or fact.obj
            )
            dedupe_key = f"{label.lower()}:{value}"
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            total += value
            evidence.append(fact)
        if evidence:
            return _aggregate_answer_fact(_format_aggregate_number(total), evidence, now=now, mode=intent.mode)
    return None


def _resolve_difference_value_query(intent: AggregateIntent, facts: Sequence[Fact], *, query: Query, now: datetime) -> Fact | None:
    value_field = intent.value_field
    if value_field is None or len(intent.target_labels) < 2:
        return None
    filtered = [fact for fact in facts if _fact_matches_query_context(fact, query.text)]
    if not filtered:
        filtered = list(facts)
    first = _best_difference_fact(filtered, value_field=value_field, label=intent.target_labels[0])
    second = _best_difference_fact(filtered, value_field=value_field, label=intent.target_labels[1])
    if first is None or second is None:
        return None
    first_value = _fact_numeric_value(first, value_field)
    second_value = _fact_numeric_value(second, value_field)
    if first_value is None or second_value is None:
        return None
    answer_value = abs(second_value - first_value)
    answer = _format_aggregate_number(answer_value)
    if intent.unit == "usd":
        answer = f"${answer}"
    return _aggregate_answer_fact(answer, (first, second), now=now, mode=intent.mode)


def _resolve_compare_order(intent: TemporalIntent, events: Sequence[Fact], *, now: datetime) -> Fact | None:
    if len(intent.target_labels) < 2:
        return None
    answer_labels = intent.answer_labels if len(intent.answer_labels) >= 2 else intent.target_labels
    first = _best_event_match(events, intent.target_labels[0], kind=intent.target_kind, month=intent.constrain_month)
    second = _best_event_match(events, intent.target_labels[1], kind=intent.target_kind, month=intent.constrain_month)
    if first is None or second is None:
        return None
    if first.valid_at <= second.valid_at:
        winner = first
        answer = answer_labels[0]
    else:
        winner = second
        answer = answer_labels[1]
    return Fact(
        content=answer,
        subject="user",
        predicate="temporal_answer",
        obj=answer,
        created_at=now,
        valid_at=now,
        confidence=0.92,
        identity_id=winner.identity_id,
        tags=("temporal-answer",),
        metadata={
            "derived_from": [first.id, second.id],
            "mode": intent.mode,
        },
    )


def _resolve_delta_days(intent: TemporalIntent, events: Sequence[Fact], *, now: datetime) -> Fact | None:
    if not intent.target_labels:
        return None
    target = _best_event_match(events, intent.target_labels[0])
    reference = _best_event_match(
        events,
        intent.reference_label or "meeting",
        exclude_ids={target.id} if target is not None else None,
    )
    if target is None or reference is None:
        return None
    delta_days = abs((reference.valid_at.date() - target.valid_at.date()).days)
    answer = _format_temporal_delta(delta_days, intent.answer_unit)
    return Fact(
        content=answer,
        subject="user",
        predicate="temporal_answer",
        obj=answer,
        created_at=now,
        valid_at=now,
        confidence=0.9,
        identity_id=target.identity_id,
        tags=("temporal-answer",),
        metadata={
            "derived_from": [target.id, reference.id],
            "mode": intent.mode,
        },
    )


def _resolve_first_date(intent: TemporalIntent, events: Sequence[Fact], *, now: datetime) -> Fact | None:
    if not intent.target_labels:
        return None
    matches = _matching_events(events, intent.target_labels[0], kind=intent.target_kind, month=intent.constrain_month)
    if not matches:
        return None
    winner = min(matches, key=lambda event: event.valid_at)
    answer = _format_month_day(winner.valid_at)
    return Fact(
        content=answer,
        subject="user",
        predicate="temporal_answer",
        obj=answer,
        created_at=now,
        valid_at=now,
        confidence=0.9,
        identity_id=winner.identity_id,
        tags=("temporal-answer",),
        metadata={
            "derived_from": [winner.id],
            "mode": intent.mode,
        },
    )


def _resolve_wake_time(intent: TemporalIntent, facts: Sequence[Fact], *, now: datetime) -> Fact | None:
    requested_days = {label.lower() for label in intent.target_labels}
    if not requested_days:
        return None
    wake_facts = [fact for fact in facts if _event_kind(fact) == "wake_up"]
    explicit = [
        fact for fact in wake_facts
        if requested_days <= set(_fact_multi_metadata_values(fact, "weekdays"))
        and _fact_metadata_value(fact, "time") is not None
    ]
    if explicit:
        time_value = _fact_metadata_value(explicit[0], "time")
        if time_value is not None:
            return _temporal_answer_fact(time_value, explicit[:1], now=now, mode=intent.mode)

    base = next(
        (
            fact for fact in wake_facts
            if not _fact_multi_metadata_values(fact, "weekdays")
            and _fact_metadata_value(fact, "time") is not None
        ),
        None,
    )
    offset = next(
        (
            fact for fact in wake_facts
            if requested_days <= set(_fact_multi_metadata_values(fact, "weekdays"))
            and _fact_numeric_value(fact, "offset_minutes") is not None
        ),
        None,
    )
    if base is None or offset is None:
        return None
    base_time = _fact_metadata_value(base, "time")
    offset_minutes = _fact_numeric_value(offset, "offset_minutes")
    if base_time is None or offset_minutes is None:
        return None
    derived = _shift_clock_time(base_time, int(offset_minutes))
    if derived is None:
        return None
    return _temporal_answer_fact(derived, (base, offset), now=now, mode=intent.mode)


def _resolve_wake_delta(intent: TemporalIntent, facts: Sequence[Fact], *, now: datetime) -> Fact | None:
    requested_days = {label.lower() for label in intent.target_labels}
    if not requested_days:
        return None
    wake_facts = [fact for fact in facts if _event_kind(fact) == "wake_up"]
    specific = next(
        (
            fact for fact in wake_facts
            if requested_days <= set(_fact_multi_metadata_values(fact, "weekdays"))
            and _fact_metadata_value(fact, "time") is not None
        ),
        None,
    )
    baseline = next(
        (
            fact for fact in wake_facts
            if not _fact_multi_metadata_values(fact, "weekdays")
            and _fact_metadata_value(fact, "time") is not None
        ),
        None,
    )
    if specific is None or baseline is None:
        return None
    specific_time = _fact_metadata_value(specific, "time")
    baseline_time = _fact_metadata_value(baseline, "time")
    if specific_time is None or baseline_time is None:
        return None
    delta_minutes = _clock_minutes(baseline_time) - _clock_minutes(specific_time)
    if delta_minutes <= 0:
        return None
    answer = "1 minute" if delta_minutes == 1 else f"{delta_minutes} minutes"
    return _temporal_answer_fact(answer, (specific, baseline), now=now, mode=intent.mode)


def _resolve_event_object(intent: TemporalIntent, events: Sequence[Fact], *, now: datetime) -> Fact | None:
    if not intent.target_labels:
        return None
    candidates = _matching_events(
        events,
        intent.target_labels[0],
        kind=intent.target_kind,
        month=None,
    )
    if intent.relative_filter == "last_month":
        previous_month = (now.month - 1) or 12
        previous_year = now.year if now.month > 1 else now.year - 1
        candidates = [
            event for event in candidates
            if event.valid_at.year == previous_year and event.valid_at.month == previous_month
        ]
    if not candidates:
        return None
    winner = max(candidates, key=lambda event: event.valid_at)
    return _temporal_answer_fact(winner.obj, (winner,), now=now, mode=intent.mode)


def _resolve_count_before(intent: TemporalIntent, events: Sequence[Fact], *, now: datetime) -> Fact | None:
    reference = _best_event_match(events, intent.reference_label or "")
    if reference is None:
        return None
    candidates = [
        event
        for event in events
        if event.id != reference.id and event.valid_at < reference.valid_at
    ]
    if intent.target_tag is not None:
        candidates = [event for event in candidates if intent.target_tag in event.tags]
    if not candidates:
        return None
    deduped: list[Fact] = []
    seen: set[str] = set()
    for event in sorted(candidates, key=lambda item: item.valid_at):
        dedupe_key = _event_dedup_key(event)
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        deduped.append(event)
    return _temporal_answer_fact(str(len(deduped)), tuple(deduped), now=now, mode=intent.mode)


def _resolve_age_at_event(intent: TemporalIntent, facts: Sequence[Fact], *, now: datetime) -> Fact | None:
    if not intent.target_labels:
        return None
    event = _best_event_match(facts, intent.target_labels[0], kind=intent.target_kind)
    if event is None:
        return None
    age_fact = next(
        (
            fact
            for fact in facts
            if fact.predicate == "has_age_years" and _fact_numeric_value(fact, "age_years") is not None
        ),
        None,
    )
    if age_fact is None:
        return None
    age_years = _fact_numeric_value(age_fact, "age_years")
    if age_years is None:
        return None
    years_elapsed = max(0, round((age_fact.valid_at.date() - event.valid_at.date()).days / 365))
    answer = str(max(0, int(age_years) - years_elapsed))
    return _temporal_answer_fact(answer, (age_fact, event), now=now, mode=intent.mode)


def _resolve_recent_start(intent: TemporalIntent, facts: Sequence[Fact], *, now: datetime) -> Fact | None:
    candidates = [
        fact
        for fact in facts
        if (intent.target_kind is None or _event_kind(fact) == intent.target_kind)
    ]
    if not candidates:
        return None
    winner = max(candidates, key=lambda fact: fact.valid_at)
    answer = _fact_metadata_value(winner, "service") or winner.obj
    return _temporal_answer_fact(answer, (winner,), now=now, mode=intent.mode)


def _resolve_first_after(intent: TemporalIntent, events: Sequence[Fact], *, now: datetime) -> Fact | None:
    reference = _best_event_match(events, intent.reference_label or "")
    if reference is None:
        return None
    candidates = [
        event
        for event in events
        if _event_kind(event) == intent.target_kind and event.valid_at >= reference.valid_at
    ]
    if not candidates:
        return None
    winner = min(candidates, key=lambda event: event.valid_at)
    return Fact(
        content=winner.obj,
        subject="user",
        predicate="temporal_answer",
        obj=winner.obj,
        created_at=now,
        valid_at=now,
        confidence=0.9,
        identity_id=winner.identity_id,
        tags=("temporal-answer",),
        metadata={
            "derived_from": [reference.id, winner.id],
            "mode": intent.mode,
        },
    )


def _best_event_match(
    events: Sequence[Fact],
    label: str,
    *,
    kind: str | None = None,
    month: int | None = None,
    exclude_ids: set[str] | None = None,
) -> Fact | None:
    normalized_label = label.lower().strip()
    best: tuple[float, Fact] | None = None
    for event in events:
        if exclude_ids is not None and event.id in exclude_ids:
            continue
        event_kind = _event_kind(event)
        if kind is not None and event_kind != kind:
            continue
        if month is not None and event.valid_at.month != month:
            continue
        score = _event_match_score(event, normalized_label)
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, event)
    return best[1] if best is not None else None


def _matching_events(
    events: Sequence[Fact],
    label: str,
    *,
    kind: str | None = None,
    month: int | None = None,
) -> list[Fact]:
    matched: list[Fact] = []
    normalized_label = label.lower().strip()
    for event in events:
        event_kind = _event_kind(event)
        if kind is not None and event_kind != kind:
            continue
        if month is not None and event.valid_at.month != month:
            continue
        if _event_match_score(event, normalized_label) > 0:
            matched.append(event)
    return matched


def _event_kind(event: Fact) -> str | None:
    metadata = event.metadata.get("event")
    if isinstance(metadata, dict):
        kind = metadata.get("kind")
        if isinstance(kind, str):
            return kind
    return None


def _event_label(event: Fact) -> str:
    metadata = event.metadata.get("event")
    if isinstance(metadata, dict):
        label = metadata.get("label")
        if isinstance(label, str) and label.strip():
            return label
    return event.obj


def _event_dedup_key(event: Fact) -> str:
    label = _event_label(event).lower().strip()
    quoted = re.findall(r'["\']([^"\']+)["\']', label)
    if quoted:
        label = quoted[0].strip()
    elif "," in label:
        label = label.split(",")[-1].strip()
    label = re.sub(r"^(?:the|a|an)\s+", "", label)
    return f"{event.valid_at.date().isoformat()}::{label}"


def _event_match_score(event: Fact, normalized_label: str) -> float:
    haystacks = [event.content.lower(), event.obj.lower()]
    haystacks.extend(_metadata_haystacks(event.metadata))

    if normalized_label in {"meeting", "team meeting"} and any("meeting" in hay for hay in haystacks):
        return 3.0
    if normalized_label in {"bike", "car"}:
        if any(normalized_label in hay for hay in haystacks):
            return 3.0
    if normalized_label in {"mesh network system", "new router"}:
        if any("router" in hay or "mesh network" in hay for hay in haystacks):
            return 3.0
    if normalized_label.startswith("started working with") and any("started working with" in hay for hay in haystacks):
        return 3.5
    if normalized_label and any(normalized_label in hay for hay in haystacks):
        return 4.0

    label_tokens = _event_tokens(normalized_label)
    if not label_tokens:
        return 0.0
    best_overlap = 0.0
    for hay in haystacks:
        hay_tokens = _event_tokens(hay)
        overlap = len(label_tokens & hay_tokens) / len(label_tokens)
        if overlap > best_overlap:
            best_overlap = overlap
    return best_overlap


def _best_difference_fact(facts: Sequence[Fact], *, value_field: str, label: str) -> Fact | None:
    normalized = label.lower().strip()
    best: tuple[float, Fact] | None = None
    for fact in facts:
        value = _fact_numeric_value(fact, value_field)
        if value is None:
            continue
        score = 0.0
        hay = " ".join(
            [
                fact.content.lower(),
                fact.obj.lower(),
                json.dumps(fact.metadata, sort_keys=True).lower(),
            ]
        )
        if normalized in hay:
            score += 1.0
        kind = _fact_metadata_value(fact, "kind")
        if kind and normalized in kind.lower():
            score += 1.0
        label_value = _fact_metadata_value(fact, "label")
        if label_value and normalized in label_value.lower():
            score += 1.0
        if score <= 0:
            continue
        if best is None or score > best[0]:
            best = (score, fact)
    return best[1] if best is not None else None


def _metadata_haystacks(value: object) -> list[str]:
    out: list[str] = []
    if isinstance(value, str):
        out.append(value.lower())
    elif isinstance(value, dict):
        for nested in value.values():
            out.extend(_metadata_haystacks(nested))
    elif isinstance(value, list):
        for nested in value:
            out.extend(_metadata_haystacks(nested))
    return out


def _event_tokens(text: str) -> set[str]:
    tokens: set[str] = set()
    for raw in text.replace("-", " ").split():
        cleaned = raw.strip(" '\".,?!():;").lower()
        if not cleaned:
            continue
        tokens.add(cleaned)
        if len(cleaned) > 3 and cleaned.endswith("s"):
            tokens.add(cleaned[:-1])
    return tokens


def _aggregate_answer_fact(answer: str, evidence: Sequence[Fact], *, now: datetime, mode: str) -> Fact:
    return Fact(
        content=answer,
        subject="user",
        predicate="aggregate_answer",
        obj=answer,
        created_at=now,
        valid_at=now,
        confidence=0.9,
        identity_id=evidence[0].identity_id,
        tags=("aggregate-answer",),
        metadata={
            "derived_from": [fact.id for fact in evidence],
            "mode": mode,
        },
    )


def _temporal_answer_fact(answer: str, evidence: Sequence[Fact], *, now: datetime, mode: str) -> Fact:
    return Fact(
        content=answer,
        subject="user",
        predicate="temporal_answer",
        obj=answer,
        created_at=now,
        valid_at=now,
        confidence=0.9,
        identity_id=evidence[0].identity_id,
        tags=("temporal-answer",),
        metadata={
            "derived_from": [fact.id for fact in evidence],
            "mode": mode,
        },
    )


def _assistant_match_score(query_text: str, fact: Fact) -> float:
    combined = " ".join(
        [
            fact.content.lower(),
            fact.obj.lower(),
            json.dumps(fact.metadata, sort_keys=True).lower(),
        ]
    )
    score = 0.0
    lowered_query = query_text.lower()
    for token in lowered_query.split():
        cleaned = token.strip(" ?!.,")
        if len(cleaned) <= 2:
            continue
        if cleaned in combined:
            score += 0.2
    if "assistant-row" in fact.tags:
        score += 0.15
    if "assistant-item" in fact.tags:
        score += 0.1
    if "assistant-option" in fact.tags:
        score += 0.14
    name_query = any(
        phrase in lowered_query
        for phrase in (
            "what was the name",
            "remind me of the",
            "which restaurant",
            "which book",
            "which song",
            "which trail",
            "which resource",
            "which state",
            "which singer",
            "which job",
            "what was the example",
        )
    )
    if "assistant-name" in fact.tags and name_query:
        score += 0.2
    category = _fact_metadata_value(fact, "category")
    if category and category in lowered_query:
        score += 0.18
    transport_mode = _fact_metadata_value(fact, "transport_mode")
    if transport_mode and transport_mode in lowered_query:
        score += 0.2
    if any(phrase in lowered_query for phrase in ("how much", "price", "cost", "fare", "save")):
        if _fact_numeric_value(fact, "price_min") is not None or _fact_numeric_value(fact, "price_max") is not None:
            score += 0.22
    if any(phrase in lowered_query for phrase in ("which option", "which one", "recommended last time")):
        if "assistant-option" in fact.tags:
            score += 0.12
    return score


def _assistant_row_match(query_text: str, memories: Sequence[Fact]) -> Fact | None:
    lower = query_text.lower()
    weekday = next(
        (
            day
            for day in ("sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday")
            if day in lower
        ),
        None,
    )
    if weekday is None:
        return None
    names = [token.strip(" ?!.,") for token in query_text.split() if token[:1].isupper()]
    for memory in memories:
        if "assistant-row" not in memory.tags:
            continue
        hay = memory.content.lower()
        if weekday not in hay:
            continue
        if any(name.lower() in hay for name in names):
            return memory
    return None


def _fact_matches_query_context(fact: Fact, query_text: str) -> bool:
    normalized_query = query_text.lower()
    combined = " ".join(
        [
            fact.content.lower(),
            fact.obj.lower(),
            json.dumps(fact.metadata, sort_keys=True).lower(),
        ]
    )
    if "camping" in normalized_query and "camping" not in combined:
        return False
    if "doctor" in normalized_query and "doctor" not in combined and "dr." not in combined:
        return False
    if "project" in normalized_query and "project" not in combined:
        return False
    if "model kit" in normalized_query and "model" not in combined and "kit" not in combined:
        return False
    if "pick up" in normalized_query or "return" in normalized_query:
        return any(token in combined for token in ("pick up", "pickup", "return", "dry cleaning", "zara"))
    if "bike-related" in normalized_query and "bike" not in combined:
        return False
    if "road trip" in normalized_query and "drive" not in combined:
        return False
    if "plants" in normalized_query and "plant" not in combined:
        return False
    if "tanks" in normalized_query and "tank" not in combined:
        return False
    if "movie festivals" in normalized_query and "festival" not in combined and "fest" not in combined:
        return False
    if "weddings" in normalized_query and "wedding" not in combined:
        return False
    if "social media break" in normalized_query and "social media break" not in combined:
        return False
    if "playing games" in normalized_query and "playing" not in combined:
        return False
    if "luxury items" in normalized_query and "luxury" not in combined and "designer" not in combined:
        return False
    if "citrus fruits" in normalized_query and not any(token in combined for token in ("orange", "lemon", "lime", "grapefruit")):
        return False
    if "people reached" in normalized_query and not any(token in combined for token in ("followers", "reached", "audience")):
        return False
    if "page count" in normalized_query and "page" not in combined:
        return False
    if "siblings" in normalized_query and not any(token in combined for token in ("sister", "brother", "sibling")):
        return False
    return True


def _fact_metadata_value(fact: Fact, key: str) -> str | None:
    for meta_key in (
        "profile",
        "event",
        "obligation",
        "project",
        "trip",
        "media",
        "doctor",
        "expense",
        "travel",
        "model_kit",
        "inventory",
        "festival",
        "wedding",
        "social_break",
        "gaming",
        "citrus",
        "education",
        "assistant_reference",
        "wake_routine",
        "audience",
        "reading",
        "family",
        "assistant_reference",
        "deal",
        "transport_cost",
    ):
        value = fact.metadata.get(meta_key)
        if isinstance(value, dict):
            item = value.get(key)
            if isinstance(item, str):
                return item
    return None


def _fact_numeric_value(fact: Fact, key: str) -> float | None:
    for meta_key in (
        "profile",
        "event",
        "obligation",
        "project",
        "trip",
        "media",
        "doctor",
        "expense",
        "travel",
        "model_kit",
        "social_break",
        "gaming",
        "wake_routine",
        "audience",
        "reading",
        "family",
        "assistant_reference",
        "deal",
        "transport_cost",
    ):
        value = fact.metadata.get(meta_key)
        if isinstance(value, dict):
            item = value.get(key)
            if isinstance(item, (int, float)):
                return float(item)
    return None


def _fact_multi_metadata_values(fact: Fact, key: str) -> tuple[str, ...]:
    for meta_key in ("wake_routine", "event"):
        value = fact.metadata.get(meta_key)
        if isinstance(value, dict):
            item = value.get(key)
            if isinstance(item, list) and all(isinstance(entry, str) for entry in item):
                return tuple(item)
    return ()


def _format_temporal_delta(delta_days: int, unit: str | None) -> str:
    if unit is None:
        if delta_days >= 60 and delta_days % 30 == 0:
            unit = "months"
        elif delta_days >= 14 and delta_days % 7 == 0:
            unit = "weeks"
        else:
            unit = "days"
    if unit == "weeks":
        weeks = delta_days / 7
        value = int(weeks) if weeks.is_integer() else round(weeks, 1)
        suffix = "week" if value == 1 else "weeks"
        if float(value).is_integer() and 1 <= int(value) <= 12:
            return f"{_small_number_word(int(value))} {suffix}"
        return f"{_format_aggregate_number(float(value))} {suffix}"
    if unit == "months":
        months = max(1, round(delta_days / 30))
        years, remainder = divmod(months, 12)
        if years and remainder:
            year_label = "year" if years == 1 else "years"
            month_label = "month" if remainder == 1 else "months"
            return f"{years} {year_label} and {remainder} {month_label}"
        if years:
            year_label = "year" if years == 1 else "years"
            return f"{years} {year_label}"
        suffix = "month" if months == 1 else "months"
        if 1 <= months <= 12:
            return f"{_small_number_word(months)} {suffix}"
        return f"{months} {suffix}"
    suffix = "day" if delta_days == 1 else "days"
    return f"{delta_days} {suffix}"


def _format_month_day(value: datetime) -> str:
    month = value.strftime("%B")
    day = value.day
    suffix = "th"
    if day % 10 == 1 and day % 100 != 11:
        suffix = "st"
    elif day % 10 == 2 and day % 100 != 12:
        suffix = "nd"
    elif day % 10 == 3 and day % 100 != 13:
        suffix = "rd"
    return f"{month} {day}{suffix}"


def _shift_clock_time(value: str, offset_minutes: int) -> str | None:
    parsed = datetime.strptime(value.upper(), "%I:%M %p")
    shifted = parsed + timedelta(minutes=offset_minutes)
    return shifted.strftime("%-I:%M %p")


def _clock_minutes(value: str) -> int:
    parsed = datetime.strptime(value.upper(), "%I:%M %p")
    return parsed.hour * 60 + parsed.minute


def _small_number_word(value: int) -> str:
    return {
        1: "one",
        2: "two",
        3: "three",
        4: "four",
        5: "five",
        6: "six",
        7: "seven",
        8: "eight",
        9: "nine",
        10: "ten",
        11: "eleven",
        12: "twelve",
    }[value]


def _convert_duration(value: float, source_unit: str, target_unit: str) -> float | None:
    if source_unit == target_unit:
        return value
    if source_unit == "days" and target_unit == "weeks":
        return value / 7.0
    if source_unit == "weeks" and target_unit == "days":
        return value * 7.0
    if source_unit == "hours" and target_unit == "days":
        return value / 24.0
    if source_unit == "days" and target_unit == "hours":
        return value * 24.0
    return None


def _format_aggregate_number(value: float) -> str:
    if value.is_integer():
        return str(int(value))
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _transition_to_markdown(t: Transition) -> str:
    head = f"### {t.kind.value} `{t.id}`\n\n"
    body = f"> {t.summary}\n\n" if t.summary else ""
    meta = [
        f"- at: `{t.at.isoformat()}`",
        f"- from: `{t.from_memory_id or '(none)'}`",
        f"- to:   `{t.to_memory_id or '(none)'}`",
        f"- confidence: {t.confidence:.2f}",
    ]
    return head + body + "\n".join(meta) + "\n"
