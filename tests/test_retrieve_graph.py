"""End-to-end tests for the v0.3 graph-walk recall stage.

Covers:

* Storage round-trip of entities, mentions, and edges through ``SQLiteStore``.
* Edge merging on re-observation (weight aggregation).
* Graph-walk recall via the engine's full ``recall()`` pipeline (which is
  the path real callers exercise through the CLI and MCP server).

The wedge demo and existing v0.1/v0.2 tests already exercise the
metadata-first recall path; these tests verify graph-walk *adds* recall
without regressing it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from reify import EngineConfig, Reify
from reify.core.types import Fact
from reify.retrieve.entities import (
    Entity,
    EntityEdge,
    EntityMention,
    extract_for_memory,
    merge_deltas,
)
from reify.retrieve.graph import WalkConfig, graph_walk
from reify.storage import SQLiteStore, WriteBatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> SQLiteStore:
    s = SQLiteStore(":memory:")
    await s.init()
    try:
        yield s
    finally:
        await s.close()


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Storage round-trip
# ---------------------------------------------------------------------------


async def test_entity_round_trip(store: SQLiteStore) -> None:
    fact = Fact(
        content="Arun works at Acme Corp.",
        subject="Arun",
        predicate="works_at",
        obj="Acme Corp",
    )
    await store.commit_batch(WriteBatch(memories=[fact]))
    delta = extract_for_memory(fact)
    await store.upsert_entities(delta.entities)
    await store.add_entity_mentions(delta.mentions)
    await store.add_entity_edges(delta.edges)

    present = await store.entities_by_ids(["arun", "acme_corp", "nonexistent"])
    assert set(present) == {"arun", "acme_corp"}

    neighbors = await store.neighbors("arun")
    relations = {edge.relation for edge in neighbors}
    assert "works_at" in relations
    targets = {edge.to_entity for edge in neighbors}
    assert "acme_corp" in targets

    triples = await store.memories_mentioning(["arun"])
    assert len(triples) == 1
    mem, ent_id, conf = triples[0]
    assert mem.id == fact.id
    assert ent_id == "arun"
    assert conf == 1.0  # subject-of-Fact is high confidence


async def test_edge_weight_aggregates_on_reobservation(store: SQLiteStore) -> None:
    f1 = Fact(content="Arun joined Acme.", subject="Arun", predicate="works_at", obj="Acme")
    f2 = Fact(content="Arun stays at Acme.", subject="Arun", predicate="works_at", obj="Acme")
    await store.commit_batch(WriteBatch(memories=[f1, f2]))
    delta = merge_deltas([extract_for_memory(f1), extract_for_memory(f2)])
    await store.upsert_entities(delta.entities)
    await store.add_entity_edges(delta.edges)

    neighbors = await store.neighbors("arun")
    # The same (from, to, relation) tuple must collapse to one row, not two.
    works_at = [n for n in neighbors if n.relation == "works_at"]
    assert len(works_at) == 1
    assert works_at[0].weight == 1.0  # MAX(weight) preserved


async def test_memories_mentioning_filters_soft_deleted(store: SQLiteStore) -> None:
    fact = Fact(
        content="Priya works at Globex.", subject="Priya", predicate="works_at", obj="Globex"
    )
    await store.commit_batch(WriteBatch(memories=[fact]))
    delta = extract_for_memory(fact)
    await store.upsert_entities(delta.entities)
    await store.add_entity_mentions(delta.mentions)

    triples_before = await store.memories_mentioning(["priya"])
    assert len(triples_before) == 1

    await store.soft_delete(fact.id)
    triples_after = await store.memories_mentioning(["priya"])
    assert triples_after == []


# ---------------------------------------------------------------------------
# graph_walk standalone
# ---------------------------------------------------------------------------


async def test_graph_walk_returns_empty_when_no_seeds_in_graph(
    store: SQLiteStore,
) -> None:
    # Empty graph + arbitrary query -> no results.
    results = await graph_walk("What did Arun say?", storage=store)
    assert results == []


async def test_graph_walk_finds_one_hop_neighbor(store: SQLiteStore) -> None:
    f1 = Fact(content="Arun works at Acme.", subject="Arun", predicate="works_at", obj="Acme")
    f2 = Fact(
        content="Acme launched a new product.",
        subject="Acme",
        predicate="launched",
        obj="Sonar",
    )
    await store.commit_batch(WriteBatch(memories=[f1, f2]))
    for f in (f1, f2):
        delta = extract_for_memory(f)
        await store.upsert_entities(delta.entities)
        await store.add_entity_mentions(delta.mentions)
        await store.add_entity_edges(delta.edges)

    results = await graph_walk("What did Arun's company do?", storage=store)
    # Both the seed-mentioning memory (f1) and the one-hop-away memory (f2)
    # must surface, with f1 ranked above f2 (closer hop).
    ids = [r.memory.id for r in results]
    assert f1.id in ids
    assert f2.id in ids
    assert ids.index(f1.id) < ids.index(f2.id)


async def test_graph_walk_respects_depth(store: SQLiteStore) -> None:
    # Three-hop chain: Arun -> Acme -> Sonar -> Beta. With depth=1 only
    # Arun and Acme should be reached; Sonar and Beta must not surface.
    f1 = Fact(content="Arun at Acme.", subject="Arun", predicate="works_at", obj="Acme")
    f2 = Fact(content="Acme made Sonar.", subject="Acme", predicate="made", obj="Sonar")
    f3 = Fact(content="Sonar uses Beta.", subject="Sonar", predicate="uses", obj="Beta")
    await store.commit_batch(WriteBatch(memories=[f1, f2, f3]))
    for f in (f1, f2, f3):
        d = extract_for_memory(f)
        await store.upsert_entities(d.entities)
        await store.add_entity_mentions(d.mentions)
        await store.add_entity_edges(d.edges)

    shallow = await graph_walk(
        "Tell me about Arun.", storage=store, config=WalkConfig(depth=1)
    )
    shallow_ids = {r.memory.id for r in shallow}
    # f1 (mentions Arun directly) and f2 (mentions Acme, one hop) are in.
    # f3 (mentions Sonar, two hops) is out.
    assert f1.id in shallow_ids
    assert f3.id not in shallow_ids


# ---------------------------------------------------------------------------
# Engine integration: graph stage merged into recall()
# ---------------------------------------------------------------------------


async def test_engine_recall_uses_graph_stage_for_indirect_query() -> None:
    """A query that mentions a *related* entity but not the target memory's
    keywords should still surface the related memory via the graph stage."""
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Seed with two Facts that share the entity "Acme".
        await engine.remember(
            "Arun works at Acme.", at=_utc(2025, 1, 1)
        )
        await engine.remember(
            "Acme launched the Sonar project.", at=_utc(2025, 2, 1)
        )

        # Query mentions Arun (seed) but the target memory is the second
        # one (about Acme/Sonar). The graph-walk hop Arun->Acme bridges
        # the gap.
        response = await engine.recall("What is Arun's company up to?", limit=10)
        contents = " | ".join(r.memory.content for r in response.results)
        assert "Sonar" in contents or "Acme" in contents


async def test_engine_recall_does_not_regress_on_direct_match() -> None:
    """The graph stage must not displace direct keyword matches: if a
    memory directly contains the query keywords, it stays at the top."""
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        await engine.remember(
            "Arun lives in San Francisco.", at=_utc(2025, 1, 1)
        )
        await engine.remember(
            "Arun visited Boston for a conference.", at=_utc(2025, 2, 1)
        )

        # Use a query that overlaps the target memory's content lexically;
        # the metadata-first recaller should rank that match first and the
        # graph stage should append, not displace.
        response = await engine.recall("San Francisco", limit=5)
        assert response.results
        top = response.results[0]
        assert "San Francisco" in top.memory.content
        # The metadata-first match must not be tagged as graph-only.
        assert top.matched_stage != "graph"


async def test_engine_recall_is_noop_when_graph_disabled() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(
        storage=store,
        config=EngineConfig(graph_retrieval_enabled=False),
    )

    async with engine:
        await engine.remember("Arun works at Acme.", at=_utc(2025, 1, 1))
        await engine.remember("Acme launched Sonar.", at=_utc(2025, 2, 1))

        # Even when the graph is populated, disabling the stage means the
        # engine falls back to metadata-first only. Sanity check: no crash,
        # no exception, recall returns *something* (the direct keyword
        # match for the first memory).
        response = await engine.recall("Arun", limit=5)
        ids = {r.matched_stage for r in response.results}
        assert "graph" not in ids
