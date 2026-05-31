"""Graph-walk retrieval strategy.

Given a query, walk k hops from query-derived seed entities through the
entity graph populated at write time, and return the memories mentioning
any reached entity, scored by walk distance and edge weight.

This is one of the five recall stages from FIRST_PRINCIPLES.md
commandment 5 (``tag -> time -> entity -> BM25 -> embeddings``). The
walk earns its place when entity overlap is the strongest signal: queries
like *"what did I say about Acme last quarter?"* where the topic anchor
is a named entity but the relevant memories are scattered across many
sessions.

Design constraints
------------------
* **Pure read.** No mutation; safe to run on any storage snapshot.
* **Bounded.** Walk depth and per-hop fanout are capped so worst-case
  cost stays linear in ``depth * fanout``.
* **No LLM.** The walk is deterministic graph traversal. The reranker
  stage (v0.3 next slice) is where learned scoring lives.
* **Returns ``RecallResult``.** Score is normalized to ``[0, 1]`` so
  it composes with the other recall stages via simple weighted fusion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from reify.core.ids import MemoryId
from reify.core.types import Memory, Network
from reify.recall import RecallResult
from reify.retrieve.entities import Entity, extract_for_query


# ---------------------------------------------------------------------------
# Storage protocol: the minimum surface graph-walk needs.
# Decouples this module from SQLiteStore so tests can inject fakes.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Edge:
    to_entity: str
    relation: str
    weight: float


@runtime_checkable
class GraphStorage(Protocol):
    async def entities_by_ids(self, ids: Sequence[str]) -> list[str]:
        """Return the subset of ``ids`` that exist in the entities table."""

    async def neighbors(
        self,
        entity_id: str,
        *,
        limit: int = 32,
    ) -> list[_Edge]:
        """Return outgoing edges from ``entity_id`` ordered by weight desc."""

    async def memories_mentioning(
        self,
        entity_ids: Sequence[str],
        *,
        identity_id: str | None = None,
        valid_at: datetime | None = None,
        limit: int = 200,
    ) -> list[tuple[Memory, str, float]]:
        """Return ``(memory, entity_id, mention_confidence)`` triples for any
        memory mentioning any of ``entity_ids``. Newest-first."""


# ---------------------------------------------------------------------------
# Walk
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WalkConfig:
    """Knobs for :func:`graph_walk`. Defaults sized for v0.3 corpora.

    * ``depth`` — number of hops; 2 is enough for "things related to my
      employer" (employer -> coworker / project / location).
    * ``fanout`` — per-hop neighbor cap so a hub entity does not explode
      the frontier. 32 is generous for the deterministic graph.
    * ``decay`` — multiplicative score decay per hop. 0.5 means second-hop
      memories are scored at half the weight of first-hop ones.
    * ``min_seed_kind_priority`` — kinds the seed must belong to, in
      priority order. ``None`` means accept any seed.
    """

    depth: int = 2
    fanout: int = 32
    decay: float = 0.5
    seed_limit: int = 8
    memory_limit: int = 200


async def graph_walk(
    query_text: str,
    *,
    storage: GraphStorage,
    config: WalkConfig | None = None,
    identity_id: str | None = None,
    valid_at: datetime | None = None,
    limit: int = 20,
    seeds: Sequence[Entity] | None = None,
) -> list[RecallResult]:
    """Run a k-hop weighted walk and return ranked :class:`RecallResult`.

    Steps
    -----
    1. Derive seed entities from ``query_text`` (or use the supplied
       ``seeds``).
    2. Filter seeds to those that actually exist in storage.
    3. BFS up to ``config.depth`` hops, accumulating per-entity scores
       weighted by edge weight and decayed by hop distance.
    4. For each reached entity, fetch memories mentioning it and combine
       its walk score with the mention confidence.
    5. Deduplicate memories (a memory mentioning multiple reached
       entities sums their contributions, capped at 1.0).
    6. Rank by combined score, return top ``limit``.

    Returns an empty list if no seeds are present in the graph; the recall
    pipeline treats that the same as a stage that contributed nothing.
    """
    cfg = config or WalkConfig()
    seed_entities = list(seeds) if seeds is not None else list(extract_for_query(query_text))
    if not seed_entities:
        return []

    seed_entities = seed_entities[: cfg.seed_limit]
    seed_ids = [e.id for e in seed_entities]
    present = set(await storage.entities_by_ids(seed_ids))
    if not present:
        return []

    # BFS with score accumulation. ``scores[entity_id]`` is the best score
    # found by any path that reaches it.
    scores: dict[str, float] = {eid: 1.0 for eid in present}
    frontier: list[tuple[str, float]] = [(eid, 1.0) for eid in present]
    visited: set[str] = set(present)
    for hop in range(cfg.depth):
        next_frontier: list[tuple[str, float]] = []
        decay = cfg.decay ** (hop + 1)
        for eid, parent_score in frontier:
            edges = await storage.neighbors(eid, limit=cfg.fanout)
            for edge in edges:
                if edge.to_entity in visited:
                    # Reinforce: take the better score if a new path is stronger.
                    candidate = parent_score * edge.weight * decay
                    prev = scores.get(edge.to_entity, 0.0)
                    if candidate > prev:
                        scores[edge.to_entity] = candidate
                    continue
                visited.add(edge.to_entity)
                child_score = parent_score * edge.weight * decay
                scores[edge.to_entity] = child_score
                next_frontier.append((edge.to_entity, child_score))
        frontier = next_frontier
        if not frontier:
            break

    # Fetch memories mentioning any reached entity.
    reached_ids = list(scores.keys())
    triples = await storage.memories_mentioning(
        reached_ids,
        identity_id=identity_id,
        valid_at=valid_at,
        limit=cfg.memory_limit,
    )

    # Combine per-(memory, entity) contributions. A memory mentioning
    # multiple reached entities sums their contributions, then is squashed
    # into ``[0, 1]`` via tanh-style saturation so highly-connected memories
    # still beat singly-connected ones without dominating the result set.
    by_memory: dict[MemoryId, tuple[Memory, float]] = {}
    for memory, entity_id, mention_conf in triples:
        walk_score = scores.get(entity_id, 0.0)
        contribution = walk_score * max(mention_conf, 0.0)
        if memory.id not in by_memory:
            by_memory[memory.id] = (memory, contribution)
        else:
            existing_mem, existing_score = by_memory[memory.id]
            by_memory[memory.id] = (existing_mem, existing_score + contribution)

    if not by_memory:
        return []

    # Saturate: 1 - exp(-score) maps [0, inf) -> [0, 1) monotonically.
    results = [
        RecallResult(
            memory=mem,
            score=1.0 - math.exp(-score),
            matched_stage="graph",
        )
        for mem, score in by_memory.values()
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:limit]


__all__ = [
    "GraphStorage",
    "WalkConfig",
    "graph_walk",
    "_Edge",
]
