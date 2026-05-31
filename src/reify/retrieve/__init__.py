"""Hybrid retrieval pipeline.

Five parallel strategies fused with reciprocal rank fusion, then reranked
with a cross-encoder, then packed into a token budget for the target model.

Strategies
----------
* ``semantic``   - vector similarity over network-scoped indexes (storage layer)
* ``lexical``    - BM25 over canonical content (lives in :mod:`reify.recall`)
* ``graph_walk`` - k-hop traversal through extracted entity edges
  (:mod:`reify.retrieve.graph`); seeds via :mod:`reify.retrieve.entities`
* ``temporal``   - bitemporal filtering (``valid_at`` / ``valid_until``)
* ``causal``     - walk along LLM-annotated causal-parent edges (v0.4)

Pipeline stages
---------------
1. ``intent`` - fast classifier picks strategies and budgets.
2. ``parallel_fetch`` - strategies run concurrently.
3. ``fusion`` - RRF combines per-strategy ranks.
4. ``rerank`` - cross-encoder re-scores.
5. ``pack`` - token-budget-aware packer formats for the target model.
"""

from reify.retrieve.entities import (
    Entity,
    EntityEdge,
    EntityGraphDelta,
    EntityMention,
    canonicalize,
    extract_for_memory,
    extract_for_query,
    merge_deltas,
)
from reify.retrieve.graph import GraphStorage, WalkConfig, graph_walk
from reify.retrieve.rerank import (
    BGEReranker,
    FeatureReranker,
    FeatureWeights,
    NoopReranker,
    Reranker,
)

__all__ = [
    "BGEReranker",
    "Entity",
    "EntityEdge",
    "EntityGraphDelta",
    "EntityMention",
    "FeatureReranker",
    "FeatureWeights",
    "GraphStorage",
    "NoopReranker",
    "Reranker",
    "WalkConfig",
    "canonicalize",
    "extract_for_memory",
    "extract_for_query",
    "graph_walk",
    "merge_deltas",
]
