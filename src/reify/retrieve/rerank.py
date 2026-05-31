"""Reranking stage of the v0.3 hybrid retrieval pipeline.

After the parallel recall stages produce a fused candidate set, a reranker
re-scores them with a richer signal that the cheap stages cannot afford
per-candidate. This module ships two implementations:

* :class:`FeatureReranker` — deterministic, pure-Python, dependency-free.
  Blends keyword overlap, entity overlap, recency, mention confidence,
  and provenance-source trust. Always available; the no-LLM, no-embedder
  floor (FIRST_PRINCIPLES.md commandment 4).
* :class:`BGEReranker` — adapter for the BGE cross-encoder family. Behind
  an optional dependency (``transformers`` + ``sentence-transformers``).
  Imports lazily so the core works when the optional deps are absent.

Both implementations satisfy the :class:`Reranker` protocol so the engine
can swap them via :class:`reify.engine.EngineConfig` without code changes.
The reranker stage may run *only on the merged candidate window*; it never
runs on the full memory store, keeping its cost bounded regardless of corpus
size (commandment 5: pre-filter aggressively).
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from reify.core.types import Fact, Memory
from reify.recall import RecallResult


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Reranker(Protocol):
    """A pluggable reranker.

    Implementations re-score a candidate set without changing its size:
    the input length and output length match. Order may change. Score
    semantics are implementation-defined but should remain in ``[0, 1]``
    so downstream consumers (the context packer, the user-value gate)
    can compare across implementations.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RecallResult],
        *,
        now: datetime | None = None,
    ) -> list[RecallResult]: ...


# ---------------------------------------------------------------------------
# Noop
# ---------------------------------------------------------------------------


class NoopReranker:
    """Pass-through reranker. The explicit identity for the no-LLM floor.

    Useful as a configuration sentinel: callers that intentionally disable
    reranking get a stable, typed object rather than ``None`` checks
    sprinkled across the codebase.
    """

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RecallResult],
        *,
        now: datetime | None = None,
    ) -> list[RecallResult]:
        return list(candidates)


# ---------------------------------------------------------------------------
# Deterministic feature reranker
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[\w']+")
_STOP = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "what", "where", "when", "who", "whom", "why",
    "how", "this", "that", "these", "those", "it", "its", "i", "you",
    "your", "my", "me", "we", "us", "they", "them", "their", "with",
    "about", "from",
})


def _tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if t.lower() not in _STOP}


@dataclass(frozen=True, slots=True)
class FeatureWeights:
    """Knobs for :class:`FeatureReranker`.

    Defaults sized to keep the reranker score in ``[0, 1]`` and to weight
    cheap-but-strong signals (keyword overlap, entity overlap) above
    softer ones (recency, source trust). Exposed as a dataclass so callers
    can A/B without subclassing.
    """

    keyword_overlap: float = 0.40
    entity_overlap: float = 0.25
    recency: float = 0.15
    source_trust: float = 0.10
    base_score: float = 0.10
    """Weight applied to the input :class:`RecallResult.score` so
    upstream stage rankings still bias the output."""

    half_life_days: float = 30.0
    """Recency falloff. A memory ``half_life_days`` old contributes 0.5
    of its full recency signal."""


# Provenance source -> trust prior. Higher means more reliable. Values
# come from the existing extractor/source taxonomy. Unknown sources get
# the default 0.5 (neutral).
_SOURCE_TRUST: dict[str, float] = {
    "user": 1.0,
    "cli:remember": 1.0,
    "claude_desktop": 0.95,
    "windsurf": 0.95,
    "cursor": 0.95,
    "extractor:deterministic": 0.9,
    "extractor:llm": 0.7,
    "ingest:browser": 0.5,
    "ingest:screen": 0.4,
    "library": 0.85,
}


class FeatureReranker:
    """Deterministic, dependency-free reranker.

    Blends:

    1. **Keyword overlap** — Jaccard between query tokens and memory
       content tokens (stop-words removed).
    2. **Entity overlap** — fraction of query-derived entities mentioned
       by the memory's content. Uses :func:`extract_for_query` and a
       cheap presence check; no DB calls.
    3. **Recency** — exponential decay on ``created_at`` with half-life
       from :class:`FeatureWeights`.
    4. **Source trust** — provenance-source prior from
       :data:`_SOURCE_TRUST`.
    5. **Base score** — preserves a slice of the upstream stage score
       so metadata-first wins do not invert.

    All five signals are clipped to ``[0, 1]`` and combined as a
    weighted sum, then re-normalized by the sum of weights so the final
    score sits cleanly in ``[0, 1]`` regardless of weight tuning.
    """

    def __init__(self, weights: FeatureWeights | None = None) -> None:
        self._w = weights or FeatureWeights()

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RecallResult],
        *,
        now: datetime | None = None,
    ) -> list[RecallResult]:
        if not candidates:
            return []
        # Local import: avoid forcing reify.retrieve.entities to load
        # when the engine is constructed without rerank in the loop.
        from reify.retrieve.entities import extract_for_query

        ts_now = now or datetime.now(timezone.utc)
        q_tokens = _tokens(query)
        q_entity_ids = {e.id for e in extract_for_query(query)}
        w = self._w
        weight_sum = (
            w.keyword_overlap
            + w.entity_overlap
            + w.recency
            + w.source_trust
            + w.base_score
        )
        if weight_sum <= 0:
            return list(candidates)

        rescored: list[RecallResult] = []
        for cand in candidates:
            mem = cand.memory
            f_keyword = _jaccard(q_tokens, _tokens(mem.content))
            f_entity = _entity_overlap(q_entity_ids, mem)
            f_recent = _recency_score(mem.created_at, ts_now, w.half_life_days)
            f_trust = _source_trust(mem)
            f_base = max(min(cand.score, 1.0), 0.0)

            score = (
                w.keyword_overlap * f_keyword
                + w.entity_overlap * f_entity
                + w.recency * f_recent
                + w.source_trust * f_trust
                + w.base_score * f_base
            ) / weight_sum

            rescored.append(
                RecallResult(
                    memory=mem,
                    score=score,
                    matched_stage=cand.matched_stage,
                )
            )

        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored


# ---------------------------------------------------------------------------
# BGE adapter (optional dependency)
# ---------------------------------------------------------------------------


class BGEReranker:
    """Adapter around a BGE cross-encoder model.

    Requires ``sentence-transformers`` (and indirectly ``transformers``)
    which are *not* part of Reify's core dependencies. Construction is
    cheap (model load is deferred to first call); the model is loaded
    lazily so importing this module has no side effects.

    If the optional dependency is missing the constructor raises
    :class:`ImportError`. Callers that want graceful degradation can
    catch it and fall back to :class:`FeatureReranker`.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier. Defaults to ``BAAI/bge-reranker-base``.
    device:
        Passed through to ``CrossEncoder``. ``None`` lets the underlying
        library auto-select (CUDA / MPS / CPU).
    max_pair_chars:
        Truncate (query, memory) pair to this many characters before
        scoring; protects against pathological long inputs.
    """

    def __init__(
        self,
        *,
        model_name: str = "BAAI/bge-reranker-base",
        device: str | None = None,
        max_pair_chars: int = 4096,
    ) -> None:
        try:
            # Lazy import: keep core install slim.
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - exercised manually
            raise ImportError(
                "BGEReranker requires `sentence-transformers`. Install with: "
                "pip install 'reify[rerank]' or pip install sentence-transformers"
            ) from exc
        self._cross_encoder_cls = CrossEncoder
        self._model = None
        self._model_name = model_name
        self._device = device
        self._max_pair_chars = max_pair_chars

    def _ensure_model(self):  # pragma: no cover - exercised manually with the dep
        if self._model is None:
            self._model = self._cross_encoder_cls(self._model_name, device=self._device)
        return self._model

    async def rerank(
        self,
        query: str,
        candidates: Sequence[RecallResult],
        *,
        now: datetime | None = None,
    ) -> list[RecallResult]:  # pragma: no cover - exercised manually with the dep
        if not candidates:
            return []
        import asyncio

        model = self._ensure_model()
        pairs = [
            (
                query[: self._max_pair_chars],
                cand.memory.content[: self._max_pair_chars],
            )
            for cand in candidates
        ]

        # CrossEncoder.predict is sync and CPU/GPU-bound; run off the loop.
        scores = await asyncio.to_thread(model.predict, pairs)
        # BGE rerankers emit logits in roughly [-10, 10]; squash to [0, 1].
        rescored = [
            RecallResult(
                memory=c.memory,
                score=1.0 / (1.0 + math.exp(-float(s))),
                matched_stage=c.matched_stage,
            )
            for c, s in zip(candidates, scores, strict=True)
        ]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / len(a | b)


def _recency_score(created_at: datetime, now: datetime, half_life_days: float) -> float:
    if half_life_days <= 0:
        return 1.0
    age = (now - created_at).total_seconds() / 86400.0
    if age < 0:
        return 1.0
    return float(math.pow(0.5, age / half_life_days))


def _source_trust(memory: Memory) -> float:
    if not memory.provenance:
        return 0.5
    # Take the highest-trust source attached to the memory.
    best = 0.0
    for p in memory.provenance:
        # Look up either the full source or its "namespace" prefix.
        prefix = p.source.split(":", 1)[0]
        score = _SOURCE_TRUST.get(p.source, _SOURCE_TRUST.get(prefix, 0.5))
        if score > best:
            best = score
    return best


def _entity_overlap(query_entity_ids: set[str], memory: Memory) -> float:
    if not query_entity_ids:
        return 0.0
    # Cheap presence check without a DB roundtrip: extract entities from
    # the memory content and intersect with the query's seed entities.
    from reify.retrieve.entities import extract_for_memory

    delta = extract_for_memory(memory)
    mem_ids = {e.id for e in delta.entities}
    if not mem_ids:
        return 0.0
    inter = query_entity_ids & mem_ids
    if not inter:
        return 0.0
    return len(inter) / len(query_entity_ids)


__all__ = [
    "BGEReranker",
    "FeatureReranker",
    "FeatureWeights",
    "NoopReranker",
    "Reranker",
]
