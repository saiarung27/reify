"""Tests for the v0.3 reranker stage.

Covers the deterministic FeatureReranker (always available) and the
NoopReranker. The BGEReranker requires `sentence-transformers` and is
covered by manual integration tests, not the default pytest run.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reify import EngineConfig, Reify
from reify.core.types import Experience, Fact, Provenance
from reify.recall import RecallResult
from reify.retrieve.rerank import (
    FeatureReranker,
    FeatureWeights,
    NoopReranker,
    Reranker,
)
from reify.storage import SQLiteStore


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_fact(content: str, **overrides) -> Fact:
    defaults: dict = {
        "subject": "Subj",
        "predicate": "rel",
        "obj": "Obj",
        "valid_at": _utc(2025, 1, 1),
        "created_at": _utc(2025, 1, 1),
    }
    defaults.update(overrides)
    return Fact(content=content, **defaults)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_feature_reranker_satisfies_protocol() -> None:
    assert isinstance(FeatureReranker(), Reranker)


def test_noop_reranker_satisfies_protocol() -> None:
    assert isinstance(NoopReranker(), Reranker)


# ---------------------------------------------------------------------------
# Noop
# ---------------------------------------------------------------------------


async def test_noop_preserves_order_and_scores() -> None:
    f1 = _make_fact("first")
    f2 = _make_fact("second")
    cands = [
        RecallResult(memory=f1, score=0.9, matched_stage="lexical"),
        RecallResult(memory=f2, score=0.5, matched_stage="lexical"),
    ]
    out = await NoopReranker().rerank("anything", cands)
    assert [r.memory.id for r in out] == [f1.id, f2.id]
    assert [r.score for r in out] == [0.9, 0.5]


# ---------------------------------------------------------------------------
# FeatureReranker
# ---------------------------------------------------------------------------


async def test_feature_reranker_promotes_keyword_match() -> None:
    """A memory that lexically matches the query should beat one that
    does not, even if the upstream stage gave them equal scores."""
    target = _make_fact("Arun lives in San Francisco.")
    distractor = _make_fact("Arun visited Boston for a conference.")
    cands = [
        RecallResult(memory=distractor, score=0.5, matched_stage="recent"),
        RecallResult(memory=target, score=0.5, matched_stage="recent"),
    ]
    # Use lexical-overlap form "lives" (matches target) so the Jaccard
    # signal differentiates without needing stemming.
    out = await FeatureReranker().rerank("Where Arun lives", cands)
    assert out[0].memory.id == target.id
    assert out[0].score >= out[1].score


async def test_feature_reranker_keeps_scores_in_zero_one() -> None:
    cands = [
        RecallResult(
            memory=_make_fact("text " * 50), score=2.0, matched_stage="x"
        ),
        RecallResult(memory=_make_fact("y"), score=-1.0, matched_stage="x"),
    ]
    out = await FeatureReranker().rerank("query", cands)
    for r in out:
        assert 0.0 <= r.score <= 1.0


async def test_feature_reranker_recency_breaks_ties() -> None:
    """When everything else is equal, newer wins."""
    older = _make_fact(
        "Arun is a Staff Engineer.",
        valid_at=_utc(2024, 1, 1),
        created_at=_utc(2024, 1, 1),
    )
    newer = _make_fact(
        "Arun is a Staff Engineer.",
        valid_at=_utc(2025, 6, 1),
        created_at=_utc(2025, 6, 1),
    )
    cands = [
        RecallResult(memory=older, score=0.4, matched_stage="lex"),
        RecallResult(memory=newer, score=0.4, matched_stage="lex"),
    ]
    out = await FeatureReranker().rerank(
        "Arun's title", cands, now=_utc(2025, 7, 1)
    )
    assert out[0].memory.id == newer.id


async def test_feature_reranker_handles_empty_candidates() -> None:
    out = await FeatureReranker().rerank("anything", [])
    assert out == []


async def test_feature_reranker_preserves_matched_stage() -> None:
    cands = [
        RecallResult(memory=_make_fact("Arun lives here."), score=0.5, matched_stage="lexical"),
        RecallResult(memory=_make_fact("Acme launched."), score=0.5, matched_stage="graph"),
    ]
    out = await FeatureReranker().rerank("Arun", cands)
    stages = {r.matched_stage for r in out}
    assert stages == {"lexical", "graph"}


async def test_feature_weights_can_disable_a_signal() -> None:
    """If keyword overlap weight is zeroed, ties are broken by other signals
    only. This documents the override surface for A/B tuning."""
    target = _make_fact("Arun lives in San Francisco.")
    distractor = _make_fact("Arun visited Boston.")
    cands = [
        RecallResult(memory=distractor, score=0.7, matched_stage="x"),
        RecallResult(memory=target, score=0.3, matched_stage="x"),
    ]
    out = await FeatureReranker(
        FeatureWeights(
            keyword_overlap=0.0,
            entity_overlap=0.0,
            recency=0.0,
            source_trust=0.0,
            base_score=1.0,
        )
    ).rerank("Where does Arun live?", cands)
    # With only the upstream score retained, the distractor's higher
    # base score puts it first.
    assert out[0].memory.id == distractor.id


# ---------------------------------------------------------------------------
# Engine integration
# ---------------------------------------------------------------------------


async def test_engine_uses_reranker_when_configured() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(
        storage=store,
        config=EngineConfig(reranker=FeatureReranker()),
    )

    async with engine:
        await engine.remember(
            "Arun lives in San Francisco.", at=_utc(2025, 1, 1)
        )
        await engine.remember(
            "Arun visited Boston for a conference.", at=_utc(2025, 2, 1)
        )

        response = await engine.recall("San Francisco", limit=5)
        assert response.results
        assert "San Francisco" in response.results[0].memory.content


async def test_engine_recall_unchanged_when_reranker_is_none() -> None:
    """Default config => no reranker => behavior must match v0.2 + entity graph."""
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store, config=EngineConfig())  # reranker=None

    async with engine:
        await engine.remember("Arun lives in San Francisco.", at=_utc(2025, 1, 1))
        response = await engine.recall("San Francisco", limit=5)
        assert response.results
        assert "San Francisco" in response.results[0].memory.content
