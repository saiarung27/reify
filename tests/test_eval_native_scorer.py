"""Tests for the Reify-native multi-session synthesis scorer."""

from __future__ import annotations

import pytest

from reify import Reify
from reify.eval.native_scorer import (
    NativeScorer,
    NativeScorerConfig,
    NativeScorerResult,
)
from reify.storage import SQLiteStore


@pytest.fixture
async def engine() -> Reify:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    await engine.start()
    yield engine
    await engine.stop()


def test_native_scorer_config_defaults() -> None:
    config = NativeScorerConfig()
    assert config.continuity_threshold == 0.8
    assert config.abstention_threshold == 0.3


def test_native_scorer_config_custom() -> None:
    config = NativeScorerConfig(
        continuity_threshold=0.9,
        abstention_threshold=0.4,
    )
    assert config.continuity_threshold == 0.9
    assert config.abstention_threshold == 0.4


async def test_native_scorer_continuity(engine: Reify) -> None:
    scorer = NativeScorer()
    score = await scorer.score_continuity(engine, "test query")
    # Placeholder returns 0.0
    assert score == 0.0


async def test_native_scorer_abstention(engine: Reify) -> None:
    scorer = NativeScorer()
    score = await scorer.score_abstention(engine, "test query")
    # Empty engine -> correct abstention -> 1.0
    assert score == 1.0


async def test_native_scorer_provenance(engine: Reify) -> None:
    scorer = NativeScorer()
    score = await scorer.score_provenance(engine, "test query")
    # Placeholder returns 0.0
    assert score == 0.0


async def test_native_scorer_update_truth(engine: Reify) -> None:
    scorer = NativeScorer()
    score = await scorer.score_update_truth(engine, "test query")
    # Placeholder returns 0.0
    assert score == 0.0


async def test_native_scorer_case(engine: Reify) -> None:
    scorer = NativeScorer()
    result = await scorer.score_case(engine, "test query")
    assert isinstance(result, NativeScorerResult)
    # Empty engine: no recall results, no transitions.
    # Continuity, provenance, update_truth = 0.0; abstention = 1.0 (correct abstention).
    assert result.continuity_score == 0.0
    assert result.abstention_score == 1.0
    assert result.provenance_score == 0.0
    assert result.update_truth_score == 0.0
    assert result.overall_score == 0.25
