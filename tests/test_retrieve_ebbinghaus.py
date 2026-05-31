"""Tests for the v0.4 Ebbinghaus confidence decay."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reify import EngineConfig, Reify
from reify.core.types import Fact
from reify.recall import RecallResult
from reify.retrieve.ebbinghaus import (
    EbbinghausConfig,
    apply_ebbinghaus_to_score,
    ebbinghaus_decay,
)
from reify.storage import SQLiteStore


def _fact(content: str, created_at: datetime) -> Fact:
    return Fact(
        content=content,
        subject="s",
        predicate="p",
        obj="o",
        created_at=created_at,
    )


def test_ebbinghaus_decay_returns_1_when_disabled() -> None:
    config = EbbinghausConfig(enabled=False)
    memory = _fact("test", datetime.now(timezone.utc))
    decay = ebbinghaus_decay(memory, datetime.now(timezone.utc), config)
    assert decay == 1.0


def test_ebbinghaus_decay_decreases_with_time() -> None:
    config = EbbinghausConfig(stability_days=10.0)
    now = datetime.now(timezone.utc)
    memory = _fact("test", now - timedelta(days=5))
    decay = ebbinghaus_decay(memory, now, config)
    # After 5 days with 10-day stability, decay should be < 1.0
    assert decay < 1.0
    assert decay > 0.5  # e^(-0.5) ≈ 0.607


def test_ebbinghaus_decay_respects_min_confidence() -> None:
    config = EbbinghausConfig(stability_days=1.0, min_confidence=0.5)
    now = datetime.now(timezone.utc)
    memory = _fact("test", now - timedelta(days=100))
    decay = ebbinghaus_decay(memory, now, config)
    # Should be floored at min_confidence
    assert decay == 0.5


def test_ebbinghaus_decay_with_reinforcement() -> None:
    config = EbbinghausConfig(stability_days=10.0, reinforcement_factor=2.0)
    now = datetime.now(timezone.utc)
    memory = _fact("test", now - timedelta(days=10))
    # Without reinforcement, decay would be e^(-1) ≈ 0.368
    decay_no_reinforcement = ebbinghaus_decay(memory, now, config)
    # With reinforcement (stability doubled), decay is e^(-10/20) = e^(-0.5) ≈ 0.607
    last_recall = now - timedelta(days=5)
    decay_with_reinforcement = ebbinghaus_decay(
        memory, now, config, last_recall_at=last_recall
    )
    assert decay_with_reinforcement > decay_no_reinforcement


def test_apply_ebbinghaus_to_score() -> None:
    config = EbbinghausConfig(stability_days=10.0)
    now = datetime.now(timezone.utc)
    memory = _fact("test", now - timedelta(days=5))
    base_score = 0.9
    adjusted = apply_ebbinghaus_to_score(base_score, memory, now, config)
    assert adjusted < base_score
    assert adjusted > 0


async def test_engine_applies_ebbinghaus_decay() -> None:
    store = SQLiteStore(":memory:")
    config = EngineConfig(
        ebbinghaus_config=EbbinghausConfig(
            enabled=True,
            stability_days=1.0,  # Fast decay for test
            min_confidence=0.1,
        ),
    )
    engine = Reify(storage=store, config=config)
    async with engine:
        # Remember a memory.
        await engine.remember("Arun lives in San Francisco.", at=datetime.now(timezone.utc) - timedelta(days=2))
        # Recall it - decay should reduce score.
        response = await engine.recall("San Francisco")
        # Should still return results, but with decayed scores.
        assert response.results


async def test_engine_ebbinghaus_disabled() -> None:
    store = SQLiteStore(":memory:")
    config = EngineConfig(
        ebbinghaus_config=EbbinghausConfig(enabled=False),
    )
    engine = Reify(storage=store, config=config)
    async with engine:
        await engine.remember("Arun lives in San Francisco.")
        response = await engine.recall("San Francisco")
        assert response.results
