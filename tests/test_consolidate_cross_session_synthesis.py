"""Tests for cross-session synthesis consolidator."""

from __future__ import annotations

import pytest

from reify import Reify
from reify.consolidate.cross_session_synthesis import (
    CrossSessionSynthesizer,
    CrossSessionSynthesisConfig,
)
from reify.storage import SQLiteStore


@pytest.fixture
async def engine() -> Reify:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    await engine.start()
    yield engine
    await engine.stop()


def test_synthesizer_config_defaults() -> None:
    config = CrossSessionSynthesisConfig()
    assert config.enabled is True
    assert config.min_sessions == 2
    assert config.similarity_threshold == 0.7


def test_synthesizer_config_custom() -> None:
    config = CrossSessionSynthesisConfig(
        enabled=False,
        min_sessions=3,
        similarity_threshold=0.8,
    )
    assert config.enabled is False
    assert config.min_sessions == 3
    assert config.similarity_threshold == 0.8


async def test_synthesizer_without_engine() -> None:
    synthesizer = CrossSessionSynthesizer()
    count = await synthesizer.run_synthesis()
    assert count == 0


async def test_synthesizer_disabled() -> None:
    synthesizer = CrossSessionSynthesizer(
        config=CrossSessionSynthesisConfig(enabled=False)
    )
    synthesizer.engine = None
    count = await synthesizer.run_synthesis()
    assert count == 0


async def test_synthesizer_with_engine(engine: Reify) -> None:
    synthesizer = CrossSessionSynthesizer()
    synthesizer.engine = engine
    count = await synthesizer.run_synthesis()
    # Placeholder returns 0, actual implementation would create inferences
    assert count == 0
