"""Tests for the v0.4 adversarial probing."""

from __future__ import annotations

import pytest

from reify import Reify
from reify.consolidate.adversarial import (
    AdversarialProbe,
    AdversarialProbeResult,
    AdversarialProber,
)
from reify.storage import SQLiteStore


async def test_adversarial_prober_generates_no_probes_placeholder() -> None:
    """Placeholder implementation returns no probes."""
    prober = AdversarialProber()
    probes = await prober._generate_probes()
    assert probes == ()


async def test_adversarial_prober_runs_probe_no_results() -> None:
    """Test running a probe that expects no results."""
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    prober = AdversarialProber(engine=engine)

    async with engine:
        probe = AdversarialProbe(
            query="I live on Mars",
            expected_behavior="no_results",
            target_memory_ids=(),
        )

        result = await prober._run_probe(probe)
        assert result.probe == probe
        assert result.success  # No results, so success
        assert result.actual_result_count == 0


async def test_adversarial_prober_runs_probe_with_results() -> None:
    """Test running a probe when results are returned."""
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    prober = AdversarialProber(engine=engine)

    async with engine:
        await engine.remember("I live in San Francisco.")

        probe = AdversarialProbe(
            query="San Francisco",
            expected_behavior="no_results",
            target_memory_ids=(),
        )

        result = await prober._run_probe(probe)
        assert result.probe == probe
        assert not result.success  # Results returned when none expected
        assert result.actual_result_count > 0
        assert result.actual_top_hit is not None


async def test_adversarial_prober_run_probes_without_engine() -> None:
    """Prober returns empty results when no engine is set."""
    prober = AdversarialProber(engine=None)
    results = await prober.run_probes()
    assert results == ()


async def test_adversarial_prober_run_probes_with_engine() -> None:
    """Prober runs probes when engine is set."""
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    prober = AdversarialProber(engine=engine)

    async with engine:
        await engine.remember("I live in San Francisco.")

    results = await prober.run_probes()
    # Placeholder implementation returns no probes, so no results.
    assert results == ()
