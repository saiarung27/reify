"""Tests for the v0.4 BEAM evaluation harness."""

from __future__ import annotations

import pytest

from reify import Reify
from reify.eval.beam import (
    BEAMCase,
    BEAMReport,
    default_beam_fixture_cases,
    run_beam_suite,
)
from reify.storage import SQLiteStore


async def test_beam_fixture_cases() -> None:
    cases = default_beam_fixture_cases()
    assert len(cases) == 3
    assert all(isinstance(case, BEAMCase) for case in cases)


async def test_beam_suite_with_fixture() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    async with engine:
        await engine.remember("I work at Stripe.")
        await engine.remember("I have a Border Collie named Mochi.")
        await engine.remember("I live in San Francisco.")

        report = await run_beam_suite(engine=engine, cases=default_beam_fixture_cases())
        assert isinstance(report, BEAMReport)
        assert report.total_cases == 3
        assert report.scale == 1
        assert report.dataset_label == "bundled fixture"


async def test_beam_suite_with_scale() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    async with engine:
        await engine.remember("I work at Stripe.")

        report = await run_beam_suite(
            engine=engine,
            cases=default_beam_fixture_cases(),
            scale=2,
        )
        assert report.total_cases == 6  # 3 cases * 2 scale
        assert report.scale == 2


async def test_beam_suite_latency_metrics() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    async with engine:
        await engine.remember("I work at Stripe.")

        report = await run_beam_suite(engine=engine, cases=default_beam_fixture_cases())
        assert report.p50_latency_ms >= 0
        assert report.p95_latency_ms >= 0
        assert report.p99_latency_ms >= 0


async def test_beam_suite_target_check() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    async with engine:
        await engine.remember("I work at Stripe.")

        # With low target, should pass.
        report = await run_beam_suite(
            engine=engine,
            cases=default_beam_fixture_cases(),
            target=0.1,
        )
        assert report.passed_target

        # With high target, should fail.
        report = await run_beam_suite(
            engine=engine,
            cases=default_beam_fixture_cases(),
            target=1.0,
        )
        assert not report.passed_target
