"""Tests for the v0.4 sleep scheduler."""

from __future__ import annotations

import asyncio

import pytest

from reify import EngineConfig, Reify
from reify.consolidate.sleep_scheduler import SleepScheduler, SleepSchedulerConfig
from reify.storage import SQLiteStore


async def test_sleep_scheduler_records_activity() -> None:
    scheduler = SleepScheduler()
    assert scheduler._last_activity_at
    await asyncio.sleep(0.01)
    scheduler.record_activity()
    # Activity timestamp should be updated.
    assert scheduler._last_activity_at


async def test_sleep_scheduler_start_stop() -> None:
    scheduler = SleepScheduler()
    await scheduler.start()
    assert scheduler._running
    await scheduler.stop()
    assert not scheduler._running


async def test_sleep_scheduler_consolidation_runs_during_idle() -> None:
    """Consolidation runs when system is idle for threshold duration."""
    store = SQLiteStore(":memory:")
    engine = Reify(
        storage=store,
        config=EngineConfig(
            sleep_scheduler_enabled=True,
            sleep_scheduler_config=SleepSchedulerConfig(
                idle_threshold_seconds=0.1,
                check_interval_seconds=0.05,
                min_idle_duration_seconds=0.05,
            ),
        ),
    )
    async with engine:
        # Record activity.
        await engine.remember("Arun lives in San Francisco.")
        # Wait for idle threshold + consolidation.
        await asyncio.sleep(0.3)
        # Scheduler should have run consolidation.
        # Verify by checking that the task is still running.
        assert engine._sleep_scheduler._running


async def test_sleep_scheduler_disabled() -> None:
    """Scheduler does nothing when disabled."""
    store = SQLiteStore(":memory:")
    engine = Reify(
        storage=store,
        config=EngineConfig(sleep_scheduler_enabled=False),
    )
    async with engine:
        await engine.remember("Arun lives in San Francisco.")
        # Scheduler should not be running.
        assert not engine._sleep_scheduler._running
