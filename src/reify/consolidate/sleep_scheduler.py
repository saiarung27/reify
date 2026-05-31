"""Sleep-phase scheduler for background consolidation.

The sleep scheduler runs consolidation tasks during idle periods. It detects
when the system is idle (no recent writes or recalls) and triggers consolidation
passes such as cross-session synthesis, adversarial probing, and confidence decay.

Per FIRST_PRINCIPLES.md commandment 7, the sleep scheduler is *off the foreground
write/read path*: it runs in the background and does not block user operations.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reify import Reify

from reify.consolidate.adversarial import AdversarialProber
from reify.consolidate.cross_session_synthesis import (
    CrossSessionSynthesizer,
    CrossSessionSynthesisConfig,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SleepSchedulerConfig:
    """Configuration for the sleep scheduler."""

    idle_threshold_seconds: float = 60.0
    """Seconds of no activity before the system is considered idle."""

    check_interval_seconds: float = 10.0
    """How often to check for idle state."""

    min_idle_duration_seconds: float = 30.0
    """Minimum idle duration before running consolidation tasks."""


@dataclass(slots=True)
class SleepScheduler:
    """Background scheduler that runs consolidation during idle periods.

    The scheduler monitors engine activity (writes and recalls) and triggers
    consolidation when the system has been idle for a configurable duration.
    """

    config: SleepSchedulerConfig = field(default_factory=SleepSchedulerConfig)
    _adversarial_prober: AdversarialProber = field(default_factory=AdversarialProber)
    _cross_session_synthesizer: CrossSessionSynthesizer = field(
        default_factory=lambda: CrossSessionSynthesizer(
            config=CrossSessionSynthesisConfig()
        )
    )
    engine: Reify | None = None
    """Injected after engine start."""

    _last_activity_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    _task: asyncio.Task[None] | None = None
    _running: bool = False

    def record_activity(self) -> None:
        """Record that the engine is active (called on writes and recalls)."""
        self._last_activity_at = datetime.now(timezone.utc)

    async def start(self) -> None:
        """Start the background idle detection loop."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._idle_detection_loop())

    async def stop(self) -> None:
        """Stop the background loop."""
        if not self._running:
            return
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _idle_detection_loop(self) -> None:
        """Background loop that checks for idle state and runs consolidation."""
        while self._running:
            await asyncio.sleep(self.config.check_interval_seconds)
            if not self._running:
                break

            idle_seconds = (datetime.now(timezone.utc) - self._last_activity_at).total_seconds()
            if idle_seconds >= self.config.idle_threshold_seconds:
                # System is idle; wait for minimum duration before consolidating.
                await asyncio.sleep(self.config.min_idle_duration_seconds)
                if not self._running:
                    break

                # Re-check activity in case a write/recall happened during the wait.
                idle_seconds = (datetime.now(timezone.utc) - self._last_activity_at).total_seconds()
                if idle_seconds >= self.config.idle_threshold_seconds:
                    await self._run_consolidation()

    async def _run_consolidation(self) -> None:
        """Run consolidation tasks during idle period."""
        if self.engine is None:
            return

        log.info("Sleep scheduler: starting consolidation during idle period")
        try:
            # Flush ARW queue first.
            await self.engine.flush_arw()
            # Trigger decay sweep.
            await self.engine.sweep_decay(now=self.engine.now(), reason="sleep_scheduler")
            # Run adversarial probing.
            self._adversarial_prober.engine = self.engine
            probe_results = await self._adversarial_prober.run_probes()
            log.info(f"Sleep scheduler: adversarial probing complete, {len(probe_results)} probes run")
            # Run cross-session synthesis.
            self._cross_session_synthesizer.engine = self.engine
            synthesized_count = await self._cross_session_synthesizer.run_synthesis()
            log.info(f"Sleep scheduler: cross-session synthesis complete, {synthesized_count} inferences created")
            log.info("Sleep scheduler: consolidation complete")
        except Exception as e:
            log.error(f"Sleep scheduler: consolidation failed: {e}", exc_info=True)


__all__ = [
    "SleepScheduler",
    "SleepSchedulerConfig",
]
