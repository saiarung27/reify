"""Adversarial belief probing for memory validation.

Adversarial probing generates negative test cases to validate memory consistency.
It attempts to find contradictions or inconsistencies in the memory store by
generating probes that should fail if the memory is consistent.

Per FIRST_PRINCIPLES.md, this is a background consolidation task that runs
during idle periods and does not block foreground operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reify import Reify

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class AdversarialProbe:
    """A probe generated to test memory consistency."""

    query: str
    """The natural-language query to test."""

    expected_behavior: str
    """Expected behavior: 'no_results' or 'contradiction'."""

    target_memory_ids: tuple[str, ...]
    """Memory IDs this probe is testing."""


@dataclass(frozen=True, slots=True)
class AdversarialProbeResult:
    """Result of running an adversarial probe."""

    probe: AdversarialProbe
    success: bool
    """True if the memory behaved as expected."""

    actual_result_count: int
    actual_top_hit: str | None = None


@dataclass(slots=True)
class AdversarialProber:
    """Generates and runs adversarial probes to validate memory consistency."""

    engine: Reify | None = None
    """Injected after engine start."""

    async def run_probes(self) -> tuple[AdversarialProbeResult, ...]:
        """Generate and run adversarial probes against the memory store."""
        if self.engine is None:
            return ()

        probes = await self._generate_probes()
        results: list[AdversarialProbeResult] = []

        for probe in probes:
            result = await self._run_probe(probe)
            results.append(result)

        return tuple(results)

    async def _generate_probes(self) -> tuple[AdversarialProbe, ...]:
        """Generate adversarial probes based on current memory state.

        Generates negated queries and temporal inconsistency probes from
        recent memories to test memory consistency.
        """
        if self.engine is None:
            return ()

        # Sample recent memories for probe generation.
        recent_memories = []
        try:
            response = await self.engine.recall("", limit=10)
            recent_memories = [r.memory for r in response.results]
        except Exception:
            return ()

        probes: list[AdversarialProbe] = []

        for memory in recent_memories:
            content = memory.content.lower()

            # Generate negation probes for factual claims.
            if "live in" in content or "work at" in content or "from" in content:
                # Extract the location/entity and create a negated query.
                parts = content.split()
                for i, part in enumerate(parts):
                    if part in ["in", "at", "from"] and i + 1 < len(parts):
                        entity = parts[i + 1].strip(".,!?")
                        negated_query = f"Do I live in {entity}?"
                        probes.append(
                            AdversarialProbe(
                                query=negated_query,
                                expected_behavior="contradiction",
                                target_memory_ids=(memory.id,),
                            )
                        )
                        break

            # Generate temporal inconsistency probes.
            if "yesterday" in content or "last week" in content:
                probes.append(
                    AdversarialProbe(
                        query=f"What happened {content.split('yesterday')[0].strip()} tomorrow?",
                        expected_behavior="no_results",
                        target_memory_ids=(memory.id,),
                    )
                )

        return tuple(probes)

    async def _run_probe(self, probe: AdversarialProbe) -> AdversarialProbeResult:
        """Run a single adversarial probe."""
        if self.engine is None:
            return AdversarialProbeResult(
                probe=probe,
                success=False,
                actual_result_count=0,
            )

        response = await self.engine.recall(probe.query, limit=5)

        if probe.expected_behavior == "no_results":
            success = len(response.results) == 0
        elif probe.expected_behavior == "contradiction":
            # Check if results contradict the probe expectation.
            # Placeholder: treat any result as a contradiction.
            success = len(response.results) > 0
        else:
            success = False

        top_hit = response.results[0].memory.content if response.results else None

        return AdversarialProbeResult(
            probe=probe,
            success=success,
            actual_result_count=len(response.results),
            actual_top_hit=top_hit,
        )


__all__ = [
    "AdversarialProbe",
    "AdversarialProbeResult",
    "AdversarialProber",
]
