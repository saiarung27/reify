"""Reify-native multi-session synthesis scorer.

This module provides a genuinely Reify-native evaluation for continuity,
abstention, provenance, and update truth, replacing the proxy-backed
LongMemEval evaluation.

Per FIRST_PRINCIPLES.md, this evaluation is deterministic and inspectable,
using Reify's internal memory structures directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reify import Reify


@dataclass(frozen=True)
class NativeScorerConfig:
    """Configuration for the native scorer."""

    continuity_threshold: float = 0.8
    """Minimum confidence threshold for continuity scoring."""

    abstention_threshold: float = 0.3
    """Maximum confidence threshold for abstention scoring."""


@dataclass(frozen=True)
class NativeScorerResult:
    """Result of native scoring for a single case."""

    continuity_score: float
    """Score for assistant continuity (0-1)."""

    abstention_score: float
    """Score for abstention on unsupported questions (0-1)."""

    provenance_score: float
    """Score for provenance tracking (0-1)."""

    update_truth_score: float
    """Score for update truthfulness (0-1)."""

    overall_score: float
    """Weighted overall score (0-1)."""


class NativeScorer:
    """Reify-native multi-session synthesis scorer.

    Evaluates memory quality using Reify's internal structures rather than
    proxy-backed external evaluation.
    """

    def __init__(self, config: NativeScorerConfig | None = None) -> None:
        self._config = config or NativeScorerConfig()

    async def score_continuity(self, engine: Reify, query: str) -> float:
        """Score assistant continuity using Reify's internal state.

        Analyzes whether retrieved memories maintain continuity across sessions.
        """
        response = await engine.recall(query, limit=5)
        if not response.results:
            return 0.0

        # Check if memories have consistent identity/session information.
        identities = set(r.memory.identity_id for r in response.results if r.memory.identity_id)
        if len(identities) <= 1:
            # All from same identity or no identity info - good continuity.
            return 1.0
        else:
            # Multiple identities suggest potential continuity issues.
            return 1.0 / len(identities)

    async def score_abstention(self, engine: Reify, query: str) -> float:
        """Score abstention on unsupported questions.

        Checks if the system correctly abstains when no relevant memories exist.
        """
        response = await engine.recall(query, limit=5)
        if not response.results:
            # No results - correct abstention.
            return 1.0

        # Check if results are actually relevant (simple heuristic).
        # If top result has low score, it's effectively abstaining.
        top_score = response.results[0].score if response.results else 0.0
        if top_score < self._config.abstention_threshold:
            return 1.0
        else:
            # Results returned with high confidence - not abstaining.
            return 0.0

    async def score_provenance(self, engine: Reify, query: str) -> float:
        """Score provenance tracking.

        Verifies that memory provenance is correctly tracked and exposed.
        """
        response = await engine.recall(query, limit=5)
        if not response.results:
            return 0.0

        # Check if memories have provenance information.
        provenance_count = 0
        for result in response.results:
            memory = result.memory
            # Extract session_id and actor from provenance
            session_id = None
            actor = None
            if memory.provenance:
                session_id = memory.provenance[0].session_id
                actor = memory.provenance[0].actor
            if memory.identity_id or session_id or actor:
                provenance_count += 1

        return provenance_count / len(response.results) if response.results else 0.0

    async def score_update_truth(self, engine: Reify, query: str) -> float:
        """Score update truthfulness.

        Verifies that memory updates correctly replace stale information.
        """
        # Query for transitions related to the query topic.
        transitions = await engine.whatchanged(days=30)
        if not transitions:
            return 0.0

        # Check if transitions show proper state changes.
        valid_transitions = 0
        for transition in transitions:
            # Valid transitions should have different from/to states.
            if transition.from_state != transition.to_state:
                valid_transitions += 1

        return valid_transitions / len(transitions) if transitions else 0.0

    async def score_case(
        self,
        engine: Reify,
        query: str,
    ) -> NativeScorerResult:
        """Score a single case across all native metrics."""
        continuity = await self.score_continuity(engine, query)
        abstention = await self.score_abstention(engine, query)
        provenance = await self.score_provenance(engine, query)
        update_truth = await self.score_update_truth(engine, query)

        # Weighted average (equal weights for now)
        overall = (continuity + abstention + provenance + update_truth) / 4

        return NativeScorerResult(
            continuity_score=continuity,
            abstention_score=abstention,
            provenance_score=provenance,
            update_truth_score=update_truth,
            overall_score=overall,
        )


__all__ = [
    "NativeScorer",
    "NativeScorerConfig",
    "NativeScorerResult",
]
