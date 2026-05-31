"""Ebbinghaus confidence decay for time-based memory confidence curves.

The Ebbinghaus forgetting curve describes how memories decay over time without
reinforcement. This module implements a confidence decay function that reduces
memory confidence based on time since creation or last recall, following the
exponential decay pattern R(t) = e^(-t/S) where S is the stability constant.

Per FIRST_PRINCIPLES.md, this is a recall-time scoring adjustment, not a
validity change. It affects retrieval ranking but not memory persistence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reify.core.types import Memory


@dataclass(frozen=True, slots=True)
class EbbinghausConfig:
    """Configuration for Ebbinghaus confidence decay."""

    enabled: bool = True
    """Whether to apply confidence decay."""

    stability_days: float = 30.0
    """Stability constant S in days. Higher values mean slower decay."""

    min_confidence: float = 0.1
    """Minimum confidence floor to prevent memories from becoming invisible."""

    reinforcement_factor: float = 1.5
    """Multiplier applied to stability when a memory is recalled (reinforcement)."""


def ebbinghaus_decay(
    memory: Memory,
    now: datetime,
    config: EbbinghausConfig,
    last_recall_at: datetime | None = None,
) -> float:
    """Calculate the Ebbinghaus confidence decay factor for a memory.

    The decay factor is a multiplier in [min_confidence, 1.0] that is applied
    to the memory's base confidence during recall scoring.

    Parameters
    ----------
    memory:
        The memory to calculate decay for.
    now:
        The current time for decay calculation.
    config:
        Ebbinghaus configuration.
    last_recall_at:
        Optional timestamp of last recall. If provided, decay is calculated
        from this time (reinforcement). Otherwise, decay is from creation time.

    Returns
    -------
    float
        Decay factor in [config.min_confidence, 1.0].
    """
    if not config.enabled:
        return 1.0

    # Use last recall time if available (reinforcement), otherwise creation time.
    base_time = last_recall_at if last_recall_at is not None else memory.created_at
    elapsed_seconds = (now - base_time).total_seconds()
    elapsed_days = elapsed_seconds / 86400.0

    # Apply reinforcement to stability if the memory was recalled recently.
    stability = config.stability_days
    if last_recall_at is not None:
        stability *= config.reinforcement_factor

    # Ebbinghaus formula: R(t) = e^(-t/S)
    decay_factor = math.exp(-elapsed_days / stability)

    # Apply minimum confidence floor.
    return max(decay_factor, config.min_confidence)


def apply_ebbinghaus_to_score(
    base_score: float,
    memory: Memory,
    now: datetime,
    config: EbbinghausConfig,
    last_recall_at: datetime | None = None,
) -> float:
    """Apply Ebbinghaus decay to a base recall score.

    Parameters
    ----------
    base_score:
        The base recall score (e.g., from metadata-first retrieval).
    memory:
        The memory being scored.
    now:
        Current time.
    config:
        Ebbinghaus configuration.
    last_recall_at:
        Optional last recall timestamp for reinforcement.

    Returns
    -------
    float
        The adjusted score after applying confidence decay.
    """
    decay_factor = ebbinghaus_decay(memory, now, config, last_recall_at)
    return base_score * decay_factor


__all__ = [
    "EbbinghausConfig",
    "ebbinghaus_decay",
    "apply_ebbinghaus_to_score",
]
