"""Cross-session synthesis consolidator.

This module performs synthesis across sessions during idle periods to improve
multi-session memory quality. It identifies related memories from different
sessions and synthesizes them into unified inferences.

Per FIRST_PRINCIPLES.md, this runs off the foreground path during sleep phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reify import Reify

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CrossSessionSynthesisConfig:
    """Configuration for cross-session synthesis."""

    enabled: bool = True
    """Whether cross-session synthesis is enabled."""

    min_sessions: int = 2
    """Minimum number of sessions to consider for synthesis."""

    similarity_threshold: float = 0.7
    """Minimum similarity threshold for considering memories related."""


@dataclass(slots=True)
class CrossSessionSynthesizer:
    """Cross-session synthesis consolidator.

    Identifies related memories across sessions and synthesizes them into
    unified inferences during idle periods.
    """

    config: CrossSessionSynthesisConfig = field(default_factory=CrossSessionSynthesisConfig)
    engine: Reify | None = None

    async def run_synthesis(self) -> int:
        """Run cross-session synthesis pass.

        Returns the number of synthesized inferences created.
        """
        if self.engine is None:
            log.warning("CrossSessionSynthesizer: engine not set, skipping synthesis")
            return 0

        if not self.config.enabled:
            log.info("CrossSessionSynthesizer: disabled, skipping")
            return 0

        log.info("CrossSessionSynthesizer: starting synthesis pass")

        # Query recent memories for synthesis.
        recent_response = await self.engine.recall("", limit=100)
        recent_memories = [r.memory for r in recent_response.results]

        if len(recent_memories) < self.config.min_sessions:
            log.info(f"CrossSessionSynthesizer: insufficient memories ({len(recent_memories)} < {self.config.min_sessions})")
            return 0

        # Group memories by identity for cross-session analysis.
        from collections import defaultdict, Counter
        import re
        from datetime import datetime, timezone

        by_identity: dict[str, list] = defaultdict(list)
        for memory in recent_memories:
            identity_id = memory.identity_id or "default"
            by_identity[identity_id].append(memory)

        synthesized_count = 0

        # For each identity with multiple memories, find patterns.
        for identity_id, memories in by_identity.items():
            if len(memories) < 2:
                continue

            # Pattern 1: Extract named entities (capitalized words, proper nouns)
            entity_counts: Counter = Counter()
            for memory in memories:
                content = memory.content
                # Extract capitalized words (2+ letters)
                entities = re.findall(r'\b[A-Z][a-z]{2,}\b', content)
                entity_counts.update(entities)

            # Create inferences for frequently mentioned entities.
            for entity, count in entity_counts.items():
                if count >= 2:
                    from reify.core.types import Inference, RelationType
                    confidence = min(0.5 + (count * 0.1), 0.95)
                    inference = Inference(
                        content=f"You frequently mention {entity} across multiple sessions.",
                        confidence=confidence,
                        identity_id=identity_id,
                        created_at=datetime.now(timezone.utc),
                        relation=RelationType.EXTENDS,
                        parent_memory_ids=tuple(m.id for m in memories[:3]),  # Link to source memories
                        statement=f"{entity} is a recurring entity in your memories.",
                    )
                    from reify.storage import WriteBatch
                    batch = WriteBatch(memories=[inference], transitions=[])
                    await self.engine._storage.commit_batch(batch)
                    synthesized_count += 1
                    log.info(f"CrossSessionSynthesizer: synthesized entity inference about {entity} (confidence={confidence:.2f})")

            # Pattern 2: Detect temporal patterns (time expressions)
            time_expressions: Counter = Counter()
            for memory in memories:
                content = memory.content.lower()
                # Common time patterns
                time_patterns = [
                    r'\b(yesterday|today|tomorrow)\b',
                    r'\b(last|this|next) (week|month|year)\b',
                    r'\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
                    r'\b(morning|afternoon|evening|night)\b',
                ]
                for pattern in time_patterns:
                    matches = re.findall(pattern, content)
                    time_expressions.update(matches)

            # Create inferences for temporal patterns.
            for time_expr, count in time_expressions.items():
                if count >= 2:
                    from reify.core.types import Inference, RelationType
                    confidence = min(0.5 + (count * 0.1), 0.9)
                    inference = Inference(
                        content=f"You frequently reference {time_expr} in your memories.",
                        confidence=confidence,
                        identity_id=identity_id,
                        created_at=datetime.now(timezone.utc),
                        relation=RelationType.EXTENDS,
                        parent_memory_ids=tuple(m.id for m in memories[:3]),
                        statement=f"{time_expr} is a recurring temporal pattern in your memories.",
                    )
                    from reify.storage import WriteBatch
                    batch = WriteBatch(memories=[inference], transitions=[])
                    await self.engine._storage.commit_batch(batch)
                    synthesized_count += 1
                    log.info(f"CrossSessionSynthesizer: synthesized temporal inference about {time_expr} (confidence={confidence:.2f})")

            # Pattern 3: Detect recurring themes using n-grams
            all_text = " ".join(m.content.lower() for m in memories)
            words = all_text.split()
            # Extract bigrams (2-word phrases)
            bigrams: Counter = Counter()
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                if len(bigram) > 5:  # Filter short bigrams
                    bigrams[bigram] += 1

            # Create inferences for recurring bigrams.
            for bigram, count in bigrams.most_common(5):
                if count >= 2:
                    from reify.core.types import Inference, RelationType
                    confidence = min(0.4 + (count * 0.15), 0.85)
                    inference = Inference(
                        content=f"The phrase '{bigram}' appears frequently in your memories.",
                        confidence=confidence,
                        identity_id=identity_id,
                        created_at=datetime.now(timezone.utc),
                        relation=RelationType.EXTENDS,
                        parent_memory_ids=tuple(m.id for m in memories[:3]),
                        statement=f"'{bigram}' is a recurring theme in your memories.",
                    )
                    from reify.storage import WriteBatch
                    batch = WriteBatch(memories=[inference], transitions=[])
                    await self.engine._storage.commit_batch(batch)
                    synthesized_count += 1
                    log.info(f"CrossSessionSynthesizer: synthesized theme inference about '{bigram}' (confidence={confidence:.2f})")

        log.info(f"CrossSessionSynthesizer: synthesized {synthesized_count} inferences")
        return synthesized_count


__all__ = [
    "CrossSessionSynthesizer",
    "CrossSessionSynthesisConfig",
]
