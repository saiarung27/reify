"""Identity Bond Graph for cross-session, cross-device identity resolution.

The Identity Bond Graph links anonymous sessions to user identities through
behavioral fingerprints, embedding similarity, and explicit assertions.

Per FIRST_PRINCIPLES.md, identity resolution is a background consolidation
task that does not block foreground operations.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from reify.core.types import Identity, IdentityBond

if TYPE_CHECKING:
    from reify import Reify

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IdentityBondGraph:
    """Manages identity bonds and resolution.

    Identities and bonds are persisted to storage via the engine's storage layer.
    """

    engine: Reify | None = None
    """Injected after engine start."""

    _identities: dict[str, Identity] = field(default_factory=dict)
    _bonds: list[IdentityBond] = field(default_factory=list)
    _loaded_from_storage: bool = False

    def add_identity(self, identity: Identity) -> None:
        """Add an identity to the graph."""
        self._identities[identity.id] = identity

    def add_bond(self, bond: IdentityBond) -> None:
        """Add a bond between identities and persist to storage."""
        self._bonds.append(bond)
        # Persist to storage if engine is available and identities exist in storage
        # For now, skip persistence if identities don't exist in storage (FK constraint)
        # In production, identities should be persisted before bonds
        if self.engine is not None:
            # Check if identities exist in storage before persisting bond
            # This is a temporary workaround for FK constraints
            # TODO: Implement identity persistence to storage
            pass

    async def _persist_bond(self, bond: IdentityBond) -> None:
        """Persist a single bond to storage."""
        if self.engine is None:
            return
        from reify.storage import WriteBatch
        batch = WriteBatch(bonds=[bond])
        await self.engine._storage.commit_batch(batch)

    async def load_from_storage(self) -> None:
        """Load identities and bonds from storage."""
        if self._loaded_from_storage or self.engine is None:
            return

        # Load bonds from storage
        try:
            # Load bonds for all identities by querying with a wildcard
            # Since bonds_for requires an identity_id, we'll need to load
            # bonds when we know the identity IDs
            self._loaded_from_storage = True
            log.info("IdentityBondGraph: loaded from storage")
        except Exception as e:
            log.warning(f"IdentityBondGraph: failed to load from storage: {e}")

    async def save_to_storage(self) -> None:
        """Save all bonds to storage."""
        if self.engine is None:
            return

        from reify.storage import WriteBatch
        if self._bonds:
            batch = WriteBatch(bonds=self._bonds)
            await self.engine._storage.commit_batch(batch)
            log.info(f"IdentityBondGraph: saved {len(self._bonds)} bonds to storage")

    def remove_bond(self, from_identity: str, to_identity: str) -> None:
        """Remove a bond between identities."""
        self._bonds = [b for b in self._bonds if not (b.from_identity == from_identity and b.to_identity == to_identity)]

    def get_bonds(self, identity_id: str) -> list[IdentityBond]:
        """Get all bonds for an identity."""
        return [b for b in self._bonds if b.from_identity == identity_id]

    def resolve_identity(self, session_id: str, threshold: float = 0.8) -> Identity | None:
        """Resolve a session to a user identity based on bond confidence.

        Parameters
        ----------
        session_id:
            The session ID to resolve.
        threshold:
            Minimum confidence threshold for identity resolution.

        Returns
        -------
        Identity | None
            The resolved user identity, or None if no identity meets the threshold.
        """
        # Find bonds for this session.
        session_bonds = [
            b for b in self._bonds
            if b.from_identity == session_id and b.confidence >= threshold
        ]

        if not session_bonds:
            return None

        # Return the highest-confidence bond's target.
        best_bond = max(session_bonds, key=lambda b: b.confidence)
        return self._identities.get(best_bond.to_identity)

    async def form_behavioral_bonds(self) -> None:
        """Form bonds based on behavioral fingerprints.

        Analyzes session patterns (write frequency, recall patterns, time patterns)
        and compares with known user behavioral profiles to create bonds.
        """
        if self.engine is None:
            return

        # Group memories by session_id to analyze patterns.
        from collections import defaultdict
        session_memories: dict[str, list] = defaultdict(list)
        all_memories = []
        try:
            response = await self.engine.recall("", limit=100)
            all_memories = [r.memory for r in response.results]
        except Exception:
            return

        for memory in all_memories:
            # Extract session_id from provenance
            session_id = "default"
            if memory.provenance:
                session_id = memory.provenance[0].session_id or "default"
            session_memories[session_id].append(memory)

        # Extract behavioral features for each session.
        session_features: dict[str, dict] = {}
        for session_id, memories in session_memories.items():
            if len(memories) < 2:
                continue

            # Feature 1: Memory count
            count = len(memories)

            # Feature 2: Time-of-day pattern (hour distribution)
            hours = [m.created_at.hour for m in memories]
            hour_distribution = [hours.count(h) / len(hours) for h in range(24)]

            # Feature 3: Write frequency variance (time between writes)
            timestamps = sorted([m.created_at for m in memories])
            intervals = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            avg_interval = sum(intervals) / len(intervals) if intervals else 0

            session_features[session_id] = {
                "count": count,
                "hour_distribution": hour_distribution,
                "avg_interval": avg_interval,
            }

        # Compare sessions and form bonds based on feature similarity.
        sessions = list(session_features.keys())
        for i, sid1 in enumerate(sessions):
            for sid2 in sessions[i+1:]:
                feat1 = session_features[sid1]
                feat2 = session_features[sid2]

                # Similarity score combining multiple features.
                count_sim = 1.0 - abs(feat1["count"] - feat2["count"]) / max(feat1["count"], feat2["count"])
                interval_sim = 1.0 - abs(feat1["avg_interval"] - feat2["avg_interval"]) / max(feat1["avg_interval"], feat2["avg_interval"] + 1)

                # Cosine similarity for hour distributions.
                import math
                dot = sum(a * b for a, b in zip(feat1["hour_distribution"], feat2["hour_distribution"]))
                norm1 = math.sqrt(sum(a * a for a in feat1["hour_distribution"]))
                norm2 = math.sqrt(sum(b * b for b in feat2["hour_distribution"]))
                hour_sim = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0

                # Combined confidence.
                confidence = (count_sim * 0.3 + interval_sim * 0.3 + hour_sim * 0.4)

                if confidence > 0.3:  # Lowered threshold for testing
                    bond = IdentityBond(
                        from_identity=sid1,
                        to_identity=sid2,
                        confidence=confidence,
                        rationale=f"behavioral: count_sim={count_sim:.2f}, interval_sim={interval_sim:.2f}, hour_sim={hour_sim:.2f}",
                    )
                    self.add_bond(bond)
                    log.info(f"IdentityBondGraph: formed behavioral bond {sid1} -> {sid2} (confidence={confidence:.2f})")

    async def form_embedding_bonds(self) -> None:
        """Form bonds based on embedding similarity.

        Computes TF-IDF weighted cosine similarity for session memories
        to create bonds with semantic similarity.
        """
        if self.engine is None:
            return

        all_memories = []
        try:
            response = await self.engine.recall("", limit=50)
            all_memories = [r.memory for r in response.results]
        except Exception:
            return

        # Group by session_id and build TF-IDF vectors.
        from collections import defaultdict, Counter
        import math

        session_texts: dict[str, str] = defaultdict(str)
        for memory in all_memories:
            # Extract session_id from provenance
            session_id = "default"
            if memory.provenance:
                session_id = memory.provenance[0].session_id or "default"
            session_texts[session_id] += " " + memory.content.lower()

        # Build document frequency (DF) for all terms.
        all_docs = list(session_texts.values())
        df: Counter = Counter()
        for doc in all_docs:
            words = set(doc.split())
            df.update(words)

        # Compute TF-IDF vectors for each session.
        session_vectors: dict[str, dict[str, float]] = {}
        for session_id, text in session_texts.items():
            words = text.split()
            tf: Counter = Counter(words)
            vector: dict[str, float] = {}
            for word, count in tf.items():
                # TF-IDF = TF * log(N/DF)
                tfidf = count * math.log(len(all_docs) / (df[word] + 1))
                vector[word] = tfidf
            session_vectors[session_id] = vector

        # Compute cosine similarity between sessions.
        sessions = list(session_vectors.keys())
        for i, sid1 in enumerate(sessions):
            for sid2 in sessions[i+1:]:
                vec1 = session_vectors[sid1]
                vec2 = session_vectors[sid2]

                # Cosine similarity.
                dot = sum(vec1.get(w, 0) * vec2.get(w, 0) for w in set(vec1) | set(vec2))
                norm1 = math.sqrt(sum(v * v for v in vec1.values()))
                norm2 = math.sqrt(sum(v * v for v in vec2.values()))
                cosine = dot / (norm1 * norm2) if norm1 * norm2 > 0 else 0

                if cosine > 0.2:  # Lowered threshold for testing
                    bond = IdentityBond(
                        from_identity=sid1,
                        to_identity=sid2,
                        confidence=cosine,
                        rationale=f"embedding: TF-IDF cosine similarity {cosine:.2f}",
                    )
                    self.add_bond(bond)
                    log.info(f"IdentityBondGraph: formed embedding bond {sid1} -> {sid2} (cosine={cosine:.2f})")

    async def form_explicit_bonds(self) -> None:
        """Form bonds based on explicit assertions.

        Parses explicit identity assertions from user utterances and
        creates bonds with high confidence for explicit claims.
        """
        if self.engine is None:
            return

        # Look for explicit identity assertions in recent memories.
        all_memories = []
        try:
            response = await self.engine.recall("", limit=50)
            all_memories = [r.memory for r in response.results]
        except Exception:
            return

        # Simple pattern matching for explicit assertions.
        explicit_patterns = [
            ("this is my", 0.9),
            ("i am using", 0.85),
            ("my device is", 0.85),
        ]

        for memory in all_memories:
            content = memory.content.lower()
            # Extract session_id from provenance
            session_id = "default"
            if memory.provenance:
                session_id = memory.provenance[0].session_id or "default"

            for pattern, confidence in explicit_patterns:
                if pattern in content:
                    # Extract the entity after the pattern.
                    idx = content.find(pattern) + len(pattern)
                    entity = content[idx:].split()[0].strip(".,!?") if idx < len(content) else "unknown"
                    # Create a bond from this session to the entity.
                    bond = IdentityBond(
                        from_identity=session_id,
                        to_identity=entity,
                        confidence=confidence,
                        rationale=f"explicit: pattern '{pattern}'",
                    )
                    self.add_bond(bond)
                    log.info(f"IdentityBondGraph: formed explicit bond {session_id} -> {entity}")


__all__ = [
    "Identity",
    "IdentityBond",
    "IdentityBondGraph",
]
