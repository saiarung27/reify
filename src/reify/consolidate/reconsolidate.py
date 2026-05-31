"""Active Reconsolidation on Write (ARW).

This worker turns newly committed claims into materialized conclusions and
transitions. It is intentionally background-oriented: the foreground write
path commits first, enqueues durable work, and returns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from collections.abc import Callable
from typing import TYPE_CHECKING

from reify.core.types import (
    Experience,
    Fact,
    Inference,
    Memory,
    Network,
    RelationType,
    Transition,
    TransitionKind,
)
from reify.llm.base import LLMProvider, LLMRequest
from reify.storage.base import RecallFilter, Storage

log = logging.getLogger(__name__)

Clock = Callable[[], datetime]


@dataclass(frozen=True)
class Relation:
    """Relationship between two memories."""

    type: RelationType
    confidence: float
    explanation: str | None = None


@dataclass(frozen=True)
class ARWResult:
    """Derived artifacts from reconsolidating one memory."""

    inferences: tuple[Inference, ...] = ()
    transitions: tuple[Transition, ...] = ()
    updates: tuple[tuple[str, datetime], ...] = ()


class ARWWorker:
    """Active Reconsolidation on Write worker."""

    def __init__(
        self,
        storage: Storage,
        llm: LLMProvider | None = None,
        *,
        k_candidates: int = 8,
        clock: Clock | None = None,
    ) -> None:
        self._storage = storage
        self._llm = llm
        self._k_candidates = k_candidates
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def enabled(self) -> bool:
        return self._llm is not None

    async def process(self, memory: Memory) -> ARWResult:
        """Process one memory and derive inferences/transitions.

        Raises on provider errors so the queue row can be retained for retry.
        """
        if self._llm is None:
            raise RuntimeError("ARW requires a configured LLM provider")
        if not self._is_reconsolidatable(memory):
            return ARWResult()

        candidates = await self._retrieve_candidates(memory)
        if not candidates:
            return ARWResult()

        relations = await self._classify_relationships(memory, candidates)

        inferences: list[Inference] = []
        transitions: list[Transition] = []
        updates: list[tuple[str, datetime]] = []
        for candidate, relation in zip(candidates, relations, strict=True):
            if relation.type is RelationType.NONE:
                continue
            inferences.append(self._create_inference(memory, candidate, relation))
            if relation.type in (RelationType.CONTRADICTS, RelationType.RESOLVES):
                transition = self._create_transition(memory, candidate, relation)
                transitions.append(transition)
                updates.append((candidate.id, transition.at))

        return ARWResult(
            inferences=tuple(inferences),
            transitions=tuple(transitions),
            updates=tuple(updates),
        )

    def _is_reconsolidatable(self, memory: Memory) -> bool:
        return isinstance(memory, Fact) and memory.valid_until is None

    async def _retrieve_candidates(self, memory: Memory) -> list[Fact]:
        """Retrieve recent, identity-scoped claim candidates for one memory."""
        since = self._clock() - timedelta(days=90)
        filter_ = RecallFilter(
            identity_ids=(memory.identity_id,) if memory.identity_id else None,
            networks=(Network.FACT,),
            since=since,
            include_deleted=False,
        )
        candidates = await self._storage.recall(filter_, limit=max(self._k_candidates * 4, 16))
        out: list[Fact] = []
        seen_ids: set[str] = set()
        for candidate in candidates:
            if not isinstance(candidate, Fact):
                continue
            if candidate.id == memory.id or candidate.id in seen_ids:
                continue
            if candidate.predicate != memory.predicate:
                continue
            if not self._shares_context(memory, candidate):
                continue
            out.append(candidate)
            seen_ids.add(candidate.id)
            if len(out) >= self._k_candidates:
                break
        return out

    def _shares_context(self, memory: Fact, candidate: Fact) -> bool:
        if memory.subject == candidate.subject and memory.predicate == candidate.predicate:
            return True
        return bool(set(memory.tags) & set(candidate.tags))

    async def _classify_relationships(
        self, memory: Fact, candidates: list[Fact]
    ) -> list[Relation]:
        prompt = self._build_classification_prompt(memory, candidates)
        model = getattr(self._llm, "default_model", None) or "gpt-5-mini"
        request = LLMRequest(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You classify relationships between memories. "
                        "Return JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=600,
        )
        response = await self._llm.complete(request)
        return self._parse_classification_response(response.content, len(candidates))

    def _build_classification_prompt(self, memory: Fact, candidates: list[Fact]) -> str:
        prompt = """Classify the relation between the NEW fact and each EXISTING fact.

Allowed relation values:
- none
- confirms
- contradicts
- extends
- refines
- resolves

Rules:
- Use contradicts for two incompatible current states of the same predicate.
- Use resolves when the new fact clearly supersedes the older fact as the new current truth.
- Use confirms when they assert the same state.
- Use extends or refines only when both can remain true together.
- Output JSON array only. Each element must include: type, confidence, explanation.

NEW fact:
"""
        prompt += (
            f"content={memory.content}\n"
            f"subject={memory.subject}\n"
            f"predicate={memory.predicate}\n"
            f"object={memory.obj}\n"
            f"valid_at={memory.valid_at.isoformat()}\n"
        )

        prompt += "\nEXISTING facts:\n"
        for i, candidate in enumerate(candidates, start=1):
            valid_until = candidate.valid_until.isoformat() if candidate.valid_until else "open"
            prompt += (
                f"{i}. content={candidate.content}; "
                f"subject={candidate.subject}; "
                f"predicate={candidate.predicate}; "
                f"object={candidate.obj}; "
                f"valid_at={candidate.valid_at.isoformat()}; "
                f"valid_until={valid_until}\n"
            )
        return prompt

    def _parse_classification_response(
        self, response: str, num_candidates: int
    ) -> list[Relation]:
        normalized = self._normalize_json_payload(response)
        try:
            data = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError(f"ARW response was not valid JSON: {response}") from exc
        if not isinstance(data, list):
            raise ValueError(f"ARW response must be a JSON array: {response}")

        relations: list[Relation] = []
        for item in data[:num_candidates]:
            if not isinstance(item, dict):
                relations.append(Relation(type=RelationType.NONE, confidence=0.0))
                continue
            raw_type = str(item.get("type", "none")).lower()
            try:
                relation_type = RelationType(raw_type)
            except ValueError:
                relation_type = RelationType.NONE
            confidence = float(item.get("confidence", 0.0))
            confidence = min(max(confidence, 0.0), 1.0)
            explanation = item.get("explanation")
            relations.append(
                Relation(
                    type=relation_type,
                    confidence=confidence,
                    explanation=str(explanation) if explanation is not None else None,
                )
            )

        while len(relations) < num_candidates:
            relations.append(Relation(type=RelationType.NONE, confidence=0.0))
        return relations

    def _normalize_json_payload(self, response: str) -> str:
        text = response.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines:
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        start = min(
            [idx for idx in (text.find("["), text.find("{")) if idx != -1],
            default=-1,
        )
        if start > 0:
            text = text[start:].strip()
        return text

    def _create_inference(
        self, memory: Fact, candidate: Fact, relation: Relation
    ) -> Inference:
        statement = self._inference_statement(memory, candidate, relation)
        inherited_tags = tuple(
            dict.fromkeys(
                ("inference", relation.type.value, memory.predicate, *candidate.tags, *memory.tags)
            )
        )
        return Inference(
            content=statement,
            relation=relation.type,
            parent_memory_ids=(candidate.id, memory.id),
            statement=statement,
            valid_at=memory.valid_at,
            confidence=relation.confidence,
            evidence_strength=relation.confidence,
            identity_id=memory.identity_id,
            tags=inherited_tags,
            derivation_method="arw:v1",
            metadata={"explanation": relation.explanation or ""},
            provenance=memory.provenance,
        )

    def _inference_statement(
        self, memory: Fact, candidate: Fact, relation: Relation
    ) -> str:
        if relation.type in (RelationType.CONTRADICTS, RelationType.RESOLVES):
            return (
                f"{memory.subject} {self._predicate_label(memory.predicate)} changed "
                f"from {candidate.obj} to {memory.obj}."
            )
        if relation.type is RelationType.CONFIRMS:
            return (
                f"{memory.subject} {self._predicate_label(memory.predicate)} is consistently "
                f"recorded as {memory.obj}."
            )
        return (
            f"{memory.subject} {self._predicate_label(memory.predicate)} relates to "
            f"{candidate.obj} and {memory.obj}."
        )

    def _create_transition(
        self, memory: Fact, candidate: Fact, relation: Relation
    ) -> Transition:
        kind = self._transition_kind(memory.predicate)
        evidence = tuple(
            p.raw_event_id for p in memory.provenance if p.raw_event_id
        ) or tuple(
            p.raw_event_id for p in candidate.provenance if p.raw_event_id
        )
        return Transition(
            kind=kind,
            from_memory_id=candidate.id,
            to_memory_id=memory.id,
            at=memory.valid_at,
            recorded_at=self._clock(),
            confidence=relation.confidence,
            summary=(
                f"{memory.subject} {self._predicate_label(memory.predicate)} changed "
                f"from {candidate.obj} to {memory.obj}."
            ),
            identity_id=None,
            evidence=evidence,
        )

    def _transition_kind(self, predicate: str) -> TransitionKind:
        if predicate == "has_title":
            return TransitionKind.ROLE_CHANGE
        if predicate == "lives_in":
            return TransitionKind.LOCATION_CHANGE
        if predicate == "works_at":
            return TransitionKind.EMPLOYER_CHANGE
        return TransitionKind.GENERIC

    def _predicate_label(self, predicate: str) -> str:
        labels = {
            "has_title": "title",
            "lives_in": "location",
            "works_at": "employer",
        }
        return labels.get(predicate, predicate.replace("_", " "))
