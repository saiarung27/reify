"""Relation extraction for the Zep-graph adapter.

Why this module exists: Reify's deterministic entity indexer emits a
typed edge only for :class:`Fact` memories (subject/predicate/object).
Simulation posts are free-text opinions, so without extraction the
graph carries mentions plus ``co_occurs`` edges and nothing else --
verified on persona-style posts (0 typed edges). Zep, by contrast,
LLM-extracts relations from free text. Parity therefore needs an
extraction step, and it must be opt-in: extraction costs an LLM call
per episode and must never break the write path.

Design:

* :class:`RelationExtractor` is a tiny sync protocol:
  ``extract(text) -> list[RelationTriple]``. The adapter runs it in a
  thread so the event loop never blocks.
* :class:`LLMRelationExtractor` implements it over Reify's own
  :class:`LLMProvider` (OpenAI-compatible). Temperature 0, strict JSON
  list output, max 8 triples. Any provider error or unparseable output
  yields ``[]`` -- the episode is still stored and the graph still has
  its deterministic edges. Fail-soft by design, never fail-loud.
* Triples use short snake_case relation labels (``finds_too_expensive``)
  so they behave like Reify's predicate vocabulary. The adapter renders
  edge facts as ``"<subject label> <relation> <object label>"``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Protocol, Sequence

from reify.llm.base import LLMProvider, LLMRequest

log = logging.getLogger(__name__)

# NOTE: no str.format() on this template -- the JSON example's braces
# would be parsed as placeholders (that bug shipped once: KeyError
# on '"subject"'). The caller appends the text instead.
_PROMPT = """Extract subject-relation-object triples from the text below.
Return ONLY a JSON list, no prose. Each item: {"subject": ..., "relation": ..., "object": ...}.
Rules: subjects/objects are short noun phrases from the text (people, groups, prices, products);
relations are short snake_case verbs (finds_too_expensive, would_pay_only_if, worries_about);
at most 8 triples; skip triples you are unsure about; empty list [] when nothing is extractable.

Text:
"""


@dataclass(frozen=True)
class RelationTriple:
    subject: str
    relation: str
    object: str


class RelationExtractor(Protocol):
    """Sync triple extraction from free text. May return []."""

    def extract(self, text: str) -> Sequence[RelationTriple]: ...


class LLMRelationExtractor:
    """Relation extraction over a Reify LLM provider (async under the hood).

    ``extract`` is sync for the adapter's thread-offload call; it runs
    the provider call on a fresh loop only when no loop is running in
    this thread (the adapter always calls it from a worker thread).
    """

    def __init__(self, provider: LLMProvider, *, model: str, max_triples: int = 8) -> None:
        self._provider = provider
        self._model = model
        self._max_triples = max_triples

    def extract(self, text: str) -> list[RelationTriple]:
        import asyncio

        try:
            return asyncio.run(self._extract_async(text))
        except RuntimeError:
            # A loop is already running in this thread; run it as a task
            # is the caller's job -- fail soft instead of deadlocking.
            log.warning("relation_extraction_skipped_loop_running")
            return []
        except Exception as exc:  # noqa: BLE001 - extraction is best-effort
            log.warning("relation_extraction_failed", extra={"error": repr(exc)})
            return []

    async def _extract_async(self, text: str) -> list[RelationTriple]:
        try:
            resp = await self._provider.complete(
                LLMRequest(
                    messages=[{"role": "user", "content": _PROMPT + text}],
                    model=self._model,
                    temperature=0.0,
                )
            )
        except Exception as exc:  # noqa: BLE001 - extraction is best-effort
            log.warning("relation_extraction_failed", extra={"error": repr(exc)})
            return []
        try:
            raw = json.loads(resp.content)
        except (json.JSONDecodeError, TypeError):
            # Tolerate fenced JSON from providers that wrap output.
            try:
                fenced = resp.content.strip().strip("`")
                if fenced.lower().startswith("json"):
                    fenced = fenced[4:].strip()
                raw = json.loads(fenced)
            except (json.JSONDecodeError, TypeError, AttributeError):
                log.warning("relation_extraction_unparseable")
                return []
        if not isinstance(raw, list):
            return []
        triples: list[RelationTriple] = []
        for item in raw[: self._max_triples]:
            if not isinstance(item, dict):
                continue
            subj = str(item.get("subject", "")).strip()
            rel = str(item.get("relation", "")).strip()
            obj = str(item.get("object", "")).strip()
            if subj and rel and obj and subj != obj:
                triples.append(RelationTriple(subject=subj, relation=rel, object=obj))
        return triples
