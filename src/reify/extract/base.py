"""Extractor protocol and shared types.

An extractor takes a single utterance plus context and returns a tuple of
typed memories. Multiple extractors can run in sequence; the engine merges
their outputs.

Two extractor flavors are anticipated:

* :class:`reify.extract.deterministic.DeterministicExtractor` \u2014 pure-Python
  pattern matcher. No LLM, no embedder, no network. Always runs in the
  foreground stage of the write path.
* :class:`reify.extract.llm.LLMExtractor` \u2014 schema-validated extraction via
  a configured :class:`reify.llm.LLMProvider`. Runs in the background stage,
  and only on content the deterministic extractor flagged as residual or
  ambiguous.

Both implement the same :class:`Extractor` protocol so the engine treats
them uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal, Protocol, runtime_checkable

from reify.core.types import Memory, Transition


Actor = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class ExtractionContext:
    """Per-call context handed to every extractor.

    Fields
    ------
    source:
        Free-form string describing where the utterance came from. Examples:
        ``"claude_desktop"``, ``"windsurf"``, ``"cli:remember"``, ``"ingest:browser"``.
        Surfaces in the :class:`Provenance` record attached to every emitted memory.
    actor:
        Whose turn produced the utterance.
    session_id:
        Optional opaque session identifier. Memories within the same session
        share retrieval bonuses during recall.
    identity_id:
        Optional identity scope. ``None`` means the global default identity.
    now:
        The instant the utterance is being recorded. Injectable for tests
        (the bitemporal model treats time as data, not wall-clock).
    """

    source: str
    actor: Actor = "user"
    session_id: str | None = None
    identity_id: str | None = None
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class ExtractionResult:
    """What an extractor returns.

    ``memories`` holds the typed records to commit. ``transitions`` holds
    Bitemporal Transition Graph records emitted by patterns that are
    inherently changes (e.g., "promoted from X to Y", "moved from A to
    B"). ``residual_text`` is the portion of the original utterance the
    extractor could not classify; downstream extractors (typically the
    LLM-driven one) get a chance to process it. If ``residual_text`` is
    ``None`` the input was fully handled.
    """

    memories: tuple[Memory, ...] = ()
    transitions: tuple[Transition, ...] = ()
    residual_text: str | None = None


@runtime_checkable
class Extractor(Protocol):
    """The single method every extractor exposes.

    Implementations must be pure with respect to the input \u2014 no global state,
    no mutation of ``ctx``. They return a new :class:`ExtractionResult` each
    call. Thread- and coroutine-safety is the caller's contract; the engine
    serializes write-path extractors per-utterance.
    """

    async def extract(self, text: str, ctx: ExtractionContext) -> ExtractionResult: ...


def merge_results(results: Sequence[ExtractionResult]) -> ExtractionResult:
    """Combine the output of multiple extractors into a single result.

    Memories and transitions are concatenated in extractor order.
    ``residual_text`` is propagated from the *last* extractor that
    produced a non-``None`` residual, on the assumption that extractors
    run earliest-cheapest-first and a later extractor's view of the
    residual is more accurate.
    """
    merged_mem: list[Memory] = []
    merged_tr: list[Transition] = []
    residual: str | None = None
    for r in results:
        merged_mem.extend(r.memories)
        merged_tr.extend(r.transitions)
        if r.residual_text is not None:
            residual = r.residual_text
    return ExtractionResult(
        memories=tuple(merged_mem),
        transitions=tuple(merged_tr),
        residual_text=residual,
    )
