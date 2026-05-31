"""The Reify data model.

This module defines every memory artifact the engine knows about. The design
follows section 3 of ``ARCHITECTURE.md``.

Five memory networks
--------------------
* :class:`Fact` — declarative atoms with bitemporal validity.
* :class:`Experience` — time-stamped episodic events.
* :class:`Observation` — passive captures (browser, screen, file changes).
* :class:`Belief` — confidence-scored inferred patterns.
* :class:`Inference` — *materialized* cross-session conclusions. The novel
  network. Created by Active Reconsolidation on Write (ARW) and the sleep
  phase consolidator.

Bitemporal change
-----------------
* :class:`Transition` — a first-class change event linking memories. This is
  what distinguishes Reify from Zep, which only stores ``valid_at`` /
  ``valid_until`` ranges. Reify additionally stores *the change itself* with
  evidence trails.

Identity
--------
* :class:`Identity` — a user, device, or anonymous session.
* :class:`IdentityBond` — a weighted edge in the Identity Bond Graph.

All models are frozen by default. Mutation is achieved by writing a new record
and (for facts) by emitting a Transition that bounds the prior record's
validity. This is the bitemporal discipline that gives the engine its
correctness guarantees.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from reify.core.ids import MemoryId, new_id


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Network(StrEnum):
    """The five memory networks. See module docstring for semantics."""

    FACT = "fact"
    EXPERIENCE = "experience"
    OBSERVATION = "observation"
    BELIEF = "belief"
    INFERENCE = "inference"


class RelationType(StrEnum):
    """How a new memory relates to an existing one.

    Emitted by Active Reconsolidation on Write (ARW). Each non-``NONE``
    classification produces a derived :class:`Inference`.
    """

    NONE = "none"
    CONFIRMS = "confirms"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    REFINES = "refines"
    RESOLVES = "resolves"


class TransitionKind(StrEnum):
    """High-level taxonomy of state transitions for the BTG."""

    GENERIC = "generic"
    LOCATION_CHANGE = "location_change"
    EMPLOYER_CHANGE = "employer_change"
    RELATIONSHIP_CHANGE = "relationship_change"
    PREFERENCE_CHANGE = "preference_change"
    GOAL_CHANGE = "goal_change"
    ROLE_CHANGE = "role_change"


# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------


Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
"""A scalar in ``[0, 1]`` representing belief in a memory's accuracy."""


def _utcnow() -> datetime:
    """Return the current UTC time. Indirection to make tests deterministic."""
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """The chain of custody for a memory.

    A single memory can carry multiple :class:`Provenance` entries when it has
    been merged, deduplicated, or reconsolidated.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    """Logical source identifier. Examples: ``claude_desktop``, ``cursor``,
    ``user``, ``extractor:openai:v1``. Convention: ``namespace[:detail]``."""

    session_id: str | None = None
    """Session that produced this provenance entry, if applicable."""

    actor: Literal["user", "assistant", "system", "extractor"] = "user"
    """Who or what authored the underlying event."""

    raw_event_id: str | None = None
    """Optional pointer back to a raw, unprocessed event in an upstream log."""

    recorded_at: datetime = Field(default_factory=_utcnow)
    """When this provenance entry was attached."""


# ---------------------------------------------------------------------------
# Memory base
# ---------------------------------------------------------------------------


class _MemoryBase(BaseModel):
    """Fields shared by every memory in every network.

    Bitemporal semantics
    ~~~~~~~~~~~~~~~~~~~~
    * ``valid_at`` is the *world time* at which this memory's claim begins to
      be true.
    * ``valid_until`` is the world time at which it stops being true. ``None``
      means open-ended (still believed to be true).
    * ``created_at`` is the *system time* at which the record was written.
      It is independent of ``valid_at``: a memory can describe something that
      happened in the past or will happen in the future.

    Together these support queries such as ``as_of(t)`` (what did the system
    believe at time ``t``) and ``during(a, b)`` (what changed between two
    points), which are the foundation of the Bitemporal Transition Graph.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: MemoryId = Field(default_factory=new_id)
    """Stable, opaque, time-ordered identifier."""

    content: str
    """Canonical text representation. This is what gets embedded and what the
    retrieval pipeline returns when packing context."""

    created_at: datetime = Field(default_factory=_utcnow)
    """System time of record creation."""

    valid_at: datetime = Field(default_factory=_utcnow)
    """World time at which the memory's claim becomes valid."""

    valid_until: datetime | None = None
    """World time at which the claim stops being valid. ``None`` = open."""

    confidence: Confidence = 1.0
    """Belief in this memory's accuracy. Decays for :class:`Belief`."""

    identity_id: str | None = None
    """The :class:`Identity` that owns this memory. ``None`` = global."""

    provenance: tuple[Provenance, ...] = ()
    """Chain of custody. Append-only; never mutated in place."""

    tags: tuple[str, ...] = ()
    """Free-form labels for filtered retrieval."""

    metadata: dict[str, Any] = Field(default_factory=dict)
    """Open-ended structured payload. Avoid placing semantically-significant
    data here; prefer adding a typed field."""

    soft_deleted_at: datetime | None = None
    """If set, the memory is tombstoned and excluded from default retrieval."""


# ---------------------------------------------------------------------------
# The five networks
# ---------------------------------------------------------------------------


class Fact(_MemoryBase):
    """A declarative atom: subject–predicate–object with bitemporal validity.

    Examples
    --------
    * ``("Arun", "lives_in", "San Francisco")``
    * ``("project:reify", "uses", "sqlite-vec")``
    """

    network: Literal[Network.FACT] = Network.FACT

    subject: str
    """The entity the claim is about."""

    predicate: str
    """The relation. Should be drawn from a controlled vocabulary where
    possible to support graph queries."""

    obj: str
    """The value or entity on the other side of the relation. (``object`` is
    a Python builtin, so we use ``obj``.)"""


class Experience(_MemoryBase):
    """A time-stamped event: a conversation turn, tool call, or session
    boundary.

    Episodic memory in the cognitive-science sense.
    """

    network: Literal[Network.EXPERIENCE] = Network.EXPERIENCE

    event_type: str
    """Examples: ``user_message``, ``assistant_message``, ``tool_call``,
    ``session_start``, ``session_end``."""

    actor: Literal["user", "assistant", "system"] = "user"
    summary: str
    details: str = ""


class Observation(_MemoryBase):
    """A passive capture from a sensor: browser tab, screen frame, file
    change.

    Lower trust than :class:`Experience` because the user did not necessarily
    intend the event to be remembered.
    """

    network: Literal[Network.OBSERVATION] = Network.OBSERVATION

    capture_kind: str
    """Examples: ``browser``, ``screen``, ``file_change``, ``shell_history``."""

    summary: str
    raw_ref: str | None = None
    """Opaque pointer to a large raw payload (image path, log line range)."""


class Belief(_MemoryBase):
    """An inferred pattern with confidence and evidence.

    Beliefs decay over time without confirming evidence. They are subject to
    adversarial probing during the sleep phase: counter-hypotheses are
    generated and checked against subsequent :class:`Experience` and
    :class:`Observation` records.
    """

    network: Literal[Network.BELIEF] = Network.BELIEF

    statement: str
    evidence: tuple[MemoryId, ...] = ()
    last_validated_at: datetime | None = None
    decay_half_life_days: float = 30.0
    """Confidence halves every ``decay_half_life_days`` without revalidation.
    Set to ``inf`` to disable decay."""


class Inference(_MemoryBase):
    """A materialized cross-session conclusion. The novel fifth network.

    Created by Active Reconsolidation on Write (ARW) at write time, or by the
    sleep-phase synthesis pass during idle periods. Each inference records
    the relation type, the parent memories it combines, and the strength of
    the evidence supporting it.

    Why this is important
    ~~~~~~~~~~~~~~~~~~~~~
    Existing systems leave cross-session synthesis to the model at retrieval
    time. Reify materializes the conclusion as a first-class memory so that
    the retrieval pipeline can return it directly. This is the wedge that
    breaks the 87 % multi-session ceiling that current leaders admit they hit.
    """

    network: Literal[Network.INFERENCE] = Network.INFERENCE

    relation: RelationType
    """How the parent memories relate. Never ``NONE`` for a stored inference."""

    parent_memory_ids: tuple[MemoryId, ...]
    """The memories that were combined to produce this inference. At least
    one; typically two or more."""

    statement: str
    """Natural-language statement of the conclusion."""

    evidence_strength: Confidence = 0.7
    """How strongly the parents support the conclusion. Not the same as
    ``confidence`` (which describes the inference's accuracy)."""

    derivation_method: str = "arw:v1"
    """Which subsystem produced this inference. Examples: ``arw:v1``,
    ``sleep:v1``, ``manual``."""


Memory = Annotated[
    Fact | Experience | Observation | Belief | Inference,
    Field(discriminator="network"),
]
"""Discriminated union of every memory network. Use this anywhere you would
otherwise want a generic ``Memory`` argument."""


# ---------------------------------------------------------------------------
# Bitemporal Transition Graph
# ---------------------------------------------------------------------------


class Transition(BaseModel):
    """A first-class change event linking two memories in the BTG.

    Where ``valid_at`` / ``valid_until`` describe *when* a memory was true,
    a transition records *what changed and why*. This is the artifact that
    distinguishes Reify's bitemporal model from Zep's.

    Examples
    --------
    User moves from New York to San Francisco. The system writes:

    1. A new :class:`Fact` ``(Arun, lives_in, San Francisco)`` valid from t.
    2. An update to the prior fact's ``valid_until = t``.
    3. A :class:`Transition` of kind ``LOCATION_CHANGE`` linking the two
       facts, with the supporting :class:`Experience` records as evidence.

    Queries can then ask "what changed between January and June?" and receive
    a structured answer rather than two contradictory facts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: MemoryId = Field(default_factory=new_id)
    kind: TransitionKind = TransitionKind.GENERIC

    from_memory_id: MemoryId | None = None
    """The prior memory whose validity is bounded by this transition.
    ``None`` for a pure addition (no prior state)."""

    to_memory_id: MemoryId | None = None
    """The new memory that takes effect after the transition. ``None`` for a
    pure retraction (the world simply stopped containing this fact)."""

    at: datetime
    """World time at which the transition is considered to have occurred."""

    recorded_at: datetime = Field(default_factory=_utcnow)
    """System time at which the transition was written."""

    evidence: tuple[MemoryId, ...] = ()
    """Memories (typically :class:`Experience` or :class:`Observation`) that
    support the transition."""

    confidence: Confidence = 1.0
    summary: str = ""
    identity_id: str | None = None


# ---------------------------------------------------------------------------
# Identity Bond Graph
# ---------------------------------------------------------------------------


class Identity(BaseModel):
    """A user, device, or anonymous session.

    Identities form a probabilistic graph. A retrieval scoped to ``user:arun``
    can transitively include memories owned by ``device:macbook_a1`` and
    ``anonymous:session_xyz`` if the bonds between them clear the configured
    confidence threshold.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    """Stable identifier. Convention: ``kind:slug`` (``user:arun``,
    ``device:macbook_a1``, ``anonymous:01HX...``)."""

    kind: Literal["user", "device", "anonymous"]
    label: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdentityBond(BaseModel):
    """A weighted, directed edge in the Identity Bond Graph.

    Bonds are formed by the :mod:`reify.identity` subsystem from three signal
    classes:

    * **Behavioral fingerprint** — typing cadence, working hours, vocabulary.
    * **Embedding similarity** — semantic similarity of recent memories.
    * **Explicit assertion** — the user told the system "this is me."

    The bond's ``confidence`` aggregates these signals. Retrieval applies a
    user-configurable threshold (default 0.8) when deciding which identities
    to include.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_identity: str
    to_identity: str
    confidence: Confidence
    rationale: str = ""
    formed_at: datetime = Field(default_factory=_utcnow)
    last_reinforced_at: datetime | None = None
