"""Tests for the five-network memory model.

These are the schema's truth tests: if any of these regress, the data model
contract has been broken and downstream subsystems will misbehave.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from pydantic import TypeAdapter, ValidationError

from reify.core.ids import new_id
from reify.core.types import (
    Belief,
    Experience,
    Fact,
    Identity,
    IdentityBond,
    Inference,
    Memory,
    Network,
    Observation,
    Provenance,
    RelationType,
    Transition,
    TransitionKind,
)


# ---------------------------------------------------------------------------
# Per-network shape
# ---------------------------------------------------------------------------


def test_fact_minimal() -> None:
    f = Fact(content="Arun lives in SF.", subject="Arun", predicate="lives_in", obj="SF")
    assert f.network is Network.FACT
    assert f.confidence == 1.0
    assert f.valid_until is None
    assert isinstance(f.id, str) and len(f.id) == 26


def test_experience_minimal() -> None:
    e = Experience(
        content="User said hello.",
        event_type="user_message",
        actor="user",
        summary="Greeting",
    )
    assert e.network is Network.EXPERIENCE
    assert e.actor == "user"


def test_observation_minimal() -> None:
    o = Observation(
        content="Browser navigated to example.com.",
        capture_kind="browser",
        summary="Visited example.com",
    )
    assert o.network is Network.OBSERVATION


def test_belief_has_evidence_tuple() -> None:
    eid = new_id()
    b = Belief(content="User prefers Python.", statement="User prefers Python.", evidence=(eid,))
    assert b.network is Network.BELIEF
    assert b.evidence == (eid,)
    assert b.decay_half_life_days == 30.0


def test_inference_requires_relation_and_parents() -> None:
    parent_a = new_id()
    parent_b = new_id()
    inf = Inference(
        content="User moved from NY to SF in 2024.",
        relation=RelationType.RESOLVES,
        parent_memory_ids=(parent_a, parent_b),
        statement="User moved from NY to SF in 2024.",
        evidence_strength=0.85,
    )
    assert inf.network is Network.INFERENCE
    assert inf.relation is RelationType.RESOLVES
    assert inf.parent_memory_ids == (parent_a, parent_b)
    assert inf.derivation_method == "arw:v1"


# ---------------------------------------------------------------------------
# Frozen-ness: data model is immutable
# ---------------------------------------------------------------------------


def test_memories_are_frozen() -> None:
    f = Fact(content="x", subject="a", predicate="r", obj="b")
    with pytest.raises(ValidationError):
        f.confidence = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Confidence bounds
# ---------------------------------------------------------------------------


def test_confidence_clamped_to_unit_interval() -> None:
    with pytest.raises(ValidationError):
        Fact(
            content="x",
            subject="a",
            predicate="r",
            obj="b",
            confidence=1.5,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        Fact(
            content="x",
            subject="a",
            predicate="r",
            obj="b",
            confidence=-0.1,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Discriminated union round-trip
# ---------------------------------------------------------------------------


def test_memory_union_discriminates_by_network() -> None:
    adapter = TypeAdapter(Memory)
    raw_fact = {
        "network": "fact",
        "content": "x",
        "subject": "a",
        "predicate": "r",
        "obj": "b",
    }
    raw_inf = {
        "network": "inference",
        "content": "y",
        "relation": "extends",
        "parent_memory_ids": [new_id()],
        "statement": "y",
    }

    parsed_fact = adapter.validate_python(raw_fact)
    parsed_inf = adapter.validate_python(raw_inf)

    assert isinstance(parsed_fact, Fact)
    assert isinstance(parsed_inf, Inference)


# ---------------------------------------------------------------------------
# Bitemporal sanity
# ---------------------------------------------------------------------------


def test_valid_until_can_bound_validity() -> None:
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    t1 = datetime(2024, 6, 1, tzinfo=timezone.utc)
    f = Fact(
        content="x",
        subject="a",
        predicate="r",
        obj="b",
        valid_at=t0,
        valid_until=t1,
    )
    assert f.valid_until is not None
    assert f.valid_until - f.valid_at == timedelta(days=152)


# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


def test_transition_links_two_facts() -> None:
    a = new_id()
    b = new_id()
    t = datetime(2024, 6, 1, tzinfo=timezone.utc)
    tr = Transition(
        kind=TransitionKind.LOCATION_CHANGE,
        from_memory_id=a,
        to_memory_id=b,
        at=t,
        summary="User moved from NY to SF.",
    )
    assert tr.kind is TransitionKind.LOCATION_CHANGE
    assert tr.from_memory_id == a
    assert tr.to_memory_id == b
    assert tr.at == t


def test_transition_supports_pure_addition_and_retraction() -> None:
    t = datetime(2024, 6, 1, tzinfo=timezone.utc)
    addition = Transition(to_memory_id=new_id(), at=t)
    retraction = Transition(from_memory_id=new_id(), at=t)
    assert addition.from_memory_id is None
    assert retraction.to_memory_id is None


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------


def test_identity_kinds() -> None:
    user = Identity(id="user:arun", kind="user", label="Arun")
    device = Identity(id="device:macbook_a1", kind="device")
    anon = Identity(id="anonymous:01HX", kind="anonymous")
    assert user.kind == "user"
    assert device.kind == "device"
    assert anon.kind == "anonymous"


def test_identity_bond_confidence_is_clamped() -> None:
    with pytest.raises(ValidationError):
        IdentityBond(
            from_identity="anonymous:01HX",
            to_identity="user:arun",
            confidence=1.2,  # type: ignore[arg-type]
        )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_provenance_defaults() -> None:
    p = Provenance(source="claude_desktop")
    assert p.actor == "user"
    assert p.session_id is None
