"""Defensive coercion in ARW for smaller / less rule-disciplined models.

Smaller local models (quantized 7B/8B Ollama models such as ``llama3``)
sometimes return ``refines`` for what is clearly a supersession on a
mutually-exclusive predicate (``has_title``, ``lives_in``, ``works_at``).
The engine post-processes these into ``resolves`` so the Bitemporal
Transition Graph still fires, with a confidence discount to reflect that
we are second-guessing the model.

These are pure unit tests against
:meth:`reify.consolidate.reconsolidate.ARWWorker._coerce_exclusive_supersessions`
— no LLM, no live storage.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from reify.consolidate.reconsolidate import ARWWorker, Relation
from reify.core.types import Fact, RelationType
from reify.storage import SQLiteStore


def _fact(content: str, predicate: str, obj: str, *, valid_at: datetime) -> Fact:
    return Fact(
        content=content,
        subject="user",
        predicate=predicate,
        obj=obj,
        valid_at=valid_at,
        confidence=0.9,
        identity_id="user:test",
    )


@pytest.fixture
def worker() -> ARWWorker:
    """A worker built against a throwaway in-memory store. The store is not
    exercised by these tests; we only call the pure post-processing helper."""
    store = SQLiteStore(":memory:")
    return ARWWorker(store)


def test_refines_on_title_change_is_promoted_to_resolves(worker: ARWWorker) -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=60))
    candidate = _fact("Title: Senior Engineer", "has_title", "Senior Engineer", valid_at=now)
    raw = [Relation(type=RelationType.REFINES, confidence=1.0, explanation="model said refines")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.RESOLVES
    assert 0.5 <= out[0].confidence <= 1.0
    assert "coerced" in (out[0].explanation or "")


def test_refines_on_location_change_is_promoted_to_resolves(worker: ARWWorker) -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Location: Berkeley", "lives_in", "Berkeley", valid_at=now + timedelta(days=60))
    candidate = _fact("Location: Oakland", "lives_in", "Oakland", valid_at=now)
    raw = [Relation(type=RelationType.REFINES, confidence=0.9, explanation="")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.RESOLVES


def test_refines_on_employer_change_is_promoted_to_resolves(worker: ARWWorker) -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Employer: Stripe", "works_at", "Stripe", valid_at=now + timedelta(days=60))
    candidate = _fact("Employer: Linear", "works_at", "Linear", valid_at=now)
    raw = [Relation(type=RelationType.REFINES, confidence=0.85, explanation=None)]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.RESOLVES


def test_refines_on_non_exclusive_predicate_is_left_alone(worker: ARWWorker) -> None:
    """Preferences and similar non-exclusive predicates must not be coerced —
    you can prefer both vim AND dark mode at the same time."""
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Prefers dark mode", "prefers", "dark mode", valid_at=now + timedelta(days=1))
    candidate = _fact("Prefers vim", "prefers", "vim", valid_at=now)
    raw = [Relation(type=RelationType.REFINES, confidence=0.9, explanation="")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.REFINES


def test_refines_with_same_object_is_left_alone(worker: ARWWorker) -> None:
    """If both facts assert the same value, ``refines`` is fine — it is just a
    re-statement of the same current truth, not a change."""
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=1))
    candidate = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now)
    raw = [Relation(type=RelationType.REFINES, confidence=1.0, explanation="")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.REFINES


def test_resolves_is_left_alone(worker: ARWWorker) -> None:
    """The model already got it right — don't touch RESOLVES."""
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=60))
    candidate = _fact("Title: Senior Engineer", "has_title", "Senior Engineer", valid_at=now)
    raw = [Relation(type=RelationType.RESOLVES, confidence=0.95, explanation="model said resolves")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.RESOLVES
    assert out[0].confidence == 0.95
    assert "coerced" not in (out[0].explanation or "")


def test_contradicts_is_left_alone(worker: ARWWorker) -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=60))
    candidate = _fact("Title: Senior Engineer", "has_title", "Senior Engineer", valid_at=now)
    raw = [Relation(type=RelationType.CONTRADICTS, confidence=0.9, explanation="")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.CONTRADICTS


def test_confidence_is_discounted_on_coercion(worker: ARWWorker) -> None:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=60))
    candidate = _fact("Title: Senior Engineer", "has_title", "Senior Engineer", valid_at=now)
    raw = [Relation(type=RelationType.REFINES, confidence=1.0, explanation="")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.RESOLVES
    # Discounted by 0.8 from input 1.0
    assert out[0].confidence == pytest.approx(0.8, abs=0.01)


def test_case_insensitive_object_comparison(worker: ARWWorker) -> None:
    """``Staff Engineer`` and ``staff engineer`` describe the same role —
    do not coerce on capitalization differences alone."""
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=1))
    candidate = _fact("Title: staff engineer", "has_title", "staff engineer", valid_at=now)
    raw = [Relation(type=RelationType.REFINES, confidence=0.9, explanation="")]

    out = worker._coerce_exclusive_supersessions(memory, [candidate], raw)

    assert out[0].type is RelationType.REFINES


def test_coercion_preserves_other_relations_in_batch(worker: ARWWorker) -> None:
    """Only the offending entry should be touched in a multi-candidate call."""
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    memory = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=60))
    cand1 = _fact("Title: Senior Engineer", "has_title", "Senior Engineer", valid_at=now)
    cand2 = _fact("Title: Staff Engineer", "has_title", "Staff Engineer", valid_at=now + timedelta(days=30))
    raw = [
        Relation(type=RelationType.REFINES, confidence=0.9, explanation=""),
        Relation(type=RelationType.CONFIRMS, confidence=0.95, explanation=""),
    ]

    out = worker._coerce_exclusive_supersessions(memory, [cand1, cand2], raw)

    assert out[0].type is RelationType.RESOLVES
    assert out[1].type is RelationType.CONFIRMS
