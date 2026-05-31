"""Tests for :mod:`reify.retrieve.entities` (v0.3 entity extractor).

Covers canonicalization, kind classification, surface extraction, and
per-memory delta construction (entities, mentions, edges).
"""

from __future__ import annotations

import pytest

from reify.core.types import Experience, Fact, Inference, RelationType
from reify.retrieve.entities import (
    Entity,
    canonicalize,
    classify_kind,
    extract_for_memory,
    extract_for_query,
    merge_deltas,
    slugify,
)


# ---------------------------------------------------------------------------
# slugify / canonicalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "label,expected",
    [
        ("Staff Engineer", "staff_engineer"),
        ("  San Francisco  ", "san_francisco"),
        ("Arun", "arun"),
        ("Bank of America", "bank_of_america"),
        ("café", "cafe"),
        ("", ""),
    ],
)
def test_slugify(label: str, expected: str) -> None:
    assert slugify(label) == expected


def test_canonicalize_drops_empty_and_articles() -> None:
    assert canonicalize("") is None
    assert canonicalize("the") is None


def test_canonicalize_yields_kind_and_label() -> None:
    ent = canonicalize("Staff Engineer")
    assert ent is not None
    assert ent.kind == "role"
    # ID is slug-only; kind is upgradeable metadata to avoid graph fragmentation.
    assert ent.id == "staff_engineer"
    assert ent.label == "Staff Engineer"


def test_classify_kind_uses_predicate_hint() -> None:
    assert classify_kind("Google", hint="works_at") == "org"
    assert classify_kind("San Francisco", hint="lives_in") == "location"
    assert classify_kind("Staff Engineer", hint="title") == "role"


def test_classify_kind_falls_back_on_lexicon() -> None:
    assert classify_kind("Staff Engineer") == "role"
    assert classify_kind("Acme Corp") == "org"
    # Single capitalized name without role/org markers -> person.
    assert classify_kind("Arun") == "person"


# ---------------------------------------------------------------------------
# extract_for_memory
# ---------------------------------------------------------------------------


def test_fact_yields_subject_object_and_predicate_edge() -> None:
    fact = Fact(
        content="Arun works at Acme Corp.",
        subject="Arun",
        predicate="works_at",
        obj="Acme Corp",
    )
    delta = extract_for_memory(fact)

    by_id = {e.id: e for e in delta.entities}
    assert "arun" in by_id
    assert "acme_corp" in by_id
    assert by_id["acme_corp"].kind == "org"

    role_by_entity = {m.entity_id: m.role for m in delta.mentions}
    assert role_by_entity["arun"] == "subject"
    assert role_by_entity["acme_corp"] == "object"

    pred_edges = [e for e in delta.edges if e.relation == "works_at"]
    assert len(pred_edges) == 1
    assert pred_edges[0].from_entity == "arun"
    assert pred_edges[0].to_entity == "acme_corp"


def test_experience_extracts_capitalized_phrases() -> None:
    exp = Experience(
        content="I had lunch with Arun in San Francisco at Blue Bottle.",
        event_type="user_message",
        actor="user",
        summary="lunch",
    )
    delta = extract_for_memory(exp)
    ids = {e.id for e in delta.entities}
    assert "arun" in ids
    assert "san_francisco" in ids


def test_extract_skips_sentence_initial_pronouns() -> None:
    exp = Experience(
        content="I went home.",
        event_type="user_message",
        actor="user",
        summary="went home",
    )
    delta = extract_for_memory(exp)
    # "I" should not become an entity, and "went home" is lowercase so
    # nothing else is captured. The delta should be empty.
    assert delta.entities == ()
    assert delta.mentions == ()


def test_co_occurrence_edges_are_symmetric() -> None:
    exp = Experience(
        content="Arun and Priya both joined Acme Corp last year.",
        event_type="user_message",
        actor="user",
        summary="both joined Acme",
    )
    delta = extract_for_memory(exp)
    # Multiple entities -> co-occurrence edges in both directions.
    co = [e for e in delta.edges if e.relation == "co_occurs"]
    assert co, "expected at least one co_occurs edge"
    # For every (a -> b) there should be a (b -> a).
    pairs = {(e.from_entity, e.to_entity) for e in co}
    for a, b in list(pairs):
        assert (b, a) in pairs


def test_inference_runs_through_capitalized_extraction() -> None:
    inf = Inference(
        content="Arun has been promoted twice at Acme Corp.",
        statement="promoted twice at Acme Corp",
        relation=RelationType.EXTENDS,
        parent_memory_ids=("01H",),
    )
    delta = extract_for_memory(inf)
    ids = {e.id for e in delta.entities}
    assert "arun" in ids


# ---------------------------------------------------------------------------
# extract_for_query
# ---------------------------------------------------------------------------


def test_extract_for_query_finds_seeds() -> None:
    seeds = extract_for_query("What does Arun think about Acme Corp lately?")
    ids = {e.id for e in seeds}
    assert "arun" in ids
    assert "acme_corp" in ids


def test_extract_for_query_deduplicates() -> None:
    seeds = extract_for_query("Arun and Arun and Arun.")
    assert len(seeds) == 1
    assert seeds[0].id == "arun"


# ---------------------------------------------------------------------------
# merge_deltas
# ---------------------------------------------------------------------------


def test_merge_deltas_dedupes_entities_and_concatenates_lists() -> None:
    f1 = Fact(content="Arun is at Acme.", subject="Arun", predicate="works_at", obj="Acme")
    f2 = Fact(content="Arun likes Acme.", subject="Arun", predicate="likes", obj="Acme")
    merged = merge_deltas([extract_for_memory(f1), extract_for_memory(f2)])
    by_id = {e.id: e for e in merged.entities}
    assert set(by_id) == {"arun", "acme"}
    # The stronger "org" classification (from the works_at hint) should win
    # over the weaker "person" guess (from the likes-hint pass) because
    # ``works_at`` is a kind-upgradable hint.
    assert by_id["acme"].kind == "org"
    rels = sorted(e.relation for e in merged.edges if e.relation in {"works_at", "likes"})
    assert rels == ["likes", "works_at"]
