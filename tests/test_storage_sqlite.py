"""Tests for :mod:`reify.storage.sqlite_store`.

The storage layer is the foundation every other v0.1 subsystem stacks on. If
any of these regress, the wedge demo cannot ship.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from reify.core.errors import StorageError, UnknownMemoryError
from reify.core.ids import new_id
from reify.core.types import (
    Belief,
    Experience,
    Fact,
    Identity,
    IdentityBond,
    Inference,
    Network,
    Observation,
    Provenance,
    RelationType,
    Transition,
    TransitionKind,
)
from reify.storage import RecallFilter, SCHEMA_VERSION, SQLiteStore, WriteBatch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def store() -> SQLiteStore:
    """An initialized, empty in-memory store. One per test."""
    s = SQLiteStore(":memory:")
    await s.init()
    try:
        yield s
    finally:
        await s.close()


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def _make_fact(**overrides: object) -> Fact:
    defaults: dict[str, object] = {
        "content": "Arun lives in San Francisco.",
        "subject": "Arun",
        "predicate": "lives_in",
        "obj": "San Francisco",
    }
    defaults.update(overrides)
    return Fact(**defaults)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_init_is_idempotent() -> None:
    s = SQLiteStore(":memory:")
    await s.init()
    await s.init()  # must not raise
    await s.close()


async def test_close_is_idempotent() -> None:
    s = SQLiteStore(":memory:")
    await s.init()
    await s.close()
    await s.close()  # must not raise


async def test_operations_before_init_raise() -> None:
    s = SQLiteStore(":memory:")
    with pytest.raises(StorageError):
        await s.get("01HX")  # type: ignore[arg-type]


async def test_schema_version_is_recorded(tmp_path: Path) -> None:
    db = tmp_path / "store.db"
    s = SQLiteStore(db)
    await s.init()
    # Inspect via the raw connection so we test the on-disk truth.
    raw = s._require_conn()  # type: ignore[attr-defined]
    row = raw.execute("SELECT version FROM schema_version").fetchone()
    assert row["version"] == SCHEMA_VERSION
    await s.close()


# ---------------------------------------------------------------------------
# Per-network round trips
# ---------------------------------------------------------------------------


async def test_fact_round_trip(store: SQLiteStore) -> None:
    fact = _make_fact()
    await store.commit_batch(WriteBatch(memories=[fact]))
    got = await store.get(fact.id)
    assert got is not None
    assert isinstance(got, Fact)
    assert got.id == fact.id
    assert got.subject == "Arun"
    assert got.predicate == "lives_in"
    assert got.obj == "San Francisco"


async def test_experience_round_trip(store: SQLiteStore) -> None:
    exp = Experience(
        content="User said: I just got promoted.",
        event_type="user_message",
        actor="user",
        summary="Promotion announcement",
    )
    await store.commit_batch(WriteBatch(memories=[exp]))
    got = await store.get(exp.id)
    assert isinstance(got, Experience)
    assert got.event_type == "user_message"
    assert got.summary == "Promotion announcement"


async def test_observation_round_trip(store: SQLiteStore) -> None:
    obs = Observation(
        content="Browser opened example.com",
        capture_kind="browser",
        summary="Visited example.com",
    )
    await store.commit_batch(WriteBatch(memories=[obs]))
    got = await store.get(obs.id)
    assert isinstance(got, Observation)
    assert got.capture_kind == "browser"


async def test_belief_round_trip(store: SQLiteStore) -> None:
    eid = new_id()
    b = Belief(
        content="User prefers Python.",
        statement="User prefers Python.",
        evidence=(eid,),
    )
    await store.commit_batch(WriteBatch(memories=[b]))
    got = await store.get(b.id)
    assert isinstance(got, Belief)
    assert got.evidence == (eid,)


async def test_inference_round_trip(store: SQLiteStore) -> None:
    parents = (new_id(), new_id())
    inf = Inference(
        content="Promoted twice in 18 months.",
        relation=RelationType.EXTENDS,
        parent_memory_ids=parents,
        statement="Promoted twice in 18 months.",
        evidence_strength=0.82,
    )
    await store.commit_batch(WriteBatch(memories=[inf]))
    got = await store.get(inf.id)
    assert isinstance(got, Inference)
    assert got.relation is RelationType.EXTENDS
    assert got.parent_memory_ids == parents


async def test_provenance_round_trips(store: SQLiteStore) -> None:
    p1 = Provenance(source="claude_desktop", actor="user", session_id="s1")
    p2 = Provenance(source="extractor:v1", actor="extractor")
    fact = _make_fact(provenance=(p1, p2))
    await store.commit_batch(WriteBatch(memories=[fact]))
    got = await store.get(fact.id)
    assert got is not None
    assert len(got.provenance) == 2
    assert got.provenance[0].source == "claude_desktop"
    assert got.provenance[1].actor == "extractor"


# ---------------------------------------------------------------------------
# Recall filtering
# ---------------------------------------------------------------------------


async def test_recall_filters_by_network(store: SQLiteStore) -> None:
    await store.commit_batch(
        WriteBatch(
            memories=[
                _make_fact(content="f1", subject="A", predicate="r", obj="B"),
                Experience(content="e1", event_type="user_message", summary="x"),
                Experience(content="e2", event_type="user_message", summary="y"),
            ]
        )
    )
    facts = await store.recall(RecallFilter(networks=[Network.FACT]))
    experiences = await store.recall(RecallFilter(networks=[Network.EXPERIENCE]))
    assert len(facts) == 1
    assert len(experiences) == 2


async def test_recall_filters_by_identity(store: SQLiteStore) -> None:
    await store.commit_batch(
        WriteBatch(
            memories=[
                _make_fact(identity_id="user:arun"),
                _make_fact(identity_id="user:other"),
                _make_fact(identity_id=None),
            ]
        )
    )
    mine = await store.recall(RecallFilter(identity_ids=["user:arun"]))
    assert len(mine) == 1
    assert mine[0].identity_id == "user:arun"


async def test_recall_filters_by_tag(store: SQLiteStore) -> None:
    await store.commit_batch(
        WriteBatch(
            memories=[
                _make_fact(tags=("promotion",)),
                _make_fact(tags=("location",)),
                _make_fact(tags=("promotion", "title")),
            ]
        )
    )
    promos = await store.recall(RecallFilter(tags_any=["promotion"]))
    assert len(promos) == 2


async def test_recall_bitemporal_as_of(store: SQLiteStore) -> None:
    old = _make_fact(
        content="Senior Engineer",
        obj="Senior",
        valid_at=_utc(2024, 1, 1),
        valid_until=_utc(2024, 6, 1),
    )
    new = _make_fact(
        content="Staff Engineer",
        obj="Staff",
        valid_at=_utc(2024, 6, 1),
    )
    await store.commit_batch(WriteBatch(memories=[old, new]))

    # Before the change: only the old fact is valid.
    asof_march = await store.recall(RecallFilter(valid_at=_utc(2024, 3, 1)))
    assert len(asof_march) == 1
    assert asof_march[0].id == old.id

    # After the change: only the new fact is valid.
    asof_august = await store.recall(RecallFilter(valid_at=_utc(2024, 8, 1)))
    assert len(asof_august) == 1
    assert asof_august[0].id == new.id


async def test_recall_excludes_soft_deleted_by_default(store: SQLiteStore) -> None:
    fact = _make_fact()
    await store.commit_batch(WriteBatch(memories=[fact]))
    await store.soft_delete(fact.id)
    visible = await store.recall(RecallFilter())
    assert visible == []
    visible_with_deleted = await store.recall(RecallFilter(include_deleted=True))
    assert len(visible_with_deleted) == 1


async def test_recall_orders_newest_first(store: SQLiteStore) -> None:
    a = _make_fact(content="a")
    b = _make_fact(content="b")
    c = _make_fact(content="c")
    # commit in two batches so the IDs (ULIDs) are time-ordered.
    await store.commit_batch(WriteBatch(memories=[a]))
    await store.commit_batch(WriteBatch(memories=[b]))
    await store.commit_batch(WriteBatch(memories=[c]))
    out = await store.recall(RecallFilter())
    assert [m.content for m in out] == ["c", "b", "a"]


# ---------------------------------------------------------------------------
# Soft delete
# ---------------------------------------------------------------------------


async def test_soft_delete_unknown_id_raises(store: SQLiteStore) -> None:
    with pytest.raises(UnknownMemoryError):
        await store.soft_delete(new_id())


async def test_soft_delete_is_idempotent(store: SQLiteStore) -> None:
    fact = _make_fact()
    await store.commit_batch(WriteBatch(memories=[fact]))
    await store.soft_delete(fact.id)
    await store.soft_delete(fact.id)  # must not raise


async def test_expired_memory_ids_returns_only_live_expired_claims(store: SQLiteStore) -> None:
    now = _utc(2024, 2, 15)
    expired = _make_fact(
        content="expired",
        metadata={
            "retention": {
                "ttl_days": 30.0,
                "expires_at": _utc(2024, 2, 1).isoformat(),
                "reinforced_count": 0,
            }
        },
    )
    active = _make_fact(
        content="active",
        metadata={
            "retention": {
                "ttl_days": 30.0,
                "expires_at": _utc(2024, 3, 1).isoformat(),
                "reinforced_count": 0,
            }
        },
    )
    deleted = _make_fact(
        content="deleted",
        metadata={
            "retention": {
                "ttl_days": 30.0,
                "expires_at": _utc(2024, 1, 15).isoformat(),
                "reinforced_count": 0,
            }
        },
    )
    await store.commit_batch(WriteBatch(memories=[expired, active, deleted]))
    await store.soft_delete(deleted.id, at=now)

    expired_ids = await store.expired_memory_ids(now=now)
    assert expired_ids == [expired.id]


# ---------------------------------------------------------------------------
# Atomicity
# ---------------------------------------------------------------------------


async def test_commit_batch_is_atomic(store: SQLiteStore) -> None:
    fact = _make_fact()
    # Reference a non-existent memory in a transition; the FK is permissive
    # (ON DELETE SET NULL), so the transition will land. Instead, force a
    # failure by inserting two memories with the same id.
    twin = _make_fact(content="twin")
    object.__setattr__(twin, "id", fact.id)  # bypass frozen for a forced collision

    # The store uses INSERT OR REPLACE, so a duplicate id is not an error.
    # Instead, force a constraint violation via a bond pointing at unknown
    # identities (FK ON DELETE CASCADE requires both endpoints to exist).
    bond = IdentityBond(
        from_identity="ghost:a", to_identity="ghost:b", confidence=0.5
    )
    with pytest.raises(StorageError):
        await store.commit_batch(WriteBatch(memories=[fact], bonds=[bond]))
    # The fact must not have landed.
    assert await store.get(fact.id) is None


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------


async def test_transition_round_trip(store: SQLiteStore) -> None:
    old = _make_fact(content="Senior", obj="Senior")
    new = _make_fact(content="Staff", obj="Staff")
    t = Transition(
        kind=TransitionKind.ROLE_CHANGE,
        from_memory_id=old.id,
        to_memory_id=new.id,
        at=_utc(2024, 6, 1),
        evidence=(new_id(),),
        summary="Promotion from Senior to Staff.",
    )
    await store.commit_batch(WriteBatch(memories=[old, new], transitions=[t]))

    got = await store.transitions_for(memory_ids=[old.id])
    assert len(got) == 1
    assert got[0].kind is TransitionKind.ROLE_CHANGE
    assert got[0].from_memory_id == old.id
    assert got[0].to_memory_id == new.id
    assert len(got[0].evidence) == 1


async def test_transitions_for_between_window(store: SQLiteStore) -> None:
    a = _make_fact()
    b = _make_fact()
    await store.commit_batch(WriteBatch(memories=[a, b]))
    early = Transition(
        kind=TransitionKind.GENERIC,
        from_memory_id=a.id,
        to_memory_id=b.id,
        at=_utc(2024, 1, 15),
    )
    late = Transition(
        kind=TransitionKind.GENERIC,
        from_memory_id=a.id,
        to_memory_id=b.id,
        at=_utc(2024, 8, 15),
    )
    await store.commit_batch(WriteBatch(transitions=[early, late]))

    window = await store.transitions_for(
        between=(_utc(2024, 6, 1), _utc(2024, 12, 1))
    )
    assert len(window) == 1
    assert window[0].id == late.id


# ---------------------------------------------------------------------------
# Embeddings + vector search
# ---------------------------------------------------------------------------


async def test_attach_embedding_unknown_memory_raises(store: SQLiteStore) -> None:
    with pytest.raises(UnknownMemoryError):
        await store.attach_embedding(new_id(), [0.1, 0.2, 0.3], model="test")


async def test_vector_search_returns_closest_first(store: SQLiteStore) -> None:
    a = _make_fact(content="apple")
    b = _make_fact(content="banana")
    c = _make_fact(content="cherry")
    await store.commit_batch(WriteBatch(memories=[a, b, c]))
    # Hand-crafted embeddings so the math is obvious.
    await store.attach_embedding(a.id, [1.0, 0.0, 0.0], model="test")
    await store.attach_embedding(b.id, [0.0, 1.0, 0.0], model="test")
    await store.attach_embedding(c.id, [0.95, 0.05, 0.0], model="test")

    results = await store.vector_search([1.0, 0.0, 0.0], limit=3)
    ids = [m.id for m, _ in results]
    # a (identical) < c (close) < b (orthogonal)
    assert ids == [a.id, c.id, b.id]
    assert results[0][1] == pytest.approx(0.0, abs=1e-6)


async def test_vector_search_with_no_embeddings_returns_empty(store: SQLiteStore) -> None:
    await store.commit_batch(WriteBatch(memories=[_make_fact()]))
    out = await store.vector_search([0.1, 0.2, 0.3])
    assert out == []


# ---------------------------------------------------------------------------
# Identity bonds
# ---------------------------------------------------------------------------


async def test_identity_bond_round_trip(store: SQLiteStore) -> None:
    # Identities must exist before bonds (FK constraint).
    raw = store._require_conn()  # type: ignore[attr-defined]
    for ident in (
        Identity(id="anonymous:s1", kind="anonymous"),
        Identity(id="user:arun", kind="user"),
    ):
        raw.execute(
            "INSERT INTO identities (id, kind, label, created_at, metadata_json) VALUES (?, ?, ?, ?, '{}')",
            (ident.id, ident.kind, ident.label, ident.created_at.isoformat()),
        )

    bond = IdentityBond(
        from_identity="anonymous:s1",
        to_identity="user:arun",
        confidence=0.83,
        rationale="behavioral fingerprint match",
    )
    await store.commit_batch(WriteBatch(bonds=[bond]))

    bonds = await store.bonds_for("anonymous:s1")
    assert len(bonds) == 1
    assert bonds[0].confidence == pytest.approx(0.83)

    weak = await store.bonds_for("anonymous:s1", min_confidence=0.9)
    assert weak == []


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


async def test_data_persists_across_close_and_reopen(tmp_path: Path) -> None:
    db = tmp_path / "store.db"
    s1 = SQLiteStore(db)
    await s1.init()
    fact = _make_fact(content="Persists across sessions.")
    await s1.commit_batch(WriteBatch(memories=[fact]))
    await s1.close()

    s2 = SQLiteStore(db)
    await s2.init()
    got = await s2.get(fact.id)
    assert got is not None
    assert got.content == "Persists across sessions."
    await s2.close()
