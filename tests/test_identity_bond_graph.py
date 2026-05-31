"""Tests for the v0.5 Identity Bond Graph."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from reify import Reify
from reify.identity.bond_graph import Identity, IdentityBond, IdentityBondGraph
from reify.storage import SQLiteStore


def test_identity_creation() -> None:
    identity = Identity(id="user:123", kind="user")
    assert identity.id == "user:123"
    assert identity.kind == "user"
    assert identity.parent_id is None


def test_identity_with_parent() -> None:
    identity = Identity(id="device:456", kind="device", parent_id="user:123")
    assert identity.parent_id == "user:123"


def test_identity_bond_creation() -> None:
    bond = IdentityBond(
        from_identity="session:abc",
        to_identity="user:123",
        confidence=0.9,
        rationale="behavioral",
    )
    assert bond.from_identity == "session:abc"
    assert bond.to_identity == "user:123"
    assert bond.confidence == 0.9
    assert bond.rationale == "behavioral"


def test_identity_bond_graph_add_identity() -> None:
    graph = IdentityBondGraph()
    identity = Identity(id="user:123", kind="user")
    graph.add_identity(identity)
    assert "user:123" in graph._identities
    assert graph._identities["user:123"] == identity


def test_identity_bond_graph_add_bond() -> None:
    graph = IdentityBondGraph()
    bond = IdentityBond(
        from_identity="session:abc",
        to_identity="user:123",
        confidence=0.9,
        rationale="behavioral",
    )
    graph.add_bond(bond)
    assert bond in graph._bonds


def test_identity_bond_graph_resolve_identity() -> None:
    graph = IdentityBondGraph()
    user_identity = Identity(id="user:123", kind="user")
    graph.add_identity(user_identity)

    bond = IdentityBond(
        from_identity="session:abc",
        to_identity="user:123",
        confidence=0.9,
        rationale="behavioral",
    )
    graph.add_bond(bond)

    resolved = graph.resolve_identity("session:abc", threshold=0.8)
    assert resolved == user_identity


def test_identity_bond_graph_resolve_identity_below_threshold() -> None:
    graph = IdentityBondGraph()
    user_identity = Identity(id="user:123", kind="user")
    graph.add_identity(user_identity)

    bond = IdentityBond(
        from_identity="session:abc",
        to_identity="user:123",
        confidence=0.7,
        rationale="behavioral",
    )
    graph.add_bond(bond)

    resolved = graph.resolve_identity("session:abc", threshold=0.8)
    assert resolved is None


def test_identity_bond_graph_resolve_identity_no_bonds() -> None:
    graph = IdentityBondGraph()
    resolved = graph.resolve_identity("session:abc", threshold=0.8)
    assert resolved is None


def test_identity_bond_graph_resolve_identity_highest_confidence() -> None:
    graph = IdentityBondGraph()
    user1 = Identity(id="user:1", kind="user")
    user2 = Identity(id="user:2", kind="user")
    graph.add_identity(user1)
    graph.add_identity(user2)

    bond1 = IdentityBond(
        from_identity="session:abc",
        to_identity="user:1",
        confidence=0.7,
        rationale="behavioral",
    )
    bond2 = IdentityBond(
        from_identity="session:abc",
        to_identity="user:2",
        confidence=0.9,
        rationale="embedding",
    )
    graph.add_bond(bond1)
    graph.add_bond(bond2)

    resolved = graph.resolve_identity("session:abc", threshold=0.5)
    assert resolved == user2  # Higher confidence


async def test_identity_bond_graph_form_behavioral_bonds() -> None:
    graph = IdentityBondGraph()
    await graph.form_behavioral_bonds()
    # Placeholder: no-op


async def test_identity_bond_graph_form_embedding_bonds() -> None:
    graph = IdentityBondGraph()
    await graph.form_embedding_bonds()
    # Placeholder: no-op


async def test_identity_bond_graph_form_explicit_bonds() -> None:
    graph = IdentityBondGraph()
    await graph.form_explicit_bonds()
    # Placeholder: no-op


async def test_engine_has_identity_bond_graph() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    assert engine._identity_bond_graph is not None
    async with engine:
        assert engine._identity_bond_graph.engine == engine
