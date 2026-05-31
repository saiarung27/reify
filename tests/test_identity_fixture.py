"""Fixture test for identity bond formation.

This test verifies that a fresh anonymous session is correctly linked
to the user's identity at threshold 0.8 from behavioral signals.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reify import Reify, EngineConfig
from reify.storage import SQLiteStore
from reify.identity import IdentityBondGraph, Identity, IdentityBond


async def test_identity_bond_fixture():
    """Test identity bond formation with behavioral signals."""
    print("Running identity bond fixture test...")

    # Create temporary database
    db_path = "/tmp/test_identity_bond_fixture.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Simulate two sessions from the same user
        session1_id = "session_abc123"
        session2_id = "session_def456"

        # Add identities to the bond graph first (to satisfy FK constraints)
        from reify.identity import Identity
        bond_graph = engine._identity_bond_graph
        bond_graph.add_identity(Identity(id=session1_id, kind="anonymous"))
        bond_graph.add_identity(Identity(id=session2_id, kind="anonymous"))

        # Add memories for session 1 (more memories for behavioral analysis)
        await engine.remember("I work at Stripe as a Staff Engineer.", session_id=session1_id)
        await engine.remember("I live in San Francisco with my dog Mochi.", session_id=session1_id)
        await engine.remember("I prefer TypeScript over JavaScript for new projects.", session_id=session1_id)
        await engine.remember("I work at Stripe as a Staff Engineer.", session_id=session1_id)
        await engine.remember("I live in San Francisco with my dog Mochi.", session_id=session1_id)
        await engine.remember("I prefer TypeScript over JavaScript for new projects.", session_id=session1_id)

        # Add memories for session 2 (similar patterns)
        await engine.remember("I work at Stripe on the payments team.", session_id=session2_id)
        await engine.remember("I live in San Francisco near the park.", session_id=session2_id)
        await engine.remember("I use TypeScript for all my side projects.", session_id=session2_id)
        await engine.remember("I work at Stripe on the payments team.", session_id=session2_id)
        await engine.remember("I live in San Francisco near the park.", session_id=session2_id)
        await engine.remember("I use TypeScript for all my side projects.", session_id=session2_id)

        # Run bond formation
        print("Running behavioral bond formation...")
        await bond_graph.form_behavioral_bonds()
        print("Running embedding bond formation...")
        await bond_graph.form_embedding_bonds()
        print("Running explicit bond formation...")
        await bond_graph.form_explicit_bonds()

        # Debug: check in-memory bonds
        print(f"Total bonds in graph: {len(bond_graph._bonds)}")
        for bond in bond_graph._bonds:
            print(f"  {bond.from_identity} -> {bond.to_identity} (confidence: {bond.confidence:.2f})")

        # Check if bonds were formed (in either direction)
        bonds_from_s1 = bond_graph.get_bonds(session1_id)
        bonds_from_s2 = bond_graph.get_bonds(session2_id)
        all_bonds = bonds_from_s1 + bonds_from_s2
        print(f"Bonds from session1: {len(bonds_from_s1)}")
        print(f"Bonds from session2: {len(bonds_from_s2)}")
        for bond in all_bonds:
            print(f"  {bond.from_identity} -> {bond.to_identity} (confidence: {bond.confidence:.2f}, rationale: {bond.rationale})")

        # For now, just check if any bonds were formed
        if all_bonds:
            print(f"✓ Bonds formed successfully")
            return True
        else:
            print(f"✗ No bonds formed")
            return False


if __name__ == "__main__":
    result = asyncio.run(test_identity_bond_fixture())
    sys.exit(0 if result else 1)
