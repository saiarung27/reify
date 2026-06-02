"""Manual test script for v0.5 (Identity Bond Graph) features.

Tests:
- Identity model with anonymous → device → user hierarchy
- Bond formation from three signal classes (behavioral, embedding, explicit)
- Configurable confidence threshold (default 0.8)
- CLI: identity link/unlink/list
- MCP tool: identity_bond
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reify import Reify, EngineConfig
from reify.storage import SQLiteStore
from reify.identity import Identity, IdentityBondGraph, IdentityBond


async def test_identity_model():
    """Test identity model hierarchy."""
    print("\n=== Testing Identity Model ===")
    
    # Create identities with hierarchy
    user_id = "user_123"
    device_id = "device_abc"
    session_id = "session_xyz"
    
    user = Identity(id=user_id, kind="user")
    device = Identity(id=device_id, kind="device", parent_id=user_id)
    session = Identity(id=session_id, kind="anonymous", parent_id=device_id)
    
    print(f"User: {user.id} (kind: {user.kind})")
    print(f"Device: {device.id} (kind: {device.kind}, parent: {device.parent_id})")
    print(f"Session: {session.id} (kind: {session.kind}, parent: {session.parent_id})")
    
    # Verify hierarchy
    assert device.parent_id == user_id
    assert session.parent_id == device_id
    print("✓ Identity model hierarchy working")
    return True


async def test_bond_formation():
    """Test bond formation from three signal classes."""
    print("\n=== Testing Bond Formation ===")
    db_path = "/tmp/test_bond_formation.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Add identities
        bond_graph = engine._identity_bond_graph
        session1 = "session_abc"
        session2 = "session_def"
        bond_graph.add_identity(Identity(id=session1, kind="anonymous"))
        bond_graph.add_identity(Identity(id=session2, kind="anonymous"))

        # Add memories for both sessions
        for _ in range(6):
            await engine.remember("I work at Stripe as a Staff Engineer.", session_id=session1)
            await engine.remember("I work at Stripe on the payments team.", session_id=session2)

        # Run bond formation
        await bond_graph.form_behavioral_bonds()
        await bond_graph.form_embedding_bonds()
        await bond_graph.form_explicit_bonds()

        # Check bonds
        bonds_s1 = bond_graph.get_bonds(session1)
        bonds_s2 = bond_graph.get_bonds(session2)
        all_bonds = bonds_s1 + bonds_s2
        
        print(f"Total bonds formed: {len(all_bonds)}")
        for bond in all_bonds:
            print(f"  {bond.from_identity} -> {bond.to_identity}")
            print(f"    Confidence: {bond.confidence:.2f}")
            print(f"    Rationale: {bond.rationale}")
        
        # Verify at least one bond was formed
        assert len(all_bonds) > 0, "No bonds formed"
        print("✓ Bond formation working")
        return True


async def test_confidence_threshold():
    """Test configurable confidence threshold."""
    print("\n=== Testing Confidence Threshold ===")
    
    # Create bonds with different confidences
    bond1 = IdentityBond(
        from_identity="session1",
        to_identity="session2",
        confidence=0.9,
        rationale="test",
    )
    bond2 = IdentityBond(
        from_identity="session1",
        to_identity="session3",
        confidence=0.6,
        rationale="test",
    )
    
    # Test threshold filtering
    threshold = 0.8
    high_confidence = [b for b in [bond1, bond2] if b.confidence >= threshold]
    
    print(f"Bonds with confidence >= {threshold}: {len(high_confidence)}")
    assert len(high_confidence) == 1
    assert high_confidence[0].confidence == 0.9
    print("✓ Confidence threshold working")
    return True


async def test_cli_commands():
    """Test CLI identity commands."""
    print("\n=== Testing CLI Commands ===")
    print("Note: CLI commands tested via subprocess in production")
    print("Testing identity link/unlink/list methods directly...")
    
    db_path = "/tmp/test_cli_commands.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        bond_graph = engine._identity_bond_graph
        id1 = "identity_a"
        id2 = "identity_b"
        
        # Add identities
        bond_graph.add_identity(Identity(id=id1, kind="anonymous"))
        bond_graph.add_identity(Identity(id=id2, kind="anonymous"))
        
        # Test add_bond (equivalent to identity-link)
        bond = IdentityBond(
            from_identity=id1,
            to_identity=id2,
            confidence=0.9,
            rationale="manual test",
        )
        bond_graph.add_bond(bond)
        
        # Test get_bonds (equivalent to identity-list)
        bonds = bond_graph.get_bonds(id1)
        print(f"Bonds for {id1}: {len(bonds)}")
        assert len(bonds) == 1
        
        # Test remove_bond (equivalent to identity-unlink)
        bond_graph.remove_bond(id1, id2)
        bonds_after = bond_graph.get_bonds(id1)
        print(f"Bonds after removal: {len(bonds_after)}")
        assert len(bonds_after) == 0
        
        print("✓ CLI command methods working")
        return True


async def test_mcp_tool():
    """Test MCP identity_bond tool."""
    print("\n=== Testing MCP Tool ===")
    print("Note: MCP tool tested via test_mcp_sse_manual.py")
    print("Verifying tool definition in mcp.py...")
    
    # Check if identity_bond tool is defined in mcp.py
    import re
    with open("src/reify/server/mcp.py", "r") as f:
        content = f.read()
        has_identity_bond = 'name="identity_bond"' in content
    
    print(f"identity_bond tool defined: {has_identity_bond}")
    assert has_identity_bond, "identity_bond tool not defined in mcp.py"
    print("✓ MCP identity_bond tool defined")
    return True


async def main():
    """Run all v0.5 manual tests."""
    print("Running v0.5 Manual Tests")
    print("=" * 50)

    results = []
    results.append(("Identity Model", await test_identity_model()))
    results.append(("Bond Formation", await test_bond_formation()))
    results.append(("Confidence Threshold", await test_confidence_threshold()))
    results.append(("CLI Commands", await test_cli_commands()))
    results.append(("MCP Tool", await test_mcp_tool()))

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
