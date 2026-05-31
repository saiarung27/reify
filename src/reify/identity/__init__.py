"""Identity Bond Graph subsystem (v0.5).

Probabilistic linking of anonymous sessions, devices, and users. Bonds are
formed from three signal classes: behavioral fingerprints, embedding
similarity of recent memories, and explicit user assertions.

Retrieval applies a configurable confidence threshold (default 0.8) when
deciding which identities' memories to include in a query. The threshold is
the user's privacy dial.
"""

from reify.identity.bond_graph import Identity, IdentityBond, IdentityBondGraph

__all__ = ["Identity", "IdentityBond", "IdentityBondGraph"]
