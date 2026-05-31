"""Core types, identifiers, errors, and settings.

This package has no runtime dependencies on storage, retrieval, or LLM
providers. It is the data-model spine; every other package imports from here.
"""

from reify.core.errors import (
    ReifyError,
    SchemaError,
    StorageError,
    UnknownMemoryError,
)
from reify.core.ids import MemoryId, new_id
from reify.core.types import (
    Belief,
    Confidence,
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

__all__ = [
    "Belief",
    "Confidence",
    "Experience",
    "Fact",
    "Identity",
    "IdentityBond",
    "Inference",
    "Memory",
    "MemoryId",
    "Network",
    "Observation",
    "Provenance",
    "ReifyError",
    "RelationType",
    "SchemaError",
    "StorageError",
    "Transition",
    "TransitionKind",
    "UnknownMemoryError",
    "new_id",
]
