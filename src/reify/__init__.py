"""Reify: a model-portable cognitive substrate for AI.

This is the top-level package. The public API is intentionally small:

* :class:`reify.Reify` — the engine.
* :data:`reify.__version__` — the installed version string.

Subpackages
-----------
* :mod:`reify.core` — the data model and primitive types.
* :mod:`reify.storage` — storage interface + SQLite implementation.
* :mod:`reify.embed` — embedder interface + local default.
* :mod:`reify.llm` — LLM provider interface + adapters.
* :mod:`reify.ingest` — raw event capture (observer + reflector).
* :mod:`reify.extract` — schema-validated extraction.
* :mod:`reify.consolidate` — Active Reconsolidation, transitions, sleep phase.
* :mod:`reify.retrieve` — hybrid retrieval pipeline.
* :mod:`reify.synthesize` — token-budget-aware context synthesis.
* :mod:`reify.identity` — Identity Bond Graph.
* :mod:`reify.eval` — benchmark harness.
* :mod:`reify.server` — MCP, REST, OpenAI-compatible surfaces.

The :class:`Reify` engine is the only object most callers need; everything
else is composable substrate.
"""

from reify.engine import EngineConfig, RecallResponse, Reify, WriteReceipt
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
from reify.version import __version__

__all__ = [
    "Belief",
    "Confidence",
    "EngineConfig",
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
    "RecallResponse",
    "Reify",
    "RelationType",
    "Transition",
    "TransitionKind",
    "WriteReceipt",
    "__version__",
    "new_id",
]
