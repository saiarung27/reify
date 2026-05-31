"""Extraction subsystem.

Two extractor flavors share the :class:`Extractor` protocol:

* :class:`DeterministicExtractor` (this package) \u2014 pure-Python pattern
  matcher. No LLM, no embedder, no network. Foreground stage of the write
  path.
* ``LLMExtractor`` \u2014 schema-validated LLM extraction (v0.1, lands with the
  LLM provider implementations).

Both never return free-form text; schema violations surface as
:class:`reify.core.errors.SchemaError`.
"""

from reify.extract.base import (
    Actor,
    Extractor,
    ExtractionContext,
    ExtractionResult,
    merge_results,
)
from reify.extract.deterministic import DeterministicExtractor

__all__ = [
    "Actor",
    "DeterministicExtractor",
    "ExtractionContext",
    "ExtractionResult",
    "Extractor",
    "merge_results",
]
