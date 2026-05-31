"""Storage layer.

The interface is defined in :mod:`reify.storage.base`. The default
implementation is :mod:`reify.storage.sqlite_store`, a plain-SQLite backend
with the schema in ``_schema.sql``.

Storage is intentionally minimal: it is a *substrate* for the higher-level
Active Reconsolidation, Bitemporal Transition Graph, and Identity Bond Graph
subsystems. It does not know about LLMs, embedders, or retrieval pipelines.
"""

from reify.storage.base import RecallFilter, Storage, WriteBatch
from reify.storage.sqlite_store import SCHEMA_VERSION, SQLiteStore

__all__ = [
    "SCHEMA_VERSION",
    "RecallFilter",
    "SQLiteStore",
    "Storage",
    "WriteBatch",
]
