"""Typed exceptions for Reify.

Every public API raises one of the exceptions defined here. Internal failures
are wrapped before they reach the boundary so that callers never receive
backend-specific exceptions (sqlite3, httpx, etc.).
"""

from __future__ import annotations


class ReifyError(Exception):
    """Base class for all errors raised by Reify."""


class SchemaError(ReifyError):
    """Raised when a memory artifact fails Pydantic validation.

    Almost always indicates a bug in an extractor or in calling code that
    bypassed the schema.
    """


class StorageError(ReifyError):
    """Raised by the storage layer for operational failures.

    Examples: schema migration failure, unique constraint violation, IO error
    against the underlying database file.
    """


class UnknownMemoryError(ReifyError):
    """Raised when a referenced :data:`MemoryId` does not exist."""


class ProviderError(ReifyError):
    """Raised by LLM or embedding providers for upstream failures.

    Wraps backend-specific exceptions so callers can handle them uniformly.
    """


class ConfigError(ReifyError):
    """Raised when configuration is invalid or required values are missing."""
