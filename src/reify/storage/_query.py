"""Constrained SQL fragment helpers for the SQLite store.

The store assembles WHERE clauses dynamically. Values are always bound via
``?`` placeholders, but the *fragments* themselves are concatenated with
``f"..."`` formatting. This module makes that contract explicit and
auditable so a future contributor cannot accidentally embed an unbound
string into the SQL.

Rules enforced by :class:`WhereBuilder`:

1. Every clause is a Python *literal* string (the call site cannot pass
   user-controlled text without going through the explicit ``params``
   list).
2. Clauses may only use ``?`` for value binding; quote/backtick/semicolon
   characters are rejected.
3. The number of parameters must match the number of ``?`` placeholders
   in the assembled SQL.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

# Allowed characters in a clause fragment: letters, digits, whitespace,
# and a small set of SQL punctuation. We deliberately reject quote
# characters because every value should be supplied via params.
_ALLOWED_CLAUSE_RE = re.compile(
    r"^[A-Za-z0-9_\s\.,\(\)\?<>=!\-\+\*/]+$"
)


def assert_safe_fragment(fragment: str) -> None:
    """Raise ``ValueError`` if a SQL fragment contains banned characters.

    Banned characters include single/double quotes (any literal value should
    be a parameter), backticks, and semicolons (statement chaining).
    """
    if "'" in fragment or '"' in fragment or "`" in fragment or ";" in fragment:
        raise ValueError(
            f"unsafe SQL fragment: quotes, backticks, or semicolons are not allowed (use ? placeholders for values): {fragment!r}"
        )
    if "--" in fragment or "/*" in fragment or "*/" in fragment:
        raise ValueError(
            f"unsafe SQL fragment: SQL comment markers are not allowed: {fragment!r}"
        )
    flat = " ".join(fragment.split())
    if not _ALLOWED_CLAUSE_RE.match(flat):
        raise ValueError(f"unsafe SQL fragment: contains disallowed characters: {fragment!r}")


def in_placeholders(n: int) -> str:
    """Return ``"?,?,..."`` with ``n`` placeholders. ``n`` must be >= 0."""
    if not isinstance(n, int) or n < 0:
        raise ValueError(f"in_placeholders requires non-negative int, got {n!r}")
    return ",".join("?" for _ in range(n))


@dataclass
class WhereBuilder:
    """Accumulates AND-joined WHERE clause fragments and their parameters.

    Every fragment is validated by :func:`assert_safe_fragment`. The final
    SQL is assembled via :meth:`build`, which also asserts that the number
    of bound parameters matches the number of placeholders.
    """

    clauses: list[str] = field(default_factory=list)
    params: list[Any] = field(default_factory=list)

    def add(self, fragment: str, *params: Any) -> None:
        assert_safe_fragment(fragment)
        expected = fragment.count("?")
        if expected != len(params):
            raise ValueError(
                f"placeholder/param mismatch in {fragment!r}: "
                f"expected {expected}, got {len(params)}"
            )
        self.clauses.append(fragment)
        self.params.extend(params)

    def add_in(self, column: str, values: Sequence[Any]) -> None:
        """Add ``column IN (?,?,...)`` with one placeholder per value.

        An empty ``values`` sequence emits a tautologically-false clause
        (``0=1``) so the result set is empty as the caller intends; SQLite
        rejects ``IN ()`` syntactically.
        """
        assert_safe_fragment(column)
        if not values:
            self.clauses.append("0=1")
            return
        placeholders = in_placeholders(len(values))
        self.clauses.append(f"{column} IN ({placeholders})")
        self.params.extend(values)

    def build(self, *, default: str = "1=1") -> tuple[str, list[Any]]:
        """Assemble the final ``(where_sql, params)`` tuple."""
        if not self.clauses:
            return default, list(self.params)
        sql = " AND ".join(self.clauses)
        if sql.count("?") != len(self.params):
            raise AssertionError(
                f"WhereBuilder placeholder/param mismatch: "
                f"sql={sql!r} params={self.params!r}"
            )
        return sql, list(self.params)


__all__ = ["WhereBuilder", "assert_safe_fragment", "in_placeholders"]
