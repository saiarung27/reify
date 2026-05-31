"""Memory identifiers.

Reify uses ULIDs as the canonical identifier for every memory artifact.

Why ULID and not UUID:
  * Lexicographically sortable by creation time, which matches our bitemporal
    write path and makes range scans cheap.
  * 128-bit, collision-resistant.
  * Compact 26-character Crockford base32 representation.
  * Generation requires no coordination (good for offline-first clients).

Identifiers are opaque strings at every public boundary. The ULID type is a
private implementation detail.
"""

from __future__ import annotations

from typing import NewType

from ulid import ULID

MemoryId = NewType("MemoryId", str)
"""Opaque, lexicographically time-ordered identifier for a memory artifact."""


def new_id() -> MemoryId:
    """Mint a fresh memory identifier.

    The returned value is a 26-character Crockford base32 ULID string. It is
    monotonically increasing within a process at millisecond resolution.
    """
    return MemoryId(str(ULID()))


def parse_id(value: str) -> MemoryId:
    """Validate and normalize a string as a :data:`MemoryId`.

    Raises :class:`ValueError` if ``value`` is not a well-formed ULID. The
    returned identifier is canonicalized to upper-case Crockford base32.
    """
    parsed = ULID.from_str(value)
    return MemoryId(str(parsed))
