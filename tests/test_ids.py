"""Tests for :mod:`reify.core.ids`."""

from __future__ import annotations

import time

import pytest

from reify.core.ids import MemoryId, new_id, parse_id


def test_new_id_returns_unique_values() -> None:
    ids = {new_id() for _ in range(1024)}
    assert len(ids) == 1024


def test_new_id_is_lexicographically_time_ordered() -> None:
    first = new_id()
    # ULID timestamp resolution is 1 ms; sleep enough to guarantee ordering.
    time.sleep(0.002)
    second = new_id()
    assert first < second


def test_new_id_returns_memoryid_typed_str() -> None:
    value = new_id()
    # NewType is a runtime no-op but the value should be a 26-char Crockford
    # base32 string.
    assert isinstance(value, str)
    assert len(value) == 26


def test_parse_id_round_trips() -> None:
    original = new_id()
    assert parse_id(original) == original


def test_parse_id_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_id("not-a-ulid")


def test_memoryid_alias_is_string() -> None:
    # MemoryId is a NewType over str; cast through it for clarity.
    value: MemoryId = new_id()
    assert isinstance(value, str)
