"""Tests for the constrained SQL builder."""

from __future__ import annotations

import pytest

from reify.storage._query import (
    WhereBuilder,
    assert_safe_fragment,
    in_placeholders,
)


def test_in_placeholders() -> None:
    assert in_placeholders(0) == ""
    assert in_placeholders(1) == "?"
    assert in_placeholders(3) == "?,?,?"


def test_in_placeholders_rejects_negatives() -> None:
    with pytest.raises(ValueError):
        in_placeholders(-1)


def test_assert_safe_fragment_accepts_clean() -> None:
    assert_safe_fragment("created_at < ?")
    assert_safe_fragment("identity_id = ?")
    assert_safe_fragment("network IN (?,?,?)")
    assert_safe_fragment("valid_at <= ? AND (valid_until IS NULL OR valid_until > ?)")


@pytest.mark.parametrize(
    "bad",
    [
        "name = 'admin'",
        'name = "admin"',
        "name = `x`",
        "DROP TABLE users; --",
    ],
)
def test_assert_safe_fragment_rejects_quotes_and_semicolons(bad: str) -> None:
    with pytest.raises(ValueError):
        assert_safe_fragment(bad)


def test_where_builder_basic() -> None:
    b = WhereBuilder()
    b.add("identity_id = ?", "user-1")
    b.add("created_at < ?", "2025-01-01")
    sql, params = b.build()
    assert sql == "identity_id = ? AND created_at < ?"
    assert params == ["user-1", "2025-01-01"]


def test_where_builder_default_when_empty() -> None:
    b = WhereBuilder()
    sql, params = b.build()
    assert sql == "1=1"
    assert params == []


def test_where_builder_add_in_with_values() -> None:
    b = WhereBuilder()
    b.add_in("network", ["a", "b", "c"])
    sql, params = b.build()
    assert sql == "network IN (?,?,?)"
    assert params == ["a", "b", "c"]


def test_where_builder_add_in_empty_values_is_tautologically_false() -> None:
    b = WhereBuilder()
    b.add_in("network", [])
    sql, params = b.build()
    assert sql == "0=1"
    assert params == []


def test_where_builder_rejects_unsafe_column() -> None:
    b = WhereBuilder()
    with pytest.raises(ValueError):
        b.add_in("name) OR 1=1 --", ["x"])


def test_where_builder_param_count_mismatch() -> None:
    b = WhereBuilder()
    with pytest.raises(ValueError):
        b.add("identity_id = ?", "user-1", "extra")
