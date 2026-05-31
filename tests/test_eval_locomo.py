"""Tests for the v0.3 LoCoMo proxy harness.

The fixture suite exercises every code path end-to-end so the CLI's
``reify eval locomo`` command stays smoke-tested. The bundled cases must
clear the proxy's default 80% accuracy target on a healthy
metadata-first stack.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from reify.eval import (
    LoCoMoCase,
    LoCoMoEvent,
    LoCoMoReport,
    default_locomo_fixture_cases,
    load_locomo_cases,
    run_locomo_suite,
)


def _utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


def test_default_fixture_has_taxonomy_coverage() -> None:
    cases = default_locomo_fixture_cases()
    categories = {c.category for c in cases}
    # The fixture should cover at least four LoCoMo categories so the
    # per-category accuracy breakdown is non-trivial.
    assert {"single_hop", "multi_hop", "temporal", "open_domain"} <= categories


async def test_run_locomo_suite_meets_target() -> None:
    report = await run_locomo_suite()
    assert isinstance(report, LoCoMoReport)
    assert len(report.case_results) == len(default_locomo_fixture_cases())
    assert report.overall_accuracy >= report.target
    assert report.passed_target is True


async def test_run_locomo_suite_returns_per_category_breakdown() -> None:
    report = await run_locomo_suite()
    assert report.per_category_accuracy
    # Every reported category must have a value in [0, 1].
    for category, accuracy in report.per_category_accuracy.items():
        assert 0.0 <= accuracy <= 1.0, f"category {category!r}: {accuracy}"


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_load_locomo_cases_from_json(tmp_path) -> None:
    payload = [
        {
            "name": "loaded_case",
            "category": "single_hop",
            "identity_id": "test:loaded",
            "events": [
                {"content": "I joined Vercel.", "at": "2025-01-01T00:00:00+00:00"},
            ],
            "query": "Where do I work?",
            "query_at": "2025-02-01T00:00:00+00:00",
            "expected_substrings": ["Vercel"],
        }
    ]
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    cases = load_locomo_cases(path)
    assert len(cases) == 1
    case = cases[0]
    assert case.name == "loaded_case"
    assert case.category == "single_hop"
    assert case.expected_substrings == ("Vercel",)
    assert len(case.events) == 1
    assert case.events[0].content == "I joined Vercel."


def test_load_locomo_cases_from_jsonl(tmp_path) -> None:
    lines = [
        json.dumps({
            "name": "a",
            "category": "single_hop",
            "identity_id": "test:a",
            "events": [{"content": "x", "at": "2025-01-01T00:00:00+00:00"}],
            "query": "?",
            "query_at": "2025-02-01T00:00:00+00:00",
            "expected_substrings": [],
        }),
        json.dumps({
            "name": "b",
            "category": "temporal",
            "identity_id": "test:b",
            "events": [{"content": "y", "at": "2025-01-02T00:00:00+00:00"}],
            "query": "?",
            "query_at": "2025-02-02T00:00:00+00:00",
            "expected_substrings": [],
        }),
    ]
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    cases = load_locomo_cases(path)
    assert tuple(c.name for c in cases) == ("a", "b")


def test_load_empty_file_returns_empty(tmp_path) -> None:
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    assert load_locomo_cases(path) == ()


# ---------------------------------------------------------------------------
# Custom case execution
# ---------------------------------------------------------------------------


async def test_run_locomo_suite_with_custom_cases() -> None:
    case = LoCoMoCase(
        name="custom",
        category="single_hop",
        identity_id="custom:test",
        events=(LoCoMoEvent(content="I bought a Tesla Model 3.", at=_utc(2025, 1, 1)),),
        query="What car do I drive?",
        query_at=_utc(2025, 2, 1),
        expected_substrings=("Tesla",),
    )
    report = await run_locomo_suite(cases=(case,), dataset_label="custom")
    assert report.dataset_label == "custom"
    assert len(report.case_results) == 1
    assert report.case_results[0].correct is True
