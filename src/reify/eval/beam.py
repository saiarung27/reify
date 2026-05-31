"""BEAM (Benchmark for Evaluation of Associative Memory) harness.

BEAM is a large-scale evaluation harness for testing memory systems at scale.
It supports configurable dataset sizes and measures recall accuracy, latency,
and throughput.

Per FIRST_PRINCIPLES.md, evaluation harnesses are independent of the core
engine and provide standardized metrics for comparing performance.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reify import Reify


def _percentile(sorted_values: list[float], p: float) -> float:
    """Calculate the p-th percentile using linear interpolation.

    Parameters
    ----------
    sorted_values:
        Sorted list of values.
    p:
        Percentile in [0, 1].

    Returns
    -------
    float
        The interpolated percentile value.
    """
    if not sorted_values:
        return 0.0
    if p <= 0:
        return sorted_values[0]
    if p >= 1:
        return sorted_values[-1]

    n = len(sorted_values)
    # Linear interpolation method (Type 7 from NumPy)
    index = (n - 1) * p
    lower = int(index)
    upper = lower + 1
    if upper >= n:
        return sorted_values[-1]
    weight = index - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


@dataclass(frozen=True, slots=True)
class BEAMCase:
    """A single test case in the BEAM benchmark."""

    query: str
    """The natural-language query to test."""

    expected_substrings: tuple[str, ...]
    """Substrings that must appear in the top result for correctness."""

    identity_id: str | None = None
    """Optional identity ID for the query."""


@dataclass(frozen=True, slots=True)
class BEAMCaseResult:
    """Result of running a single BEAM case."""

    case: BEAMCase
    correct: bool
    result_count: int
    top_hit: str | None
    latency_ms: float


@dataclass(frozen=True, slots=True)
class BEAMReport:
    """Aggregate report for a BEAM benchmark run."""

    case_results: tuple[BEAMCaseResult, ...]
    overall_accuracy: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    total_cases: int
    dataset_label: str
    scale: int
    passed_target: bool
    target: float


def default_beam_fixture_cases() -> tuple[BEAMCase, ...]:
    """Bundled fixture cases for quick BEAM testing."""
    return (
        BEAMCase(
            query="Where do I work?",
            expected_substrings=("Stripe",),
        ),
        BEAMCase(
            query="What pet do I have?",
            expected_substrings=("Mochi",),
        ),
        BEAMCase(
            query="Where do I live?",
            expected_substrings=("San Francisco",),
        ),
    )


async def run_beam_suite(
    *,
    engine: Reify,
    cases: tuple[BEAMCase, ...] | None = None,
    dataset_label: str = "bundled fixture",
    scale: int = 1,
    target: float = 0.8,
) -> BEAMReport:
    """Run the BEAM benchmark suite.

    Parameters
    ----------
    engine:
        The Reify engine to test.
    cases:
        Test cases to run. If None, uses bundled fixture cases.
    dataset_label:
        Label for the dataset (e.g., file path or "bundled fixture").
    scale:
        Scale multiplier for the dataset (for large-scale testing).
    target:
        Target accuracy threshold.

    Returns
    -------
    BEAMReport
        Aggregate results including accuracy and latency percentiles.
    """
    if cases is None:
        cases = default_beam_fixture_cases()

    # Scale the dataset by repeating cases if scale > 1.
    if scale > 1:
        scaled_cases = []
        for _ in range(scale):
            scaled_cases.extend(cases)
        cases = tuple(scaled_cases)

    case_results: list[BEAMCaseResult] = []
    latencies: list[float] = []

    for case in cases:
        start = time.perf_counter()
        response = await engine.recall(case.query, identity_id=case.identity_id)
        latency_ms = (time.perf_counter() - start) * 1000
        latencies.append(latency_ms)

        top_hit = response.results[0].memory.content if response.results else None
        correct = False
        if top_hit:
            correct = any(
                substr in top_hit for substr in case.expected_substrings
            )

        case_results.append(
            BEAMCaseResult(
                case=case,
                correct=correct,
                result_count=len(response.results),
                top_hit=top_hit,
                latency_ms=latency_ms,
            )
        )

    # Calculate accuracy.
    correct_count = sum(1 for r in case_results if r.correct)
    overall_accuracy = correct_count / len(case_results) if case_results else 0.0

    # Calculate latency percentiles with linear interpolation.
    sorted_latencies = sorted(latencies)
    p50 = _percentile(sorted_latencies, 0.50) if sorted_latencies else 0.0
    p95 = _percentile(sorted_latencies, 0.95) if sorted_latencies else 0.0
    p99 = _percentile(sorted_latencies, 0.99) if sorted_latencies else 0.0

    return BEAMReport(
        case_results=tuple(case_results),
        overall_accuracy=overall_accuracy,
        p50_latency_ms=p50,
        p95_latency_ms=p95,
        p99_latency_ms=p99,
        total_cases=len(case_results),
        dataset_label=dataset_label,
        scale=scale,
        passed_target=overall_accuracy >= target,
        target=target,
    )


__all__ = [
    "BEAMCase",
    "BEAMCaseResult",
    "BEAMReport",
    "default_beam_fixture_cases",
    "run_beam_suite",
]
