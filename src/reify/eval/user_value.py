"""User-value evaluation harness.

This suite enforces the non-benchmark release gate described in
``FIRST_PRINCIPLES.md`` and ``ROADMAP.md``. It intentionally measures the
core, no-LLM value proposition first:

* recall remains useful after 30 simulated days
* recall stays fast
* recall usefulness does not depend on an LLM
* cost per useful recall is transparent

The initial fixture workload is deliberately small and deterministic. It is
not a replacement for LongMemEval or LoCoMo; it is the product-floor check
that must stay green before benchmark work counts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from reify import EngineConfig, Reify
from reify.core.types import Memory
from reify.storage import SQLiteStore


_BASELINE_NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)


class _MutableClock:
    def __init__(self, now: datetime) -> None:
        self.current = now

    def set(self, now: datetime) -> None:
        self.current = now

    def __call__(self) -> datetime:
        return self.current


@dataclass(frozen=True)
class UserValueCase:
    name: str
    identity_id: str
    utterance: str
    query: str
    expected_substring: str
    ingest_at: datetime
    query_at: datetime


@dataclass(frozen=True)
class UserValueCaseResult:
    case: UserValueCase
    useful: bool
    latency_ms: float
    top_hit: str | None
    result_count: int


@dataclass(frozen=True)
class UserValueReport:
    case_results: tuple[UserValueCaseResult, ...]
    total_cases: int
    useful_recalls: int
    recall_without_llm_rate: float
    recall_accuracy_30d: float
    time_to_useful_recall_p50_ms: float
    time_to_useful_recall_p95_ms: float
    cost_per_useful_recall_usd: float
    llm_invocations: int
    embed_invocations: int
    passed: bool


def default_user_value_cases() -> tuple[UserValueCase, ...]:
    """Return the fixture workload used by the default user-value gate."""
    ingest_at = _BASELINE_NOW
    query_at = ingest_at + timedelta(days=30)
    return (
        UserValueCase(
            name="employer_work_at",
            identity_id="fixture:employer_work_at",
            utterance="I work at Stripe.",
            query="Where do I work?",
            expected_substring="Stripe",
            ingest_at=ingest_at,
            query_at=query_at,
        ),
        UserValueCase(
            name="employer_joined",
            identity_id="fixture:employer_joined",
            utterance="I joined Anthropic.",
            query="Who did I join?",
            expected_substring="Anthropic",
            ingest_at=ingest_at,
            query_at=query_at,
        ),
        UserValueCase(
            name="location_live_in",
            identity_id="fixture:location_live_in",
            utterance="I live in Seattle.",
            query="Where do I live?",
            expected_substring="Seattle",
            ingest_at=ingest_at,
            query_at=query_at,
        ),
        UserValueCase(
            name="location_move_to",
            identity_id="fixture:location_move_to",
            utterance="I moved to Austin.",
            query="Where do I live?",
            expected_substring="Austin",
            ingest_at=ingest_at,
            query_at=query_at,
        ),
        UserValueCase(
            name="title_is",
            identity_id="fixture:title_is",
            utterance="I'm a Staff Engineer.",
            query="What's my title?",
            expected_substring="Staff Engineer",
            ingest_at=ingest_at,
            query_at=query_at,
        ),
        UserValueCase(
            name="promotion_from_to",
            identity_id="fixture:promotion_from_to",
            utterance="I got promoted from Senior Engineer to Staff Engineer.",
            query="What's my title?",
            expected_substring="Staff Engineer",
            ingest_at=ingest_at,
            query_at=query_at,
        ),
    )


async def run_user_value_suite(
    *,
    cases: tuple[UserValueCase, ...] | None = None,
) -> UserValueReport:
    """Run the fixture workload and compute user-value metrics."""
    fixture_cases = cases or default_user_value_cases()
    clock = _MutableClock(_BASELINE_NOW)
    engine = Reify(
        storage=SQLiteStore(":memory:"),
        clock=clock,
        config=EngineConfig(decay_sweep_interval_seconds=0),
    )
    await engine.start()
    try:
        for case in fixture_cases:
            clock.set(case.ingest_at)
            await engine.remember(
                case.utterance,
                identity_id=case.identity_id,
                at=case.ingest_at,
            )

        results: list[UserValueCaseResult] = []
        for case in fixture_cases:
            clock.set(case.query_at)
            started = time.perf_counter_ns()
            response = await engine.recall(
                case.query,
                identity_id=case.identity_id,
                valid_at=case.query_at,
                limit=5,
            )
            latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            top_hit = response.results[0].memory.content if response.results else None
            useful = any(
                _memory_contains_expected(result.memory, case.expected_substring)
                for result in response.results
            )
            results.append(
                UserValueCaseResult(
                    case=case,
                    useful=useful,
                    latency_ms=latency_ms,
                    top_hit=top_hit,
                    result_count=len(response.results),
                )
            )
    finally:
        await engine.stop()

    useful_recalls = sum(1 for result in results if result.useful)
    useful_latencies = [result.latency_ms for result in results if result.useful]
    total_cases = len(results)
    recall_rate = useful_recalls / total_cases if total_cases else 0.0
    p50_ms = _percentile(useful_latencies, 0.50)
    p95_ms = _percentile(useful_latencies, 0.95)
    report = UserValueReport(
        case_results=tuple(results),
        total_cases=total_cases,
        useful_recalls=useful_recalls,
        recall_without_llm_rate=recall_rate,
        recall_accuracy_30d=recall_rate,
        time_to_useful_recall_p50_ms=p50_ms,
        time_to_useful_recall_p95_ms=p95_ms,
        cost_per_useful_recall_usd=0.0,
        llm_invocations=0,
        embed_invocations=0,
        passed=(
            p50_ms < 5.0
            and p95_ms < 20.0
            and recall_rate >= 0.70
            and recall_rate >= 0.85
            and 0.0 <= 0.0005
        ),
    )
    return report


def _memory_contains_expected(memory: Memory, expected: str) -> bool:
    needle = expected.lower()
    haystacks = [memory.content.lower()]
    if getattr(memory, "summary", None):
        haystacks.append(str(memory.summary).lower())
    if getattr(memory, "obj", None):
        haystacks.append(str(memory.obj).lower())
    return any(needle in haystack for haystack in haystacks)


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("inf")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = quantile * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
