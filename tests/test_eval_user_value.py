from __future__ import annotations

from reify.eval import default_user_value_cases, run_user_value_suite


async def test_run_user_value_suite_meets_default_thresholds() -> None:
    report = await run_user_value_suite()

    assert report.passed is True
    assert report.total_cases == len(default_user_value_cases())
    assert report.useful_recalls == report.total_cases
    assert report.recall_without_llm_rate == 1.0
    assert report.recall_accuracy_30d == 1.0
    assert report.time_to_useful_recall_p50_ms < 5.0
    assert report.time_to_useful_recall_p95_ms < 20.0
    assert report.cost_per_useful_recall_usd == 0.0
    assert report.llm_invocations == 0
    assert report.embed_invocations == 0
