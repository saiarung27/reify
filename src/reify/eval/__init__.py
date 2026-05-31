"""Evaluation harness.

Currently implemented:

* ``user-value`` — fixture workload for the release-gate metrics documented
  in ``FIRST_PRINCIPLES.md`` and ``ROADMAP.md``.
* ``longmemeval-fixture`` — bundled fixture workload for local and CI smoke
  coverage.
* ``longmemeval`` — official dataset runner with separate Reify proxy and
  official-style QA reporting.
* ``multisession-synthesis`` — capability-track breakdown derived from
  official LongMemEval histories, scored by the current Reify retrieval proxy
  and grouped by product-aligned capabilities.

Planned suites for v0.3+:

* ``locomo`` — single-hop, multi-hop, open-domain, temporal.
* ``beam`` — 1M and 10M token scales.
* ``multisession-synthesis`` — a future fully native scorer beyond the current
  proxy-backed capability breakdown.
"""

from reify.eval.user_value import (
    UserValueCase,
    UserValueCaseResult,
    UserValueReport,
    default_user_value_cases,
    run_user_value_suite,
)
from reify.eval.longmemeval import (
    LongMemEvalBenchmarkReport,
    LongMemEvalCase,
    LongMemEvalCaseResult,
    LongMemEvalEvent,
    LongMemEvalReport,
    OfficialLongMemEvalInstance,
    OfficialLongMemEvalReport,
    OfficialLongMemEvalResult,
    adapt_official_longmemeval_instances,
    build_official_longmemeval_judge_prompt,
    default_longmemeval_fixture_cases,
    load_longmemeval_cases,
    load_official_longmemeval_instances,
    run_longmemeval_benchmark,
    run_longmemeval_suite,
    run_official_longmemeval_suite,
)
from reify.eval.multisession_synthesis import (
    MultiSessionSynthesisCase,
    MultiSessionSynthesisCaseResult,
    MultiSessionSynthesisReport,
    adapt_multisession_synthesis_cases,
    run_multisession_synthesis_suite,
)
from reify.eval.locomo import (
    LoCoMoCase,
    LoCoMoCaseResult,
    LoCoMoEvent,
    LoCoMoReport,
    default_locomo_fixture_cases,
    load_locomo_cases,
    run_locomo_suite,
)

__all__ = [
    "LongMemEvalBenchmarkReport",
    "MultiSessionSynthesisCase",
    "MultiSessionSynthesisCaseResult",
    "MultiSessionSynthesisReport",
    "LongMemEvalCase",
    "LongMemEvalCaseResult",
    "LongMemEvalEvent",
    "LongMemEvalReport",
    "OfficialLongMemEvalInstance",
    "OfficialLongMemEvalReport",
    "OfficialLongMemEvalResult",
    "adapt_official_longmemeval_instances",
    "adapt_multisession_synthesis_cases",
    "build_official_longmemeval_judge_prompt",
    "default_longmemeval_fixture_cases",
    "load_longmemeval_cases",
    "load_official_longmemeval_instances",
    "run_longmemeval_benchmark",
    "run_longmemeval_suite",
    "run_multisession_synthesis_suite",
    "run_official_longmemeval_suite",
    "UserValueCase",
    "UserValueCaseResult",
    "UserValueReport",
    "default_user_value_cases",
    "run_user_value_suite",
    "LoCoMoCase",
    "LoCoMoCaseResult",
    "LoCoMoEvent",
    "LoCoMoReport",
    "default_locomo_fixture_cases",
    "load_locomo_cases",
    "run_locomo_suite",
]
