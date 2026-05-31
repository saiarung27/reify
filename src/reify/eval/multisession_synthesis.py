"""Capability-track evaluation derived from LongMemEval inputs.

This suite keeps the released LongMemEval conversation histories as the input
corpus and groups the existing Reify retrieval proxy into product-aligned
capability tracks.

This is intentionally not a new judge or a replacement scorer. It is a
breakdown view over the current deterministic retrieval proxy so we can see
which product promises are actually carrying or failing on a real corpus.

The track is intentionally deterministic and inspectable:

* direct fact recall
* assistant continuity
* cross-session continuity
* temporal recall
* update truthfulness
* preference grounding
* abstention on unsupported questions
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from reify.eval.longmemeval import (
    LongMemEvalCase,
    LongMemEvalCaseResult,
    LongMemEvalReport,
    OfficialLongMemEvalInstance,
    adapt_official_longmemeval_instances,
    load_official_longmemeval_instances,
    run_longmemeval_suite,
)


@dataclass(frozen=True)
class MultiSessionSynthesisCase:
    name: str
    track: str
    base_case: LongMemEvalCase


@dataclass(frozen=True)
class MultiSessionSynthesisCaseResult:
    case: MultiSessionSynthesisCase
    correct: bool
    result_count: int
    top_hit: str | None


@dataclass(frozen=True)
class MultiSessionSynthesisReport:
    case_results: tuple[MultiSessionSynthesisCaseResult, ...]
    overall_accuracy: float
    per_track_accuracy: dict[str, float]
    dataset_label: str
    passed_target: bool
    track_target: float


def adapt_multisession_synthesis_cases(
    instances: tuple[OfficialLongMemEvalInstance, ...],
) -> tuple[MultiSessionSynthesisCase, ...]:
    """Map official LongMemEval instances into Reify-native tracks."""
    base_cases = adapt_official_longmemeval_instances(instances)
    adapted: list[MultiSessionSynthesisCase] = []
    for instance, base_case in zip(instances, base_cases, strict=True):
        adapted.append(
            MultiSessionSynthesisCase(
                name=instance.question_id,
                track=_reify_native_track(instance),
                base_case=base_case,
            )
        )
    return tuple(adapted)


async def run_multisession_synthesis_suite(
    *,
    dataset_path: Path,
) -> MultiSessionSynthesisReport:
    """Run the capability-track breakdown over an official LongMemEval JSON file."""
    instances = load_official_longmemeval_instances(dataset_path)
    adapted = adapt_multisession_synthesis_cases(instances)
    base_report = await run_longmemeval_suite(
        cases=tuple(case.base_case for case in adapted),
        dataset_label=str(dataset_path),
    )
    by_name = {result.case.name: result for result in base_report.case_results}
    case_results = tuple(
        MultiSessionSynthesisCaseResult(
            case=case,
            correct=by_name[case.name].correct,
            result_count=by_name[case.name].result_count,
            top_hit=by_name[case.name].top_hit,
        )
        for case in adapted
    )
    overall_accuracy = base_report.overall_accuracy
    per_track_accuracy = _aggregate_track_accuracy(case_results)
    return MultiSessionSynthesisReport(
        case_results=case_results,
        overall_accuracy=overall_accuracy,
        per_track_accuracy=per_track_accuracy,
        dataset_label=str(dataset_path),
        passed_target=_meets_target(overall_accuracy, per_track_accuracy, target=0.85),
        track_target=0.85,
    )


def _reify_native_track(instance: OfficialLongMemEvalInstance) -> str:
    if instance.question_id.endswith("_abs"):
        return "abstention"
    return {
        "single-session-user": "direct-recall",
        "single-session-assistant": "assistant-continuity",
        "single-session-preference": "preference-grounding",
        "multi-session": "cross-session",
        "temporal-reasoning": "temporal-recall",
        "knowledge-update": "update-truth",
    }.get(instance.question_type, instance.question_type)


def _aggregate_track_accuracy(
    results: tuple[MultiSessionSynthesisCaseResult, ...],
) -> dict[str, float]:
    grouped: dict[str, list[bool]] = {}
    for result in results:
        grouped.setdefault(result.case.track, []).append(result.correct)
    return {
        track: sum(labels) / len(labels)
        for track, labels in grouped.items()
        if labels
    }


def _meets_target(
    overall_accuracy: float,
    per_track_accuracy: dict[str, float],
    *,
    target: float,
) -> bool:
    return overall_accuracy >= target and all(
        accuracy >= target for accuracy in per_track_accuracy.values()
    )
