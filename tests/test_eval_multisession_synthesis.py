from __future__ import annotations

import json
from pathlib import Path

from reify.eval import (
    adapt_multisession_synthesis_cases,
    load_official_longmemeval_instances,
    run_multisession_synthesis_suite,
)


def test_adapt_multisession_synthesis_cases_groups_into_reify_tracks(tmp_path: Path) -> None:
    dataset = tmp_path / "official.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "Where do I live?",
                    "answer": "Seattle",
                    "question_date": "2026/01/31",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/01/01 (Thu) 12:00"],
                    "haystack_sessions": [[{"role": "user", "content": "I live in Seattle."}]],
                    "answer_session_ids": ["s1"],
                },
                {
                    "question_id": "q2_abs",
                    "question_type": "multi-session",
                    "question": "What is my dog's name?",
                    "answer": "The history never mentioned a dog.",
                    "question_date": "2026/01/31",
                    "haystack_session_ids": ["s2"],
                    "haystack_dates": ["2026/01/01 (Thu) 12:00"],
                    "haystack_sessions": [[{"role": "user", "content": "I live in Seattle."}]],
                    "answer_session_ids": ["s2"],
                },
            ]
        ),
        encoding="utf-8",
    )
    instances = load_official_longmemeval_instances(dataset)
    cases = adapt_multisession_synthesis_cases(instances)
    assert cases[0].track == "direct-recall"
    assert cases[1].track == "abstention"


async def test_run_multisession_synthesis_suite_reports_reify_tracks(tmp_path: Path) -> None:
    dataset = tmp_path / "official.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "Where do I live?",
                    "answer": "Seattle",
                    "question_date": "2026/01/31",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/01/01 (Thu) 12:00"],
                    "haystack_sessions": [[{"role": "user", "content": "I live in Seattle."}]],
                    "answer_session_ids": ["s1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    report = await run_multisession_synthesis_suite(dataset_path=dataset)
    assert report.overall_accuracy == 1.0
    assert report.per_track_accuracy == {"direct-recall": 1.0}
    assert report.case_results[0].case.track == "direct-recall"
    assert report.passed_target is True


async def test_run_multisession_synthesis_suite_requires_all_tracks_to_clear_target(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "official.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "question_id": "q1",
                    "question_type": "single-session-user",
                    "question": "Where do I live?",
                    "answer": "Seattle",
                    "question_date": "2026/01/31",
                    "haystack_session_ids": ["s1"],
                    "haystack_dates": ["2026/01/01 (Thu) 12:00"],
                    "haystack_sessions": [[{"role": "user", "content": "I live in Seattle."}]],
                    "answer_session_ids": ["s1"],
                },
                {
                    "question_id": "q2",
                    "question_type": "temporal-reasoning",
                    "question": "When did I move to Austin?",
                    "answer": "2025-01-01",
                    "question_date": "2026/01/31",
                    "haystack_session_ids": ["s2"],
                    "haystack_dates": ["2026/01/01 (Thu) 12:00"],
                    "haystack_sessions": [[{"role": "user", "content": "I live in Seattle."}]],
                    "answer_session_ids": ["s2"],
                },
            ]
        ),
        encoding="utf-8",
    )
    report = await run_multisession_synthesis_suite(dataset_path=dataset)
    assert report.overall_accuracy == 0.5
    assert report.per_track_accuracy == {"direct-recall": 1.0, "temporal-recall": 0.0}
    assert report.passed_target is False
