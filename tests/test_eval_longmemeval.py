from __future__ import annotations

import json
from pathlib import Path

from reify.eval import (
    adapt_official_longmemeval_instances,
    build_official_longmemeval_judge_prompt,
    default_longmemeval_fixture_cases,
    load_longmemeval_cases,
    load_official_longmemeval_instances,
    run_official_longmemeval_suite,
    run_longmemeval_suite,
)
from reify.eval.longmemeval import (
    LongMemEvalCase,
    LongMemEvalEvent,
    _extract_preference_rubric_concepts,
    _preference_proxy_supported,
)
from reify.core.types import Experience
from datetime import datetime, timezone
from reify.llm.base import LLMRequest, LLMResponse


class FakeJudgeProvider:
    name = "fake"
    default_model = "fake-model"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = request.messages[0]["content"]
        if "Retrieved Memory" in prompt:
            if "Seattle" in prompt:
                content = "Seattle."
            else:
                content = "I don't know based on the provided history."
        else:
            content = "yes" if "Seattle" in prompt and "Model Response: Seattle." in prompt else "no"
        return LLMResponse(content=content, model=self.default_model)


async def test_run_longmemeval_fixture_suite_meets_initial_target() -> None:
    report = await run_longmemeval_suite()

    assert report.passed_target is True
    assert len(report.case_results) == len(default_longmemeval_fixture_cases())
    assert report.overall_accuracy == 1.0
    assert report.per_category_accuracy == {
        "single_hop": 1.0,
        "temporal": 1.0,
    }


def test_load_longmemeval_cases_from_jsonl(tmp_path: Path) -> None:
    dataset = tmp_path / "longmemeval.jsonl"
    dataset.write_text(
        (
            '{"name":"sample","category":"single_hop","identity_id":"user123",'
            '"events":[{"content":"I live in Seattle.","at":"2026-01-01T12:00:00+00:00"}],'
            '"query":"Where do I live?","query_at":"2026-01-31T12:00:00+00:00",'
            '"expected_substrings":["Seattle"]}\n'
        ),
        encoding="utf-8",
    )

    cases = load_longmemeval_cases(dataset)
    assert len(cases) == 1
    assert cases[0].name == "sample"
    assert cases[0].category == "single_hop"
    assert cases[0].expected_substrings == ("Seattle",)


def test_load_official_longmemeval_instances_and_adapt(tmp_path: Path) -> None:
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
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "I live in Seattle.", "has_answer": True},
                            {"role": "assistant", "content": "Thanks for sharing."},
                        ]
                    ],
                    "answer_session_ids": ["s1"],
                }
            ]
        ),
        encoding="utf-8",
    )

    instances = load_official_longmemeval_instances(dataset)
    assert len(instances) == 1
    assert instances[0].question_id == "q1"
    assert instances[0].question_type == "single-session-user"

    cases = adapt_official_longmemeval_instances(instances)
    assert len(cases) == 1
    assert cases[0].query == "Where do I live?"
    assert cases[0].expected_substrings == ("Seattle",)
    assert len(cases[0].events) == 2


async def test_run_official_longmemeval_suite_with_fake_judge(tmp_path: Path) -> None:
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
                    "haystack_sessions": [
                        [
                            {"role": "user", "content": "I live in Seattle.", "has_answer": True},
                            {"role": "assistant", "content": "Thanks for sharing."},
                        ]
                    ],
                    "answer_session_ids": ["s1"],
                }
            ]
        ),
        encoding="utf-8",
    )
    instances = load_official_longmemeval_instances(dataset)
    provider = FakeJudgeProvider()
    report = await run_official_longmemeval_suite(
        instances=instances,
        answer_provider=provider,
        answer_model=provider.default_model,
        judge_provider=provider,
        judge_model=provider.default_model,
        dataset_label=str(dataset),
    )

    assert report.passed_target is True
    assert report.overall_accuracy == 1.0
    assert report.task_averaged_accuracy == 1.0
    assert report.per_question_type_accuracy == {"single-session-user": 1.0}
    assert report.results[0].hypothesis == "Seattle."


def test_build_official_longmemeval_judge_prompt_abstention() -> None:
    prompt = build_official_longmemeval_judge_prompt(
        question_type="single-session-user",
        question="What is my dog called?",
        answer="The history never mentioned a dog.",
        response="I don't know based on the provided history.",
        abstention=True,
    )
    assert "unanswerable question" in prompt
    assert "Answer yes or no only." in prompt


def test_extract_preference_rubric_concepts() -> None:
    concepts = _extract_preference_rubric_concepts(
        "The user would prefer responses that suggest resources specifically tailored to "
        "Adobe Premiere Pro, especially those that delve into its advanced settings."
    )
    assert "Adobe Premiere Pro" in concepts
    assert any("advanced settings" in concept.lower() for concept in concepts)


def test_preference_proxy_supported_by_retrieved_memories() -> None:
    memories = [
        Experience(
            content="I'm trying to learn more about some advanced settings for video editing with Adobe Premiere Pro.",
            event_type="user_message",
            actor="user",
            summary="Adobe Premiere Pro advanced settings",
        ),
        Experience(
            content="I enjoy to use Adobe Premiere Pro for color grading work.",
            event_type="user_message",
            actor="user",
            summary="Premiere Pro enjoyment",
        ),
    ]
    assert _preference_proxy_supported(
        memories,
        rubric=(
            "The user would prefer responses that suggest resources specifically tailored "
            "to Adobe Premiere Pro, especially those that delve into its advanced settings."
        ),
    )


async def test_run_longmemeval_suite_accepts_word_number_expected_answers() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    case = LongMemEvalCase(
        name="festival_count",
        category="multi-session",
        identity_id="festival-count",
        events=(
            LongMemEvalEvent(
                content="I volunteered at the Portland Film Festival.",
                at=now,
            ),
            LongMemEvalEvent(
                content="I recently participated in the 48-hour film challenge at the Austin Film Festival.",
                at=now,
            ),
            LongMemEvalEvent(
                content="I just got back from AFI Fest in LA.",
                at=now,
            ),
            LongMemEvalEvent(
                content="I got to discuss the narrative structure after the screening at the Seattle International Film Festival.",
                at=now,
            ),
        ),
        query="How many movie festivals that I attended?",
        query_at=now,
        expected_substrings=("I attended four movie festivals.",),
    )

    report = await run_longmemeval_suite(cases=(case,), dataset_label="festival-test")
    assert report.overall_accuracy == 1.0


async def test_run_longmemeval_suite_accepts_study_abroad_school_without_country() -> None:
    now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    case = LongMemEvalCase(
        name="study_abroad",
        category="single-session-user",
        identity_id="study-abroad",
        events=(
            LongMemEvalEvent(
                content="I actually went there with some friends during my study abroad program at the University of Melbourne.",
                at=now,
            ),
        ),
        query="Where did I attend for my study abroad program?",
        query_at=now,
        expected_substrings=("University of Melbourne in Australia",),
    )

    report = await run_longmemeval_suite(cases=(case,), dataset_label="study-abroad-test")
    assert report.overall_accuracy == 1.0
