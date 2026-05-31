"""Tests for :class:`reify.Reify` (the engine).

The engine sits between extraction, storage, and recall. These tests focus
on the engine's contract: lifecycle, the foreground write path, recall
intent inference, transitions surfacing, ``whatdoyouknow``, ``forget``,
and markdown export.

The four-step wedge demo lives in :mod:`tests.test_wedge_demo` so that the
acceptance bar is one named test you can run in isolation.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from reify import EngineConfig, Reify
from reify.core.errors import ReifyError, UnknownMemoryError
from reify.core.ids import new_id
from reify.core.types import Fact, Inference, Network, RelationType, Transition, TransitionKind
from reify.llm.base import LLMRequest, LLMResponse
from reify.storage import RecallFilter, SQLiteStore, WriteBatch


NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


class FakeLLMProvider:
    name = "fake"
    default_model = "fake-model"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        return LLMResponse(
            content=(
                '[{"type":"resolves","confidence":0.93,"explanation":"new fact supersedes old"}]'
            ),
            model=self.default_model,
        )


class FencedJsonLLMProvider:
    name = "fake"
    default_model = "fake-model"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        return LLMResponse(
            content=(
                "```json\n"
                '[{"type":"resolves","confidence":0.91,"explanation":"new fact supersedes old"}]\n'
                "```"
            ),
            model=self.default_model,
        )


class MutableClock:
    def __init__(self, now: datetime) -> None:
        self.current = now

    def set(self, now: datetime) -> None:
        self.current = now

    def __call__(self) -> datetime:
        return self.current


def _clock_at(t: datetime):
    """Return a clock callable that always returns ``t``."""
    return lambda: t


@pytest.fixture
async def engine() -> Reify:
    """A fresh in-memory engine with a frozen clock."""
    e = Reify(storage=SQLiteStore(":memory:"), clock=_clock_at(NOW))
    await e.start()
    try:
        yield e
    finally:
        await e.stop()


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


async def test_start_is_idempotent() -> None:
    e = Reify(storage=SQLiteStore(":memory:"), clock=_clock_at(NOW))
    await e.start()
    await e.start()
    await e.stop()


async def test_operations_before_start_raise() -> None:
    e = Reify(storage=SQLiteStore(":memory:"))
    with pytest.raises(ReifyError):
        await e.remember("hello")


async def test_engine_async_context_manager() -> None:
    async with Reify(storage=SQLiteStore(":memory:"), clock=_clock_at(NOW)) as e:
        receipt = await e.remember("I'm a Senior Engineer.")
        assert len(receipt.committed) >= 1


async def test_engine_requires_at_least_one_extractor() -> None:
    with pytest.raises(ReifyError):
        Reify(storage=SQLiteStore(":memory:"), extractors=[])


# ---------------------------------------------------------------------------
# remember()
# ---------------------------------------------------------------------------


async def test_remember_returns_committed_memories(engine: Reify) -> None:
    receipt = await engine.remember("I just got promoted to Staff Engineer.")
    # 1 Experience + 1 Fact = 2
    assert len(receipt.committed) == 2
    assert receipt.enqueued_count == 1


async def test_remember_records_provenance_source(engine: Reify) -> None:
    receipt = await engine.remember(
        "I work at Anthropic.", source="windsurf", session_id="sess-7"
    )
    fact = next(m for m in receipt.committed if isinstance(m, Fact))
    assert any("windsurf" in p.source for p in fact.provenance)
    assert any(p.session_id == "sess-7" for p in fact.provenance)


async def test_remember_with_explicit_at_overrides_clock(engine: Reify) -> None:
    custom = datetime(2024, 6, 1, tzinfo=timezone.utc)
    receipt = await engine.remember(
        "I just got promoted to Staff Engineer.", at=custom
    )
    fact = next(m for m in receipt.committed if isinstance(m, Fact))
    assert fact.valid_at == custom


async def test_remember_attaches_default_retention_to_claims(engine: Reify) -> None:
    receipt = await engine.remember("I work at Anthropic.")
    fact = next(m for m in receipt.committed if isinstance(m, Fact))
    retention = fact.metadata["retention"]
    assert retention["ttl_days"] == 30.0
    assert retention["expires_at"] == (fact.valid_at + timedelta(days=30)).isoformat()


async def test_remember_empty_returns_empty_receipt(engine: Reify) -> None:
    receipt = await engine.remember("   ")
    assert receipt.committed == ()
    assert receipt.enqueued_count == 0


async def test_flush_arw_materializes_inference_and_transition() -> None:
    e = Reify(
        storage=SQLiteStore(":memory:"),
        clock=_clock_at(NOW),
        config=EngineConfig(llm=FakeLLMProvider()),
    )
    async with e:
        await e.remember("I just got promoted from Senior to Staff Engineer.")
        processed = await e.flush_arw()
        assert processed == 1

        response = await e.recall("inference")
        inferences = [r.memory for r in response.results if isinstance(r.memory, Inference)]
        assert len(inferences) == 1
        assert inferences[0].relation is RelationType.RESOLVES

        transitions = await e.whatchanged(days=30)
        assert len(transitions) == 1
        assert transitions[0].kind is TransitionKind.ROLE_CHANGE
        assert "changed from Senior to Staff Engineer" in transitions[0].summary


async def test_flush_arw_accepts_fenced_json_from_provider() -> None:
    e = Reify(
        storage=SQLiteStore(":memory:"),
        clock=_clock_at(NOW),
        config=EngineConfig(llm=FencedJsonLLMProvider()),
    )
    async with e:
        await e.remember("I just got promoted from Senior to Staff Engineer.")
        processed = await e.flush_arw()
        assert processed == 1
        transitions = await e.whatchanged(days=30)
        assert len(transitions) == 1


# ---------------------------------------------------------------------------
# recall()
# ---------------------------------------------------------------------------


async def test_recall_finds_current_title(engine: Reify) -> None:
    await engine.remember("I just got promoted from Senior to Staff Engineer.")
    response = await engine.recall("What's my title?")
    assert response.results, "expected at least one result for 'what's my title?'"

    # The current Staff Engineer fact must rank first; the Senior fact has
    # been bitemporally bounded and should be filtered out at the storage
    # stage by the ``valid_at = now`` constraint.
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "Staff Engineer"
    assert "title" in top.tags


async def test_recall_query_intent_extracts_tags(engine: Reify) -> None:
    response = await engine.recall("Where do I live?")
    assert "location" in response.query.tags


async def test_recall_previous_chat_prefers_assistant_memories(engine: Reify) -> None:
    await engine.remember("Can you recommend any good restaurants or cafes in the area?", actor="user")
    await engine.remember(
        "1. Miss Bee Providore: This restaurant serves a mix of western and Indonesian cuisine and has a cozy and stylish interior.",
        actor="assistant",
    )

    response = await engine.recall(
        "I'm going back to our previous conversation. Can you remind me of the restaurant name?",
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert "Miss Bee Providore" in top.obj


async def test_recall_previous_chat_prefers_typed_assistant_options(engine: Reify) -> None:
    await engine.remember("Can you help me get from the airport to my hotel in Tokyo?", actor="user")
    await engine.remember(
        "1. N'EX (Narita Express) Train: This is the fastest way to get to Tokyo Station ( approx. 60 minutes, ¥3,000-4,000).\n"
        "4. Taxi: Taxis are available, but this is the most expensive option (approx. 1-2 hours, ¥15,000-20,000).",
        actor="assistant",
    )

    response = await engine.recall("What was the taxi option you mentioned?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert "assistant-option" in top.tags
    assert top.metadata["assistant_reference"]["transport_mode"] == "taxi"


async def test_recall_temporal_compare_order(engine: Reify) -> None:
    await engine.remember(
        'I participated in a webinar on "Data Analysis using Python" two months ago.',
        at=datetime(2023, 3, 28, tzinfo=timezone.utc),
    )
    await engine.remember(
        'I attended the workshop on "Effective Time Management" on May 20th.',
        at=datetime(2023, 5, 28, tzinfo=timezone.utc),
    )
    response = await engine.recall(
        'Which event did I attend first, the "Effective Time Management" workshop or the "Data Analysis using Python" webinar?',
    )
    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "Data Analysis using Python"


async def test_recall_temporal_delta_days(engine: Reify) -> None:
    await engine.remember(
        'I attended the workshop on "Effective Communication in the Workplace" on January 10th.',
        at=datetime(2023, 1, 13, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I remember making a note to myself to practice those skills in my upcoming team meeting on January 17th.",
        at=datetime(2023, 1, 13, tzinfo=timezone.utc),
    )
    response = await engine.recall(
        "How many days before the team meeting I was preparing for did I attend the workshop on 'Effective Communication in the Workplace'?",
    )
    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "7 days"


async def test_recall_temporal_first_issue_after_service(engine: Reify) -> None:
    await engine.remember(
        "I just got my car serviced for the first time on March 1st.",
        at=datetime(2023, 4, 10, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I recently had an issue with my GPS system not functioning correctly.",
        at=datetime(2023, 4, 10, tzinfo=timezone.utc),
    )
    response = await engine.recall(
        "What was the first issue I had with my new car after its first service?",
    )
    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert "GPS" in top.obj


async def test_recall_temporal_between_generic_events(engine: Reify) -> None:
    await engine.remember(
        "I recently attended the Sunday mass at St. Mary's Church on January 2nd.",
        at=datetime(2023, 2, 20, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I just came from the Ash Wednesday service at the cathedral on February 1st.",
        at=datetime(2023, 2, 20, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        "How many days had passed between the Sunday mass at St. Mary's Church and the Ash Wednesday service at the cathedral?",
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "30 days"


async def test_recall_temporal_house_after_agent_start(engine: Reify) -> None:
    await engine.remember(
        "Since I started working with her on 2/15, I'm hoping she can give me a better sense of what's available in my budget.",
        at=datetime(2022, 3, 2, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I recently saw a house that I really love on 3/1, and I'm considering making an offer.",
        at=datetime(2022, 3, 2, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        "How many days did it take for me to find a house I loved after starting to work with Rachel?",
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "14 days"


async def test_recall_temporal_age_at_move_event(engine: Reify) -> None:
    await engine.remember(
        "I'm 32-year-old male and I've been living in the United States for the past five years on a work visa.",
        at=datetime(2023, 5, 29, tzinfo=timezone.utc),
    )

    response = await engine.recall("How old was I when I moved to the United States?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "27"


async def test_recall_temporal_recently_started_streaming_service(engine: Reify) -> None:
    await engine.remember(
        "I've been using Netflix, Hulu, and Amazon Prime for the past 6 months.",
        at=datetime(2023, 5, 22, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I've also been using Apple TV+ for a few months now.",
        at=datetime(2023, 5, 22, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I started a free trial of Disney+ last month.",
        at=datetime(2023, 5, 22, tzinfo=timezone.utc),
    )

    response = await engine.recall("Which streaming service did I start using most recently?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "Disney+"


async def test_recall_aggregate_total_people_reached(engine: Reify) -> None:
    await engine.remember(
        "I recently collaborated with an influencer who promoted my product to her 10,000 followers.",
        at=datetime(2023, 5, 25, tzinfo=timezone.utc),
    )
    await engine.remember(
        "My previous Facebook ad campaign reached around 2,000 people and resulted in 50 clicks to my website.",
        at=datetime(2023, 5, 27, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        "What was the total number of people reached by my Facebook ad campaign and Instagram influencer collaboration?"
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "12000"


async def test_recall_aggregate_page_count_by_month(engine: Reify) -> None:
    await engine.remember(
        "I just finished a 416-page novel.",
        at=datetime(2023, 1, 12, tzinfo=timezone.utc),
    )
    await engine.remember(
        'Before that, I read "The Power" by Naomi Alderman in December, which had 341 pages and took me around 5 weeks to finish.',
        at=datetime(2023, 1, 12, tzinfo=timezone.utc),
    )
    await engine.remember(
        'I just finished reading "The Nightingale" by Kristin Hannah, which had 440 pages and took me around 3 weeks to complete.',
        at=datetime(2023, 3, 8, tzinfo=timezone.utc),
    )

    response = await engine.recall("What was the page count of the two novels I finished in January and March?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "856"


async def test_recall_aggregate_page_count_falls_back_when_query_months_do_not_match_event_timestamps(
    engine: Reify,
) -> None:
    await engine.remember(
        "I just finished a 416-page novel.",
        at=datetime(2023, 5, 22, 10, 41, tzinfo=timezone.utc),
    )
    await engine.remember(
        'I just finished a historical fiction novel, "The Nightingale" by Kristin Hannah, which had 440 pages and took me around 3 weeks to complete.',
        at=datetime(2023, 5, 27, 7, 7, tzinfo=timezone.utc),
    )

    response = await engine.recall("What was the page count of the two novels I finished in January and March?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "856"


async def test_recall_aggregate_total_siblings(engine: Reify) -> None:
    await engine.remember(
        "I have a brother.",
        at=datetime(2023, 2, 10, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I come from a family with 3 sisters.",
        at=datetime(2023, 4, 10, tzinfo=timezone.utc),
    )

    response = await engine.recall("What is the total number of siblings I have?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "4"


async def test_recall_aggregate_handbag_savings(engine: Reify) -> None:
    await engine.remember(
        "I've had luck finding great deals at TK Maxx before, like that designer handbag I got for $200.",
        at=datetime(2023, 4, 10, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I got a fantastic deal on the bag - it was originally $500!",
        at=datetime(2023, 4, 12, tzinfo=timezone.utc),
    )
    response = await engine.recall("How much did I save on the designer handbag?")
    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "$300"


async def test_recall_aggregate_airport_transport_savings(engine: Reify) -> None:
    await engine.remember(
        "No, I'm pretty sure my friend said it's $10 to get to my hotel from the airport by train.",
        at=datetime(2023, 5, 10, tzinfo=timezone.utc),
    )
    await engine.remember(
        "By the way, I was told that taking a taxi from the airport to my hotel would cost around $60, which is a bit pricey for me.",
        at=datetime(2023, 5, 11, tzinfo=timezone.utc),
    )
    response = await engine.recall("How much will I save by taking the train from the airport to my hotel instead of a taxi?")
    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "$50"


async def test_recall_temporal_guitar_lessons_before_amp_purchase(engine: Reify) -> None:
    await engine.remember(
        "I've been taking weekly guitar lessons with a new instructor, Alex, for six weeks now and it's really helped me understand music theory better.",
        at=datetime(2023, 5, 1, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I've been using a metronome to help with my timing, but I'm not sure if I'm using it to its full potential. By the way, I recently bought a new guitar amp two weeks ago, which has made a huge difference in the sound quality.",
        at=datetime(2023, 5, 1, tzinfo=timezone.utc),
    )
    response = await engine.recall("How long had I been taking guitar lessons when I bought the new guitar amp?")
    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "four weeks"


async def test_recall_temporal_compare_people_met_first(engine: Reify) -> None:
    await engine.remember(
        "I met Tom at a conference last month.",
        at=datetime(2023, 5, 28, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I met Mark and Sarah at a coffee shop last week.",
        at=datetime(2023, 5, 28, tzinfo=timezone.utc),
    )

    response = await engine.recall("Who did I meet first, Mark and Sarah or Tom?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "Tom"


async def test_recall_temporal_compare_parenthood(engine: Reify) -> None:
    await engine.remember(
        "My cousin Alex just adopted a baby girl from China in January.",
        at=datetime(2023, 3, 17, tzinfo=timezone.utc),
    )
    await engine.remember(
        "My sister-in-law, Rachel, is doing great with the twins, Jackson and Julia, who recently celebrated their first birthday.",
        at=datetime(2023, 3, 17, tzinfo=timezone.utc),
    )

    response = await engine.recall("Who became a parent first, Rachel or Alex?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "Rachel"


async def test_recall_temporal_first_date_in_month(engine: Reify) -> None:
    await engine.remember(
        "I attended the first BBQ event on June 3rd.",
        at=datetime(2023, 7, 1, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I attended another BBQ event on June 18th.",
        at=datetime(2023, 7, 1, tzinfo=timezone.utc),
    )

    response = await engine.recall("What was the date on which I attended the first BBQ event in June?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "June 3rd"


async def test_recall_temporal_wake_time_with_relative_offset(engine: Reify) -> None:
    await engine.remember(
        "I've recently started waking up at 7:00 AM.",
        at=datetime(2023, 5, 30, tzinfo=timezone.utc),
    )
    await engine.remember(
        "On Tuesdays and Thursdays, I've also started waking up 15 minutes earlier to meditate.",
        at=datetime(2023, 5, 30, tzinfo=timezone.utc),
    )

    response = await engine.recall("What time do I wake up on Tuesdays and Thursdays?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "6:45 AM"


async def test_recall_temporal_seed_compare_order(engine: Reify) -> None:
    await engine.remember(
        "I just started some marigold seeds that arrived on March 3rd, and they're already germinating.",
        at=datetime(2023, 3, 10, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I've been starting seeds indoors under grow lights in my basement since February 20th - tomatoes, peppers, and cucumbers are all doing well.",
        at=datetime(2023, 3, 10, tzinfo=timezone.utc),
    )

    response = await engine.recall("Which seeds were started first, the tomatoes or the marigolds?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "tomatoes"


async def test_recall_temporal_purchase_compare_order_uses_query_labels(engine: Reify) -> None:
    await engine.remember(
        "I'm having some issues with my coffee maker's performance lately. I bought it about three weeks ago, and I've been using it every morning since then.",
        at=datetime(2023, 5, 13, tzinfo=timezone.utc),
    )
    await engine.remember(
        "Do you have any recommendations for cakes or cookies that don't require a stand mixer? By the way, I had to take mine to a repair shop last month and it took two weeks to fix.",
        at=datetime(2023, 5, 13, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        "Which event happened first, the purchase of the coffee maker or the malfunction of the stand mixer?",
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "malfunction of the stand mixer"


async def test_recall_temporal_item_compare_order_with_owner_suffixes(engine: Reify) -> None:
    await engine.remember(
        "I've been using those eco-friendly training pads from Chewy.com. I got a set of 10 for $25 about a month ago, and they've been a lifesaver.",
        at=datetime(2023, 5, 29, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I recently got a new Orthopedic Memory Foam dog bed for my golden retriever, Max, about three weeks ago from Petco.",
        at=datetime(2023, 5, 29, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        "Which item did I purchase first, the dog bed for Max or the training pads for Luna?",
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "training pads for Luna"


async def test_recall_temporal_lens_vs_road_trip_compare_order(engine: Reify) -> None:
    await engine.remember(
        "I'm especially interested in mastering my 50mm lens that I got a month ago.",
        at=datetime(2023, 5, 30, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I just got back from a road trip to the coast with friends last week and took some amazing shots of the cliffs and beaches.",
        at=datetime(2023, 5, 30, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        "Which event happened first, the road trip to the coast or the arrival of the new prime lens?",
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "arrival of the new prime lens"


async def test_recall_temporal_vehicle_compare_order_with_month_constraint(engine: Reify) -> None:
    await engine.remember(
        "I've been having some issues with my bike lately - in mid-February, I had to take it in for repairs because the gears were acting up.",
        at=datetime(2023, 3, 10, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I just washed my current Corolla on Monday, February 27th, after not doing so for over a month.",
        at=datetime(2023, 3, 10, tzinfo=timezone.utc),
    )

    response = await engine.recall("Which vehicle did I take care of first in February, the bike or the car?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "bike"


async def test_recall_temporal_days_since_purchase_when_event(engine: Reify) -> None:
    await engine.remember(
        "I recently got a new pair of Adidas running shoes on January 10th, and I want to make sure I take good care of them.",
        at=datetime(2023, 2, 10, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I realized that one of the shoelaces on my old Converse sneakers had broken on January 24th, so I had to replace it.",
        at=datetime(2023, 2, 10, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        "How many days had passed since I bought my Adidas running shoes when I realized one of the shoelaces on my old Converse sneakers had broken?",
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "14 days"


async def test_recall_shoe_cleaning_fact(engine: Reify) -> None:
    await engine.remember(
        "I cleaned my white Adidas sneakers last month, which I'd been meaning to do for weeks.",
        at=datetime(2026, 1, 15, tzinfo=timezone.utc),
    )

    response = await engine.recall("Which pair of shoes did I clean last month?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "white Adidas sneakers"


async def test_recall_temporal_charity_event_count_before_reference(engine: Reify) -> None:
    await engine.remember(
        'I volunteered at the "Food for Thought" charity gala on September 25th.',
        at=datetime(2023, 11, 29, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I attended a charity golf tournament on July 17th.",
        at=datetime(2023, 11, 29, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I volunteered at the Walk for Wildlife event in June.",
        at=datetime(2023, 11, 29, tzinfo=timezone.utc),
    )
    await engine.remember(
        'I participated in the "Dance for a Cause" event on May 1st.',
        at=datetime(2023, 11, 29, tzinfo=timezone.utc),
    )
    await engine.remember(
        'I ran 5 kilometers in the "Run for the Cure" event on October 15th.',
        at=datetime(2023, 11, 29, tzinfo=timezone.utc),
    )

    response = await engine.recall(
        'How many charity events did I participate in before the "Run for the Cure" event?'
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "4"


async def test_recall_temporal_working_before_current_job_in_months(engine: Reify) -> None:
    await engine.remember(
        "I've been working professionally for 9 years.",
        at=datetime(2023, 5, 1, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I've been working at NovaTech for about 4 years and 3 months now.",
        at=datetime(2023, 5, 1, tzinfo=timezone.utc),
    )

    response = await engine.recall("How long have I been working before I started my current job at NovaTech?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "4 years and 9 months"


async def test_recall_temporal_friday_wake_delta(engine: Reify) -> None:
    await engine.remember(
        "I usually do them right after waking up at 6:30 AM on weekdays.",
        at=datetime(2023, 5, 30, tzinfo=timezone.utc),
    )
    await engine.remember(
        "On Fridays, I like to get a head start, so I wake up at 6:00 AM.",
        at=datetime(2023, 5, 30, tzinfo=timezone.utc),
    )

    response = await engine.recall("How much earlier do I wake up on Fridays compared to other weekdays?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "30 minutes"


async def test_recall_temporal_membership_duration_in_weeks(engine: Reify) -> None:
    await engine.remember(
        'I recently joined a Facebook group called "Book Lovers Unite" three weeks ago.',
        at=datetime(2023, 5, 24, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I attended a meetup organized by Book Lovers Unite last week.",
        at=datetime(2023, 5, 24, tzinfo=timezone.utc),
    )

    response = await engine.recall("How long had I been a member of 'Book Lovers Unite' when I attended the meetup?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "two weeks"


async def test_recall_temporal_setup_compare_order_with_mesh_network(engine: Reify) -> None:
    await engine.remember(
        "Since I set up my smart thermostat a month ago, I've noticed lower energy bills.",
        at=datetime(2023, 5, 24, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I recently upgraded my home Wi-Fi router 3 weeks ago, which improved coverage throughout the house.",
        at=datetime(2023, 5, 24, tzinfo=timezone.utc),
    )

    response = await engine.recall("Which device did I set up first, the smart thermostat or the mesh network system?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "smart thermostat"


async def test_recall_temporal_bird_watching_duration_in_months(engine: Reify) -> None:
    await engine.remember(
        "I've been getting into bird watching for about three months now.",
        at=datetime(2023, 5, 25, tzinfo=timezone.utc),
    )
    await engine.remember(
        "I recently attended a bird watching workshop at the local Audubon society a month ago.",
        at=datetime(2023, 5, 25, tzinfo=timezone.utc),
    )

    response = await engine.recall("How long had I been bird watching when I attended the bird watching workshop?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "two months"


async def test_recall_aggregate_obligations_count(engine: Reify) -> None:
    await engine.remember("I still need to pick up my dry cleaning for the navy blue blazer.")
    await engine.remember("I need to return some boots to Zara.")
    await engine.remember("I just exchanged a pair of boots and I still need to pick up the new pair.")

    response = await engine.recall("How many items of clothing do I need to pick up or return from a store?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "3"


async def test_recall_aggregate_project_count(engine: Reify) -> None:
    await engine.remember("I've been working on a solo project for my Data Mining class.")
    await engine.remember("I'm working on a project that involves analyzing customer data to identify trends and patterns.")
    await engine.remember("I led the data analysis team for a comprehensive market analysis for a new product launch.")

    response = await engine.recall("How many projects have I led or am currently leading?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "2"


async def test_recall_aggregate_trip_days(engine: Reify) -> None:
    await engine.remember("I just got back from a 3-day solo camping trip to Big Sur in early April.")
    await engine.remember("I just got back from an amazing 5-day camping trip to Yellowstone National Park last month.")
    await engine.remember(
        "We had a 7-day family road trip there in February. We did a lot of driving and hiking, but not camping for this time."
    )

    response = await engine.recall("How many days did I spend on camping trips in the United States this year?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "8 days"


async def test_recall_aggregate_media_weeks(engine: Reify) -> None:
    await engine.remember("I watched all 22 Marvel Cinematic Universe movies in two weeks.")
    await engine.remember("I just finished a Star Wars marathon, watched all the main films in a week and a half, it was a wild ride!")

    response = await engine.recall(
        "How many weeks did it take me to watch all the Marvel Cinematic Universe movies and the main Star Wars films?"
    )

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "3.5 weeks"


async def test_recall_aggregate_bike_expenses(engine: Reify) -> None:
    await engine.remember("I recently got a new set of bike lights installed, which were $40.")
    await engine.remember("I bought my Bell Zephyr helmet for $120.")
    await engine.remember("I needed to replace the chain, which cost me $25.")

    response = await engine.recall("How much total money have I spent on bike-related expenses since the start of the year?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "$185"


async def test_recall_aggregate_doctor_count(engine: Reify) -> None:
    await engine.remember("I just got back from a follow-up appointment with my dermatologist, Dr. Lee.")
    await engine.remember("I recently had a UTI and was prescribed antibiotics by my primary care physician, Dr. Smith.")
    await engine.remember("I just got diagnosed with chronic sinusitis by an ENT specialist, Dr. Patel.")

    response = await engine.recall("How many different doctors did I visit?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "3"


async def test_recall_aggregate_plant_count(engine: Reify) -> None:
    await engine.remember("I bought the peace lily and a succulent plant two weeks ago.")
    await engine.remember("I'm also wondering if I should repot my snake plant, which I got from my sister last month.")

    response = await engine.recall("How many plants did I acquire in the last month?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "3"


async def test_recall_aggregate_social_media_break_days(engine: Reify) -> None:
    await engine.remember(
        "I've been making an effort to cut down on social media lately - I even took a week-long break from it in mid-January."
    )
    await engine.remember(
        "I've been making an effort to cut down on social media lately - I actually just got back from a 10-day break in mid-February."
    )

    response = await engine.recall("How many days did I take social media breaks in total?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "17 days"


async def test_recall_aggregate_luxury_spend_dedupes_brand_only_duplicate(engine: Reify) -> None:
    await engine.remember(
        "I recently bought a luxury evening gown for a wedding. It was a big purchase, $800, but I felt like I needed to make a good impression."
    )
    await engine.remember(
        "I've also made some luxury purchases, like a pair of leather boots from a high-end Italian designer that I got for $500."
    )
    await engine.remember(
        "I've been noticing that I tend to splurge on luxury items every now and then, like that designer handbag I just got from Gucci for $1,200."
    )

    response = await engine.recall("What is the total amount I spent on luxury items in the past few months?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "$2500"


async def test_recall_aggregate_gaming_hours_dedupes_repeat_mentions(engine: Reify) -> None:
    await engine.remember("I spent around 70 hours playing Assassin's Creed Odyssey.")
    await engine.remember("I'm looking for some recommendations on games similar to The Last of Us Part II. By the way, I just finished it on normal difficulty and it took me 25 hours to complete.")
    await engine.remember("I spent around 30 hours playing The Last of Us Part II on hard difficulty.")
    await engine.remember("I'm trying to find some new indie games similar to Celeste, which took me 10 hours to complete.")
    await engine.remember("I'm looking for some recommendations for indie games similar to Hyper Light Drifter, which took me 5 hours to finish, by the way.")
    await engine.remember("Do you have any recommendations for games similar to The Last of Us Part II, which I recently completed on both normal and hard difficulties?")

    response = await engine.recall("How many hours have I spent playing games in total?")

    assert response.results
    top = response.results[0].memory
    assert isinstance(top, Fact)
    assert top.obj == "140 hours"


async def test_recall_with_no_data_returns_empty(engine: Reify) -> None:
    response = await engine.recall("What's my title?")
    assert response.results == ()


async def test_recall_honors_explicit_valid_at(engine: Reify) -> None:
    # First-event clock for the write; later clock for the recall.
    earlier = NOW - timedelta(days=60)
    later = NOW
    await engine.remember(
        "I just got promoted from Senior to Staff Engineer.", at=earlier
    )
    # Querying as-of just after the promotion: only the new title is current.
    just_after = earlier + timedelta(seconds=1)
    response = await engine.recall("title", valid_at=just_after)
    titles = [r.memory.obj for r in response.results if isinstance(r.memory, Fact)]
    assert "Staff Engineer" in titles
    # Senior was bounded with valid_until = earlier, so it must be gone.
    assert "Senior" not in titles
    assert later  # silence unused warning if test simplifies


# ---------------------------------------------------------------------------
# whatchanged()
# ---------------------------------------------------------------------------


async def test_whatchanged_returns_recent_transitions(engine: Reify) -> None:
    # Manually plant a transition so we don't depend on ARW for this test.
    a = Fact(content="Senior", subject="user", predicate="has_title", obj="Senior")
    b = Fact(content="Staff", subject="user", predicate="has_title", obj="Staff")
    transition_at = NOW - timedelta(days=2)
    t = Transition(
        kind=TransitionKind.ROLE_CHANGE,
        from_memory_id=a.id,
        to_memory_id=b.id,
        at=transition_at,
        summary="Promoted from Senior to Staff.",
    )
    # Reach into storage directly for the planting; the engine's surface
    # won't accept handcrafted transitions until v0.2.
    await engine._storage.commit_batch(
        WriteBatch(memories=[a, b], transitions=[t])
    )

    out = await engine.whatchanged(days=30)
    assert len(out) == 1
    assert out[0].kind is TransitionKind.ROLE_CHANGE
    assert out[0].summary.startswith("Promoted")


async def test_whatchanged_window_excludes_old_transitions(engine: Reify) -> None:
    a = Fact(content="A", subject="user", predicate="has_title", obj="A")
    b = Fact(content="B", subject="user", predicate="has_title", obj="B")
    long_ago = NOW - timedelta(days=120)
    t = Transition(
        kind=TransitionKind.ROLE_CHANGE,
        from_memory_id=a.id,
        to_memory_id=b.id,
        at=long_ago,
    )
    await engine._storage.commit_batch(
        WriteBatch(memories=[a, b], transitions=[t])
    )
    out = await engine.whatchanged(days=30)
    assert out == []


# ---------------------------------------------------------------------------
# whatdoyouknow()
# ---------------------------------------------------------------------------


async def test_whatdoyouknow_returns_everything_recent_first(engine: Reify) -> None:
    await engine.remember("I work at Anthropic.")
    await engine.remember("I live in San Francisco.")
    out = await engine.whatdoyouknow()
    # Each remember produces 1 Experience + 1 Fact => 4 total.
    assert len(out) >= 4
    # Most recent first.
    assert out[0].created_at >= out[-1].created_at


async def test_whatdoyouknow_with_topic_filters(engine: Reify) -> None:
    await engine.remember("I work at Stripe.")
    await engine.remember("I live in Seattle.")
    out = await engine.whatdoyouknow("employer")
    contents = [m.content for m in out]
    assert any("Stripe" in c for c in contents)
    # Location facts should not dominate; the employer-tagged one should be present.


# ---------------------------------------------------------------------------
# forget()
# ---------------------------------------------------------------------------


async def test_forget_removes_memory_from_default_recall(engine: Reify) -> None:
    receipt = await engine.remember("I'm a Senior Engineer.")
    fact = next(m for m in receipt.committed if isinstance(m, Fact))

    response_before = await engine.recall("title")
    assert any(r.memory.id == fact.id for r in response_before.results)

    await engine.forget(fact.id)

    response_after = await engine.recall("title")
    assert all(r.memory.id != fact.id for r in response_after.results)


async def test_forget_unknown_id_raises(engine: Reify) -> None:
    with pytest.raises(UnknownMemoryError):
        await engine.forget(new_id())


async def test_forget_by_tag_tombstones_matching_memories(engine: Reify) -> None:
    await engine.remember("I work at Stripe.")
    await engine.remember("I live in Seattle.")

    deleted = await engine.forget(tag="employer")
    assert deleted >= 1

    remaining = await engine.recall("employer")
    assert all("Stripe" not in r.memory.content for r in remaining.results)


async def test_recall_reinforces_ttl(engine: Reify) -> None:
    receipt = await engine.remember("I work at Anthropic.")
    fact = next(m for m in receipt.committed if isinstance(m, Fact))
    before = datetime.fromisoformat(fact.metadata["retention"]["expires_at"])

    await engine.recall("employer")

    refreshed = await engine._storage.get(fact.id)
    assert refreshed is not None
    after = datetime.fromisoformat(refreshed.metadata["retention"]["expires_at"])
    assert after > before
    assert refreshed.metadata["retention"]["reinforced_count"] == 1


async def test_expired_claims_are_swept_before_recall() -> None:
    expired_now = NOW + timedelta(days=31)
    engine = Reify(
        storage=SQLiteStore(":memory:"),
        clock=_clock_at(expired_now),
    )
    await engine.start()
    try:
        old_fact = Fact(
            content="Employer: OldCo",
            subject="user",
            predicate="works_at",
            obj="OldCo",
            valid_at=NOW,
            created_at=NOW,
            metadata={
                "retention": {
                    "ttl_days": 30.0,
                    "expires_at": (NOW + timedelta(days=30)).isoformat(),
                    "reinforced_count": 0,
                }
            },
        )
        await engine._storage.commit_batch(WriteBatch(memories=[old_fact]))

        response = await engine.recall("employer")
        assert response.results == ()

        swept = await engine._storage.get(old_fact.id)
        assert swept is not None
        assert swept.soft_deleted_at == expired_now
    finally:
        await engine.stop()


async def test_sweep_decay_emits_audit_log(caplog: pytest.LogCaptureFixture) -> None:
    expired_now = NOW + timedelta(days=31)
    engine = Reify(
        storage=SQLiteStore(":memory:"),
        clock=_clock_at(expired_now),
        config=EngineConfig(decay_sweep_interval_seconds=0),
    )
    await engine.start()
    try:
        old_fact = Fact(
            content="Employer: OldCo",
            subject="user",
            predicate="works_at",
            obj="OldCo",
            valid_at=NOW,
            created_at=NOW,
            metadata={
                "retention": {
                    "ttl_days": 30.0,
                    "expires_at": (NOW + timedelta(days=30)).isoformat(),
                    "reinforced_count": 0,
                }
            },
        )
        await engine._storage.commit_batch(WriteBatch(memories=[old_fact]))

        with caplog.at_level(logging.INFO, logger="reify.engine"):
            swept = await engine.sweep_decay(reason="test")

        assert swept == 1
        records = [r for r in caplog.records if getattr(r, "audit_event", "") == "reify_decay_sweep"]
        assert len(records) == 1
        record = records[0]
        assert record.reason == "test"
        assert record.swept_count == 1
        assert record.expired_memory_ids == [old_fact.id]
    finally:
        await engine.stop()


async def test_background_decay_sweeper_tombstones_expired_claims(
    caplog: pytest.LogCaptureFixture,
) -> None:
    expired_now = NOW + timedelta(days=31)
    engine = Reify(
        storage=SQLiteStore(":memory:"),
        clock=_clock_at(expired_now),
        config=EngineConfig(decay_sweep_interval_seconds=0.01),
    )
    await engine.start()
    try:
        old_fact = Fact(
            content="Employer: OldCo",
            subject="user",
            predicate="works_at",
            obj="OldCo",
            valid_at=NOW,
            created_at=NOW,
            metadata={
                "retention": {
                    "ttl_days": 30.0,
                    "expires_at": (NOW + timedelta(days=30)).isoformat(),
                    "reinforced_count": 0,
                }
            },
        )
        await engine._storage.commit_batch(WriteBatch(memories=[old_fact]))

        with caplog.at_level(logging.INFO, logger="reify.engine"):
            await asyncio.sleep(0.05)

        swept = await engine._storage.get(old_fact.id)
        assert swept is not None
        assert swept.soft_deleted_at == expired_now
        records = [r for r in caplog.records if getattr(r, "audit_event", "") == "reify_decay_sweep"]
        assert records
        assert any(old_fact.id in r.expired_memory_ids for r in records)
    finally:
        await engine.stop()


async def test_retention_fixture_workload_produces_expected_survivor_set() -> None:
    base_now = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    clock = MutableClock(base_now)
    engine = Reify(
        storage=SQLiteStore(":memory:"),
        clock=clock,
        config=EngineConfig(decay_sweep_interval_seconds=0),
    )
    await engine.start()
    try:
        def _retention(valid_at: datetime) -> dict[str, object]:
            return {
                "retention": {
                    "ttl_days": 30.0,
                    "expires_at": (valid_at + timedelta(days=30)).isoformat(),
                    "reinforced_count": 0,
                }
            }

        def _fact(index: int, *, cohort: str, valid_at: datetime) -> Fact:
            return Fact(
                content=f"Employer claim {cohort} #{index}",
                subject=f"user-{cohort}-{index}",
                predicate="works_at",
                obj=f"Company-{index}",
                identity_id=f"cohort:{cohort}",
                valid_at=valid_at,
                created_at=valid_at,
                metadata=_retention(valid_at),
            )

        stale_at = base_now
        mid_at = base_now + timedelta(days=20)
        late_at = base_now + timedelta(days=70)
        memories = [
            *[_fact(i, cohort="stale", valid_at=stale_at) for i in range(400)],
            *[_fact(i, cohort="reinforced", valid_at=stale_at) for i in range(300)],
            *[_fact(i, cohort="mid", valid_at=mid_at) for i in range(100)],
            *[_fact(i, cohort="late", valid_at=late_at) for i in range(200)],
        ]
        await engine._storage.commit_batch(WriteBatch(memories=memories))

        for day in (20, 40, 59, 74):
            clock.set(base_now + timedelta(days=day))
            reinforced = await engine.whatdoyouknow(
                identity_id="cohort:reinforced",
                limit=1000,
            )
            assert len(reinforced) == 300

        clock.set(base_now + timedelta(days=90))
        swept = await engine.sweep_decay(reason="fixture_survivor_set")
        assert swept == 0

        stale = await engine.whatdoyouknow(identity_id="cohort:stale", limit=1000)
        reinforced = await engine.whatdoyouknow(identity_id="cohort:reinforced", limit=1000)
        mid = await engine.whatdoyouknow(identity_id="cohort:mid", limit=1000)
        late = await engine.whatdoyouknow(identity_id="cohort:late", limit=1000)

        assert len(stale) == 0
        assert len(reinforced) == 300
        assert len(mid) == 0
        assert len(late) == 200

        all_rows = await engine._storage.recall(RecallFilter(include_deleted=True), limit=2000)
        live_rows = [m for m in all_rows if m.soft_deleted_at is None]
        deleted_rows = [m for m in all_rows if m.soft_deleted_at is not None]
        assert len(all_rows) == 1000
        assert len(live_rows) == 500
        assert len(deleted_rows) == 500
    finally:
        await engine.stop()


# ---------------------------------------------------------------------------
# export_markdown()
# ---------------------------------------------------------------------------


async def test_export_markdown_writes_human_readable_file(
    engine: Reify, tmp_path: Path
) -> None:
    await engine.remember("I just got promoted from Senior to Staff Engineer.")
    out = tmp_path / "export.md"
    count = await engine.export_markdown(out)
    assert count >= 2
    text = out.read_text("utf-8")
    assert "# Reify export" in text
    assert "Staff Engineer" in text
    assert "## Memories" in text


async def test_export_markdown_creates_parent_dirs(
    engine: Reify, tmp_path: Path
) -> None:
    await engine.remember("I'm a Staff Engineer.")
    out = tmp_path / "deep" / "nested" / "export.md"
    count = await engine.export_markdown(out)
    assert count >= 1
    assert out.exists()


# ---------------------------------------------------------------------------
# flush_arw()
# ---------------------------------------------------------------------------


async def test_flush_arw_preserves_queue_in_no_llm_mode(engine: Reify) -> None:
    await engine.remember("I'm a Staff Engineer.")
    # No configured LLM means the durable queue remains intact for a later worker.
    assert await engine.flush_arw() == 0
    assert await engine.flush_arw() == 0
