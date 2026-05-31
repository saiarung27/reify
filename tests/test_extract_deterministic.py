"""Tests for :mod:`reify.extract.deterministic`.

These tests pin down the precision contract of the deterministic extractor.
Recall is the LLM stage's job; here we care that:

1. Every recognized pattern produces the right typed memory shape.
2. False positives stay out (negative cases).
3. Bitemporal bounds are correct: prior facts get ``valid_until = now`` so
   recall queries with ``as_of`` past the change return the new fact only.
4. The raw utterance is always recorded as an :class:`Experience`.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from reify.core.types import Experience, Fact, Network
from reify.extract import DeterministicExtractor, ExtractionContext


NOW = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)


def _ctx(**overrides: object) -> ExtractionContext:
    base: dict[str, object] = {"source": "windsurf", "actor": "user", "now": NOW}
    base.update(overrides)
    return ExtractionContext(**base)  # type: ignore[arg-type]


@pytest.fixture
def extractor() -> DeterministicExtractor:
    return DeterministicExtractor()


# ---------------------------------------------------------------------------
# Always-an-Experience invariant
# ---------------------------------------------------------------------------


async def test_empty_input_emits_nothing(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract("   ", _ctx())
    assert result.memories == ()


async def test_unrecognized_input_still_emits_experience(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract("Hi there, just saying hello.", _ctx())
    networks = [m.network for m in result.memories]
    assert networks == [Network.EXPERIENCE]
    exp = result.memories[0]
    assert isinstance(exp, Experience)
    assert exp.content == "Hi there, just saying hello."
    assert exp.actor == "user"
    assert exp.event_type == "user_message"


async def test_actor_is_propagated_to_event_type(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract("Acknowledged.", _ctx(actor="assistant"))
    exp = result.memories[0]
    assert isinstance(exp, Experience)
    assert exp.actor == "assistant"
    assert exp.event_type == "assistant_message"


async def test_assistant_turns_do_not_emit_user_claims(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract("I live in Seattle.", _ctx(actor="assistant"))
    experiences = [m for m in result.memories if isinstance(m, Experience)]
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(experiences) == 1
    assert facts == []


async def test_assistant_turns_emit_structured_reference_items(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "1. Miss Bee Providore: cozy and stylish. 2. Bakery & Co: coffee and pastry.",
        _ctx(actor="assistant"),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert any(f.obj == "Miss Bee Providore: cozy and stylish" for f in facts)
    assert any(f.metadata["assistant_reference"]["item_index"] == "1" for f in facts)
    assert any(f.obj == "Miss Bee Providore" and "assistant-name" in f.tags for f in facts)


async def test_assistant_turns_emit_sentence_reference_notes(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "One of the best hiking trails with breathtaking views in the Natural Park of the Moncayo mountain in Aragón is the GR-90.",
        _ctx(actor="assistant"),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert any("GR-90" in f.obj for f in facts)


async def test_assistant_turns_emit_typed_option_metadata(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "1. N'EX (Narita Express) Train: This is the fastest way to get to Tokyo Station ( approx. 60 minutes, ¥3,000-4,000).\n"
        "4. Taxi: Taxis are available, but this is the most expensive option (approx. 1-2 hours, ¥15,000-20,000).",
        _ctx(actor="assistant"),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    option_facts = [f for f in facts if "assistant-option" in f.tags]
    assert any(
        f.metadata["assistant_reference"].get("transport_mode") == "train"
        and f.metadata["assistant_reference"].get("price_min") == 3000
        and f.metadata["assistant_reference"].get("price_max") == 4000
        for f in option_facts
    )
    assert any(
        f.metadata["assistant_reference"].get("transport_mode") == "taxi"
        and f.metadata["assistant_reference"].get("price_min") == 15000
        and f.metadata["assistant_reference"].get("price_max") == 20000
        for f in option_facts
    )


async def test_extracts_age_and_country_move_event(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I'm 32-year-old and I've been living in the United States for the past five years on a work visa.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert any(f.predicate == "has_age_years" and f.obj == "32" for f in facts)
    assert any(f.predicate == "experienced_event" and f.obj == "moved to United States" for f in facts)


async def test_extracts_streaming_service_start_events(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I've been using Netflix, Hulu, and Amazon Prime for the past 6 months. I've also been using Apple TV+ for a few months now. Since I started a free trial of Disney+ last month, I've been watching more documentaries.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    labels = {f.obj for f in facts if f.predicate == "experienced_event"}
    assert {"Netflix", "Hulu", "Amazon Prime", "Apple TV+", "Disney+"} <= labels


async def test_extracts_audience_reach_and_finished_book_pages(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I recently collaborated with an influencer who promoted my product to her 10,000 followers. "
        "I just finished reading \"The Nightingale\" by Kristin Hannah, which had 440 pages.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert any(f.predicate == "reached_audience" and f.metadata["audience"]["count_value"] == 10000 for f in facts)
    assert any(f.predicate == "finished_book" and f.metadata["reading"]["page_count"] == 440 for f in facts)


async def test_extracts_sibling_counts(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I have a brother, and I come from a family with 3 sisters.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    sibling_counts = {f.obj: f.metadata["family"]["count_value"] for f in facts if f.predicate == "has_siblings"}
    assert sibling_counts == {"brothers": 1, "sisters": 3}


async def test_extracts_deal_quotes_and_guitar_lesson_timing(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I've had luck finding great deals at TK Maxx before, like that designer handbag I got for $200. "
        "It was originally $500! "
        "I've been taking weekly guitar lessons with a new instructor, Alex, for six weeks now. "
        "I recently bought a new guitar amp two weeks ago.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert any(f.metadata.get("deal", {}).get("kind") == "paid" and f.metadata["deal"]["amount_value"] == 200 for f in facts)
    assert any(f.metadata.get("deal", {}).get("kind") == "original" and f.metadata["deal"]["amount_value"] == 500 for f in facts)
    assert any(f.predicate == "experienced_event" and f.obj == "taking guitar lessons" for f in facts)
    assert any(f.predicate == "experienced_event" and f.obj == "bought the new guitar amp" for f in facts)


async def test_provenance_carries_source_and_session(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "Just chatting.", _ctx(source="windsurf", session_id="sess-1")
    )
    exp = result.memories[0]
    assert any(p.session_id == "sess-1" for p in exp.provenance)
    assert any("windsurf" in p.source for p in exp.provenance)


# ---------------------------------------------------------------------------
# Promotion (the wedge demo's bread and butter)
# ---------------------------------------------------------------------------


async def test_promotion_from_to_emits_old_and_new_titles(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I just got promoted from Senior to Staff Engineer.", _ctx()
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 2

    by_obj = {f.obj: f for f in facts}
    assert "Senior" in by_obj
    assert "Staff Engineer" in by_obj

    old = by_obj["Senior"]
    new = by_obj["Staff Engineer"]

    # The old title's validity is bounded at NOW; the new title's is open.
    assert old.valid_until == NOW
    assert new.valid_until is None

    # Both share the predicate.
    assert old.predicate == "has_title"
    assert new.predicate == "has_title"

    # Tags distinguish former vs current.
    assert "former" in old.tags
    assert "current" in new.tags
    assert "promotion" in new.tags


async def test_promotion_to_only_emits_single_title(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract("I was promoted to Principal Engineer.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 1
    assert facts[0].obj == "Principal Engineer"
    assert facts[0].valid_until is None


async def test_promotion_just_got_variant(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract(
        "I just got promoted from Engineer to Senior Engineer.", _ctx()
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 2


async def test_promotion_does_not_double_fire_with_title_pattern(
    extractor: DeterministicExtractor,
) -> None:
    # If a promotion is detected, the title-statement pattern must not also
    # fire (would create duplicates).
    result = await extractor.extract(
        "I'm a Staff Engineer because I got promoted from Senior to Staff.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    # Exactly two: old + new from the promotion pattern. NOT three.
    assert len(facts) == 2


# ---------------------------------------------------------------------------
# Title statements
# ---------------------------------------------------------------------------


async def test_title_statement_im_a(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract("I'm a Senior Engineer.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 1
    assert facts[0].obj == "Senior Engineer"
    assert facts[0].predicate == "has_title"


async def test_title_statement_my_title_is(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract("My title is Staff Engineer.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 1
    assert facts[0].obj == "Staff Engineer"


async def test_title_statement_negative_non_role(
    extractor: DeterministicExtractor,
) -> None:
    # "I am a person who likes dogs" must NOT extract "person" as a title.
    result = await extractor.extract("I am a person who likes dogs.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert facts == []


# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------


async def test_location_move_from_to(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract(
        "I moved from New York to San Francisco.", _ctx()
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 2
    by_obj = {f.obj: f for f in facts}
    assert "New York" in by_obj
    assert "San Francisco" in by_obj
    assert by_obj["New York"].valid_until == NOW
    assert by_obj["San Francisco"].valid_until is None


async def test_location_move_to_only(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract("I just moved to Austin.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 1
    assert facts[0].obj == "Austin"
    assert facts[0].predicate == "lives_in"


async def test_location_live_in(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract("I live in Seattle.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 1
    assert facts[0].obj == "Seattle"


async def test_location_live_in_stops_before_second_clause(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract("I live in Seattle and I work at Stripe.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    location = next(f for f in facts if f.predicate == "lives_in")
    assert location.obj == "Seattle"


async def test_location_negative_lowercase(extractor: DeterministicExtractor) -> None:
    # Locations require capitalization to avoid matching common-noun phrases.
    result = await extractor.extract("I live in the moment.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert facts == []


# ---------------------------------------------------------------------------
# Employer
# ---------------------------------------------------------------------------


async def test_employer_work_at(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract("I work at Stripe.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 1
    assert facts[0].obj == "Stripe"
    assert facts[0].predicate == "works_at"


async def test_employer_joined(extractor: DeterministicExtractor) -> None:
    result = await extractor.extract("I just joined Anthropic.", _ctx())
    facts = [m for m in result.memories if isinstance(m, Fact)]
    assert len(facts) == 1
    assert facts[0].obj == "Anthropic"
    assert "joined" in facts[0].tags


async def test_event_workshop_with_date_extracts_normalized_event(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        'I attended the workshop on "Effective Communication in the Workplace" on January 10th.',
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(f for f in facts if f.predicate == "experienced_event")
    assert event.obj == "Effective Communication in the Workplace"
    assert "event" in event.tags
    assert event.metadata["event"]["kind"] == "workshop"
    assert event.valid_at.month == 1
    assert event.valid_at.day == 10


async def test_preference_extraction_emits_preference_fact(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I'm leaning more towards the Kuat Transfer.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    pref = next(f for f in facts if f.predicate == "prefers")
    assert pref.obj == "the Kuat Transfer"
    assert "preference" in pref.tags


async def test_preference_extraction_enjoy_pattern_is_cleaned(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I enjoy to use Adobe Premiere Pro, especially for advanced color grading.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    pref = next(f for f in facts if f.predicate == "prefers")
    assert pref.obj == "Adobe Premiere Pro"
    assert pref.metadata["preference"]["source_pattern"] == "enjoy"


async def test_device_named_pattern_extracts_clean_device_label(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I need adapters for my new laptop, Dell XPS 13, before my trip.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(f for f in facts if f.predicate == "experienced_event")
    assert event.obj == "Dell XPS 13"


async def test_study_abroad_extraction_emits_school_fact(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I actually went there with some friends during my study abroad program at the University of Melbourne.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    education = next(f for f in facts if "education" in f.tags)
    assert education.obj == "University of Melbourne"


async def test_inventory_plant_reference_got_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I'm also wondering if I should repot my snake plant, which I got from my sister last month.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    plant = next(f for f in facts if "inventory" in f.tags)
    assert plant.obj == "snake plant"
    assert plant.metadata["inventory"]["kind"] == "plant"


async def test_luxury_evening_gown_cross_sentence_amount_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I recently bought a luxury evening gown for a wedding. It was a big purchase, $800, but I felt like I needed to make a good impression.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    luxury = next(f for f in facts if "luxury" in f.tags)
    assert luxury.metadata["expense"]["amount_value"] == 800.0


async def test_game_duration_without_it_prefix_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I'm trying to find some new indie games similar to Celeste, which took me 10 hours to complete.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    game = next(f for f in facts if "gaming" in f.tags)
    assert game.obj == "Celeste"
    assert game.metadata["gaming"]["duration_value"] == 10


async def test_generic_event_slash_date_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I'd like to ask Rachel about the new construction listings in these areas. Since I started working with her on 2/15, I'm hoping she can give me a better sense of what's available in my budget.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(f for f in facts if f.predicate == "experienced_event")
    assert "started working with Rachel" in event.obj
    assert event.valid_at.month == 2
    assert event.valid_at.day == 15


async def test_purchase_event_with_relative_date_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I bought a laptop backpack from Amazon on 1/15.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(f for f in facts if f.metadata.get("event", {}).get("kind") == "purchase")
    assert event.obj == "laptop backpack"
    assert event.valid_at.month == 1
    assert event.valid_at.day == 15


async def test_setup_event_with_relative_weeks_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "Since I set up my smart thermostat a month ago, my energy bills have dropped.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(f for f in facts if f.metadata.get("event", {}).get("kind") == "setup")
    assert event.obj == "smart thermostat"
    assert event.valid_at.month == 12
    assert event.valid_at.day == 16


async def test_wake_up_time_fact_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "On Fridays, I like to get a head start, so I wake up at 6:00 AM.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    wake = next(f for f in facts if f.predicate == "wakes_up_at")
    assert wake.obj == "6:00 AM"
    assert wake.metadata["wake_routine"]["weekdays"] == ["friday"]


async def test_wake_up_offset_fact_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "On Tuesdays and Thursdays, I've also started waking up 15 minutes earlier to meditate.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    wake = next(f for f in facts if f.predicate == "wakes_up_at")
    assert wake.metadata["wake_routine"]["offset_minutes"] == -15
    assert wake.metadata["wake_routine"]["weekdays"] == ["tuesday", "thursday"]


async def test_church_service_date_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I just got back from Sunday mass at St. Mary's Church on March 19th, and Father John's sermon really inspired me.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(f for f in facts if f.metadata.get("event", {}).get("kind") == "service")
    assert event.obj == "Sunday mass at St. Mary's Church"
    assert event.valid_at.month == 3
    assert event.valid_at.day == 19


async def test_stand_mixer_repair_event_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "Do you have any recommendations for cakes or cookies that don't require a stand mixer, just in case mine breaks down again? By the way, I had to take mine to a repair shop last month and it took two weeks to fix.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(f for f in facts if f.metadata.get("event", {}).get("kind") == "repair")
    assert event.obj == "stand mixer"
    assert event.valid_at.month == 12
    assert event.valid_at.day == 16


async def test_training_pads_context_purchase_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I've been using those eco-friendly training pads from Chewy.com. I got a set of 10 for $25 about a month ago, and they've been a lifesaver.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("kind") == "purchase"
        and f.obj == "eco-friendly training pads"
    )
    assert event.valid_at.month == 12
    assert event.valid_at.day == 16


async def test_coffee_maker_context_purchase_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I'm having some issues with my coffee maker's performance lately. I bought it about three weeks ago, and I've been using it every morning since then.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("kind") == "purchase"
        and f.obj == "coffee maker"
    )
    assert event.valid_at.month == 12
    assert event.valid_at.day == 25


async def test_lens_purchase_after_label_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I'm especially interested in mastering my 50mm lens that I got a month ago.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("kind") == "purchase"
        and "lens" in f.obj
    )
    assert event.valid_at.month == 12
    assert event.valid_at.day == 16


async def test_pronoun_repair_with_qualified_month_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I've been having some issues with my bike lately - in mid-February, I had to take it in for repairs because the gears were acting up.",
        _ctx(now=datetime(2023, 3, 10, 12, 0, tzinfo=timezone.utc)),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("kind") == "repair" and f.obj == "bike"
    )
    assert event.valid_at.month == 2
    assert event.valid_at.day == 15


async def test_shoe_cleaning_event_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I cleaned my white Adidas sneakers last month, which I'd been meaning to do for weeks.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("label") == "cleaned white Adidas sneakers"
    )
    assert event.valid_at.month == 12
    assert event.valid_at.day == 16


async def test_shoelace_break_event_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I realized that one of the shoelaces on my old Converse sneakers had broken on January 24th, so I had to replace it.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("label") == "broken shoelace on old Converse sneakers"
    )
    assert event.valid_at.month == 1
    assert event.valid_at.day == 24


async def test_shoe_purchase_on_date_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I recently got a new pair of Adidas running shoes on January 10th, and I want to take good care of them.",
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("kind") == "purchase" and f.obj == "Adidas running shoes"
    )
    assert event.valid_at.month == 1
    assert event.valid_at.day == 10


async def test_charity_event_with_month_only_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        "I volunteered at the Walk for Wildlife event in June.",
        _ctx(now=datetime(2023, 11, 29, 12, 0, tzinfo=timezone.utc)),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    event = next(
        f for f in facts
        if f.metadata.get("event", {}).get("label") == "Walk for Wildlife event"
    )
    assert event.valid_at == datetime(2023, 6, 15, 12, 0, tzinfo=timezone.utc)
    assert "charity" in event.tags


async def test_media_duration_two_and_a_half_weeks_is_extracted(
    extractor: DeterministicExtractor,
) -> None:
    result = await extractor.extract(
        'I just finished "The Seven Husbands of Evelyn Hugo", which took me two and a half weeks to finish.',
        _ctx(),
    )
    facts = [m for m in result.memories if isinstance(m, Fact)]
    media = next(f for f in facts if "media" in f.tags)
    assert media.metadata["media"]["duration_value"] == 2.5


# ---------------------------------------------------------------------------
# Composition: the full wedge-demo opener
# ---------------------------------------------------------------------------


async def test_wedge_demo_opener(extractor: DeterministicExtractor) -> None:
    """Exact wording used in the wedge demo (FIRST_PRINCIPLES.md \u00a74)."""
    result = await extractor.extract(
        "I just got promoted from Senior to Staff Engineer.", _ctx()
    )

    # 1 Experience + 2 Facts (former + current title).
    assert len(result.memories) == 3

    networks = sorted(m.network.value for m in result.memories)
    assert networks == ["experience", "fact", "fact"]

    facts = [m for m in result.memories if isinstance(m, Fact)]
    objs = sorted(f.obj for f in facts)
    assert objs == ["Senior", "Staff Engineer"]

    # The current-title fact must be retrievable by the natural query
    # "what's my title?" \u2014 i.e., it must be tagged 'current' and have
    # an open-ended validity.
    current = next(f for f in facts if "current" in f.tags)
    assert current.obj == "Staff Engineer"
    assert current.valid_until is None
