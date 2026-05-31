"""Tests for the v0.3 token-budget context packer.

Covers profile selection, budget enforcement, format differentiation,
truncation legibility, and the engine's ``recall_packed`` surface.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from reify import EngineConfig, Reify
from reify.core.types import Fact
from reify.recall import RecallResult
from reify.storage import SQLiteStore
from reify.synthesize import (
    CLAUDE_PROFILE,
    GEMINI_PROFILE,
    GPT_PROFILE,
    LOCAL_PROFILE,
    ModelProfile,
    PackedContext,
    get_profile,
    pack,
)


def _utc(y: int, m: int, d: int) -> datetime:
    return datetime(y, m, d, tzinfo=timezone.utc)


def _result(content: str, *, score: float = 0.5, stage: str = "lexical") -> RecallResult:
    return RecallResult(
        memory=Fact(
            content=content,
            subject="s",
            predicate="p",
            obj="o",
            valid_at=_utc(2025, 1, 1),
        ),
        score=score,
        matched_stage=stage,
    )


# ---------------------------------------------------------------------------
# Profile lookup
# ---------------------------------------------------------------------------


def test_get_profile_returns_builtins() -> None:
    assert get_profile("claude") is CLAUDE_PROFILE
    assert get_profile("gpt") is GPT_PROFILE
    assert get_profile("gemini") is GEMINI_PROFILE
    assert get_profile("local") is LOCAL_PROFILE


def test_get_profile_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_profile("not-a-real-profile")


# ---------------------------------------------------------------------------
# pack() basics
# ---------------------------------------------------------------------------


def test_pack_includes_all_when_budget_is_generous() -> None:
    results = [_result(f"memory {i}") for i in range(3)]
    out = pack(results, profile="gpt", budget_tokens=10_000)
    assert isinstance(out, PackedContext)
    assert out.included_count == 3
    assert out.dropped_ids == ()
    assert out.truncated is False
    for r in results:
        assert r.memory.id in out.included_ids


def test_pack_drops_overflow_keeping_highest_ranked() -> None:
    """Items are listed highest-score first; the packer iterates in that
    order and drops from where the budget runs out, so the most relevant
    memories always survive."""
    results = [_result(f"x" * 200, score=1.0 - i * 0.1) for i in range(5)]
    out = pack(results, profile="gpt", budget_tokens=80)
    assert out.included_count >= 1
    assert out.included_count + len(out.dropped_ids) == 5
    assert out.truncated is True
    # The highest-scoring memory must be in the included set.
    assert results[0].memory.id in out.included_ids


def test_pack_zero_budget_drops_all() -> None:
    results = [_result("anything")]
    out = pack(results, profile="gpt", budget_tokens=0)
    assert out.included_count == 0
    assert out.dropped_ids == (results[0].memory.id,)
    assert out.truncated is True
    assert out.text == ""


def test_pack_empty_results() -> None:
    out = pack([], profile="gpt", budget_tokens=1000)
    assert out.included_count == 0
    assert out.dropped_ids == ()
    assert out.truncated is False


# ---------------------------------------------------------------------------
# Format differentiation
# ---------------------------------------------------------------------------


def test_claude_profile_uses_xml_tags() -> None:
    out = pack([_result("hi")], profile="claude")
    assert "<memories>" in out.text
    assert "<memory" in out.text
    assert "</memory>" in out.text
    assert "</memories>" in out.text


def test_gpt_profile_uses_delimiters() -> None:
    out = pack([_result("a"), _result("b")], profile="gpt")
    assert "Reify memories" in out.text
    assert "[1]" in out.text
    assert "---" in out.text  # item separator


def test_gemini_profile_uses_markdown_headings() -> None:
    out = pack([_result("a")], profile="gemini")
    assert "## Memory" in out.text


def test_local_profile_is_minimal() -> None:
    out = pack([_result("a")], profile="local")
    # Minimal format: bullet list, no XML, no markdown headings.
    assert "<memory" not in out.text
    assert "## Memory" not in out.text
    assert "- (lexical) a" in out.text


# ---------------------------------------------------------------------------
# Custom profile override
# ---------------------------------------------------------------------------


def test_pack_accepts_custom_profile() -> None:
    custom = ModelProfile(
        name="custom",
        max_tokens=500,
        format="markdown",
        header="<<HEAD>>\n",
        footer="\n<<FOOT>>",
        item_separator="\n~\n",
    )
    out = pack([_result("a"), _result("b")], profile=custom)
    assert out.text.startswith("<<HEAD>>")
    assert out.text.endswith("<<FOOT>>")
    assert "~" in out.text


# ---------------------------------------------------------------------------
# Engine.recall_packed integration
# ---------------------------------------------------------------------------


async def test_engine_recall_packed_returns_packed_context() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        await engine.remember("Arun lives in San Francisco.", at=_utc(2025, 1, 1))
        out = await engine.recall_packed("San Francisco", profile="claude")
        assert isinstance(out, PackedContext)
        assert "<memories>" in out.text
        assert out.included_count >= 1


async def test_engine_recall_packed_respects_budget() -> None:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        for i in range(5):
            await engine.remember(
                f"Arun event number {i} long content " * 5,
                at=_utc(2025, 1, i + 1),
            )
        out = await engine.recall_packed(
            "Arun", profile="local", budget_tokens=40
        )
        # Tight budget => truncation must be visible.
        assert out.truncated is True
        assert out.included_count + len(out.dropped_ids) == out.total_count
