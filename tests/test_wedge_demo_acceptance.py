"""Wedge demo acceptance test - CLI-runnable, MCP-host-agnostic.

This test exercises the four-step wedge demo using the engine directly with
controllable timestamps, simulating the CLI workflow:

Step 1: Ingest promotion (deterministic extraction)
Step 2: Query title (bitemporal recall)
Step 3: Query changes (BTG - will be empty until ARW is implemented)
Step 4: Cross-session inference (ARW - will be empty until ARW is implemented)

The test is host-agnostic and can run on any machine without any MCP host
installed. It uses the engine's injectable clock to compress the 60-day story
into seconds.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from reify import Reify, EngineConfig
from reify.core.types import Inference, RelationType, TransitionKind
from reify.llm.base import LLMRequest, LLMResponse
from reify.storage import SQLiteStore


def _utc_date(year: int, month: int, day: int) -> datetime:
    """Helper to create UTC datetime at midnight."""
    return datetime(year, month, day, tzinfo=timezone.utc)


class FakeLLMProvider:
    name = "fake"
    default_model = "fake-model"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        prompt = request.messages[-1]["content"]
        if "Title: Staff Engineer" in prompt and "Title: Senior" in prompt:
            payload = [
                {
                    "type": "resolves",
                    "confidence": 0.95,
                    "explanation": "new title supersedes old title",
                }
            ]
        elif "Title: Principal Engineer" in prompt and "Title: Staff" in prompt:
            payload = [
                {
                    "type": "resolves",
                    "confidence": 0.96,
                    "explanation": "new title supersedes prior title",
                }
            ]
        else:
            payload = [{"type": "none", "confidence": 0.0, "explanation": ""}]
        return LLMResponse(content=json.dumps(payload), model=self.default_model)


@pytest.mark.asyncio
async def test_wedge_demo_acceptance() -> None:
    """Full wedge demo acceptance test using engine with controllable clock.

    This is the v0.1 acceptance bar. When this passes, the wedge demo works
    end-to-end via CLI (and by extension, via any MCP host).
    """
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "store.db"
        store = SQLiteStore(path=db_path)

        # Create engine with frozen clock at Day 1
        day1 = _utc_date(2025, 1, 1)
        clock = lambda: day1
        engine = Reify(
            storage=store,
            config=EngineConfig(llm=FakeLLMProvider()),
            clock=clock,
        )

        async with engine:
            # ------------------------------------------------------------------
            # Step 1: Day 1 - Ingest promotion
            # ------------------------------------------------------------------
            # User says: "I got promoted from Senior to Staff Engineer."
            # Expected: Experience + Title (Staff) + Employer (if mentioned) + Title (Senior, valid_until=now)
            receipt = await engine.remember(
                "I got promoted from Senior to Staff Engineer at Acme Corp",
                source="cli:remember",
                identity_id="user123",
            )
            assert len(receipt.committed) >= 2  # At least Experience + new title
            assert any(m.network == "experience" for m in receipt.committed)
            # Should have both old (Senior) and new (Staff) title facts
            title_facts = [m for m in receipt.committed if m.network == "fact" and "Title" in m.content]
            assert len(title_facts) == 2
            # Old title should have valid_until set
            old_title = [m for m in title_facts if "Senior" in m.content][0]
            new_title = [m for m in title_facts if "Staff" in m.content][0]
            assert old_title.valid_until == day1
            assert new_title.valid_until is None
            processed = await engine.flush_arw()
            assert processed == 1

        # ------------------------------------------------------------------
        # Step 2: Day 30 - Query title
        # ------------------------------------------------------------------
        # Advance clock to Day 30
        day30 = _utc_date(2025, 1, 30)
        clock2 = lambda: day30
        engine2 = Reify(
            storage=store,
            config=EngineConfig(llm=FakeLLMProvider()),
            clock=clock2,
        )

        async with engine2:
            # User asks: "What's my title?"
            # Expected: "Staff Engineer" (not Senior, not both)
            response = await engine2.recall("what is my title?", identity_id="user123")
            assert len(response.results) > 0
            # Find current title facts (valid_at <= day30 and valid_until is None or > day30)
            current_titles = [
                r for r in response.results
                if r.memory.network == "fact"
                and "Title" in r.memory.content
                and r.memory.valid_at <= day30
                and (r.memory.valid_until is None or r.memory.valid_until > day30)
            ]
            assert len(current_titles) == 1
            assert "Staff Engineer" in current_titles[0].memory.content

        # ------------------------------------------------------------------
        # Step 3: Day 30 - Query changes
        # ------------------------------------------------------------------
        async with engine2:
            # User asks: "Did anything change recently?"
            # Expected: Transition record materialized by ARW
            transitions = await engine2.whatchanged(identity_id="user123", days=30)
            assert len(transitions) == 1
            assert transitions[0].kind is TransitionKind.ROLE_CHANGE
            assert "Senior to Staff Engineer" in transitions[0].summary

        # ------------------------------------------------------------------
        # Step 4: Day 60 - Cross-session inference
        # ------------------------------------------------------------------
        # Advance clock to Day 60 and add second promotion
        day60 = _utc_date(2025, 3, 1)  # ~60 days later
        clock3 = lambda: day60
        engine3 = Reify(
            storage=store,
            config=EngineConfig(llm=FakeLLMProvider()),
            clock=clock3,
        )

        async with engine3:
            # User says: "I got promoted from Staff to Principal."
            receipt2 = await engine3.remember(
                "I got promoted from Staff to Principal Engineer",
                source="cli:remember",
                identity_id="user123",
            )
            assert len(receipt2.committed) >= 2
            processed = await engine3.flush_arw()
            assert processed == 1

        # Advance clock to Day 90 and query for pattern
        day90 = _utc_date(2025, 5, 1)
        clock4 = lambda: day90
        engine4 = Reify(
            storage=store,
            config=EngineConfig(llm=FakeLLMProvider()),
            clock=clock4,
        )

        async with engine4:
            # User asks: "What do I usually emphasize in promotion writeups?"
            memories = await engine4.whatdoyouknow("promotion", identity_id="user123")
            assert len(memories) > 0
            inferences = [m for m in memories if isinstance(m, Inference)]
            assert len(inferences) >= 2
            assert any(m.relation is RelationType.RESOLVES for m in inferences)


@pytest.mark.asyncio
async def test_wedge_demo_cli_timestamp_simulation() -> None:
    """Test that --at flag works for timestamp injection.

    This verifies that the CLI can simulate the wedge demo without
    actually waiting 60 days.
    """
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "store.db"
        store = SQLiteStore(path=db_path)

        # Simulate CLI: reify remember "..." --at 2025-01-01
        day1 = _utc_date(2025, 1, 1)
        engine = Reify(storage=store, config=EngineConfig(), clock=lambda: day1)

        async with engine:
            await engine.remember(
                "I got promoted from Senior to Staff Engineer",
                source="cli:remember",
                at=day1,
            )

        # Simulate CLI: reify recall "title" --at 2025-01-30
        day30 = _utc_date(2025, 1, 30)
        engine2 = Reify(storage=store, config=EngineConfig(), clock=lambda: day30)

        async with engine2:
            response = await engine2.recall("title", valid_at=day30)
            assert len(response.results) > 0
            # Should find Staff Engineer as current title
            titles = [r for r in response.results if "Staff" in r.memory.content]
            assert len(titles) > 0
