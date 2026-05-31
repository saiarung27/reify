"""End-to-end wedge demo test via CLI-like engine usage.

This test runs the four-step wedge demo:
1. Promotion (remember)
2. Title query (recall)
3. Change query (what-changed)
4. Derived claim query (what-do-you-know or recall)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from reify import Reify, EngineConfig
from reify.llm.base import LLMRequest, LLMResponse
from reify.storage import SQLiteStore


class FakeLLMProvider:
    name = "fake"
    default_model = "fake-model"

    async def complete(self, request: LLMRequest) -> LLMResponse:
        del request
        return LLMResponse(
            content=json.dumps(
                [
                    {
                        "type": "resolves",
                        "confidence": 0.9,
                        "explanation": "new fact supersedes old fact",
                    }
                ]
            ),
            model=self.default_model,
        )


@pytest.mark.asyncio
async def test_wedge_demo_cli_style() -> None:
    """Run the wedge demo end-to-end using the engine directly (CLI style)."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "store.db"
        store = SQLiteStore(path=db_path)
        engine = Reify(storage=store, config=EngineConfig(llm=FakeLLMProvider()))

        async with engine:
            # Step 1: Promotion (remember)
            receipt1 = await engine.remember(
                "I was promoted from Senior Engineer to Staff Engineer at Acme Corp",
                source="cli:remember",
                identity_id="user123",
            )
            assert len(receipt1.committed) > 0
            # Should have: Experience, Title (Staff Engineer), Employer (Acme Corp)
            # Note: Transitions are created by ARW, not the deterministic extractor
            assert any(m.network == "experience" for m in receipt1.committed)
            assert any(m.network == "fact" for m in receipt1.committed)
            processed = await engine.flush_arw()
            assert processed == 1

            # Step 2: Title query (recall)
            response = await engine.recall("what is my title?", identity_id="user123")
            assert len(response.results) > 0
            # Should find the Staff Engineer title
            title_results = [r for r in response.results if r.memory.network == "fact"]
            assert len(title_results) > 0
            assert "Staff Engineer" in title_results[0].memory.content

            # Step 3: Change query (what-changed)
            transitions = await engine.whatchanged(identity_id="user123", days=30)
            assert len(transitions) == 1

            # Step 4: What do you know (knowledge summary)
            memories = await engine.whatdoyouknow(identity_id="user123")
            assert len(memories) > 0
            # Should have experience, facts
            networks = {m.network for m in memories}
            assert "experience" in networks
            assert "fact" in networks


@pytest.mark.asyncio
async def test_cli_export() -> None:
    """Test the export functionality."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "store.db"
        export_path = Path(tmpdir) / "export.md"

        store = SQLiteStore(path=db_path)
        engine = Reify(storage=store, config=EngineConfig())

        async with engine:
            # Add some memories
            await engine.remember("I work at Acme Corp as a Staff Engineer", source="cli:remember")
            await engine.remember("I live in San Francisco", source="cli:remember")

            # Export
            count = await engine.export_markdown(export_path)
            assert count > 0
            assert export_path.exists()

            content = export_path.read_text()
            assert "# Reify export" in content
            assert "Acme Corp" in content or "Staff Engineer" in content
