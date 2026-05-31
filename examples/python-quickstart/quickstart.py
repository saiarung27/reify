"""Reify Python quickstart.

Run this script after ``pip install -e ".[dev]"`` from the repo root, or
``pip install reify`` from PyPI once published.

It exercises the four Reify primitives end-to-end against an isolated
SQLite store and the deterministic extractor only - no LLM API key
required.

    python examples/python-quickstart/quickstart.py

Expected output: a four-step narrative that proves remember, recall,
what-changed, and what-do-you-know all work locally.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from reify import Reify
from reify.storage import SQLiteStore


def utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


async def main() -> None:
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)

        # Pin a deterministic clock so the demo is reproducible. In a real
        # app you would omit ``clock`` and use the wall clock.
        clock_value = utc(2025, 1, 1)

        def clock() -> datetime:
            return clock_value

        engine = Reify(storage=store, clock=clock)

        async with engine:
            user = "demo-user"

            print("== Day 1: remember ==")
            await engine.remember(
                "I just got promoted from Senior to Staff Engineer.",
                identity_id=user,
            )
            # ``flush_arw`` drains the durable Active Reconsolidation queue
            # (the engine equivalent of the ``reify reflect`` CLI command).
            await engine.flush_arw()
            print("  ingested promotion #1\n")

            print("== Day 30: what's my title? ==")
            clock_value = utc(2025, 1, 30)
            response = await engine.recall("What's my title?", identity_id=user)
            for r in response.results:
                print(f"  [{r.score:0.2f}] {r.memory.content}")
            print()

            print("== Day 90: second promotion ==")
            clock_value = utc(2025, 4, 1)
            await engine.remember(
                "I got promoted from Staff to Principal Engineer.",
                identity_id=user,
            )
            await engine.flush_arw()
            print("  ingested promotion #2\n")

            print("== Day 120: what's my title now? ==")
            clock_value = utc(2025, 4, 30)
            response = await engine.recall("What's my title?", identity_id=user)
            for r in response.results:
                print(f"  [{r.score:0.2f}] {r.memory.content}")
            print()

            print("== Day 120: full knowledge for this identity ==")
            memories = await engine.whatdoyouknow(identity_id=user)
            for m in memories:
                print(f"  {m.network}: {m.content}")

    print("\nDone. The wedge demo passed end-to-end with no LLM and no network.")


if __name__ == "__main__":
    asyncio.run(main())
