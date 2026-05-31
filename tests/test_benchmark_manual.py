"""Manual benchmark test for BEAM harness and native scorer.

This script runs a simple benchmark to verify the evaluation framework works.
It creates test cases, runs them through the engine, and scores them.

Usage:
    python tests/test_benchmark_manual.py
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime, timezone

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reify import Reify, EngineConfig
from reify.storage import SQLiteStore
from reify.eval.beam import BEAMCase, run_beam_suite
from reify.eval.native_scorer import NativeScorer, NativeScorerConfig


async def test_benchmark():
    """Test the BEAM harness and native scorer."""
    print("Running benchmark test...")

    # Create temporary database
    db_path = "/tmp/test_benchmark.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Add some test memories
        await engine.remember("I work at Stripe as a Staff Engineer.")
        await engine.remember("I live in San Francisco with my dog Mochi.")
        await engine.remember("I prefer TypeScript over JavaScript for new projects.")

        # Create BEAM test cases
        beam_cases = [
            BEAMCase(
                query="Where do I work?",
                expected_substrings=["Stripe", "Staff Engineer"],
            ),
            BEAMCase(
                query="What pet do I have?",
                expected_substrings=["dog", "Mochi"],
            ),
            BEAMCase(
                query="What programming language do I prefer?",
                expected_substrings=["TypeScript"],
            ),
        ]

        # Run BEAM suite
        print("\nRunning BEAM suite...")
        report = await run_beam_suite(engine=engine, cases=beam_cases)
        print(f"BEAM Report:")
        print(f"  Total cases: {report.total_cases}")
        print(f"  Accuracy: {report.overall_accuracy:.2%}")
        print(f"  P50 latency: {report.p50_latency_ms:.2f}ms")
        print(f"  P95 latency: {report.p95_latency_ms:.2f}ms")

        # Run native scorer
        print("\nRunning native scorer...")
        scorer = NativeScorer(NativeScorerConfig())
        for case in beam_cases:
            result = await scorer.score_case(engine, case.query)
            print(f"Query: {case.query}")
            print(f"  Continuity: {result.continuity_score:.2f}")
            print(f"  Abstention: {result.abstention_score:.2f}")
            print(f"  Provenance: {result.provenance_score:.2f}")
            print(f"  Update truth: {result.update_truth_score:.2f}")
            print(f"  Overall: {result.overall_score:.2f}")

        print("\n✓ Benchmark test completed successfully")
        return True


if __name__ == "__main__":
    result = asyncio.run(test_benchmark())
    sys.exit(0 if result else 1)
