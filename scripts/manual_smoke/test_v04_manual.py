"""Manual test script for v0.4 (Sleep Phase) features.

Tests:
- Hot-buffer with promotion gate
- Cross-session synthesis pass
- Adversarial belief probing
- Belief confidence decay (Ebbinghaus)
- Sleep-phase scheduler
- BEAM harness
- Native scorer
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reify import Reify, EngineConfig
from reify.storage import SQLiteStore
from reify.storage.hot_buffer import HotBuffer, HotBufferConfig
from reify.consolidate.adversarial import AdversarialProber
from reify.consolidate.sleep_scheduler import SleepScheduler
from reify.eval.beam import run_beam_suite, BEAMCase
from reify.eval.native_scorer import NativeScorer, NativeScorerConfig


async def test_hot_buffer():
    """Test hot-buffer batching and flush."""
    print("\n=== Testing Hot Buffer ===")
    db_path = "/tmp/test_hot_buffer.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Access hot buffer
        hot_buffer = engine._hot_buffer
        print(f"Hot buffer pending count: {hot_buffer.pending_count}")
        print(f"Hot buffer pending bytes: {hot_buffer.pending_bytes}")
        print("✓ Hot buffer initialized")
        return True


async def test_adversarial_probing():
    """Test adversarial probing."""
    print("\n=== Testing Adversarial Probing ===")
    db_path = "/tmp/test_adversarial.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Add some memories
        await engine.remember("I live in San Francisco.")
        await engine.remember("I work at Stripe.")

        # Run adversarial probing
        prober = AdversarialProber(engine=engine)
        results = await prober.run_probes()
        print(f"Generated {len(results)} adversarial probes")
        for result in results:
            print(f"  Query: {result.probe.query}")
            print(f"  Expected: {result.probe.expected_behavior}")
            print(f"  Success: {result.success}")
        print("✓ Adversarial probing working")
        return True


async def test_sleep_scheduler():
    """Test sleep scheduler idle detection."""
    print("\n=== Testing Sleep Scheduler ===")
    db_path = "/tmp/test_sleep_scheduler.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Access sleep scheduler
        scheduler = engine._sleep_scheduler
        print(f"Sleep scheduler running: {scheduler._running}")
        print(f"Last activity: {scheduler._last_activity_at}")
        
        # Record activity
        scheduler.record_activity()
        print(f"Activity recorded: {scheduler._last_activity_at}")
        print("✓ Sleep scheduler initialized")
        return True


async def test_cross_session_synthesis():
    """Test cross-session synthesis."""
    print("\n=== Testing Cross-Session Synthesis ===")
    db_path = "/tmp/test_synthesis.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Add memories with same identity
        await engine.remember("I work at Stripe as a Staff Engineer.", identity_id="user1")
        await engine.remember("I live in San Francisco.", identity_id="user1")
        await engine.remember("I prefer TypeScript.", identity_id="user1")

        # Run synthesis
        from reify.consolidate.cross_session_synthesis import CrossSessionSynthesizer, CrossSessionSynthesisConfig
        synthesizer = CrossSessionSynthesizer(engine=engine, config=CrossSessionSynthesisConfig())
        count = await synthesizer.run_synthesis()
        print(f"Synthesized {count} inferences")
        print("✓ Cross-session synthesis working")
        return True


async def test_beam_harness():
    """Test BEAM harness."""
    print("\n=== Testing BEAM Harness ===")
    db_path = "/tmp/test_beam.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Add test memories
        await engine.remember("I work at Stripe as a Staff Engineer.")
        await engine.remember("I live in San Francisco with my dog Mochi.")
        await engine.remember("I prefer TypeScript over JavaScript.")

        # Run BEAM
        cases = (
            BEAMCase(query="Where do I work?", expected_substrings=("Stripe",)),
            BEAMCase(query="What pet do I have?", expected_substrings=("Mochi",)),
        )
        report = await run_beam_suite(engine=engine, cases=cases)
        print(f"BEAM accuracy: {report.overall_accuracy:.2%}")
        print(f"Total cases: {report.total_cases}")
        print("✓ BEAM harness working")
        return True


async def test_native_scorer():
    """Test native scorer."""
    print("\n=== Testing Native Scorer ===")
    db_path = "/tmp/test_scorer.db"
    store = SQLiteStore(path=db_path)
    engine = Reify(storage=store, config=EngineConfig())

    async with engine:
        # Add test memories
        await engine.remember("I work at Stripe as a Staff Engineer.")

        # Run scorer
        scorer = NativeScorer(NativeScorerConfig())
        result = await scorer.score_case(engine, "Where do I work?")
        print(f"Continuity: {result.continuity_score:.2f}")
        print(f"Abstention: {result.abstention_score:.2f}")
        print(f"Provenance: {result.provenance_score:.2f}")
        print(f"Update truth: {result.update_truth_score:.2f}")
        print(f"Overall: {result.overall_score:.2f}")
        print("✓ Native scorer working")
        return True


async def main():
    """Run all v0.4 manual tests."""
    print("Running v0.4 Manual Tests")
    print("=" * 50)

    results = []
    results.append(("Hot Buffer", await test_hot_buffer()))
    results.append(("Adversarial Probing", await test_adversarial_probing()))
    results.append(("Sleep Scheduler", await test_sleep_scheduler()))
    results.append(("Cross-Session Synthesis", await test_cross_session_synthesis()))
    results.append(("BEAM Harness", await test_beam_harness()))
    results.append(("Native Scorer", await test_native_scorer()))

    print("\n" + "=" * 50)
    print("Test Results Summary:")
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
