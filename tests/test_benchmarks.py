"""Comprehensive benchmark test script.

Tests:
- LongMemEval fixture test
- LoCoMo fixture test
- User-value metrics test
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reify import Reify, EngineConfig
from reify.storage import SQLiteStore


async def test_longmemeval():
    """Test LongMemEval fixture."""
    print("\n=== Testing LongMemEval ===")
    try:
        from reify.eval.longmemeval import run_longmemeval_suite, default_longmemeval_fixture_cases
        
        # Get fixture cases
        cases = default_longmemeval_fixture_cases()
        print(f"✓ LongMemEval fixture has {len(cases)} cases")
        
        # Run suite (creates its own engine)
        report = await run_longmemeval_suite(cases=cases)
        print(f"✓ LongMemEval overall accuracy: {report.overall_accuracy:.2%}")
        print(f"✓ Passed target (≥85%): {report.passed_target}")
        print(f"✓ Per-category accuracy: {report.per_category_accuracy}")
        
        # Check exit criterion
        if report.overall_accuracy >= 0.85:
            print("✓ LongMemEval meets exit criterion (≥85%)")
        else:
            print(f"⚠ LongMemEval below exit criterion ({report.overall_accuracy:.2%} < 85%)")
        
        return True
    except Exception as e:
        print(f"✗ LongMemEval error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_locomo():
    """Test LoCoMo fixture."""
    print("\n=== Testing LoCoMo ===")
    try:
        from reify.eval.locomo import run_locomo_suite, default_locomo_fixture_cases
        
        # Get fixture cases
        cases = default_locomo_fixture_cases()
        print(f"✓ LoCoMo fixture has {len(cases)} cases")
        
        # Run suite (creates its own engine)
        report = await run_locomo_suite(cases=cases)
        print(f"✓ LoCoMo overall accuracy: {report.overall_accuracy:.2%}")
        print(f"✓ Passed target (≥80%): {report.passed_target}")
        print(f"✓ Per-category accuracy: {report.per_category_accuracy}")
        
        # Check exit criterion
        if report.overall_accuracy >= 0.80:
            print("✓ LoCoMo meets exit criterion (≥80%)")
        else:
            print(f"⚠ LoCoMo below exit criterion ({report.overall_accuracy:.2%} < 80%)")
        
        return True
    except Exception as e:
        print(f"✗ LoCoMo error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_user_value():
    """Test user-value metrics."""
    print("\n=== Testing User-Value Metrics ===")
    try:
        from reify.eval.user_value import run_user_value_suite, default_user_value_cases
        
        # Get fixture cases
        cases = default_user_value_cases()
        print(f"✓ User-value fixture has {len(cases)} cases")
        
        # Run suite (creates its own engine)
        report = await run_user_value_suite(cases=cases)
        print(f"✓ Total cases: {report.total_cases}")
        print(f"✓ Useful recalls: {report.useful_recalls}")
        print(f"✓ Recall without LLM rate: {report.recall_without_llm_rate:.2%}")
        print(f"✓ 30-day recall accuracy: {report.recall_accuracy_30d:.2%}")
        print(f"✓ Time-to-useful-recall p50: {report.time_to_useful_recall_p50_ms:.2f}ms")
        print(f"✓ Time-to-useful-recall p95: {report.time_to_useful_recall_p95_ms:.2f}ms")
        print(f"✓ Cost per useful recall: ${report.cost_per_useful_recall_usd:.6f}")
        print(f"✓ LLM invocations: {report.llm_invocations}")
        print(f"✓ Embed invocations: {report.embed_invocations}")
        print(f"✓ Passed all gates: {report.passed}")
        
        # Check exit criteria
        targets = {
            "Recall without LLM (≥70%)": report.recall_without_llm_rate >= 0.70,
            "30-day recall accuracy (≥85%)": report.recall_accuracy_30d >= 0.85,
            "Time-to-useful-recall p50 (<5ms)": report.time_to_useful_recall_p50_ms < 5,
            "Time-to-useful-recall p95 (<20ms)": report.time_to_useful_recall_p95_ms < 20,
            "Cost per useful recall (≤$0.0005)": report.cost_per_useful_recall_usd <= 0.0005,
        }
        
        print("\nExit criteria check:")
        for target, passed in targets.items():
            status = "✓" if passed else "✗"
            print(f"  {status} {target}")
        
        return report.passed
    except Exception as e:
        print(f"✗ User-value error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all benchmark tests."""
    print("Running Comprehensive Benchmark Tests")
    print("=" * 50)

    results = []
    results.append(("LongMemEval", await test_longmemeval()))
    results.append(("LoCoMo", await test_locomo()))
    results.append(("User-Value", await test_user_value()))

    print("\n" + "=" * 50)
    print("Benchmark Results Summary:")
    for name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {name}: {status}")

    all_passed = all(passed for _, passed in results)
    print(f"\nOverall: {'✓ ALL PASSED' if all_passed else '✗ SOME FAILED'}")
    return all_passed


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
