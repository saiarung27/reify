"""Evidence-style recall: tag-filtered queries and tag-misfire fallback.

Regression context: a free-text question mentioning a business name
("Queen City") inferred the ``location`` tag from the token "city",
which matched no stored memory, so the metadata-first pipeline returned
zero candidates and the answer came solely from a graph-walk fallback
presented at score 1.00. These tests pin the two fixes:

1. Inferred tags that match nothing degrade to the identity-scoped set
   instead of returning empty.
2. Caller-supplied tags at ingest plus explicit tag filters at query
   give exact evidence retrieval for callers that know the section
   they want (the MiroFish adapter pattern).
"""

from __future__ import annotations

import pytest

from reify import EngineConfig, Reify
from reify.core.types import Fact
from reify.engine import _merge_recall_results
from reify.recall import RecallResult
from reify.storage import SQLiteStore

OFFER = (
    "Queen City Home Refresh sells weekly lead lists to Charlotte house "
    "cleaners, built from Facebook group posts asking for cleaning"
)
PRICING = (
    "After a free 3-week pilot, the price is 99 dollars per month, "
    "or 8 to 12 dollars per lead"
)
OBJECTIONS = (
    "Cleaners worry most about lead quality, trust in a new seller, "
    "and whether the same ZIP goes to competing cleaners"
)


@pytest.fixture
async def engine():
    eng = Reify(storage=SQLiteStore(":memory:"), config=EngineConfig())
    async with eng:
        yield eng


async def _remember_brief(engine: Reify, *, tagged: bool = False) -> None:
    await engine.remember(OFFER, identity_id="sim1", tags=("offer",) if tagged else ())
    await engine.remember(PRICING, identity_id="sim1", tags=("pricing",) if tagged else ())
    await engine.remember(OBJECTIONS, identity_id="sim1", tags=("objection",) if tagged else ())


async def test_tag_misfire_falls_back_to_identity_scoped_candidates(engine: Reify) -> None:
    """A tag inferred from a business name must not empty the result."""
    await _remember_brief(engine)
    resp = await engine.recall(
        "What does Queen City Home Refresh charge after the free pilot?",
        identity_id="sim1",
    )
    assert len(resp.results) == 3
    assert any(PRICING in r.memory.content for r in resp.results)
    # No graph-only result may present at full confidence.
    assert all(r.score < 1.0 for r in resp.results if r.matched_stage == "graph")


async def test_caller_tags_enable_exact_evidence_recall(engine: Reify) -> None:
    await _remember_brief(engine, tagged=True)
    resp = await engine.recall("pricing", identity_id="sim1", tags=("pricing",))
    assert len(resp.results) == 1
    assert PRICING in resp.results[0].memory.content
    resp2 = await engine.recall("objections", identity_id="sim1", tags=("objection",))
    assert len(resp2.results) == 1
    assert OBJECTIONS in resp2.results[0].memory.content


async def test_remember_merges_caller_tags(engine: Reify) -> None:
    receipt = await engine.remember(PRICING, identity_id="sim1", tags=("pricing",))
    assert receipt.committed
    assert all("pricing" in m.tags for m in receipt.committed)


def test_merge_keeps_graph_only_scores_unscaled() -> None:
    graph_only = RecallResult(memory=Fact(content="graph hit", subject="x", predicate="rel", obj="y"), score=0.4, matched_stage="graph")
    merged = _merge_recall_results([], [graph_only], limit=20)
    assert len(merged) == 1
    assert merged[0].score == pytest.approx(0.4)


def test_merge_caps_secondary_below_primary() -> None:
    primary = RecallResult(memory=Fact(content="primary", subject="x", predicate="rel", obj="y"), score=0.5, matched_stage="lexical")
    secondary = RecallResult(memory=Fact(content="graph", subject="x", predicate="rel", obj="y"), score=0.4, matched_stage="graph")
    merged = _merge_recall_results([primary], [secondary], limit=20)
    assert merged[0].memory.content == "primary"
    assert merged[1].score < 0.5


async def test_explicit_tag_with_no_match_returns_empty(engine: Reify) -> None:
    """An evidence query's tag filter is exact: no degrade to untagged rows."""
    await _remember_brief(engine, tagged=True)
    resp = await engine.recall("anything", identity_id="sim1", tags=("nonexistent",))
    assert len(resp.results) == 0


async def test_pure_tag_query_does_not_resurrect_untagged(engine: Reify) -> None:
    """Mirrors the forget-by-tag contract: a query that IS a tag stays a filter."""
    await engine.remember("I work at Stripe.", identity_id="sim1")
    await engine.forget(tag="employer")
    resp = await engine.recall("employer", identity_id="sim1")
    assert all("Stripe" not in r.memory.content for r in resp.results)
