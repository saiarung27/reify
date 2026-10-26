"""Tests for the Zep-graph-shaped adapter backed by Reify stores.

Covers the call surface MiroFish actually uses: graph create/delete,
episode add/get, search, node/edge reads, ontology recording, and the
batch ingestion shape. Edge facts here are structural (co-occurrence),
not LLM-extracted relations -- the tests pin the shape, not Zep parity.
"""

from __future__ import annotations

import pytest

from reify.adapters import ReifyZepGraph

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
def adapter(tmp_path):
    with ReifyZepGraph(tmp_path / "graphs") as adapter:
        yield adapter


def _seed(adapter, graph_id="sim-1"):
    adapter.create_graph(graph_id)
    ep1 = adapter.add_episode(graph_id, OFFER, identity_id="persona-a", tags=("offer",))
    ep2 = adapter.add_episode(graph_id, PRICING, identity_id="persona-a", tags=("pricing",))
    ep3 = adapter.add_episode(graph_id, OBJECTIONS, identity_id="persona-b", tags=("objection",))
    return ep1, ep2, ep3


def test_create_add_search_roundtrip(adapter):
    ep1, ep2, _ep3 = _seed(adapter)
    assert ep1["uuid"] and ep2["uuid"] and ep1["uuid"] != ep2["uuid"]

    found = adapter.get_episode("sim-1", ep2["uuid"])
    assert found is not None
    assert "99 dollars per month" in found["content"]

    result = adapter.search("sim-1", "price 99 dollars per month", limit=10)
    contents = [e["content"] for e in result["episodes"]]
    assert any("99 dollars per month" in c for c in contents)
    assert result["nodes"] or result["edges"] or result["episodes"]


def test_search_with_tag_filter_is_exact(adapter):
    _seed(adapter)
    result = adapter.search("sim-1", "anything", tags=("pricing",))
    assert len(result["episodes"]) == 1
    assert "99 dollars per month" in result["episodes"][0]["content"]


def test_node_and_edge_reads(adapter):
    _seed(adapter)
    nodes = adapter.list_nodes("sim-1")
    assert nodes, "entity indexing should surface nodes from plain text"
    names = [n["name"] for n in nodes]
    assert any("Queen City" in name for name in names)

    node = next(n for n in nodes if "Queen City" in n["name"])
    detail = adapter.get_node("sim-1", node["uuid"])
    assert detail is not None and detail["name"] == node["name"]
    assert detail["summary"]

    edges = adapter.get_node_edges("sim-1", node["uuid"])
    assert edges, "co-occurrence edges should exist for a mentioned entity"
    assert all("fact" in e and e["source"] == node["uuid"] for e in edges)

    all_edges = adapter.list_edges("sim-1")
    assert all_edges


def test_ontology_recorded_not_enforced(adapter):
    adapter.create_graph("sim-1")
    ontology = {"entity_types": [{"name": "Cleaner"}], "edge_types": [{"name": "worries_about"}]}
    stored = adapter.set_ontology("sim-1", ontology)
    assert stored["ontology"] == ontology
    assert adapter.get_ontology("sim-1") == ontology
    # Recording must not block writes of unlisted types.
    ep = adapter.add_episode("sim-1", "A new cleaner joined today.")
    assert ep["uuid"]


def test_batch_flow(adapter):
    adapter.create_graph("sim-1")
    batch_id = adapter.create_batch("sim-1")
    adapter.add_to_batch(batch_id, [{"content": OFFER}, {"content": PRICING}])
    summary = adapter.process_batch(batch_id)
    assert summary["status"] == "completed"
    assert summary["added_count"] == 2
    status = adapter.get_batch(batch_id)
    assert status["status"] == "completed" and status["item_count"] == 2
    assert len(adapter.list_batch_items(batch_id)) == 2
    result = adapter.search("sim-1", "99 dollars per month", limit=10)
    assert any("99 dollars" in e["content"] for e in result["episodes"])


def test_delete_graph_removes_store(adapter, tmp_path):
    _seed(adapter)
    assert adapter.delete_graph("sim-1") is True
    assert list((tmp_path / "graphs").glob("sim-1.db")) == []
    assert adapter.get_ontology("sim-1") is None


class FakeExtractor:
    """Deterministic stand-in for LLM extraction."""

    def __init__(self, triples=(), *, fail=False):
        self._triples = list(triples)
        self._fail = fail
        self.calls: list[str] = []

    def extract(self, text):
        self.calls.append(text)
        if self._fail:
            raise RuntimeError("provider down")
        from reify.adapters.relations import RelationTriple

        return [RelationTriple(*t) for t in self._triples]


def test_relation_extraction_adds_typed_edges(adapter, tmp_path):
    from reify.adapters import ReifyZepGraph

    extractor = FakeExtractor(
        [("Maya", "would_pay_only_if", "three bookings"),
         ("Maya", "finds_too_expensive", "99 dollars a month")]
    )
    with ReifyZepGraph(tmp_path / "graphs2", relation_extractor=extractor) as a2:
        a2.create_graph("sim-typed")
        a2.add_episode(
            "sim-typed",
            "Maya says she would pay 99 dollars a month only if three bookings come in.",
            identity_id="persona-1",
        )
        edges = a2.list_edges("sim-typed")
        relations = {e["relation"] for e in edges}
        assert "would_pay_only_if" in relations
        assert "finds_too_expensive" in relations
        typed = [e for e in edges if e["relation"] == "would_pay_only_if"][0]
        assert "Maya" in typed["fact"] and "three bookings" in typed["fact"]
        assert extractor.calls


def test_extractor_failure_never_breaks_episode_write(adapter, tmp_path):
    from reify.adapters import ReifyZepGraph

    with ReifyZepGraph(tmp_path / "graphs3", relation_extractor=FakeExtractor(fail=True)) as a3:
        a3.create_graph("sim-fail")
        ep = a3.add_episode("sim-fail", "Maya worries about lead quality.")
        assert ep["uuid"]
        assert a3.get_episode("sim-fail", ep["uuid"]) is not None


class FakeLLMProvider:
    """Stand-in for an LLM provider returning a fixed JSON triple list."""

    name = "fake"
    default_model = "fake-model"

    def __init__(self, content):
        self.content = content
        self.seen_prompts: list[str] = []

    async def complete(self, request):
        self.seen_prompts.append(request.messages[0]["content"])
        from reify.llm.base import LLMResponse

        return LLMResponse(content=self.content, model="fake-model")


def test_llm_extractor_parses_json_triples():
    """Regression: the prompt template once crashed on its own JSON example
    (str.format parsed {"subject"} as a placeholder). A fake provider run
    through the real extractor must yield triples, not []."""
    import asyncio

    from reify.adapters.relations import LLMRelationExtractor

    provider = FakeLLMProvider(
        '[{"subject": "Maya", "relation": "finds_too_expensive", '
        '"object": "99 dollars a month"}]'
    )
    extractor = LLMRelationExtractor(provider, model="fake-model")
    triples = asyncio.run(extractor._extract_async("Maya says 99 dollars is too much."))
    assert len(triples) == 1
    assert triples[0].subject == "Maya"
    assert triples[0].relation == "finds_too_expensive"
    assert triples[0].object == "99 dollars a month"
    assert '{"subject"' in provider.seen_prompts[0]
