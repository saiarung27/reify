"""Fix 2: persona fidelity — brief-derived descriptions must reach profiles.

Regression coverage for the Telegram E2E finding: generated personae did
not match the brief (Maya Chen came out a Chinese marketer, Marcus Webb
an environmental activist) because (a) Reify nodes carried no summary at
the profile-generation path (list_nodes returned ""), and (b) the
free-text scan turned sentence-initial common words ("Holds", "Wants",
"It") into bogus "person" entities, so the sim filled with invented
people. These tests pin both fixes: sentence descriptions on every node
path, and junk singles kept out of the entity set.
"""

from __future__ import annotations

import pytest

from reify.adapters import ReifyZepGraph
from reify.core.types import Experience
from reify.retrieve.entities import classify_kind, extract_for_memory

BRIEF = (
    "MoveLensAlertBot is live on Telegram. Pick your tickers, get stock "
    "news, filings, spike alerts and dip levels in your own private DM.\n"
    "## People in the simulated world\n"
    "- Maya Chen, 29, beginner investor. Holds index funds and a few "
    "individual stocks. Checks her phone at lunch. Afraid of making a "
    "dumb mistake.\n"
    "- Darnell Brooks, 41, swing trader. Watches more than 15 tickers. "
    "Hates noisy alerts and unsubscribes fast when spammed.\n"
    "- Marcus Webb, 38, former day trader. Trusts charts, not tips. "
    "Will test any levels claim against real price history.\n"
    # Regression lines (2026-10-27 live rerun): sentence-initial verbs
    # that the first junk list missed became "person" nodes.
    "- Ben Ortiz, 47, manages a family portfolio. Privacy conscious. "
    "Dislikes spam and dislikes group chats.\n"
    "- Sara Lindqvist, 28, tech worker. Compares every new tool to the "
    "free alerts already inside her broker app.\n"
    "- David Okafor, 44, small-business owner. Invests weekly. Wants "
    "instant push alerts when a stock he watches spikes.\n"
    "CPI and FOMC decisions move markets. NVTS is a watched ticker.\n"
)


def _ids_for(text: str) -> dict[str, str]:
    exp = Experience(
        content=text, event_type="user_message", actor="user", summary="brief"
    )
    delta = extract_for_memory(exp)
    return {e.id: e.kind for e in delta.entities}


def test_sentence_start_common_words_are_not_entities() -> None:
    kinds = _ids_for(BRIEF)
    for junk in (
        "it", "holds", "checks", "afraid", "watches", "hates", "trusts",
        "will", "nobody", "would", "why", "what", "questions",
        "dislikes", "compares", "invests",
    ):
        assert junk not in kinds, f"{junk!r} must not become an entity"
    for person in ("maya_chen", "darnell_brooks", "marcus_webb",
                   "ben_ortiz", "sara_lindqvist", "david_okafor"):
        assert person in kinds, f"{person!r} must survive the filter"
        assert kinds[person] == "person"


def test_acronyms_classify_as_topic_not_person() -> None:
    assert classify_kind("CPI") == "topic"
    assert classify_kind("FOMC") == "topic"
    assert classify_kind("NVTS") == "topic"
    assert classify_kind("DM") == "topic"
    kinds = _ids_for(BRIEF)
    assert kinds.get("cpi") == "topic"
    assert kinds.get("nvts") == "topic"


@pytest.fixture
def adapter(tmp_path):
    with ReifyZepGraph(tmp_path / "graphs") as adapter:
        yield adapter


def test_adapter_node_summaries_are_defining_sentences(adapter) -> None:
    adapter.create_graph("sim-1")
    adapter.add_episode("sim-1", BRIEF, identity_id="brief")
    nodes = {n["name"]: n for n in adapter.list_nodes("sim-1")}
    assert "Maya Chen" in nodes
    maya = nodes["Maya Chen"]
    # The defining line, not the product preamble or an empty string.
    assert "29" in maya["summary"]
    assert "beginner investor" in maya["summary"]
    assert "MoveLensAlertBot is live" not in maya["summary"]
    assert "Darnell Brooks" not in maya["summary"]
    # Junk singles never reach the node list at all.
    for junk in ("It", "Holds", "Wants", "Would", "What"):
        assert junk not in nodes
    # Every node path agrees: get_node and search carry the same summary.
    detail = adapter.get_node("sim-1", maya["uuid"])
    assert detail is not None
    assert detail["summary"] == maya["summary"]
    found = adapter.search("sim-1", "Maya Chen beginner investor", limit=10)
    search_nodes = {n["name"]: n for n in found["nodes"]}
    assert "Maya Chen" in search_nodes
    assert search_nodes["Maya Chen"]["summary"] == maya["summary"]


def test_concurrent_adapter_searches_do_not_race_on_sqlite(adapter) -> None:
    """Parallel graph searches must not raise sqlite3.InterfaceError.

    Found live during Fix 2 profile verification: profile generation runs
    edge/node searches in parallel threads, and two interleaved recall
    coroutines ran their storage bodies on different default-executor
    threads against the store's single connection ("bad parameter or
    other API misuse"), degrading that persona to a bare fallback. The
    store now routes all blocking work through one worker thread, so
    concurrent adapter calls must all succeed.
    """
    import concurrent.futures

    adapter.create_graph("sim-race")
    adapter.add_episode("sim-race", BRIEF, identity_id="brief")

    def _search_once(_: int):
        found = adapter.search("sim-race", "Maya Chen investor", limit=10)
        names = [n["name"] for n in found["nodes"]]
        assert "Maya Chen" in names
        return True

    for _round in range(2):
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(_search_once, range(16)))
        assert all(results)
