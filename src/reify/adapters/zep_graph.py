"""A Zep Cloud graph-shaped surface backed by local Reify stores.

Purpose: MiroFish drives its simulation memory through Zep Cloud
(``graph.create/delete/add/search``, ``node.get/get_edges``,
``episode.get``, ``set_ontology``, batch ingestion). Zep's terms grant
the vendor a perpetual license to train on customer data, so this
adapter replaces that layer with Reify: one SQLite store per graph,
everything on local disk, nothing sent anywhere.

Mapping (deliberate, documented where it is lossy):

* Zep graph_id -> one Reify store file (``<graph_id>.db``) under the
  adapter root, plus an optional ``<graph_id>.ontology.json``.
* Zep episode -> one Reify ``remember`` call. The episode UUID is the
  first committed memory's id. A ``remember`` can yield several
  memories for structured text; the episode identity is the first.
* Zep search -> Reify ``recall`` (metadata-first + graph walk), with
  the evidence-query tag filter available via ``tags=``.
* Zep node -> a Reify entity (``entities`` table). Node ids are Reify
  entity slugs (``queen_city_home_refresh``), NOT UUIDs: callers must
  treat them as opaque strings.
* Zep edge fact -> Reify entity edges. Without a relation extractor,
  edges are deterministic mentions + ``co_occurs`` only. With a
  ``relation_extractor`` (e.g. :class:`LLMRelationExtractor`), each
  episode additionally yields typed relation edges, which is the
  Zep-parity path. ``fact`` strings render with entity labels:
  ``"<subject label> <relation> <object label>"`` (``co_occurs``
  renders as ``"<a> co-occurs with <b>"``). Extracted relations are
  model output, not verified claims -- same caveat as Zep's.
* Zep batch -> a synchronous in-adapter registry. ``process_batch``
  ingests immediately, so ``get_batch`` reports ``completed`` right
  away. MiroFish's poll loop still works; it just never waits.
* ``set_ontology`` is recorded, not enforced: entity/edge type names
  are stored for inspection, but Reify does not validate against them.

Threading: MiroFish calls Zep synchronously, while Reify's engine is
async. The adapter owns one private event-loop thread; every engine
call runs on that loop via ``run_coroutine_threadsafe``. Public methods
are sync and safe to call from any thread. Do NOT call them from the
adapter's own loop thread (there is no public path that does).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import uuid
from pathlib import Path
from typing import Any

from reify import EngineConfig, Reify
from reify.adapters.relations import RelationExtractor

log = logging.getLogger(__name__)
from reify.retrieve.entities import EntityEdge, canonicalize
from reify.storage import SQLiteStore


def _episode_dict(memory: Any) -> dict[str, Any]:
    return {
        "uuid": memory.id,
        "content": memory.content,
        "created_at": memory.created_at.isoformat(),
        "tags": list(memory.tags),
    }


def _node_dict(entity: dict[str, Any], summary: str = "") -> dict[str, Any]:
    return {
        "uuid": entity["id"],
        "name": entity["label"],
        "labels": [entity["kind"]] if entity.get("kind") else [],
        "summary": summary,
    }


def _describe_mention(content: str, span_start: int, span_end: int) -> str:
    """Best short description of an entity from its defining mention.

    Takes the LINE containing the mention span: brief-style list items put
    one person per line, so the line IS the description ("Maya Chen, 29,
    beginner investor. Holds index funds..."). If the line is long prose,
    narrows to the sentence containing the span. Bounded to 500 chars so
    node summaries stay prompt-sized. The old content[:200] truncation
    regularly cut mid-sentence and, for big chunks, described the wrong
    subject entirely (product preamble instead of the person).
    """
    if not content:
        return ""
    span_start = max(0, min(span_start, len(content)))
    span_end = max(span_start, min(span_end, len(content)))
    line_start = content.rfind("\n", 0, span_start) + 1
    line_end = content.find("\n", span_end)
    if line_end == -1:
        line_end = len(content)
    line = content[line_start:line_end].strip(" \t-•*")
    if len(line) > 300:
        rel = span_start - line_start
        bounds = [0]
        bounds += [m.end() for m in re.finditer(r"[.!?]\s+", line)]
        bounds.append(len(line))
        for i in range(len(bounds) - 1):
            if bounds[i] <= rel < bounds[i + 1]:
                line = line[bounds[i]:bounds[i + 1]]
                break
    return line[:500].strip()


async def _mention_summaries(store: Any, entity_ids: list[str]) -> dict[str, str]:
    """entity_id -> sentence description from its earliest mention.

    Uses stored mention spans (definitional-first order), so the summary
    is the text that introduces the entity, not an arbitrary chunk head.
    """
    summaries: dict[str, str] = {}
    mention_spans = getattr(store, "mention_spans", None)
    if mention_spans is None or not entity_ids:
        return summaries
    spans = await mention_spans(entity_ids)
    if not spans:
        return summaries
    wanted = {eid for eid, _, _, _ in spans}
    memories = await store.memories_mentioning(list(wanted), limit=1000)
    contents: dict[str, str] = {}
    for mem, _eid, _conf in memories:
        contents.setdefault(mem.id, mem.content)
    seen: set[str] = set()
    for eid, mid, start, end in spans:
        if eid in seen or mid not in contents:
            continue
        seen.add(eid)
        summaries[eid] = _describe_mention(contents[mid], start, end)
    return summaries


def _edge_dict(
    from_id: str, to_id: str, relation: str, labels: dict[str, str]
) -> dict[str, Any]:
    from_label = labels.get(from_id, from_id)
    to_label = labels.get(to_id, to_id)
    if relation == "co_occurs":
        fact = f"{from_label} co-occurs with {to_label}"
    else:
        fact = f"{from_label} {relation} {to_label}"
    return {
        "source": from_id,
        "target": to_id,
        "relation": relation,
        "fact": fact,
    }


class ReifyZepGraph:
    """Sync Zep-graph-shaped client backed by Reify stores."""

    def __init__(
        self,
        root_dir: str | Path,
        *,
        relation_extractor: RelationExtractor | None = None,
    ) -> None:
        """Create an adapter.

        ``relation_extractor`` is optional and off by default: without
        it the graph carries deterministic mentions + ``co_occurs``
        edges only. With it (e.g. :class:`LLMRelationExtractor`), each
        episode also yields typed relation edges, which is what makes
        the graph comparable to Zep's LLM-extracted graph. Extraction
        failures never fail the episode write.
        """
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
        self._relation_extractor = relation_extractor
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop_forever, daemon=True)
        self._thread.start()
        self._engines: dict[str, Reify] = {}
        self._batches: dict[str, dict[str, Any]] = {}
        self._closed = False

    # -- loop plumbing ---------------------------------------------------

    def _loop_forever(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run(self, coro: Any) -> Any:
        if self._closed:
            raise RuntimeError("ReifyZepGraph is closed")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=300)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        async def _stop_all() -> None:
            for engine in self._engines.values():
                await engine.stop()

        try:
            asyncio.run_coroutine_threadsafe(_stop_all(), self._loop).result(timeout=60)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10)

    def __enter__(self) -> "ReifyZepGraph":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # -- graphs ----------------------------------------------------------

    def create_graph(self, graph_id: str) -> dict[str, Any]:
        self._run(self._ensure_engine(graph_id))
        return {"graph_id": graph_id}

    def delete_graph(self, graph_id: str) -> bool:
        async def _delete() -> bool:
            engine = self._engines.pop(graph_id, None)
            if engine is not None:
                await engine.stop()
            removed = False
            for path in (self._store_path(graph_id), self._ontology_path(graph_id)):
                if path.exists():
                    path.unlink()
                    removed = True
            return removed

        return self._run(_delete())

    def _store_path(self, graph_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in graph_id)
        return self._root / f"{safe}.db"

    def _ontology_path(self, graph_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in graph_id)
        return self._root / f"{safe}.ontology.json"

    async def _ensure_engine(self, graph_id: str) -> Reify:
        engine = self._engines.get(graph_id)
        if engine is None:
            engine = Reify(
                storage=SQLiteStore(path=str(self._store_path(graph_id))),
                config=EngineConfig(),
            )
            await engine.start()
            self._engines[graph_id] = engine
        return engine

    # -- ontology --------------------------------------------------------

    def set_ontology(self, graph_id: str, ontology: dict[str, Any]) -> dict[str, Any]:
        self._run(self._ensure_engine(graph_id))
        path = self._ontology_path(graph_id)
        path.write_text(json.dumps(ontology, indent=2, sort_keys=True))
        return {"graph_id": graph_id, "ontology": ontology}

    def get_ontology(self, graph_id: str) -> dict[str, Any] | None:
        path = self._ontology_path(graph_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    # -- episodes ----------------------------------------------------------

    def add_episode(
        self,
        graph_id: str,
        content: str,
        *,
        identity_id: str | None = None,
        tags: tuple[str, ...] = (),
        episode_type: str = "message",
    ) -> dict[str, Any]:
        async def _add() -> dict[str, Any]:
            engine = await self._ensure_engine(graph_id)
            receipt = await engine.remember(
                content,
                identity_id=identity_id,
                tags=(f"episode:{episode_type}", *tags),
            )
            if not receipt.committed:
                raise ValueError("episode produced no memories (empty extraction)")
            if self._relation_extractor is not None:
                await self._extract_relations(engine, content)
            return _episode_dict(receipt.committed[0])

        return self._run(_add())

    async def _extract_relations(self, engine: Reify, content: str) -> None:
        """Best-effort typed edges for one episode. Never raises."""
        extractor = self._relation_extractor
        if extractor is None:
            return
        try:
            triples = await asyncio.to_thread(extractor.extract, content)
        except Exception as exc:  # noqa: BLE001 - extraction is best-effort
            log.warning("relation_extraction_failed", extra={"error": repr(exc)})
            return
        if not triples:
            return
        store = engine.storage
        upsert = getattr(store, "upsert_entities", None)
        add_edges = getattr(store, "add_entity_edges", None)
        if upsert is None or add_edges is None:
            return
        entities: dict[str, Any] = {}
        edges: list[EntityEdge] = []
        for triple in triples:
            subj = canonicalize(triple.subject)
            obj = canonicalize(triple.object)
            if subj is None or obj is None or subj.id == obj.id:
                continue
            entities[subj.id] = subj
            entities[obj.id] = obj
            edges.append(
                EntityEdge(
                    from_entity=subj.id,
                    to_entity=obj.id,
                    relation=triple.relation,
                    weight=1.0,
                )
            )
        if not edges:
            return
        try:
            await upsert(list(entities.values()))
            await add_edges(edges)
        except Exception as exc:  # noqa: BLE001 - extraction is best-effort
            log.warning("relation_extraction_store_failed", extra={"error": repr(exc)})

    def add_episodes(
        self, graph_id: str, items: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            self.add_episode(
                graph_id,
                item["content"],
                identity_id=item.get("identity_id"),
                tags=tuple(item.get("tags", ())),
                episode_type=item.get("episode_type", "message"),
            )
            for item in items
        ]

    def get_episode(self, graph_id: str, uuid_: str) -> dict[str, Any] | None:
        async def _get() -> dict[str, Any] | None:
            engine = await self._ensure_engine(graph_id)
            memory = await engine.get_memory(uuid_)
            return _episode_dict(memory) if memory is not None else None

        return self._run(_get())

    # -- search ------------------------------------------------------------

    def search(
        self,
        graph_id: str,
        query: str,
        *,
        limit: int = 20,
        identity_id: str | None = None,
        tags: tuple[str, ...] | None = None,
    ) -> dict[str, Any]:
        async def _search() -> dict[str, Any]:
            engine = await self._ensure_engine(graph_id)
            resp = await engine.recall(
                query, identity_id=identity_id, limit=limit, tags=tags
            )
            episodes = [_episode_dict(r.memory) for r in resp.results]
            episode_ids = {e["uuid"] for e in episodes}
            nodes, edges = await self._nodes_and_edges_for(
                engine, episode_ids, limit=limit
            )
            return {"episodes": episodes, "nodes": nodes, "edges": edges}

        return self._run(_search())

    async def _nodes_and_edges_for(
        self, engine: Reify, episode_ids: set[str], *, limit: int
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        store = engine.storage
        list_entities = getattr(store, "list_entities", None)
        list_edges = getattr(store, "list_edges", None)
        if list_entities is None:
            return [], []
        entities = await list_entities(limit=1000)
        if not entities:
            return [], []
        mentions = await store.memories_mentioning(
            [e["id"] for e in entities], limit=1000
        )
        episode_entity_ids = {
            eid for m, eid, _ in mentions if m.id in episode_ids
        }
        summaries = await _mention_summaries(
            store, [e["id"] for e in entities]
        )
        nodes = [
            _node_dict(e, summary=summaries.get(e["id"], ""))
            for e in entities
            if e["id"] in episode_entity_ids
        ]
        node_ids = {n["uuid"] for n in nodes}
        edges: list[dict[str, Any]] = []
        if list_edges is not None:
            labels = {e["id"]: e["label"] for e in entities}
            for row in await list_edges(limit=1000):
                if row["from_entity"] in node_ids and row["to_entity"] in node_ids:
                    edges.append(
                        _edge_dict(
                            row["from_entity"], row["to_entity"], row["relation"], labels
                        )
                    )
        return nodes[:limit], edges[:limit]

    # -- nodes / edges -------------------------------------------------------

    def get_node(self, graph_id: str, uuid_: str) -> dict[str, Any] | None:
        async def _get() -> dict[str, Any] | None:
            engine = await self._ensure_engine(graph_id)
            store = engine.storage
            list_entities = getattr(store, "list_entities", None)
            if list_entities is None:
                return None
            for entity in await list_entities(limit=1000):
                if entity["id"] == uuid_:
                    summaries = await _mention_summaries(store, [uuid_])
                    return _node_dict(
                        entity, summary=summaries.get(uuid_, "")
                    )
            return None

        return self._run(_get())

    def get_node_edges(self, graph_id: str, uuid_: str) -> list[dict[str, Any]]:
        async def _edges() -> list[dict[str, Any]]:
            engine = await self._ensure_engine(graph_id)
            store = engine.storage
            rows = await store.neighbors(uuid_)
            labels: dict[str, str] = {uuid_: uuid_}
            list_entities = getattr(store, "list_entities", None)
            if list_entities is not None:
                for entity in await list_entities(limit=1000):
                    labels[entity["id"]] = entity["label"]
            return [
                _edge_dict(uuid_, row.to_entity, row.relation, labels) for row in rows
            ]

        return self._run(_edges())

    def list_nodes(self, graph_id: str) -> list[dict[str, Any]]:
        async def _list() -> list[dict[str, Any]]:
            engine = await self._ensure_engine(graph_id)
            store = engine.storage
            list_entities = getattr(store, "list_entities", None)
            if list_entities is None:
                return []
            entities = await list_entities(limit=1000)
            summaries = await _mention_summaries(
                store, [e["id"] for e in entities]
            )
            return [
                _node_dict(e, summary=summaries.get(e["id"], ""))
                for e in entities
            ]

        return self._run(_list())

    def list_edges(self, graph_id: str) -> list[dict[str, Any]]:
        async def _list() -> list[dict[str, Any]]:
            engine = await self._ensure_engine(graph_id)
            store = engine.storage
            list_edges = getattr(store, "list_edges", None)
            if list_edges is None:
                return []
            labels: dict[str, str] = {}
            list_entities = getattr(store, "list_entities", None)
            if list_entities is not None:
                for entity in await list_entities(limit=1000):
                    labels[entity["id"]] = entity["label"]
            return [
                _edge_dict(
                    row["from_entity"], row["to_entity"], row["relation"], labels
                )
                for row in await list_edges(limit=1000)
            ]

        return self._run(_list())

    # -- batches ---------------------------------------------------------------

    def create_batch(self, graph_id: str) -> str:
        batch_id = uuid.uuid4().hex
        self._batches[batch_id] = {
            "graph_id": graph_id,
            "items": [],
            "status": "created",
        }
        return batch_id

    def add_to_batch(self, batch_id: str, items: list[dict[str, Any]]) -> int:
        batch = self._batches[batch_id]
        batch["items"].extend(items)
        return len(batch["items"])

    def process_batch(self, batch_id: str) -> dict[str, Any]:
        batch = self._batches[batch_id]
        added = self.add_episodes(batch["graph_id"], batch["items"])
        batch["status"] = "completed"
        batch["added"] = added
        return {
            "batch_id": batch_id,
            "status": "completed",
            "added_count": len(added),
        }

    def get_batch(self, batch_id: str) -> dict[str, Any]:
        batch = self._batches[batch_id]
        return {
            "batch_id": batch_id,
            "status": batch["status"],
            "item_count": len(batch["items"]),
            "added_count": len(batch.get("added", [])),
        }

    def list_batch_items(self, batch_id: str) -> list[dict[str, Any]]:
        batch = self._batches[batch_id]
        return list(batch["items"])
