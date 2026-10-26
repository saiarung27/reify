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
* Zep edge fact -> Reify entity edges, which are co-occurrence edges
  from the deterministic extractor plus extractor facts. This is the
  known fidelity gap: Zep's edges are LLM-extracted relations with
  natural-language facts; Reify's are structural. ``fact`` strings
  here read as ``"<source> <relation> <target>"`` and must not be
  presented as LLM-extracted claims.
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
import threading
import uuid
from pathlib import Path
from typing import Any

from reify import EngineConfig, Reify
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


class ReifyZepGraph:
    """Sync Zep-graph-shaped client backed by Reify stores."""

    def __init__(self, root_dir: str | Path) -> None:
        self._root = Path(root_dir)
        self._root.mkdir(parents=True, exist_ok=True)
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
            return _episode_dict(receipt.committed[0])

        return self._run(_add())

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
        by_entity: dict[str, list[str]] = {}
        for memory, entity_id, _conf in mentions:
            by_entity.setdefault(entity_id, []).append(memory.content)
        episode_entity_ids = {
            eid for m, eid, _ in mentions if m.id in episode_ids
        }
        nodes = [
            _node_dict(e, summary=(by_entity.get(e["id"], [""])[0][:200]))
            for e in entities
            if e["id"] in episode_entity_ids
        ]
        node_ids = {n["uuid"] for n in nodes}
        edges: list[dict[str, Any]] = []
        if list_edges is not None:
            for row in await list_edges(limit=1000):
                if row["from_entity"] in node_ids and row["to_entity"] in node_ids:
                    edges.append(
                        {
                            "source": row["from_entity"],
                            "target": row["to_entity"],
                            "relation": row["relation"],
                            "fact": (
                                f"{row['from_entity']} {row['relation']} "
                                f"{row['to_entity']}"
                            ),
                        }
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
                    mentions = await store.memories_mentioning([uuid_], limit=1)
                    summary = mentions[0][0].content[:200] if mentions else ""
                    return _node_dict(entity, summary=summary)
            return None

        return self._run(_get())

    def get_node_edges(self, graph_id: str, uuid_: str) -> list[dict[str, Any]]:
        async def _edges() -> list[dict[str, Any]]:
            engine = await self._ensure_engine(graph_id)
            rows = await engine.storage.neighbors(uuid_)
            return [
                {
                    "source": uuid_,
                    "target": row.to_entity,
                    "relation": row.relation,
                    "fact": f"{uuid_} {row.relation} {row.to_entity}",
                }
                for row in rows
            ]

        return self._run(_edges())

    def list_nodes(self, graph_id: str) -> list[dict[str, Any]]:
        async def _list() -> list[dict[str, Any]]:
            engine = await self._ensure_engine(graph_id)
            store = engine.storage
            list_entities = getattr(store, "list_entities", None)
            if list_entities is None:
                return []
            return [_node_dict(e) for e in await list_entities(limit=1000)]

        return self._run(_list())

    def list_edges(self, graph_id: str) -> list[dict[str, Any]]:
        async def _list() -> list[dict[str, Any]]:
            engine = await self._ensure_engine(graph_id)
            store = engine.storage
            list_edges = getattr(store, "list_edges", None)
            if list_edges is None:
                return []
            return [
                {
                    "source": row["from_entity"],
                    "target": row["to_entity"],
                    "relation": row["relation"],
                    "fact": (
                        f"{row['from_entity']} {row['relation']} "
                        f"{row['to_entity']}"
                    ),
                }
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
