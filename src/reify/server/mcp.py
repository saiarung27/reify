"""MCP server entry point.

This is the primary surface through which AI hosts (Claude Desktop, Cursor,
Cline, Zed, Windsurf, ChatGPT desktop, and any future MCP-compatible client)
talk to Reify.

Exposed tools (the surface a host sees)
---------------------------------------
* ``remember`` — submit raw content; the engine extracts, canonicalizes, and commits.
* ``reflect`` — drain the Active Reconsolidation queue.
* ``recall`` — retrieve relevant memories for a query, returning structured records.
* ``whatchanged`` — query transitions within a time window.
* ``whatdoyouknow`` — recall all memories matching a topic tag.
* ``forget`` — tombstone one memory or a filtered set.

This is a thin adapter (< 200 LOC) that delegates to the Reify engine.
"""

from __future__ import annotations

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from reify.engine import Reify, EngineConfig
from reify.config import get_settings
from reify.llm import create_llm_provider
from reify.storage import SQLiteStore


def register_tools(server: Server, engine: Reify) -> None:
    """Register MCP tools with the server."""

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name="remember",
                description="Remember content. Extracts facts and stores them.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "Content to remember"},
                        "identity": {"type": "string", "description": "User identity (optional)"},
                    },
                    "required": ["content"],
                },
            ),
            Tool(
                name="recall",
                description="Recall memories matching a query.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "identity": {"type": "string", "description": "User identity (optional)"},
                        "limit": {"type": "number", "description": "Max results (default: 20)"},
                    },
                    "required": ["query"],
                },
            ),
            Tool(
                name="reflect",
                description="Drain the Active Reconsolidation queue.",
                inputSchema={"type": "object", "properties": {}},
            ),
            Tool(
                name="whatchanged",
                description="Query transitions within a time window.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "days": {"type": "number", "description": "Look back this many days"},
                        "identity": {"type": "string", "description": "User identity (optional)"},
                    },
                    "required": ["days"],
                },
            ),
            Tool(
                name="whatdoyouknow",
                description="Recall all memories with a specific tag/topic.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "topic": {"type": "string", "description": "Topic tag to filter by"},
                        "identity": {"type": "string", "description": "User identity (optional)"},
                    },
                    "required": ["topic"],
                },
            ),
            Tool(
                name="forget",
                description="Tombstone one memory or a filtered set.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {"type": "string", "description": "Memory id to forget"},
                        "tag": {"type": "string", "description": "Forget memories with this tag"},
                        "identity": {"type": "string", "description": "User identity (optional)"},
                    },
                },
            ),
            Tool(
                name="identity_bond",
                description="Link two identities with a bond.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "source_id": {"type": "string", "description": "Source identity ID"},
                        "target_id": {"type": "string", "description": "Target identity ID"},
                        "confidence": {"type": "number", "description": "Bond confidence (0-1, default 0.8)"},
                    },
                    "required": ["source_id", "target_id"],
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "remember":
            receipt = await engine.remember(
                arguments["content"],
                identity_id=arguments.get("identity"),
            )
            return [TextContent(type="text", text=f"Remembered {len(receipt.committed)} memories")]

        elif name == "reflect":
            processed = await engine.flush_arw()
            return [TextContent(type="text", text=f"Processed {processed} queued memories")]

        elif name == "recall":
            response = await engine.recall(
                arguments["query"],
                identity_id=arguments.get("identity"),
                limit=arguments.get("limit", 20),
            )
            formatted = "\n".join(f"- {r.memory.content}" for r in response.results)
            return [TextContent(type="text", text=formatted or "No memories found")]

        elif name == "whatchanged":
            transitions = await engine.whatchanged(
                days=arguments["days"],
                identity_id=arguments.get("identity"),
            )
            formatted = "\n".join(f"- {t.summary}" for t in transitions)
            return [TextContent(type="text", text=formatted or "No changes found")]

        elif name == "whatdoyouknow":
            memories = await engine.whatdoyouknow(
                arguments["topic"],
                identity_id=arguments.get("identity"),
            )
            formatted = "\n".join(f"- {m.content}" for m in memories)
            return [TextContent(type="text", text=formatted or "No memories found")]

        elif name == "forget":
            deleted = await engine.forget(
                arguments.get("memory_id"),
                tag=arguments.get("tag"),
                identity_id=arguments.get("identity"),
            )
            return [TextContent(type="text", text=f"Forgot {deleted} memories")]

        elif name == "identity_bond":
            from reify.identity.bond_graph import IdentityBond

            bond = IdentityBond(
                from_identity=arguments["source_id"],
                to_identity=arguments["target_id"],
                confidence=arguments.get("confidence", 0.8),
                rationale="explicit",
            )
            engine._identity_bond_graph.add_bond(bond)
            return [TextContent(type="text", text=f"Linked {arguments['source_id']} to {arguments['target_id']} with confidence {arguments.get('confidence', 0.8)}")]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def serve_stdio() -> None:
    """Run the MCP server over stdio.

    Exposes remember, recall, whatchanged, whatdoyouknow tools wired to the
    Reify engine.
    """
    # Ensure Reify directory exists
    settings = get_settings()
    settings.ensure_dirs()

    server = Server("reify")
    try:
        llm_provider = create_llm_provider(settings.llm_reconsolidate)
    except Exception:
        llm_provider = None

    engine = Reify(
        storage=SQLiteStore(path=settings.db_path),
        config=EngineConfig(
            llm=llm_provider,
            default_ttl_days=settings.default_ttl_days,
            reinforcement_factor=settings.reinforcement_factor,
            decay_sweep_interval_seconds=settings.decay_sweep_interval_seconds,
        ),
    )

    register_tools(server, engine)

    await engine.start()
    try:
        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await engine.stop()
