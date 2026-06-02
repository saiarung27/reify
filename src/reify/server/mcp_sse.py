"""MCP server with HTTP/SSE transport.

This provides an HTTP/SSE transport for the MCP protocol, allowing
MCP clients to connect over HTTP instead of stdio.

Per FIRST_PRINCIPLES.md, this adapter is thin and transport-only - it does
not contain unique memory logic absent from the library.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from mcp.server import Server
from mcp.server.sse import SseServerTransport

from reify import EngineConfig, Reify
from reify.config import get_settings
from reify.llm import create_llm_provider
from reify.storage import SQLiteStore


def create_mcp_server() -> Server:
    """Create and configure the MCP server instance."""
    settings = get_settings()
    settings.ensure_dirs()

    server = Server("reify")
    try:
        llm_provider = create_llm_provider(settings.llm_reconsolidate)
    except Exception:
        llm_provider = None

    engine = Reify(
        storage=SQLiteStore(path=settings.db_path),
        config=EngineConfig(llm=llm_provider),
    )

    # Register tools (same as stdio server)
    from reify.server.mcp import register_tools

    register_tools(server, engine)

    return server


def create_sse_app() -> FastAPI:
    """Create a FastAPI app with SSE transport for MCP."""
    mcp_server = create_mcp_server()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await mcp_server.startup()
        yield
        await mcp_server.shutdown()

    app = FastAPI(title="Reify MCP SSE Server", lifespan=lifespan)
    from reify.server.security import harden_app
    harden_app(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/sse")
    async def sse_endpoint(request: Request) -> StreamingResponse:
        """SSE endpoint for MCP communication."""
        transport = SseServerTransport("/messages")

        async def handle_sse() -> None:
            async with transport.connect_sse(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )

        return StreamingResponse(
            handle_sse(),
            media_type="text/event-stream",
        )

    @app.post("/messages")
    async def messages_endpoint(request: Request) -> StreamingResponse:
        """Messages endpoint for MCP communication."""
        transport = SseServerTransport("/sse")

        async def handle_messages() -> None:
            async with transport.connect_messages(
                request.scope, request.receive, request._send
            ) as (read_stream, write_stream):
                await mcp_server.run(
                    read_stream,
                    write_stream,
                    mcp_server.create_initialization_options(),
                )

        return StreamingResponse(
            handle_messages(),
            media_type="text/event-stream",
        )

    return app


__all__ = [
    "create_mcp_server",
    "create_sse_app",
]
