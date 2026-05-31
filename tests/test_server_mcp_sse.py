"""Tests for the v0.6 MCP-SSE server."""

from __future__ import annotations

from reify.server.mcp_sse import create_mcp_server, create_sse_app


def test_create_mcp_server() -> None:
    server = create_mcp_server()
    assert server is not None
    assert server.name == "reify"


def test_create_sse_app() -> None:
    app = create_sse_app()
    assert app is not None
    assert app.title == "Reify MCP SSE Server"


def test_sse_app_routes_exist() -> None:
    app = create_sse_app()
    # Check that routes are registered
    route_paths = [route.path for route in app.routes]
    assert "/sse" in route_paths
    assert "/messages" in route_paths
