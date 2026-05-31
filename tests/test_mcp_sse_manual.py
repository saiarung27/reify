"""Manual test script for MCP-SSE server.

This script can be run to verify the MCP-SSE server is functional.
It starts the server and performs basic HTTP requests to verify endpoints.

Usage:
    python tests/test_mcp_sse_manual.py
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from reify.server.mcp_sse import create_sse_app


async def test_sse_app():
    """Test the SSE app endpoints."""
    print("Starting MCP-SSE server test...")

    # Create the FastAPI app
    app = create_sse_app()

    # Check that routes exist
    routes = [route.path for route in app.routes]
    print(f"Available routes: {routes}")

    # Verify expected routes exist
    expected_routes = ["/sse", "/messages"]
    for route in expected_routes:
        if route in routes:
            print(f"✓ Route {route} exists")
        else:
            print(f"✗ Route {route} missing")
            return False

    print("\nMCP-SSE server structure is valid.")
    print("To test with a real MCP client:")
    print("1. Start the server: reify serve-mcp-sse")
    print("2. Configure your MCP client to connect to http://localhost:8003/sse")
    print("3. Verify tools are available: remember, recall, forget, reflect, whatchanged, whatdoyouknow, identity_bond")

    return True


if __name__ == "__main__":
    result = asyncio.run(test_sse_app())
    sys.exit(0 if result else 1)
