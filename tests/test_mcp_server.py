"""Real stdio MCP integration tests."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _server_params(home: Path) -> StdioServerParameters:
    env = os.environ.copy()
    env["REIFY_HOME"] = str(home)
    return StdioServerParameters(
        command=str(Path(sys.executable).with_name("reify")),
        args=["serve"],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
    )


@pytest.mark.asyncio
async def test_mcp_stdio_lists_tools(tmp_path: Path) -> None:
    async with stdio_client(_server_params(tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [tool.name for tool in tools.tools]
            assert names == [
                "remember",
                "recall",
                "reflect",
                "whatchanged",
                "whatdoyouknow",
                "forget",
                "identity_bond",
            ]


@pytest.mark.asyncio
async def test_mcp_stdio_remember_recall_flow(tmp_path: Path) -> None:
    async with stdio_client(_server_params(tmp_path)) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            remember = await session.call_tool(
                "remember",
                {"content": "I got promoted from Senior to Staff Engineer", "identity": "user123"},
            )
            remember_text = "\n".join(item.text for item in remember.content if hasattr(item, "text"))
            assert "Remembered" in remember_text

            reflect = await session.call_tool("reflect", {})
            reflect_text = "\n".join(item.text for item in reflect.content if hasattr(item, "text"))
            assert "Processed" in reflect_text

            recall = await session.call_tool(
                "recall",
                {"query": "what is my title?", "identity": "user123", "limit": 5},
            )
            recall_text = "\n".join(item.text for item in recall.content if hasattr(item, "text"))
            assert "Staff Engineer" in recall_text

            knowledge = await session.call_tool(
                "whatdoyouknow",
                {"topic": "promotion", "identity": "user123"},
            )
            knowledge_text = "\n".join(item.text for item in knowledge.content if hasattr(item, "text"))
            assert "Staff Engineer" in knowledge_text

            changed = await session.call_tool(
                "whatchanged",
                {"days": 30, "identity": "user123"},
            )
            changed_text = "\n".join(item.text for item in changed.content if hasattr(item, "text"))
            assert changed_text

            forgot = await session.call_tool(
                "forget",
                {"tag": "employer", "identity": "user123"},
            )
            forgot_text = "\n".join(item.text for item in forgot.content if hasattr(item, "text"))
            assert "Forgot" in forgot_text
