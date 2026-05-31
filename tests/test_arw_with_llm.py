"""Test ARW with real LLM (Ollama).

This test requires Ollama to be running with llama3 model installed.
Run with: pytest tests/test_arw_with_llm.py
"""

import pytest

from reify.engine import Reify, EngineConfig
from reify.llm import OllamaProvider
from reify.storage import SQLiteStore


@pytest.mark.asyncio
async def test_arw_with_ollama_llama3():
    """Test ARW with real Ollama LLM (llama3)."""
    # Create Ollama provider
    llm = OllamaProvider(base_url="http://localhost:11434")

    # Create engine with LLM
    config = EngineConfig(llm=llm)
    store = SQLiteStore(path=":memory:")
    engine = Reify(storage=store, config=config)

    async with engine:
        # Add a memory about a title
        receipt = await engine.remember("I'm a Senior Engineer at Acme Corp")
        print(f"Committed {len(receipt.committed)} memories")

        # Flush ARW to process with LLM
        processed = await engine.flush_arw()
        print(f"ARW processed {processed} memories")

        # For now, just verify it doesn't crash
        assert processed >= 0

        # Add a conflicting memory
        receipt2 = await engine.remember("I'm a Staff Engineer at Acme Corp")
        print(f"Committed {len(receipt2.committed)} memories")

        # Flush ARW again
        processed2 = await engine.flush_arw()
        print(f"ARW processed {processed2} memories")

        assert processed2 >= 0


if __name__ == "__main__":
    import asyncio

    asyncio.run(test_arw_with_ollama_llama3())
