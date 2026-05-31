# Reify Python SDK

The official Python SDK for Reify memory engine.

## Installation

```bash
pip install reify
```

## Quick Start

```python
import asyncio
from reify import Reify
from reify.storage import SQLiteStore

async def main():
    # Initialize the engine
    store = SQLiteStore(path="reify.db")
    engine = Reify(storage=store)
    await engine.start()

    try:
        # Remember content
        receipt = await engine.remember("I live in San Francisco")
        print(f"Remembered {len(receipt.committed)} memories")

        # Recall memories
        response = await engine.recall("San Francisco")
        for result in response.results:
            print(f"- {result.memory.content}")
    finally:
        await engine.stop()

asyncio.run(main())
```

## API Reference

### Reify

The main engine class for memory operations.

#### Methods

- `remember(text: str, identity_id: str | None = None) -> WriteReceipt`
  - Ingest content and extract memories

- `recall(query: str, identity_id: str | None = None, limit: int = 20) -> RecallResponse`
  - Retrieve relevant memories for a query

- `forget(memory_id: str | None = None, tag: str | None = None, identity_id: str | None = None) -> int`
  - Soft-delete memories

- `whatchanged(days: int, identity_id: str | None = None) -> tuple[Transition, ...]`
  - Query transitions within a time window

- `whatdoyouknow(identity_id: str) -> tuple[Memory, ...]`
  - Recall all memories for an identity

### Storage

#### SQLiteStore

```python
from reify.storage import SQLiteStore

store = SQLiteStore(path="reify.db")
```

## Configuration

```python
from reify import Reify, EngineConfig
from reify.storage import SQLiteStore

config = EngineConfig(
    default_ttl_days=30,
    reinforcement_factor=2.0,
)

engine = Reify(storage=store, config=config)
```

## License

MIT
