"""STATE-Bench memory adapter.

STATE-Bench is an agent framework benchmark that requires a bring-your-own-memory
interface. This module provides an adapter to plug Reify into STATE-Bench.

To use:
1. Clone STATE-Bench: git clone https://github.com/microsoft/STATE-Bench
2. Copy this file to STATE-Bench/agents/reify_agent.py
3. Configure .env with Reify settings
4. Run: uv run python -m state_bench.scripts.run_batch --agent-class ReifyMemoryAgent ...

STATE-Bench repository: https://github.com/microsoft/STATE-Bench
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class ReifyMemoryAgent:
    """Reify memory adapter for STATE-Bench.
    
    This implements the STATE-Bench memory interface:
    - retrieve_learnings(query, top_k=3) -> list[str]
    
    Note: This is a stub that demonstrates the interface. Full integration
    requires running STATE-Bench as a separate process and calling Reify
    via its CLI or Python API.
    """
    
    def __init__(self) -> None:
        # In a full implementation, this would initialize a Reify engine
        # and configure it with the appropriate settings
        self._learnings_path = Path.home() / ".reify" / "statebench_learnings.json"
        
    def retrieve_learnings(self, query: str, top_k: int = 3) -> list[str]:
        """Retrieve relevant memories for the given query.
        
        Args:
            query: The query to retrieve memories for
            top_k: Number of memories to return
            
        Returns:
            List of memory strings
        """
        # Stub implementation - in full version, this would call Reify.recall()
        # For now, return empty list to demonstrate interface
        return []
    
    def store_learning(self, learning: str) -> None:
        """Store a learning/memory.
        
        Args:
            learning: The learning to store
        """
        # Stub implementation - in full version, this would call Reify.remember()
        learnings = []
        if self._learnings_path.exists():
            learnings = json.loads(self._learnings_path.read_text())
        learnings.append(learning)
        self._learnings_path.parent.mkdir(parents=True, exist_ok=True)
        self._learnings_path.write_text(json.dumps(learnings))


__all__ = [
    "ReifyMemoryAgent",
]
