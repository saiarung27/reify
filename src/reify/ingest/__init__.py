"""Ingestion subsystem (v0.1).

Two pieces, modelled on Mastra's Observational Memory pattern but decoupled
from the Mastra framework:

* **Observer** — passively watches incoming events (chat turns, tool calls,
  observations) and emits raw, time-stamped records.
* **Reflector** — runs after a session boundary or window and produces
  :class:`reify.core.types.Belief` records via dialectic inference.

Implementations are deferred to v0.1.
"""
