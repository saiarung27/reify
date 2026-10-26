"""Adapters that let Reify stand in for other memory backends.

The first adapter is :class:`ReifyZepGraph`, a Zep Cloud graph-shaped
surface backed by local Reify stores. It exists so a Zep-based caller
(MiroFish) can swap its memory layer without a cloud vendor.
"""

from reify.adapters.zep_graph import ReifyZepGraph

__all__ = ["ReifyZepGraph"]
