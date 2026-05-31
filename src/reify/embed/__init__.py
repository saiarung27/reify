"""Embedding provider interface.

The default implementation uses ``fastembed`` for local, CPU-friendly
embeddings (BAAI/bge-small-en-v1.5, 384 dimensions). It is the right default
for a local-first system; cloud-hosted providers can be added by implementing
:class:`Embedder`.
"""

from reify.embed.base import Embedder

__all__ = ["Embedder"]
