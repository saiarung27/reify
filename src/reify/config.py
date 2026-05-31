"""Runtime configuration for Reify.

Configuration is read from environment variables and an optional ``.env`` file
at the repo root. Every value has a sensible default so that ``reify`` runs
locally with zero configuration after installing optional extras.

The settings object is constructed exactly once per process and cached.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ReifySettings(BaseSettings):
    """Process-wide settings.

    All values can be overridden via environment variables prefixed with
    ``REIFY_``. See ``.env.example`` for the full surface.
    """

    model_config = SettingsConfigDict(
        env_prefix="REIFY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------

    home: Path = Field(default=Path.home() / ".reify")
    """Root directory for all Reify state. Defaults to ``~/.reify``."""

    @property
    def db_path(self) -> Path:
        """Absolute path to the SQLite store."""
        return self.home / "store.db"

    @property
    def cache_dir(self) -> Path:
        """Directory for transient caches (downloaded models, eval data)."""
        return self.home / "cache"

    # ------------------------------------------------------------------
    # LLM providers (provider:model strings; parsed by the LLM registry)
    # ------------------------------------------------------------------

    llm_extract: str = "openai:gpt-5-mini"
    """Model used by the extractor in the write path. Should be cheap+fast."""

    llm_reconsolidate: str = "openai:gpt-5"
    """Model used by Active Reconsolidation on Write. Should be capable."""

    llm_intent: str = "openai:gpt-5-nano"
    """Model used by the retrieval intent classifier. Should be very fast."""

    # ------------------------------------------------------------------
    # Embeddings + reranker
    # ------------------------------------------------------------------

    embed_model: str = "BAAI/bge-small-en-v1.5"
    """``fastembed`` model identifier. 384-dim by default."""

    rerank_model: str = "BAAI/bge-reranker-base"

    # ------------------------------------------------------------------
    # Server
    # ------------------------------------------------------------------

    http_host: str = "127.0.0.1"
    http_port: int = 7878

    # ------------------------------------------------------------------
    # Behavior
    # ------------------------------------------------------------------

    arw_top_k: int = 8
    """How many related memories ARW considers when classifying relations."""

    arw_min_similarity: float = 0.55
    """Floor for considering a candidate related during ARW."""

    identity_bond_threshold: float = 0.8
    """Default minimum bond confidence for cross-identity inheritance."""

    log_level: str = "INFO"
    default_ttl_days: float = 30.0
    reinforcement_factor: float = 1.5
    decay_sweep_interval_seconds: float = 60.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def ensure_dirs(self) -> None:
        """Create ``home`` and ``cache_dir`` if they do not exist.

        Idempotent. Called by :func:`reify.cli.init` and by long-running
        servers on startup.
        """
        self.home.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> ReifySettings:
    """Return the process-wide :class:`ReifySettings` singleton."""
    return ReifySettings()
