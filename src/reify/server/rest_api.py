"""REST API server for Reify.

This provides a FastAPI-based REST interface for core Reify operations:
remember, recall, forget, export, and identity management.

Per FIRST_PRINCIPLES.md, this adapter is thin and transport-only - it does
not contain unique memory logic absent from the library.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from reify import Reify
from reify.storage import SQLiteStore


class RememberRequest(BaseModel):
    """Request to remember a memory."""

    text: str
    identity_id: str | None = None
    at: str | None = None  # ISO format timestamp

    @field_validator("at")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError as e:
                raise ValueError(f"Invalid ISO format timestamp: {v}") from e
        return v


class RecallRequest(BaseModel):
    """Request to recall memories."""

    query: str
    identity_id: str | None = None
    limit: int = 20
    valid_at: str | None = None  # ISO format timestamp

    @field_validator("valid_at")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError as e:
                raise ValueError(f"Invalid ISO format timestamp: {v}") from e
        return v


class ForgetRequest(BaseModel):
    """Request to forget memories."""

    memory_id: str | None = None
    tag: str | None = None
    before: str | None = None  # ISO format timestamp
    identity_id: str | None = None

    @field_validator("before")
    @classmethod
    def validate_timestamp(cls, v: str | None) -> str | None:
        if v is not None:
            try:
                datetime.fromisoformat(v)
            except ValueError as e:
                raise ValueError(f"Invalid ISO format timestamp: {v}") from e
        return v


class MemoryResponse(BaseModel):
    """Response containing a memory."""

    id: str
    content: str
    created_at: str
    network: str


class RecallResponse(BaseModel):
    """Response containing recall results."""

    query: str
    results: list[MemoryResponse]
    count: int


class RestAPI:
    """REST API server for Reify."""

    def __init__(self, engine: Reify) -> None:
        self._engine = engine
        self._app = FastAPI(title="Reify REST API", version="0.1.0")
        from reify.server.security import harden_app
        harden_app(self._app)
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Setup FastAPI routes."""

        @self._app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self._app.post("/remember")
        async def remember(request: RememberRequest) -> dict[str, Any]:
            at_dt: datetime | None = None
            if request.at:
                at_dt = datetime.fromisoformat(request.at)

            receipt = await self._engine.remember(
                request.text,
                identity_id=request.identity_id,
                at=at_dt,
            )
            return {
                "committed": len(receipt.committed),
                "memory_ids": [m.id for m in receipt.committed],
            }

        @self._app.post("/recall")
        async def recall(request: RecallRequest) -> RecallResponse:
            valid_at_dt: datetime | None = None
            if request.valid_at:
                valid_at_dt = datetime.fromisoformat(request.valid_at)

            response = await self._engine.recall(
                request.query,
                identity_id=request.identity_id,
                limit=request.limit,
                valid_at=valid_at_dt,
            )

            return RecallResponse(
                query=request.query,
                results=[
                    MemoryResponse(
                        id=r.memory.id,
                        content=r.memory.content,
                        created_at=r.memory.created_at.isoformat(),
                        network=r.memory.network,
                    )
                    for r in response.results
                ],
                count=len(response.results),
            )

        @self._app.post("/forget")
        async def forget(request: ForgetRequest) -> dict[str, int]:
            before_dt: datetime | None = None
            if request.before:
                before_dt = datetime.fromisoformat(request.before)

            deleted = await self._engine.forget(
                request.memory_id,
                tag=request.tag,
                before=before_dt,
                identity_id=request.identity_id,
            )
            return {"deleted": deleted}

        @self._app.get("/what-changed")
        async def what_changed(days: int = 7) -> dict[str, Any]:
            transitions = await self._engine.whatchanged(days=days)
            return {
                "transitions": [
                    {
                        "memory_id": t.memory_id,
                        "from_state": t.from_state,
                        "to_state": t.to_state,
                        "at": t.at.isoformat(),
                    }
                    for t in transitions
                ],
                "count": len(transitions),
            }

        @self._app.get("/what-do-you-know")
        async def what_do_you_know(identity_id: str) -> dict[str, Any]:
            knowledge = await self._engine.whatdoyouknow(identity_id=identity_id)
            return {
                "identity_id": identity_id,
                "memories": [
                    {
                        "id": m.id,
                        "content": m.content,
                        "created_at": m.created_at.isoformat(),
                        "network": m.network,
                    }
                    for m in knowledge
                ],
                "count": len(knowledge),
            }

    @property
    def app(self) -> FastAPI:
        """Get the FastAPI application."""
        return self._app


# Create a default app instance for uvicorn
def create_app() -> FastAPI:
    """Create a default REST API app instance."""
    from reify.config import get_settings
    from reify.cli import _engine_config
    from reify.storage import SQLiteStore

    settings = get_settings()
    settings.ensure_dirs()

    store = SQLiteStore(path=settings.db_path)
    engine = Reify(storage=store, config=_engine_config(settings))

    # Start the engine
    import asyncio
    asyncio.run(engine.start())

    app = RestAPI(engine).app

    # Shutdown hook
    @app.on_event("shutdown")
    async def shutdown() -> None:
        await engine.stop()

    return app


__all__ = [
    "RestAPI",
    "create_app",
]
