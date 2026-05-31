"""REST API endpoint tests.

Tests for the FastAPI-based REST interface covering all endpoints:
- /health
- /remember
- /recall
- /forget
- /what-changed
- /what-do-you-know
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient

from reify import Reify, EngineConfig
from reify.server.rest_api import RestAPI
from reify.storage import SQLiteStore


@pytest.fixture
def client():
    """Create a test client for the REST API."""
    with TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "store.db"
        store = SQLiteStore(path=db_path)
        engine = Reify(storage=store, config=EngineConfig(llm=None))

        # Start the engine
        import asyncio
        asyncio.run(engine.start())

        api = RestAPI(engine)
        test_client = TestClient(api.app)

        yield test_client

        # Cleanup
        asyncio.run(engine.stop())


def test_health_endpoint(client: TestClient) -> None:
    """Health endpoint should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_remember_endpoint(client: TestClient) -> None:
    """Remember endpoint should store a memory."""
    response = client.post(
        "/remember",
        json={
            "text": "I'm a Senior Engineer at Acme Corp",
            "identity_id": "user-123",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["committed"] >= 1
    assert len(data["memory_ids"]) >= 1
    assert isinstance(data["memory_ids"][0], str)


def test_remember_with_timestamp(client: TestClient) -> None:
    """Remember endpoint should accept ISO timestamp."""
    timestamp = datetime.now(timezone.utc).isoformat()
    response = client.post(
        "/remember",
        json={
            "text": "I live in Seattle",
            "identity_id": "user-123",
            "at": timestamp,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["committed"] >= 1


def test_remember_invalid_timestamp(client: TestClient) -> None:
    """Remember endpoint should reject invalid timestamp."""
    response = client.post(
        "/remember",
        json={
            "text": "Test memory",
            "at": "invalid-timestamp",
        },
    )
    assert response.status_code == 422  # Validation error


def test_recall_endpoint(client: TestClient) -> None:
    """Recall endpoint should return memories matching query."""
    # First, remember something
    client.post(
        "/remember",
        json={
            "text": "I'm a Senior Engineer at Acme Corp",
            "identity_id": "user-123",
        },
    )

    # Then recall it
    response = client.post(
        "/recall",
        json={
            "query": "What's my title?",
            "identity_id": "user-123",
            "limit": 10,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What's my title?"
    assert data["count"] >= 0
    assert isinstance(data["results"], list)


def test_recall_with_valid_at(client: TestClient) -> None:
    """Recall endpoint should accept valid_at timestamp."""
    # Remember a memory
    client.post(
        "/remember",
        json={
            "text": "I live in Seattle",
            "identity_id": "user-123",
        },
    )

    # Recall with valid_at
    timestamp = datetime.now(timezone.utc).isoformat()
    response = client.post(
        "/recall",
        json={
            "query": "Where do I live?",
            "identity_id": "user-123",
            "valid_at": timestamp,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "Where do I live?"


def test_recall_invalid_timestamp(client: TestClient) -> None:
    """Recall endpoint should reject invalid timestamp."""
    response = client.post(
        "/recall",
        json={
            "query": "test",
            "valid_at": "invalid-timestamp",
        },
    )
    assert response.status_code == 422  # Validation error


def test_forget_by_memory_id(client: TestClient) -> None:
    """Forget endpoint should delete memory by ID."""
    # First, remember something
    remember_response = client.post(
        "/remember",
        json={
            "text": "Temporary memory to forget",
            "identity_id": "user-123",
        },
    )
    memory_id = remember_response.json()["memory_ids"][0]

    # Then forget it
    response = client.post(
        "/forget",
        json={
            "memory_id": memory_id,
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["deleted"] >= 1


def test_forget_by_tag(client: TestClient) -> None:
    """Forget endpoint should delete memories by tag."""
    # Remember memories with tags (if supported)
    client.post(
        "/remember",
        json={
            "text": "Tagged memory 1",
            "identity_id": "user-123",
        },
    )

    response = client.post(
        "/forget",
        json={
            "tag": "test-tag",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "deleted" in data


def test_forget_by_before(client: TestClient) -> None:
    """Forget endpoint should delete memories before timestamp."""
    # Remember a memory
    client.post(
        "/remember",
        json={
            "text": "Old memory",
            "identity_id": "user-123",
        },
    )

    # Forget memories before now
    timestamp = datetime.now(timezone.utc).isoformat()
    response = client.post(
        "/forget",
        json={
            "before": timestamp,
            "identity_id": "user-123",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert "deleted" in data


def test_forget_invalid_timestamp(client: TestClient) -> None:
    """Forget endpoint should reject invalid timestamp."""
    response = client.post(
        "/forget",
        json={
            "before": "invalid-timestamp",
        },
    )
    assert response.status_code == 422  # Validation error


def test_what_changed_endpoint(client: TestClient) -> None:
    """What-changed endpoint should return transitions."""
    response = client.get("/what-changed?days=7")
    assert response.status_code == 200
    data = response.json()
    assert "transitions" in data
    assert "count" in data
    assert isinstance(data["transitions"], list)


def test_what_changed_default_days(client: TestClient) -> None:
    """What-changed endpoint should use default 7 days."""
    response = client.get("/what-changed")
    assert response.status_code == 200
    data = response.json()
    assert "transitions" in data


def test_what_do_you_know_endpoint(client: TestClient) -> None:
    """What-do-you-know endpoint should return all memories for identity."""
    # First, remember something
    client.post(
        "/remember",
        json={
            "text": "I'm a Senior Engineer at Acme Corp",
            "identity_id": "user-123",
        },
    )

    # Then query what we know
    response = client.get("/what-do-you-know?identity_id=user-123")
    assert response.status_code == 200
    data = response.json()
    assert data["identity_id"] == "user-123"
    assert "memories" in data
    assert "count" in data
    assert isinstance(data["memories"], list)


def test_what_do_you_know_empty_identity(client: TestClient) -> None:
    """What-do-you-know should return empty list for unknown identity."""
    response = client.get("/what-do-you-know?identity_id=unknown-user")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["memories"] == []


def test_remember_empty_text(client: TestClient) -> None:
    """Remember endpoint should handle empty text."""
    response = client.post(
        "/remember",
        json={
            "text": "",
            "identity_id": "user-123",
        },
    )
    # Should either succeed or fail gracefully
    assert response.status_code in [200, 422]


def test_recall_empty_query(client: TestClient) -> None:
    """Recall endpoint should handle empty query."""
    response = client.post(
        "/recall",
        json={
            "query": "",
            "identity_id": "user-123",
        },
    )
    # Should either succeed or fail gracefully
    assert response.status_code in [200, 422]


def test_forget_no_params(client: TestClient) -> None:
    """Forget endpoint should handle no parameters."""
    response = client.post(
        "/forget",
        json={},
    )
    # Should either succeed or fail gracefully
    assert response.status_code in [200, 422]


def test_404_endpoint(client: TestClient) -> None:
    """Non-existent endpoint should return 404."""
    response = client.get("/non-existent")
    assert response.status_code == 404


def test_method_not_allowed(client: TestClient) -> None:
    """Wrong HTTP method should return 405."""
    response = client.get("/remember")  # Should be POST
    assert response.status_code == 405
