"""Tests for the v0.6 REST API server."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from reify import Reify
from reify.server.rest_api import RestAPI
from reify.storage import SQLiteStore


@pytest.fixture
async def engine() -> Reify:
    store = SQLiteStore(":memory:")
    engine = Reify(storage=store)
    await engine.start()
    yield engine
    await engine.stop()


@pytest.fixture
def api(engine: Reify) -> RestAPI:
    return RestAPI(engine)


def test_rest_health_endpoint(api: RestAPI) -> None:
    client = TestClient(api.app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_rest_remember(api: RestAPI) -> None:
    client = TestClient(api.app)
    response = client.post(
        "/remember",
        json={"text": "I live in San Francisco."},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["committed"] >= 1
    assert len(data["memory_ids"]) == data["committed"]


def test_rest_recall(api: RestAPI) -> None:
    client = TestClient(api.app)
    # First remember something.
    client.post("/remember", json={"text": "I live in San Francisco."})

    # Then recall it.
    response = client.post(
        "/recall",
        json={"query": "San Francisco"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "San Francisco"
    assert data["count"] >= 1
    assert len(data["results"]) == data["count"]


def test_rest_forget_by_id(api: RestAPI) -> None:
    client = TestClient(api.app)
    # Remember something.
    remember_response = client.post("/remember", json={"text": "I live in San Francisco."})
    memory_id = remember_response.json()["memory_ids"][0]

    # Forget it.
    response = client.post("/forget", json={"memory_id": memory_id})
    assert response.status_code == 200
    assert response.json()["deleted"] >= 1


def test_rest_what_changed(api: RestAPI) -> None:
    client = TestClient(api.app)
    response = client.get("/what-changed?days=7")
    assert response.status_code == 200
    data = response.json()
    assert "transitions" in data
    assert "count" in data


def test_rest_what_do_you_know(api: RestAPI) -> None:
    client = TestClient(api.app)
    # Remember something for an identity.
    client.post(
        "/remember",
        json={"text": "I live in San Francisco.", "identity_id": "user:123"},
    )

    # Query what we know.
    response = client.get("/what-do-you-know?identity_id=user:123")
    assert response.status_code == 200
    data = response.json()
    assert data["identity_id"] == "user:123"
    assert "memories" in data
    assert "count" in data
