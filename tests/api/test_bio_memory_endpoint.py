from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from main import app
from src.don_memory.dependencies import get_trace_storage
from src.don_memory.trace_storage import TraceStorage


def test_memory_endpoint_returns_latest_first(api_client: TestClient, tmp_path, monkeypatch):
    storage = TraceStorage(db_url=f"sqlite:///{tmp_path/'mem.db'}")

    def _override_storage() -> TraceStorage:
        return storage

    app.dependency_overrides[get_trace_storage] = _override_storage

    trace_ids = []
    for event_type in ["compress", "qac_fit", "qac_apply"]:
        payload = {
            "id": uuid4().hex,
            "project_id": "proj-42",
            "user_id": "user-7",
            "event_type": event_type,
            "status": "succeeded",
            "metrics": {"event": event_type},
            "artifacts": {},
            "engine_used": event_type,
            "health": {"don_stack": {"healthy": True}},
            "seed": 11,
            "started_at": "2025-10-24T00:00:00Z",
            "finished_at": "2025-10-24T00:00:01Z",
        }
        trace_ids.append(storage.store_trace(payload))

    response = api_client.get(
        "/api/v1/bio/memory/proj-42",
        headers={"Authorization": "Bearer demo_token"},
    )
    assert response.status_code == 200
    data = response.json()
    events = [trace["event_type"] for trace in data["traces"]]
    assert events == ["qac_apply", "qac_fit", "compress"]

    app.dependency_overrides.pop(get_trace_storage, None)
