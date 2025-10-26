from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from main import app
from src.don_memory.dependencies import get_trace_storage
from src.don_memory.trace_storage import TraceStorage


def test_compress_returns_trace_id(api_client: TestClient, tmp_path, monkeypatch):
    storage = TraceStorage(db_url=f"sqlite:///{tmp_path/'trace.db'}")

    def _override_storage() -> TraceStorage:
        return storage

    app.dependency_overrides[get_trace_storage] = _override_storage

    payload = {
        "data": {
            "gene_names": ["G1", "G2"],
            "expression_matrix": np.array([[1.0, 2.0], [3.0, 4.0]]).tolist(),
            "cell_metadata": [],
        },
        "compression_target": 2,
        "project_id": "proj-99",
        "user_id": "user-21",
        "seed": 123,
        "stabilize": False,
    }

    response = api_client.post(
        "/api/v1/genomics/compress",
        json=payload,
        headers={"Authorization": "Bearer demo_token"},
    )
    assert response.status_code == 200
    body = response.json()
    assert "trace_id" in body

    traces = storage.list_traces(project_id="proj-99")
    assert traces[0]["id"] == body["trace_id"]
    assert traces[0]["event_type"] == "compress"

    app.dependency_overrides.pop(get_trace_storage, None)
