from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from src.don_memory.trace_storage import TraceStorage


@pytest.fixture()
def storage(tmp_path) -> TraceStorage:
    db_path = tmp_path / "memory.db"
    return TraceStorage(db_url=f"sqlite:///{db_path}")


def _sample_trace(project_id: str = "proj-1", user_id: str = "user-9", **extra):
    base = {
        "id": uuid4().hex,
        "project_id": project_id,
        "user_id": user_id,
        "event_type": "compress",
        "status": "succeeded",
        "metrics": {"compression_ratio": "32.0x"},
        "artifacts": {},
        "engine_used": "compress",
        "health": {"don_stack": {"don_gpu": True}},
        "seed": 42,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }
    base.update(extra)
    return base


def test_store_trace_and_list(storage: TraceStorage) -> None:
    t1 = _sample_trace()
    t2 = _sample_trace()
    t2["event_type"] = "qac_fit"
    t2["started_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=5)
    ).isoformat()
    t2["finished_at"] = (
        datetime.now(timezone.utc) + timedelta(seconds=6)
    ).isoformat()

    trace_id_1 = storage.store_trace(t1)
    trace_id_2 = storage.store_trace(t2)

    assert trace_id_1 == t1["id"]
    assert trace_id_2 == t2["id"]

    traces = storage.list_traces(project_id="proj-1")
    assert [trace["id"] for trace in traces] == [trace_id_2, trace_id_1]
    assert traces[0]["event_type"] == "qac_fit"


def test_link_edges(storage: TraceStorage) -> None:
    t1 = _sample_trace()
    t2 = _sample_trace(event_type="qac_apply")

    storage.store_trace(t1)
    storage.store_trace(t2)
    storage.link(t1["id"], t2["id"], relation="follows")

    edges = storage.list_edges(source_id=t1["id"])
    assert edges == [
        {
            "source_id": t1["id"],
            "target_id": t2["id"],
            "relation": "follows",
        }
    ]
