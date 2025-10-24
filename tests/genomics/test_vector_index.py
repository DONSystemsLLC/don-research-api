from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from don_research.genomics.index import VectorIndex


def _mock_items() -> list[dict]:
    base = np.eye(3, dtype=np.float32)
    return [
        {"vector_id": f"vec_{i}", "psi": base[i].tolist(), "meta": {"cluster": str(i)} }
        for i in range(3)
    ]


def test_vector_index_numpy_search() -> None:
    index = VectorIndex(dim=3, metric="cosine")
    index.build(_mock_items())
    hits = index.search([1.0, 0.0, 0.0], k=2)
    assert hits[0]["vector_id"] == "vec_0"
    assert hits[0]["rank"] == 1
    assert hits[1]["rank"] == 2


def test_vector_index_filters(tmp_path) -> None:
    items = _mock_items()
    path = tmp_path / "vectors.jsonl"
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item) + "\n")
    index = VectorIndex.load_jsonl(str(path), metric="cosine")
    hits = index.search([0.0, 0.0, 1.0], k=3, filters={"cluster": "2"})
    assert len(hits) == 1
    assert hits[0]["vector_id"] == "vec_2"