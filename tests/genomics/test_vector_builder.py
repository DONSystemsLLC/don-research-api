from __future__ import annotations

from pathlib import Path

import numpy as np

from don_research.genomics.vector_builder import build_vectors_from_h5ad


def test_build_vectors_cluster_mode(h5ad_file: str) -> None:
    vectors = build_vectors_from_h5ad(h5ad_file, mode="cluster")
    assert len(vectors) == 2
    for item in vectors:
        assert len(item["psi"]) == 128
        assert item["type"] == "cluster"
        assert "cluster" in item["meta"]


def test_build_vectors_cell_mode(h5ad_file: str) -> None:
    vectors = build_vectors_from_h5ad(h5ad_file, mode="cell")
    assert len(vectors) == 24
    sample = vectors[0]
    assert len(sample["psi"]) == 128
    entropy = np.asarray(sample["psi"][0:16])
    assert entropy.sum() > 0