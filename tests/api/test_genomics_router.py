from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from don_research.api import genomics_router


def _write_fixture_h5ad(small_adata, filename: str) -> Path:
    path = Path(genomics_router.DATA_DIR) / filename
    small_adata.write_h5ad(path)
    return path


def test_load_endpoint_local_path(api_client, small_adata) -> None:
    path = _write_fixture_h5ad(small_adata, "fixture.h5ad")
    response = api_client.post(
        "/api/v1/genomics/load",
        data={"accession_or_path": str(path)},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["h5ad_path"] == str(path.resolve())


def test_vectors_build_and_search_flow(api_client, small_adata) -> None:
    path = _write_fixture_h5ad(small_adata, "build.h5ad")
    with open(path, "rb") as handle:
        response = api_client.post(
            "/api/v1/genomics/vectors/build",
            files={"file": ("build.h5ad", handle, "application/octet-stream")},
            data={"mode": "cluster"},
        )
    assert response.status_code == 200
    payload = response.json()
    jsonl_path = Path(payload["jsonl"])
    assert jsonl_path.exists()

    preview_vector = payload["preview"][0]["psi"]
    search = api_client.post(
        "/api/v1/genomics/vectors/search",
        data={
            "jsonl_path": str(jsonl_path),
            "psi": json.dumps(preview_vector),
            "k": 2,
        },
    )
    hits = search.json()["hits"]
    assert len(hits) == 2
    assert hits[0]["rank"] == 1


def test_query_encode_endpoint(api_client, small_adata) -> None:
    path = _write_fixture_h5ad(small_adata, "encode.h5ad")
    response = api_client.post(
        "/api/v1/genomics/query/encode",
        data={
            "text": "Interferon gamma PBMC T-cell",
            "gene_list_json": json.dumps(["IFNG", "GZMB"]),
            "h5ad_path": str(path),
        },
    )
    payload = response.json()
    psi = np.asarray(payload["psi"], dtype=np.float32)
    assert psi.shape == (128,)
    assert np.isclose(np.linalg.norm(psi), 1.0, atol=1e-6)


def test_entropy_map_endpoint(api_client, small_adata) -> None:
    path = _write_fixture_h5ad(small_adata, "entropy.h5ad")
    with open(path, "rb") as handle:
        response = api_client.post(
            "/api/v1/genomics/entropy-map",
            files={"file": ("entropy.h5ad", handle, "application/octet-stream")},
        )
    payload = response.json()
    assert response.status_code == 200
    png_path = Path(payload["png"])
    assert png_path.exists()
    assert payload["stats"]["cells"] > 0