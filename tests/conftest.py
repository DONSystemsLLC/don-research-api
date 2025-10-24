from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Dict, Iterator

import anndata as ad
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from don_research.api import genomics_router
from don_research.genomics import vector_builder
from main import app, usage_tracker, verify_token


if vector_builder.ad is None:  # pragma: no cover - ensure tests have anndata access
    import anndata as _ad

    vector_builder.ad = _ad


if vector_builder.sc is None:  # pragma: no cover - provide minimal scanpy stub for tests
    class _StubScanpy:
        class pp:
            @staticmethod
            def highly_variable_genes(adata, *args, **kwargs) -> None:
                if "highly_variable" not in adata.var.columns:
                    adata.var["highly_variable"] = np.zeros(adata.n_vars, dtype=bool)

            @staticmethod
            def pca(adata, n_comps: int = 32, *_, **__) -> None:
                if "X_pca" in adata.obsm:
                    return
                matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
                comps = min(n_comps, matrix.shape[1])
                adata.obsm["X_pca"] = matrix[:, :comps].astype(np.float32)

            @staticmethod
            def neighbors(adata, *_, **__) -> None:
                return None

        class tl:
            @staticmethod
            def leiden(adata, key_added: str = "leiden_tmp", *_, **__) -> None:
                if key_added in adata.obs:
                    return
                default = adata.obs.get("cluster", pd.Series(["0"] * adata.n_obs)).astype(str).values
                adata.obs[key_added] = default

    vector_builder.sc = _StubScanpy()


@pytest.fixture(scope="session")
def small_adata() -> ad.AnnData:
    rng = np.random.default_rng(42)
    matrix = rng.poisson(lam=5.0, size=(24, 12)).astype(np.float32)
    var_names = [f"GENE{i}" for i in range(matrix.shape[1])]
    obs_index = pd.Index([f"cell_{i}" for i in range(matrix.shape[0])], dtype=object)
    obs = pd.DataFrame(
        {
            "cluster": ["A"] * 12 + ["B"] * 12,
            "cell_type": ["T-cell"] * 12 + ["B-cell"] * 12,
            "tissue": ["PBMC"] * matrix.shape[0],
        },
        index=obs_index,
    )
    var_index = pd.Index(var_names, dtype=object)
    adata = ad.AnnData(X=matrix, obs=obs, var=pd.DataFrame(index=var_index))
    adata.var_names = pd.Index(var_names, dtype=object)
    adata.obsm["X_pca"] = rng.normal(size=(matrix.shape[0], 8)).astype(np.float32)
    adata.var["highly_variable"] = np.ones(matrix.shape[1], dtype=bool)
    return adata


@pytest.fixture
def h5ad_file(tmp_path, small_adata: ad.AnnData) -> str:
    path = tmp_path / "mock.h5ad"
    small_adata.write_h5ad(path)
    return str(path)


@pytest.fixture
def raw_h5ad_bytes(small_adata: ad.AnnData) -> bytes:
    buffer = io.BytesIO()
    small_adata.write_h5ad(buffer)
    return buffer.getvalue()


@pytest.fixture
def api_client(tmp_path, monkeypatch) -> Iterator[TestClient]:
    tmp_str = str(tmp_path)
    monkeypatch.setattr(genomics_router, "DATA_DIR", tmp_str)
    monkeypatch.setattr(genomics_router, "VEC_DIR", os.path.join(tmp_str, "vectors"))
    monkeypatch.setattr(genomics_router, "MAP_DIR", os.path.join(tmp_str, "maps"))
    os.makedirs(genomics_router.VEC_DIR, exist_ok=True)
    os.makedirs(genomics_router.MAP_DIR, exist_ok=True)

    def _allow() -> Dict[str, str]:
        return {"name": "Test Lab", "rate_limit": 1000}

    app.dependency_overrides[verify_token] = _allow
    usage_tracker.clear()
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.pop(verify_token, None)
        usage_tracker.clear()
