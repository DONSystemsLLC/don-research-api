from __future__ import annotations

import warnings

import pytest

from don_research.genomics.vector_builder import build_vectors_from_h5ad


def _parse_version(version: str) -> tuple[int, ...]:
    parts = []
    normalized = version.replace("+", ".").replace("-", ".")
    for token in normalized.split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            break
    return tuple(parts)


def test_scanpy_numpy2_compatible(small_adata) -> None:
    try:
        import scanpy as sc  # type: ignore
    except Exception as exc:  # pragma: no cover - ensures informative failure
        pytest.fail(
            f"scanpy import failed: {exc!r}. Please install a NumPy 2 compatible version.",
        )

    minimum = (1, 10, 4)
    assert _parse_version(sc.__version__) >= minimum, (
        "scanpy version is too old. Expected >= 1.10.4 for NumPy 2 compatibility, "
        f"found {sc.__version__}."
    )

    adata = small_adata.copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        sc.pp.pca(adata, n_comps=min(5, adata.n_vars))
    assert "X_pca" in adata.obsm


@pytest.mark.parametrize("mode", ["cluster", "cell"])
def test_build_vectors_without_runtime_warnings(h5ad_file: str, mode: str) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=RuntimeWarning)
        warnings.filterwarnings("error", category=RuntimeWarning, module="sklearn.utils.extmath")
        build_vectors_from_h5ad(h5ad_file, mode=mode)