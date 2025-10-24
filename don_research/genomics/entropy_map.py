from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
except Exception:  # pragma: no cover
    ad = None

try:
    import scanpy as sc  # type: ignore
except Exception:  # pragma: no cover
    sc = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import networkx as nx  # noqa: F401
except Exception:  # pragma: no cover
    nx = None  # noqa: F841


def _to_dense_array(matrix) -> np.ndarray:
    dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
    return np.asarray(dense, dtype=np.float32)


def _ensure_embedding(adata, requested_key: Optional[str]) -> Tuple[np.ndarray, str]:
    candidate_keys = [requested_key] if requested_key else []
    candidate_keys += ["X_scVI", "X_scvi", "X_totalVI", "X_umap", "X_tsne", "X_pca", "X_pca_harmony"]
    for key in candidate_keys:
        if key and key in getattr(adata, "obsm", {}):
            return np.asarray(adata.obsm[key]), key

    matrix = _to_dense_array(adata.X)
    n_components = min(32, matrix.shape[1]) if matrix.ndim == 2 else 2

    if sc is not None:  # pragma: no cover - exercised when real scanpy available
        sc.pp.pca(adata, n_comps=n_components)
        return np.asarray(adata.obsm["X_pca"]), "X_pca"

    from sklearn.decomposition import PCA

    pca = PCA(n_components=max(2, n_components))
    embedding = pca.fit_transform(matrix)
    if embedding.shape[1] < 2:
        embedding = np.pad(embedding, ((0, 0), (0, 2 - embedding.shape[1])), mode="constant")
    adata.obsm["X_pca"] = embedding.astype(np.float32)
    return adata.obsm["X_pca"], "X_pca"


def _ensure_labels(adata, requested_key: Optional[str]) -> Tuple[np.ndarray, str]:
    if requested_key and requested_key in adata.obs.columns:
        return adata.obs[requested_key].astype(str).values, requested_key

    for key in ("cell_type", "cluster", "leiden", "louvain"):
        if key in adata.obs.columns:
            return adata.obs[key].astype(str).values, key

    if sc is not None:  # pragma: no cover - exercised when real scanpy available
        sc.pp.neighbors(adata, use_rep=None)
        sc.tl.leiden(adata, key_added="leiden_tmp", resolution=1.0)
        return adata.obs["leiden_tmp"].astype(str).values, "leiden_tmp"

    embedding, _ = _ensure_embedding(adata, None)
    clusters = max(2, min(adata.n_obs // 4, 10))
    from sklearn.cluster import KMeans

    labels = KMeans(n_clusters=clusters, n_init="auto", random_state=0).fit_predict(embedding)
    adata.obs["leiden_tmp"] = labels.astype(str)
    return adata.obs["leiden_tmp"].values, "leiden_tmp"


def _build_knn(embedding: np.ndarray, k: int = 15) -> np.ndarray:
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(embedding)
    indices = nn.kneighbors(return_distance=False)
    return indices[:, 1:]


def _neighbor_label_entropy(labels: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    entropy = np.zeros(labels.shape[0], dtype=np.float32)
    for idx in range(labels.shape[0]):
        neighbors = labels[knn_idx[idx]]
        values, counts = np.unique(neighbors, return_counts=True)
        probabilities = counts / counts.sum()
        entropy[idx] = -np.sum(probabilities * np.log(probabilities + 1e-12))
    entropy /= np.log(len(np.unique(labels))) + 1e-12
    return entropy


def _density_scores(embedding: np.ndarray, knn_idx: np.ndarray) -> np.ndarray:
    neighbors = embedding[knn_idx]
    diffs = embedding[:, None, :] - neighbors
    distances = np.linalg.norm(diffs, axis=2)
    mean_distance = np.mean(distances, axis=1)
    return 1.0 / (mean_distance + 1e-6)


def generate_entropy_map(
    h5ad_path: str,
    label_key: Optional[str] = None,
    emb_key: Optional[str] = None,
    k: int = 15,
    out_png: Optional[str] = None,
) -> Tuple[Optional[str], Dict]:
    if ad is None:
        raise RuntimeError("anndata required.")

    adata = ad.read_h5ad(h5ad_path)
    labels, resolved_label_key = _ensure_labels(adata, label_key)
    embedding, resolved_emb_key = _ensure_embedding(adata, emb_key)

    knn_idx = _build_knn(embedding, k=k)
    entropy = _neighbor_label_entropy(labels, knn_idx)
    density = _density_scores(embedding, knn_idx)
    collapse_score = (entropy * density).astype(np.float32)

    stats = {
        "cells": int(adata.n_obs),
        "n_cells": int(adata.n_obs),
        "label_key": resolved_label_key,
        "embedding_key": resolved_emb_key,
        "entropy_mean": float(entropy.mean()),
        "entropy_std": float(entropy.std()),
        "collapse_mean": float(collapse_score.mean()),
        "collapse_std": float(collapse_score.std()),
        "neighbors_k": int(k),
    }

    png_path = None
    if out_png and plt is not None:
        selection = np.arange(adata.n_obs)
        if adata.n_obs > 4000:
            selection = np.random.choice(adata.n_obs, 4000, replace=False)
        if "X_umap" in adata.obsm:
            x_coords, y_coords = adata.obsm["X_umap"][selection, 0], adata.obsm["X_umap"][selection, 1]
        elif "X_tsne" in adata.obsm:
            x_coords, y_coords = adata.obsm["X_tsne"][selection, 0], adata.obsm["X_tsne"][selection, 1]
        else:
            x_coords, y_coords = embedding[selection, 0], embedding[selection, 1]

        figure, axis = plt.subplots(1, 1, figsize=(7, 6), dpi=160)
        scatter = axis.scatter(x_coords, y_coords, c=collapse_score[selection], cmap="magma", s=6, alpha=0.9)
        figure.colorbar(scatter, ax=axis, label="collapse score (entropy * density)")
        axis.set_title(f"Entropy halo map ({os.path.basename(h5ad_path)})")
        axis.set_xticks([])
        axis.set_yticks([])
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        figure.tight_layout()
        figure.savefig(out_png)
        plt.close(figure)
        png_path = out_png

    return png_path, stats
