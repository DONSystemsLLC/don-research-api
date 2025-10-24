from __future__ import annotations

import hashlib
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import anndata as ad
    import scanpy as sc
except Exception:  # pragma: no cover
    ad = None
    sc = None

try:
    from sklearn.metrics import silhouette_score
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover
    silhouette_score = None
    NearestNeighbors = None


def _safe_log(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def _entropy_signature(values: np.ndarray, bins: int = 16) -> np.ndarray:
    if values.ndim > 1:
        values = values.ravel()
    if not np.any(np.isfinite(values)):
        return np.zeros(bins, dtype=np.float32)
    hist, _ = np.histogram(values[~np.isnan(values)], bins=bins)
    hist = hist.astype(np.float32)
    total = hist.sum()
    return hist / total if total > 0 else hist


def _percent_mito(adata, layer: Optional[str] = None) -> np.ndarray:
    genes = adata.var_names.str.upper()
    mito_mask = genes.str.startswith(("MT-", "MT."))
    matrix = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    mito = np.asarray(matrix[:, mito_mask]).sum(axis=1)
    total = np.asarray(matrix).sum(axis=1) + 1e-12
    return (mito / total).astype(np.float32)


def _hvg_fraction(adata, n_top: int = 2000) -> float:
    try:
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor="seurat_v3", inplace=True)
        return float(np.mean(adata.var["highly_variable"].values))
    except Exception:  # pragma: no cover
        return 0.0


def _neighborhood_purity(embedding: np.ndarray, labels: np.ndarray, k: int = 15) -> float:
    if NearestNeighbors is None or embedding.shape[0] < k + 1:
        return 0.0
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(embedding)
    indices = nn.kneighbors(return_distance=False)
    neighbors = labels[indices[:, 1:]]
    purity = (neighbors == labels[:, None]).mean()
    return float(purity)


def _silhouette(embedding: np.ndarray, labels: np.ndarray) -> float:
    if silhouette_score is None:
        return -1.0
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2 or len(unique_labels) > embedding.shape[0] - 1:
        return -1.0
    try:
        # Use float64 to avoid RuntimeWarnings from BLAS matmul on float32 inputs.
        embedding64 = embedding.astype(np.float64, copy=False)
        sample_count = embedding.shape[0]
        if sample_count > 5000:
            selection = np.random.choice(sample_count, 5000, replace=False)
            return float(silhouette_score(embedding64[selection], labels[selection]))
        return float(silhouette_score(embedding64, labels))
    except Exception:  # pragma: no cover
        return -1.0


def _marker_tokens(adata, cluster: str, top_n: int = 20, label_key: str = "cluster") -> List[str]:
    try:
        if "highly_variable" not in adata.var.columns:
            sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3", inplace=True)
        mask = adata.obs[label_key].astype(str).values == cluster
        matrix = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        mean_in = np.asarray(matrix[mask]).mean(axis=0)
        mean_out = np.asarray(matrix[~mask]).mean(axis=0) + 1e-9
        logfc = np.log2((mean_in + 1e-9) / mean_out)
        top_idx = np.argsort(logfc)[::-1][:top_n]
        return [f"GENE:{gene}" for gene in adata.var_names[top_idx].tolist()]
    except Exception:  # pragma: no cover
        return []


def _hash_bucket(token: str, dim: int) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % dim


def _hashed_bigrams(tokens: Iterable[str], out_dim: int = 100) -> np.ndarray:
    vector = np.zeros(out_dim, dtype=np.float32)
    token_list = list(tokens)
    for idx in range(len(token_list) - 1):
        bigram = token_list[idx] + "||" + token_list[idx + 1]
        bucket = _hash_bucket(bigram, out_dim)
        vector[bucket] += 1.0
    total = vector.sum()
    return vector / total if total > 0 else vector


def _choose_embedding(adata) -> Tuple[np.ndarray, str]:
    for key in ("X_scVI", "X_scvi", "X_totalVI", "X_pca", "X_pca_harmony"):
        if key in getattr(adata, "obsm", {}):
            return np.array(adata.obsm[key]), key
    sc.pp.pca(adata, n_comps=32)
    return np.array(adata.obsm["X_pca"]), "X_pca"


def _default_labels(adata) -> Tuple[np.ndarray, str]:
    for key in ("cluster", "leiden", "louvain", "cell_type"):
        if key in adata.obs.columns:
            return adata.obs[key].astype(str).values, key
    sc.pp.neighbors(adata, use_rep=None)
    sc.tl.leiden(adata, key_added="leiden_tmp", resolution=1.0)
    return adata.obs["leiden_tmp"].astype(str).values, "leiden_tmp"


def build_vectors_from_h5ad(
    h5ad_path: str,
    mode: str = "cluster",
    pathway_scores_key: Optional[str] = None,
    layer: Optional[str] = None,
    out_dim: int = 128,
) -> List[Dict]:
    if ad is None or sc is None:
        raise RuntimeError("anndata/scanpy not available. Please `pip install scanpy anndata`.")

    adata = ad.read_h5ad(h5ad_path)
    matrix = adata.layers[layer] if layer and layer in adata.layers else adata.X
    if "n_counts" not in adata.obs.columns:
        dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        adata.obs["n_counts"] = np.asarray(dense).sum(axis=1)
    adata.obs["pct_mito"] = _percent_mito(adata, layer=layer)

    embedding, embedding_key = _choose_embedding(adata)
    labels, label_key = _default_labels(adata)

    purity = _neighborhood_purity(embedding, labels)
    silhouette_val = _silhouette(embedding, labels)
    hvg_frac = _hvg_fraction(adata)

    outputs: List[Dict] = []

    if mode == "cell":
        dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        for idx in range(adata.n_obs):
            entropy_signature = _entropy_signature(np.asarray(dense[idx]), bins=16)
            metrics = np.array(
                [
                    hvg_frac,
                    float(adata.obs["pct_mito"].iloc[idx]),
                    float(adata.obs["n_counts"].iloc[idx]),
                    1.0,
                    silhouette_val,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    purity,
                    float(adata.obs.get("velocity_confidence", pd.Series([-1] * adata.n_obs)).iloc[idx]),
                    float(adata.obs.get("doublet_score", pd.Series([-1] * adata.n_obs)).iloc[idx]),
                ],
                dtype=np.float32,
            )

            tokens = [
                f"CELLTYPE:{adata.obs.get('cell_type', pd.Series(['NA'] * adata.n_obs)).iloc[idx]}",
                f"TISSUE:{adata.obs.get('tissue', pd.Series(['NA'] * adata.n_obs)).iloc[idx]}",
            ]
            bio = _hashed_bigrams(tokens, out_dim=(out_dim - 28))
            vector = np.zeros(out_dim, dtype=np.float32)
            vector[0:16] = entropy_signature
            vector[16:28] = metrics
            vector[28:] = bio

            outputs.append(
                {
                    "vector_id": f"{os.path.basename(h5ad_path)}:cell:{idx}",
                    "psi": vector.tolist(),
                    "space": embedding_key,
                    "metric": "cosine",
                    "type": "cell",
                    "meta": {
                        "file": os.path.basename(h5ad_path),
                        "cluster": str(labels[idx]),
                        "cell_index": int(idx),
                        "cell_type": str(adata.obs.get("cell_type", pd.Series(["NA"] * adata.n_obs)).iloc[idx]),
                        "tissue": str(adata.obs.get("tissue", pd.Series(["NA"] * adata.n_obs)).iloc[idx]),
                    },
                }
            )

    else:
        dense = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        obs_df = adata.obs.copy()
        obs_df["_label"] = labels.astype(str)
        for cluster in sorted(obs_df["_label"].unique().tolist()):
            mask = (obs_df["_label"] == cluster).values
            mean_expr = np.asarray(dense[mask]).mean(axis=0)
            entropy_signature = _entropy_signature(mean_expr, bins=16)

            cells_in_cluster = int(mask.sum())
            median_umi = float(np.median(adata.obs["n_counts"].values[mask])) if cells_in_cluster > 0 else 0.0
            pct_mito_med = float(np.median(adata.obs["pct_mito"].values[mask])) if cells_in_cluster > 0 else 0.0

            try:
                mean_in = np.asarray(dense[mask]).mean(axis=0)
                mean_out = np.asarray(dense[~mask]).mean(axis=0) + 1e-9
                logfc = np.log2((mean_in + 1e-9) / mean_out)
                de_strength = float(np.mean(np.sort(np.abs(logfc))[::-1][:200]))
            except Exception:  # pragma: no cover
                de_strength = 0.0

            metrics = np.array(
                [
                    hvg_frac,
                    pct_mito_med,
                    median_umi,
                    float(cells_in_cluster),
                    silhouette_val,
                    0.0,
                    0.0,
                    de_strength,
                    0.0,
                    purity,
                    -1.0,
                    float(
                        np.median(
                            adata.obs.get("doublet_score", pd.Series([-1] * adata.n_obs)).values[mask]
                        )
                    ),
                ],
                dtype=np.float32,
            )

            tokens = _marker_tokens(adata, cluster=cluster, top_n=20, label_key=label_key)
            if cells_in_cluster > 0:
                celltype = pd.Series(
                    adata.obs.get("cell_type", pd.Series(["NA"] * adata.n_obs)).values[mask]
                ).mode().iloc[0]
                tissue = pd.Series(
                    adata.obs.get("tissue", pd.Series(["NA"] * adata.n_obs)).values[mask]
                ).mode().iloc[0]
            else:
                celltype = "NA"
                tissue = "NA"
            tokens.extend([f"CELLTYPE:{celltype}", f"TISSUE:{tissue}"])

            bio = _hashed_bigrams(tokens, out_dim=(out_dim - 28))
            vector = np.zeros(out_dim, dtype=np.float32)
            vector[0:16] = entropy_signature
            vector[16:28] = metrics
            vector[28:] = bio

            outputs.append(
                {
                    "vector_id": f"{os.path.basename(h5ad_path)}:cluster:{cluster}",
                    "psi": vector.tolist(),
                    "space": embedding_key,
                    "metric": "cosine",
                    "type": "cluster",
                    "meta": {
                        "file": os.path.basename(h5ad_path),
                        "cluster": cluster,
                        "cells": cells_in_cluster,
                        "cell_type": celltype,
                        "tissue": tissue,
                    },
                }
            )

    return outputs
