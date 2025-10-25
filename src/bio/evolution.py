"""
Evolution Tracker - Run-over-Run Stability Analysis
Compares pipeline runs to detect drift and assess reproducibility

Integrates with DON Stack TACE for temporal adjacency collapse analysis
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

logger = logging.getLogger(__name__)

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from sklearn.metrics import silhouette_score, adjusted_rand_score
import logging

logger = logging.getLogger(__name__)


def evolution_report(
    baseline_run: str,
    candidate_run: str,
    artifacts_dir: str = "artifacts"
) -> Dict[str, Any]:
    """
    Compare two analysis runs for stability and improvement
    
    Args:
        baseline_run: Baseline run ID (e.g., "run_2025_10_24_01")
        candidate_run: Candidate run ID to compare
        artifacts_dir: Directory containing run artifacts
        
    Returns:
        Dict with deltas and stability metrics
    """
    artifacts_path = Path(artifacts_dir)
    
    # Load baseline metrics
    baseline_metrics = load_run_metrics(artifacts_path / baseline_run)
    if not baseline_metrics:
        raise ValueError(f"Could not load metrics for baseline run: {baseline_run}")
    
    # Load candidate metrics
    candidate_metrics = load_run_metrics(artifacts_path / candidate_run)
    if not candidate_metrics:
        raise ValueError(f"Could not load metrics for candidate run: {candidate_run}")
    
    # Compute deltas
    deltas = compute_deltas(baseline_metrics, candidate_metrics)
    
    # Assess stability
    stability = assess_stability(deltas)
    
    # Generate notes
    notes = generate_evolution_notes(deltas, stability)
    
    return {
        "baseline_run": baseline_run,
        "candidate_run": candidate_run,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "deltas": deltas,
        "stability": stability,
        "notes": notes,
        "recommendation": generate_recommendation(deltas, stability)
    }


def load_run_metrics(run_path: Path) -> Optional[Dict[str, Any]]:
    """
    Load metrics from a run directory
    Expected files: collapse_map.json, collapse_vectors.jsonl, metrics.json
    """
    if not run_path.exists():
        logger.warning(f"Run path does not exist: {run_path}")
        return None
    
    metrics = {}
    
    # Load from metrics.json if available
    metrics_file = run_path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            stored_metrics = json.load(f)
            metrics.update(stored_metrics)
    
    # Load collapse map for cluster count
    map_file = run_path / "collapse_map.json"
    if map_file.exists():
        with open(map_file, 'r') as f:
            collapse_map = json.load(f)
            metrics["n_clusters"] = collapse_map["metadata"]["n_clusters"]
            metrics["n_cells"] = collapse_map["metadata"]["n_cells"]
            metrics["n_edges"] = len(collapse_map["edges"])
    
    # Load vectors for detailed metrics if needed
    vectors_file = run_path / "collapse_vectors.jsonl"
    if vectors_file.exists() and "silhouette_score" not in metrics:
        from .adapter_anndata import stream_jsonl
        vectors = stream_jsonl(vectors_file)
        
        # Compute silhouette score
        if len(vectors) > 0 and "latent" in vectors[0]:
            latents = np.array([v["latent"] for v in vectors])
            clusters = np.array([v["cluster"] for v in vectors])
            
            # Only compute if we have multiple clusters
            if len(set(clusters)) > 1:
                try:
                    silhouette = silhouette_score(latents, clusters, metric='euclidean')
                    metrics["silhouette_score"] = float(silhouette)
                except:
                    metrics["silhouette_score"] = 0.0
            else:
                metrics["silhouette_score"] = 0.0
        
        # Compute neighbor purity (simplified version)
        if "neighbor_purity" not in metrics:
            metrics["neighbor_purity"] = estimate_neighbor_purity(vectors)
    
    # Set defaults for missing metrics
    if "silhouette_score" not in metrics:
        metrics["silhouette_score"] = 0.0
    if "neighbor_purity" not in metrics:
        metrics["neighbor_purity"] = 0.0
    if "psi_fidelity" not in metrics:
        metrics["psi_fidelity"] = 0.0
    if "evr_captured" not in metrics:
        metrics["evr_captured"] = 0.0
    if "runtime_seconds" not in metrics:
        metrics["runtime_seconds"] = 0.0
    
    return metrics


def estimate_neighbor_purity(vectors: list, k: int = 15) -> float:
    """
    Estimate neighborhood purity: fraction of k-nearest neighbors from same cluster
    """
    if len(vectors) < k + 1:
        return 1.0
    
    try:
        from sklearn.neighbors import NearestNeighbors
        
        latents = np.array([v["latent"] for v in vectors])
        clusters = np.array([v["cluster"] for v in vectors])
        
        # Fit kNN
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='euclidean').fit(latents)
        distances, indices = nbrs.kneighbors(latents)
        
        # Compute purity for each cell
        purities = []
        for i in range(len(vectors)):
            cell_cluster = clusters[i]
            neighbor_indices = indices[i][1:]  # Exclude self
            neighbor_clusters = clusters[neighbor_indices]
            purity = (neighbor_clusters == cell_cluster).mean()
            purities.append(purity)
        
        return float(np.mean(purities))
    except Exception as e:
        logger.warning(f"Could not compute neighbor purity: {e}")
        return 0.0


def compute_deltas(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Compute metric deltas between baseline and candidate"""
    deltas = {}
    
    metrics_to_compare = [
        "silhouette_score",
        "neighbor_purity",
        "psi_fidelity",
        "evr_captured",
        "n_clusters",
        "runtime_seconds"
    ]
    
    for metric in metrics_to_compare:
        if metric in baseline and metric in candidate:
            baseline_val = baseline[metric]
            candidate_val = candidate[metric]
            
            delta = candidate_val - baseline_val
            
            # For runtime, also compute percentage change
            if metric == "runtime_seconds" and baseline_val > 0:
                pct_change = ((candidate_val - baseline_val) / baseline_val) * 100
                deltas[metric] = {
                    "absolute": round(float(delta), 3),
                    "percent": round(float(pct_change), 2)
                }
            else:
                deltas[metric] = round(float(delta), 4)
    
    return deltas


def assess_stability(deltas: Dict[str, Any]) -> Dict[str, str]:
    """
    Assess stability based on deltas
    Returns stability rating for each metric
    """
    stability = {}
    
    # Silhouette score: ±0.05 is stable
    if "silhouette_score" in deltas:
        delta = abs(deltas["silhouette_score"])
        if delta < 0.02:
            stability["silhouette_score"] = "stable"
        elif delta < 0.05:
            stability["silhouette_score"] = "minor_drift"
        else:
            stability["silhouette_score"] = "significant_drift"
    
    # Neighbor purity: ±0.03 is stable
    if "neighbor_purity" in deltas:
        delta = abs(deltas["neighbor_purity"])
        if delta < 0.01:
            stability["neighbor_purity"] = "stable"
        elif delta < 0.03:
            stability["neighbor_purity"] = "minor_drift"
        else:
            stability["neighbor_purity"] = "significant_drift"
    
    # ψ-fidelity: ±0.05 is stable
    if "psi_fidelity" in deltas:
        delta = abs(deltas["psi_fidelity"])
        if delta < 0.02:
            stability["psi_fidelity"] = "stable"
        elif delta < 0.05:
            stability["psi_fidelity"] = "minor_drift"
        else:
            stability["psi_fidelity"] = "significant_drift"
    
    # Cluster count: no change is stable
    if "n_clusters" in deltas:
        delta = abs(deltas["n_clusters"])
        if delta == 0:
            stability["n_clusters"] = "stable"
        elif delta <= 2:
            stability["n_clusters"] = "minor_drift"
        else:
            stability["n_clusters"] = "significant_drift"
    
    return stability


def generate_evolution_notes(deltas: Dict[str, Any], stability: Dict[str, str]) -> list:
    """Generate human-readable notes about evolution"""
    notes = []
    
    # Silhouette score
    if "silhouette_score" in deltas:
        delta = deltas["silhouette_score"]
        status = stability.get("silhouette_score", "unknown")
        if delta > 0.05:
            notes.append(f"✅ Silhouette score improved by {delta:+.3f} - better cluster separation")
        elif delta < -0.05:
            notes.append(f"⚠️ Silhouette score decreased by {delta:+.3f} - worse cluster separation")
        else:
            notes.append(f"✓ Silhouette score stable ({delta:+.3f})")
    
    # Neighbor purity
    if "neighbor_purity" in deltas:
        delta = deltas["neighbor_purity"]
        if delta > 0.03:
            notes.append(f"✅ Neighbor purity improved by {delta:+.3f} - cleaner neighborhoods")
        elif delta < -0.03:
            notes.append(f"⚠️ Neighbor purity decreased by {delta:+.3f} - noisier neighborhoods")
        else:
            notes.append(f"✓ Neighbor purity stable ({delta:+.3f})")
    
    # ψ-fidelity
    if "psi_fidelity" in deltas:
        delta = deltas["psi_fidelity"]
        if delta > 0.05:
            notes.append(f"✅ ψ-fidelity improved by {delta:+.3f} - better QAC stabilization")
        elif delta < -0.05:
            notes.append(f"⚠️ ψ-fidelity decreased by {delta:+.3f} - QAC drift detected")
        else:
            notes.append(f"✓ ψ-fidelity stable ({delta:+.3f})")
    
    # Cluster count
    if "n_clusters" in deltas:
        delta = deltas["n_clusters"]
        if delta != 0:
            notes.append(f"ℹ️ Cluster count changed by {delta:+d} clusters")
    
    # Runtime
    if "runtime_seconds" in deltas and isinstance(deltas["runtime_seconds"], dict):
        pct = deltas["runtime_seconds"]["percent"]
        if pct < -10:
            notes.append(f"✅ Runtime improved by {abs(pct):.1f}%")
        elif pct > 10:
            notes.append(f"⚠️ Runtime increased by {pct:.1f}%")
    
    return notes


def generate_recommendation(deltas: Dict[str, Any], stability: Dict[str, str]) -> str:
    """Generate overall recommendation"""
    
    # Count improvements, regressions, and stable metrics
    improvements = 0
    regressions = 0
    stable = 0
    
    quality_metrics = ["silhouette_score", "neighbor_purity", "psi_fidelity"]
    
    for metric in quality_metrics:
        if metric in deltas:
            delta = deltas[metric]
            if abs(delta) < 0.02:
                stable += 1
            elif delta > 0:
                improvements += 1
            else:
                regressions += 1
    
    # Generate recommendation
    if improvements > regressions:
        return "ACCEPT - Candidate run shows overall improvement"
    elif regressions > improvements:
        return "REJECT - Candidate run shows overall regression"
    elif stable > 0:
        return "STABLE - Both runs show comparable quality"
    else:
        return "REVIEW - Mixed results, manual review recommended"


# ============================================================================
# AnnData-Compatible Wrappers (for FastAPI endpoints and tests)
# Following ResoTrace universal_bridge.py pattern - TACE temporal stability
# ============================================================================

try:
    import anndata as ad
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    from scipy.spatial.distance import euclidean
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("AnnData/sklearn not available")


def compare_runs(
    run1_path: str,
    run2_path: str,
    cluster_key: str,
    latent_key: str,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Compare two pipeline runs for stability and drift
    
    Similar to TACE temporal collapse analysis - monitors runs for
    stability thresholds and triggers measurement when drift detected.
    
    Args:
        run1_path: Path to first run H5AD file
        run2_path: Path to second run H5AD file
        cluster_key: Key in adata.obs for cluster assignments
        latent_key: Key in adata.obsm for latent space (e.g., 'X_pca')
        seed: Random seed
        
    Returns:
        Dict with delta metrics and stability score
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData and sklearn required for compare_runs")
    
    # Load both runs
    adata1 = ad.read_h5ad(run1_path)
    adata2 = ad.read_h5ad(run2_path)
    
    # Validate keys
    if cluster_key not in adata1.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in run1")
    if cluster_key not in adata2.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in run2")
    if latent_key not in adata1.obsm:
        raise ValueError(f"Latent key '{latent_key}' not found in run1.obsm")
    if latent_key not in adata2.obsm:
        raise ValueError(f"Latent key '{latent_key}' not found in run2.obsm")
    
    # Compute deltas
    deltas = compute_deltas(adata1, adata2, cluster_key, latent_key)
    
    # Compute stability metrics
    metrics = stability_metrics(adata1, adata2, cluster_key, latent_key)
    
    return {
        "n_cells_run1": adata1.n_obs,
        "n_cells_run2": adata2.n_obs,
        "stability_score": metrics["overall_stability"],
        "delta_metrics": deltas,
        "report": {
            "cluster_stability": metrics["cluster_stability"],
            "latent_stability": metrics["latent_stability"],
            "interpretation": metrics["interpretation"]
        }
    }


def compute_deltas(
    adata1,
    adata2,
    cluster_key: str,
    latent_key: str
) -> Dict[str, float]:
    """
    Compute delta metrics between two runs
    
    Args:
        adata1: First AnnData object
        adata2: Second AnnData object
        cluster_key: Cluster key
        latent_key: Latent space key
        
    Returns:
        Dict of delta metrics
    """
    deltas = {}
    
    # Cell count delta
    deltas["cell_count_delta"] = adata2.n_obs - adata1.n_obs
    
    # Cluster count delta
    n_clusters1 = len(adata1.obs[cluster_key].unique())
    n_clusters2 = len(adata2.obs[cluster_key].unique())
    deltas["cluster_count_delta"] = n_clusters2 - n_clusters1
    
    # Cluster consistency (using smaller dataset size for fair comparison)
    min_cells = min(adata1.n_obs, adata2.n_obs)
    if min_cells > 1:
        # Sample equal number from both
        np.random.seed(42)
        idx1 = np.random.choice(adata1.n_obs, min(min_cells, adata1.n_obs), replace=False)
        idx2 = np.random.choice(adata2.n_obs, min(min_cells, adata2.n_obs), replace=False)
        
        labels1 = adata1.obs[cluster_key].values[idx1]
        labels2 = adata2.obs[cluster_key].values[idx2]
        
        # Adjusted Rand Index (1 = identical clustering)
        deltas["cluster_consistency"] = adjusted_rand_score(labels1, labels2)
    else:
        deltas["cluster_consistency"] = 0.0
    
    # Latent space drift (average Euclidean distance between centroids)
    if min_cells > 0:
        X1 = adata1.obsm[latent_key][:min_cells]
        X2 = adata2.obsm[latent_key][:min_cells]
        
        # Calculate mean drift
        drift = np.mean([euclidean(X1[i], X2[i]) for i in range(min(len(X1), len(X2)))])
        deltas["latent_drift"] = float(drift)
    else:
        deltas["latent_drift"] = 0.0
    
    # Gene expression delta (if X is available)
    if hasattr(adata1, 'X') and hasattr(adata2, 'X'):
        # Calculate mean expression difference
        X1_dense = adata1.X[:min_cells].toarray() if hasattr(adata1.X, 'toarray') else adata1.X[:min_cells]
        X2_dense = adata2.X[:min_cells].toarray() if hasattr(adata2.X, 'toarray') else adata2.X[:min_cells]
        
        expr_delta = np.mean(np.abs(X1_dense - X2_dense))
        deltas["gene_expression_delta"] = float(expr_delta)
    
    return deltas


def stability_metrics(
    adata1,
    adata2,
    cluster_key: str,
    latent_key: str
) -> Dict[str, Any]:
    """
    Calculate comprehensive stability metrics
    
    Uses TACE-style field tension analysis to determine
    if runs have reached stable collapse state.
    
    Args:
        adata1: First AnnData
        adata2: Second AnnData
        cluster_key: Cluster key
        latent_key: Latent key
        
    Returns:
        Dict with stability scores and interpretation
    """
    metrics = {}
    
    # Cluster stability (based on ARI)
    min_cells = min(adata1.n_obs, adata2.n_obs)
    if min_cells > 1:
        np.random.seed(42)
        idx1 = np.random.choice(adata1.n_obs, min(min_cells, adata1.n_obs), replace=False)
        idx2 = np.random.choice(adata2.n_obs, min(min_cells, adata2.n_obs), replace=False)
        
        labels1 = adata1.obs[cluster_key].values[idx1]
        labels2 = adata2.obs[cluster_key].values[idx2]
        
        ari = adjusted_rand_score(labels1, labels2)
        metrics["cluster_stability"] = (ari + 1.0) / 2.0 * 100.0  # Normalize to 0-100
    else:
        metrics["cluster_stability"] = 0.0
    
    # Latent stability (inverse of drift)
    if min_cells > 0:
        X1 = adata1.obsm[latent_key][:min_cells]
        X2 = adata2.obsm[latent_key][:min_cells]
        
        drift = np.mean([euclidean(X1[i], X2[i]) for i in range(min(len(X1), len(X2)))])
        # Normalize drift to 0-100 stability score (lower drift = higher stability)
        max_expected_drift = 10.0  # Heuristic
        latent_stability = max(0, 100.0 - (drift / max_expected_drift * 100.0))
        metrics["latent_stability"] = latent_stability
    else:
        metrics["latent_stability"] = 0.0
    
    # Overall stability (average of components)
    metrics["overall_stability"] = (metrics["cluster_stability"] + metrics["latent_stability"]) / 2.0
    
    # Interpretation
    if metrics["overall_stability"] > 90:
        metrics["interpretation"] = "Excellent - Runs are highly consistent"
    elif metrics["overall_stability"] > 75:
        metrics["interpretation"] = "Good - Runs are largely stable"
    elif metrics["overall_stability"] > 50:
        metrics["interpretation"] = "Moderate - Some drift detected"
    else:
        metrics["interpretation"] = "Poor - Significant instability between runs"
    
    return metrics

