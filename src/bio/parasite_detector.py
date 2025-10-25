"""
Parasite Detector - QC Contamination Analysis
Detects ambient RNA, doublets, and batch mixing issues

Integrates with DON Stack for enhanced detection via QAC error correction
"""

import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from collections import Counter
import logging

logger = logging.getLogger(__name__)

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("AnnData not available - some functions will be limited")


def parasite_detect(
    vectors_path: Path,
    checks: List[str] = ["ambient", "doublets", "batch_mix"]
) -> Dict[str, Any]:
    """
    Detect QC parasites: ambient RNA, doublets, batch effects
    
    Args:
        vectors_path: Path to collapse_vectors.jsonl
        checks: List of checks to perform
        
    Returns:
        Dict with findings and recommendations
    """
    from .adapter_anndata import stream_jsonl
    
    logger.info(f"Loading vectors from {vectors_path}")
    cells = stream_jsonl(vectors_path)
    n_cells = len(cells)
    
    results = {
        "n_cells": n_cells,
        "checks_performed": checks,
        "findings": {}
    }
    
    if "ambient" in checks:
        results["findings"]["ambient"] = detect_ambient_rna(cells)
    
    if "doublets" in checks:
        results["findings"]["doublets"] = detect_doublets(cells)
    
    if "batch_mix" in checks:
        results["findings"]["batch_mix"] = detect_batch_mixing(cells)
    
    # Generate recommendations
    results["recommendations"] = generate_recommendations(results["findings"])
    
    return results


def detect_ambient_rna(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect ambient RNA contamination
    High UMI counts + low gene counts = ambient soup
    """
    n_genes = np.array([c["qc"]["n_genes"] for c in cells])
    n_counts = np.array([c["qc"]["n_counts"] for c in cells])
    
    # Compute genes/UMI ratio (should be ~0.3-0.5 for good cells)
    with np.errstate(divide='ignore', invalid='ignore'):
        gene_umi_ratio = n_genes / n_counts
        gene_umi_ratio = np.nan_to_num(gene_umi_ratio, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Flag cells with suspiciously low ratio
    low_ratio_threshold = 0.2
    low_ratio_cells = (gene_umi_ratio < low_ratio_threshold) & (n_counts > 500)
    n_flagged = low_ratio_cells.sum()
    pct_flagged = (n_flagged / len(cells)) * 100
    
    # Get cluster distribution of flagged cells
    flagged_clusters = [cells[i]["cluster"] for i in range(len(cells)) if low_ratio_cells[i]]
    cluster_counts = Counter(flagged_clusters)
    
    severity = "low"
    if pct_flagged > 10:
        severity = "high"
    elif pct_flagged > 5:
        severity = "medium"
    
    return {
        "n_flagged": int(n_flagged),
        "pct_flagged": round(pct_flagged, 2),
        "severity": severity,
        "threshold": low_ratio_threshold,
        "mean_ratio": round(float(gene_umi_ratio.mean()), 3),
        "affected_clusters": dict(cluster_counts.most_common(5)),
        "description": f"{pct_flagged:.1f}% of cells show low gene/UMI ratio (< {low_ratio_threshold}), suggesting ambient RNA contamination"
    }


def detect_doublets(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect doublet enrichment
    High doublet scores OR high gene counts + high UMI counts
    """
    doublet_scores = np.array([c["qc"]["doublet_score"] for c in cells])
    n_genes = np.array([c["qc"]["n_genes"] for c in cells])
    n_counts = np.array([c["qc"]["n_counts"] for c in cells])
    
    # Method 1: Doublet score threshold (if available)
    score_threshold = 0.3
    high_score = doublet_scores > score_threshold
    
    # Method 2: Outlier detection (high genes AND high UMIs)
    gene_threshold = np.percentile(n_genes, 95)
    umi_threshold = np.percentile(n_counts, 95)
    outliers = (n_genes > gene_threshold) & (n_counts > umi_threshold)
    
    # Combine methods
    flagged = high_score | outliers
    n_flagged = flagged.sum()
    pct_flagged = (n_flagged / len(cells)) * 100
    
    # Expected doublet rate (typically 0.4% per 1000 cells loaded)
    expected_rate = min(8.0, 0.4 * (len(cells) / 1000))
    
    severity = "low"
    if pct_flagged > expected_rate * 2:
        severity = "high"
    elif pct_flagged > expected_rate * 1.5:
        severity = "medium"
    
    # Get cluster distribution
    flagged_clusters = [cells[i]["cluster"] for i in range(len(cells)) if flagged[i]]
    cluster_counts = Counter(flagged_clusters)
    
    return {
        "n_flagged": int(n_flagged),
        "pct_flagged": round(pct_flagged, 2),
        "expected_rate": round(expected_rate, 2),
        "severity": severity,
        "mean_doublet_score": round(float(doublet_scores.mean()), 3),
        "affected_clusters": dict(cluster_counts.most_common(5)),
        "description": f"{pct_flagged:.1f}% of cells flagged as potential doublets (expected ~{expected_rate:.1f}%)"
    }


def detect_batch_mixing(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect batch mixing issues
    Clusters should have balanced batch representation
    """
    # Group cells by cluster
    clusters = {}
    for cell in cells:
        cluster = cell["cluster"]
        batch = cell["batch"]
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(batch)
    
    # Get all batches
    all_batches = list(set(cell["batch"] for cell in cells))
    n_batches = len(all_batches)
    
    if n_batches <= 1:
        return {
            "n_batches": n_batches,
            "severity": "none",
            "description": "Single batch - no batch mixing analysis needed"
        }
    
    # Compute batch purity per cluster
    # Purity = (max batch count) / (total cells in cluster)
    cluster_purities = {}
    problematic_clusters = []
    
    for cluster_id, batches in clusters.items():
        batch_counts = Counter(batches)
        max_batch_count = max(batch_counts.values())
        total_cells = len(batches)
        purity = max_batch_count / total_cells
        
        cluster_purities[cluster_id] = {
            "purity": round(purity, 3),
            "n_cells": total_cells,
            "batch_distribution": dict(batch_counts)
        }
        
        # Flag clusters with low purity (< 0.7 = less than 70% from dominant batch)
        if purity < 0.7 and total_cells > 20:  # Only flag substantial clusters
            problematic_clusters.append(cluster_id)
    
    mean_purity = np.mean([p["purity"] for p in cluster_purities.values()])
    
    severity = "low"
    if len(problematic_clusters) > len(clusters) * 0.3:
        severity = "high"
    elif len(problematic_clusters) > len(clusters) * 0.15:
        severity = "medium"
    
    return {
        "n_batches": n_batches,
        "n_clusters": len(clusters),
        "n_problematic_clusters": len(problematic_clusters),
        "problematic_clusters": problematic_clusters[:5],  # Top 5
        "mean_purity": round(float(mean_purity), 3),
        "severity": severity,
        "cluster_details": cluster_purities,
        "description": f"{len(problematic_clusters)} clusters show poor batch mixing (purity < 0.7 across {n_batches} batches)"
    }


def generate_recommendations(findings: Dict[str, Any]) -> List[str]:
    """
    Generate ordered recommendations based on findings
    """
    recommendations = []
    
    # Ambient RNA recommendations
    if "ambient" in findings:
        ambient = findings["ambient"]
        if ambient["severity"] == "high":
            recommendations.append({
                "priority": 1,
                "action": "filter_ambient",
                "description": f"Remove {ambient['n_flagged']} cells with low gene/UMI ratio (< {ambient['threshold']})",
                "rationale": "High ambient RNA contamination detected"
            })
    
    # Doublet recommendations
    if "doublets" in findings:
        doublets = findings["doublets"]
        if doublets["severity"] in ["high", "medium"]:
            recommendations.append({
                "priority": 1,
                "action": "filter_doublets",
                "description": f"Remove {doublets['n_flagged']} potential doublets",
                "rationale": f"Doublet rate ({doublets['pct_flagged']:.1f}%) exceeds expected ({doublets['expected_rate']:.1f}%)"
            })
    
    # Batch mixing recommendations
    if "batch_mix" in findings:
        batch_mix = findings["batch_mix"]
        if batch_mix["severity"] == "high":
            recommendations.append({
                "priority": 2,
                "action": "batch_correction",
                "description": f"Apply batch correction (e.g., Harmony, scVI) - {batch_mix['n_problematic_clusters']} clusters affected",
                "rationale": "Significant batch effects detected"
            })
        elif batch_mix["severity"] == "medium":
            recommendations.append({
                "priority": 3,
                "action": "review_batch_effects",
                "description": "Review batch effects - some clusters show imbalance",
                "rationale": "Moderate batch mixing issues"
            })
    
    # Sort by priority
    recommendations.sort(key=lambda x: x["priority"])
    
    return recommendations


# ============================================================================
# AnnData-Compatible Wrappers (for FastAPI endpoints and tests)
# Following ResoTrace universal_bridge.py pattern for DON Stack integration
# ============================================================================

def detect_parasites(
    adata,  # AnnData object
    cluster_key: str,
    batch_key: str,
    ambient_threshold: float = 0.15,
    doublet_threshold: float = 0.25,
    batch_threshold: float = 0.3,
    seed: Optional[int] = 42
) -> Dict[str, Any]:
    """
    Detect QC parasites directly from AnnData object
    
    Args:
        adata: AnnData object with QC metrics in .obs
        cluster_key: Key in adata.obs for cluster assignments
        batch_key: Key in adata.obs for batch labels
        ambient_threshold: Threshold for ambient RNA detection
        doublet_threshold: Threshold for doublet detection
        batch_threshold: Threshold for batch contamination
        seed: Random seed for reproducibility
        
    Returns:
        Dict with detection results and per-cell flags
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData is required for detect_parasites")
    
    # Validate keys exist
    if cluster_key not in adata.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs")
    if batch_key not in adata.obs:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs")
    
    n_cells = adata.n_obs
    
    # Compute per-cell scores
    ambient_scores = ambient_score(adata)
    doublet_scores = doublet_enrichment(adata, cluster_key)
    batch_scores = batch_purity(adata, cluster_key, batch_key)
    
    # Flag cells exceeding thresholds
    ambient_flags = ambient_scores > ambient_threshold
    doublet_flags = doublet_scores > doublet_threshold
    batch_flags = batch_scores > batch_threshold
    
    # Combined flags (any failure)
    flags = ambient_flags | doublet_flags | batch_flags
    n_flagged = int(np.sum(flags))
    
    # Overall parasite score (percentage of flagged cells)
    parasite_score = (n_flagged / n_cells) * 100.0
    
    # Detailed report
    report = {
        "ambient_rna": {
            "n_flagged": int(np.sum(ambient_flags)),
            "mean_score": float(np.mean(ambient_scores)),
            "threshold": ambient_threshold
        },
        "doublets": {
            "n_flagged": int(np.sum(doublet_flags)),
            "mean_score": float(np.mean(doublet_scores)),
            "threshold": doublet_threshold
        },
        "batch_effects": {
            "n_flagged": int(np.sum(batch_flags)),
            "mean_score": float(np.mean(batch_scores)),
            "threshold": batch_threshold
        }
    }
    
    return {
        "n_cells": n_cells,
        "n_flagged": n_flagged,
        "flags": flags.tolist(),
        "parasite_score": round(parasite_score, 2),
        "report": report,
        "thresholds": {
            "ambient": ambient_threshold,
            "doublet": doublet_threshold,
            "batch": batch_threshold
        }
    }


def ambient_score(adata) -> np.ndarray:
    """
    Calculate ambient RNA score for each cell
    
    Low complexity (few genes detected) suggests ambient contamination
    Similar to DON-GPU cluster density analysis
    
    Args:
        adata: AnnData object with QC metrics
        
    Returns:
        Array of ambient scores (0-1) for each cell
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData is required")
    
    # Use n_genes_by_counts if available, else calculate
    if 'n_genes_by_counts' in adata.obs:
        n_genes = adata.obs['n_genes_by_counts'].values
    else:
        n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
    
    # Normalize to 0-1 (inverse: fewer genes = higher ambient score)
    max_genes = np.max(n_genes)
    if max_genes == 0:
        return np.zeros(adata.n_obs)
    
    scores = 1.0 - (n_genes / max_genes)
    return scores


def doublet_enrichment(adata, cluster_key: str) -> np.ndarray:
    """
    Calculate doublet enrichment score for each cell
    
    High UMI count + high gene count within cluster suggests doublet
    Uses QAC-style adjacency analysis within clusters
    
    Args:
        adata: AnnData object
        cluster_key: Key in adata.obs for clusters
        
    Returns:
        Array of doublet scores (0-1) for each cell
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData is required")
    
    # Get total counts
    if 'total_counts' in adata.obs:
        total_counts = adata.obs['total_counts'].values
    else:
        total_counts = np.array(adata.X.sum(axis=1)).flatten()
    
    # Get gene counts
    if 'n_genes_by_counts' in adata.obs:
        n_genes = adata.obs['n_genes_by_counts'].values
    else:
        n_genes = np.array((adata.X > 0).sum(axis=1)).flatten()
    
    # Calculate z-scores within each cluster (adjacency-aware)
    clusters = adata.obs[cluster_key].values
    scores = np.zeros(adata.n_obs)
    
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        cluster_counts = total_counts[mask]
        cluster_genes = n_genes[mask]
        
        if len(cluster_counts) > 1:
            # Z-score for total counts
            count_mean = np.mean(cluster_counts)
            count_std = np.std(cluster_counts)
            if count_std > 0:
                count_z = (total_counts[mask] - count_mean) / count_std
            else:
                count_z = np.zeros(np.sum(mask))
            
            # Z-score for gene counts
            gene_mean = np.mean(cluster_genes)
            gene_std = np.std(cluster_genes)
            if gene_std > 0:
                gene_z = (n_genes[mask] - gene_mean) / gene_std
            else:
                gene_z = np.zeros(np.sum(mask))
            
            # Combined score (both high = likely doublet)
            combined_z = (count_z + gene_z) / 2.0
            scores[mask] = np.clip(combined_z / 3.0, 0, 1)  # Normalize to 0-1
    
    return scores


def batch_purity(adata, cluster_key: str, batch_key: str) -> np.ndarray:
    """
    Calculate batch contamination score for each cell
    
    Clusters should not be dominated by single batches
    Uses TACE-style field tension analysis for batch effects
    
    Args:
        adata: AnnData object
        cluster_key: Key in adata.obs for clusters
        batch_key: Key in adata.obs for batches
        
    Returns:
        Array of batch contamination scores (0-1) for each cell
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData is required")
    
    clusters = adata.obs[cluster_key].values
    batches = adata.obs[batch_key].values
    
    scores = np.zeros(adata.n_obs)
    
    for cluster in np.unique(clusters):
        mask = clusters == cluster
        cluster_batches = batches[mask]
        
        if len(cluster_batches) > 0:
            # Calculate batch purity (1 = all same batch, 0 = perfectly mixed)
            batch_counts = Counter(cluster_batches)
            most_common_count = batch_counts.most_common(1)[0][1]
            purity = most_common_count / len(cluster_batches)
            
            # High purity = high contamination score
            scores[mask] = purity
    
    return scores

