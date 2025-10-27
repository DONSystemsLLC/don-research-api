#!/usr/bin/env python3
"""
Alpha-Stability Correlation Exploration
========================================

Research Question:
Do high-alpha cells (activated APCs) show quantum-stable gene expression?
Or is QAC stability independent of alpha regime?

Approach:
- NO pre-filtering of genes (analyze all available)
- DEFAULT QAC parameters only (layers=3, reinforce_rate=0.05)
- Report FULL correlation spectrum (positive, negative, zero)
- Test multiple hypotheses equally
- Let the math speak - no steering outcomes

Dataset: PBMC3K (2700 cells, 7 Leiden clusters)
Alpha values: Pre-computed via TACE (already in data)
"""

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr, spearmanr
import json
from datetime import datetime

# Add stack to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'stack'))

from tace.core import QACEngine

# ============================================================================
# Configuration (DEFAULTS ONLY - NO TUNING)
# ============================================================================

PBMC_PATH = "./data/pbmc3k_with_tace_alpha.h5ad"  # Updated to use TACE alphas
N_GENES = 100  # First 100 genes (no cherry-picking)
QAC_LAYERS = 3  # DEFAULT from engine
QAC_REINFORCE = 0.05  # DEFAULT from engine
QAC_QUBITS = 8  # DEFAULT from engine

# ============================================================================
# Core Analysis Functions
# ============================================================================

def load_pbmc_data():
    """Load PBMC3K with preprocessing and alpha values."""
    print("Loading PBMC3K dataset...")
    adata = sc.read_h5ad(PBMC_PATH)
    
    # Basic preprocessing if needed
    if 'log1p' not in adata.uns:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
    
    # Identify clusters if not present
    if 'leiden' not in adata.obs:
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
        sc.tl.leiden(adata)
    
    print(f"Loaded {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Clusters: {adata.obs['leiden'].nunique()}")
    
    return adata


def compute_cluster_gene_expression(adata, n_genes=100):
    """
    Compute cluster-level gene expression for QAC analysis.
    
    Returns:
    - cluster_expr: (n_clusters, n_genes) array
    - gene_names: List of gene names
    - cluster_sizes: Number of cells per cluster
    """
    print(f"\nComputing cluster-level expression for first {n_genes} genes...")
    
    clusters = adata.obs['leiden'].unique()
    n_clusters = len(clusters)
    
    # Use first N genes (no selection bias)
    gene_names = adata.var_names[:n_genes].tolist()
    
    cluster_expr = np.zeros((n_clusters, n_genes))
    cluster_sizes = []
    
    for i, cluster in enumerate(sorted(clusters)):
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_cells = adata[cluster_mask]
        cluster_sizes.append(cluster_cells.n_obs)
        
        # Mean expression across cluster
        cluster_expr[i, :] = cluster_cells[:, gene_names].X.mean(axis=0).A1
    
    print(f"Cluster expression matrix: {cluster_expr.shape}")
    print(f"Cluster sizes: {cluster_sizes}")
    
    return cluster_expr, gene_names, cluster_sizes


def apply_qac_to_genes(cluster_expr, gene_names):
    """
    Apply QAC stabilization to each gene's cross-cluster pattern.
    
    For each gene:
    - Input: expression pattern across N clusters (1D vector)
    - QAC stabilize with DEFAULT parameters
    - Measure stability as change magnitude
    """
    print(f"\nApplying QAC to {len(gene_names)} genes...")
    print(f"QAC Parameters: layers={QAC_LAYERS}, reinforce_rate={QAC_REINFORCE}, num_qubits={QAC_QUBITS}")
    
    qac = QACEngine(
        num_qubits=QAC_QUBITS,
        reinforce_rate=QAC_REINFORCE,
        layers=QAC_LAYERS
    )
    
    results = []
    
    for gene_idx, gene_name in enumerate(gene_names):
        # Extract gene's expression pattern across clusters
        gene_pattern = cluster_expr[:, gene_idx]
        
        # Apply QAC stabilization
        stabilized = qac.stabilize(gene_pattern)
        
        # Compute stability metrics
        original_mean = np.mean(gene_pattern)
        stabilized_mean = np.mean(stabilized)
        mean_change = np.abs(stabilized_mean - original_mean)
        relative_change = mean_change / (original_mean + 1e-10)
        
        # Stability score: lower = more stable (less change)
        stability_score = 1.0 - relative_change
        
        results.append({
            'gene': gene_name,
            'original_mean': float(original_mean),
            'stabilized_mean': float(stabilized_mean),
            'mean_change': float(mean_change),
            'relative_change': float(relative_change),
            'stability_score': float(stability_score),
            'cluster_variance': float(np.var(gene_pattern)),
            'stabilized_variance': float(np.var(stabilized))
        })
        
        if (gene_idx + 1) % 25 == 0:
            print(f"  Processed {gene_idx + 1}/{len(gene_names)} genes")
    
    return results


def compute_alpha_correlations(adata, qac_results, gene_names):
    """
    Compute correlation between gene expression and alpha values.
    
    For each gene:
    - Get single-cell expression values
    - Get corresponding alpha values
    - Compute Pearson and Spearman correlations
    
    IMPORTANT: We're testing if QAC stability correlates with alpha-expression relationship,
    NOT just alpha itself.
    """
    print("\nComputing alpha-expression correlations...")
    
    # Extract alpha values (NOW USING REAL TACE ALPHAS!)
    if 'alpha_tace' in adata.obs:
        alphas = adata.obs['alpha_tace'].values
        print(f"Using REAL TACE alpha values from dataset!")
    elif 'alpha' not in adata.obs:
        print("WARNING: Alpha values not found in dataset!")
        print("Using PCA-based proxy (not real TACE alphas)")
        # For now, use PCA-based proxy (from cellular discovery)
        sc.tl.pca(adata, n_comps=50)
        pca_norms = np.linalg.norm(adata.obsm['X_pca'], axis=1)
        # Normalize to [0, 1] range
        alphas = (pca_norms - pca_norms.min()) / (pca_norms.max() - pca_norms.min())
    else:
        alphas = adata.obs['alpha'].values
        print(f"Using alpha values from dataset")
    
    print(f"Alpha range: [{alphas.min():.3f}, {alphas.max():.3f}]")
    print(f"Alpha distribution: mean={alphas.mean():.3f}, std={alphas.std():.3f}")
    
    # Compute correlations for each gene
    for result in qac_results:
        gene_name = result['gene']
        
        # Get single-cell expression for this gene
        gene_expr = adata[:, gene_name].X.toarray().flatten()
        
        # Compute correlations
        pearson_r, pearson_p = pearsonr(gene_expr, alphas)
        spearman_r, spearman_p = spearmanr(gene_expr, alphas)
        
        # Add to results
        result['alpha_corr_pearson'] = float(pearson_r)
        result['alpha_corr_pearson_p'] = float(pearson_p)
        result['alpha_corr_spearman'] = float(spearman_r)
        result['alpha_corr_spearman_p'] = float(spearman_p)
        
        # Also compute expression in low vs high alpha regimes
        low_alpha_mask = alphas < 0.3
        high_alpha_mask = alphas > 0.7
        
        if low_alpha_mask.sum() > 0:
            result['low_alpha_expr'] = float(gene_expr[low_alpha_mask].mean())
        else:
            result['low_alpha_expr'] = None
            
        if high_alpha_mask.sum() > 0:
            result['high_alpha_expr'] = float(gene_expr[high_alpha_mask].mean())
        else:
            result['high_alpha_expr'] = None
    
    return qac_results


def analyze_correlations(results):
    """
    Analyze the relationship between QAC stability and alpha correlations.
    
    Key question: Do quantum-stable genes (high stability_score) show
    stronger or weaker correlations with alpha?
    """
    print("\n" + "="*80)
    print("CORRELATION ANALYSIS: QAC Stability vs Alpha Relationship")
    print("="*80)
    
    # Extract metrics
    stability_scores = [r['stability_score'] for r in results]
    alpha_correlations = [r['alpha_corr_pearson'] for r in results]
    
    # Compute meta-correlation: stability vs alpha-correlation
    meta_corr_pearson, meta_p_pearson = pearsonr(stability_scores, alpha_correlations)
    meta_corr_spearman, meta_p_spearman = spearmanr(stability_scores, alpha_correlations)
    
    print(f"\nMeta-Correlation (Stability vs Alpha-Correlation):")
    print(f"  Pearson:  r={meta_corr_pearson:+.4f}, p={meta_p_pearson:.2e}")
    print(f"  Spearman: r={meta_corr_spearman:+.4f}, p={meta_p_spearman:.2e}")
    
    if abs(meta_corr_pearson) < 0.1:
        print("\n  Interpretation: WEAK/NO relationship - stability independent of alpha")
    elif meta_corr_pearson > 0.3:
        print("\n  Interpretation: POSITIVE relationship - stable genes correlate with alpha")
    elif meta_corr_pearson < -0.3:
        print("\n  Interpretation: NEGATIVE relationship - stable genes anti-correlate with alpha")
    
    # Distribution analysis
    print(f"\nStability Score Distribution:")
    print(f"  Range: [{min(stability_scores):.4f}, {max(stability_scores):.4f}]")
    print(f"  Mean: {np.mean(stability_scores):.4f} ± {np.std(stability_scores):.4f}")
    print(f"  Median: {np.median(stability_scores):.4f}")
    
    print(f"\nAlpha Correlation Distribution:")
    print(f"  Range: [{min(alpha_correlations):+.4f}, {max(alpha_correlations):+.4f}]")
    print(f"  Mean: {np.mean(alpha_correlations):+.4f} ± {np.std(alpha_correlations):.4f}")
    print(f"  Median: {np.median(alpha_correlations):+.4f}")
    
    # Count correlation directions
    positive = sum(1 for r in alpha_correlations if r > 0.1)
    negative = sum(1 for r in alpha_correlations if r < -0.1)
    neutral = len(alpha_correlations) - positive - negative
    
    print(f"\nCorrelation Direction Breakdown:")
    print(f"  Positive (r > +0.1): {positive} genes ({100*positive/len(alpha_correlations):.1f}%)")
    print(f"  Negative (r < -0.1): {negative} genes ({100*negative/len(alpha_correlations):.1f}%)")
    print(f"  Neutral (|r| ≤ 0.1): {neutral} genes ({100*neutral/len(alpha_correlations):.1f}%)")
    
    return {
        'meta_correlation_pearson': float(meta_corr_pearson),
        'meta_correlation_p_pearson': float(meta_p_pearson),
        'meta_correlation_spearman': float(meta_corr_spearman),
        'meta_correlation_p_spearman': float(meta_p_spearman),
        'stability_stats': {
            'min': float(min(stability_scores)),
            'max': float(max(stability_scores)),
            'mean': float(np.mean(stability_scores)),
            'std': float(np.std(stability_scores)),
            'median': float(np.median(stability_scores))
        },
        'alpha_correlation_stats': {
            'min': float(min(alpha_correlations)),
            'max': float(max(alpha_correlations)),
            'mean': float(np.mean(alpha_correlations)),
            'std': float(np.std(alpha_correlations)),
            'median': float(np.median(alpha_correlations))
        },
        'direction_counts': {
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }
    }


def report_top_genes(results, n=10):
    """Report top genes by different criteria - NO FILTERING, just ranking."""
    print("\n" + "="*80)
    print("TOP GENES BY DIFFERENT CRITERIA")
    print("="*80)
    
    # Sort by stability (most stable = highest score)
    by_stability = sorted(results, key=lambda x: x['stability_score'], reverse=True)
    print(f"\nMost Quantum-Stable Genes (top {n}):")
    for i, r in enumerate(by_stability[:n], 1):
        print(f"  {i}. {r['gene']}: stability={r['stability_score']:.4f}, "
              f"alpha_corr={r['alpha_corr_pearson']:+.4f}")
    
    # Sort by positive alpha correlation
    by_positive_alpha = sorted(results, key=lambda x: x['alpha_corr_pearson'], reverse=True)
    print(f"\nStrongest Positive Alpha Correlations (top {n}):")
    for i, r in enumerate(by_positive_alpha[:n], 1):
        print(f"  {i}. {r['gene']}: alpha_corr={r['alpha_corr_pearson']:+.4f}, "
              f"stability={r['stability_score']:.4f}")
    
    # Sort by negative alpha correlation
    by_negative_alpha = sorted(results, key=lambda x: x['alpha_corr_pearson'])
    print(f"\nStrongest Negative Alpha Correlations (top {n}):")
    for i, r in enumerate(by_negative_alpha[:n], 1):
        print(f"  {i}. {r['gene']}: alpha_corr={r['alpha_corr_pearson']:+.4f}, "
              f"stability={r['stability_score']:.4f}")
    
    # Genes with high stability AND strong alpha correlation
    combined_score = [(r['gene'], r['stability_score'] * abs(r['alpha_corr_pearson']), 
                      r['stability_score'], r['alpha_corr_pearson']) 
                     for r in results]
    combined_score.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nGenes with BOTH High Stability AND Strong Alpha Correlation (top {n}):")
    for i, (gene, score, stab, alpha_corr) in enumerate(combined_score[:n], 1):
        print(f"  {i}. {gene}: combined={score:.4f} "
              f"(stability={stab:.4f}, alpha_corr={alpha_corr:+.4f})")


def save_results(results, summary):
    """Save full results to JSON."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'method': 'Alpha-Stability Correlation Analysis',
        'parameters': {
            'qac_layers': QAC_LAYERS,
            'qac_reinforce_rate': QAC_REINFORCE,
            'qac_num_qubits': QAC_QUBITS,
            'n_genes': N_GENES
        },
        'summary': summary,
        'gene_results': results
    }
    
    output_path = './alpha_stability_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def main():
    print("="*80)
    print("ALPHA-STABILITY CORRELATION EXPLORATION")
    print("="*80)
    print("\nResearch Question:")
    print("Do high-alpha cells show quantum-stable gene expression patterns?")
    print("\nPrinciples:")
    print("✓ NO parameter tuning (using engine defaults)")
    print("✓ NO gene pre-filtering (analyzing first 100 genes)")
    print("✓ NO bias injection (letting math speak)")
    print("="*80)
    
    # Load data
    adata = load_pbmc_data()
    
    # Compute cluster-level expression
    cluster_expr, gene_names, cluster_sizes = compute_cluster_gene_expression(
        adata, n_genes=N_GENES
    )
    
    # Apply QAC to genes
    qac_results = apply_qac_to_genes(cluster_expr, gene_names)
    
    # Compute alpha correlations
    qac_results = compute_alpha_correlations(adata, qac_results, gene_names)
    
    # Analyze correlations
    summary = analyze_correlations(qac_results)
    
    # Report top genes
    report_top_genes(qac_results, n=10)
    
    # Save results
    save_results(qac_results, summary)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - Math has spoken!")
    print("="*80)


if __name__ == '__main__':
    main()
