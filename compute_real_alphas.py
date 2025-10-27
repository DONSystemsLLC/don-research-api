#!/usr/bin/env python3
"""
Compute Real TACE Alpha Values for PBMC3K
==========================================

Purpose:
- Compute actual TACE alpha values for all 2700 cells
- Use DEFAULT TACE parameters (no tuning)
- Save alpha values to adata.obs['alpha_tace']
- Compare with PCA-proxy to validate difference

Approach:
- For each cell: get gene expression vector
- Apply TACE tune_alpha() with DEFAULT parameters
- Use DON-GPU compressed cluster vectors as reference tensions
- Let TACE engine determine alpha values (no steering)

Dataset: PBMC3K (2700 cells × 13714 genes)
"""

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
from datetime import datetime
import json

# Add stack to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'stack'))
sys.path.insert(0, str(current_dir / 'src'))

from tace.core import TACEController, QACEngine, tune_alpha
from don_memory.adapters.don_stack_adapter import DONStackAdapter

# ============================================================================
# Configuration (DEFAULTS ONLY)
# ============================================================================

PBMC_PATH = "./data/pbmc3k.h5ad"
OUTPUT_PATH = "./data/pbmc3k_with_tace_alpha.h5ad"

# TACE defaults (from engine)
TACE_COLLAPSE_THRESHOLD = 0.975
TACE_FEEDBACK_GAIN = 0.3
TACE_MAX_ITERATIONS = 10

# QAC defaults (used by tune_alpha)
QAC_NUM_QUBITS = 8
QAC_REINFORCE_RATE = 0.05
QAC_LAYERS = 3

# Default alpha for tune_alpha
DEFAULT_ALPHA = 0.5

# ============================================================================
# Core Functions
# ============================================================================

def load_pbmc_data():
    """Load PBMC3K with preprocessing."""
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


def compute_cluster_centroids(adata):
    """
    Compute cluster centroids as "tensions" for TACE.
    
    In DON theory, tensions represent the background field state.
    Cluster centroids = average transcriptional state of each population.
    """
    print("\nComputing cluster centroids as TACE tensions...")
    
    clusters = sorted(adata.obs['leiden'].unique())
    n_clusters = len(clusters)
    
    # Get full gene expression matrix
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    
    centroids = []
    for cluster in clusters:
        cluster_mask = adata.obs['leiden'] == cluster
        centroid = X[cluster_mask].mean(axis=0)
        centroids.append(centroid)
        print(f"  Cluster {cluster}: {cluster_mask.sum()} cells, "
              f"centroid norm={np.linalg.norm(centroid):.2f}")
    
    centroids = np.array(centroids)
    print(f"\nCentroid matrix shape: {centroids.shape}")
    
    return centroids


def compute_tace_alpha_for_cell(cell_pca, don_adapter, default_alpha=0.5):
    """
    Compute TACE alpha for a single cell using the CORRECT approach.
    
    Args:
        cell_pca: PCA representation for one cell (first 32 PCs)
        don_adapter: DONStackAdapter for normalize() operation
        default_alpha: Starting alpha value
    
    Returns:
        alpha: TACE-computed alpha value [0.1, 0.9]
    
    Method (following cellular_discovery.py exactly):
    1. Take first 32 PCA components
    2. Apply DON-GPU normalize (fractal compression reveals quantum structure)
    3. Take first 8 values from normalized result
    4. Pass to tune_alpha as list
    """
    # Apply DON-GPU normalize (fractal clustering - CRITICAL STEP!)
    # This reveals the quantum structure that TACE needs
    normalized = don_adapter.normalize(cell_pca)
    
    # Take first 8 values (match num_qubits) and convert to list
    # This is exactly how cellular_discovery.py did it
    tensions_for_tace = normalized[:8].tolist()
    
    # Compute alpha using tune_alpha
    alpha = tune_alpha(
        tensions=tensions_for_tace,
        default_alpha=default_alpha
    )
    
    return alpha


def compute_all_tace_alphas(adata, batch_size=100):
    """
    Compute TACE alpha for all cells using CORRECT DON-GPU approach.
    
    Args:
        adata: AnnData object with PCA computed
        batch_size: Process cells in batches for progress reporting
    
    Returns:
        alphas: Array of alpha values for all cells
    """
    print("\n" + "="*80)
    print("COMPUTING REAL TACE ALPHA VALUES (DON-GPU + TACE Pipeline)")
    print("="*80)
    print(f"\nParameters:")
    print(f"  QAC: num_qubits={QAC_NUM_QUBITS}, layers={QAC_LAYERS}, reinforce_rate={QAC_REINFORCE_RATE}")
    print(f"  TACE: collapse_threshold={TACE_COLLAPSE_THRESHOLD}, feedback_gain={TACE_FEEDBACK_GAIN}")
    print(f"  Default alpha: {DEFAULT_ALPHA}")
    print(f"\nMethod:")
    print(f"  1. Get PCA representation (first 32 PCs)")
    print(f"  2. Apply DON-GPU normalize (fractal clustering)")
    print(f"  3. Take first 8 values → TACE tune_alpha")
    print(f"  4. This matches cellular_discovery.py exactly!")
    print()
    
    # Compute PCA if not present
    if 'X_pca' not in adata.obsm:
        print("Computing PCA (50 components)...")
        sc.tl.pca(adata, n_comps=50)
        print("  ✓ PCA computed")
    
    # Initialize DON Stack adapter
    don_adapter = DONStackAdapter()
    
    n_cells = adata.n_obs
    alphas = np.zeros(n_cells)
    
    # Get PCA matrix
    X_pca = adata.obsm['X_pca']
    
    print(f"Processing {n_cells} cells with DON-GPU + TACE pipeline...")
    
    for i in range(n_cells):
        # Get PCA representation (first 32 components, as in cellular_discovery.py)
        cell_pca = X_pca[i, :32]
        
        # Compute TACE alpha with DON-GPU normalize
        alpha = compute_tace_alpha_for_cell(
            cell_pca=cell_pca,
            don_adapter=don_adapter,
            default_alpha=DEFAULT_ALPHA
        )
        
        alphas[i] = alpha
        
        # Progress reporting
        if (i + 1) % batch_size == 0:
            current_alphas = alphas[:i+1]
            unique_alphas = len(np.unique(current_alphas))
            print(f"  Processed {i+1}/{n_cells} cells "
                  f"(alpha range: [{current_alphas.min():.3f}, {current_alphas.max():.3f}], "
                  f"{unique_alphas} unique values)")
    
    print(f"\n✓ Computed alpha values for all {n_cells} cells")
    
    return alphas


def analyze_alpha_distribution(alphas, adata):
    """Analyze the computed alpha distribution."""
    print("\n" + "="*80)
    print("ALPHA DISTRIBUTION ANALYSIS")
    print("="*80)
    
    print(f"\nOverall Statistics:")
    print(f"  Range: [{alphas.min():.4f}, {alphas.max():.4f}]")
    print(f"  Mean: {alphas.mean():.4f} ± {alphas.std():.4f}")
    print(f"  Median: {np.median(alphas):.4f}")
    print(f"  Q1: {np.percentile(alphas, 25):.4f}")
    print(f"  Q3: {np.percentile(alphas, 75):.4f}")
    
    # Check for bimodality (from immune state discovery)
    low_alpha = (alphas < 0.3).sum()
    mid_alpha = ((alphas >= 0.3) & (alphas <= 0.7)).sum()
    high_alpha = (alphas > 0.7).sum()
    
    print(f"\nRegime Distribution:")
    print(f"  Low α (<0.3):     {low_alpha:4d} cells ({100*low_alpha/len(alphas):5.1f}%)")
    print(f"  Mid α (0.3-0.7):  {mid_alpha:4d} cells ({100*mid_alpha/len(alphas):5.1f}%)")
    print(f"  High α (>0.7):    {high_alpha:4d} cells ({100*high_alpha/len(alphas):5.1f}%)")
    
    # Check if bimodal
    if low_alpha > 100 and high_alpha > 100:
        print("\n  → BIMODAL distribution detected (matches immune state discovery!)")
    else:
        print("\n  → Distribution does not show clear bimodality")
    
    # Per-cluster analysis
    print(f"\nAlpha by Cluster:")
    clusters = sorted(adata.obs['leiden'].unique())
    
    cluster_stats = []
    for cluster in clusters:
        cluster_mask = adata.obs['leiden'] == cluster
        cluster_alphas = alphas[cluster_mask]
        
        stats = {
            'cluster': str(cluster),
            'n_cells': int(cluster_mask.sum()),
            'mean_alpha': float(cluster_alphas.mean()),
            'std_alpha': float(cluster_alphas.std()),
            'min_alpha': float(cluster_alphas.min()),
            'max_alpha': float(cluster_alphas.max()),
            'median_alpha': float(np.median(cluster_alphas))
        }
        
        cluster_stats.append(stats)
        
        print(f"  Cluster {cluster} (n={stats['n_cells']:4d}): "
              f"α={stats['mean_alpha']:.3f}±{stats['std_alpha']:.3f} "
              f"[{stats['min_alpha']:.3f}, {stats['max_alpha']:.3f}]")
    
    return {
        'overall': {
            'min': float(alphas.min()),
            'max': float(alphas.max()),
            'mean': float(alphas.mean()),
            'std': float(alphas.std()),
            'median': float(np.median(alphas)),
            'q1': float(np.percentile(alphas, 25)),
            'q3': float(np.percentile(alphas, 75))
        },
        'regimes': {
            'low': int(low_alpha),
            'mid': int(mid_alpha),
            'high': int(high_alpha)
        },
        'by_cluster': cluster_stats
    }


def compare_with_pca_proxy(alphas, adata):
    """Compare TACE alphas with PCA-based proxy."""
    print("\n" + "="*80)
    print("COMPARISON: TACE Alpha vs PCA Proxy")
    print("="*80)
    
    # Compute PCA proxy (same as in alpha_stability_exploration.py)
    if 'X_pca' not in adata.obsm:
        sc.tl.pca(adata, n_comps=50)
    
    pca_norms = np.linalg.norm(adata.obsm['X_pca'], axis=1)
    pca_proxy = (pca_norms - pca_norms.min()) / (pca_norms.max() - pca_norms.min())
    
    # Compute correlation
    from scipy.stats import pearsonr, spearmanr
    
    corr_pearson, p_pearson = pearsonr(alphas, pca_proxy)
    corr_spearman, p_spearman = spearmanr(alphas, pca_proxy)
    
    print(f"\nCorrelation between TACE alpha and PCA proxy:")
    print(f"  Pearson:  r={corr_pearson:+.4f}, p={p_pearson:.2e}")
    print(f"  Spearman: r={corr_spearman:+.4f}, p={p_spearman:.2e}")
    
    if abs(corr_pearson) < 0.3:
        print("\n  → WEAK correlation - TACE alpha encodes different information than PCA!")
    elif corr_pearson > 0.5:
        print("\n  → STRONG positive correlation - TACE alpha related to PCA magnitude")
    
    print(f"\nPCA Proxy Statistics:")
    print(f"  Range: [{pca_proxy.min():.4f}, {pca_proxy.max():.4f}]")
    print(f"  Mean: {pca_proxy.mean():.4f} ± {pca_proxy.std():.4f}")
    
    return {
        'correlation_pearson': float(corr_pearson),
        'correlation_p_pearson': float(p_pearson),
        'correlation_spearman': float(corr_spearman),
        'correlation_p_spearman': float(p_spearman)
    }


def save_results(adata, alphas, analysis, comparison):
    """Save results to file."""
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Add alpha to adata.obs
    adata.obs['alpha_tace'] = alphas
    
    # Save updated adata
    print(f"\nSaving AnnData with TACE alphas to: {OUTPUT_PATH}")
    adata.write_h5ad(OUTPUT_PATH)
    print("  ✓ Saved")
    
    # Save analysis summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'method': 'TACE Alpha Computation',
        'parameters': {
            'qac_num_qubits': QAC_NUM_QUBITS,
            'qac_layers': QAC_LAYERS,
            'qac_reinforce_rate': QAC_REINFORCE_RATE,
            'tace_collapse_threshold': TACE_COLLAPSE_THRESHOLD,
            'tace_feedback_gain': TACE_FEEDBACK_GAIN,
            'tace_max_iterations': TACE_MAX_ITERATIONS,
            'default_alpha': DEFAULT_ALPHA
        },
        'n_cells': int(adata.n_obs),
        'n_clusters': int(adata.obs['leiden'].nunique()),
        'alpha_distribution': analysis,
        'pca_comparison': comparison
    }
    
    summary_path = './tace_alpha_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaving analysis summary to: {summary_path}")
    print("  ✓ Saved")


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    print("="*80)
    print("COMPUTING REAL TACE ALPHA VALUES FOR PBMC3K")
    print("="*80)
    print("\nPrinciples:")
    print("✓ Using DEFAULT TACE/QAC parameters (no tuning)")
    print("✓ Computing alpha for ALL 2700 cells")
    print("✓ Using DON-GPU normalize (fractal clustering) FIRST")
    print("✓ Following cellular_discovery.py methodology exactly")
    print("✓ Letting TACE engine determine values (no bias)")
    print("="*80)
    
    # Load data
    adata = load_pbmc_data()
    
    # Compute TACE alpha for all cells (DON-GPU + TACE pipeline)
    alphas = compute_all_tace_alphas(adata, batch_size=100)
    
    # Analyze distribution
    analysis = analyze_alpha_distribution(alphas, adata)
    
    # Compare with PCA proxy
    comparison = compare_with_pca_proxy(alphas, adata)
    
    # Save results
    save_results(adata, alphas, analysis, comparison)
    
    print("\n" + "="*80)
    print("COMPUTATION COMPLETE")
    print("="*80)
    print(f"\nNext step: Re-run alpha_stability_exploration.py with real alphas")
    print(f"  → Use {OUTPUT_PATH} as input")
    print(f"  → Compare results with PCA-proxy analysis")
    print("="*80)


if __name__ == '__main__':
    main()
