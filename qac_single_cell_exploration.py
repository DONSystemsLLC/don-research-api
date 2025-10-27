#!/usr/bin/env python3
"""
QAC Single-Cell Exploration
============================

Research Question:
What happens when we apply QAC to individual cells instead of clusters?

Hypotheses to Test:
1. Does QAC stability vary across individual cells?
2. Do low-alpha vs high-alpha cells show different QAC stability?
3. Is quantum stability cell-intrinsic or emergent from populations?

Approach:
- Apply QAC to single-cell gene expression (first 100 genes)
- Compare stability across alpha regimes (low/mid/high)
- Test correlation between QAC stability and alpha values
- NO BIAS: use DEFAULT parameters, analyze ALL sampled cells

Dataset: PBMC3K with real TACE alphas
"""

import sys
from pathlib import Path
import numpy as np
import scanpy as sc
from scipy.stats import pearsonr, spearmanr, mannwhitneyu
import json
from datetime import datetime

# Add stack to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'stack'))

from tace.core import QACEngine

# ============================================================================
# Configuration (DEFAULTS ONLY)
# ============================================================================

PBMC_PATH = "./data/pbmc3k_with_tace_alpha.h5ad"
N_GENES = 100  # First 100 genes (no cherry-picking)
N_CELLS_SAMPLE = 500  # Sample 500 cells for computational feasibility
RANDOM_SEED = 42  # For reproducibility

# QAC defaults
QAC_LAYERS = 3
QAC_REINFORCE = 0.05
QAC_QUBITS = 8

# ============================================================================
# Core Functions
# ============================================================================

def load_data_with_sampling():
    """Load PBMC3K and sample cells for analysis."""
    print("Loading PBMC3K with TACE alphas...")
    adata = sc.read_h5ad(PBMC_PATH)
    
    print(f"Total dataset: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Check alpha distribution
    if 'alpha_tace' not in adata.obs:
        raise ValueError("TACE alphas not found! Run compute_real_alphas.py first.")
    
    alphas = adata.obs['alpha_tace'].values
    print(f"\nAlpha distribution:")
    print(f"  Range: [{alphas.min():.3f}, {alphas.max():.3f}]")
    print(f"  Mean: {alphas.mean():.3f} ± {alphas.std():.3f}")
    
    low = (alphas < 0.3).sum()
    mid = ((alphas >= 0.3) & (alphas <= 0.7)).sum()
    high = (alphas > 0.7).sum()
    print(f"  Low (<0.3):   {low:4d} cells ({100*low/len(alphas):.1f}%)")
    print(f"  Mid (0.3-0.7): {mid:4d} cells ({100*mid/len(alphas):.1f}%)")
    print(f"  High (>0.7):  {high:4d} cells ({100*high/len(alphas):.1f}%)")
    
    # Sample cells (stratified by alpha regime for balanced representation)
    np.random.seed(RANDOM_SEED)
    
    low_idx = np.where(alphas < 0.3)[0]
    mid_idx = np.where((alphas >= 0.3) & (alphas <= 0.7))[0]
    high_idx = np.where(alphas > 0.7)[0]
    
    # Sample proportionally
    n_low = min(len(low_idx), int(N_CELLS_SAMPLE * 0.36))
    n_mid = min(len(mid_idx), int(N_CELLS_SAMPLE * 0.17))
    n_high = min(len(high_idx), int(N_CELLS_SAMPLE * 0.47))
    
    sampled_low = np.random.choice(low_idx, n_low, replace=False)
    sampled_mid = np.random.choice(mid_idx, n_mid, replace=False)
    sampled_high = np.random.choice(high_idx, n_high, replace=False)
    
    sampled_idx = np.concatenate([sampled_low, sampled_mid, sampled_high])
    np.random.shuffle(sampled_idx)
    
    adata_sample = adata[sampled_idx, :].copy()
    
    print(f"\nSampled {len(sampled_idx)} cells:")
    print(f"  Low-alpha:  {n_low} cells")
    print(f"  Mid-alpha:  {n_mid} cells")
    print(f"  High-alpha: {n_high} cells")
    
    return adata_sample


def apply_qac_to_single_cell(cell_expr, gene_names, qac_engine):
    """
    Apply QAC to a single cell's gene expression.
    
    For each gene:
    - Take expression value across all genes (8-dimensional input to QAC)
    - Apply QAC stabilization
    - Measure stability as change magnitude
    
    Returns per-gene stability scores for this cell.
    """
    results = []
    
    # We'll apply QAC to each gene's value in context of other genes
    # Take first N_GENES genes
    gene_expr = cell_expr[:N_GENES]
    
    # For single-cell QAC, we treat the gene expression vector as a quantum state
    # Apply QAC to the full vector (not per-gene)
    original_vector = gene_expr.copy()
    stabilized_vector = qac_engine.stabilize(original_vector)
    
    # Compute per-gene stability
    for gene_idx, gene_name in enumerate(gene_names):
        original_val = original_vector[gene_idx]
        stabilized_val = stabilized_vector[gene_idx] if gene_idx < len(stabilized_vector) else 0.0
        
        change = abs(stabilized_val - original_val)
        relative_change = change / (abs(original_val) + 1e-10)
        stability_score = 1.0 - relative_change
        
        results.append({
            'gene': gene_name,
            'original': float(original_val),
            'stabilized': float(stabilized_val),
            'change': float(change),
            'relative_change': float(relative_change),
            'stability': float(stability_score)
        })
    
    return results


def compute_single_cell_qac(adata_sample):
    """Apply QAC to individual cells."""
    print("\n" + "="*80)
    print("APPLYING QAC TO INDIVIDUAL CELLS")
    print("="*80)
    print(f"\nParameters:")
    print(f"  QAC: layers={QAC_LAYERS}, reinforce_rate={QAC_REINFORCE}, num_qubits={QAC_QUBITS}")
    print(f"  Analyzing first {N_GENES} genes per cell")
    print()
    
    qac = QACEngine(
        num_qubits=QAC_QUBITS,
        reinforce_rate=QAC_REINFORCE,
        layers=QAC_LAYERS
    )
    
    gene_names = adata_sample.var_names[:N_GENES].tolist()
    n_cells = adata_sample.n_obs
    
    # Get expression matrix
    X = adata_sample.X.toarray() if hasattr(adata_sample.X, 'toarray') else adata_sample.X
    
    # Store results
    cell_stability_scores = []  # Average stability per cell
    gene_stability_by_cell = {gene: [] for gene in gene_names}  # Per-gene across cells
    
    print(f"Processing {n_cells} cells...")
    
    for i in range(n_cells):
        cell_expr = X[i, :]
        
        # Apply QAC to this cell
        cell_results = apply_qac_to_single_cell(cell_expr, gene_names, qac)
        
        # Compute average stability for this cell
        avg_stability = np.mean([r['stability'] for r in cell_results])
        cell_stability_scores.append(avg_stability)
        
        # Store per-gene stability
        for result in cell_results:
            gene_stability_by_cell[result['gene']].append(result['stability'])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{n_cells} cells "
                  f"(avg stability: {np.mean(cell_stability_scores):.4f})")
    
    print(f"\n✓ Completed QAC analysis for {n_cells} cells")
    
    return cell_stability_scores, gene_stability_by_cell, gene_names


def analyze_stability_by_alpha(adata_sample, cell_stabilities):
    """Analyze how QAC stability varies with alpha regime."""
    print("\n" + "="*80)
    print("QAC STABILITY vs ALPHA REGIME")
    print("="*80)
    
    alphas = adata_sample.obs['alpha_tace'].values
    stabilities = np.array(cell_stabilities)
    
    # Overall correlation
    corr_pearson, p_pearson = pearsonr(alphas, stabilities)
    corr_spearman, p_spearman = spearmanr(alphas, stabilities)
    
    print(f"\nCorrelation (Alpha vs Cell-Average Stability):")
    print(f"  Pearson:  r={corr_pearson:+.4f}, p={p_pearson:.2e}")
    print(f"  Spearman: r={corr_spearman:+.4f}, p={p_spearman:.2e}")
    
    if abs(corr_pearson) < 0.1:
        print("\n  → WEAK/NO correlation - stability independent of alpha")
    elif corr_pearson > 0.3:
        print("\n  → POSITIVE correlation - high-alpha cells more stable")
    elif corr_pearson < -0.3:
        print("\n  → NEGATIVE correlation - low-alpha cells more stable")
    
    # By regime
    low_mask = alphas < 0.3
    mid_mask = (alphas >= 0.3) & (alphas <= 0.7)
    high_mask = alphas > 0.7
    
    low_stab = stabilities[low_mask]
    mid_stab = stabilities[mid_mask]
    high_stab = stabilities[high_mask]
    
    print(f"\nStability by Alpha Regime:")
    print(f"  Low-alpha (<0.3):   {low_stab.mean():.4f} ± {low_stab.std():.4f} (n={len(low_stab)})")
    print(f"  Mid-alpha (0.3-0.7): {mid_stab.mean():.4f} ± {mid_stab.std():.4f} (n={len(mid_stab)})")
    print(f"  High-alpha (>0.7):  {high_stab.mean():.4f} ± {high_stab.std():.4f} (n={len(high_stab)})")
    
    # Statistical test: low vs high
    if len(low_stab) > 0 and len(high_stab) > 0:
        u_stat, p_val = mannwhitneyu(low_stab, high_stab, alternative='two-sided')
        print(f"\nMann-Whitney U test (Low vs High):")
        print(f"  U={u_stat:.0f}, p={p_val:.2e}")
        
        if p_val < 0.05:
            diff = high_stab.mean() - low_stab.mean()
            direction = "MORE" if diff > 0 else "LESS"
            print(f"  → SIGNIFICANT: High-alpha cells {direction} stable than low-alpha")
        else:
            print(f"  → NOT significant - no difference between regimes")
    
    return {
        'correlation_pearson': float(corr_pearson),
        'correlation_p': float(p_pearson),
        'correlation_spearman': float(corr_spearman),
        'low_alpha_stability': {
            'mean': float(low_stab.mean()),
            'std': float(low_stab.std()),
            'n': int(len(low_stab))
        },
        'mid_alpha_stability': {
            'mean': float(mid_stab.mean()),
            'std': float(mid_stab.std()),
            'n': int(len(mid_stab))
        },
        'high_alpha_stability': {
            'mean': float(high_stab.mean()),
            'std': float(high_stab.std()),
            'n': int(len(high_stab))
        }
    }


def analyze_gene_stability_variance(gene_stability_by_cell, gene_names):
    """Analyze which genes show most variable stability across cells."""
    print("\n" + "="*80)
    print("GENE-LEVEL STABILITY VARIANCE")
    print("="*80)
    
    gene_stats = []
    
    for gene in gene_names:
        stabilities = np.array(gene_stability_by_cell[gene])
        
        gene_stats.append({
            'gene': gene,
            'mean_stability': float(stabilities.mean()),
            'std_stability': float(stabilities.std()),
            'min_stability': float(stabilities.min()),
            'max_stability': float(stabilities.max()),
            'cv_stability': float(stabilities.std() / (stabilities.mean() + 1e-10))  # Coefficient of variation
        })
    
    # Sort by coefficient of variation (most variable)
    gene_stats_sorted = sorted(gene_stats, key=lambda x: x['cv_stability'], reverse=True)
    
    print(f"\nMost Variable Stability Across Cells (top 10):")
    for i, stats in enumerate(gene_stats_sorted[:10], 1):
        print(f"  {i}. {stats['gene']}: "
              f"mean={stats['mean_stability']:.4f}, "
              f"std={stats['std_stability']:.4f}, "
              f"CV={stats['cv_stability']:.4f}")
    
    print(f"\nMost Consistent Stability Across Cells (bottom 10):")
    for i, stats in enumerate(gene_stats_sorted[-10:][::-1], 1):
        print(f"  {i}. {stats['gene']}: "
              f"mean={stats['mean_stability']:.4f}, "
              f"std={stats['std_stability']:.4f}, "
              f"CV={stats['cv_stability']:.4f}")
    
    return gene_stats


def compare_cluster_vs_single_cell():
    """Compare cluster-level vs single-cell QAC results."""
    print("\n" + "="*80)
    print("COMPARISON: Cluster-Level vs Single-Cell QAC")
    print("="*80)
    
    # Load cluster results
    try:
        with open('./qac_cluster_genomics_results.json', 'r') as f:
            cluster_results = json.load(f)
        
        print("\nCluster-level results loaded:")
        print(f"  Analyzed {cluster_results['n_genes']} genes")
        print(f"  Method: {cluster_results['method']}")
        
        # Compare top genes
        cluster_genes = cluster_results['results']
        cluster_top = sorted(cluster_genes, key=lambda x: x['stability_score'], reverse=True)[:10]
        
        print(f"\nTop 10 most stable genes (cluster-level):")
        for i, g in enumerate(cluster_top, 1):
            print(f"  {i}. {g['gene']}: stability={g['stability_score']:.4f}")
        
    except FileNotFoundError:
        print("\nCluster-level results not found (run qac_cluster_genomics.py first)")


def save_results(adata_sample, cell_stabilities, gene_stats, alpha_analysis):
    """Save results to JSON."""
    output = {
        'timestamp': datetime.now().isoformat(),
        'method': 'Single-Cell QAC Stability Analysis',
        'parameters': {
            'qac_layers': QAC_LAYERS,
            'qac_reinforce_rate': QAC_REINFORCE,
            'qac_num_qubits': QAC_QUBITS,
            'n_genes': N_GENES,
            'n_cells_sampled': len(adata_sample)
        },
        'alpha_analysis': alpha_analysis,
        'gene_stability_stats': gene_stats,
        'cell_stabilities': {
            'mean': float(np.mean(cell_stabilities)),
            'std': float(np.std(cell_stabilities)),
            'min': float(np.min(cell_stabilities)),
            'max': float(np.max(cell_stabilities))
        }
    }
    
    output_path = './qac_single_cell_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_path}")


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("="*80)
    print("QAC SINGLE-CELL EXPLORATION")
    print("="*80)
    print("\nResearch Questions:")
    print("1. Does QAC stability vary across individual cells?")
    print("2. Do low-alpha vs high-alpha cells show different stability?")
    print("3. Is quantum stability cell-intrinsic or emergent?")
    print("\nPrinciples:")
    print("✓ NO parameter tuning (DEFAULT QAC parameters)")
    print("✓ NO gene pre-filtering (first 100 genes)")
    print("✓ Stratified sampling (balanced alpha regimes)")
    print("✓ Letting math speak (no bias)")
    print("="*80)
    
    # Load and sample data
    adata_sample = load_data_with_sampling()
    
    # Apply QAC to single cells
    cell_stabilities, gene_stability_by_cell, gene_names = compute_single_cell_qac(adata_sample)
    
    # Analyze stability by alpha regime
    alpha_analysis = analyze_stability_by_alpha(adata_sample, cell_stabilities)
    
    # Analyze gene-level variance
    gene_stats = analyze_gene_stability_variance(gene_stability_by_cell, gene_names)
    
    # Compare with cluster-level results
    compare_cluster_vs_single_cell()
    
    # Save results
    save_results(adata_sample, cell_stabilities, gene_stats, alpha_analysis)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE - Math has spoken!")
    print("="*80)


if __name__ == '__main__':
    main()
