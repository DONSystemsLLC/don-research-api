#!/usr/bin/env python3
"""
Deep Dive: Individual Cell Patterns

The cluster centroids showed uniform alpha, but individual cells showed variation.
Let's explore what's happening at the single-cell level.
"""

import sys
import numpy as np
import scanpy as sc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))
from don_memory.adapters.don_stack_adapter import DONStackAdapter

print("="*70)
print("SINGLE-CELL DEEP DIVE")
print("="*70 + "\n")

# Load data
adata = sc.read_h5ad("data/pbmc3k.h5ad")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.5)

print(f"Dataset: {adata.n_obs} cells in {len(adata.obs['leiden'].unique())} clusters\n")

adapter = DONStackAdapter()

# ============================================================================
# Sample cells from each cluster
# ============================================================================
print("="*70)
print("SAMPLING CELLS FROM EACH CLUSTER")
print("="*70 + "\n")

cells_per_cluster = 10
cell_data = []

for cluster_id in sorted(adata.obs['leiden'].unique()):
    mask = adata.obs['leiden'] == cluster_id
    cluster_indices = np.where(mask)[0]
    
    # Sample multiple cells
    n_sample = min(cells_per_cluster, len(cluster_indices))
    sampled = np.random.choice(cluster_indices, n_sample, replace=False)
    
    print(f"Cluster {cluster_id} ({mask.sum()} cells) - sampling {n_sample} cells:")
    
    cluster_alphas = []
    for cell_idx in sampled:
        cell_pca = adata.obsm['X_pca'][cell_idx, :32]
        normalized = adapter.normalize(cell_pca)
        alpha = adapter.tune_alpha(normalized[:8].tolist(), 0.5)
        
        cluster_alphas.append(alpha)
        cell_data.append({
            'cluster': cluster_id,
            'cell_idx': cell_idx,
            'alpha': alpha,
            'pca_norm': np.linalg.norm(cell_pca)
        })
    
    print(f"  Alpha range: [{min(cluster_alphas):.6f}, {max(cluster_alphas):.6f}]")
    print(f"  Alpha mean: {np.mean(cluster_alphas):.6f}")
    print(f"  Alpha std: {np.std(cluster_alphas):.6f}")
    print()

# ============================================================================
# Overall distribution
# ============================================================================
print("="*70)
print("OVERALL ALPHA DISTRIBUTION")
print("="*70 + "\n")

all_alphas = np.array([c['alpha'] for c in cell_data])

print(f"Total cells sampled: {len(all_alphas)}")
print(f"Alpha range: [{all_alphas.min():.6f}, {all_alphas.max():.6f}]")
print(f"Alpha mean: {all_alphas.mean():.6f}")
print(f"Alpha std: {all_alphas.std():.6f}")
print(f"Alpha median: {np.median(all_alphas):.6f}")

# Check for distinct values
unique_alphas = np.unique(all_alphas)
print(f"\nDistinct alpha values: {len(unique_alphas)}")
if len(unique_alphas) <= 20:
    print("Values:")
    for val in unique_alphas:
        count = np.sum(all_alphas == val)
        pct = 100 * count / len(all_alphas)
        print(f"  {val:.6f}: {count:3d} cells ({pct:5.1f}%)")

# ============================================================================
# Correlation with PCA norm
# ============================================================================
print("\n" + "="*70)
print("WHAT PREDICTS ALPHA?")
print("="*70 + "\n")

pca_norms = np.array([c['pca_norm'] for c in cell_data])

corr = np.corrcoef(pca_norms, all_alphas)[0,1]
print(f"PCA Norm ↔ Alpha: {corr:+.4f}")

if abs(corr) > 0.3:
    print(f"→ Moderate to strong correlation")
    print(f"→ Cell 'magnitude' affects alpha")
else:
    print(f"→ Weak correlation")
    print(f"→ Alpha not predicted by simple magnitude")

# ============================================================================
# Look at extreme cases
# ============================================================================
print("\n" + "="*70)
print("EXTREME CASES")
print("="*70 + "\n")

# Lowest alphas
sorted_by_alpha = sorted(cell_data, key=lambda x: x['alpha'])

print("LOWEST ALPHA cells:")
for c in sorted_by_alpha[:5]:
    print(f"  Cell {c['cell_idx']} (Cluster {c['cluster']}): α={c['alpha']:.6f}, norm={c['pca_norm']:.4f}")

print("\nHIGHEST ALPHA cells:")
for c in sorted_by_alpha[-5:]:
    print(f"  Cell {c['cell_idx']} (Cluster {c['cluster']}): α={c['alpha']:.6f}, norm={c['pca_norm']:.4f}")

# ============================================================================
# Check if it's bimodal
# ============================================================================
print("\n" + "="*70)
print("DISTRIBUTION ANALYSIS")
print("="*70 + "\n")

# Create histogram bins
bins = np.linspace(all_alphas.min(), all_alphas.max(), 11)
hist, bin_edges = np.histogram(all_alphas, bins=bins)

print("Alpha distribution (histogram):")
for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
    bar = "█" * int(40 * count / hist.max())
    print(f"  [{edge:.3f}-{bin_edges[i+1]:.3f}]: {bar} {count}")

# Check for bimodality
low_threshold = 0.3
high_threshold = 0.7

n_low = np.sum(all_alphas < low_threshold)
n_high = np.sum(all_alphas > high_threshold)
n_mid = len(all_alphas) - n_low - n_high

print(f"\nCategorization:")
print(f"  Low alpha (<{low_threshold}): {n_low} cells ({100*n_low/len(all_alphas):.1f}%)")
print(f"  Mid alpha ({low_threshold}-{high_threshold}): {n_mid} cells ({100*n_mid/len(all_alphas):.1f}%)")
print(f"  High alpha (>{high_threshold}): {n_high} cells ({100*n_high/len(all_alphas):.1f}%)")

if n_low > 0 and n_high > 0 and n_mid < len(all_alphas) * 0.3:
    print("\n→ BIMODAL distribution detected!")
    print("→ Cells fall into two distinct alpha regimes")
else:
    print("\n→ Continuous distribution")

# ============================================================================
# What distinguishes low vs high alpha cells?
# ============================================================================
if n_low > 5 and n_high > 5:
    print("\n" + "="*70)
    print("LOW vs HIGH ALPHA CELLS")
    print("="*70 + "\n")
    
    low_alpha_cells = [c for c in cell_data if c['alpha'] < low_threshold]
    high_alpha_cells = [c for c in cell_data if c['alpha'] > high_threshold]
    
    low_norms = [c['pca_norm'] for c in low_alpha_cells]
    high_norms = [c['pca_norm'] for c in high_alpha_cells]
    
    print(f"Low alpha cells (n={len(low_alpha_cells)}):")
    print(f"  Mean norm: {np.mean(low_norms):.4f}")
    print(f"  Std norm: {np.std(low_norms):.4f}")
    
    print(f"\nHigh alpha cells (n={len(high_alpha_cells)}):")
    print(f"  Mean norm: {np.mean(high_norms):.4f}")
    print(f"  Std norm: {np.std(high_norms):.4f}")
    
    norm_diff = np.mean(high_norms) - np.mean(low_norms)
    print(f"\nDifference in norms: {norm_diff:+.4f}")
    
    if abs(norm_diff) > 1.0:
        direction = "stronger" if norm_diff > 0 else "weaker"
        print(f"→ High-alpha cells have {direction} PCA signatures")
    else:
        print(f"→ Similar PCA magnitudes")
    
    # Check cluster distribution
    low_clusters = [c['cluster'] for c in low_alpha_cells]
    high_clusters = [c['cluster'] for c in high_alpha_cells]
    
    print(f"\nCluster distribution:")
    print(f"  Low alpha: clusters {sorted(set(low_clusters))}")
    print(f"  High alpha: clusters {sorted(set(high_clusters))}")
    
    if set(low_clusters) != set(high_clusters):
        print(f"→ Different clusters show different alpha regimes")
    else:
        print(f"→ Both alpha regimes appear in same clusters")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70 + "\n")
