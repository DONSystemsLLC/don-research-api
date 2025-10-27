#!/usr/bin/env python3
"""
Cellular Discovery: Let the Data Speak

Direct analysis of PBMC3K single-cell data using DON Stack.
No theoretical assumptions - just observe what emerges.
"""

import sys
import numpy as np
import scanpy as sc
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from don_memory.adapters.don_stack_adapter import DONStackAdapter

def print_section(title):
    """Print section divider"""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")

# ============================================================================
# LOAD REAL CELLULAR DATA
# ============================================================================
print_section("📊 LOADING PBMC3K DATA")

adata = sc.read_h5ad("data/pbmc3k.h5ad")
print(f"Dataset: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"Shape: {adata.shape}")

# Basic preprocessing
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
print(f"Using {adata.var['highly_variable'].sum()} highly variable genes")

# Dimensionality reduction
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata)
sc.tl.leiden(adata, resolution=0.5)
print(f"Identified {len(adata.obs['leiden'].unique())} clusters")

# Get cluster sizes
cluster_sizes = adata.obs['leiden'].value_counts().sort_index()
print(f"\nCluster distribution:")
for cluster, count in cluster_sizes.items():
    pct = 100 * count / len(adata)
    print(f"  Cluster {cluster}: {count:4d} cells ({pct:5.2f}%)")

# ============================================================================
# EXPLORATION 1: DON Stack Processing per Cluster
# ============================================================================
print_section("🔬 EXPLORATION 1: Cluster-Level Analysis")

adapter = DONStackAdapter()

cluster_data = []

for cluster_id in sorted(adata.obs['leiden'].unique()):
    # Get cells in this cluster
    mask = adata.obs['leiden'] == cluster_id
    cluster_cells = adata[mask]
    n_cells = mask.sum()
    
    # Get mean expression vector (cluster centroid)
    if hasattr(cluster_cells.X, 'toarray'):
        mean_expr = np.array(cluster_cells.X.toarray().mean(axis=0)).flatten()
    else:
        mean_expr = np.array(cluster_cells.X.mean(axis=0)).flatten()
    
    # Use PCA representation for DON Stack
    pca_centroid = cluster_cells.obsm['X_pca'].mean(axis=0)
    
    # Process with DON-GPU (normalize)
    normalized = adapter.normalize(pca_centroid[:32])  # Use first 32 PCs
    
    # Process with TACE (tune alpha)
    alpha = adapter.tune_alpha(normalized[:8].tolist(), default_alpha=0.5)
    
    # Calculate cluster statistics
    expr_mean = mean_expr.mean()
    expr_std = mean_expr.std()
    expr_cv = expr_std / (expr_mean + 1e-10)  # Coefficient of variation
    
    cluster_data.append({
        'cluster': cluster_id,
        'n_cells': n_cells,
        'pct': 100 * n_cells / len(adata),
        'alpha': alpha,
        'expr_mean': expr_mean,
        'expr_std': expr_std,
        'expr_cv': expr_cv,
        'normalized_norm': np.linalg.norm(normalized)
    })
    
    print(f"Cluster {cluster_id} ({n_cells} cells, {100*n_cells/len(adata):.1f}%):")
    print(f"  Expression: mean={expr_mean:.4f}, std={expr_std:.4f}, CV={expr_cv:.4f}")
    print(f"  DON-GPU norm: {np.linalg.norm(normalized):.4f}")
    print(f"  TACE alpha: {alpha:.6f}")
    print()

# ============================================================================
# EXPLORATION 2: What Correlates with Alpha?
# ============================================================================
print_section("📈 EXPLORATION 2: Alpha Correlation Analysis")

n_cells_arr = np.array([c['n_cells'] for c in cluster_data])
alpha_arr = np.array([c['alpha'] for c in cluster_data])
expr_mean_arr = np.array([c['expr_mean'] for c in cluster_data])
expr_cv_arr = np.array([c['expr_cv'] for c in cluster_data])
norm_arr = np.array([c['normalized_norm'] for c in cluster_data])

print("Correlations with TACE alpha:\n")

correlations = {}

if len(alpha_arr) > 2:
    corr_size = np.corrcoef(n_cells_arr, alpha_arr)[0,1]
    correlations['Cell Count'] = corr_size
    print(f"Cell Count ↔ Alpha: {corr_size:+.4f}")
    
    corr_mean = np.corrcoef(expr_mean_arr, alpha_arr)[0,1]
    correlations['Expression Level'] = corr_mean
    print(f"Expression Level ↔ Alpha: {corr_mean:+.4f}")
    
    corr_cv = np.corrcoef(expr_cv_arr, alpha_arr)[0,1]
    correlations['Variability (CV)'] = corr_cv
    print(f"Variability (CV) ↔ Alpha: {corr_cv:+.4f}")
    
    corr_norm = np.corrcoef(norm_arr, alpha_arr)[0,1]
    correlations['DON-GPU Norm'] = corr_norm
    print(f"DON-GPU Norm ↔ Alpha: {corr_norm:+.4f}")
    
    print(f"\nAlpha statistics:")
    print(f"  Range: [{alpha_arr.min():.6f}, {alpha_arr.max():.6f}]")
    print(f"  Mean: {alpha_arr.mean():.6f}")
    print(f"  Std Dev: {alpha_arr.std():.6f}")
    
    if alpha_arr.std() < 0.001:
        print(f"  → Nearly constant across all clusters")
    else:
        print(f"  → Varies across clusters")
        strongest = max(correlations.items(), key=lambda x: abs(x[1]))
        print(f"  → Strongest correlation: {strongest[0]} ({strongest[1]:+.4f})")

# ============================================================================
# EXPLORATION 3: Rare vs Common Cell Populations
# ============================================================================
print_section("🔍 EXPLORATION 3: Population Size Effects")

# Sort by size
sorted_clusters = sorted(cluster_data, key=lambda x: x['n_cells'])

print("Rarest clusters:")
for c in sorted_clusters[:3]:
    print(f"  C{c['cluster']}: {c['n_cells']:4d} cells ({c['pct']:5.2f}%), α={c['alpha']:.6f}")

print(f"\nMost common clusters:")
for c in sorted_clusters[-3:]:
    print(f"  C{c['cluster']}: {c['n_cells']:4d} cells ({c['pct']:5.2f}%), α={c['alpha']:.6f}")

# Compare alpha values
rare_alphas = [c['alpha'] for c in sorted_clusters[:3]]
common_alphas = [c['alpha'] for c in sorted_clusters[-3:]]

print(f"\nAlpha comparison:")
print(f"  Rare cells: mean={np.mean(rare_alphas):.6f}, std={np.std(rare_alphas):.6f}")
print(f"  Common cells: mean={np.mean(common_alphas):.6f}, std={np.std(common_alphas):.6f}")
print(f"  Difference: {np.mean(rare_alphas) - np.mean(common_alphas):+.6f}")

# ============================================================================
# EXPLORATION 4: Cluster-to-Cluster Relationships
# ============================================================================
print_section("🔗 EXPLORATION 4: Inter-Cluster Relationships")

print("Computing pairwise similarities...\n")

# Get PCA centroids for all clusters
centroids = []
for cluster_id in sorted(adata.obs['leiden'].unique()):
    mask = adata.obs['leiden'] == cluster_id
    cluster_cells = adata[mask]
    pca_centroid = cluster_cells.obsm['X_pca'].mean(axis=0)
    centroids.append(pca_centroid[:32])

# Compute similarity matrix
n_clusters = len(centroids)
similarity_matrix = np.zeros((n_clusters, n_clusters))

for i in range(n_clusters):
    for j in range(n_clusters):
        if i != j:
            # Cosine similarity
            dot = np.dot(centroids[i], centroids[j])
            norm_i = np.linalg.norm(centroids[i])
            norm_j = np.linalg.norm(centroids[j])
            similarity_matrix[i, j] = dot / (norm_i * norm_j + 1e-10)

print("Similarity matrix (top values):")
print("  (Only showing similarities > 0.9)\n")

high_sim_pairs = []
for i in range(n_clusters):
    for j in range(i+1, n_clusters):
        sim = similarity_matrix[i, j]
        if sim > 0.9:
            high_sim_pairs.append((i, j, sim))
            print(f"  C{i} ↔ C{j}: {sim:.4f}")

if not high_sim_pairs:
    print("  (No pairs with similarity > 0.9)")
    print("\n  Showing highest similarities:")
    all_pairs = []
    for i in range(n_clusters):
        for j in range(i+1, n_clusters):
            all_pairs.append((i, j, similarity_matrix[i, j]))
    all_pairs.sort(key=lambda x: x[2], reverse=True)
    for i, j, sim in all_pairs[:5]:
        print(f"  C{i} ↔ C{j}: {sim:.4f}")

# ============================================================================
# EXPLORATION 5: What Happens with Cell-Level Data?
# ============================================================================
print_section("🧬 EXPLORATION 5: Individual Cell Analysis")

print("Testing DON Stack on individual cells...\n")

# Sample cells from different clusters
sample_cells = []
for cluster_id in sorted(adata.obs['leiden'].unique())[:3]:  # First 3 clusters
    mask = adata.obs['leiden'] == cluster_id
    cluster_indices = np.where(mask)[0]
    
    # Sample one cell
    if len(cluster_indices) > 0:
        cell_idx = cluster_indices[0]
        sample_cells.append((cluster_id, cell_idx))

for cluster_id, cell_idx in sample_cells:
    cell_pca = adata.obsm['X_pca'][cell_idx, :32]
    
    # Process with DON Stack
    normalized = adapter.normalize(cell_pca)
    alpha = adapter.tune_alpha(normalized[:8].tolist(), 0.5)
    
    print(f"Cell from Cluster {cluster_id}:")
    print(f"  PCA norm: {np.linalg.norm(cell_pca):.4f}")
    print(f"  DON-GPU norm: {np.linalg.norm(normalized):.4f}")
    print(f"  TACE alpha: {alpha:.6f}")
    print()

# ============================================================================
# SYNTHESIS: What Did We Discover?
# ============================================================================
print_section("💡 DISCOVERIES")

print("What the data reveals:\n")

print("1. CLUSTER STRUCTURE")
print(f"   • {len(cluster_sizes)} distinct cell populations identified")
print(f"   • Size range: {n_cells_arr.min()}-{n_cells_arr.max()} cells")
print(f"   • Largest cluster: {100*n_cells_arr.max()/len(adata):.1f}% of cells")
print(f"   • Smallest cluster: {100*n_cells_arr.min()/len(adata):.1f}% of cells")

print("\n2. TACE ALPHA PATTERNS")
print(f"   • Alpha range: [{alpha_arr.min():.6f}, {alpha_arr.max():.6f}]")
print(f"   • Variation: {alpha_arr.std():.6f}")
if alpha_arr.std() < 0.001:
    print(f"   • Interpretation: Uniform across cell types")
else:
    print(f"   • Interpretation: Cell-type dependent")
    if len(correlations) > 0:
        strongest = max(correlations.items(), key=lambda x: abs(x[1]))
        print(f"   • Strongest predictor: {strongest[0]} (r={strongest[1]:+.4f})")

print("\n3. POPULATION DIFFERENCES")
rare_common_diff = np.mean(rare_alphas) - np.mean(common_alphas)
if abs(rare_common_diff) > 0.01:
    direction = "higher" if rare_common_diff > 0 else "lower"
    print(f"   • Rare cells show {direction} alpha ({rare_common_diff:+.4f})")
    print(f"   • Population size affects TACE response")
else:
    print(f"   • No significant difference between rare/common")
    print(f"   • Population size independent")

print("\n4. CLUSTER RELATIONSHIPS")
if high_sim_pairs:
    print(f"   • Found {len(high_sim_pairs)} highly similar cluster pairs")
    print(f"   • Suggests related cell lineages or states")
else:
    print(f"   • Clusters are well-separated")
    print(f"   • Distinct cell populations")

print("\n" + "="*70)
print("Analysis complete.")
print("="*70 + "\n")
