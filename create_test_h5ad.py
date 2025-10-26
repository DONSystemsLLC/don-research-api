#!/usr/bin/env python3
"""
Create a small test H5AD file for bio module testing.
"""

import numpy as np
import pandas as pd
import anndata as ad
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

# Create small dataset
n_cells = 100
n_genes = 50

# Random expression matrix
X = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes)).astype(float)

# Gene names
gene_names = [f"GENE{i}" for i in range(n_genes)]

# Cell metadata
obs = pd.DataFrame({
    'cell_id': [f"CELL{i}" for i in range(n_cells)],
    'batch': np.random.choice(['batch1', 'batch2'], n_cells),
})
obs.index = obs['cell_id']

# Gene metadata
var = pd.DataFrame({
    'gene_name': gene_names
})
var.index = gene_names

# Create AnnData object
adata = ad.AnnData(X=X, obs=obs, var=var)

# Add mock clustering results
adata.obs['leiden'] = np.random.choice(['0', '1', '2', '3'], n_cells)

# Add mock UMAP embedding (2D)
adata.obsm['X_umap'] = np.random.randn(n_cells, 2)

# Add mock PCA (for latent space)
adata.obsm['X_pca'] = np.random.randn(n_cells, 10)

# Calculate percentage mitochondrial (mock)
adata.obs['pct_counts_mt'] = np.random.uniform(0, 15, n_cells)

# Save to test_data directory
output_dir = Path("test_data")
output_dir.mkdir(exist_ok=True)

output_path = output_dir / "pbmc_small.h5ad"
adata.write_h5ad(output_path)

print(f"✅ Created test H5AD file: {output_path}")
print(f"   Cells: {n_cells}")
print(f"   Genes: {n_genes}")
print(f"   Clusters: {adata.obs['leiden'].nunique()}")
print(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")
