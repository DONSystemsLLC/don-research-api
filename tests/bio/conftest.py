"""
Fixtures for bio module tests
"""

import pytest
import numpy as np
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix


@pytest.fixture
def small_adata():
    """Create a small AnnData object for testing"""
    # Create synthetic data
    n_obs = 100
    n_vars = 50
    
    # Random expression matrix
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars))
    X = csr_matrix(X.astype(np.float32))
    
    # Create AnnData
    adata = ad.AnnData(X)
    adata.obs_names = [f"cell_{i}" for i in range(n_obs)]
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    
    # Add basic QC metrics
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    
    # Add MT percentage (simulated)
    adata.obs['pct_counts_mt'] = np.random.rand(n_obs) * 10
    
    return adata


@pytest.fixture
def small_adata_with_paga(small_adata):
    """Create AnnData with PAGA connectivity"""
    # Add clustering
    clusters = ['0', '1', '2'] * (small_adata.n_obs // 3)
    clusters += ['0'] * (small_adata.n_obs - len(clusters))  # Fill remainder
    small_adata.obs['leiden'] = clusters[:small_adata.n_obs]
    small_adata.obs['leiden'] = small_adata.obs['leiden'].astype('category')
    
    # Add PCA (without highly variable genes to avoid scikit-misc dependency)
    sc.pp.normalize_total(small_adata, target_sum=1e4)
    sc.pp.log1p(small_adata)
    # Skip highly_variable_genes (requires scikit-misc)
    sc.pp.pca(small_adata, n_comps=10)
    
    # Compute neighbors
    sc.pp.neighbors(small_adata, n_neighbors=10, n_pcs=10)
    
    # Compute PAGA
    sc.tl.paga(small_adata, groups='leiden')
    
    return small_adata


@pytest.fixture
def bio_test_data(tmpdir):
    """Create test data directory structure"""
    test_dir = tmpdir.mkdir("bio_test_data")
    
    # Create subdirectories
    artifacts_dir = test_dir.mkdir("artifacts")
    runs_dir = test_dir.mkdir("runs")
    
    return {
        "root": str(test_dir),
        "artifacts": str(artifacts_dir),
        "runs": str(runs_dir)
    }
