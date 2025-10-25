"""
Tests for parasite detector
TDD approach: Write tests first, then implement
"""

import pytest
import numpy as np
from .test_helpers import create_repeated_labels


def test_detect_parasites_basic(small_adata):
    """Test basic parasite detection"""
    from src.bio.parasite_detector import detect_parasites
    
    # Add required QC metrics if missing
    if 'n_genes_by_counts' not in small_adata.obs:
        small_adata.obs['n_genes_by_counts'] = np.random.randint(100, 1000, size=small_adata.n_obs)
    if 'total_counts' not in small_adata.obs:
        small_adata.obs['total_counts'] = np.random.randint(1000, 10000, size=small_adata.n_obs)
    if 'pct_counts_mt' not in small_adata.obs:
        small_adata.obs['pct_counts_mt'] = np.random.rand(small_adata.n_obs) * 20
    if 'batch' not in small_adata.obs:
        small_adata.obs['batch'] = create_repeated_labels(['batch1', 'batch2'], small_adata.n_obs)
    if 'leiden' not in small_adata.obs:
        small_adata.obs['leiden'] = create_repeated_labels(['0', '1'], small_adata.n_obs)
    
    result = detect_parasites(
        adata=small_adata,
        cluster_key="leiden",
        batch_key="batch"
    )
    
    # Assertions
    assert "n_cells" in result
    assert result["n_cells"] == small_adata.n_obs
    assert "n_flagged" in result
    assert result["n_flagged"] >= 0
    assert "flags" in result
    assert len(result["flags"]) == small_adata.n_obs
    assert "parasite_score" in result
    assert 0.0 <= result["parasite_score"] <= 100.0
    assert "report" in result
    assert "ambient_rna" in result["report"]
    assert "doublets" in result["report"]
    assert "batch_effects" in result["report"]


def test_detect_parasites_with_thresholds(small_adata):
    """Test detection with custom thresholds"""
    from src.bio.parasite_detector import detect_parasites
    
    # Add required metrics
    small_adata.obs['n_genes_by_counts'] = np.random.randint(100, 1000, size=small_adata.n_obs)
    small_adata.obs['total_counts'] = np.random.randint(1000, 10000, size=small_adata.n_obs)
    small_adata.obs['pct_counts_mt'] = np.random.rand(small_adata.n_obs) * 20
    small_adata.obs['batch'] = ['batch1'] * small_adata.n_obs
    small_adata.obs['leiden'] = ['0'] * small_adata.n_obs
    
    result = detect_parasites(
        adata=small_adata,
        cluster_key="leiden",
        batch_key="batch",
        ambient_threshold=0.1,
        doublet_threshold=0.3,
        batch_threshold=0.5
    )
    
    assert "thresholds" in result
    assert result["thresholds"]["ambient"] == 0.1
    assert result["thresholds"]["doublet"] == 0.3
    assert result["thresholds"]["batch"] == 0.5


def test_ambient_score_calculation(small_adata):
    """Test ambient RNA scoring"""
    from src.bio.parasite_detector import ambient_score
    
    small_adata.obs['n_genes_by_counts'] = create_repeated_labels([100, 200, 500, 800], small_adata.n_obs)
    small_adata.obs['total_counts'] = create_repeated_labels([1000, 2000, 5000, 8000], small_adata.n_obs)
    
    scores = ambient_score(small_adata)
    
    assert len(scores) == small_adata.n_obs
    assert all(0.0 <= s <= 1.0 for s in scores)
    # Low complexity cells should have higher ambient scores
    assert scores[0] > scores[3]  # 100 genes vs 800 genes


def test_doublet_enrichment(small_adata):
    """Test doublet enrichment scoring"""
    from src.bio.parasite_detector import doublet_enrichment
    
    # Add required metrics
    small_adata.obs['n_genes_by_counts'] = np.random.randint(100, 1000, size=small_adata.n_obs)
    small_adata.obs['total_counts'] = np.random.randint(1000, 10000, size=small_adata.n_obs)
    small_adata.obs['leiden'] = create_repeated_labels(['0', '1', '2'], small_adata.n_obs)
    
    scores = doublet_enrichment(small_adata, cluster_key="leiden")
    
    assert len(scores) == small_adata.n_obs
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_batch_purity(small_adata):
    """Test batch purity scoring"""
    from src.bio.parasite_detector import batch_purity
    
    # Create clean batches
    small_adata.obs['leiden'] = create_repeated_labels(['0', '0', '1', '1'], small_adata.n_obs)
    small_adata.obs['batch'] = create_repeated_labels(['batch1', 'batch1', 'batch2', 'batch2'], small_adata.n_obs)
    
    scores = batch_purity(small_adata, cluster_key="leiden", batch_key="batch")
    
    assert len(scores) == small_adata.n_obs
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_detect_parasites_missing_cluster_key(small_adata):
    """Test error handling for missing cluster key"""
    from src.bio.parasite_detector import detect_parasites
    
    with pytest.raises(ValueError, match="Cluster key.*not found"):
        detect_parasites(
            adata=small_adata,
            cluster_key="nonexistent",
            batch_key="batch"
        )


def test_detect_parasites_missing_batch_key(small_adata):
    """Test error handling for missing batch key"""
    from src.bio.parasite_detector import detect_parasites
    
    small_adata.obs['leiden'] = ['0'] * small_adata.n_obs
    
    with pytest.raises(ValueError, match="Batch key.*not found"):
        detect_parasites(
            adata=small_adata,
            cluster_key="leiden",
            batch_key="nonexistent"
        )


def test_detect_parasites_high_contamination(small_adata):
    """Test detection with artificially high contamination"""
    from src.bio.parasite_detector import detect_parasites
    
    # Create obviously bad cells
    small_adata.obs['n_genes_by_counts'] = [50] * small_adata.n_obs  # Very low complexity
    small_adata.obs['total_counts'] = [500] * small_adata.n_obs
    small_adata.obs['pct_counts_mt'] = [80.0] * small_adata.n_obs  # Very high MT
    small_adata.obs['batch'] = ['batch1'] * small_adata.n_obs
    small_adata.obs['leiden'] = ['0'] * small_adata.n_obs
    
    result = detect_parasites(
        adata=small_adata,
        cluster_key="leiden",
        batch_key="batch",
        ambient_threshold=0.2
    )
    
    # Should flag most/all cells
    assert result["n_flagged"] > small_adata.n_obs * 0.5
    assert result["parasite_score"] > 50.0
