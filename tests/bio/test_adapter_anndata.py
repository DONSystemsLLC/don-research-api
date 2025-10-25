"""
Tests for AnnData adapter
TDD approach: Write tests first, then implement
"""

import json
import pytest
import numpy as np
from pathlib import Path


def test_export_artifacts_basic(small_adata, tmpdir):
    """Test basic H5AD to collapse map export"""
    from src.bio.adapter_anndata import export_artifacts
    
    # Ensure required keys exist in test data
    clusters = ['0', '1'] * (small_adata.n_obs // 2)
    clusters += ['0'] * (small_adata.n_obs - len(clusters))
    small_adata.obs['leiden'] = clusters[:small_adata.n_obs]
    
    # Add PCA if not present
    if 'X_pca' not in small_adata.obsm:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, small_adata.n_obs, small_adata.n_vars))
        small_adata.obsm['X_pca'] = pca.fit_transform(small_adata.X.toarray() if hasattr(small_adata.X, 'toarray') else small_adata.X)
    
    # Save to temp file
    h5ad_path = Path(tmpdir) / "test.h5ad"
    small_adata.write_h5ad(h5ad_path)
    
    # Export artifacts
    result = export_artifacts(
        h5ad_path=str(h5ad_path),
        cluster_key="leiden",
        latent_key="X_pca",
        output_dir=str(tmpdir / "artifacts")
    )
    
    # Assertions
    assert result["nodes"] >= 2  # At least 2 clusters
    assert result["vectors"] == small_adata.n_obs
    assert len(result["artifacts"]) == 2
    
    # Verify collapse map
    map_path = Path(result["artifacts"][0])
    assert map_path.exists()
    with open(map_path) as f:
        collapse_map = json.load(f)
    assert "nodes" in collapse_map
    assert "edges" in collapse_map
    assert "metadata" in collapse_map
    assert collapse_map["metadata"]["n_clusters"] == result["nodes"]
    
    # Verify vectors
    vectors_path = Path(result["artifacts"][1])
    assert vectors_path.exists()
    from src.bio.adapter_anndata import stream_jsonl
    vectors = stream_jsonl(vectors_path)
    assert len(vectors) == small_adata.n_obs
    assert "cell_id" in vectors[0]
    assert "cluster" in vectors[0]
    assert "latent" in vectors[0]
    assert "qc" in vectors[0]


def test_export_artifacts_with_paga(small_adata_with_paga, tmpdir):
    """Test export with PAGA connectivity"""
    from src.bio.adapter_anndata import export_artifacts
    
    h5ad_path = Path(tmpdir) / "test_paga.h5ad"
    small_adata_with_paga.write_h5ad(h5ad_path)
    
    result = export_artifacts(
        h5ad_path=str(h5ad_path),
        cluster_key="leiden",
        latent_key="X_pca",
        paga_key="paga",
        output_dir=str(tmpdir / "artifacts")
    )
    
    # Should have edges from PAGA
    assert result["edges"] > 0
    
    map_path = Path(result["artifacts"][0])
    with open(map_path) as f:
        collapse_map = json.load(f)
    assert len(collapse_map["edges"]) == result["edges"]
    assert "source" in collapse_map["edges"][0]
    assert "target" in collapse_map["edges"][0]
    assert "weight" in collapse_map["edges"][0]


def test_export_artifacts_subsample(small_adata, tmpdir):
    """Test cell subsampling"""
    from src.bio.adapter_anndata import export_artifacts
    
    small_adata.obs['leiden'] = ['0'] * small_adata.n_obs
    if 'X_pca' not in small_adata.obsm:
        small_adata.obsm['X_pca'] = np.random.randn(small_adata.n_obs, 10)
    
    h5ad_path = Path(tmpdir) / "test.h5ad"
    small_adata.write_h5ad(h5ad_path)
    
    sample_size = small_adata.n_obs // 2
    result = export_artifacts(
        h5ad_path=str(h5ad_path),
        cluster_key="leiden",
        latent_key="X_pca",
        sample_cells=sample_size,
        output_dir=str(tmpdir / "artifacts")
    )
    
    assert result["vectors"] == sample_size


def test_export_artifacts_missing_cluster_key(small_adata, tmpdir):
    """Test error handling for missing cluster key"""
    from src.bio.adapter_anndata import export_artifacts
    
    h5ad_path = Path(tmpdir) / "test.h5ad"
    small_adata.write_h5ad(h5ad_path)
    
    with pytest.raises(ValueError, match="Cluster key.*not found"):
        export_artifacts(
            h5ad_path=str(h5ad_path),
            cluster_key="nonexistent",
            latent_key="X_pca",
            output_dir=str(tmpdir / "artifacts")
        )


def test_export_artifacts_missing_latent_key(small_adata, tmpdir):
    """Test error handling for missing latent key"""
    from src.bio.adapter_anndata import export_artifacts
    
    small_adata.obs['leiden'] = ['0'] * small_adata.n_obs
    h5ad_path = Path(tmpdir) / "test.h5ad"
    small_adata.write_h5ad(h5ad_path)
    
    with pytest.raises(ValueError, match="Latent key.*not found"):
        export_artifacts(
            h5ad_path=str(h5ad_path),
            cluster_key="leiden",
            latent_key="nonexistent",
            output_dir=str(tmpdir / "artifacts")
        )


def test_stream_jsonl(tmpdir):
    """Test JSONL streaming utility"""
    from src.bio.adapter_anndata import stream_jsonl
    
    # Create test JSONL
    test_data = [
        {"id": 1, "value": "a"},
        {"id": 2, "value": "b"},
        {"id": 3, "value": "c"}
    ]
    
    jsonl_path = Path(tmpdir) / "test.jsonl"
    with open(jsonl_path, 'w') as f:
        for record in test_data:
            f.write(json.dumps(record) + '\n')
    
    # Stream and verify
    records = stream_jsonl(jsonl_path)
    assert len(records) == 3
    assert records[0]["id"] == 1
    assert records[2]["value"] == "c"
