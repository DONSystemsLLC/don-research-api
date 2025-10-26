"""
Tests for bio routes
TDD approach: Write tests first, then implement
"""

import pytest
from pathlib import Path
from io import BytesIO
import time
import json
from .test_helpers import create_repeated_labels


def test_export_artifacts_sync(api_client, small_adata, tmpdir):
    """Test synchronous export-artifacts endpoint"""
    from src.bio.routes import router
    
    # Save test data to H5AD
    h5ad_path = Path(tmpdir) / "test.h5ad"
    small_adata.obs['leiden'] = create_repeated_labels(['0', '1'], small_adata.n_obs)
    
    if 'X_pca' not in small_adata.obsm:
        import numpy as np
        small_adata.obsm['X_pca'] = np.random.randn(small_adata.n_obs, 10)
    
    small_adata.write_h5ad(h5ad_path)
    
    # Read file as bytes
    with open(h5ad_path, 'rb') as f:
        file_content = f.read()
    
    # Make API request
    response = api_client.post(
        "/api/v1/bio/export-artifacts",
        files={"file": ("test.h5ad", BytesIO(file_content), "application/octet-stream")},
        data={
            "cluster_key": "leiden",
            "latent_key": "X_pca",
            "sync": "true"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["nodes"] >= 2
    assert data["vectors"] == small_adata.n_obs
    assert len(data["artifacts"]) == 2


def test_export_artifacts_async(api_client, small_adata, tmpdir):
    """Test asynchronous export-artifacts endpoint"""
    h5ad_path = Path(tmpdir) / "test.h5ad"
    small_adata.obs['leiden'] = ['0'] * small_adata.n_obs
    
    if 'X_pca' not in small_adata.obsm:
        import numpy as np
        small_adata.obsm['X_pca'] = np.random.randn(small_adata.n_obs, 10)
    
    small_adata.write_h5ad(h5ad_path)
    
    with open(h5ad_path, 'rb') as f:
        file_content = f.read()
    
    response = api_client.post(
        "/api/v1/bio/export-artifacts",
        files={"file": ("test.h5ad", BytesIO(file_content), "application/octet-stream")},
        data={
            "cluster_key": "leiden",
            "latent_key": "X_pca",
            "sync": "false"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "pending"
    assert data["job_id"] is not None
    
    # Check job status
    job_id = data["job_id"]
    # Poll for completion (background task should process quickly in tests)
    for _ in range(20):
        job_response = api_client.get(f"/api/v1/bio/jobs/{job_id}")
        assert job_response.status_code == 200
        job_data = job_response.json()
        if job_data["status"] == "completed":
            break
        time.sleep(0.1)
    else:
        pytest.fail("async export job did not complete in time")

    assert job_data["endpoint"] == "export-artifacts"
    assert job_data["status"] == "completed"
    assert job_data["result"] is not None
    result = job_data["result"]
    assert result["nodes"] >= 1
    assert result["vectors"] == small_adata.n_obs
    assert len(result["artifacts"]) == 2
    for artifact_path in result["artifacts"]:
        assert Path(artifact_path).exists()


def test_signal_sync_sync(api_client, tmpdir):
    """Test synchronous signal-sync endpoint"""
    # Create two test collapse maps
    map1 = {
        "nodes": [
            {"id": "0", "size": 50, "label": "cluster_0"},
            {"id": "1", "size": 30, "label": "cluster_1"}
        ],
        "edges": [
            {"source": "0", "target": "1", "weight": 0.8}
        ],
        "metadata": {"n_clusters": 2}
    }
    
    map2 = {
        "nodes": [
            {"id": "0", "size": 55, "label": "cluster_0"},
            {"id": "1", "size": 35, "label": "cluster_1"}
        ],
        "edges": [
            {"source": "0", "target": "1", "weight": 0.75}
        ],
        "metadata": {"n_clusters": 2}
    }
    
    # Convert to bytes
    map1_bytes = json.dumps(map1).encode('utf-8')
    map2_bytes = json.dumps(map2).encode('utf-8')
    
    response = api_client.post(
        "/api/v1/bio/signal-sync",
        files={
            "artifact1": ("map1.json", BytesIO(map1_bytes), "application/json"),
            "artifact2": ("map2.json", BytesIO(map2_bytes), "application/json")
        },
        data={
            "coherence_threshold": "0.8",
            "sync": "true"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert 0.0 <= data["coherence_score"] <= 1.0
    assert 0.0 <= data["node_overlap"] <= 1.0
    assert data["node_overlap"] == 1.0  # Both maps have same nodes
    assert "report" in data


def test_parasite_detect_sync(api_client, small_adata, tmpdir):
    """Test synchronous parasite-detect endpoint"""
    h5ad_path = Path(tmpdir) / "test.h5ad"
    
    # Add required metadata
    import numpy as np
    small_adata.obs['leiden'] = create_repeated_labels(['0', '1'], small_adata.n_obs)
    small_adata.obs['batch'] = create_repeated_labels(['batch1', 'batch2'], small_adata.n_obs)
    
    # Ensure QC metrics exist
    if 'n_genes_by_counts' not in small_adata.obs:
        small_adata.obs['n_genes_by_counts'] = np.random.randint(100, 1000, size=small_adata.n_obs)
    if 'total_counts' not in small_adata.obs:
        small_adata.obs['total_counts'] = np.random.randint(1000, 10000, size=small_adata.n_obs)
    if 'pct_counts_mt' not in small_adata.obs:
        small_adata.obs['pct_counts_mt'] = np.random.rand(small_adata.n_obs) * 20
    
    small_adata.write_h5ad(h5ad_path)
    
    with open(h5ad_path, 'rb') as f:
        file_content = f.read()
    
    response = api_client.post(
        "/api/v1/bio/qc/parasite-detect",
        files={"file": ("test.h5ad", BytesIO(file_content), "application/octet-stream")},
        data={
            "cluster_key": "leiden",
            "batch_key": "batch",
            "sync": "true"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["n_cells"] == small_adata.n_obs
    assert data["n_flagged"] >= 0
    assert len(data["flags"]) == small_adata.n_obs
    assert 0.0 <= data["parasite_score"] <= 100.0
    assert "report" in data


def test_evolution_report_sync(api_client, small_adata, tmpdir):
    """Test synchronous evolution-report endpoint"""
    run1_path = Path(tmpdir) / "run1.h5ad"
    run2_path = Path(tmpdir) / "run2.h5ad"
    
    # Prepare run1
    import numpy as np
    small_adata.obs['leiden'] = ['0'] * small_adata.n_obs
    small_adata.obsm['X_pca'] = np.random.randn(small_adata.n_obs, 10)
    small_adata.write_h5ad(run1_path)
    
    # Prepare run2 (slight variation)
    small_adata.obsm['X_pca'] += np.random.randn(small_adata.n_obs, 10) * 0.1
    small_adata.write_h5ad(run2_path)
    
    with open(run1_path, 'rb') as f1, open(run2_path, 'rb') as f2:
        file1_content = f1.read()
        file2_content = f2.read()
    
    response = api_client.post(
        "/api/v1/bio/evolution/report",
        files={
            "run1_file": ("run1.h5ad", BytesIO(file1_content), "application/octet-stream"),
            "run2_file": ("run2.h5ad", BytesIO(file2_content), "application/octet-stream")
        },
        data={
            "run2_name": "run2_test",
            "cluster_key": "leiden",
            "latent_key": "X_pca",
            "sync": "true"
        }
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "completed"
    assert data["n_cells_run1"] == small_adata.n_obs
    assert data["n_cells_run2"] == small_adata.n_obs
    assert 0.0 <= data["stability_score"] <= 100.0
    assert "delta_metrics" in data


def test_export_artifacts_invalid_file(api_client):
    """Test export with invalid file format"""
    response = api_client.post(
        "/api/v1/bio/export-artifacts",
        files={"file": ("test.txt", BytesIO(b"invalid"), "text/plain")},
        data={
            "cluster_key": "leiden",
            "latent_key": "X_pca",
            "sync": "true"
        }
    )
    
    assert response.status_code == 400
    assert "must be .h5ad" in response.json()["detail"]


def test_job_not_found(api_client):
    """Test job status for non-existent job"""
    response = api_client.get("/api/v1/bio/jobs/nonexistent-job-id")
    
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_export_artifacts_missing_cluster_key(api_client, small_adata, tmpdir):
    """Test export with missing cluster key"""
    h5ad_path = Path(tmpdir) / "test.h5ad"
    
    import numpy as np
    small_adata.obsm['X_pca'] = np.random.randn(small_adata.n_obs, 10)
    small_adata.write_h5ad(h5ad_path)
    
    with open(h5ad_path, 'rb') as f:
        file_content = f.read()
    
    response = api_client.post(
        "/api/v1/bio/export-artifacts",
        files={"file": ("test.h5ad", BytesIO(file_content), "application/octet-stream")},
        data={
            "cluster_key": "nonexistent",
            "latent_key": "X_pca",
            "sync": "true"
        }
    )
    
    assert response.status_code == 500
    assert "not found" in response.json()["detail"].lower()
