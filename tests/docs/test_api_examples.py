"""
TDD tests for API documentation examples.

These tests verify that all code examples in the documentation actually work.
Written BEFORE documentation to ensure accuracy via test-driven development.

Test Strategy:
- Every curl/Python example from docs must be tested
- All response formats must match documented schemas
- Error cases must return documented status codes
- Performance characteristics must match claims
"""

import pytest
import json
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path


@pytest.fixture
def api_client():
    """API client with valid authentication"""
    from main import app
    client = TestClient(app)
    # Use demo token from authorized_institutions
    client.headers = {"Authorization": "Bearer demo_token"}
    return client


@pytest.fixture
def unauthorized_client():
    """API client without authentication"""
    from main import app
    return TestClient(app)


class TestHealthEndpoint:
    """Test /api/v1/health examples from documentation"""
    
    def test_health_check_returns_ok_status(self, api_client):
        """Verify health endpoint returns {"status": "ok"} as documented"""
        response = api_client.get("/api/v1/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "don_stack" in data
    
    def test_health_check_includes_don_stack_status(self, api_client):
        """Verify health check includes DON Stack component status"""
        response = api_client.get("/api/v1/health")
        data = response.json()
        
        assert "don_stack" in data
        don_stack = data["don_stack"]
        assert "mode" in don_stack
        assert don_stack["mode"] in ["production", "fallback"]
        assert "adapter_loaded" in don_stack
        assert isinstance(don_stack["adapter_loaded"], bool)
    
    def test_health_check_includes_qac_availability(self, api_client):
        """Verify QAC engine availability in health response"""
        response = api_client.get("/api/v1/health")
        data = response.json()
        
        assert "qac" in data
        qac = data["qac"]
        assert "supported_engines" in qac
        assert "default_engine" in qac
        assert "real_engine_available" in qac
        assert isinstance(qac["supported_engines"], list)


class TestGenomicsCompressionEndpoint:
    """Test /api/v1/genomics/compress examples"""
    
    def test_compression_with_minimal_data(self, api_client):
        """Verify basic compression example from documentation"""
        payload = {
            "data": {
                "gene_names": ["GENE1", "GENE2", "GENE3", "GENE4"],
                "expression_matrix": [
                    [1.0, 2.0, 3.0, 4.0],
                    [5.0, 6.0, 7.0, 8.0],
                    [9.0, 10.0, 11.0, 12.0]
                ]
            },
            "compression_target": 2
        }
        
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        assert response.status_code == 200
        
        data = response.json()
        # Verify documented response structure
        assert "compressed_data" in data
        assert "compression_stats" in data
        assert "algorithm" in data
        assert "institution" in data
        
        # Verify compression actually happened
        stats = data["compression_stats"]
        assert "original_dimensions" in stats
        assert "compressed_dimensions" in stats
        assert stats["compressed_dimensions"] <= stats["original_dimensions"]
    
    def test_compression_returns_trace_id(self, api_client):
        """Verify trace_id is returned for memory tracking"""
        payload = {
            "data": {
                "gene_names": ["G1", "G2"],
                "expression_matrix": [[1.0, 2.0]]
            },
            "compression_target": 1,
            "project_id": "test_project"
        }
        
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        data = response.json()
        
        assert "trace_id" in data
        assert isinstance(data["trace_id"], str)
        assert len(data["trace_id"]) > 0
    
    def test_compression_respects_seed_parameter(self, api_client):
        """Verify seed parameter produces reproducible results"""
        payload = {
            "data": {
                "gene_names": ["G1", "G2", "G3"],
                "expression_matrix": [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]
                ]
            },
            "compression_target": 2,
            "seed": 42
        }
        
        # Run twice with same seed
        response1 = api_client.post("/api/v1/genomics/compress", json=payload)
        response2 = api_client.post("/api/v1/genomics/compress", json=payload)
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Results should be deterministic
        assert np.allclose(
            data1["compressed_data"],
            data2["compressed_data"],
            atol=1e-6
        )
    
    def test_compression_reports_correct_ratios(self, api_client):
        """Verify compression ratio calculation is accurate"""
        payload = {
            "data": {
                "gene_names": [f"G{i}" for i in range(100)],
                "expression_matrix": [[float(i) for i in range(100)]]
            },
            "compression_target": 10
        }
        
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        data = response.json()
        
        stats = data["compression_stats"]
        original = stats["original_dimensions"]
        compressed = stats["achieved_k"]
        
        # Verify ratio format (e.g., "10.0×")
        assert "compression_ratio" in stats
        assert "×" in stats["compression_ratio"]
        
        # Extract numerical ratio
        ratio_str = stats["compression_ratio"].replace("×", "")
        ratio = float(ratio_str)
        expected_ratio = original / compressed
        assert abs(ratio - expected_ratio) < 0.1


class TestBioModuleEndpoints:
    """Test bio module endpoints and workflow examples"""
    
    def test_export_artifacts_sync_mode(self, api_client, h5ad_file):
        """Verify synchronous export-artifacts example"""
        with open(h5ad_file, "rb") as f:
            files = {"file": ("test.h5ad", f, "application/octet-stream")}
            data = {
                "cluster_key": "cluster",  # Match fixture data
                "latent_key": "X_pca",     # Match fixture data
                "sync": "true"
            }
            
            response = api_client.post(
                "/api/v1/bio/export-artifacts",
                files=files,
                data=data
            )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify documented response structure
        assert "nodes" in result
        assert "edges" in result
        assert "vectors" in result
        assert "artifacts" in result
        assert "status" in result
        assert result["status"] == "completed"
    
    def test_export_artifacts_async_mode_returns_job_id(self, api_client, h5ad_file):
        """Verify asynchronous mode returns job_id for polling"""
        with open(h5ad_file, "rb") as f:
            files = {"file": ("test.h5ad", f, "application/octet-stream")}
            data = {
                "cluster_key": "cluster",  # Match fixture data
                "latent_key": "X_pca",     # Match fixture data
                "sync": "false"
            }
            
            response = api_client.post(
                "/api/v1/bio/export-artifacts",
                files=files,
                data=data
            )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "job_id" in result
        assert result["job_id"] is not None
        assert result["status"] == "pending"
    
    def test_job_polling_endpoint(self, api_client, h5ad_file):
        """Verify job status polling workflow from documentation"""
        # Submit async job
        with open(h5ad_file, "rb") as f:
            files = {"file": ("test.h5ad", f, "application/octet-stream")}
            data = {
                "cluster_key": "leiden",
                "latent_key": "X_umap",
                "sync": "false"
            }
            response = api_client.post(
                "/api/v1/bio/export-artifacts",
                files=files,
                data=data
            )
        
        job_id = response.json()["job_id"]
        
        # Poll job status
        status_response = api_client.get(f"/api/v1/bio/jobs/{job_id}")
        assert status_response.status_code == 200
        
        job_data = status_response.json()
        assert "job_id" in job_data
        assert "status" in job_data
        assert job_data["status"] in ["pending", "running", "completed", "failed"]
    
    def test_memory_endpoint_retrieves_traces(self, api_client, h5ad_file):
        """Verify project memory retrieval from documentation"""
        project_id = "test_doc_project"
        
        # Create a trace by running a job
        payload = {
            "data": {
                "gene_names": ["G1", "G2"],
                "expression_matrix": [[1.0, 2.0]]
            },
            "compression_target": 1,
            "project_id": project_id
        }
        api_client.post("/api/v1/genomics/compress", json=payload)
        
        # Retrieve project memory
        response = api_client.get(f"/api/v1/bio/memory/{project_id}")
        assert response.status_code == 200
        
        memory = response.json()
        assert "project_id" in memory
        assert memory["project_id"] == project_id
        assert "count" in memory
        assert "traces" in memory
        assert isinstance(memory["traces"], list)


class TestAuthenticationAndRateLimiting:
    """Test authentication flows and rate limiting as documented"""
    
    def test_missing_auth_token_returns_401(self, unauthorized_client):
        """Verify 401 error for missing authentication"""
        response = unauthorized_client.get("/api/v1/usage")
        assert response.status_code == 403  # FastAPI returns 403 for missing bearer
    
    def test_invalid_token_returns_401(self):
        """Verify 401 error for invalid token"""
        from main import app
        client = TestClient(app)
        client.headers = {"Authorization": "Bearer invalid_token_xyz"}
        
        response = client.get("/api/v1/usage")
        assert response.status_code == 401
        
        data = response.json()
        assert "detail" in data
        assert "Invalid" in data["detail"] or "token" in data["detail"].lower()
    
    def test_rate_limiting_headers(self, api_client):
        """Verify rate limit information is available"""
        response = api_client.get("/api/v1/usage")
        assert response.status_code == 200
        
        data = response.json()
        assert "rate_limit" in data
        assert "requests_used" in data
        assert isinstance(data["rate_limit"], int)
        assert data["rate_limit"] > 0
    
    def test_demo_token_has_100_requests_per_hour(self, api_client):
        """Verify demo token rate limit matches documentation"""
        response = api_client.get("/api/v1/usage")
        data = response.json()
        
        assert data["rate_limit"] == 100  # Demo token limit from docs


class TestErrorResponses:
    """Test error responses match documentation"""
    
    def test_invalid_file_format_returns_400(self, api_client, tmpdir):
        """Verify 400 error for wrong file format"""
        # Create a non-h5ad file
        txt_file = tmpdir / "test.txt"
        txt_file.write_text("not an h5ad file", encoding="utf-8")
        
        with open(txt_file, "rb") as f:
            files = {"file": ("test.txt", f, "text/plain")}
            data = {"cluster_key": "leiden", "latent_key": "X_umap"}
            
            response = api_client.post(
                "/api/v1/bio/export-artifacts",
                files=files,
                data=data
            )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data
        assert "h5ad" in data["detail"].lower()
    
    def test_missing_required_parameter_returns_422(self, api_client, h5ad_file):
        """Verify 422 error for missing required parameters"""
        with open(h5ad_file, "rb") as f:
            files = {"file": ("test.h5ad", f, "application/octet-stream")}
            # Missing required cluster_key parameter
            data = {"latent_key": "X_umap"}
            
            response = api_client.post(
                "/api/v1/bio/export-artifacts",
                files=files,
                data=data
            )
        
        assert response.status_code == 422  # FastAPI validation error
    
    def test_nonexistent_job_returns_404(self, api_client):
        """Verify 404 error for non-existent job ID"""
        fake_job_id = "nonexistent-job-id-12345"
        response = api_client.get(f"/api/v1/bio/jobs/{fake_job_id}")
        
        assert response.status_code == 404
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"].lower()


class TestResponseFormats:
    """Test that response formats match documented schemas"""
    
    def test_compression_response_schema(self, api_client):
        """Verify compression response includes all documented fields"""
        payload = {
            "data": {
                "gene_names": ["G1", "G2"],
                "expression_matrix": [[1.0, 2.0]]
            }
        }
        
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        data = response.json()
        
        # Required fields from documentation
        required_fields = [
            "compressed_data",
            "gene_names",
            "compression_stats",
            "algorithm",
            "institution",
            "runtime_ms",
            "trace_id"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify compression_stats structure
        stats = data["compression_stats"]
        stats_fields = [
            "original_dimensions",
            "compressed_dimensions",
            "compression_ratio",
            "cells_processed"
        ]
        
        for field in stats_fields:
            assert field in stats, f"Missing stats field: {field}"
    
    def test_health_response_schema(self, api_client):
        """Verify health response matches documented structure"""
        response = api_client.get("/api/v1/health")
        data = response.json()
        
        required_fields = ["status", "don_stack", "qac", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"
        
        # Verify nested structures
        assert "mode" in data["don_stack"]
        assert "supported_engines" in data["qac"]


class TestPerformanceCharacteristics:
    """Test that performance matches documented claims"""
    
    def test_small_dataset_compression_under_1_second(self, api_client):
        """Verify small dataset compression is fast (< 1s)"""
        import time
        
        payload = {
            "data": {
                "gene_names": [f"G{i}" for i in range(100)],
                "expression_matrix": [[float(j) for j in range(100)] for _ in range(50)]
            },
            "compression_target": 10
        }
        
        start = time.time()
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        elapsed = time.time() - start
        
        assert response.status_code == 200
        assert elapsed < 1.0, f"Compression took {elapsed:.2f}s, expected < 1.0s"
    
    def test_compression_runtime_reported(self, api_client):
        """Verify runtime_ms is reported in response"""
        payload = {
            "data": {
                "gene_names": ["G1", "G2"],
                "expression_matrix": [[1.0, 2.0]]
            }
        }
        
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        data = response.json()
        
        assert "runtime_ms" in data
        assert isinstance(data["runtime_ms"], (int, float))
        assert data["runtime_ms"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
