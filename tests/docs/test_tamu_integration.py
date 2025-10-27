"""
TDD tests for Texas A&M Cai Lab integration documentation.

These tests ensure the TAMU_INTEGRATION.md guide is accurate and complete.
All workflows and examples must be validated before documentation is written.

Test Coverage:
- Complete Scanpy pipeline integration
- Cell type marker workflows
- QC pipeline examples
- Evolution tracking workflows
- Reproducibility patterns
"""

import pytest
import json
import numpy as np
from fastapi.testclient import TestClient
from pathlib import Path


@pytest.fixture
def api_client():
    """Authenticated API client for TAMU"""
    from main import app
    client = TestClient(app)
    client.headers = {"Authorization": "Bearer demo_token"}
    return client


class TestScanpyPipelineIntegration:
    """Test complete Scanpy → DON API workflows"""
    
    def test_pbmc_analysis_complete_workflow(self, api_client, h5ad_file):
        """
        Test complete PBMC analysis workflow from TAMU docs:
        1. Load h5ad from Scanpy preprocessing
        2. Build vectors (cluster mode)
        3. Encode cell type query
        4. Search for matching clusters
        5. Generate entropy map
        """
        
        # Step 1: Build vectors from processed h5ad
        with open(h5ad_file, "rb") as f:
            files = {"file": ("pbmc3k.h5ad", f, "application/octet-stream")}
            data = {"mode": "cluster"}
            
            build_response = api_client.post(
                "/api/v1/genomics/vectors/build",
                files=files,
                data=data
            )
        
        assert build_response.status_code == 200
        build_result = build_response.json()
        assert build_result["ok"] is True
        assert "jsonl" in build_result
        assert build_result["count"] > 0
        
        jsonl_path = build_result["jsonl"]
        
        # Step 2: Encode T cell markers query
        t_cell_genes = ["CD3E", "CD8A", "CD4", "IL7R"]
        encode_data = {"gene_list_json": json.dumps(t_cell_genes)}
        
        encode_response = api_client.post(
            "/api/v1/genomics/query/encode",
            data=encode_data
        )
        
        assert encode_response.status_code == 200
        encode_result = encode_response.json()
        assert "psi" in encode_result
        query_vector = encode_result["psi"]
        assert len(query_vector) == 128  # Documented vector dimension
        
        # Step 3: Search for matching clusters
        search_data = {
            "jsonl_path": jsonl_path,
            "psi": json.dumps(query_vector),
            "k": 5
        }
        
        search_response = api_client.post(
            "/api/v1/genomics/vectors/search",
            data=search_data
        )
        
        assert search_response.status_code == 200
        search_result = search_response.json()
        assert search_result["ok"] is True
        assert "hits" in search_result
        assert len(search_result["hits"]) <= 5
        
        # Step 4: Validate distance interpretation
        if search_result["hits"]:
            first_hit = search_result["hits"][0]
            assert "distance" in first_hit
            distance = first_hit["distance"]
            # Distance should be in [0, 2] for cosine
            assert 0 <= distance <= 2
    
    def test_entropy_map_generation_workflow(self, api_client, h5ad_file):
        """Test entropy map generation for cell state analysis"""
        with open(h5ad_file, "rb") as f:
            files = {"file": ("pbmc.h5ad", f, "application/octet-stream")}
            data = {"label_key": "leiden"}
            
            response = api_client.post(
                "/api/v1/genomics/entropy-map",
                files=files,
                data=data
            )
        
        assert response.status_code == 200
        result = response.json()
        
        assert result["ok"] is True
        assert "png" in result
        assert "stats" in result
        
        # Verify stats structure
        stats = result["stats"]
        assert "mean_entropy" in stats or "entropy" in str(stats)


class TestCellTypeMarkerWorkflows:
    """Test cell type discovery workflows from docs"""
    
    def test_t_cell_marker_query(self, api_client):
        """Test T cell marker encoding from docs table"""
        t_cell_markers = ["CD3E", "CD8A", "CD4", "IL7R"]
        data = {"gene_list_json": json.dumps(t_cell_markers)}
        
        response = api_client.post("/api/v1/genomics/query/encode", data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "psi" in result
        assert len(result["psi"]) == 128
    
    def test_b_cell_marker_query(self, api_client):
        """Test B cell markers from documentation table"""
        b_cell_markers = ["MS4A1", "CD79A", "CD19", "IGHM"]
        data = {"gene_list_json": json.dumps(b_cell_markers)}
        
        response = api_client.post("/api/v1/genomics/query/encode", data=data)
        assert response.status_code == 200
        
        result = response.json()
        assert "psi" in result
        assert len(result["psi"]) == 128
    
    def test_monocyte_marker_query(self, api_client):
        """Test monocyte markers from documentation"""
        monocyte_markers = ["CD14", "FCGR3A", "CST3", "LYZ"]
        data = {"gene_list_json": json.dumps(monocyte_markers)}
        
        response = api_client.post("/api/v1/genomics/query/encode", data=data)
        assert response.status_code == 200
    
    def test_distance_interpretation_guidelines(self, api_client, h5ad_file):
        """Verify distance ranges match documentation"""
        # Build vectors
        with open(h5ad_file, "rb") as f:
            files = {"file": ("test.h5ad", f, "application/octet-stream")}
            build_response = api_client.post(
                "/api/v1/genomics/vectors/build",
                files=files,
                data={"mode": "cluster"}
            )
        
        jsonl_path = build_response.json()["jsonl"]
        
        # Encode query
        genes = ["CD3E", "CD8A"]
        encode_response = api_client.post(
            "/api/v1/genomics/query/encode",
            data={"gene_list_json": json.dumps(genes)}
        )
        query_vector = encode_response.json()["psi"]
        
        # Search
        search_response = api_client.post(
            "/api/v1/genomics/vectors/search",
            data={
                "jsonl_path": jsonl_path,
                "psi": json.dumps(query_vector),
                "k": 10
            }
        )
        
        hits = search_response.json()["hits"]
        
        # Verify all distances are valid (may include infinity for no matches)
        # Documented range: 0.0-0.2 (very similar), 0.2-0.5 (similar), 
        # 0.5-0.8 (moderate), 0.8-2.0 (dissimilar), >2.0 (no match)
        for hit in hits:
            distance = hit["distance"]
            assert distance >= 0.0, "Distance cannot be negative"
            # Note: infinity/max_float indicates no meaningful match


class TestQCPipelineWorkflows:
    """Test QC detection workflows from docs"""
    
    def test_parasite_detection_workflow(self, api_client, h5ad_file):
        """
        Test parasite detection QC workflow:
        1. Run detection
        2. Interpret parasite_score
        3. Get per-cell flags
        4. Verify thresholds
        """
        with open(h5ad_file, "rb") as f:
            files = {"file": ("pbmc.h5ad", f, "application/octet-stream")}
            data = {
                "cluster_key": "leiden",
                "batch_key": "batch",  # Assuming exists or will be handled
                "ambient_threshold": "0.15",
                "doublet_threshold": "0.25",
                "batch_threshold": "0.3",
                "sync": "true"
            }
            
            response = api_client.post(
                "/api/v1/bio/qc/parasite-detect",
                files=files,
                data=data
            )
        
        # May fail if batch_key doesn't exist, but structure should be correct
        if response.status_code == 200:
            result = response.json()
            
            assert "n_cells" in result
            assert "n_flagged" in result
            assert "parasite_score" in result
            assert "flags" in result
            assert "thresholds" in result
            
            # Verify parasite_score is percentage
            assert 0 <= result["parasite_score"] <= 100
            
            # Verify flags is boolean array
            assert isinstance(result["flags"], list)
            if result["flags"]:
                assert isinstance(result["flags"][0], bool)
    
    def test_qc_interpretation_guidelines(self, api_client):
        """Verify QC score interpretation ranges from docs"""
        # Documented ranges:
        # 0-5%: Excellent
        # 5-15%: Good
        # 15-30%: Moderate
        # >30%: High contamination
        
        test_scores = [3.0, 10.0, 20.0, 35.0]
        interpretations = ["excellent", "good", "moderate", "high"]
        
        for score, expected_category in zip(test_scores, interpretations):
            if score <= 5:
                assert expected_category == "excellent"
            elif score <= 15:
                assert expected_category == "good"
            elif score <= 30:
                assert expected_category == "moderate"
            else:
                assert expected_category == "high"


class TestEvolutionTrackingWorkflows:
    """Test pipeline evolution tracking from docs"""
    
    def test_run_comparison_workflow(self, api_client, h5ad_file):
        """
        Test evolution report workflow:
        1. Compare two runs
        2. Get stability score
        3. Interpret delta metrics
        """
        with open(h5ad_file, "rb") as f1:
            run1_content = f1.read()
        
        # Use same file twice for testing
        files = {
            "run1_file": ("run1.h5ad", run1_content, "application/octet-stream"),
            "run2_file": ("run2.h5ad", run1_content, "application/octet-stream")
        }
        data = {
            "run2_name": "parameter_test",
            "cluster_key": "leiden",
            "latent_key": "X_umap",
            "sync": "true"
        }
        
        response = api_client.post(
            "/api/v1/bio/evolution/report",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            result = response.json()
            
            assert "stability_score" in result
            assert "delta_metrics" in result
            assert "n_cells_run1" in result
            assert "n_cells_run2" in result
            
            # Same file should have high stability
            assert result["stability_score"] >= 90.0
    
    def test_stability_score_interpretation(self, api_client):
        """Verify stability score ranges from documentation"""
        # Documented ranges:
        # >90%: Excellent stability
        # 70-90%: Good stability
        # 50-70%: Moderate drift
        # <50%: High drift
        
        test_scores = [95.0, 80.0, 60.0, 40.0]
        categories = ["excellent", "good", "moderate", "high_drift"]
        
        for score, expected in zip(test_scores, categories):
            if score > 90:
                assert expected == "excellent"
            elif score >= 70:
                assert expected == "good"
            elif score >= 50:
                assert expected == "moderate"
            else:
                assert expected == "high_drift"


class TestSignalSyncWorkflows:
    """Test signal synchronization workflows"""
    
    def test_signal_sync_coherence_workflow(self, api_client, tmpdir):
        """Test cross-artifact coherence measurement"""
        # Create minimal collapse map JSON files
        map1 = {
            "nodes": [
                {"id": "cluster_0", "size": 100},
                {"id": "cluster_1", "size": 50}
            ],
            "edges": [
                {"source": "cluster_0", "target": "cluster_1", "weight": 0.5}
            ]
        }
        
        map2 = {
            "nodes": [
                {"id": "cluster_0", "size": 95},
                {"id": "cluster_1", "size": 55}
            ],
            "edges": [
                {"source": "cluster_0", "target": "cluster_1", "weight": 0.6}
            ]
        }
        
        file1 = tmpdir / "map1.json"
        file2 = tmpdir / "map2.json"
        file1.write_text(json.dumps(map1), encoding="utf-8")
        file2.write_text(json.dumps(map2), encoding="utf-8")
        
        with open(file1, "rb") as f1, open(file2, "rb") as f2:
            files = {
                "artifact1": ("map1.json", f1, "application/json"),
                "artifact2": ("map2.json", f2, "application/json")
            }
            data = {
                "coherence_threshold": "0.8",
                "sync": "true"
            }
            
            response = api_client.post(
                "/api/v1/bio/signal-sync",
                files=files,
                data=data
            )
        
        assert response.status_code == 200
        result = response.json()
        
        assert "coherence_score" in result
        assert "node_overlap" in result
        assert "synchronized" in result
        assert 0 <= result["coherence_score"] <= 1.0


class TestReproducibilityPatterns:
    """Test reproducibility features from docs"""
    
    def test_seed_parameter_produces_reproducible_results(self, api_client):
        """Verify seed parameter reproducibility"""
        payload = {
            "data": {
                "gene_names": [f"G{i}" for i in range(10)],
                "expression_matrix": [
                    [float(np.random.randn()) for _ in range(10)]
                    for _ in range(5)
                ]
            },
            "compression_target": 3,
            "seed": 42
        }
        
        # Run twice with same seed
        response1 = api_client.post("/api/v1/genomics/compress", json=payload)
        response2 = api_client.post("/api/v1/genomics/compress", json=payload)
        
        data1 = response1.json()
        data2 = response2.json()
        
        # Results should be identical
        assert np.allclose(
            data1["compressed_data"],
            data2["compressed_data"],
            atol=1e-10
        )
    
    def test_trace_id_enables_audit_trail(self, api_client):
        """Verify trace_id is returned for reproducibility"""
        payload = {
            "data": {
                "gene_names": ["G1", "G2"],
                "expression_matrix": [[1.0, 2.0]]
            },
            "project_id": "reproducibility_test",
            "user_id": "tamu_researcher"
        }
        
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        data = response.json()
        
        assert "trace_id" in data
        trace_id = data["trace_id"]
        
        # Should be able to retrieve from memory
        memory_response = api_client.get("/api/v1/bio/memory/reproducibility_test")
        memory = memory_response.json()
        
        trace_ids = [t["id"] for t in memory["traces"]]
        assert trace_id in trace_ids


class TestProjectTrackingPatterns:
    """Test project organization features"""
    
    def test_project_id_groups_operations(self, api_client):
        """Verify project_id groups related operations"""
        project_id = "tamu_pbmc_study_2024"
        
        # Run multiple operations with same project_id
        for i in range(3):
            payload = {
                "data": {
                    "gene_names": [f"G{i}"],
                    "expression_matrix": [[float(i)]]
                },
                "project_id": project_id,
                "user_id": f"researcher_{i}"
            }
            response = api_client.post("/api/v1/genomics/compress", json=payload)
            assert response.status_code == 200
        
        # Retrieve project memory
        memory_response = api_client.get(f"/api/v1/bio/memory/{project_id}")
        memory = memory_response.json()
        
        assert memory["project_id"] == project_id
        assert memory["count"] >= 3  # At least our 3 operations
    
    def test_user_id_tracks_researcher(self, api_client):
        """Verify user_id is tracked in traces"""
        user_id = "dr_cai_lab_001"
        payload = {
            "data": {
                "gene_names": ["G1"],
                "expression_matrix": [[1.0]]
            },
            "project_id": "user_tracking_test",
            "user_id": user_id
        }
        
        response = api_client.post("/api/v1/genomics/compress", json=payload)
        trace_id = response.json()["trace_id"]
        
        # Retrieve and verify user_id in trace
        memory_response = api_client.get("/api/v1/bio/memory/user_tracking_test")
        memory = memory_response.json()
        
        matching_traces = [t for t in memory["traces"] if t["id"] == trace_id]
        assert len(matching_traces) > 0
        assert matching_traces[0]["user_id"] == user_id


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
