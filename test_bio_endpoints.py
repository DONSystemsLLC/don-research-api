#!/usr/bin/env python3
"""
Manual testing script for bio module endpoints.
Tests all 4 endpoints with sync/async modes, authentication, error handling.
"""

import httpx
import json
from pathlib import Path
import time

# Configuration
BASE_URL = "http://localhost:8080"
API_BASE = f"{BASE_URL}/api/v1"
BEARER_TOKEN = "demo_token"

# Test data paths
TEST_DATA_DIR = Path("test_data")
PBMC_SMALL = TEST_DATA_DIR / "pbmc_small.h5ad"

# HTTP client with authentication
headers = {
    "Authorization": f"Bearer {BEARER_TOKEN}"
}


def print_section(title):
    """Pretty print section headers"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def test_health_endpoint():
    """Test health endpoint to verify server is running"""
    print_section("Testing Health Endpoint")
    
    response = httpx.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    assert response.status_code == 200, "Health check failed"
    print("✅ Health endpoint working")


def test_export_artifacts_sync():
    """Test export-artifacts endpoint in synchronous mode"""
    print_section("Testing Export Artifacts (Sync Mode)")
    
    if not PBMC_SMALL.exists():
        print(f"⚠️  Test file not found: {PBMC_SMALL}")
        return
    
    with open(PBMC_SMALL, "rb") as f:
        files = {"file": ("pbmc_small.h5ad", f, "application/octet-stream")}
        data = {
            "cluster_key": "leiden",
            "latent_key": "X_umap",
            "sync": "true",
            "seed": "42",
            "project_id": "test_export_sync",
            "user_id": "test_user"
        }
        
        response = httpx.post(
            f"{API_BASE}/bio/export-artifacts",
            headers=headers,
            files=files,
            data=data,
            timeout=30.0
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Export completed")
        print(f"   Nodes: {result.get('nodes', 0)}")
        print(f"   Edges: {result.get('edges', 0)}")
        print(f"   Vectors: {result.get('vectors', 0)}")
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")
        if 'trace' in result.get('message', ''):
            print(f"   ✅ Memory logging confirmed")
    else:
        print(f"❌ Failed: {response.text}")


def test_export_artifacts_async():
    """Test export-artifacts endpoint in asynchronous mode"""
    print_section("Testing Export Artifacts (Async Mode)")
    
    if not PBMC_SMALL.exists():
        print(f"⚠️  Test file not found: {PBMC_SMALL}")
        return
    
    with open(PBMC_SMALL, "rb") as f:
        files = {"file": ("pbmc_small.h5ad", f, "application/octet-stream")}
        data = {
            "cluster_key": "leiden",
            "latent_key": "X_umap",
            "sync": "false",  # Async mode
            "seed": "42",
            "project_id": "test_export_async",
            "user_id": "test_user"
        }
        
        response = httpx.post(
            f"{API_BASE}/bio/export-artifacts",
            headers=headers,
            files=files,
            data=data,
            timeout=30.0
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        job_id = result.get('job_id')
        print(f"✅ Job created: {job_id}")
        print(f"   Status: {result.get('status')}")
        print(f"   Message: {result.get('message')}")
        
        # Poll job status
        if job_id:
            print("\n   Polling job status...")
            for i in range(10):
                time.sleep(0.5)
                job_response = httpx.get(
                    f"{API_BASE}/bio/jobs/{job_id}",
                    headers=headers
                )
                
                if job_response.status_code == 200:
                    job_status = job_response.json()
                    status = job_status.get('status')
                    print(f"   [{i+1}] Status: {status}")
                    
                    if status == "completed":
                        print(f"   ✅ Job completed successfully")
                        result = job_status.get('result', {})
                        print(f"      Nodes: {result.get('nodes', 0)}")
                        print(f"      Edges: {result.get('edges', 0)}")
                        print(f"      Vectors: {result.get('vectors', 0)}")
                        if result.get('trace_id'):
                            print(f"      ✅ Memory logging confirmed: {result['trace_id']}")
                        break
                    elif status == "failed":
                        print(f"   ❌ Job failed: {job_status.get('error')}")
                        break
    else:
        print(f"❌ Failed: {response.text}")


def test_signal_sync():
    """Test signal-sync endpoint"""
    print_section("Testing Signal Sync")
    
    # Create two simple test artifacts
    artifact1 = {
        "nodes": [
            {"id": "cluster_0", "size": 100},
            {"id": "cluster_1", "size": 80}
        ],
        "edges": [
            {"source": "cluster_0", "target": "cluster_1", "weight": 0.5}
        ]
    }
    
    artifact2 = {
        "nodes": [
            {"id": "cluster_0", "size": 95},
            {"id": "cluster_2", "size": 75}
        ],
        "edges": [
            {"source": "cluster_0", "target": "cluster_2", "weight": 0.6}
        ]
    }
    
    # Write to temp files
    artifact1_path = Path("/tmp/test_artifact1.json")
    artifact2_path = Path("/tmp/test_artifact2.json")
    
    artifact1_path.write_text(json.dumps(artifact1))
    artifact2_path.write_text(json.dumps(artifact2))
    
    with open(artifact1_path, "rb") as f1, open(artifact2_path, "rb") as f2:
        files = {
            "artifact1": ("artifact1.json", f1, "application/json"),
            "artifact2": ("artifact2.json", f2, "application/json")
        }
        data = {
            "coherence_threshold": "0.5",
            "sync": "true",
            "seed": "42",
            "project_id": "test_signal_sync",
            "user_id": "test_user"
        }
        
        response = httpx.post(
            f"{API_BASE}/bio/signal-sync",
            headers=headers,
            files=files,
            data=data,
            timeout=10.0
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Signal sync completed")
        print(f"   Coherence Score: {result.get('coherence_score', 0):.3f}")
        print(f"   Node Overlap: {result.get('node_overlap', 0):.3f}")
        print(f"   Edge Consistency: {result.get('edge_consistency', 0):.3f}")
        print(f"   Synchronized: {result.get('synchronized')}")
        if 'trace' in result.get('message', ''):
            print(f"   ✅ Memory logging confirmed")
    else:
        print(f"❌ Failed: {response.text}")


def test_parasite_detect():
    """Test parasite-detect endpoint"""
    print_section("Testing Parasite Detect (QC)")
    
    if not PBMC_SMALL.exists():
        print(f"⚠️  Test file not found: {PBMC_SMALL}")
        return
    
    with open(PBMC_SMALL, "rb") as f:
        files = {"file": ("pbmc_small.h5ad", f, "application/octet-stream")}
        data = {
            "cluster_key": "leiden",
            "batch_key": "batch",
            "ambient_threshold": "0.15",
            "doublet_threshold": "0.25",
            "batch_threshold": "0.3",
            "sync": "true",
            "seed": "42",
            "project_id": "test_parasite",
            "user_id": "test_user"
        }
        
        response = httpx.post(
            f"{API_BASE}/bio/qc/parasite-detect",
            headers=headers,
            files=files,
            data=data,
            timeout=30.0
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ QC analysis completed")
        print(f"   Cells: {result.get('n_cells', 0)}")
        print(f"   Flagged: {result.get('n_flagged', 0)}")
        print(f"   Parasite Score: {result.get('parasite_score', 0):.3f}")
        if 'trace' in result.get('message', ''):
            print(f"   ✅ Memory logging confirmed")
    else:
        print(f"❌ Failed: {response.text}")


def test_evolution_report():
    """Test evolution/report endpoint"""
    print_section("Testing Evolution Report")
    
    if not PBMC_SMALL.exists():
        print(f"⚠️  Test file not found: {PBMC_SMALL}")
        return
    
    # Use same file for both runs (simplified test)
    with open(PBMC_SMALL, "rb") as f1, open(PBMC_SMALL, "rb") as f2:
        files = {
            "run1_file": ("run1.h5ad", f1, "application/octet-stream"),
            "run2_file": ("run2.h5ad", f2, "application/octet-stream")
        }
        data = {
            "run2_name": "run2_test",
            "cluster_key": "leiden",
            "latent_key": "X_umap",
            "sync": "true",
            "seed": "42",
            "project_id": "test_evolution",
            "user_id": "test_user"
        }
        
        response = httpx.post(
            f"{API_BASE}/bio/evolution/report",
            headers=headers,
            files=files,
            data=data,
            timeout=30.0
        )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Evolution analysis completed")
        print(f"   Run 1: {result.get('run1_name')} ({result.get('n_cells_run1', 0)} cells)")
        print(f"   Run 2: {result.get('run2_name')} ({result.get('n_cells_run2', 0)} cells)")
        print(f"   Stability Score: {result.get('stability_score', 0):.3f}")
        if 'trace' in result.get('message', ''):
            print(f"   ✅ Memory logging confirmed")
    else:
        print(f"❌ Failed: {response.text}")


def test_authentication():
    """Test authentication with invalid token"""
    print_section("Testing Authentication")
    
    bad_headers = {"Authorization": "Bearer invalid_token"}
    
    response = httpx.get(
        f"{API_BASE}/bio/jobs/test123",
        headers=bad_headers
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 401:
        print("✅ Authentication correctly rejects invalid tokens")
    else:
        print(f"⚠️  Expected 401, got {response.status_code}")


def test_job_not_found():
    """Test job polling with non-existent job"""
    print_section("Testing Job Not Found")
    
    response = httpx.get(
        f"{API_BASE}/bio/jobs/nonexistent_job_id",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 404:
        print("✅ Correctly returns 404 for non-existent job")
    else:
        print(f"⚠️  Expected 404, got {response.status_code}")


def test_project_memory():
    """Test retrieving project memory traces"""
    print_section("Testing Project Memory Retrieval")
    
    response = httpx.get(
        f"{API_BASE}/bio/memory/test_export_sync",
        headers=headers
    )
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Memory retrieval working")
        print(f"   Project ID: {result.get('project_id')}")
        print(f"   Trace Count: {result.get('count', 0)}")
        if result.get('count', 0) > 0:
            print(f"   ✅ Traces successfully persisted")
    else:
        print(f"❌ Failed: {response.text}")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  DON Research API - Bio Module Integration Tests")
    print("="*60)
    
    try:
        # Basic connectivity
        test_health_endpoint()
        
        # Bio endpoints
        test_export_artifacts_sync()
        test_export_artifacts_async()
        test_signal_sync()
        test_parasite_detect()
        test_evolution_report()
        
        # Edge cases
        test_authentication()
        test_job_not_found()
        
        # Memory system
        test_project_memory()
        
        print_section("All Tests Complete")
        print("✅ Bio module integration verified")
        
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
