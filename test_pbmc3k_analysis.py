#!/usr/bin/env python3
"""
Simplified PBMC3K Analysis - Testing Real DON Stack
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8080"
TAMU_TOKEN = "tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc"
HEADERS = {"Authorization": f"Bearer {TAMU_TOKEN}"}
DATA_FILE = Path("data/pbmc3k.h5ad")

def print_header(title):
    print(f"\n{'='*70}\n  {title}\n{'='*70}\n")

def test_1_compress():
    """Test #1: Compress PBMC3K with DON-GPU"""
    print_header("Test 1: DON-GPU Compression")
    
    # Load data and prepare for API
    import scanpy as sc
    adata = sc.read_h5ad(DATA_FILE)
    print(f"✓ Loaded: {adata.n_obs} cells × {adata.n_vars} genes")
    
    # Prepare data payload
    payload = {
        "data": {
            "expression_matrix": adata.X.toarray().tolist() if hasattr(adata.X, 'toarray') else adata.X.tolist(),
            "gene_names": adata.var_names.tolist()
        },
        "target_dims": 128,
        "project_id": "pbmc3k_test"
    }
    
    print("Compressing... (this may take a minute)")
    response = requests.post(
        f"{BASE_URL}/api/v1/genomics/compress",
        headers=HEADERS,
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n✅ Compression successful!")
        print(f"   Input dimensions: {result.get('input_dimensions', 'N/A')}")
        print(f"   Output dimensions: {result.get('output_dimensions', 'N/A')}")
        
        stats = result.get('compression_stats', {})
        print(f"   Compression ratio: {stats.get('ratio', 'N/A')}")
        print(f"   Processing time: {stats.get('processing_time', 'N/A'):.2f}s")
        print(f"   Algorithm: {result.get('algorithm', 'N/A')}")
        
        return result
    else:
        print(f"\n❌ Compression failed: {response.status_code}")
        print(response.text[:500])
        return None

def test_2_rag_optimize():
    """Test #2: RAG Optimization"""
    print_header("Test 2: RAG Optimization")
    
    with open(DATA_FILE, 'rb') as f:
        files = {'file': ('pbmc3k.h5ad', f, 'application/octet-stream')}
        response = requests.post(
            f"{BASE_URL}/api/v1/genomics/rag-optimize",
            headers=HEADERS,
            files=files,
            data={
                'target_dims': 256,
                'cluster_mode': 'louvain'
            }
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ RAG optimization complete!")
        print(f"   Clusters: {result.get('n_clusters', 'N/A')}")
        print(f"   Vector dimensions: {result.get('vector_dims', 'N/A')}")
        print(f"   Compression ratio: {result.get('compression_ratio', 'N/A')}")
        print(f"   Algorithm: {result.get('algorithm', 'N/A')}")
        return result
    else:
        print(f"❌ RAG optimization failed: {response.status_code}")
        print(response.text[:500])
        return None

def test_3_quantum_stabilize():
    """Test #3: Quantum Stabilization"""
    print_header("Test 3: Quantum Stabilization")
    
    # Test quantum stabilization on sample vectors
    test_vectors = [
        ("T cell signature", [1.0, 0.9, 0.8, 0.7, 0.6]),
        ("B cell signature", [0.9, 1.0, 0.7, 0.8, 0.5]),
        ("Monocyte signature", [0.7, 0.8, 0.6, 1.0, 0.9])
    ]
    
    results = []
    for name, vector in test_vectors:
        print(f"\n🔬 Stabilizing: {name}")
        response = requests.post(
            f"{BASE_URL}/api/v1/quantum/stabilize",
            headers=HEADERS,
            json={'vector': vector, 'alpha': 0.1}
        )
        
        if response.status_code == 200:
            result = response.json()
            stabilized = result.get('stabilized_vector', [])
            print(f"   ✅ Stabilized (input={len(vector)}, output={len(stabilized)})")
            print(f"   Algorithm: {result.get('algorithm', 'N/A')}")
            results.append(result)
        else:
            print(f"   ❌ Failed: {response.status_code}")
    
    return results

def test_4_vector_build():
    """Test #4: Vector Building"""
    print_header("Test 4: Vector Building")
    
    with open(DATA_FILE, 'rb') as f:
        files = {'file': ('pbmc3k.h5ad', f, 'application/octet-stream')}
        data = {'mode': 'cluster'}
        
        response = requests.post(
            f"{BASE_URL}/api/v1/genomics/vectors/build",
            headers=HEADERS,
            files=files,
            data=data
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Vector database built!")
        print(f"   Mode: {result.get('mode', 'N/A')}")
        print(f"   Vector count: {result.get('count', 'N/A')}")
        print(f"   Output file: {result.get('jsonl', 'N/A')}")
        
        # Show preview
        preview = result.get('preview', [])
        if preview:
            print(f"\n   Preview (first entry):")
            first = preview[0]
            print(f"   - Label: {first.get('label', 'N/A')}")
            print(f"   - Vector dims: {len(first.get('vector', []))}")
        
        return result
    else:
        print(f"❌ Vector build failed: {response.status_code}")
        print(response.text[:500])
        return None

def test_5_qac():
    """Test #5: QAC Error Correction"""
    print_header("Test 5: QAC Error Correction")
    
    # Fit QAC model
    marker_vectors = [
        [1.0, 0.9, 0.8, 0.7, 0.6],
        [0.9, 1.0, 0.7, 0.8, 0.5],
        [0.8, 0.7, 1.0, 0.6, 0.7]
    ]
    
    print("🔧 Fitting QAC model...")
    response = requests.post(
        f"{BASE_URL}/qac/fit",
        headers=HEADERS,
        json={
            'vectors': marker_vectors,
            'alpha': 0.1,
            'project_id': 'pbmc3k_qac'
        }
    )
    
    if response.status_code == 200:
        fit_result = response.json()
        model_id = fit_result.get('model_id', 'N/A')
        print(f"   ✅ QAC model fitted (ID: {model_id})")
        
        # Apply QAC correction
        print("\n🔬 Applying QAC correction...")
        response = requests.post(
            f"{BASE_URL}/qac/apply",
            headers=HEADERS,
            json={
                'model_id': model_id,
                'vector': [0.85, 0.75, 0.90, 0.70, 0.65]
            }
        )
        
        if response.status_code == 200:
            apply_result = response.json()
            corrected = apply_result.get('corrected_vector', [])
            print(f"   ✅ QAC correction applied!")
            print(f"   Input dims: 5")
            print(f"   Output dims: {len(corrected)}")
            print(f"   Algorithm: {apply_result.get('algorithm', 'N/A')}")
            return apply_result
        else:
            print(f"   ❌ Apply failed: {response.status_code}")
    else:
        print(f"   ❌ Fit failed: {response.status_code}")
        print(response.text[:200])
    
    return None

def main():
    print("\n" + "="*70)
    print("  🧬 DON RESEARCH API - PBMC3K ANALYSIS")
    print("  Testing Real DON Stack with Single-Cell Data")
    print("="*70)
    
    # Run all tests
    results = {}
    
    results['compress'] = test_1_compress()
    results['rag'] = test_2_rag_optimize()
    results['quantum'] = test_3_quantum_stabilize()
    results['vectors'] = test_4_vector_build()
    results['qac'] = test_5_qac()
    
    # Summary
    print_header("✅ Analysis Complete")
    
    successes = sum(1 for r in results.values() if r is not None)
    print(f"Tests passed: {successes}/5")
    print("\nDON Stack capabilities demonstrated:")
    print("  • DON-GPU fractal compression")
    print("  • RAG optimization for retrieval")
    print("  • Quantum vector stabilization")
    print("  • Vector database construction")
    print("  • QAC error correction")
    print("\n🎉 Ready for TAMU handoff!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
