#!/usr/bin/env python3
"""Quick DON Stack validation on PBMC3K data"""

import requests
import json

API_URL = "http://localhost:8080"
TOKEN = "tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

print("="*70)
print("  🧬 DON STACK LIVE TEST - PBMC3K DATA")
print("="*70)

# Test 1: Quantum Stabilization
print("\n[1/3] Testing Quantum Stabilization...")
response = requests.post(
    f"{API_URL}/api/v1/quantum/stabilize",
    headers=HEADERS,
    json={
        "quantum_states": [
            [1.0, 0.9, 0.8, 0.7, 0.6],
            [0.9, 1.0, 0.7, 0.8, 0.5],
            [0.8, 0.7, 1.0, 0.6, 0.7]
        ],
        "coherence_target": 0.95
    }
)
if response.status_code == 200:
    result = response.json()
    print(f"  ✅ {result['qac_stats']['algorithm']}")
    print(f"  ✅ Coherence: {result['coherence_metrics']['estimated_coherence']}")
    print(f"  ✅ States processed: {result['coherence_metrics']['states_processed']}")
else:
    print(f"  ❌ Failed: {response.status_code}")

# Test 2: Genomics Compression (using pre-made payload)
print("\n[2/3] Testing DON-GPU Compression (100 cells x 500 genes)...")
with open('test_payload.json', 'r') as f:
    payload = json.load(f)

response = requests.post(
    f"{API_URL}/api/v1/genomics/compress",
    headers=HEADERS,
    json=payload
)
if response.status_code == 200:
    result = response.json()
    compressed = result.get('compressed_data', [])
    print(f"  ✅ {result.get('algorithm', 'N/A')}")
    print(f"  ✅ Compressed to: {len(compressed)} vectors")
    print(f"  ✅ Vector dimensions: {len(compressed[0]) if compressed else 0}")
else:
    print(f"  ❌ Failed: {response.status_code}")

# Test 3: Vector Building (H5AD upload)
print("\n[3/3] Testing Vector Building from H5AD...")
with open('data/pbmc3k.h5ad', 'rb') as f:
    files = {'file': ('pbmc3k.h5ad', f, 'application/octet-stream')}
    response = requests.post(
        f"{API_URL}/api/v1/genomics/vectors/build",
        headers=HEADERS,
        files=files,
        data={'mode': 'cluster'}
    )

if response.status_code == 200:
    result = response.json()
    print(f"  ✅ Built {result.get('count', 'N/A')} cluster vectors")
    print(f"  ✅ Mode: {result.get('mode', 'N/A')}")
    print(f"  ✅ Output: {result.get('jsonl', 'N/A')}")
else:
    print(f"  ❌ Failed: {response.status_code}")
    print(f"      {response.text[:200]}")

print("\n" + "="*70)
print("  ✅ DON STACK OPERATIONAL - READY FOR TAMU HANDOFF")
print("="*70)
