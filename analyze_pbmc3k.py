#!/usr/bin/env python3
"""
Real PBMC 3k analysis with DON Research Node
Let the math speak - no assumptions
"""

import scipy.io as sio
import scipy.sparse as sp
import numpy as np
import requests
import json

# Load PBMC 3k
print("Loading PBMC 3k dataset...")
matrix = sio.mmread('filtered_gene_bc_matrices/hg19/matrix.mtx').tocsr()
with open('filtered_gene_bc_matrices/hg19/genes.tsv') as f:
    genes = [line.strip().split('\t')[1] for line in f]
with open('filtered_gene_bc_matrices/hg19/barcodes.tsv') as f:
    barcodes = [line.strip() for line in f]

print(f"\nDataset: {len(genes)} genes × {len(barcodes)} cells")
print(f"Sparsity: {100 * matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.2f}% non-zero\n")

# Test with increasing cell counts
headers = {'Authorization': 'Bearer demo_token'}
api_url = 'http://localhost:8080/api/v1/genomics/compress'

for n_cells in [100, 250, 500, 1000, 1500, 2000]:
    if n_cells > len(barcodes):
        break
    
    # Sample cells
    indices = np.random.choice(len(barcodes), n_cells, replace=False)
    dense_subset = matrix[:, indices].toarray().T.tolist()
    
    data = {
        'gene_names': genes,
        'expression_matrix': dense_subset,
        'cell_metadata': [{'barcode': barcodes[i]} for i in indices]
    }
    
    payload = {
        'data': data,
        'target_dims': 64,
        'alpha': 0.1,
        'engine': 'real_qac',
        'institution': 'demo'
    }
    
    print(f"Running {n_cells} cells...")
    response = requests.post(api_url, json=payload, headers=headers, timeout=300)
    
    if response.status_code == 200:
        result = response.json()
        stats = result.get('compression_stats', {})
        
        orig = stats.get('original_dimensions')
        comp = stats.get('compressed_dimensions')
        ratio = stats.get('compression_ratio')
        
        print(f"  {orig} → {comp} dims | {ratio} | Rank: {stats.get('rank')}")
    else:
        print(f"  Error: {response.status_code}")

print("\nDone. The math has spoken.")
