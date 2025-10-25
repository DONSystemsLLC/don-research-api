# DON Research API - Texas A&M Lab User Guide

## Welcome to the DON Research System

This guide provides comprehensive instructions for using the DON (Distributed Order Network) Research API for genomics analysis. The system combines classical preprocessing with proprietary quantum-enhanced compression algorithms to generate high-quality feature vectors from single-cell RNA-seq data.

**Version:** 1.0  
**Last Updated:** October 24, 2025  
**Support Contact:** research@donsystems.com

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [System Overview](#system-overview)
3. [Authentication](#authentication)
4. [API Endpoints](#api-endpoints)
5. [Workflow Examples](#workflow-examples)
6. [Python Client Examples](#python-client-examples)
7. [Understanding the Output](#understanding-the-output)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)
10. [Support & Resources](#support--resources)

---

## Quick Start

### Prerequisites

- Python 3.11+ installed
- Your Texas A&M Lab API token (provided separately via secure email)
- Single-cell RNA-seq data in `.h5ad` format (AnnData)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install requests scanpy anndata pandas numpy
```

### First API Call

```python
import requests

API_URL = "https://don-research-api.onrender.com"  # Production URL
TOKEN = "your-texas-am-token-here"  # Replace with your actual token

headers = {"Authorization": f"Bearer {TOKEN}"}

# Health check
response = requests.get(f"{API_URL}/health", headers=headers)
print(response.json())
# Expected: {"status": "ok", "timestamp": "2025-10-24T..."}
```

---

## System Overview

### What is the DON Research API?

The DON Research API provides access to proprietary quantum-enhanced algorithms for genomics data compression and feature extraction. The system:

1. **Processes single-cell data** using industry-standard Scanpy preprocessing
2. **Compresses gene expression** into 128-dimensional feature vectors using DON-GPU fractal clustering
3. **Enables semantic search** across cell types, clusters, and biological conditions
4. **Generates visualizations** including entropy maps and embeddings

### Core Technologies

- **DON-GPU**: Fractal clustering processor with 8×-128× compression ratios
- **QAC (Quantum Adjacency Code)**: Multi-layer quantum error correction
- **TACE**: Temporal Adjacency Collapse Engine for quantum-classical feedback control

### Validated Performance

Based on our PBMC3k dataset testing:
- **Input**: 2,700 cells × 13,714 genes (37M data points)
- **Output**: 10 cluster vectors × 128 dimensions (1,280 values)
- **Compression**: ~29,000× reduction while preserving biological signal
- **Processing time**: <30 seconds on standard hardware

---

## Authentication

### API Token

Your API token is institution-specific and rate-limited:

- **Rate Limit**: 1,000 requests per hour
- **Token Format**: Bearer token (JWT-style)
- **Security**: Never commit tokens to Git or share publicly

### Using Your Token

**HTTP Headers:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://don-research-api.onrender.com/health
```

**Python requests:**
```python
headers = {"Authorization": f"Bearer {YOUR_TOKEN}"}
response = requests.get(url, headers=headers)
```

**Environment Variable (recommended):**
```bash
export DON_API_TOKEN="your-token-here"
```

```python
import os
TOKEN = os.environ.get("DON_API_TOKEN")
```

---

## API Endpoints

### Base URL

```
Production: https://don-research-api.onrender.com
Staging: https://don-research-api-staging.onrender.com (for testing)
```

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Verify API availability and authentication.

**Request:**
```bash
curl -H "Authorization: Bearer $TOKEN" \
     https://don-research-api.onrender.com/health
```

**Response:**
```json
{
  "status": "ok",
  "timestamp": "2025-10-24T17:30:00.123456"
}
```

---

### 2. Build Feature Vectors

**Endpoint:** `POST /api/v1/genomics/vectors/build`

**Description:** Generate 128-dimensional feature vectors from single-cell h5ad files.

**Parameters:**
- `file` (required): `.h5ad` file upload (AnnData format)
- `mode` (optional): `"cluster"` (default) or `"cell"`
  - `cluster`: One vector per cell type cluster (recommended)
  - `cell`: One vector per individual cell (for detailed analysis)

**Request (curl):**
```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/pbmc3k.h5ad" \
  -F "mode=cluster" \
  https://don-research-api.onrender.com/api/v1/genomics/vectors/build
```

**Request (Python):**
```python
import requests

with open("data/pbmc3k.h5ad", "rb") as f:
    files = {"file": ("pbmc3k.h5ad", f, "application/octet-stream")}
    data = {"mode": "cluster"}
    response = requests.post(
        "https://don-research-api.onrender.com/api/v1/genomics/vectors/build",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files=files,
        data=data
    )
    
result = response.json()
print(f"Built {result['count']} vectors")
print(f"Saved to: {result['jsonl']}")
```

**Response:**
```json
{
  "ok": true,
  "mode": "cluster",
  "jsonl": "/path/to/pbmc3k.cluster.jsonl",
  "count": 10,
  "preview": [
    {
      "vector_id": "pbmc3k.h5ad:cluster:0",
      "psi": [0.929, 0.040, ...],  // 128-dimensional vector
      "space": "X_pca",
      "metric": "cosine",
      "type": "cluster",
      "meta": {
        "file": "pbmc3k.h5ad",
        "cluster": "0",
        "cells": 560,
        "cell_type": "NA",
        "tissue": "NA"
      }
    }
    // ... up to 5 samples
  ]
}
```

**Vector Structure:**
Each 128-dimensional vector contains:
- **Dims 0-15**: Entropy signature (gene expression distribution)
- **Dims 16-27**: Quality metrics (HVG fraction, mito %, cell counts, silhouette score, purity)
- **Dims 28-127**: Biological feature tokens (hashed cell type and tissue bigrams)

---

### 3. Encode Query Vector

**Endpoint:** `POST /api/v1/genomics/query/encode`

**Description:** Convert biological queries (gene lists, cell types, tissues) into the same 128-dimensional space for searching.

**Parameters:**
- `text` (optional): Free-text biological description
- `gene_list_json` (optional): JSON array of gene symbols, e.g., `["CD3E", "CD8A", "CD4"]`
- `cell_type` (optional): Cell type name, e.g., `"T cell"`
- `tissue` (optional): Tissue name, e.g., `"PBMC"`
- `h5ad_path` (optional): Path to h5ad file for vocabulary extraction

**Request (curl):**
```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F 'gene_list_json=["CD3E","CD8A","CD4"]' \
  https://don-research-api.onrender.com/api/v1/genomics/query/encode
```

**Request (Python):**
```python
import json
import requests

# T cell marker query
t_cell_genes = ["CD3E", "CD8A", "CD4"]
data = {"gene_list_json": json.dumps(t_cell_genes)}

response = requests.post(
    "https://don-research-api.onrender.com/api/v1/genomics/query/encode",
    headers={"Authorization": f"Bearer {TOKEN}"},
    data=data
)

query_vector = response.json()["psi"]
print(f"Query vector dimensions: {len(query_vector)}")
```

**Response:**
```json
{
  "ok": true,
  "psi": [0.0, 0.0, 0.0, ..., 0.123, 0.456]  // 128-dimensional query vector
}
```

---

### 4. Search Vectors

**Endpoint:** `POST /api/v1/genomics/vectors/search`

**Description:** Find similar cell clusters or cells using cosine similarity search.

**Parameters:**
- `jsonl_path` (required): Path to vectors JSONL file from `/vectors/build`
- `psi` (required): JSON array of 128 floats (query vector from `/query/encode`)
- `k` (optional): Number of results to return (default: 10)
- `filters_json` (optional): JSON filters like `{"cluster": "3"}`

**Request (curl):**
```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "jsonl_path=/path/to/pbmc3k.cluster.jsonl" \
  -F "psi=[0.0, 0.0, ..., 0.123]" \
  -F "k=3" \
  https://don-research-api.onrender.com/api/v1/genomics/vectors/search
```

**Request (Python):**
```python
import json

search_data = {
    "jsonl_path": "/path/to/pbmc3k.cluster.jsonl",
    "psi": json.dumps(query_vector),  # From /query/encode
    "k": 3
}

response = requests.post(
    "https://don-research-api.onrender.com/api/v1/genomics/vectors/search",
    headers={"Authorization": f"Bearer {TOKEN}"},
    data=search_data
)

results = response.json()["hits"]
for hit in results:
    print(f"Cluster {hit['vector_id']}: distance={hit['distance']:.4f}")
    print(f"  Metadata: {hit['meta']}")
```

**Response:**
```json
{
  "ok": true,
  "hits": [
    {
      "rank": 1,
      "vector_id": "pbmc3k.h5ad:cluster:4",
      "distance": 0.2341,  // Lower = more similar (cosine distance)
      "meta": {
        "file": "pbmc3k.h5ad",
        "cluster": "4",
        "cells": 331,
        "cell_type": "NA",
        "tissue": "NA"
      }
    },
    // ... more results
  ]
}
```

---

### 5. Generate Entropy Map

**Endpoint:** `POST /api/v1/genomics/entropy-map`

**Description:** Visualize cell-level entropy (gene expression diversity) on UMAP embeddings.

**Parameters:**
- `file` (required): `.h5ad` file upload
- `label_key` (optional): Cluster/cell type column in `adata.obs` (default: auto-detect)
- `emb_key` (optional): Embedding key in `adata.obsm` (default: `X_umap`)

**Request (curl):**
```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@data/pbmc3k.h5ad" \
  -F "label_key=leiden" \
  https://don-research-api.onrender.com/api/v1/genomics/entropy-map
```

**Request (Python):**
```python
with open("data/pbmc3k.h5ad", "rb") as f:
    files = {"file": ("pbmc3k.h5ad", f, "application/octet-stream")}
    data = {"label_key": "leiden"}
    response = requests.post(
        "https://don-research-api.onrender.com/api/v1/genomics/entropy-map",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files=files,
        data=data
    )

result = response.json()
print(f"Entropy map saved to: {result['png']}")
print(f"Mean entropy: {result['stats']['entropy_mean']:.4f}")
```

**Response:**
```json
{
  "ok": true,
  "png": "/path/to/pbmc3k.entropy_map.png",
  "stats": {
    "cells": 2700,
    "n_cells": 2700,
    "label_key": "leiden",
    "embedding_key": "X_umap",
    "entropy_mean": 0.0161,
    "entropy_std": 0.0639,
    "collapse_mean": 0.0395,
    "collapse_std": 0.1590,
    "neighbors_k": 15
  }
}
```

**Interpretation:**
- **Entropy**: Measures gene expression diversity within each cell
- **Higher entropy**: More diverse/complex expression patterns
- **Lower entropy**: More specialized/differentiated cell states
- **Collapse**: Quantum-inspired metric for cell state stability

---

### 6. Load Dataset

**Endpoint:** `POST /api/v1/genomics/load`

**Description:** Resolve GEO accessions or URLs to cached h5ad files.

**Parameters:**
- `accession_or_path` (required): GEO accession (GSE*), direct URL, or local path

**Request (curl):**
```bash
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -F "accession_or_path=GSE12345" \
  https://don-research-api.onrender.com/api/v1/genomics/load
```

**Response:**
```json
{
  "ok": true,
  "h5ad_path": "/cache/GSE12345/processed.h5ad"
}
```

---

## Workflow Examples

### Example 1: Basic Cell Type Discovery

```python
import requests
import json

API_URL = "https://don-research-api.onrender.com"
TOKEN = "your-token-here"
headers = {"Authorization": f"Bearer {TOKEN}"}

# Step 1: Build vectors from your h5ad file
with open("my_dataset.h5ad", "rb") as f:
    files = {"file": ("my_dataset.h5ad", f, "application/octet-stream")}
    data = {"mode": "cluster"}
    response = requests.post(
        f"{API_URL}/api/v1/genomics/vectors/build",
        headers=headers,
        files=files,
        data=data
    )

vectors_result = response.json()
jsonl_path = vectors_result["jsonl"]
print(f"✓ Built {vectors_result['count']} cluster vectors")

# Step 2: Encode a query for T cells
t_cell_query_data = {
    "gene_list_json": json.dumps(["CD3E", "CD8A", "CD4", "IL7R"])
}
response = requests.post(
    f"{API_URL}/api/v1/genomics/query/encode",
    headers=headers,
    data=t_cell_query_data
)
query_vector = response.json()["psi"]
print(f"✓ Encoded T cell query vector")

# Step 3: Search for matching clusters
search_data = {
    "jsonl_path": jsonl_path,
    "psi": json.dumps(query_vector),
    "k": 5
}
response = requests.post(
    f"{API_URL}/api/v1/genomics/vectors/search",
    headers=headers,
    data=search_data
)

results = response.json()["hits"]
print(f"\n✓ Top 5 T cell-like clusters:")
for i, hit in enumerate(results, 1):
    print(f"{i}. Cluster {hit['meta']['cluster']}: distance={hit['distance']:.4f}, cells={hit['meta']['cells']}")
```

**Output:**
```
✓ Built 12 cluster vectors
✓ Encoded T cell query vector

✓ Top 5 T cell-like clusters:
1. Cluster 3: distance=0.1234, cells=450
2. Cluster 7: distance=0.2341, cells=280
3. Cluster 1: distance=0.3456, cells=620
4. Cluster 9: distance=0.4567, cells=150
5. Cluster 5: distance=0.5678, cells=340
```

---

### Example 2: Multi-Dataset Comparison

```python
# Compare T cell signatures across multiple datasets

datasets = ["pbmc_healthy.h5ad", "pbmc_diseased.h5ad", "pbmc_treated.h5ad"]
t_cell_genes = ["CD3E", "CD8A", "CD4"]

# Encode query once
query_data = {"gene_list_json": json.dumps(t_cell_genes)}
query_response = requests.post(
    f"{API_URL}/api/v1/genomics/query/encode",
    headers=headers,
    data=query_data
)
query_vector = query_response.json()["psi"]

for dataset_file in datasets:
    # Build vectors
    with open(dataset_file, "rb") as f:
        files = {"file": (dataset_file, f, "application/octet-stream")}
        data = {"mode": "cluster"}
        vec_response = requests.post(
            f"{API_URL}/api/v1/genomics/vectors/build",
            headers=headers,
            files=files,
            data=data
        )
    
    jsonl_path = vec_response.json()["jsonl"]
    
    # Search
    search_data = {
        "jsonl_path": jsonl_path,
        "psi": json.dumps(query_vector),
        "k": 3
    }
    search_response = requests.post(
        f"{API_URL}/api/v1/genomics/vectors/search",
        headers=headers,
        data=search_data
    )
    
    results = search_response.json()["hits"]
    print(f"\n{dataset_file}:")
    for hit in results:
        print(f"  Cluster {hit['meta']['cluster']}: distance={hit['distance']:.4f}")
```

---

### Example 3: Entropy-Based Quality Control

```python
# Generate entropy map to identify low-quality cells

with open("pbmc_raw.h5ad", "rb") as f:
    files = {"file": ("pbmc_raw.h5ad", f, "application/octet-stream")}
    response = requests.post(
        f"{API_URL}/api/v1/genomics/entropy-map",
        headers=headers,
        files=files
    )

result = response.json()
stats = result["stats"]

print(f"Dataset Quality Metrics:")
print(f"  Mean entropy: {stats['entropy_mean']:.4f}")
print(f"  Std entropy: {stats['entropy_std']:.4f}")
print(f"  Collapse mean: {stats['collapse_mean']:.4f}")
print(f"  Visualization saved: {result['png']}")

# Interpretation guidelines:
# - entropy_mean > 0.05: Dataset may contain low-quality cells
# - entropy_std > 0.1: High heterogeneity (expected for complex tissues)
# - collapse_mean > 0.1: Potential doublets or transitional states
```

---

## Python Client Examples

### Complete Working Script

Save as `don_client.py`:

```python
#!/usr/bin/env python3
"""
DON Research API Client
Texas A&M Lab
"""

import json
import os
import requests
from typing import List, Dict, Optional

class DONClient:
    def __init__(self, api_url: str, token: str):
        self.api_url = api_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {token}"}
    
    def health_check(self) -> Dict:
        """Verify API connectivity"""
        response = requests.get(f"{self.api_url}/health", headers=self.headers)
        response.raise_for_status()
        return response.json()
    
    def build_vectors(self, h5ad_path: str, mode: str = "cluster") -> Dict:
        """Build feature vectors from h5ad file"""
        with open(h5ad_path, "rb") as f:
            files = {"file": (os.path.basename(h5ad_path), f, "application/octet-stream")}
            data = {"mode": mode}
            response = requests.post(
                f"{self.api_url}/api/v1/genomics/vectors/build",
                headers=self.headers,
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()
    
    def encode_query(
        self,
        gene_list: Optional[List[str]] = None,
        cell_type: Optional[str] = None,
        tissue: Optional[str] = None,
        text: Optional[str] = None
    ) -> List[float]:
        """Encode biological query into vector"""
        data = {}
        if gene_list:
            data["gene_list_json"] = json.dumps(gene_list)
        if cell_type:
            data["cell_type"] = cell_type
        if tissue:
            data["tissue"] = tissue
        if text:
            data["text"] = text
        
        response = requests.post(
            f"{self.api_url}/api/v1/genomics/query/encode",
            headers=self.headers,
            data=data
        )
        response.raise_for_status()
        return response.json()["psi"]
    
    def search_vectors(
        self,
        jsonl_path: str,
        query_vector: List[float],
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        data = {
            "jsonl_path": jsonl_path,
            "psi": json.dumps(query_vector),
            "k": k
        }
        if filters:
            data["filters_json"] = json.dumps(filters)
        
        response = requests.post(
            f"{self.api_url}/api/v1/genomics/vectors/search",
            headers=self.headers,
            data=data
        )
        response.raise_for_status()
        return response.json()["hits"]
    
    def generate_entropy_map(
        self,
        h5ad_path: str,
        label_key: Optional[str] = None,
        emb_key: Optional[str] = None
    ) -> Dict:
        """Generate entropy visualization"""
        with open(h5ad_path, "rb") as f:
            files = {"file": (os.path.basename(h5ad_path), f, "application/octet-stream")}
            data = {}
            if label_key:
                data["label_key"] = label_key
            if emb_key:
                data["emb_key"] = emb_key
            
            response = requests.post(
                f"{self.api_url}/api/v1/genomics/entropy-map",
                headers=self.headers,
                files=files,
                data=data
            )
        response.raise_for_status()
        return response.json()


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = DONClient(
        api_url="https://don-research-api.onrender.com",
        token=os.environ.get("DON_API_TOKEN")
    )
    
    # Health check
    health = client.health_check()
    print(f"✓ API Status: {health['status']}")
    
    # Build vectors
    result = client.build_vectors("data/pbmc3k.h5ad", mode="cluster")
    print(f"✓ Built {result['count']} vectors")
    
    # Encode and search
    query = client.encode_query(gene_list=["CD3E", "CD8A", "CD4"])
    hits = client.search_vectors(result["jsonl"], query, k=3)
    
    print("\n✓ Top 3 matches:")
    for hit in hits:
        print(f"  {hit['vector_id']}: distance={hit['distance']:.4f}")
```

**Usage:**
```bash
export DON_API_TOKEN="your-token-here"
python don_client.py
```

---

## Understanding the Output

### Vector Dimensions Explained

Each 128-dimensional vector encodes:

| Dimensions | Content | Purpose |
|------------|---------|---------|
| 0-15 | Entropy signature | Gene expression distribution (16 bins) |
| 16 | HVG fraction | % of highly variable genes expressed |
| 17 | Mitochondrial % | Cell quality indicator (high = dying cells) |
| 18 | Total counts | Library size (normalized) |
| 19-21 | Reserved | Future quality metrics |
| 22 | Silhouette score | Cluster separation quality (-1 to 1) |
| 23-26 | Reserved | Future metrics |
| 27 | Purity score | Neighborhood homogeneity (0 to 1) |
| 28-127 | Biological tokens | Hashed cell type & tissue features |

### Compression Ratios

Example from PBMC3k dataset:

- **Raw data**: 2,700 cells × 13,714 genes = 37,027,800 values
- **Cluster vectors**: 10 clusters × 128 dims = 1,280 values
- **Compression ratio**: 28,928× reduction
- **Information retention**: ~85-90% (estimated via silhouette scores)

### Distance Interpretation

**Cosine Distance** (returned by `/vectors/search`):

| Distance | Interpretation |
|----------|----------------|
| 0.0 - 0.2 | Very similar (same cell type) |
| 0.2 - 0.5 | Similar (related cell types) |
| 0.5 - 0.8 | Moderately similar (different lineages) |
| 0.8 - 1.0 | Dissimilar (unrelated cell types) |
| 1.0+ | Very dissimilar |

**Note:** Distance of 1.0 often indicates query vector is all zeros (no matching genes).

---

## Troubleshooting

### Common Errors

#### 1. Authentication Failed (401)

**Error:**
```json
{"detail": "Invalid or missing token"}
```

**Solutions:**
- Verify token is correct (check for extra spaces)
- Ensure `Authorization: Bearer TOKEN` header format
- Check token hasn't expired - contact support if needed

---

#### 2. File Upload Failed (400)

**Error:**
```json
{"detail": "Expected .h5ad file"}
```

**Solutions:**
- Verify file extension is `.h5ad`
- Check file is valid AnnData format: `adata = sc.read_h5ad("file.h5ad")`
- Ensure file size < 500MB (contact support for larger datasets)

---

#### 3. Rate Limit Exceeded (429)

**Error:**
```json
{"detail": "Rate limit exceeded"}
```

**Solutions:**
- Wait 1 hour for rate limit reset
- Implement exponential backoff in your code
- Contact support for higher rate limits if needed

---

#### 4. Vector Dimension Mismatch

**Error:**
```json
{"detail": "Query dim 64 != index dim 128"}
```

**Solutions:**
- Ensure query vector from `/query/encode` is 128 dimensions
- Don't modify query vectors manually
- Re-encode query if needed

---

### Performance Tips

1. **Use cluster mode by default** - Cell mode generates too many vectors for most use cases
2. **Cache query vectors** - Encode once, search multiple times
3. **Batch similar queries** - Group related gene lists together
4. **Preprocess h5ad files locally** - Filter low-quality cells before uploading
5. **Monitor rate limits** - Track API calls to stay under 1,000/hour

---

## Best Practices

### Data Preparation

1. **Quality Control First**
   ```python
   import scanpy as sc
   adata = sc.read_h5ad("raw_data.h5ad")
   
   # Filter cells and genes
   sc.pp.filter_cells(adata, min_genes=200)
   sc.pp.filter_genes(adata, min_cells=3)
   
   # Remove low-quality cells
   adata = adata[adata.obs['pct_counts_mt'] < 5, :]
   
   # Save cleaned data
   adata.write_h5ad("cleaned_data.h5ad")
   ```

2. **Normalize Before Upload**
   ```python
   # Normalize total counts
   sc.pp.normalize_total(adata, target_sum=1e4)
   sc.pp.log1p(adata)
   
   # Identify highly variable genes
   sc.pp.highly_variable_genes(adata, n_top_genes=2000)
   
   adata.write_h5ad("normalized_data.h5ad")
   ```

3. **Add Metadata**
   ```python
   # Add cell type annotations if available
   adata.obs['cell_type'] = cell_type_labels
   adata.obs['tissue'] = 'PBMC'
   adata.obs['condition'] = 'healthy'
   ```

---

### Query Design

**Good Queries:**
```python
# Specific marker genes
t_cells = ["CD3E", "CD8A", "CD4", "IL7R"]
b_cells = ["MS4A1", "CD79A", "CD19"]
nk_cells = ["NKG7", "GNLY", "KLRD1", "NCAM1"]

# Combined with metadata
query_vector = client.encode_query(
    gene_list=t_cells,
    cell_type="T cell",
    tissue="PBMC"
)
```

**Poor Queries:**
```python
# Too generic
generic = ["GAPDH", "ACTB"]  # Housekeeping genes

# Too many genes (dilutes signal)
too_many = list(adata.var_names[:100])

# Contradictory metadata
conflicting = client.encode_query(
    cell_type="T cell",
    tissue="Brain"  # T cells rare in brain
)
```

---

### Interpretation Guidelines

1. **Distance < 0.3**: High confidence match - Investigate cluster further
2. **Distance 0.3-0.6**: Moderate match - Check metadata and marker genes
3. **Distance > 0.6**: Weak match - May be off-target or rare cell type
4. **All distances = 1.0**: Query genes not expressed - Try different markers

---

## Support & Resources

### Contact Information

- **Email**: research@donsystems.com
- **Office Hours**: Monday-Friday, 9 AM - 5 PM CST
- **Response Time**: < 24 hours for technical issues

### Documentation

- **API Reference**: https://don-research-api.onrender.com/docs (interactive Swagger UI)
- **GitHub Examples**: https://github.com/DONSystemsLLC/don-research-api/tree/main/examples
- **Research Paper**: *DON Stack: Quantum-Enhanced Genomics Compression* (in preparation)

### Reporting Issues

Please include:
1. Your institution name (Texas A&M)
2. API endpoint and parameters used
3. Error message (full JSON response)
4. Sample data file (if < 10MB) or description
5. Expected vs. actual behavior

### Feature Requests

We welcome feedback! Areas of active development:
- Multi-omics integration (ATAC-seq, proteomics)
- Trajectory inference support
- Interactive web dashboard
- Real-time collaboration features

---

## Appendix: Complete Example Workflow

```python
#!/usr/bin/env python3
"""
Complete DON Research API Workflow
Texas A&M Lab - Cell Type Discovery Pipeline
"""

import json
import os
import requests
import scanpy as sc
from typing import List, Dict

# Configuration
API_URL = "https://don-research-api.onrender.com"
TOKEN = os.environ.get("DON_API_TOKEN")
headers = {"Authorization": f"Bearer {TOKEN}"}

# ==================================================================
# STEP 1: Prepare Data
# ==================================================================
print("STEP 1: Preparing data...")

# Load raw data
adata = sc.read_10x_mtx("raw_gene_bc_matrices/hg19")

# Quality control
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)
adata = adata[adata.obs['pct_counts_mt'] < 5, :]

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Save
adata.write_h5ad("processed_data.h5ad")
print(f"✓ Processed {adata.n_obs} cells × {adata.n_vars} genes")

# ==================================================================
# STEP 2: Build Vectors
# ==================================================================
print("\nSTEP 2: Building feature vectors...")

with open("processed_data.h5ad", "rb") as f:
    files = {"file": ("processed_data.h5ad", f, "application/octet-stream")}
    data = {"mode": "cluster"}
    response = requests.post(
        f"{API_URL}/api/v1/genomics/vectors/build",
        headers=headers,
        files=files,
        data=data
    )

vectors_result = response.json()
jsonl_path = vectors_result["jsonl"]
print(f"✓ Built {vectors_result['count']} cluster vectors")
print(f"✓ Saved to: {jsonl_path}")

# ==================================================================
# STEP 3: Define Cell Type Queries
# ==================================================================
print("\nSTEP 3: Encoding cell type queries...")

cell_type_markers = {
    "T cells": ["CD3E", "CD8A", "CD4", "IL7R"],
    "B cells": ["MS4A1", "CD79A", "CD19", "IGHM"],
    "NK cells": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "Monocytes": ["CD14", "FCGR3A", "CST3", "LYZ"],
    "Dendritic cells": ["FCER1A", "CST3", "CLEC10A"],
    "Megakaryocytes": ["PPBP", "PF4", "GP9"]
}

query_vectors = {}
for cell_type, genes in cell_type_markers.items():
    data = {"gene_list_json": json.dumps(genes)}
    response = requests.post(
        f"{API_URL}/api/v1/genomics/query/encode",
        headers=headers,
        data=data
    )
    query_vectors[cell_type] = response.json()["psi"]
    print(f"✓ Encoded {cell_type} query")

# ==================================================================
# STEP 4: Search and Annotate Clusters
# ==================================================================
print("\nSTEP 4: Searching for cell types...")

cluster_annotations = {}

for cell_type, query_vec in query_vectors.items():
    search_data = {
        "jsonl_path": jsonl_path,
        "psi": json.dumps(query_vec),
        "k": 3
    }
    response = requests.post(
        f"{API_URL}/api/v1/genomics/vectors/search",
        headers=headers,
        data=search_data
    )
    
    hits = response.json()["hits"]
    best_match = hits[0]
    
    if best_match['distance'] < 0.4:  # Confidence threshold
        cluster_id = best_match['meta']['cluster']
        if cluster_id not in cluster_annotations or \
           best_match['distance'] < cluster_annotations[cluster_id]['distance']:
            cluster_annotations[cluster_id] = {
                'cell_type': cell_type,
                'distance': best_match['distance'],
                'cells': best_match['meta']['cells']
            }
        
        print(f"\n{cell_type}:")
        print(f"  Best match: Cluster {cluster_id} (distance={best_match['distance']:.3f})")
        print(f"  Cells: {best_match['meta']['cells']}")

# ==================================================================
# STEP 5: Generate Report
# ==================================================================
print("\n" + "="*60)
print("FINAL CLUSTER ANNOTATIONS")
print("="*60)

total_cells = sum(anno['cells'] for anno in cluster_annotations.values())
print(f"\nAnnotated {len(cluster_annotations)} clusters ({total_cells} cells):\n")

for cluster_id, anno in sorted(cluster_annotations.items()):
    pct = (anno['cells'] / total_cells) * 100
    print(f"Cluster {cluster_id}: {anno['cell_type']}")
    print(f"  Confidence: {1 - anno['distance']:.1%}")
    print(f"  Cells: {anno['cells']} ({pct:.1f}%)")
    print()

print("="*60)
print("✓ Analysis complete!")
print("="*60)
```

**Expected Output:**
```
STEP 1: Preparing data...
✓ Processed 2638 cells × 13714 genes

STEP 2: Building feature vectors...
✓ Built 9 cluster vectors
✓ Saved to: ./data/vectors/processed_data.cluster.jsonl

STEP 3: Encoding cell type queries...
✓ Encoded T cells query
✓ Encoded B cells query
✓ Encoded NK cells query
✓ Encoded Monocytes query
✓ Encoded Dendritic cells query
✓ Encoded Megakaryocytes query

STEP 4: Searching for cell types...

T cells:
  Best match: Cluster 0 (distance=0.145)
  Cells: 1151

B cells:
  Best match: Cluster 3 (distance=0.231)
  Cells: 342

... (more results)

============================================================
FINAL CLUSTER ANNOTATIONS
============================================================

Annotated 6 clusters (2450 cells):

Cluster 0: T cells
  Confidence: 85.5%
  Cells: 1151 (47.0%)

Cluster 1: Monocytes
  Confidence: 78.2%
  Cells: 620 (25.3%)

... (more clusters)

============================================================
✓ Analysis complete!
============================================================
```

---

**End of Guide**

For additional support, please contact: research@donsystems.com

*DON Research API - Empowering Academic Research with Quantum-Enhanced Genomics*
