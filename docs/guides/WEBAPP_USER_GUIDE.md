# DON Research API - Web Application User Guide
## Texas A&M University Cai Lab

**Version:** 1.0  
**Last Updated:** January 2025  
**Production URL:** https://don-research.onrender.com

---

## Table of Contents

1. [Web Interface Overview](#web-interface-overview)
2. [Getting Started](#getting-started)
3. [Navigation Guide](#navigation-guide)
4. [Core Features](#core-features)
5. [API Endpoint Reference](#api-endpoint-reference)
6. [Interactive Documentation (Swagger UI)](#interactive-documentation-swagger-ui)
7. [Bio Module Features](#bio-module-features)
8. [Workflow Examples](#workflow-examples)
9. [Troubleshooting](#troubleshooting)
10. [Support Resources](#support-resources)

---

## Web Interface Overview

### What is the DON Research Web App?

The DON Research API web interface serves as your **primary documentation portal** and **interactive API explorer**. It provides:

- ✅ **Comprehensive User Guide**: Step-by-step instructions for all API features
- ✅ **Interactive API Documentation**: Test endpoints directly in your browser (Swagger UI)
- ✅ **Code Examples**: Copy-paste Python snippets for common workflows
- ✅ **Real-time Testing**: Execute API calls and see responses immediately
- ✅ **Cell Type References**: Curated marker genes for common cell types
- ✅ **Troubleshooting Resources**: Common error resolutions and best practices

### Accessing the Web Interface

**Production URL:** https://don-research.onrender.com

1. **Open your web browser** (Chrome, Firefox, Safari, or Edge)
2. **Navigate to:** `https://don-research.onrender.com`
3. **You'll see:** The main user guide homepage with navigation menu

**No login required** for viewing documentation - authentication is only needed when making API calls.

---

## Getting Started

### First Time Setup

Before using the web interface, ensure you have:

1. ✅ **API Token** (provided via secure email)
2. ✅ **Python 3.11+** installed locally
3. ✅ **Required packages:** `requests`, `scanpy`, `anndata`, `pandas`, `numpy`

### Installation Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate  # Windows

# Install packages
pip install requests scanpy anndata pandas numpy
```

### Verify Connection

Test your API access with this simple Python script:

```python
import requests

API_URL = "https://don-research.onrender.com"
TOKEN = "your-tamu-token-here"  # Replace with your actual token

headers = {"Authorization": f"Bearer {TOKEN}"}
response = requests.get(f"{API_URL}/health", headers=headers)
print(response.json())
```

**Expected output:** `{"status": "ok", "timestamp": "2025-01-24T..."}`

---

## Navigation Guide

### Main Navigation Bar

The sticky navigation bar at the top provides quick links to all sections:

| Link | Description |
|------|-------------|
| **Quick Start** | Jump to installation and first API call |
| **Authentication** | Token usage and security best practices |
| **API Endpoints** | Complete endpoint reference (POST/GET) |
| **Bio Module** | ResoTrace integration features |
| **Workflows** | Complete code examples for common tasks |
| **Troubleshooting** | Error solutions and debugging tips |
| **Support** | Contact information and office hours |
| **API Docs** | Opens Swagger UI in same tab |

### Page Structure

The user guide is organized into **collapsible sections** for easy scanning:

1. **🚀 Quick Start**: Get running in 5 minutes
2. **📊 System Overview**: Technology and performance metrics
3. **🔐 Authentication**: Token management
4. **🔌 API Endpoints**: Full endpoint documentation
5. **🧬 Bio Module**: Advanced ResoTrace workflows
6. **🔬 Workflow Examples**: Complete code samples
7. **🔧 Troubleshooting**: Common issues and solutions
8. **📞 Support**: Contact and resources

---

## Core Features

### 1. Quick Start Section

**Purpose:** Get your first API call working in under 5 minutes

**What you'll find:**
- ✅ Prerequisites checklist (Python, token, data format)
- ✅ Copy-paste installation commands
- ✅ Working health check example
- ✅ Success indicators (green boxes show expected results)

**How to use:**
1. Scroll to **Quick Start** via navigation
2. Copy the installation commands → Run in your terminal
3. Copy the Python health check code → Replace token → Run
4. Look for green success box confirming connection

### 2. System Overview

**Purpose:** Understand the technology and validated performance

**What you'll find:**
- ✅ DON Stack architecture explanation (DON-GPU, QAC, TACE)
- ✅ Performance metrics cards showing:
  - **Input Data**: 2,700 cells × 13,714 genes
  - **Compression Ratio**: 28,928× (37M → 1.3K values)
  - **Processing Time**: < 30 seconds
  - **Information Retention**: 85-90%

**How to interpret:**
- **Compression Ratio** tells you how much data reduction to expect
- **Processing Time** helps estimate workflow duration
- **Information Retention** indicates biological signal preservation

### 3. Authentication Section

**Purpose:** Secure token management and rate limit understanding

**What you'll find:**
- ✅ Token format explanation (Bearer token)
- ✅ Rate limit: **1,000 requests/hour** (Academic tier)
- ✅ Security best practices (environment variables)
- ✅ Code examples for HTTP headers and Python requests

**Best practices:**
```bash
# Store token as environment variable (recommended)
export DON_API_TOKEN="tamu_cai_lab_2025_..."

# Use in Python
import os
TOKEN = os.environ.get("DON_API_TOKEN")
```

**Never:**
- ❌ Commit tokens to Git repositories
- ❌ Share tokens in Slack/email without encryption
- ❌ Hardcode tokens in shared scripts

### 4. API Endpoints Reference

**Purpose:** Complete endpoint documentation with parameters and examples

The user guide documents **5 core endpoints** and **4 Bio module endpoints**:

#### Core Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Verify API availability |
| `/api/v1/genomics/vectors/build` | POST | Generate 128D feature vectors |
| `/api/v1/genomics/query/encode` | POST | Convert gene lists to query vectors |
| `/api/v1/genomics/vectors/search` | POST | Find similar cell clusters |
| `/api/v1/genomics/entropy-map` | POST | Visualize cell-level entropy |

#### Bio Module Endpoints (ResoTrace Integration)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/bio/export-artifacts` | POST | Export collapse maps & vectors |
| `/api/v1/bio/signal-sync` | POST | Compare pipeline runs |
| `/api/v1/bio/qc/parasite-detect` | POST | QC parasite detection |
| `/api/v1/bio/evolution/report` | POST | Stability analysis |

**Each endpoint section includes:**
- ✅ HTTP method badge (GET/POST color-coded)
- ✅ Complete parameter table (required vs optional)
- ✅ Copy-paste code examples
- ✅ Expected JSON response structure
- ✅ Interpretation guides for results

### 5. Interactive Examples

**Purpose:** Working code you can copy and modify immediately

**Example structure:**
1. **Problem statement**: "I want to find T cell clusters"
2. **Complete Python code**: Imports → API calls → Result processing
3. **Expected output**: Shows what success looks like
4. **Interpretation**: How to read the results

**How to use examples:**
1. Find relevant workflow in user guide
2. Copy entire code block (triple-backticks)
3. Replace `TOKEN` and file paths with your values
4. Run in Python interpreter or script
5. Compare your output to documented expected results

---

## Interactive Documentation (Swagger UI)

### What is Swagger UI?

**Swagger UI** is an interactive API testing tool embedded at `/docs`. It allows you to:

- ✅ **Test API calls directly in your browser** (no code required)
- ✅ **See real-time request/response data** (JSON formatting)
- ✅ **Explore all endpoints** with parameter descriptions
- ✅ **Authenticate once** and test multiple endpoints
- ✅ **Download response data** for validation

### Accessing Swagger UI

**Three ways to open:**

1. **From user guide navigation:** Click "API Docs" link (top right)
2. **Direct URL:** `https://don-research.onrender.com/docs`
3. **From any endpoint documentation:** Click "Interactive docs at /docs"

### Using Swagger UI: Step-by-Step

#### Step 1: Authenticate

1. Click the **green "Authorize" button** (top right)
2. Enter your token: `Bearer tamu_cai_lab_2025_...` (include "Bearer " prefix)
3. Click **Authorize** → **Close**

**✅ Success indicator:** Padlock icons turn from open 🔓 to locked 🔒

#### Step 2: Select an Endpoint

**Endpoint sections:**
- **Genomics**: Core vector operations (build, search, encode)
- **Bio**: ResoTrace integration (export, sync, QC)
- **Health**: System status check

**Color coding:**
- 🟢 **GET** (green): Read-only operations
- 🟡 **POST** (yellow): Data upload/processing

#### Step 3: Test an Endpoint

**Example: Build Feature Vectors**

1. **Expand** `/api/v1/genomics/vectors/build` (POST)
2. Click **"Try it out"** button (top right of section)
3. **Upload file:** Click "Choose File" → Select `.h5ad` file
4. **Set parameters:**
   - `mode`: `cluster` (dropdown)
5. Click **"Execute"** (blue button)
6. **View results:**
   - **Request URL**: Actual API call made
   - **Response body**: JSON result (formatted)
   - **Response headers**: Including `X-Trace-ID` for debugging

#### Step 4: Interpret Results

**Success (200 OK):**
```json
{
  "ok": true,
  "mode": "cluster",
  "count": 10,
  "jsonl": "/path/to/vectors.jsonl"
}
```

**Error (400/401/429):**
```json
{
  "detail": "Invalid or missing token"
}
```

Refer to **Troubleshooting** section in user guide for error resolutions.

### Swagger UI Best Practices

✅ **DO:**
- Use Swagger UI for **initial testing** and **exploration**
- Copy **Request URL** and **curl command** for scripting
- Test with **small datasets** first (< 1,000 cells)
- Save successful responses for validation

❌ **DON'T:**
- Upload **large files** (> 100MB) via browser (use Python scripts)
- Test **production workflows** in Swagger (use automated scripts)
- Repeatedly execute **slow endpoints** (respect rate limits)

---

## Bio Module Features

### Overview

The **Bio Module** provides advanced single-cell analysis workflows for **ResoTrace integration**. Key capabilities:

- ✅ **Export Artifacts**: Convert H5AD → ResoTrace collapse maps
- ✅ **Signal Sync**: Compare pipeline runs for reproducibility
- ✅ **Parasite Detection**: QC for ambient RNA, doublets, batch effects
- ✅ **Evolution Report**: Track pipeline stability over time

### Sync vs Async Execution Modes

**Every Bio endpoint supports two modes:**

| Mode | When to Use | Response Time | Use Case |
|------|-------------|---------------|----------|
| **Sync** (`sync=true`) | Small datasets (< 5K cells) | Immediate (< 30s) | Quick validation, exploratory analysis |
| **Async** (`sync=false`) | Large datasets (> 10K cells) | Background job | Production pipelines, batch processing |

### Feature 1: Export Artifacts

**Endpoint:** `POST /api/v1/bio/export-artifacts`

**What it does:**
- Converts `.h5ad` files into ResoTrace-compatible formats
- Generates collapse maps (cluster graph structure)
- Exports cell-level vector collections (embeddings)
- Includes PAGA connectivity (if available)

**Key parameters:**
- `file`: Your H5AD file upload
- `cluster_key`: Column in `adata.obs` with cluster labels (e.g., "leiden")
- `latent_key`: Embedding in `adata.obsm` (e.g., "X_umap", "X_pca")
- `sync`: `true` (immediate) or `false` (background job)

**Output files:**
1. `collapse_map.json`: Cluster graph structure
2. `collapse_vectors.jsonl`: Per-cell embeddings (128D)

**Example workflow:**
```python
# Synchronous export (small dataset)
with open("pbmc_3k.h5ad", "rb") as f:
    files = {"file": ("pbmc.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_umap",
        "sync": "true",
        "project_id": "cai_lab_pbmc_study"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/bio/export-artifacts",
        headers={"Authorization": f"Bearer {TOKEN}"},
        files=files,
        data=data
    )

result = response.json()
print(f"✓ Exported {result['nodes']} clusters, {result['vectors']} cells")
print(f"✓ Trace ID: {result.get('trace_id')}")
```

### Feature 2: Signal Sync

**Endpoint:** `POST /api/v1/bio/signal-sync`

**What it does:**
- Compares two collapse maps (e.g., technical replicates)
- Computes coherence score (0-1 scale)
- Detects node overlap and structural similarity
- Validates pipeline reproducibility

**Key parameters:**
- `artifact1`: First collapse map JSON file
- `artifact2`: Second collapse map JSON file
- `coherence_threshold`: Minimum score for "synchronized" status (default: 0.8)

**Interpretation:**

| Coherence Score | Meaning | Action |
|-----------------|---------|--------|
| 0.9 - 1.0 | Excellent consistency | Proceed with analysis |
| 0.7 - 0.9 | Good consistency | Expected for biological replicates |
| 0.5 - 0.7 | Moderate similarity | Review preprocessing parameters |
| < 0.5 | Low similarity | Investigate batch effects |

**Use cases:**
- ✅ Validating technical replicates
- ✅ Comparing preprocessing strategies
- ✅ Detecting batch effects
- ✅ Pipeline quality control

### Feature 3: Parasite Detector (QC)

**Endpoint:** `POST /api/v1/bio/qc/parasite-detect`

**What it does:**
- Flags low-quality cells ("parasites")
- Detects: Ambient RNA, doublets, batch effects
- Returns per-cell boolean flags
- Computes overall contamination score

**Key parameters:**
- `file`: Raw H5AD file (before filtering)
- `cluster_key`: Cluster column in `adata.obs`
- `batch_key`: Sample/batch column in `adata.obs`
- `ambient_threshold`: Ambient RNA cutoff (default: 0.15)
- `doublet_threshold`: Doublet score cutoff (default: 0.25)

**Output:**
```json
{
  "n_cells": 2700,
  "n_flagged": 135,
  "parasite_score": 5.0,
  "flags": [false, true, false, ...]  // Per-cell flags
}
```

**Recommended actions:**

| Parasite Score | Quality | Recommended Action |
|----------------|---------|-------------------|
| 0-5% | Excellent | Proceed without filtering |
| 5-15% | Good | Minor filtering recommended |
| 15-30% | Moderate | Filter flagged cells |
| > 30% | Poor | Review QC pipeline |

**Filtering flagged cells:**
```python
# After detecting parasites
flags = result["flags"]

# Load your AnnData
adata = sc.read_h5ad("data.h5ad")

# Filter out flagged cells
adata = adata[~np.array(flags), :]

# Save cleaned data
adata.write_h5ad("data_filtered.h5ad")
```

### Feature 4: Evolution Report

**Endpoint:** `POST /api/v1/bio/evolution/report`

**What it does:**
- Compares two pipeline runs (e.g., parameter sweep)
- Computes stability score (0-100%)
- Tracks delta metrics (cell counts, clusters, entropy)
- Detects pipeline drift over time

**Key parameters:**
- `run1_file`: Baseline H5AD file
- `run2_file`: Comparison H5AD file
- `run2_name`: Label for second run (e.g., "leiden_resolution_1.0")
- `cluster_key`: Cluster column
- `latent_key`: Embedding key

**Output:**
```json
{
  "run1_name": "baseline",
  "run2_name": "leiden_resolution_1.0",
  "n_cells_run1": 2700,
  "n_cells_run2": 2700,
  "stability_score": 87.5,
  "delta_metrics": {
    "delta_clusters": 2,
    "delta_entropy": 0.05
  }
}
```

**Stability score interpretation:**

| Score | Meaning | Action |
|-------|---------|--------|
| > 90% | Excellent stability | Robust pipeline |
| 70-90% | Good stability | Acceptable variation |
| 50-70% | Moderate drift | Review parameters |
| < 50% | High drift | Investigate batch effects |

### Bio Module Job Management

#### Polling Async Jobs

**Endpoint:** `GET /api/v1/bio/jobs/{job_id}`

```python
import time

job_id = "abc123..."  # From async submission

while True:
    response = requests.get(
        f"{API_URL}/api/v1/bio/jobs/{job_id}",
        headers={"Authorization": f"Bearer {TOKEN}"}
    )
    
    job = response.json()
    status = job["status"]  # pending, running, completed, failed
    
    if status == "completed":
        result = job["result"]
        break
    elif status == "failed":
        print(f"Error: {job['error']}")
        break
    
    time.sleep(2)  # Poll every 2 seconds
```

#### Retrieving Project Memory

**Endpoint:** `GET /api/v1/bio/memory/{project_id}`

```python
response = requests.get(
    f"{API_URL}/api/v1/bio/memory/cai_lab_pbmc_study",
    headers={"Authorization": f"Bearer {TOKEN}"}
)

memory = response.json()
print(f"Project: {memory['project_id']}")
print(f"Total operations: {memory['count']}")

for trace in memory["traces"]:
    print(f"  {trace['event_type']}: {trace['metrics']}")
```

**Use cases:**
- ✅ Audit trail for experiments
- ✅ Reproduce past analyses
- ✅ Track data lineage
- ✅ Debug pipeline issues

---

## Workflow Examples

### Workflow 1: Basic Cell Type Discovery

**Goal:** Identify T cell clusters in PBMC dataset

**Steps:**
1. Build cluster vectors from H5AD file
2. Encode T cell marker query (CD3E, CD8A, CD4, IL7R)
3. Search for matching clusters
4. Interpret distance scores

**Complete code:**

```python
import requests
import json

API_URL = "https://don-research.onrender.com"
TOKEN = "your-tamu-token-here"
headers = {"Authorization": f"Bearer {TOKEN}"}

# Step 1: Build cluster vectors
print("Step 1: Building vectors...")
with open("pbmc_3k.h5ad", "rb") as f:
    files = {"file": ("pbmc_3k.h5ad", f, "application/octet-stream")}
    response = requests.post(
        f"{API_URL}/api/v1/genomics/vectors/build",
        headers=headers,
        files=files,
        data={"mode": "cluster"}
    )

vectors_result = response.json()
jsonl_path = vectors_result["jsonl"]
print(f"✓ Built {vectors_result['count']} cluster vectors")

# Step 2: Encode T cell query
print("\nStep 2: Encoding T cell markers...")
t_cell_genes = ["CD3E", "CD8A", "CD4", "IL7R"]
query_data = {"gene_list_json": json.dumps(t_cell_genes)}

response = requests.post(
    f"{API_URL}/api/v1/genomics/query/encode",
    headers=headers,
    data=query_data
)
query_vector = response.json()["psi"]
print(f"✓ Encoded query vector (128 dimensions)")

# Step 3: Search for matching clusters
print("\nStep 3: Searching for T cell-like clusters...")
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
print(f"{'Rank':<6} {'Cluster':<10} {'Distance':<12} {'Cells':<8} {'Interpretation'}")
print("-" * 60)

for i, hit in enumerate(results, 1):
    cluster_id = hit['meta']['cluster']
    distance = hit['distance']
    cells = hit['meta']['cells']
    
    # Interpret distance
    if distance < 0.2:
        interp = "Very similar (likely T cells)"
    elif distance < 0.5:
        interp = "Similar (T cell-like)"
    elif distance < 0.8:
        interp = "Moderate (related lineage)"
    else:
        interp = "Dissimilar"
    
    print(f"{i:<6} {cluster_id:<10} {distance:<12.4f} {cells:<8} {interp}")
```

**Expected output:**
```
Step 1: Building vectors...
✓ Built 10 cluster vectors

Step 2: Encoding T cell markers...
✓ Encoded query vector (128 dimensions)

Step 3: Searching for T cell-like clusters...
✓ Top 5 T cell-like clusters:
Rank   Cluster    Distance     Cells    Interpretation
------------------------------------------------------------
1      0          0.1234       560      Very similar (likely T cells)
2      5          0.2456       234      Similar (T cell-like)
3      2          0.4567       412      Similar (T cell-like)
4      7          0.6789       89       Moderate (related lineage)
5      3          0.7890       345      Moderate (related lineage)
```

### Workflow 2: QC Pipeline with Parasite Detection

**Goal:** Clean dataset by removing low-quality cells

**Steps:**
1. Detect parasites in raw data
2. Filter flagged cells
3. Validate cleaned dataset

**Complete code:**

```python
import requests
import scanpy as sc
import numpy as np

API_URL = "https://don-research.onrender.com"
TOKEN = "your-tamu-token-here"
headers = {"Authorization": f"Bearer {TOKEN}"}

# Step 1: Detect parasites
print("Step 1: Detecting QC parasites...")
with open("pbmc_raw.h5ad", "rb") as f:
    files = {"file": ("pbmc_raw.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "batch_key": "sample",
        "ambient_threshold": "0.15",
        "doublet_threshold": "0.25",
        "sync": "true"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/bio/qc/parasite-detect",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()
flags = result["flags"]
n_cells = result["n_cells"]
n_flagged = result["n_flagged"]
parasite_score = result["parasite_score"]

print(f"✓ Analyzed {n_cells} cells")
print(f"✓ Flagged {n_flagged} cells ({n_flagged/n_cells*100:.1f}%)")
print(f"✓ Parasite score: {parasite_score:.1f}%")

# Step 2: Filter flagged cells
print("\nStep 2: Filtering flagged cells...")
adata = sc.read_h5ad("pbmc_raw.h5ad")
print(f"  Original: {adata.n_obs} cells")

# Keep only non-flagged cells
adata = adata[~np.array(flags), :]
print(f"  Filtered: {adata.n_obs} cells")

# Save cleaned data
adata.write_h5ad("pbmc_cleaned.h5ad")
print(f"✓ Saved cleaned data to pbmc_cleaned.h5ad")

# Step 3: Validate cleaned dataset
print("\nStep 3: Validating cleaned dataset...")
print(f"  Mean genes/cell: {adata.obs['n_genes'].mean():.0f}")
print(f"  Mean UMI/cell: {adata.obs['total_counts'].mean():.0f}")
print(f"  Median MT%: {adata.obs['pct_counts_mt'].median():.2f}%")
```

### Workflow 3: Pipeline Stability Analysis

**Goal:** Compare two Leiden resolution parameters for stability

**Complete code:**

```python
import requests

API_URL = "https://don-research.onrender.com"
TOKEN = "your-tamu-token-here"
headers = {"Authorization": f"Bearer {TOKEN}"}

print("Comparing Leiden resolutions: 0.5 vs 1.0")

with open("pbmc_leiden05.h5ad", "rb") as f1, \
     open("pbmc_leiden10.h5ad", "rb") as f2:
    
    files = {
        "run1_file": ("run1.h5ad", f1, "application/octet-stream"),
        "run2_file": ("run2.h5ad", f2, "application/octet-stream")
    }
    data = {
        "run2_name": "leiden_resolution_1.0",
        "cluster_key": "leiden",
        "latent_key": "X_pca",
        "sync": "true",
        "project_id": "cai_lab_leiden_sweep"
    }
    
    response = requests.post(
        f"{API_URL}/api/v1/bio/evolution/report",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()

print(f"\nRun 1: {result['run1_name']}")
print(f"  Cells: {result['n_cells_run1']}")

print(f"\nRun 2: {result['run2_name']}")
print(f"  Cells: {result['n_cells_run2']}")

print(f"\nStability Score: {result['stability_score']:.1f}%")

if result['stability_score'] > 90:
    print("✓ Excellent stability - parameters are robust")
elif result['stability_score'] > 70:
    print("✓ Good stability - acceptable variation")
elif result['stability_score'] > 50:
    print("⚠️  Moderate drift - review parameters")
else:
    print("❌ High drift - investigate batch effects")

print(f"\nDelta Metrics:")
for key, value in result.get('delta_metrics', {}).items():
    print(f"  {key}: {value}")
```

---

## Troubleshooting

### Common Errors & Solutions

#### Error 1: Authentication Failed (401)

**Symptom:**
```json
{
  "detail": "Invalid or missing token"
}
```

**Causes & Solutions:**

✅ **Solution 1:** Verify token format
```python
# Correct format (includes "Bearer " prefix)
headers = {"Authorization": f"Bearer {TOKEN}"}

# WRONG - missing Bearer prefix
headers = {"Authorization": TOKEN}
```

✅ **Solution 2:** Check for extra whitespace
```python
# Trim token when loading
TOKEN = "tamu_cai_lab_2025_...".strip()
```

✅ **Solution 3:** Confirm token is active
- Contact support if token expired
- Tokens are valid for 1 year from issue date

#### Error 2: File Upload Failed (400)

**Symptom:**
```json
{
  "detail": "Expected .h5ad file"
}
```

**Causes & Solutions:**

✅ **Solution 1:** Verify file extension
```bash
# Check file extension
ls -lh *.h5ad
```

✅ **Solution 2:** Validate AnnData format
```python
import scanpy as sc

try:
    adata = sc.read_h5ad("your_file.h5ad")
    print(f"✓ Valid H5AD: {adata.n_obs} cells, {adata.n_vars} genes")
except Exception as e:
    print(f"❌ Invalid H5AD: {e}")
```

✅ **Solution 3:** Check file size
- Maximum: 500 MB per upload
- For larger files, preprocess locally to reduce size

#### Error 3: Rate Limit Exceeded (429)

**Symptom:**
```json
{
  "detail": "Rate limit exceeded for institution: Texas A&M"
}
```

**Causes & Solutions:**

✅ **Solution 1:** Implement rate limiting in code
```python
import time

def rate_limited_api_call(url, headers, **kwargs):
    max_retries = 3
    retry_delay = 60  # seconds
    
    for attempt in range(max_retries):
        response = requests.post(url, headers=headers, **kwargs)
        
        if response.status_code == 429:
            if attempt < max_retries - 1:
                print(f"Rate limit hit, waiting {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                raise Exception("Rate limit exceeded after retries")
        else:
            return response
    
    return response
```

✅ **Solution 2:** Batch requests efficiently
- Use `mode="cluster"` instead of `mode="cell"` (fewer API calls)
- Cache query vectors for reuse
- Combine searches where possible

✅ **Solution 3:** Monitor usage
- Academic tier: **1,000 requests/hour**
- Rate limit resets every hour (rolling window)
- Contact support for higher limits if needed

#### Error 4: Missing Required Parameter

**Symptom:**
```json
{
  "detail": "Field 'cluster_key' is required"
}
```

**Solution:** Check endpoint documentation for required parameters

```python
# Example: Export artifacts requires these parameters
data = {
    "cluster_key": "leiden",      # Required
    "latent_key": "X_umap",       # Required
    "sync": "true"                # Optional (defaults to false)
}
```

### Data Preparation Best Practices

#### Preprocessing Pipeline

```python
import scanpy as sc

# Load raw data
adata = sc.read_h5ad("raw_counts.h5ad")
print(f"Raw: {adata.n_obs} cells, {adata.n_vars} genes")

# Basic QC filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Mitochondrial filtering
adata = adata[adata.obs['pct_counts_mt'] < 5, :]
print(f"After QC: {adata.n_obs} cells")

# Normalization
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=2000)

# Dimensionality reduction
sc.tl.pca(adata, n_comps=50)
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.umap(adata)

# Clustering
sc.tl.leiden(adata, resolution=0.5)

# Save processed data
adata.write_h5ad("processed_for_api.h5ad")
print(f"✓ Saved processed data")
```

### Performance Optimization Tips

✅ **DO:**
- Use **cluster mode** by default (fewer vectors, faster search)
- **Cache query vectors** (encode once, search multiple times)
- **Preprocess locally** (filtering, normalization) before API upload
- **Use async mode** for large datasets (> 10K cells)
- **Monitor rate limits** (track API call count)

❌ **DON'T:**
- Upload **raw unfiltered data** (poor compression quality)
- Use **cell mode** for large datasets (generates too many vectors)
- Make **redundant API calls** (cache intermediate results)
- Upload files **> 500MB** (preprocess to reduce size)
- Test with **production data** in Swagger UI (use scripts)

---

## Support Resources

### Contact Information

| Contact Type | Email | Response Time |
|--------------|-------|---------------|
| **Research Liaison** | research@donsystems.com | < 24 hours |
| **Technical Support** | support@donsystems.com | < 24 hours |
| **Partnerships** | partnerships@donsystems.com | 2-3 business days |

### Office Hours

**Monday-Friday, 9:00 AM - 5:00 PM CST**

- Technical support available during business hours
- Emergency issues: Use "URGENT" in email subject

### Documentation Resources

| Resource | Location | Purpose |
|----------|----------|---------|
| **User Guide** | https://don-research.onrender.com | Complete documentation |
| **Swagger UI** | https://don-research.onrender.com/docs | Interactive API testing |
| **Token Handoff** | `TAMU_TOKEN_HANDOFF.md` (secure email) | Token details & first steps |
| **GitHub Examples** | (Coming soon) | Code samples & notebooks |

### Reporting Issues

When contacting support, please include:

1. ✅ **Institution name**: Texas A&M University (Cai Lab)
2. ✅ **API endpoint**: e.g., `POST /api/v1/genomics/vectors/build`
3. ✅ **Full error message**: Copy entire JSON response
4. ✅ **Sample data**: If < 10MB, attach file; otherwise, describe dataset
5. ✅ **Expected vs actual behavior**: What did you expect to happen?
6. ✅ **Trace ID**: Found in response header `X-Trace-ID` (for debugging)

**Example issue report:**

```
Subject: [TAMU] Error building vectors on PBMC dataset

Institution: Texas A&M University (Cai Lab)
Endpoint: POST /api/v1/genomics/vectors/build
Error: {"detail": "Expected .h5ad file"}

Dataset:
- File: pbmc_3k.h5ad (15 MB)
- Cells: 2,700
- Genes: 13,714

Expected: Should build 10 cluster vectors
Actual: Received 400 error

Trace ID: tamu_20250124_abc123

Attached: pbmc_3k.h5ad (Google Drive link)
```

### Academic Collaboration Opportunities

We welcome research collaborations with Texas A&M University:

- ✅ **Co-authored publications** on DON Stack applications
- ✅ **Grant proposal support** (NIH, NSF, DOD funding)
- ✅ **Student internships** at DON Systems LLC
- ✅ **Custom algorithm development** for specific research needs
- ✅ **Extended API access** for large-scale studies

**Contact:** partnerships@donsystems.com

---

## Appendix

### A. Common Cell Type Marker Genes

| Cell Type | Marker Genes |
|-----------|-------------|
| **T cells** | CD3E, CD8A, CD4, IL7R, CCR7 |
| **B cells** | MS4A1 (CD20), CD79A, CD19, IGHM |
| **NK cells** | NKG7, GNLY, KLRD1, NCAM1 |
| **Monocytes** | CD14, FCGR3A (CD16), CST3, LYZ |
| **Dendritic cells** | FCER1A, CST3, CLEC10A |
| **Neutrophils** | CSF3R, FCGR3B, S100A8, S100A9 |
| **Plasma cells** | IGHG1, MZB1, SDC1 (CD138) |
| **Megakaryocytes** | PPBP, PF4, GP9 |

### B. Vector Structure Reference

**128-Dimensional Feature Vector Breakdown:**

| Dimensions | Content | Purpose |
|------------|---------|---------|
| 0-15 | Entropy signature | Gene expression distribution (16 bins) |
| 16 | HVG fraction | % highly variable genes expressed |
| 17 | Mitochondrial % | Cell quality indicator |
| 18 | Total counts | Library size (normalized) |
| 22 | Silhouette score | Cluster separation quality (-1 to 1) |
| 27 | Purity score | Neighborhood homogeneity (0 to 1) |
| 28-127 | Biological tokens | Hashed cell type & tissue features |

### C. Compression Performance Metrics

**Validated on PBMC3k Dataset:**

| Metric | Value |
|--------|-------|
| **Input data** | 2,700 cells × 13,714 genes = 37,027,800 values |
| **Output vectors** | 10 clusters × 128 dimensions = 1,280 values |
| **Compression ratio** | 28,928× reduction |
| **Processing time** | < 30 seconds (standard hardware) |
| **Information retention** | 85-90% (biological signal preserved) |

### D. Distance Interpretation Guide

**Cosine Distance (Vector Search):**

| Distance Range | Interpretation | Example |
|----------------|----------------|---------|
| 0.0 - 0.2 | Very similar | Same cell type (e.g., CD8+ T cells) |
| 0.2 - 0.5 | Similar | Related types (e.g., T cells vs NK cells) |
| 0.5 - 0.8 | Moderate | Different lineages (e.g., T cells vs B cells) |
| 0.8 - 1.0 | Dissimilar | Unrelated types (e.g., T cells vs monocytes) |

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | January 2025 | Initial release for Texas A&M Cai Lab |

---

**Document Owner:** DON Systems LLC  
**Last Reviewed:** January 24, 2025  
**Next Review:** July 2025

---

## Quick Reference Card

**API Base URL:** `https://don-research.onrender.com`

**Authentication:**
```python
headers = {"Authorization": f"Bearer {YOUR_TOKEN}"}
```

**Rate Limit:** 1,000 requests/hour (Academic tier)

**Core Workflow:**
1. Build vectors: `POST /api/v1/genomics/vectors/build`
2. Encode query: `POST /api/v1/genomics/query/encode`
3. Search: `POST /api/v1/genomics/vectors/search`

**Support:** support@donsystems.com | Office Hours: M-F 9AM-5PM CST

---

*This document is proprietary to DON Systems LLC. External distribution is prohibited.*
