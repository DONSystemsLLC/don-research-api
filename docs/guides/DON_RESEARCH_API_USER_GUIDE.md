# Welcome to the DON Stack Research API
## Onboarding Guide for Dr. Cai's Lab at Texas A&M University

**Welcome!** This guide will help you get started with the DON Stack Research API in under 15 minutes.

---

## 🎯 What You'll Accomplish Today

By the end of this onboarding guide, you will:

1. ✅ **Understand** what the DON Stack API can do for your research
2. ✅ **Access** the web interface and authenticate successfully
3. ✅ **Complete** your first API request (no coding required!)
4. ✅ **Know** where to get help and what to do next

**Estimated Time:** 10-15 minutes

---

## 👋 Welcome to DON Stack

### What is this system?

The **DON Stack Research API** is a **quantum-enhanced data processing platform** built specifically for advanced research at Texas A&M. Think of it as a powerful computational service that can:

- **Compress** massive datasets (genomics, images, sensor data) by 4×-32× while preserving 95%+ accuracy
- **Detect** quality issues in your data automatically
- **Stabilize** machine learning models using quantum error correction
- **Compare** analysis runs to ensure reproducibility

### Why should you care?

Traditional data compression loses too much information. Traditional QC is manual and time-consuming. Traditional ML models are noisy and unstable.

**DON Stack solves these problems** using quantum-inspired algorithms that:
- Compress better than PCA, t-SNE, or UMAP
- Detect contamination that visual inspection misses
- Stabilize embeddings for downstream analysis

### Who is this for?

This system is designed for:
- 🧬 **Genomics researchers** working with single-cell RNA-seq data
- 🤖 **Data scientists** needing advanced dimensionality reduction
- 🔬 **Computational biologists** validating analysis pipelines
- 📊 **Anyone** working with high-dimensional numerical data

**No quantum computing expertise required!** The system handles all the complexity internally.

---

## 🚀 Your First 5 Minutes: Access the System

Let's get you connected to the DON Stack API right now.

### Step 1: Get Your Credentials (Already Done!)

Your lab has been provisioned with:
- ✅ **API Token** - A secure access key (provided separately via email)
- ✅ **Rate Limit** - 1,000 requests per hour (academic tier)
- ✅ **Full Access** - All endpoints and features enabled

**Action:** Locate your token email from DON Systems (subject: "Texas A&M Research API Access")

### Step 2: Open the Web Interface

**Action:** Click this link → **<https://don-research.onrender.com/docs>**

You should see an interactive interface called **Swagger UI** with a list of API endpoints.

### Step 3: Authenticate (30 seconds)

**Action:** Follow these steps exactly:

1. Look for the **green "Authorize" button** in the top-right corner
2. Click it - a popup appears
3. **Paste your token** in the "Value" field (the one from your email)
4. Click **"Authorize"**
5. Click **"Close"**

✅ **Success indicator:** The padlock icon 🔓 changes to 🔒 (locked)

**Troubleshooting:**
- ❌ If you don't see the Authorize button, refresh the page
- ❌ If authentication fails, check for extra spaces in your token
- ❌ If you lost your token, email: support@donsystems.com

---

## 🎉 Your First Success: Health Check

Let's verify the system is working by making your first API call!

### What You'll Do

You're going to ask the DON Stack API: **"Are you alive and ready?"**

This is the simplest possible API request - it requires no data, no parameters, just authentication.

### Step-by-Step

1. **Scroll down** in the Swagger UI until you find:
   ```
   GET /api/v1/health
   ```

2. **Click** on it to expand the details

3. **Click** the blue "Try it out" button

4. **Click** the black "Execute" button

5. **Scroll down** to see the response

### What Success Looks Like

You should see a **Response Code: 200** and a green success message.

The response body will look like this:

```json
{
  "status": "healthy",
  "don_stack": {
    "mode": "production",
    "don_gpu": true,
    "tace": true,
    "qac": true,
    "adapter_loaded": true
  },
  "timestamp": "2025-10-27T..."
}
```

### What This Means

- ✅ **"status": "healthy"** - The system is operational
- ✅ **"don_gpu": true** - Real DON-GPU compression engine is running (not a fallback)
- ✅ **"qac": true** - Quantum error correction is available
- ✅ **"tace": true** - Temporal analysis engine is ready

**🎊 Congratulations!** You've successfully authenticated and made your first API call.

---

## 📚 What the System Can Do

Now that you're connected, let's explore what you can actually use this for.

### 1. **Compress Genomics Data** (Most Popular)

**What it does:** Takes a standard `.h5ad` single-cell file and compresses it from millions of values down to thousands - while keeping 95%+ biological information.

**Example:**
- **Input:** 10,000 cells × 20,000 genes = 200 million values
- **Output:** 10,000 cells × 32 dimensions = 320,000 values
- **Compression:** 625× smaller
- **Information retained:** 95%+

**Why you'd use this:**
- Faster downstream analysis
- Easier data sharing and storage
- Better visualization in low dimensions
- Pre-processing for machine learning

**Endpoint:** `POST /api/v1/genomics/compress`

### 2. **Detect Quality Issues** (QC Automation)

**What it does:** Automatically identifies three types of contamination in single-cell data that are hard to spot manually:

1. **Ambient RNA** - Background contamination from lysed cells
2. **Doublets** - Two cells accidentally captured as one
3. **Batch Effects** - Technical variation between samples

**Example output:**
- Parasite score: 8.3% (120 out of 1,445 cells flagged)
- Ambient RNA: Medium severity (75 cells affected)
- Doublets: 45 detected (above expected rate)
- Batch effects: 3 clusters show poor mixing

**Why you'd use this:**
- Automate QC that normally takes hours
- Catch issues that visual inspection misses
- Document data quality for publications
- Filter low-quality cells before analysis

**Endpoint:** `POST /api/v1/bio/qc/parasite-detect`

### 3. **Check Reproducibility** (Pipeline Validation)

**What it does:** Compares two analysis runs to see if your pipeline is stable or if results are drifting.

**Example:**
- Compare: Same data, different random seeds
- Result: 89.3% stability score (excellent!)
- Interpretation: Your clustering is reproducible

**Why you'd use this:**
- Validate analysis before publication
- Test parameter changes safely
- Ensure batch-to-batch consistency
- Meet reproducibility standards

**Endpoint:** `POST /api/v1/bio/evolution/report`

### 4. **Universal Data Compression** (Beyond Genomics)

**What it does:** Compresses ANY numerical data - not just genomics!

**Examples:**
- Image embeddings (CNN features: 2048 dims → 64 dims)
- Text embeddings (BERT: 768 dims → 128 dims)
- Sensor data (IoT: 500 channels → 32 channels)
- Graph embeddings (node2vec: 256 dims → 64 dims)

**Languages supported:** Python, R, Julia, MATLAB, JavaScript, Go, Rust, C++

**Endpoint:** `POST /api/v1/genomics/compress` (accepts raw JSON)

### 5. **Quantum Error Correction** (Advanced)

**What it does:** Applies quantum-inspired error correction to stabilize embeddings and reduce noise.

**Example:**
- Train QAC model on 5,000 BERT text embeddings
- Apply to new embeddings to reduce noise
- Result: 12% improvement in downstream classification

**Why you'd use this:**
- Stabilize noisy embeddings
- Improve clustering quality
- Reduce overfitting in ML models
- Research applications in quantum ML

**Endpoints:** `POST /api/v1/quantum/qac/fit` + `POST /api/v1/quantum/qac/apply`

---

## 🎓 Learn By Doing: Your First Real Task

Now let's do something more interesting - let's check your current API usage!

### Task: See How Many Requests You've Used

This will show you how to monitor your 1,000 requests/hour rate limit.

**Steps:**

1. In the Swagger UI, find: `GET /api/v1/usage`
2. Click "Try it out"
3. Click "Execute"
4. View your usage statistics

**You'll see:**
```json
{
  "institution": "Texas A&M University",
  "requests_used": 2,
  "rate_limit": 1000,
  "reset_time": 1698768000
}
```

This tells you:
- You've used **2 requests** (the health check + this usage check)
- You have **998 requests remaining** this hour
- Rate limit resets at the `reset_time` timestamp

---

## 📖 Where to Go From Here

Congratulations! You've completed the onboarding. Here's what to do next:

### For Beginners (No Coding Yet)

**Next Step:** Try uploading a file via the web interface

1. Find an example `.h5ad` file (or download PBMC3k from 10X Genomics)
2. Navigate to `POST /api/v1/bio/export-artifacts` in Swagger UI
3. Click "Try it out"
4. Upload your file
5. Set `cluster_key` to `"leiden"` and `latent_key` to `"X_pca"`
6. Set `sync` to `true`
7. Click "Execute" and watch the magic happen!

**Read next:** [Complete Web Interface Tutorial](#web-interface-tutorial-complete-walkthrough)

### For Python Users (Ready to Automate)

**Next Step:** Run your first Python script

```python
import requests
import os

TOKEN = os.getenv("DON_API_TOKEN")  # Set this in your environment
headers = {"Authorization": f"Bearer {TOKEN}"}

# Check health
response = requests.get(
    "https://don-research-api.onrender.com/api/v1/health",
    headers=headers
)

print(response.json())
```

**Read next:** [Programmatic API Tutorial](#using-the-programmatic-api)

### For R Users (Seurat Integration)

**Next Step:** Integrate with your Seurat workflows

```r
library(httr)
library(jsonlite)

TOKEN <- Sys.getenv("DON_API_TOKEN")
headers <- add_headers(Authorization = paste("Bearer", TOKEN))

# Check health
response <- GET(
  "https://don-research-api.onrender.com/api/v1/health",
  headers
)

print(content(response))
```

**Read next:** [R Integration Examples](#r-with-seurat)

### For Advanced Users (Custom Pipelines)

**Next Step:** Explore the full API capabilities

- Review all available endpoints at: <https://don-research.onrender.com/docs>
- Read about [QAC Quantum Error Correction](#qac-quantum-error-correction)
- Learn about [Universal Data Compression](#core-compression-endpoints)
- Check out [Language-Agnostic Integration](#data-formats--language-support)

---

## 🆘 Getting Help

### Documentation Resources

| Resource | URL | Best For |
|----------|-----|----------|
| **This Onboarding Guide** | You're reading it! | Getting started (first 15 min) |
| **Interactive API Docs** | <https://don-research.onrender.com/docs> | Testing endpoints, seeing examples |
| **HTML User Guide** | <https://don-research.onrender.com/help> | Quick reference, troubleshooting |
| **Complete Guide** | Scroll down in this document | Deep dives, workflows, best practices |

### Common Questions

**Q: How do I get my token?**
A: Check your email for "Texas A&M Research API Access" from DON Systems. If you can't find it, email support@donsystems.com

**Q: What if I hit the rate limit?**
A: You have 1,000 requests/hour. The limit resets every hour. Check `/api/v1/usage` to monitor. If you need more for large studies, contact us.

**Q: Can I use this with R/Julia/MATLAB?**
A: Yes! The API is language-agnostic. Any language that can make HTTP requests will work. See [Language-Specific Integration](#language-specific-integration)

**Q: Do I need to know quantum computing?**
A: No! The system handles all quantum complexity internally. You just send data and get results.

**Q: How do I cite this in publications?**
A: See the [Citing DON Stack](#citing-don-stack-in-publications) section below.

**Q: What data formats are supported?**
A: H5AD files (genomics), raw JSON arrays (any numerical data), GEO accessions, direct URLs. See [Data Formats](#data-formats--language-support)

### Support Contacts

**Technical Issues:**
- Email: support@donsystems.com
- Response time: <24 hours
- Include: Your institution (Texas A&M), endpoint used, error message, trace_id if available

**Research Collaboration:**
- Contact: Dr. James J. Cai (jcai@tamu.edu)
- For: Methodology questions, research design, collaboration opportunities

**Token Issues:**
- Email: donnievanmetre@gmail.com
- For: Lost tokens, access issues, rate limit increases

---

## ✅ Onboarding Checklist

Before moving to the advanced sections below, make sure you've completed:

- [ ] Received your API token via email
- [ ] Opened <https://don-research.onrender.com/docs>
- [ ] Successfully authenticated (green lock icon)
- [ ] Ran your first health check (`GET /api/v1/health`)
- [ ] Checked your usage (`GET /api/v1/usage`)
- [ ] Understand the 5 main capabilities of the system
- [ ] Know where to get help (emails above)

**If you've checked all boxes, you're ready to dive deeper!** The sections below provide comprehensive documentation for advanced usage.

---

# Advanced Documentation

The sections below provide comprehensive technical documentation for users ready to integrate DON Stack into production workflows.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Getting Started](#getting-started)
   - [Using the Web Interface](#using-the-web-interface) (No coding required!)
   - [Using the Programmatic API](#using-the-programmatic-api) (For automation)
3. [Data Formats & Language Support](#data-formats--language-support)
4. [Complete API Reference](#complete-api-reference)
   - [Core Compression Endpoints](#core-compression-endpoints)
   - [Bio Module Endpoints](#bio-module-endpoints)
   - [QAC Quantum Error Correction](#qac-quantum-error-correction)
5. [Research Workflows](#research-workflows)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)
8. [Support and Collaboration](#support-and-collaboration)

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Dr. Cai's Lab (Texas A&M)                              │
│  ┌──────────────────────────────────────────┐           │
│  │ Your Python/R Scripts                    │           │
│  │ - Scanpy preprocessing                   │           │
│  │ - HTTP requests to DON API               │           │
│  │ - Result visualization & analysis        │           │
│  └──────────────────┬───────────────────────┘           │
└────────────────────┼────────────────────────────────────┘
                     │ HTTPS + Bearer Token
                     │
┌────────────────────▼────────────────────────────────────┐
│  DON Stack Research API (Render.com)                    │
│  ┌──────────────────────────────────────────┐           │
│  │ FastAPI Service Layer                    │           │
│  │ - Authentication & rate limiting         │           │
│  │ - Request validation                     │           │
│  │ - Response formatting                    │           │
│  │ - Trace logging (reproducibility)        │           │
│  └──────────────────┬───────────────────────┘           │
│                     │                                    │
│  ┌──────────────────▼───────────────────────┐           │
│  │ Protected DON Stack Engines (Black Box)  │           │
│  │ - DON-GPU fractal clustering             │           │
│  │ - QAC quantum error correction           │           │
│  │ - TACE temporal field analysis           │           │
│  │ - Proprietary optimization algorithms    │           │
│  └──────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
```

### Key Components

**1. FastAPI Service Layer (Your Interface)**
- RESTful HTTP endpoints for all operations
- Accepts standard h5ad files (AnnData format)
- Returns JSON results with metrics and artifact paths
- Handles both synchronous (immediate) and asynchronous (background job) execution
- Logs all operations to SQLite database for reproducibility

**2. Multi-Backend Logging System**
- **JSONL** - Machine-readable event stream
- **Markdown** - Human-readable operation log
- **SQLite** - Queryable trace database with lineage graphs
- Every API call generates a trace with metrics, artifacts, and system health

**3. Authentication & Rate Limiting**
- Institution-based Bearer tokens
- Configurable rate limits (1,000 req/hour for Texas A&M)
- Token managed via environment variables (no hardcoded credentials)

**4. DON Stack Engines (Protected)**
- Proprietary algorithms remain server-side
- No direct access or reverse engineering possible
- Accessed only through validated API endpoints

---

## Getting Started

### Two Ways to Use the System

The DON Stack Research API can be accessed in two ways:

#### **Option 1: Interactive Web Interface (Recommended for Beginners)**
- **URL:** https://don-research.onrender.com/
- **Best for:** Uploading files, testing endpoints, exploring examples
- **Features:** Interactive Swagger UI with "Try it out" buttons
- **No coding required** for basic operations

#### **Option 2: Programmatic API (Recommended for Automation)**
- **URL:** https://don-research-api.onrender.com (programmatic access)
- **Best for:** Batch processing, pipeline integration, reproducible workflows
- **Languages:** Python, R, Julia, MATLAB, cURL, any HTTP client

### Using the Web Interface

**Step 1: Navigate to the Interactive Documentation**

Visit: **https://don-research.onrender.com/docs**

This opens the **Swagger UI** - an interactive API explorer where you can:
- Browse all available endpoints
- See request/response examples
- Test API calls directly in your browser
- Download example code snippets

**Step 2: Authenticate**

1. Click the **"Authorize"** button (lock icon) at the top right
2. Enter your Bearer token: `your_token_here`
3. Click "Authorize" and then "Close"

You're now authenticated for all subsequent requests!

**Step 3: Try Your First Request**

1. Navigate to **GET /api/v1/health** endpoint
2. Click **"Try it out"**
3. Click **"Execute"**
4. View the response below showing system status

**Step 4: Upload a File (Example: Build Feature Vectors)**

1. Navigate to **POST /api/v1/genomics/vectors/build**
2. Click **"Try it out"**
3. Click **"Choose File"** and select your `.h5ad` file
4. Select mode: `cluster` or `cell`
5. Click **"Execute"**
6. Download the resulting JSONL file from the response

**Step 5: View Full User Guide**

Visit: **https://don-research.onrender.com/help**

This shows the complete HTML user guide with:
- Detailed endpoint documentation
- Code examples in Python
- Troubleshooting tips
- Contact information

---

### Using the Programmatic API

For automation and reproducible workflows, access the API programmatically.

#### Prerequisites

**Required Software:**
- Python 3.9+ or R 4.0+ (or any language with HTTP support)
- `requests` library (Python) or `httr` (R)
- `scanpy` (for h5ad file handling in Python)
- `anndata` (Python AnnData support)

**Install Python Dependencies:**
```bash
pip install requests scanpy anndata pandas numpy
```

#### Authentication Setup

You will receive a **Bearer token** from DON Systems for Texas A&M access. Store this securely:

**Option 1: Environment Variable (Recommended)**
```bash
export DON_API_TOKEN="your_token_here"
export DON_API_BASE_URL="https://don-research-api.onrender.com"
```

**Option 2: Configuration File**
```python
# config.py
DON_API_TOKEN = "your_token_here"
DON_API_BASE_URL = "https://don-research-api.onrender.com"
```

**IMPORTANT SECURITY NOTES:**
- Never commit tokens to Git repositories
- Do not share tokens outside authorized lab members
- Rotate tokens if compromised
- Use environment variables or encrypted configuration files

#### Your First Programmatic API Call

**Example: Check API Health**

```python
import requests
import os

# Load credentials
API_TOKEN = os.getenv("DON_API_TOKEN")
API_BASE = os.getenv("DON_API_BASE_URL")

# Set up headers
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Check system health
response = requests.get(f"{API_BASE}/api/v1/health", headers=headers)
print(response.json())
```

**Expected Response:**
```json
{
  "status": "healthy",
  "don_stack": {
    "mode": "production",
    "don_gpu": true,
    "tace": true,
    "qac": true,
    "adapter_loaded": true
  },
  "timestamp": "2025-10-27T10:30:00Z"
}
```

**Interpreting the Response:**
- `status: "healthy"` - API is operational
- `don_gpu: true` - Fractal compression engine available
- `tace: true` - Temporal field analysis engine ready
- `qac: true` - Quantum coherence checking enabled
- `adapter_loaded: true` - H5AD conversion system ready

---

### Quick Start Comparison

| Feature | Web Interface | Programmatic API |
|---------|---------------|------------------|
| **URL** | https://don-research.onrender.com/docs | https://don-research-api.onrender.com |
| **Best For** | Exploration, testing, demos | Automation, batch jobs, pipelines |
| **Coding Required** | No | Yes |
| **Authentication** | Click "Authorize" button | HTTP headers in code |
| **File Upload** | Drag & drop interface | Multipart form data |
| **Results** | View in browser | Parse JSON response |
| **Reproducibility** | Manual | Fully scriptable |
| **Learning Curve** | Easy | Moderate |

**Recommendation:** Start with the **web interface** to explore endpoints and test with sample data, then switch to **programmatic access** for production workflows

---

### Web Interface Tutorial: Complete Walkthrough

This section provides a **step-by-step visual guide** to using the web interface at <https://don-research.onrender.com/>.

#### Tutorial 1: Compress Genomics Data (No Coding)

**Goal:** Upload a PBMC h5ad file and compress it to 32 dimensions

**Step-by-Step:**

1. **Open your browser** and navigate to: <https://don-research.onrender.com/docs>

2. **Authenticate:**
   - Look for the green **"Authorize"** button in the top right
   - Click it to open the authentication dialog
   - Paste your Texas A&M token in the "Value" field
   - Click **"Authorize"**, then **"Close"**
   - You'll see a padlock icon change to show you're authenticated

3. **Find the compression endpoint:**
   - Scroll down to **`POST /api/v1/genomics/compress`**
   - Click to expand the endpoint details
   - Click the blue **"Try it out"** button in the top right

4. **Enter your data:**
   - You'll see a JSON editor with example data
   - Replace the example with your own data structure:
     ```json
     {
       "data": {
         "expression_matrix": [[...your data...]],
         "gene_names": ["GENE1", "GENE2", ...]
       },
       "compression_target": 32,
       "seed": 42
     }
     ```
   - **OR** if you have a Python script that already loaded your h5ad:
     ```python
     import scanpy as sc
     import json

     adata = sc.read_h5ad("pbmc3k.h5ad")
     payload = {
         "data": {
             "expression_matrix": adata.X.toarray().tolist(),
             "gene_names": adata.var_names.tolist()
         },
         "compression_target": 32
     }

     # Copy this JSON to the web interface
     print(json.dumps(payload))
     ```

5. **Execute the request:**
   - Click the black **"Execute"** button
   - Wait for processing (typically 5-30 seconds)
   - Scroll down to see the **Response**

6. **Interpret the results:**
   - **Status code:** Should be **200** (success)
   - **Response body:** JSON with compressed data
   - Look for:
     - `"compression_ratio": "32.0×"` - How much compression achieved
     - `"compressed_data": [[...]]` - Your compressed matrix
     - `"algorithm": "DON-GPU Fractal Clustering (REAL)"` - Confirms real DON Stack used
     - `"trace_id": "abc123..."` - Save this for reproducibility

7. **Download the results:**
   - Click **"Download"** button next to the response
   - Save as `compressed_output.json`
   - Load in Python:
     ```python
     import json
     import numpy as np

     with open("compressed_output.json") as f:
         result = json.load(f)

     compressed = np.array(result["compressed_data"])
     print(f"Compressed shape: {compressed.shape}")
     ```

#### Tutorial 2: QC Parasite Detection via Web Interface

**Goal:** Upload an h5ad file and detect contamination without writing code

**Step-by-Step:**

1. **Navigate to:** <https://don-research.onrender.com/docs>

2. **Authenticate** (if not already authenticated from Tutorial 1)

3. **Find the QC endpoint:**
   - Scroll to **`POST /api/v1/bio/qc/parasite-detect`**
   - Click to expand
   - Click **"Try it out"**

4. **Upload your file:**
   - Click **"Choose File"** under the `file` parameter
   - Select your `.h5ad` file from your computer
   - The file will upload automatically when you execute

5. **Set parameters:**
   - `cluster_key`: Enter `"leiden"` (or your cluster column name)
   - `batch_key`: Enter `"sample"` (or your batch column name)
   - `ambient_threshold`: Leave default `0.15` or adjust
   - `doublet_threshold`: Leave default `0.25` or adjust
   - `sync`: Select `true` for immediate results

6. **Execute:**
   - Click **"Execute"**
   - Wait for analysis (typically 10-60 seconds)

7. **Interpret QC results:**
   - Look for `"parasite_score"`: Percentage of contaminated cells
     - **< 5%:** Excellent quality - proceed
     - **5-15%:** Good quality - minor filtering recommended
     - **15-30%:** Moderate contamination - filter flagged cells
     - **> 30%:** High contamination - review QC pipeline

   - Check specific contamination types:
     - `"ambient": {"severity": "medium"}` - Ambient RNA contamination level
     - `"doublets": {"n_flagged": 120}` - Number of doublets detected
     - `"batch": {"poor_mixing_clusters": [2, 5, 8]}` - Clusters with batch effects

8. **Save the report:**
   - Download the JSON response
   - Share the `trace_id` with collaborators for reproducibility

#### Tutorial 3: Train a QAC Model on Custom Embeddings

**Goal:** Apply quantum error correction to any numerical embeddings (e.g., image features, text embeddings)

**Step-by-Step:**

1. **Prepare your embeddings:**
   ```python
   # Example: BERT text embeddings
   import numpy as np
   import json

   # Your embeddings (e.g., from transformers library)
   embeddings = np.random.randn(5000, 768)  # 5k samples × 768 dims

   payload = {
       "embedding": embeddings.tolist(),
       "params": {
           "k_nn": 20,
           "layers": 100,
           "engine": "real_qac"
       },
       "sync": True,
       "seed": 42
   }

   print(json.dumps(payload))  # Copy this
   ```

2. **Navigate to:** <https://don-research.onrender.com/docs>

3. **Find QAC endpoint:**
   - Scroll to **`POST /api/v1/quantum/qac/fit`**
   - Click **"Try it out"**

4. **Paste your payload:**
   - Replace the example JSON with your copied payload
   - Click **"Execute"**

5. **Wait for training:**
   - This may take 1-5 minutes for large embeddings
   - Watch the status indicator

6. **Save the model ID:**
   - Response will include: `"model_id": "qac_abc123"`
   - **Copy this ID** - you'll need it to apply the model

7. **Apply the trained model:**
   - Scroll to **`POST /api/v1/quantum/qac/apply`**
   - Click **"Try it out"**
   - Enter:
     ```json
     {
       "model_id": "qac_abc123",
       "embedding": [[...new embeddings...]],
       "sync": true
     }
     ```
   - Click **"Execute"**

8. **Retrieve corrected embeddings:**
   - Response includes `"corrected_embedding": [[...]]`
   - Use these error-corrected embeddings in downstream tasks

---

### Common Web Interface Tasks

**Task:** Check if the system is operational

- Navigate to: <https://don-research.onrender.com/docs>
- Find `GET /api/v1/health`
- Click "Try it out" → "Execute"
- Confirm `"status": "healthy"`

**Task:** View your API usage

- Navigate to: <https://don-research.onrender.com/docs>
- Find `GET /api/v1/usage`
- Click "Try it out" → "Execute"
- See: `"requests_used": 47 / 1000`

**Task:** Get help documentation

- Navigate to: <https://don-research.onrender.com/help>
- Browse the complete HTML user guide
- Copy code examples directly from the page

**Task:** Download API specification

- Navigate to: <https://don-research.onrender.com/docs>
- Scroll to top and click **"Download"** button
- Save `openapi.json` for offline reference or code generation

---

## Data Formats & Language Support

### Supported Data Formats

The DON Stack Research API is **completely language-agnostic** and accepts data in multiple formats:

#### **1. H5AD Files (Genomics - Python/R)**
- Standard AnnData format for single-cell RNA-seq
- Used by: Scanpy (Python), Seurat v5 (R)
- File extension: `.h5ad`
- Best for: Pre-processed genomics data with cluster annotations

#### **2. Raw JSON (Any Language)**
- Direct HTTP POST with JSON payload
- Language support: Python, R, Julia, MATLAB, JavaScript, C++, Rust, Go, etc.
- Format:
  ```json
  {
    "data": {
      "expression_matrix": [[1.2, 3.4, ...], [5.6, 7.8, ...]],
      "gene_names": ["GENE1", "GENE2", ...],
      "cell_metadata": {"optional": "metadata"}
    },
    "compression_target": 32,
    "seed": 42
  }
  ```

#### **3. QAC Embeddings (Pure Numerical)**
- Language-agnostic embedding arrays
- No biological assumptions - works on any numerical data
- Format:
  ```json
  {
    "embedding": [[0.1, 0.2, ...], [0.3, 0.4, ...]],
    "params": {"k_nn": 15, "layers": 50}
  }
  ```

### Language-Specific Integration

#### **Python**
```python
import requests
import numpy as np

# Example: Compress any numerical matrix
data_matrix = np.random.randn(1000, 500)  # 1000 samples × 500 features

payload = {
    "data": {
        "expression_matrix": data_matrix.tolist(),
        "gene_names": [f"feature_{i}" for i in range(500)]
    },
    "compression_target": 32
}

response = requests.post(
    "https://don-research-api.onrender.com/api/v1/genomics/compress",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json=payload
)

compressed = np.array(response.json()["compressed_data"])
print(f"Compressed from {data_matrix.shape} to {compressed.shape}")
```

#### **R**
```r
library(httr)
library(jsonlite)

# Example: Compress Seurat object embeddings
data_matrix <- matrix(rnorm(1000 * 500), nrow = 1000, ncol = 500)

payload <- list(
  data = list(
    expression_matrix = data_matrix,
    gene_names = paste0("feature_", 1:500)
  ),
  compression_target = 32
)

response <- POST(
  "https://don-research-api.onrender.com/api/v1/genomics/compress",
  add_headers(Authorization = paste("Bearer", TOKEN)),
  body = toJSON(payload, auto_unbox = TRUE),
  content_type_json()
)

result <- content(response)
compressed <- do.call(rbind, result$compressed_data)
cat("Compressed from", dim(data_matrix), "to", dim(compressed), "\n")
```

#### **Julia**
```julia
using HTTP, JSON

# Example: Compress numerical data
data_matrix = randn(1000, 500)

payload = Dict(
    "data" => Dict(
        "expression_matrix" => [data_matrix[i, :] for i in 1:size(data_matrix, 1)],
        "gene_names" => ["feature_$i" for i in 1:500]
    ),
    "compression_target" => 32
)

response = HTTP.post(
    "https://don-research-api.onrender.com/api/v1/genomics/compress",
    ["Authorization" => "Bearer $TOKEN"],
    JSON.json(payload)
)

result = JSON.parse(String(response.body))
compressed = hcat(result["compressed_data"]...)
println("Compressed from $(size(data_matrix)) to $(size(compressed))")
```

#### **MATLAB**
```matlab
% Example: Compress numerical data
data_matrix = randn(1000, 500);

payload = struct(...
    'data', struct(...
        'expression_matrix', num2cell(data_matrix, 2)', ...
        'gene_names', arrayfun(@(i) sprintf('feature_%d', i), 1:500, 'UniformOutput', false) ...
    ), ...
    'compression_target', 32 ...
);

options = weboptions(...
    'HeaderFields', {'Authorization', ['Bearer ' TOKEN]}, ...
    'MediaType', 'application/json' ...
);

response = webwrite(...
    'https://don-research-api.onrender.com/api/v1/genomics/compress', ...
    payload, ...
    options ...
);

compressed = cell2mat(response.compressed_data');
fprintf('Compressed from [%d, %d] to [%d, %d]\n', ...
    size(data_matrix), size(compressed));
```

### What Data Can You Process?

The system is **not limited to genomics**. You can compress and analyze:

- **Single-cell RNA-seq** (PBMC, tumor samples, organoids)
- **Spatial transcriptomics** (Visium, MERFISH, seqFISH)
- **Image features** (embeddings from CNNs, autoencoders)
- **Sensor data** (IoT, time-series, multivariate signals)
- **Text embeddings** (word2vec, BERT, GPT)
- **Graph embeddings** (node2vec, GraphSAGE)
- **Any high-dimensional numerical data**

**Key Requirements:**
- Data must be **numerical** (floats/integers)
- Rows = samples/observations
- Columns = features/dimensions
- No NaN/Inf values (preprocessing required)

---

## Complete API Reference

The API is organized into three main modules:

1. **Core Compression** - Universal data compression (language-agnostic)
2. **Bio Module** - Genomics-specific workflows (h5ad files)
3. **QAC Module** - Quantum error correction (pure embeddings)

### Core Compression Endpoints

These endpoints accept **raw JSON from any language** and process **any numerical data**.

#### Endpoint: Universal Data Compression

**Compress any numerical matrix using DON-GPU fractal clustering.**

```
POST /api/v1/genomics/compress
```

**Request Parameters (JSON body):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `data.expression_matrix` | List[List[float]] | Yes | Numerical data matrix (samples × features) |
| `data.gene_names` | List[str] | No | Feature labels (optional, can be generic) |
| `data.cell_metadata` | Dict | No | Optional metadata for samples |
| `compression_target` | int | No | Target compressed dimensions (default: 32) |
| `params.mode` | str | No | `"auto_evr"` (adaptive) or `"fixed_k"` (exact dimensions) |
| `params.evr_target` | float | No | Explained variance ratio target (default: 0.95) |
| `params.max_k` | int | No | Maximum allowed dimensions (default: 64) |
| `seed` | int | No | Random seed for reproducibility |
| `stabilize` | bool | No | Apply quantum stabilization (default: false) |
| `project_id` | str | No | Project identifier for grouping |
| `user_id` | str | No | User identifier for audit logging |

**Python Example (Non-Genomics Data):**

```python
import requests
import numpy as np

# Example: Compress image features from a CNN
image_features = np.random.randn(10000, 2048)  # 10k images, 2048-dim features

payload = {
    "data": {
        "expression_matrix": image_features.tolist(),
        "gene_names": [f"cnn_feature_{i}" for i in range(2048)]
    },
    "compression_target": 64,
    "params": {
        "mode": "fixed_k",  # Exact 64 dimensions
        "max_k": 128
    },
    "seed": 42,
    "project_id": "image_compression_study"
}

response = requests.post(
    f"{API_BASE}/api/v1/genomics/compress",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json=payload
)

result = response.json()
compressed = np.array(result["compressed_data"])

print(f"Original: {image_features.shape}")
print(f"Compressed: {compressed.shape}")
print(f"Compression ratio: {result['compression_stats']['compression_ratio']}")
print(f"Algorithm: {result['algorithm']}")
```

**Response:**
```json
{
  "compressed_data": [[...], [...], ...],
  "gene_names": ["cnn_feature_0", "cnn_feature_1", ...],
  "metadata": null,
  "compression_stats": {
    "original_dimensions": 2048,
    "compressed_dimensions": 64,
    "requested_k": 64,
    "achieved_k": 64,
    "rank": 2048,
    "compression_ratio": "32.0×",
    "cells_processed": 10000,
    "evr_target": 0.95,
    "mode": "fixed_k",
    "max_k": 128
  },
  "algorithm": "DON-GPU Fractal Clustering (REAL)",
  "institution": "Texas A&M University",
  "runtime_ms": 1243,
  "seed": 42,
  "stabilize": false,
  "engine_used": "real_don_gpu",
  "trace_id": "abc123..."
}
```

**Use Cases:**
- **Dimensionality reduction** for visualization (compress to 2-3 dims)
- **Feature extraction** from high-dimensional data
- **Data preprocessing** before ML training
- **Storage optimization** for large datasets
- **Transfer learning** (compress pretrained embeddings)

---

### QAC Quantum Error Correction

**Train and apply quantum error correction models on any embeddings.**

The QAC (Quantum Adjacency Code) module accepts **pure numerical embeddings** from any source - it doesn't care if your data came from genomics, images, text, or sensor data.

#### Endpoint: QAC Fit (Train Model)

```
POST /api/v1/quantum/qac/fit
```

**Request Parameters (JSON body):**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `embedding` | List[List[float]] | Yes | Numerical embeddings (n_samples × n_dims) |
| `params.k_nn` | int | No | k-nearest neighbors for graph (default: 15) |
| `params.weight` | str | No | Edge weights: `"binary"` or `"gaussian"` (default: `"binary"`) |
| `params.layers` | int | No | QAC correction layers (default: 50, max: 100000) |
| `params.reinforce_rate` | float | No | Reinforcement rate (default: 0.05, range: 0-1) |
| `params.engine` | str | No | `"real_qac"` (quantum) or `"laplace"` (classical fallback) |
| `seed` | int | No | Random seed for reproducibility |
| `sync` | bool | No | Synchronous execution (default: false) |

**Python Example:**

```python
import numpy as np

# Train QAC on any embeddings (e.g., text embeddings from BERT)
text_embeddings = np.random.randn(5000, 768)  # 5k sentences, 768-dim BERT

payload = {
    "embedding": text_embeddings.tolist(),
    "params": {
        "k_nn": 20,
        "layers": 100,
        "engine": "real_qac"
    },
    "sync": True,
    "seed": 42
}

response = requests.post(
    f"{API_BASE}/api/v1/quantum/qac/fit",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json=payload
)

result = response.json()
model_id = result["model_id"]
print(f"QAC model trained: {model_id}")
```

**Response:**
```json
{
  "status": "succeeded",
  "model_id": "qac_abc123",
  "meta": {
    "n_cells": 5000,
    "k_nn": 20,
    "layers": 100,
    "engine": "real_qac"
  }
}
```

#### Endpoint: QAC Apply (Use Trained Model)

```
POST /api/v1/quantum/qac/apply
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model_id` | str | Yes | QAC model ID from `/qac/fit` |
| `embedding` | List[List[float]] | Yes | New embeddings to correct (must match training dimensions) |
| `seed` | int | No | Random seed |
| `sync` | bool | No | Synchronous execution (default: false) |

**Python Example:**

```python
# Apply QAC error correction to new embeddings
new_embeddings = np.random.randn(1000, 768)

payload = {
    "model_id": model_id,
    "embedding": new_embeddings.tolist(),
    "sync": True
}

response = requests.post(
    f"{API_BASE}/api/v1/quantum/qac/apply",
    headers={"Authorization": f"Bearer {TOKEN}"},
    json=payload
)

result = response.json()
corrected = np.array(result["corrected_embedding"])
print(f"Coherence: {result['coherence']:.2%}")
```

**Use Cases:**
- **Noise reduction** in high-dimensional embeddings
- **Stabilizing** learned representations
- **Error correction** for sensor data
- **Improving** clustering quality

---

### Bio Module Endpoints

The Bio module is **genomics-specific** and requires **h5ad files** (AnnData format). If you're working with non-genomics data, use the Core Compression endpoints instead.

#### Base Configuration

**Base URL:** `https://don-research-api.onrender.com`
**API Version:** `v1`
**Authentication:** Bearer token in `Authorization` header
**Rate Limit:** 1,000 requests/hour (academic tier)

#### Common Parameters

All bio endpoints support these optional parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sync` | boolean | `false` | If `true`, execute synchronously and return results immediately. If `false`, start background job and return `job_id` for polling. |
| `project_id` | string | auto-generated | Project identifier for grouping related traces |
| `user_id` | string | `"unknown"` | User identifier for audit logging |

### Endpoint 1: Export H5AD to DON Artifacts

**Convert standard single-cell h5ad files into DON bio artifacts.**

```
POST /api/v1/bio/export-artifacts
```

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | H5AD file (multipart/form-data upload) |
| `cluster_key` | string | Yes | Column in `adata.obs` containing cluster assignments (e.g., "leiden", "louvain") |
| `latent_key` | string | Yes | Key in `adata.obsm` containing latent embeddings (e.g., "X_pca", "X_umap") |
| `paga_key` | string | No | Key in `adata.uns` for PAGA connectivity graph (optional) |
| `sample_cells` | integer | No | If provided, randomly sample this many cells before export |
| `seed` | integer | No | Random seed for reproducibility (default: 42) |
| `sync` | boolean | No | Synchronous execution flag (default: false) |

**Python Example (Synchronous):**

```python
import requests
import os

headers = {"Authorization": f"Bearer {os.getenv('DON_API_TOKEN')}"}
base_url = os.getenv("DON_API_BASE_URL")

# Upload h5ad file
with open("pbmc3k_processed.h5ad", "rb") as f:
    files = {"file": ("pbmc3k.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_pca",
        "paga_key": "paga",
        "sync": "true",
        "seed": 42
    }

    response = requests.post(
        f"{base_url}/api/v1/bio/export-artifacts",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()
print(f"Status: {result['status']}")
print(f"Clusters: {result['nodes']}")
print(f"Cells: {result['vectors']}")
print(f"Artifacts: {result['artifacts']}")
```

**Successful Response (Synchronous):**
```json
{
  "status": "completed",
  "nodes": 9,
  "edges": 12,
  "vectors": 2638,
  "artifacts": [
    "/app/artifacts/proj-abc123/collapse_map.json",
    "/app/artifacts/proj-abc123/collapse_vectors.jsonl"
  ],
  "trace_id": "trace-xyz789",
  "project_id": "proj-abc123"
}
```

**Python Example (Asynchronous with Polling):**

```python
# Start background job
with open("large_dataset.h5ad", "rb") as f:
    files = {"file": ("large_dataset.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_pca",
        "sync": "false"  # Async mode
    }

    response = requests.post(
        f"{base_url}/api/v1/bio/export-artifacts",
        headers=headers,
        files=files,
        data=data
    )

job_data = response.json()
job_id = job_data["job_id"]
print(f"Job started: {job_id}")

# Poll for completion
import time
for attempt in range(60):  # Poll for up to 60 seconds
    status_response = requests.get(
        f"{base_url}/api/v1/bio/jobs/{job_id}",
        headers=headers
    )
    status_data = status_response.json()

    if status_data["status"] == "completed":
        print("Job completed!")
        print(f"Result: {status_data['result']}")
        break
    elif status_data["status"] == "failed":
        print(f"Job failed: {status_data.get('error')}")
        break

    time.sleep(1)  # Wait 1 second before next poll
```

**Asynchronous Response (Initial):**
```json
{
  "status": "pending",
  "job_id": "job-456def",
  "message": "Export started in background"
}
```

**Asynchronous Response (Polling /api/v1/bio/jobs/{job_id}):**
```json
{
  "job_id": "job-456def",
  "status": "completed",
  "result": {
    "nodes": 15,
    "edges": 28,
    "vectors": 10000,
    "artifacts": [
      "/app/artifacts/proj-xyz/collapse_map.json",
      "/app/artifacts/proj-xyz/collapse_vectors.jsonl"
    ],
    "trace_id": "trace-abc"
  }
}
```

**Understanding the Artifacts:**

1. **`collapse_map.json`** - Cluster-level metadata
   - Array of cluster nodes with centroids, marker genes, sizes
   - PAGA connectivity edges (if provided)
   - Dataset metadata (n_cells, n_genes, n_clusters)

2. **`collapse_vectors.jsonl`** - Per-cell data (one JSON object per line)
   - Cell ID, cluster assignment, latent coordinates
   - QC metrics (n_genes, n_counts, pct_mt, doublet_score)
   - Batch information (if available)

**Error Responses:**

| Status Code | Condition | Example Message |
|-------------|-----------|-----------------|
| 400 | Invalid file format | "File must be .h5ad format" |
| 400 | Missing required parameter | "cluster_key is required" |
| 500 | Missing cluster key in h5ad | "Cluster key 'leiden' not found in adata.obs" |
| 500 | Missing latent key in h5ad | "Latent key 'X_pca' not found in adata.obsm" |
| 429 | Rate limit exceeded | "Rate limit exceeded: 1000 requests/hour" |

---

### Endpoint 2: Signal Synchronization Check

**Validate cross-artifact coherence using QAC quantum field analysis.**

```
POST /api/v1/bio/signal-sync
```

**Purpose:** Check that DON artifacts maintain quantum coherence - ensures compression hasn't introduced field instabilities that could corrupt downstream analysis.

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `collapse_map` | file | Yes | collapse_map.json artifact |
| `collapse_vectors` | file | Yes | collapse_vectors.jsonl artifact |
| `sync` | boolean | No | Synchronous execution (default: false) |

**Python Example:**

```python
# Load artifacts from previous export
map_path = "artifacts/collapse_map.json"
vectors_path = "artifacts/collapse_vectors.jsonl"

with open(map_path, "rb") as map_file, open(vectors_path, "rb") as vec_file:
    files = {
        "collapse_map": ("collapse_map.json", map_file, "application/json"),
        "collapse_vectors": ("collapse_vectors.jsonl", vec_file, "application/jsonl")
    }
    data = {"sync": "true"}

    response = requests.post(
        f"{base_url}/api/v1/bio/signal-sync",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()
print(f"Coherence: {result['coherence_score']:.2%}")
print(f"Field Stability: {result['field_stability']}")
```

**Successful Response:**
```json
{
  "status": "completed",
  "coherence_score": 0.97,
  "field_stability": "stable",
  "qac_metrics": {
    "syndrome_weight": 0.03,
    "correction_success_rate": 0.99,
    "entanglement_fidelity": 0.96
  },
  "trace_id": "trace-qac-001",
  "project_id": "proj-abc123"
}
```

**Interpreting Results:**

- **`coherence_score`** (0-1): Overall quantum coherence preservation
  - >0.95: Excellent - artifacts are quantum-stable
  - 0.85-0.95: Good - minor field fluctuations
  - <0.85: Poor - consider re-exporting with different parameters

- **`field_stability`**: Categorical assessment
  - `"stable"`: Safe for downstream analysis
  - `"minor_drift"`: Usable but monitor carefully
  - `"unstable"`: Re-export recommended

- **`qac_metrics.syndrome_weight`**: Error detection metric (lower is better)
- **`qac_metrics.entanglement_fidelity`**: Quantum state preservation (higher is better)

**When to Use This Endpoint:**
- After exporting artifacts to validate compression quality
- Before running computationally expensive downstream analysis
- When artifacts will be archived for long-term reproducibility

---

### Endpoint 3: QC Parasite Detection

**Detect contamination in single-cell data using TACE field tension analysis.**

```
POST /api/v1/bio/qc/parasite-detect
```

**Purpose:** Identify three types of QC contamination that corrupt biological interpretation:
1. **Ambient RNA** - High UMI counts but low gene diversity (ambient soup)
2. **Doublets** - Two cells captured as one (high genes + high UMIs)
3. **Batch Effects** - Poor batch mixing in clusters (technical artifact)

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | H5AD file for QC analysis |
| `cluster_key` | string | Yes | Cluster column in adata.obs |
| `batch_key` | string | No | Batch column in adata.obs (required for batch mixing analysis) |
| `ambient_threshold` | float | No | Ambient RNA flagging threshold (default: 0.15) |
| `doublet_threshold` | float | No | Doublet score threshold (default: 0.25) |
| `batch_threshold` | float | No | Batch purity threshold (default: 0.3) |
| `sync` | boolean | No | Synchronous execution (default: false) |

**Python Example:**

```python
with open("pbmc_raw.h5ad", "rb") as f:
    files = {"file": ("pbmc_raw.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "batch_key": "sample",
        "ambient_threshold": 0.15,
        "doublet_threshold": 0.25,
        "batch_threshold": 0.3,
        "sync": "true"
    }

    response = requests.post(
        f"{base_url}/api/v1/bio/qc/parasite-detect",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()
print(f"Parasite Score: {result['parasite_score']:.1f}%")
print(f"Flagged Cells: {result['n_flagged']} / {result['n_total']}")
print(f"Ambient RNA: {result['ambient']['severity']}")
print(f"Doublets: {result['doublets']['n_flagged']} detected")
print(f"Batch Issues: {result['batch']['n_flagged_clusters']} clusters")
```

**Successful Response:**
```json
{
  "status": "completed",
  "parasite_score": 12.3,
  "n_flagged": 325,
  "n_total": 2638,
  "ambient": {
    "n_flagged": 150,
    "severity": "medium",
    "affected_clusters": ["0", "3", "7"],
    "threshold": 0.15
  },
  "doublets": {
    "n_flagged": 120,
    "expected_rate": 2.6,
    "observed_rate": 4.5,
    "threshold": 0.25
  },
  "batch": {
    "n_flagged_clusters": 3,
    "poor_mixing_clusters": ["2", "5", "8"],
    "max_purity": 0.85,
    "threshold": 0.3
  },
  "trace_id": "trace-qc-001",
  "project_id": "proj-abc123"
}
```

**Interpreting Results:**

**Overall Parasite Score:**
- **<5%**: Excellent data quality
- **5-15%**: Moderate contamination - consider filtering
- **>15%**: High contamination - filtering strongly recommended

**Ambient RNA:**
- `severity: "low"` (<5% affected): Minimal impact
- `severity: "medium"` (5-15%): Consider SoupX/CellBender correction
- `severity: "high"` (>15%): Requires decontamination before analysis
- `affected_clusters`: Which clusters show ambient contamination

**Doublets:**
- `expected_rate`: Theoretical doublet rate based on cell count (0.4% per 1000 cells)
- `observed_rate`: Actual flagged doublet percentage
- High observed vs. expected ratio suggests doublet contamination

**Batch Effects:**
- `poor_mixing_clusters`: Clusters dominated by single batch (purity >0.7)
- `max_purity`: Worst batch purity observed (1.0 = pure single-batch cluster)
- High purity indicates technical batch effects vs. biological variation

**Recommended Actions:**

```python
# Example: Filter flagged cells
import scanpy as sc
import json

# Load QC results
qc_result = response.json()

# Load original h5ad
adata = sc.read_h5ad("pbmc_raw.h5ad")

# Get flagged cell indices (you'd need to fetch these from API)
# In practice, the API could return flagged cell IDs in response
flagged_cells = qc_result.get("flagged_cell_ids", [])

# Filter
adata_clean = adata[~adata.obs_names.isin(flagged_cells)].copy()
adata_clean.write_h5ad("pbmc_cleaned.h5ad")

print(f"Removed {len(flagged_cells)} contaminated cells")
print(f"Retained {adata_clean.n_obs} high-quality cells")
```

---

### Endpoint 4: Evolution Report (Pipeline Stability)

**Compare two analysis runs for stability and reproducibility using TACE temporal field analysis.**

```
POST /api/v1/bio/evolution/report
```

**Purpose:** Validate that your analysis pipeline produces stable, reproducible results across runs. Critical for:
- Parameter optimization (ensuring changes improve vs. destabilize results)
- Longitudinal studies (detecting real biological change vs. technical drift)
- Publication reproducibility (proving findings are robust)

**Request Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `baseline_file` | file | Yes | H5AD file from baseline run |
| `candidate_file` | file | Yes | H5AD file from candidate run |
| `cluster_key` | string | Yes | Cluster column to compare |
| `latent_key` | string | Yes | Latent space to compare (e.g., "X_umap") |
| `sync` | boolean | No | Synchronous execution (default: false) |

**Python Example:**

```python
# Compare two runs of the same dataset with different random seeds
with open("run1_seed42.h5ad", "rb") as baseline, open("run2_seed99.h5ad", "rb") as candidate:
    files = {
        "baseline_file": ("baseline.h5ad", baseline, "application/octet-stream"),
        "candidate_file": ("candidate.h5ad", candidate, "application/octet-stream")
    }
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_umap",
        "sync": "true"
    }

    response = requests.post(
        f"{base_url}/api/v1/bio/evolution/report",
        headers=headers,
        files=files,
        data=data
    )

result = response.json()
print(f"Overall Stability: {result['overall_stability']:.1f}/100")
print(f"Recommendation: {result['recommendation']}")
print("\nMetric Deltas:")
for metric, value in result['deltas'].items():
    print(f"  {metric}: {value}")
```

**Successful Response:**
```json
{
  "status": "completed",
  "overall_stability": 87.5,
  "recommendation": "accept",
  "deltas": {
    "cell_count_delta": 0,
    "cluster_count_delta": 1,
    "silhouette_delta": -0.03,
    "ari": 0.92,
    "latent_drift": 2.4
  },
  "stability": {
    "silhouette": "stable",
    "cluster_count": "minor_drift",
    "ari": "stable",
    "latent": "stable"
  },
  "trace_id": "trace-evo-001",
  "project_id": "proj-abc123"
}
```

**Interpreting Results:**

**Overall Stability Score (0-100):**
- **>85**: Excellent reproducibility - pipeline is stable
- **70-85**: Good - minor variations acceptable for biological data
- **<70**: Poor - investigate parameter settings or data quality

**Recommendation:**
- `"accept"`: Candidate run is stable, safe to proceed
- `"review"`: Minor drift detected, manual inspection recommended
- `"reject"`: Significant drift, re-run with fixed parameters

**Key Metrics:**

| Metric | Meaning | Stable Threshold |
|--------|---------|------------------|
| `cell_count_delta` | Change in total cells | 0 (should be identical) |
| `cluster_count_delta` | Change in number of clusters | ±1 cluster acceptable |
| `silhouette_delta` | Change in clustering quality (-1 to 1) | ±0.05 |
| `ari` | Adjusted Rand Index - cluster consistency (0-1) | >0.85 |
| `latent_drift` | Euclidean distance between cluster centroids | <5.0 |

**Stability Assessment per Metric:**
- `"stable"`: Within acceptable variation
- `"minor_drift"`: Borderline, review recommended
- `"significant_drift"`: Outside acceptable bounds

**Use Cases:**

**1. Parameter Optimization:**
```python
# Test if increasing resolution improves clustering
baseline = run_pipeline(resolution=0.8, seed=42)
candidate = run_pipeline(resolution=1.2, seed=42)

# Compare stability
response = compare_runs(baseline, candidate)
if response["recommendation"] == "accept" and response["deltas"]["silhouette_delta"] > 0:
    print("Higher resolution improves clustering quality")
```

**2. Reproducibility Validation:**
```python
# Run same pipeline with different random seeds
run1 = run_pipeline(seed=42)
run2 = run_pipeline(seed=99)

response = compare_runs(run1, run2)
if response["overall_stability"] > 85:
    print("Pipeline is reproducible across random seeds")
```

**3. Longitudinal Stability:**
```python
# Compare time points in longitudinal study
baseline = process("patient_timepoint_1.h5ad")
candidate = process("patient_timepoint_2.h5ad")

response = compare_runs(baseline, candidate)
if response["deltas"]["cluster_count_delta"] > 2:
    print("Significant cluster emergence - real biological change")
```

---

### Endpoint 5: Usage Metrics

**Query your institution's API usage statistics.**

```
GET /api/v1/usage
```

**Python Example:**

```python
response = requests.get(
    f"{base_url}/api/v1/usage",
    headers=headers
)

usage = response.json()
print(f"Requests this hour: {usage['requests_this_hour']} / {usage['rate_limit']}")
print(f"Total requests: {usage['total_requests']}")
```

**Response:**
```json
{
  "institution": "Texas A&M University",
  "requests_this_hour": 47,
  "rate_limit": 1000,
  "total_requests": 3421,
  "period_start": "2025-10-27T10:00:00Z"
}
```

---

### Endpoint 6: System Health

**Check DON Stack engine availability.**

```
GET /api/v1/health
```

**Python Example:**

```python
response = requests.get(f"{base_url}/api/v1/health")
health = response.json()

if health["status"] == "healthy":
    print("All systems operational")
else:
    print(f"System degraded: {health}")
```

**Response:**
```json
{
  "status": "healthy",
  "don_stack": {
    "mode": "production",
    "don_gpu": true,
    "tace": true,
    "qac": true,
    "adapter_loaded": true
  },
  "timestamp": "2025-10-27T10:30:00Z"
}
```

---

## Research Workflows

### Workflow 1: Standard Single-Cell Analysis Pipeline

**Scenario:** Process a new PBMC dataset from Cell Ranger output to publication-ready analysis.

**Step-by-step:**

```python
import scanpy as sc
import requests
import os

# Configuration
API_TOKEN = os.getenv("DON_API_TOKEN")
API_BASE = os.getenv("DON_API_BASE_URL")
headers = {"Authorization": f"Bearer {API_TOKEN}"}

# Step 1: Standard Scanpy preprocessing
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")

# Basic QC filtering
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)

# Calculate QC metrics
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# Filter low-quality cells
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

# Normalize and find highly variable genes
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# PCA
sc.tl.pca(adata, svd_solver='arpack')

# Neighbors and clustering
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
sc.tl.leiden(adata, resolution=0.8)
sc.tl.paga(adata)
sc.pl.paga(adata, plot=False)  # Compute PAGA connectivity

# UMAP
sc.tl.umap(adata)

# Add doublet scores (using scrublet or doubletdetection)
import scrublet as scr
scrub = scr.Scrublet(adata.X)
doublet_scores, predicted_doublets = scrub.scrub_doublets()
adata.obs['doublet_score'] = doublet_scores

# Add batch info if applicable
adata.obs['batch'] = 'batch1'  # Modify as needed

# Save preprocessed file
adata.write_h5ad("pbmc_preprocessed.h5ad")

# Step 2: Run QC parasite detection via DON API
print("Detecting QC parasites...")
with open("pbmc_preprocessed.h5ad", "rb") as f:
    files = {"file": ("pbmc.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "batch_key": "batch",
        "sync": "true"
    }
    qc_response = requests.post(
        f"{API_BASE}/api/v1/bio/qc/parasite-detect",
        headers=headers,
        files=files,
        data=data
    )

qc_result = qc_response.json()
print(f"Parasite Score: {qc_result['parasite_score']:.1f}%")

if qc_result['parasite_score'] > 15:
    print("WARNING: High contamination detected. Consider additional filtering.")
    print(f"  Ambient RNA: {qc_result['ambient']['severity']}")
    print(f"  Doublets: {qc_result['doublets']['n_flagged']} flagged")
else:
    print("Data quality is acceptable.")

# Step 3: Export to DON artifacts for compression
print("\nExporting to DON artifacts...")
with open("pbmc_preprocessed.h5ad", "rb") as f:
    files = {"file": ("pbmc.h5ad", f, "application/octet-stream")}
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_pca",
        "paga_key": "paga",
        "sync": "true",
        "seed": 42
    }
    export_response = requests.post(
        f"{API_BASE}/api/v1/bio/export-artifacts",
        headers=headers,
        files=files,
        data=data
    )

export_result = export_response.json()
print(f"Compression achieved:")
print(f"  Original: {adata.n_obs} cells × {adata.n_vars} genes")
print(f"  Compressed: {export_result['nodes']} clusters, {export_result['vectors']} vectors")

original_size = adata.n_obs * adata.n_vars
compressed_size = export_result['nodes'] * 50 + export_result['vectors'] * 20  # Rough estimate
compression_ratio = original_size / compressed_size
print(f"  Compression ratio: {compression_ratio:.1f}×")

# Step 4: Validate quantum coherence
print("\nValidating quantum coherence...")
# Download artifacts first (in production, artifacts would be retrievable via trace_id)
# For this example, assume artifacts are local
map_path = export_result['artifacts'][0]
vec_path = export_result['artifacts'][1]

# Note: In production deployment, you'd need artifact retrieval endpoint
# For now, this is illustrative

print("Analysis complete! Artifacts ready for downstream DON Stack processing.")
```

**Expected Output:**
```
Detecting QC parasites...
Parasite Score: 8.3%
Data quality is acceptable.

Exporting to DON artifacts...
Compression achieved:
  Original: 2638 cells × 13714 genes
  Compressed: 9 clusters, 2638 vectors
  Compression ratio: 24.7×

Validating quantum coherence...
Analysis complete! Artifacts ready for downstream DON Stack processing.
```

---

### Workflow 2: Reproducibility Validation for Publication

**Scenario:** Validate that your analysis pipeline produces consistent results before submission.

```python
import scanpy as sc
import requests
import os

headers = {"Authorization": f"Bearer {os.getenv('DON_API_TOKEN')}"}
base_url = os.getenv("DON_API_BASE_URL")

def run_analysis_pipeline(h5ad_input, seed, output_name):
    """Run complete analysis with specified random seed."""
    adata = sc.read_h5ad(h5ad_input)

    # Set random seed for reproducibility
    import numpy as np
    np.random.seed(seed)

    # Preprocessing (deterministic)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata)

    # PCA (with seed)
    sc.tl.pca(adata, random_state=seed)

    # Clustering (with seed)
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.leiden(adata, random_state=seed, resolution=0.8)
    sc.tl.umap(adata, random_state=seed)

    adata.write_h5ad(output_name)
    return output_name

# Run pipeline with 3 different random seeds
print("Running reproducibility test with 3 different seeds...")
run1 = run_analysis_pipeline("raw_data.h5ad", seed=42, output_name="run_seed42.h5ad")
run2 = run_analysis_pipeline("raw_data.h5ad", seed=99, output_name="run_seed99.h5ad")
run3 = run_analysis_pipeline("raw_data.h5ad", seed=123, output_name="run_seed123.h5ad")

# Compare run1 vs run2
print("\nComparing Run 1 (seed=42) vs Run 2 (seed=99)...")
with open(run1, "rb") as baseline, open(run2, "rb") as candidate:
    files = {
        "baseline_file": ("baseline.h5ad", baseline, "application/octet-stream"),
        "candidate_file": ("candidate.h5ad", candidate, "application/octet-stream")
    }
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_umap",
        "sync": "true"
    }

    response = requests.post(
        f"{base_url}/api/v1/bio/evolution/report",
        headers=headers,
        files=files,
        data=data
    )

result_1v2 = response.json()
print(f"Stability Score: {result_1v2['overall_stability']:.1f}/100")
print(f"Recommendation: {result_1v2['recommendation']}")
print(f"Cluster Consistency (ARI): {result_1v2['deltas']['ari']:.3f}")

# Compare run1 vs run3
print("\nComparing Run 1 (seed=42) vs Run 3 (seed=123)...")
with open(run1, "rb") as baseline, open(run3, "rb") as candidate:
    files = {
        "baseline_file": ("baseline.h5ad", baseline, "application/octet-stream"),
        "candidate_file": ("candidate.h5ad", candidate, "application/octet-stream")
    }
    data = {
        "cluster_key": "leiden",
        "latent_key": "X_umap",
        "sync": "true"
    }

    response = requests.post(
        f"{base_url}/api/v1/bio/evolution/report",
        headers=headers,
        files=files,
        data=data
    )

result_1v3 = response.json()
print(f"Stability Score: {result_1v3['overall_stability']:.1f}/100")
print(f"Recommendation: {result_1v3['recommendation']}")

# Overall assessment
avg_stability = (result_1v2['overall_stability'] + result_1v3['overall_stability']) / 2
print(f"\n{'='*60}")
print(f"REPRODUCIBILITY ASSESSMENT")
print(f"{'='*60}")
print(f"Average Stability: {avg_stability:.1f}/100")

if avg_stability > 85:
    print("✓ EXCELLENT: Pipeline is highly reproducible")
    print("  Safe to proceed with publication")
elif avg_stability > 70:
    print("⚠ GOOD: Minor random variations present")
    print("  Document random seed in methods section")
else:
    print("✗ POOR: Significant run-to-run variation")
    print("  Investigate parameter settings and data quality")
```

**Expected Output:**
```
Running reproducibility test with 3 different seeds...

Comparing Run 1 (seed=42) vs Run 2 (seed=99)...
Stability Score: 89.3/100
Recommendation: accept
Cluster Consistency (ARI): 0.934

Comparing Run 1 (seed=42) vs Run 3 (seed=123)...
Stability Score: 87.8/100
Recommendation: accept

============================================================
REPRODUCIBILITY ASSESSMENT
============================================================
Average Stability: 88.6/100
✓ EXCELLENT: Pipeline is highly reproducible
  Safe to proceed with publication
```

---

### Workflow 3: Parameter Optimization

**Scenario:** Optimize clustering resolution to maximize biological signal.

```python
import scanpy as sc
import requests
import os
import numpy as np

headers = {"Authorization": f"Bearer {os.getenv('DON_API_TOKEN')}"}
base_url = os.getenv("DON_API_BASE_URL")

# Load preprocessed data
adata = sc.read_h5ad("pbmc_normalized.h5ad")

# Test different resolution parameters
resolutions = [0.4, 0.6, 0.8, 1.0, 1.2]
results = []

print("Testing resolution parameters...")
for res in resolutions:
    print(f"\nResolution: {res}")

    # Re-cluster with new resolution
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_r{res}')

    # Calculate silhouette score
    from sklearn.metrics import silhouette_score
    latent = adata.obsm['X_pca']
    clusters = adata.obs[f'leiden_r{res}'].astype(int)
    sil_score = silhouette_score(latent, clusters)

    n_clusters = len(adata.obs[f'leiden_r{res}'].unique())

    print(f"  Clusters: {n_clusters}")
    print(f"  Silhouette: {sil_score:.3f}")

    results.append({
        'resolution': res,
        'n_clusters': n_clusters,
        'silhouette': sil_score
    })

# Find optimal resolution (maximize silhouette score)
optimal = max(results, key=lambda x: x['silhouette'])
print(f"\n{'='*60}")
print(f"OPTIMAL RESOLUTION: {optimal['resolution']}")
print(f"  Clusters: {optimal['n_clusters']}")
print(f"  Silhouette: {optimal['silhouette']:.3f}")
print(f"{'='*60}")

# Apply optimal clustering
sc.tl.leiden(adata, resolution=optimal['resolution'], key_added='leiden_optimal')
adata.write_h5ad("pbmc_optimized.h5ad")

# Validate stability of optimal clustering
print("\nValidating stability of optimal clustering...")
adata_test = sc.read_h5ad("pbmc_normalized.h5ad")
sc.tl.leiden(adata_test, resolution=optimal['resolution'], random_state=99)
adata_test.write_h5ad("pbmc_optimized_seed99.h5ad")

with open("pbmc_optimized.h5ad", "rb") as baseline, open("pbmc_optimized_seed99.h5ad", "rb") as candidate:
    files = {
        "baseline_file": ("baseline.h5ad", baseline, "application/octet-stream"),
        "candidate_file": ("candidate.h5ad", candidate, "application/octet-stream")
    }
    data = {
        "cluster_key": "leiden_optimal",
        "latent_key": "X_pca",
        "sync": "true"
    }

    response = requests.post(
        f"{base_url}/api/v1/bio/evolution/report",
        headers=headers,
        files=files,
        data=data
    )

stability = response.json()
print(f"Stability Score: {stability['overall_stability']:.1f}/100")

if stability['overall_stability'] > 85:
    print("✓ Optimal clustering is stable and reproducible")
else:
    print("⚠ Optimal clustering shows variation - consider testing broader range")
```

---

## Integration with Scanpy

### Scanpy Preprocessing Checklist

Before using DON Stack API, ensure your h5ad files contain required metadata:

**Required in `adata.obs` (cell metadata):**
- Cluster assignments (e.g., `leiden`, `louvain`)
- QC metrics: `n_genes_by_counts`, `total_counts`, `pct_counts_mt`
- Doublet scores: `doublet_score` (from scrublet/doubletdetection)
- Batch information: Custom column (e.g., `sample`, `batch`)

**Required in `adata.obsm` (cell embeddings):**
- Latent space: `X_pca` or `X_umap` (numpy array, n_cells × n_dims)

**Optional in `adata.uns` (unstructured metadata):**
- PAGA connectivity: `paga` (from `sc.tl.paga()`)

**Example Scanpy Preprocessing:**

```python
import scanpy as sc
import scrublet as scr

# Load data
adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")

# ========================================
# 1. Basic QC Metrics (REQUIRED)
# ========================================
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(
    adata,
    qc_vars=['mt'],
    percent_top=None,
    log1p=False,
    inplace=True
)

# Filter
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 5, :].copy()

# ========================================
# 2. Doublet Detection (REQUIRED)
# ========================================
scrub = scr.Scrublet(adata.X)
doublet_scores, predicted_doublets = scrub.scrub_doublets()
adata.obs['doublet_score'] = doublet_scores

# ========================================
# 3. Batch Information (OPTIONAL)
# ========================================
# If you have multiple samples/batches:
# adata.obs['batch'] = ['batch1']*1000 + ['batch2']*1638  # Example
# Or from metadata file:
# adata.obs['batch'] = batch_metadata['sample_id']

# ========================================
# 4. Normalization and HVG
# ========================================
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)

# ========================================
# 5. Dimensionality Reduction (REQUIRED)
# ========================================
sc.tl.pca(adata, random_state=42)  # Creates adata.obsm['X_pca']
sc.pp.neighbors(adata, random_state=42)
sc.tl.umap(adata, random_state=42)  # Creates adata.obsm['X_umap']

# ========================================
# 6. Clustering (REQUIRED)
# ========================================
sc.tl.leiden(adata, random_state=42, resolution=0.8)  # Creates adata.obs['leiden']

# ========================================
# 7. PAGA (OPTIONAL but recommended)
# ========================================
sc.tl.paga(adata)  # Creates adata.uns['paga']
sc.pl.paga(adata, plot=False)  # Compute without plotting

# ========================================
# 8. Save for DON API
# ========================================
adata.write_h5ad("ready_for_don_api.h5ad")

print("✓ H5AD file ready for DON Stack API")
print(f"  Cells: {adata.n_obs}")
print(f"  Genes: {adata.n_vars}")
print(f"  Clusters: {len(adata.obs['leiden'].unique())}")
print(f"  Has PAGA: {'paga' in adata.uns}")
```

### Verifying H5AD Compatibility

```python
def verify_don_compatibility(adata):
    """Check if AnnData object is ready for DON API."""
    errors = []
    warnings = []

    # Check required obs columns
    required_obs = ['n_genes_by_counts', 'total_counts', 'pct_counts_mt', 'doublet_score']
    for col in required_obs:
        if col not in adata.obs.columns:
            errors.append(f"Missing required column in adata.obs: '{col}'")

    # Check for cluster assignments
    cluster_candidates = ['leiden', 'louvain', 'clusters']
    has_clusters = any(c in adata.obs.columns for c in cluster_candidates)
    if not has_clusters:
        errors.append("No cluster assignments found (expected 'leiden', 'louvain', or 'clusters')")

    # Check obsm for latent spaces
    latent_candidates = ['X_pca', 'X_umap']
    has_latent = any(l in adata.obsm.keys() for l in latent_candidates)
    if not has_latent:
        errors.append("No latent space found (expected 'X_pca' or 'X_umap' in adata.obsm)")

    # Check for PAGA (optional)
    if 'paga' not in adata.uns:
        warnings.append("PAGA connectivity not found (optional but recommended)")

    # Check for batch info (optional)
    batch_candidates = ['batch', 'sample', 'donor']
    has_batch = any(b in adata.obs.columns for b in batch_candidates)
    if not has_batch:
        warnings.append("No batch information found (optional but needed for batch effect analysis)")

    # Report
    if errors:
        print("❌ ERRORS - Cannot use with DON API:")
        for err in errors:
            print(f"  - {err}")

    if warnings:
        print("\n⚠️  WARNINGS - API will work but with limited functionality:")
        for warn in warnings:
            print(f"  - {warn}")

    if not errors and not warnings:
        print("✅ Perfect! AnnData object is fully compatible with DON API")
    elif not errors:
        print("\n✅ Compatible with DON API (warnings are non-blocking)")

    return len(errors) == 0

# Usage
import scanpy as sc
adata = sc.read_h5ad("my_data.h5ad")
is_compatible = verify_don_compatibility(adata)
```

---

## Best Practices

### 1. Rate Limiting and Request Management

**Your Rate Limit:** 1,000 requests/hour (academic tier)

**Best Practices:**
- Use synchronous mode (`sync=true`) for small datasets (<5000 cells)
- Use asynchronous mode (`sync=false`) for large datasets to avoid timeout
- Batch process multiple files during off-hours to maximize throughput
- Monitor usage via `/api/v1/usage` endpoint

**Example: Batch Processing with Rate Limit Awareness**

```python
import requests
import time
import os

headers = {"Authorization": f"Bearer {os.getenv('DON_API_TOKEN')}"}
base_url = os.getenv("DON_API_BASE_URL")

def check_rate_limit():
    """Check current rate limit status."""
    response = requests.get(f"{base_url}/api/v1/usage", headers=headers)
    usage = response.json()
    remaining = usage['rate_limit'] - usage['requests_this_hour']
    return remaining

def batch_process_files(file_list, delay=2):
    """Process multiple h5ad files with rate limit awareness."""
    results = []

    for i, filepath in enumerate(file_list):
        # Check rate limit every 10 files
        if i % 10 == 0:
            remaining = check_rate_limit()
            print(f"Rate limit remaining: {remaining} requests")
            if remaining < 10:
                print("Approaching rate limit, waiting 60 seconds...")
                time.sleep(60)

        # Process file
        print(f"Processing {filepath}...")
        with open(filepath, "rb") as f:
            files = {"file": (os.path.basename(filepath), f, "application/octet-stream")}
            data = {
                "cluster_key": "leiden",
                "latent_key": "X_pca",
                "sync": "false"  # Async for batch processing
            }

            response = requests.post(
                f"{base_url}/api/v1/bio/export-artifacts",
                headers=headers,
                files=files,
                data=data
            )

            if response.status_code == 200:
                results.append(response.json())
            else:
                print(f"  Error: {response.json().get('detail')}")

        # Respectful delay between requests
        time.sleep(delay)

    return results

# Usage
files_to_process = [
    "patient1_pbmc.h5ad",
    "patient2_pbmc.h5ad",
    "patient3_pbmc.h5ad"
]

results = batch_process_files(files_to_process)
```

---

### 2. Async vs Sync Execution

**When to use Synchronous (`sync=true`):**
- Small datasets (<5,000 cells)
- Interactive exploratory analysis
- When you need immediate results
- Testing and debugging

**When to use Asynchronous (`sync=false`):**
- Large datasets (>5,000 cells)
- Batch processing multiple files
- Production pipelines
- When long processing time expected

**Async Polling Best Practices:**

```python
import requests
import time

def poll_job(job_id, max_wait=300, poll_interval=2):
    """
    Poll for job completion with exponential backoff.

    Args:
        job_id: Job ID from async API call
        max_wait: Maximum seconds to wait (default 5 minutes)
        poll_interval: Initial seconds between polls (default 2)

    Returns:
        Job result dict or None if timeout
    """
    start_time = time.time()
    current_interval = poll_interval

    while time.time() - start_time < max_wait:
        response = requests.get(
            f"{base_url}/api/v1/bio/jobs/{job_id}",
            headers=headers
        )

        if response.status_code != 200:
            print(f"Error checking job status: {response.json()}")
            return None

        job_data = response.json()

        if job_data["status"] == "completed":
            return job_data["result"]
        elif job_data["status"] == "failed":
            print(f"Job failed: {job_data.get('error')}")
            return None

        # Exponential backoff (2s → 4s → 8s → max 30s)
        time.sleep(current_interval)
        current_interval = min(current_interval * 2, 30)

    print(f"Job timeout after {max_wait} seconds")
    return None

# Usage
response = requests.post(
    f"{base_url}/api/v1/bio/export-artifacts",
    headers=headers,
    files=files,
    data={"cluster_key": "leiden", "latent_key": "X_pca", "sync": "false"}
)

job_id = response.json()["job_id"]
result = poll_job(job_id, max_wait=600)  # Wait up to 10 minutes

if result:
    print(f"Export completed: {result['artifacts']}")
```

---

### 3. Reproducibility and Trace Logging

Every API call generates a **trace ID** that logs:
- Input parameters and file checksums
- System health snapshot
- Processing metrics
- Output artifact paths
- Timestamps (started_at, finished_at)

**Best Practices:**

**1. Always capture trace IDs:**
```python
response = requests.post(...)
result = response.json()
trace_id = result.get("trace_id")
project_id = result.get("project_id")

# Log to your lab notebook
print(f"Trace ID: {trace_id}")
print(f"Project ID: {project_id}")
```

**2. Use consistent project IDs for related analyses:**
```python
# All analyses for same study use same project_id
PROJECT_ID = "pbmc_longtidinal_study_2025"

data = {
    "cluster_key": "leiden",
    "latent_key": "X_pca",
    "project_id": PROJECT_ID,  # Group related traces
    "user_id": "jcai_lab",
    "sync": "true"
}
```

**3. Document trace IDs in publications:**
```markdown
## Methods - Computational Analysis

Single-cell data were processed using the DON Stack Research API
(https://don-research-api.onrender.com). Compression was performed
via the /export-artifacts endpoint with cluster_key='leiden',
latent_key='X_pca', and seed=42.

Trace IDs for reproducibility:
- Patient 1: trace-a1b2c3d4
- Patient 2: trace-e5f6g7h8
- Patient 3: trace-i9j0k1l2

All analyses used DON Health Commons License (DHCL) v0.1 compliant
infrastructure.
```

**4. Query traces for reproducibility:**
```python
# In future API versions, you'll be able to query traces
# For now, trace IDs are returned with each response

def log_analysis_trace(trace_id, description, output_file="analysis_log.md"):
    """Append trace to lab analysis log."""
    import datetime

    with open(output_file, "a") as f:
        f.write(f"\n### {datetime.datetime.now().isoformat()}\n")
        f.write(f"**Description:** {description}\n")
        f.write(f"**Trace ID:** `{trace_id}`\n")
        f.write("\n---\n")

# Usage
trace_id = result["trace_id"]
log_analysis_trace(trace_id, "PBMC 3k compression with leiden clustering")
```

---

### 4. Error Handling

**Robust API Client Example:**

```python
import requests
import time
from typing import Optional, Dict, Any

class DONAPIClient:
    """Robust client for DON Stack Research API."""

    def __init__(self, token: str, base_url: str, max_retries: int = 3):
        self.token = token
        self.base_url = base_url.rstrip('/')
        self.max_retries = max_retries
        self.headers = {"Authorization": f"Bearer {token}"}

    def _request_with_retry(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make request with exponential backoff retry."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(self.max_retries):
            try:
                response = requests.request(method, url, headers=self.headers, **kwargs)

                # Don't retry on client errors (4xx except 429)
                if 400 <= response.status_code < 500 and response.status_code != 429:
                    return response

                # Retry on 429 (rate limit) and 5xx (server errors)
                if response.status_code in [429, 500, 502, 503, 504]:
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                        print(f"Request failed (status {response.status_code}), retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue

                return response

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"Request exception: {e}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise

        raise Exception(f"Max retries ({self.max_retries}) exceeded")

    def export_artifacts(
        self,
        h5ad_path: str,
        cluster_key: str,
        latent_key: str,
        sync: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """Export h5ad to DON artifacts with error handling."""
        try:
            with open(h5ad_path, "rb") as f:
                files = {"file": (os.path.basename(h5ad_path), f, "application/octet-stream")}
                data = {
                    "cluster_key": cluster_key,
                    "latent_key": latent_key,
                    "sync": str(sync).lower(),
                    **kwargs
                }

                response = self._request_with_retry(
                    "POST",
                    "/api/v1/bio/export-artifacts",
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout
                )

            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get("detail", "Unknown error")
                print(f"Export failed: {error_detail}")
                return None

        except FileNotFoundError:
            print(f"Error: File not found: {h5ad_path}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def check_health(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self._request_with_retry("GET", "/api/v1/health", timeout=10)
            if response.status_code == 200:
                health = response.json()
                return health.get("status") == "healthy"
            return False
        except:
            return False

# Usage
client = DONAPIClient(
    token=os.getenv("DON_API_TOKEN"),
    base_url=os.getenv("DON_API_BASE_URL"),
    max_retries=3
)

# Check health before processing
if not client.check_health():
    print("API is unhealthy, aborting")
    exit(1)

# Process with robust error handling
result = client.export_artifacts(
    h5ad_path="data/pbmc3k.h5ad",
    cluster_key="leiden",
    latent_key="X_pca",
    sync=True,
    seed=42
)

if result:
    print(f"Success! Trace ID: {result['trace_id']}")
else:
    print("Export failed, check logs")
```

---

### 5. Data Security and Compliance

**Data Transmission Security:**
- All API calls use **HTTPS encryption** (TLS 1.2+)
- Bearer tokens transmitted in secure headers (never in URLs)
- Files uploaded via encrypted multipart/form-data

**Data Retention Policy:**
- Uploaded h5ad files are **processed in-memory** and discarded immediately
- Generated artifacts stored temporarily (24-48 hours) for retrieval
- Trace metadata (metrics, parameters) retained for reproducibility
- **No genomic data** is permanently stored server-side

**Your Responsibilities:**
- **Never commit tokens to Git** - use environment variables or secret managers
- **Rotate tokens** if compromised or when team members leave
- **Document data provenance** - record trace IDs for all published analyses
- **Comply with IRB requirements** - ensure API usage is covered by your protocols

**DON Health Commons License (DHCL) Compliance:**
- Texas A&M qualifies as "mission-aligned entity" under DHCL v0.1
- You may publish research findings derived from DON API results
- Attribution required in methods section (see reproducibility section)
- Commercial sublicensing prohibited (contact DON Systems for enterprise licensing)

---

## Troubleshooting

### Common Errors and Solutions

#### Error: "Rate limit exceeded: 1000 requests/hour"

**Cause:** You've exceeded your hourly request quota.

**Solutions:**
1. Check usage: `GET /api/v1/usage`
2. Wait for rate limit reset (top of next hour)
3. Use async mode for large batch jobs to reduce request count
4. Contact DON Systems if you need higher limits for large studies

---

#### Error: "Cluster key 'leiden' not found in adata.obs"

**Cause:** Your h5ad file doesn't contain the specified cluster column.

**Solutions:**
1. Check available columns:
   ```python
   import scanpy as sc
   adata = sc.read_h5ad("your_file.h5ad")
   print(adata.obs.columns)  # List available columns
   ```

2. Run clustering if missing:
   ```python
   sc.pp.neighbors(adata)
   sc.tl.leiden(adata)  # Creates 'leiden' column
   adata.write_h5ad("your_file.h5ad")
   ```

3. Use correct column name in API call (e.g., `"louvain"` instead of `"leiden"`)

---

#### Error: "Latent key 'X_pca' not found in adata.obsm"

**Cause:** Your h5ad file doesn't contain the specified latent space.

**Solutions:**
1. Check available embeddings:
   ```python
   print(adata.obsm.keys())  # List available latent spaces
   ```

2. Compute PCA if missing:
   ```python
   sc.tl.pca(adata)  # Creates adata.obsm['X_pca']
   adata.write_h5ad("your_file.h5ad")
   ```

3. Use existing embedding (e.g., `"X_umap"` if available)

---

#### Error: "Job timeout after 300 seconds"

**Cause:** Asynchronous job took longer than expected.

**Solutions:**
1. Increase polling timeout:
   ```python
   result = poll_job(job_id, max_wait=600)  # 10 minutes
   ```

2. For very large datasets (>50,000 cells), contact support for status

3. Check system health: `GET /api/v1/health`

---

#### Error: "File must be .h5ad format"

**Cause:** Uploaded file is not a valid h5ad file.

**Solutions:**
1. Verify file format:
   ```python
   import scanpy as sc
   try:
       adata = sc.read_h5ad("your_file.h5ad")
       print("Valid h5ad file")
   except Exception as e:
       print(f"Invalid h5ad: {e}")
   ```

2. Convert from other formats:
   ```python
   # From 10X
   adata = sc.read_10x_h5("filtered_feature_bc_matrix.h5")
   adata.write_h5ad("converted.h5ad")

   # From CSV
   import pandas as pd
   import anndata
   expr = pd.read_csv("expression_matrix.csv", index_col=0)
   adata = anndata.AnnData(expr.T)  # Transpose: genes × cells → cells × genes
   adata.write_h5ad("converted.h5ad")
   ```

---

#### Error: "coherence_score: 0.72" (Low coherence)

**Cause:** Compression artifacts show poor quantum field stability.

**Solutions:**
1. **Check input data quality** - Run QC parasite detection first
2. **Increase latent dimensions**:
   ```python
   sc.tl.pca(adata, n_comps=50)  # Instead of default 40
   ```
3. **Use higher-quality cluster assignments**:
   ```python
   # Optimize resolution for higher silhouette score
   sc.tl.leiden(adata, resolution=0.6)  # Test different values
   ```
4. **Filter low-quality cells** before export

---

### Getting Help

**For API Technical Issues:**
- Check system health: `GET /api/v1/health`
- Review trace logs for error details
- Contact: `support@donsystems.com`

**For Scientific Questions:**
- Collaboration lead: Professor James J. Cai (jcai@tamu.edu)
- DON Stack methodology: Reference published papers in README.md

**For Token/Access Issues:**
- Contact: `donnievanmetre@gmail.com` (token provisioning)
- Include institution name and research project description

---

## Support and Collaboration

### Texas A&M Partnership

This API deployment represents a research collaboration between **DON Systems** and **Professor James J. Cai's lab** at Texas A&M University, focusing on quantum computing applications in single-cell biology.

**Collaboration Framework:**

**What Dr. Cai's Lab Can Do:**
✅ Use API for any academic research
✅ Publish scientific findings derived from DON API results
✅ Integrate DON compression into existing pipelines
✅ Share usage patterns and feedback to improve algorithms
✅ Request new features or analysis endpoints
✅ Collaborate on methodology papers

**What Requires Additional Agreement:**
❌ Commercial licensing or sublicensing
❌ Reverse engineering of DON Stack engines
❌ Sharing access tokens with external collaborators (contact DON Systems for additional tokens)
❌ Using API for non-research purposes (e.g., clinical diagnostics)

---

### Citing DON Stack in Publications

**Methods Section Template:**

> "Single-cell RNA-seq data compression was performed using the DON Stack Research API (https://don-research-api.onrender.com), which implements quantum-enhanced fractal clustering algorithms (DON-GPU) achieving [X]× compression while preserving >95% biological accuracy. Quality control contamination detection utilized TACE (Temporal Adjacency Collapse Engine) field tension analysis to identify ambient RNA, doublets, and batch effects. All analyses were conducted under the DON Health Commons License (DHCL) v0.1. Trace IDs for full reproducibility: [list trace IDs]."

**Acknowledgments Template:**

> "We thank DON Systems for providing access to the DON Stack Research API and for ongoing collaboration on quantum computing applications in genomics."

---

### Requesting New Features

If your research needs additional API capabilities (e.g., new QC metrics, different artifact formats, R client library), please contact:

**Feature Requests:**
Email: `donnievanmetre@gmail.com`
Subject: "DON API Feature Request - Texas A&M"

Include:
- Description of research use case
- Desired input/output format
- Example data (if applicable)
- Priority/timeline

---

### License Summary - DON Health Commons License (DHCL) v0.1

**Key Terms for Academic Researchers:**

**Permitted:**
- ✅ Academic research and publication
- ✅ Educational use in courses
- ✅ Non-profit clinical research (IRB-approved)
- ✅ Open-source tool integration (with attribution)

**Prohibited:**
- ❌ Commercial use by large pharmaceutical companies
- ❌ Proprietary software distribution without DHCL compliance
- ❌ Use in closed-source diagnostic products
- ❌ Sublicensing to non-mission-aligned entities

**Mission-Aligned Entities** (automatically permitted):
- Academic institutions (✅ Texas A&M qualifies)
- Non-profit research organizations
- Small biotech startups (<100 employees, mission-aligned)
- Open-source software projects

**Full License:** See `LICENSE` file in repository or contact DON Systems for clarification.

---

## Appendix: Quick Reference

### API Endpoints Summary

| Endpoint | Method | Purpose | Sync Default |
|----------|--------|---------|--------------|
| `/api/v1/health` | GET | Check system health | N/A |
| `/api/v1/usage` | GET | Query usage statistics | N/A |
| `/api/v1/bio/export-artifacts` | POST | H5AD → DON artifacts | No (async) |
| `/api/v1/bio/signal-sync` | POST | QAC coherence check | No (async) |
| `/api/v1/bio/qc/parasite-detect` | POST | Detect QC contamination | No (async) |
| `/api/v1/bio/evolution/report` | POST | Compare run stability | No (async) |
| `/api/v1/bio/jobs/{job_id}` | GET | Poll async job status | N/A |

---

### Python Client Template

```python
import requests
import os
from typing import Optional, Dict, Any

class DONClient:
    def __init__(self):
        self.token = os.getenv("DON_API_TOKEN")
        self.base = os.getenv("DON_API_BASE_URL")
        self.headers = {"Authorization": f"Bearer {self.token}"}

    def export(self, h5ad_path: str, cluster_key: str, latent_key: str) -> Optional[Dict]:
        with open(h5ad_path, "rb") as f:
            files = {"file": (os.path.basename(h5ad_path), f, "application/octet-stream")}
            data = {"cluster_key": cluster_key, "latent_key": latent_key, "sync": "true"}
            response = requests.post(f"{self.base}/api/v1/bio/export-artifacts",
                                     headers=self.headers, files=files, data=data)
            return response.json() if response.status_code == 200 else None

    def qc_check(self, h5ad_path: str, cluster_key: str, batch_key: str = None) -> Optional[Dict]:
        with open(h5ad_path, "rb") as f:
            files = {"file": (os.path.basename(h5ad_path), f, "application/octet-stream")}
            data = {"cluster_key": cluster_key, "sync": "true"}
            if batch_key:
                data["batch_key"] = batch_key
            response = requests.post(f"{self.base}/api/v1/bio/qc/parasite-detect",
                                     headers=self.headers, files=files, data=data)
            return response.json() if response.status_code == 200 else None

    def compare_runs(self, baseline: str, candidate: str, cluster_key: str, latent_key: str) -> Optional[Dict]:
        with open(baseline, "rb") as b, open(candidate, "rb") as c:
            files = {
                "baseline_file": (os.path.basename(baseline), b, "application/octet-stream"),
                "candidate_file": (os.path.basename(candidate), c, "application/octet-stream")
            }
            data = {"cluster_key": cluster_key, "latent_key": latent_key, "sync": "true"}
            response = requests.post(f"{self.base}/api/v1/bio/evolution/report",
                                     headers=self.headers, files=files, data=data)
            return response.json() if response.status_code == 200 else None

# Usage
client = DONClient()
result = client.export("pbmc3k.h5ad", "leiden", "X_pca")
print(result["trace_id"])
```

---

### Environment Variables Setup

```bash
# ~/.bashrc or ~/.zshrc
export DON_API_TOKEN="your_token_from_don_systems"
export DON_API_BASE_URL="https://don-research-api.onrender.com"

# Reload shell
source ~/.bashrc
```

---

### Contact Information

**DON Systems:**
- Technical Support: `support@donsystems.com`
- Token Provisioning: `donnievanmetre@gmail.com`
- API Base URL: `https://don-research-api.onrender.com`

**Texas A&M Collaboration Lead:**
- Professor James J. Cai: `jcai@tamu.edu`
- Department of Veterinary Integrative Biosciences

---

**Document Version:** 1.0
**Last Updated:** 2025-10-27
**License:** This guide is provided under DON Health Commons License (DHCL) v0.1 for Texas A&M research use.
