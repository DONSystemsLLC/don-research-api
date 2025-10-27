# DON Research API - Usage Guide for Texas A&M

**For: Dr. James Cai and collaborators**  
**Date: January 2025**  
**Version: 1.0**

---

## Quick Start (5 Minutes)

### 1. Setup Environment

```bash
# Install required packages
pip install requests pydantic scanpy numpy scipy pandas

# Set your API token (provided separately)
export TAMU_API_TOKEN='your_token_here'
```

### 2. Run Gene Coexpression Analysis

```bash
# Make sure you have the data file
ls data/pbmc3k_with_tace_alpha.h5ad

# Run analysis
python tamu_gene_coexpression_qac.py
```

**Expected output**: JSON file with statistical results (p-values, correlations, stability metrics)

---

## API Client Library

The `don_research_client.py` module provides a Python interface to the DON Research API.

### Basic Usage

```python
from don_research_client import DonResearchClient, QACParams

# Initialize client (reads TAMU_API_TOKEN from environment)
client = DonResearchClient()

# Check API health
health = client.health()
print(f"Status: {health['status']}")

# Check your usage limits
usage = client.usage()
print(f"Remaining requests: {usage.remaining}/{usage.limit}")
```

### QAC Operations

#### Training a Model

```python
# Prepare your embedding data (n_cells, n_features)
embedding = [
    [0.1, 0.2, 0.3],
    [0.4, 0.5, 0.6],
    [0.7, 0.8, 0.9]
]

# Train QAC model (synchronous mode - waits for result)
result = client.qac.fit(
    embedding=embedding,
    params=QACParams(k_nn=8, layers=50, reinforce_rate=0.05),
    sync=True
)

model_id = result['model_id']
print(f"Trained model: {model_id}")
```

#### Asynchronous Training (for large datasets)

```python
# Submit job and get job ID
job = client.qac.fit(
    embedding=embedding,
    params=QACParams(k_nn=15),
    sync=False
)

print(f"Job submitted: {job.id}")

# Poll until complete
result = client.qac.poll_until_complete(
    job.id,
    poll_interval=2.0,  # Check every 2 seconds
    timeout=600.0       # Max 10 minutes
)

print(f"Model ID: {result['model_id']}")
```

#### Applying a Trained Model

```python
# Apply model to new data
new_embedding = [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]

output = client.qac.apply(
    model_id=model_id,
    embedding=new_embedding,
    sync=True
)

stabilized_vectors = output['stabilized_vectors']
coherence = output['coherence']
```

### Genomics Operations

```python
# Compress gene expression data using DON-GPU
gene_names = ["GENE1", "GENE2", "GENE3"]
expression_matrix = [
    [1.0, 2.0, 3.0],  # Cell 1
    [4.0, 5.0, 6.0]   # Cell 2
]

result = client.genomics.compress(
    gene_names=gene_names,
    expression_matrix=expression_matrix,
    cell_metadata={"cell_types": ["TypeA", "TypeB"]}
)

compressed = result['compressed_vectors']
stats = result['compression_stats']
print(f"Compression ratio: {stats['ratio']}×")
```

---

## QAC Parameters Reference

### Core Parameters

| Parameter | Type | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| `k_nn` | int | 1-128 | 15 | Number of nearest neighbors |
| `layers` | int | 1-100000 | 50 | Number of QAC layers |
| `reinforce_rate` | float | 0.0-1.0 | 0.05 | Reinforcement rate for stabilization |
| `weight` | str | binary/gaussian | binary | Weight type for adjacency |
| `sigma` | float | >0 | None | Gaussian sigma (if weight=gaussian) |
| `engine` | str | real_qac/laplace | real_qac | QAC engine type |

### Parameter Tuning Guidelines

**For gene module analysis (8 modules)**:
```python
QACParams(
    k_nn=8,           # Match number of modules
    layers=50,        # Default is good
    reinforce_rate=0.05,  # Default is good
    engine="real_qac" # Use real quantum algorithm
)
```

**For single-cell analysis (high-dimensional)**:
```python
QACParams(
    k_nn=15,          # More neighbors for stability
    layers=100,       # More layers for complex data
    reinforce_rate=0.1,  # Higher reinforcement
    engine="real_qac"
)
```

**For clustering/grouping**:
```python
QACParams(
    k_nn=20,          # Higher k for grouping
    layers=50,
    weight="gaussian", # Smooth adjacency
    sigma=1.0,
    engine="real_qac"
)
```

---

## Rate Limiting

**Your institution has**: 1,000 requests per hour

### Monitoring Usage

```python
# Check before making many requests
usage = client.usage()
print(f"Remaining: {usage.remaining}")

# Check rate limit headers after any request
status = client.rate_limit_status
print(f"Limit: {status['limit']}")
print(f"Remaining: {status['remaining']}")
print(f"Resets at: {status['reset_time']}")
```

### Handling Rate Limits

The client automatically handles 429 (rate limit exceeded) responses:

```python
from don_research_client import RateLimitError

try:
    result = client.qac.fit(embedding=data, sync=True)
except RateLimitError as e:
    print(f"Rate limit hit. Retry after {e.retry_after} seconds")
    # Client will automatically retry with backoff
```

### Best Practices

1. **Use sync mode for small datasets** (<100 cells)
2. **Use async mode for large datasets** (>100 cells)
3. **Reuse trained models** instead of training repeatedly
4. **Batch your work** - don't make 1000 separate calls in a loop
5. **Monitor usage** - check `client.rate_limit_status` regularly

---

## Error Handling

### Common Errors

#### Authentication Error

```python
from don_research_client import AuthenticationError

try:
    client = DonResearchClient(token="invalid_token")
    result = client.health()
except AuthenticationError as e:
    print("Invalid token! Check TAMU_API_TOKEN environment variable")
```

#### Job Timeout

```python
from don_research_client import JobTimeoutError

try:
    result = client.qac.poll_until_complete(job_id, timeout=60)
except JobTimeoutError:
    # Job is still running, but timeout reached
    # Can check status later with client.qac.get_job(job_id)
    print("Job taking longer than expected")
```

#### Job Failed

```python
from don_research_client import JobFailedError

try:
    result = client.qac.poll_until_complete(job_id)
except JobFailedError as e:
    print(f"Job failed: {e}")
    # Check error message for details
```

### Debugging

Enable verbose logging:

```python
client = DonResearchClient(token="...", verbose=True)

# Now all API calls are logged with details
```

---

## Reproducing TAMU Discoveries

### Discovery 1: Gene Co-Expression Modules

```bash
# Run provided script
python tamu_gene_coexpression_qac.py

# Compare output with gene_coexpression_qac_results.json
```

**Key findings to verify**:
- Module coherence differs between low-α and high-α cells (p < 0.000001)
- Pearson correlation between stability and alpha
- Specific modules show quantum stability patterns

### Discovery 2: Quantum Collapse Creation

```bash
# Coming soon - conversion in progress
python tamu_quantum_collapse_creation.py
```

### Discovery 3: Memory is Structure

```bash
# Coming soon - conversion in progress
python tamu_memory_is_structure.py
```

---

## Working with Your Own Data

### Format Requirements

#### For QAC Analysis

```python
# Embedding must be 2D list/array
embedding = [
    [feature1, feature2, feature3],  # Sample 1
    [feature1, feature2, feature3],  # Sample 2
    ...
]

# Each sample should have same number of features
# Values should be normalized (e.g., 0-1 range or z-scores)
```

#### For Genomics Compression

```python
gene_names = ["GENE1", "GENE2", ...]  # List of gene identifiers
expression_matrix = [
    [expr_gene1, expr_gene2, ...],  # Cell 1
    [expr_gene1, expr_gene2, ...],  # Cell 2
]
# Rows = cells, columns = genes
```

### Example: Custom scRNA-seq Analysis

```python
import scanpy as sc
from don_research_client import DonResearchClient, QACParams

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Extract expression matrix for top variable genes
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata_subset = adata[:, adata.var.highly_variable]

# Convert to list format for API
gene_names = adata_subset.var_names.tolist()
expression_matrix = adata_subset.X.toarray().tolist()

# Initialize client
client = DonResearchClient()

# Option 1: DON-GPU compression
result = client.genomics.compress(
    gene_names=gene_names,
    expression_matrix=expression_matrix
)
compressed_vectors = result['compressed_vectors']

# Option 2: QAC stabilization
# (Use compressed vectors as embedding)
qac_result = client.qac.fit(
    embedding=compressed_vectors,
    params=QACParams(k_nn=15, layers=100),
    sync=True
)

print(f"Model trained: {qac_result['model_id']}")
```

---

## Advanced Usage

### Context Manager Pattern

```python
# Automatically closes connection when done
with DonResearchClient() as client:
    result = client.qac.fit(embedding=data, sync=True)
    # Use result...
# Connection closed here
```

### Custom Timeout and Retries

```python
client = DonResearchClient(
    token="your_token",
    timeout=60.0,      # 60 second timeout per request
    max_retries=5      # Retry up to 5 times on failure
)
```

### Multiple Models

```python
# Train model on training data
train_result = client.qac.fit(
    embedding=train_data,
    params=QACParams(k_nn=15),
    sync=True
)
model_id = train_result['model_id']

# Apply same model to test data
test_result = client.qac.apply(
    model_id=model_id,
    embedding=test_data,
    sync=True
)

# Apply same model to validation data
val_result = client.qac.apply(
    model_id=model_id,
    embedding=val_data,
    sync=True
)

# Get model metadata
model_info = client.qac.get_model(model_id)
print(f"Model trained on {model_info.n_cells} cells")
```

---

## Troubleshooting

### Issue: "No API token provided"

**Solution**: Set environment variable
```bash
export TAMU_API_TOKEN='your_token'
# Or in Python:
import os
os.environ['TAMU_API_TOKEN'] = 'your_token'
```

### Issue: "Connection error"

**Solution**: Check internet connection and API endpoint
```python
# Test connectivity
import requests
response = requests.get("https://don-research-api.onrender.com/api/v1/health")
print(response.status_code)  # Should be 200
```

### Issue: "Request timed out"

**Solution**: Increase timeout or use async mode
```python
# Increase timeout
client = DonResearchClient(timeout=120.0)

# Or use async mode for large jobs
job = client.qac.fit(embedding=large_data, sync=False)
result = client.qac.poll_until_complete(job.id, timeout=3600)
```

### Issue: "Rate limit exceeded"

**Solution**: Wait for reset or reduce request frequency
```python
usage = client.usage()
print(f"Resets at: {usage.reset_time}")

# Or space out requests
import time
for batch in data_batches:
    result = client.qac.fit(embedding=batch, sync=True)
    time.sleep(5)  # Wait 5 seconds between batches
```

### Issue: "Job failed" with error message

**Solution**: Check embedding format and parameters
```python
# Common issues:
# 1. Embedding has NaN or Inf values
embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0)

# 2. Embedding is empty or wrong shape
print(f"Shape: {np.array(embedding).shape}")  # Should be (n_samples, n_features)

# 3. Parameters out of range
params = QACParams(k_nn=15)  # Use defaults first, then customize
```

---

## API Reference

### DonResearchClient

**Constructor**: `DonResearchClient(token, base_url, timeout, max_retries, verbose)`

**Methods**:
- `health()` → Dict: Check API health status
- `usage()` → UsageInfo: Get current usage statistics
- `rate_limit_status` → Dict: Get rate limit info from last request
- `close()`: Close HTTP session

**Properties**:
- `qac`: QACClient instance for QAC operations
- `genomics`: GenomicsClient instance for genomics operations

### QACClient

**Methods**:
- `fit(embedding, params, seed, sync)` → QACJob | Dict
- `apply(model_id, embedding, seed, sync)` → QACJob | Dict
- `get_job(job_id)` → QACJob
- `poll_until_complete(job_id, poll_interval, timeout)` → Dict
- `get_model(model_id)` → QACModelMeta

### GenomicsClient

**Methods**:
- `compress(gene_names, expression_matrix, cell_metadata)` → Dict

---

## Support

### Contact Information

**For technical issues**:
- Email: research@donsystems.com
- Include: error message, code snippet, API token (first 8 characters only)

**For scientific questions**:
- Dr. James Cai (TAMU collaboration lead)
- DON Systems research team

### Useful Resources

- API Documentation: `docs/api/API_REFERENCE.md`
- TAMU Executive Summary: `TAMU_EXECUTIVE_SUMMARY.md`
- Discovery Papers: `*_results.json` files

---

## Appendix: API Response Schemas

### QAC Fit Response (sync=True)

```json
{
  "model_id": "qac_abc123",
  "n_cells": 100,
  "compression_ratio": 8.0,
  "algorithm": "real_qac",
  "params": {
    "k_nn": 15,
    "layers": 50,
    "reinforce_rate": 0.05
  }
}
```

### QAC Apply Response (sync=True)

```json
{
  "model_id": "qac_abc123",
  "stabilized_vectors": [[...], [...], ...],
  "coherence": 0.95,
  "n_cells": 50
}
```

### QAC Job Response (sync=False)

```json
{
  "id": "job_xyz789",
  "type": "fit",
  "status": "running",
  "progress": 0.5,
  "model_id": null,
  "result": null,
  "error": null,
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:00:01Z"
}
```

### Genomics Compress Response

```json
{
  "compressed_vectors": [[...], [...], ...],
  "compression_stats": {
    "original_dims": 2000,
    "compressed_dims": 64,
    "ratio": 31.25,
    "algorithm": "don_gpu_fractal"
  },
  "institution": "Texas A&M University"
}
```

---

**Document Version**: 1.0  
**Last Updated**: January 2025  
**For**: Texas A&M Collaboration Package  
**Copyright**: DON Systems LLC - All Rights Reserved
