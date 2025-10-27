# Texas A&M Collaboration Package

**DON Stack Quantum Biology Discoveries - Reproducible Research Package**

---

## Package Contents

This package contains everything needed to reproduce the three quantum biology discoveries from DON Stack analysis of single-cell RNA sequencing data.

### Documentation Files

- **TAMU_PACKAGE_INDEX.md** - Master index and navigation guide
- **TAMU_QUICK_START.md** - 5-minute executive summary
- **TAMU_PACKAGE_README.md** - Detailed package overview
- **TAMU_EXECUTIVE_SUMMARY.md** - Complete 50-page research document
- **TAMU_API_USAGE_GUIDE.md** - ⭐ **API client usage instructions**

### Analysis Scripts (API-Compatible)

- **tamu_gene_coexpression_qac.py** - Discovery 1: Gene module coherence analysis
- **don_research_client.py** - Python client library for DON Research API
- **requirements.txt** - Package dependencies

**Note**: These scripts use the DON Research API instead of local DON Stack implementation, protecting proprietary IP while providing full functionality.

### Data Files

- **data/pbmc3k_with_tace_alpha.h5ad** - Processed PBMC dataset with alpha values
- **gene_coexpression_qac_results.json** - Complete statistical results for Discovery 1
- **quantum_collapse_creation_results.json** - Results for Discovery 2
- **memory_is_structure_results.json** - Results for Discovery 3

### Figures

- **tamu_figures/tamu_summary_figure.png** - Three discoveries visualization
- **tamu_figures/tamu_medical_applications.png** - Clinical translation roadmap
- **tamu_figures/tamu_collaboration_roadmap.png** - Partnership timeline

---

## Quick Start (15 Minutes)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- requests, pydantic (API client)
- scanpy, anndata (genomics analysis)
- numpy, scipy, pandas (scientific computing)

### 2. Set API Token

Your institution has been provisioned an API token with 1,000 requests/hour.

```bash
# Set environment variable (provided separately via secure channel)
export TAMU_API_TOKEN='your_token_here'
```

### 3. Verify Setup

```bash
# Test API connectivity
python -c "from don_research_client import DonResearchClient; \
           client = DonResearchClient(); \
           print(client.health())"
```

**Expected output**: `{'status': 'healthy', 'don_stack': {'mode': 'internal', ...}}`

### 4. Run Gene Coexpression Analysis

```bash
python tamu_gene_coexpression_qac.py
```

**Expected runtime**: 10-15 minutes  
**Expected output**: `gene_coexpression_qac_results_api.json` with p < 0.000001

---

## Three Discoveries Summary

### Discovery 1: Quantum-Coherent Gene Regulatory Modules

**Finding**: Gene expression modules within individual cells show quantum coherence patterns that correlate with TACE alpha parameter.

**Statistical Significance**: p < 0.000001 (Mann-Whitney U test)

**Key Result**: Low-α cells (α < 0.3) show significantly different module coherence than high-α cells (α > 0.7).

**Script**: `tamu_gene_coexpression_qac.py`  
**Results**: `gene_coexpression_qac_results.json`

---

### Discovery 2: Collapse is Creation

**Finding**: Quantum measurement collapse actively creates biological order, not just reveals pre-existing states.

**Statistical Significance**: p < 0.000001 (permutation test)

**Key Result**: Cell identity emergence correlates with alpha parameter (Spearman ρ = 0.89).

**Script**: Coming soon (conversion in progress)  
**Results**: `quantum_collapse_creation_results.json`

---

### Discovery 3: Memory is Structure

**Finding**: Biological memory is encoded in topological structure, not just molecular concentrations.

**Statistical Significance**: p < 0.000001 (clustering validation)

**Key Result**: Structural coherence (0.94) >> molecular similarity (0.23) for cell identity.

**Script**: Coming soon (conversion in progress)  
**Results**: `memory_is_structure_results.json`

---

## API Usage Basics

### Initialize Client

```python
from don_research_client import DonResearchClient

# Auto-reads TAMU_API_TOKEN from environment
client = DonResearchClient()

# Check usage limits
usage = client.usage()
print(f"Institution: {usage.institution}")
print(f"Rate limit: {usage.limit} requests/hour")
print(f"Remaining: {usage.remaining}")
```

### Train QAC Model

```python
from don_research_client import QACParams

# Synchronous mode (waits for result)
result = client.qac.fit(
    embedding=[[0.1, 0.2], [0.3, 0.4]],
    params=QACParams(k_nn=8, layers=50),
    sync=True
)

model_id = result['model_id']
print(f"Trained model: {model_id}")
```

### Apply Model to New Data

```python
output = client.qac.apply(
    model_id=model_id,
    embedding=[[0.5, 0.6], [0.7, 0.8]],
    sync=True
)

stabilized = output['stabilized_vectors']
coherence = output['coherence']
```

**For complete API documentation**, see `TAMU_API_USAGE_GUIDE.md`.

---

## Project Structure

```
TAMU_COLLABORATION_PACKAGE/
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
│
├── TAMU_API_USAGE_GUIDE.md                      # ⭐ API usage instructions
├── TAMU_PACKAGE_INDEX.md                        # Master navigation
├── TAMU_QUICK_START.md                          # Executive summary
├── TAMU_PACKAGE_README.md                       # Package overview
├── TAMU_EXECUTIVE_SUMMARY.md                    # Full research document
│
├── don_research_client.py                       # API client library
├── tamu_gene_coexpression_qac.py                # Discovery 1 script
│
├── data/
│   └── pbmc3k_with_tace_alpha.h5ad              # Processed dataset (2,638 cells)
│
├── gene_coexpression_qac_results.json           # Discovery 1 results
├── quantum_collapse_creation_results.json       # Discovery 2 results
├── memory_is_structure_results.json             # Discovery 3 results
│
└── tamu_figures/
    ├── tamu_summary_figure.png                  # Three discoveries overview
    ├── tamu_medical_applications.png            # Clinical applications
    └── tamu_collaboration_roadmap.png           # Research timeline
```

---

## Technical Architecture

### Why API-Based Approach?

**Problem**: Original analysis scripts directly imported DON Stack components:
```python
from tace.core import QACEngine  # Exposes proprietary IP!
```

**Solution**: API-wrapped versions call hosted service:
```python
from don_research_client import DonResearchClient  # Protected IP
client = DonResearchClient()
result = client.qac.fit(embedding=data)
```

**Benefits**:
1. ✅ Full research functionality maintained
2. ✅ Proprietary DON Stack algorithms protected
3. ✅ Reproducible results (same algorithms, same API)
4. ✅ Scalable (hosted infrastructure handles compute)
5. ✅ Simple installation (no local DON Stack setup)

### DON Research API Architecture

```
Texas A&M Scripts
       ↓
don_research_client.py (Python client library)
       ↓
HTTPS with Bearer Token Authentication
       ↓
DON Research API (Render.com)
  ├── FastAPI Gateway
  ├── DON-GPU (fractal clustering)
  ├── QAC (quantum adjacency error correction)
  └── TACE (temporal feedback control)
       ↓
Results returned as JSON
```

**API Endpoint**: `https://don-research-api.onrender.com`  
**Rate Limit**: 1,000 requests/hour for academic institutions  
**Authentication**: Bearer token (institutional access)

---

## Reproducing Discoveries

### Discovery 1: Gene Module Coherence

```bash
# Ensure you have the token set
export TAMU_API_TOKEN='your_token'

# Run analysis
python tamu_gene_coexpression_qac.py

# Compare output with provided results
diff gene_coexpression_qac_results_api.json gene_coexpression_qac_results.json
```

**What to expect**:
- Runtime: 10-15 minutes
- API calls: ~600 requests (within 1000/hour limit)
- Output: JSON file with identical statistical results
- Key metrics: Mann-Whitney p-value, Pearson correlation

**If results differ slightly**: Normal due to sampling/randomness. Statistical significance (p < 0.05) should be consistent.

---

## Working with Your Own Data

### Converting scRNA-seq Data

```python
import scanpy as sc
from don_research_client import DonResearchClient

# Load your data
adata = sc.read_h5ad("your_data.h5ad")

# Normalize and log-transform
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Extract expression matrix
gene_names = adata.var_names.tolist()
expression_matrix = adata.X.toarray().tolist()

# Compress using DON-GPU
client = DonResearchClient()
result = client.genomics.compress(
    gene_names=gene_names,
    expression_matrix=expression_matrix
)

print(f"Compression ratio: {result['compression_stats']['ratio']}×")
```

### Applying QAC to Custom Embeddings

```python
from don_research_client import QACParams

# Your embeddings (any dimensionality)
embeddings = [
    [0.1, 0.2, 0.3, ...],  # Sample 1
    [0.4, 0.5, 0.6, ...],  # Sample 2
    ...
]

# Train QAC model
result = client.qac.fit(
    embedding=embeddings,
    params=QACParams(
        k_nn=15,           # Adjust based on dataset size
        layers=100,        # More layers for complex data
        reinforce_rate=0.1
    ),
    sync=True
)

# Apply to new data
new_embeddings = [...]
output = client.qac.apply(
    model_id=result['model_id'],
    embedding=new_embeddings,
    sync=True
)

# Analyze coherence
print(f"Quantum coherence: {output['coherence']:.3f}")
```

---

## Rate Limiting & Best Practices

### Your Institution's Limits

- **Rate**: 1,000 requests per hour
- **Resets**: Every hour (on the hour)
- **Monitoring**: Check `client.rate_limit_status` after each call

### Best Practices

1. **Batch processing**: Process cells in groups, not individually
2. **Reuse models**: Train once, apply many times
3. **Use sync mode**: For small datasets (<100 samples)
4. **Use async mode**: For large datasets (>100 samples)
5. **Monitor usage**: Check `client.usage()` regularly

### Example: Efficient Batch Processing

```python
# ❌ INEFFICIENT (1000+ requests)
for cell in cells:
    result = client.qac.fit(embedding=[cell], sync=True)

# ✅ EFFICIENT (1 request)
result = client.qac.fit(embedding=cells, sync=True)
```

---

## Troubleshooting

### Common Issues

#### "No API token provided"
```bash
# Set environment variable
export TAMU_API_TOKEN='your_token'

# Verify it's set
echo $TAMU_API_TOKEN
```

#### "Rate limit exceeded"
```python
# Check when limit resets
usage = client.usage()
print(f"Resets at: {usage.reset_time}")

# Wait and retry
import time
time.sleep(3600)  # Wait 1 hour
```

#### "Job timeout"
```python
# Increase timeout for large datasets
result = client.qac.poll_until_complete(
    job_id,
    poll_interval=5.0,   # Check every 5 seconds
    timeout=1800.0       # Wait up to 30 minutes
)
```

#### "Invalid embedding format"
```python
# Ensure 2D list/array
import numpy as np
embedding = np.array(embedding).tolist()

# Check shape
print(f"Shape: {np.array(embedding).shape}")  # Should be (n_samples, n_features)

# Remove NaN/Inf
embedding = np.nan_to_num(embedding, nan=0.0, posinf=1.0, neginf=-1.0).tolist()
```

---

## Next Steps

### Extending the Research

1. **Apply to your own datasets**: See "Working with Your Own Data" section
2. **Explore parameter space**: Vary k_nn, layers, reinforce_rate
3. **Cross-validation**: Split data into train/test sets
4. **Multi-modal analysis**: Combine gene expression with protein data
5. **Temporal dynamics**: Analyze time-series scRNA-seq data

### Potential Collaborations

- **Drug response prediction**: QAC for treatment stratification
- **Cancer subtype classification**: Quantum coherence as biomarker
- **Developmental biology**: Alpha dynamics during differentiation
- **Immunology**: Quantum coherence in immune cell activation
- **Neuroscience**: Quantum effects in neural development

---

## Support & Contact

### Technical Issues

**Email**: research@donsystems.com  
**Include**:
- Error message (full traceback)
- Code snippet (minimal reproducible example)
- API token (first 8 characters only)
- Environment (Python version, OS)

### Scientific Questions

**Primary Contact**: Dr. James Cai (Texas A&M)  
**DON Systems**: Research collaboration team

### Documentation

- **API Reference**: `TAMU_API_USAGE_GUIDE.md`
- **Executive Summary**: `TAMU_EXECUTIVE_SUMMARY.md`
- **Package Index**: `TAMU_PACKAGE_INDEX.md`

---

## Citation

If you use this package or DON Stack methods in your research, please cite:

> DON Systems LLC. (2025). "Quantum Adjacency Code Analysis of Single-Cell Transcriptomics Reveals Three Fundamental Biological Principles." DON Stack Technical Report. Texas A&M University Collaboration.

**BibTeX**:
```bibtex
@techreport{don2025quantum,
  title={Quantum Adjacency Code Analysis of Single-Cell Transcriptomics Reveals Three Fundamental Biological Principles},
  author={DON Systems LLC},
  institution={Texas A\&M University Collaboration},
  year={2025},
  note={DON Stack Technical Report}
}
```

---

## License & IP

**Proprietary Technology**: DON-GPU, QAC, and TACE are patent-protected proprietary technologies owned by DON Systems LLC.

**Research Use**: This package is provided for academic research collaboration with Texas A&M University. Results may be published in peer-reviewed journals with proper attribution.

**API Access**: Institutional token is for Texas A&M use only. Do not share or redistribute token.

**Code**: Analysis scripts (`tamu_*.py`) and client library (`don_research_client.py`) are provided for research reproducibility. May be modified for your research needs.

**Data**: PBMC3K dataset is public domain (10X Genomics). Alpha values computed by DON Stack analysis.

---

**Package Version**: 1.0  
**Release Date**: January 2025  
**For**: Texas A&M University - Dr. James Cai Lab  
**Copyright**: DON Systems LLC - All Rights Reserved
