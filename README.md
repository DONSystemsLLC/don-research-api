# DON Stack Research API

🧬 **Quantum-enhanced data processing for genomics research**

[![Deploy](https://img.shields.io/badge/Deploy-Render-brightgreen)](https://render.com)
[![API](https://img.shields.io/badge/API-FastAPI-blue)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Proprietary-red)](#license)

## Overview

The DON Stack Research API provides IP-protected access to quantum-enhanced computational biology algorithms for academic research collaboration. This service enables researchers to leverage the power of the Distributed Order Network (DON) for genomics data processing while protecting proprietary implementations.

### Key Features

🔬 **Genomics-Optimized Compression**
- Single-cell gene expression data compression using DON-GPU fractal clustering
- Achieves 4×-32× compression with minimal information loss
- NCBI GEO database compatibility

🧠 **RAG System Optimization** 
- Quantum-enhanced retrieval for genomics databases
- TACE temporal control for adaptive similarity thresholds
- Optimized for large-scale bioinformatics queries

⚛️ **Quantum State Stabilization**
- QAC (Quantum Adjacency Code) error correction
- Multi-layer adjacency stabilization
- Real-time coherence monitoring

## API Endpoints

### Authentication
All endpoints require bearer token authentication for authorized research institutions.

```bash
Authorization: Bearer <institution_token>
```

### Core Endpoints

#### 📊 Genomics Compression
```http
POST /api/v1/genomics/compress
```
Compress single-cell gene expression matrices using fractal clustering.

#### 🔍 RAG Optimization  
```http
POST /api/v1/genomics/rag-optimize
```
Optimize retrieval-augmented generation for genomics queries.

#### ⚛️ Quantum Stabilization
```http
POST /api/v1/quantum/stabilize
```
Apply quantum error correction to state vectors.

#### 📈 Usage Stats
```http
GET /api/v1/usage
```
Get current API usage statistics for your institution.

#### 🏥 Health Check
```http
GET /api/v1/health
```
Public endpoint for service health monitoring.

## Quick Start

### 1. Request Access
Contact **research@donsystems.com** with:
- Institution name and affiliation
- Research project description
- Principal investigator information
- Expected usage patterns

### 2. API Testing
```python
import requests

api_url = "https://your-deployment.onrender.com"
headers = {"Authorization": "Bearer your_institution_token"}

# Test connection
response = requests.get(f"{api_url}/api/v1/health")
print(response.json())

# Compress genomics data
data = {
    "data": {
        "gene_names": ["GENE1", "GENE2", "GENE3"],
        "expression_matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    },
    "compression_target": 8
}

response = requests.post(
    f"{api_url}/api/v1/genomics/compress",
    headers=headers,
    json=data
)
print(response.json())
```

### 3. Integration Examples
See `examples/` directory for:
- Single-cell RNA-seq compression workflows
- RAG system optimization for PubMed queries
- Quantum state analysis for molecular simulations

### Interactive Demo Launcher

Run the guided showcase from the project root:

```bash
python demos/demo_launcher.py
```

When you choose **Option 2 – Basic Genomics Compression Demo**, the launcher now prompts you to select which real PBMC cohort to use. The tiers map to verified datasets bundled with the repo:

| Tier  | Cells × Genes | Source File                                  |
|-------|---------------|----------------------------------------------|
| Small | 100 × 100     | `real_pbmc_small.json`                       |
| Medium| 250 × 500     | `real_pbmc_medium_correct.json`              |
| Large | 500 × 1000    | `real_pbmc_data.json`                        |

You can also bypass the prompt by setting `DON_BASIC_DEMO_DATASET` (`small`, `medium`, or `large`) or by passing the size directly to the demo module:

```bash
python demos/quick/basic_compression_demo.py medium
```

For a browser-friendly walkthrough aimed at collaborating labs, visit the new help tab once the API server is running: [http://localhost:8080/help](http://localhost:8080/help).

## Deployment

### Render.com (Recommended)

1. **Fork this repository**
2. **Connect to Render**:
   - New Web Service
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**:

    ```env
    PYTHON_VERSION=3.11
    PORT=8080
    ```

4. **Deploy**: Automatic deployment on git push

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

## Research Collaboration

### Texas A&M University - Cai Lab

**Principal Investigator**: Professor James J. Cai  
**Focus**: Quantum computing applications in single-cell biology  
**Contact**: [jcai@tamu.edu](mailto:jcai@tamu.edu)

### Collaboration Benefits

- Access to cutting-edge quantum-enhanced algorithms
- Reduced computational costs for large-scale genomics
- Joint research publication opportunities
- Technical support and algorithm customization

## Technical Specifications

### DON Stack Components

- **DON-GPU**: Fractal clustering with 4×-32× compression
- **QAC**: Quantum Adjacency Code error correction
- **TACE**: Temporal Adjacency Collapse Engine control

### Performance Metrics

- **Compression Ratio**: Up to 32× reduction
- **Processing Speed**: Real-time for datasets <10GB
- **Accuracy Preservation**: >95% fidelity
- **Quantum Coherence**: >95% stability

### Rate Limits

- **Academic Institutions**: 1,000 requests/hour
- **Demo Access**: 100 requests/hour
- **Enterprise**: Custom limits available

## Security & IP Protection

⚠️ **IMPORTANT**: This API provides access to patent-protected algorithms through a secure service layer. Direct access to DON Stack source code is not provided.

- 🔐 Token-based authentication
- 🏛️ Institution verification required
- 📊 Usage monitoring and rate limiting
- 🔒 Encrypted data transmission
- 📋 Audit logging for compliance

## Support

- 📧 **Research Inquiries**: [research@donsystems.com](mailto:research@donsystems.com)
- 🐛 **Technical Issues**: [support@donsystems.com](mailto:support@donsystems.com)
- 📖 **Documentation**: [API Docs](https://your-deployment.onrender.com/docs)
- 💬 **Collaboration**: [partnerships@donsystems.com](mailto:partnerships@donsystems.com)

## License

### DON Health Commons License (DHCL) v0.1 - Intent Draft

This software is licensed under the **DON Health Commons License (DHCL)**, which enables mission-aligned clinical and academic research while protecting the instrument from enclosure by actors whose incentives conflict with public health.

**Key Principles:**
- ✅ **Open for Mission-Aligned Entities (MAEs)**: Academic institutions, public hospitals, non-profits, and qualifying startups
- ✅ **Share Results, Protect Instruments**: Publish findings freely; keep algorithms auditable and in the Commons
- ✅ **Data Sovereignty**: Federated use patterns; raw PHI stays on-site
- ✅ **Reciprocity**: Contribute improvements back within 6 months
- ❌ **Prohibited Entities**: Companies on the Designated Exclusion List (> $50M pharma/biotech revenue)

**Full License**: See [LICENSE-DHCL-v0.1-intent.md](./LICENSE-DHCL-v0.1-intent.md)  
**Attribution Notice**: See [NOTICE](./NOTICE)  

**Status**: Intent draft for review and public comment. Not legal advice. Final terms subject to counsel review.

**Compliance Self-Attestation**: Required for MAE access. Contact research@donsystems.com for template.

### Patents Pending

- Fractal Clustering Algorithm (DON-GPU)
- Quantum Adjacency Code (QAC)
- Temporal Adjacency Collapse Engine (TACE)

**Patent Peace**: Licensees grant defensive patent licenses to other MAEs. Patent suits against the Licensed Work result in immediate license termination.

---

*© 2025 DON Systems LLC / Foundation. All rights reserved.*

**Human-readable summary (non-binding)**: Use the DON instrument to make medicine reproducible and portable—if you're aligned with public health. Share improvements, keep logs, protect patient data, and don't hand the instrument to companies that will lock it up. Share results with anyone; the instrument stays in the Commons.
