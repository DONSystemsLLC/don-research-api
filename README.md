# DON Stack Research API

🧬 **Quantum-enhanced data processing for genomics research**

[![Deploy](https://img.shields.io/badge/Deploy-Render-brightgreen)](https://render.com)
[![API](https://img.shields.io/badge/API-FastAPI-blue)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-Proprietary-red)](LICENSE)

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

## Deployment

### Render.com (Recommended)

1. **Fork this repository**
2. **Connect to Render**:
   - New Web Service
   - Connect your GitHub repository
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`

3. **Environment Variables**:
   ```
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
**Contact**: jcai@tamu.edu

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

- 📧 **Research Inquiries**: research@donsystems.com
- 🐛 **Technical Issues**: support@donsystems.com
- 📖 **Documentation**: [API Docs](https://your-deployment.onrender.com/docs)
- 💬 **Collaboration**: partnerships@donsystems.com

## License

**Proprietary Software - DON Systems LLC**

This software contains patent-protected algorithms and is provided under restricted license for authorized research institutions only. Distribution, modification, or commercial use is strictly prohibited without explicit written permission.

**Patents Pending**:
- Fractal Clustering Algorithm (DON-GPU)
- Quantum Adjacency Code (QAC)
- Temporal Adjacency Collapse Engine (TACE)

---

*© 2024 DON Systems LLC. All rights reserved.*