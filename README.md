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

## Architecture & Operational Behavior

### System Architecture

The DON Research API is a **quantum-classical hybrid system** that bridges academic research with proprietary quantum computing technology:

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Gateway                          │
│  Authentication • Rate Limiting • Audit Logging • Routing   │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌───────────────────┴───────────────────┐
        ↓                                       ↓
┌──────────────────┐                  ┌──────────────────┐
│  DON Stack       │                  │  Bio Module      │
│  Adapter         │                  │  (AnnData)       │
├──────────────────┤                  ├──────────────────┤
│ • DON-GPU        │                  │ • H5AD Export    │
│ • QAC Engine     │                  │ • Signal Sync    │
│ • TACE Control   │                  │ • QC Workflows   │
└──────────────────┘                  └──────────────────┘
        ↓                                       ↓
┌──────────────────────────────────────────────────────────────┐
│           Quantum-Classical Processing Layer                  │
│  Fractal Clustering • Error Correction • Vector Indexing     │
└──────────────────────────────────────────────────────────────┘
```

### DON Stack Integration Modes

The system operates in **dual-mode architecture** for maximum flexibility:

#### Internal Mode (Default - Production)
```python
# Direct Python calls to stack/ modules
DON_STACK_MODE=internal  # Default
```
- **Advantages**: Lower latency (~50ms saved), simpler deployment, no microservice overhead
- **Use Case**: Production deployments, academic research, Render.com hosting
- **Components**: DON-GPU (`stack/don_gpu/core.py`), QAC (`stack/tace/core.py`), TACE

#### HTTP Mode (Microservices)
```python
# Microservices via separate DON-GPU and TACE servers
DON_STACK_MODE=http
DON_GPU_ENDPOINT=http://127.0.0.1:8001
TACE_ENDPOINT=http://127.0.0.1:8002
```
- **Advantages**: Horizontal scaling, independent deployment, resource isolation
- **Use Case**: Enterprise deployments, high-throughput workloads, distributed systems
- **Components**: DON-GPU service (port 8001), TACE service (port 8002)

**Switching Modes**: Set `DON_STACK_MODE` environment variable (requires service restart)

### Health Monitoring

The `/api/v1/health` endpoint provides comprehensive system status:

```json
{
  "status": "healthy",
  "timestamp": "2025-10-26T10:30:00Z",
  "don_stack": {
    "mode": "internal",          // or "http" for microservices
    "adapter_loaded": true,
    "version": "1.2.0"
  },
  "qac": {
    "supported_engines": ["adjacency_v1", "adjacency_v2", "surface_code"],
    "default_engine": "adjacency_v2",
    "real_engine_available": true
  },
  "services": {
    "compression": "operational",
    "vector_search": "operational",
    "bio_export": "operational"
  }
}
```

**Monitoring Strategy**:
- **Production**: Poll `/health` every 60 seconds
- **Critical Systems**: Alert on `status != "healthy"` or `adapter_loaded: false`
- **Performance**: Track `don_stack.mode` to verify expected configuration

### Data Retention & Cleanup

**Automatic Retention Policies** (enforced by scheduled cleanup jobs):

| Data Type | Retention | Auto-Cleanup | Override |
|-----------|-----------|--------------|----------|
| Input gene expression matrices | 24 hours | ✅ Automatic | Contact support |
| Compressed vectors (temp) | 24 hours | ✅ Automatic | Use `project_id` |
| Vector databases (FAISS) | 7 days | ✅ Automatic | Rebuild on demand |
| Audit logs (`trace_id`) | 90 days | ✅ Archival | Extended for projects |
| QAC models | 30 days | ✅ Automatic | Export before expiry |
| Async job artifacts | 48 hours | ✅ Automatic | Download promptly |

**Cleanup Scheduler**:
- Runs hourly via background task (`@app.on_event("startup")`)
- Logs cleanup operations to audit trail
- No user intervention required
- Extended retention available for active research projects

**Best Practices**:
1. Download results within 24 hours for temporary operations
2. Use `project_id` to group related operations for longer retention
3. Export QAC models before 30-day expiration
4. Request extended retention for multi-month projects

### Rate Limiting & Authentication

**Per-Institution Hourly Limits**:

```python
# Configured in src/auth/authorized_institutions.py
AUTHORIZED_INSTITUTIONS = {
    "demo_token": {
        "name": "Demo Access",
        "rate_limit": 100  # requests/hour
    },
    "institution_token": {
        "name": "Academic Institution",
        "rate_limit": 1000  # requests/hour
    }
}
```

**Rate Limit Headers** (all responses):
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 995
X-RateLimit-Reset: 1730044800  # Unix timestamp
```

**Rate Limit Exceeded (429 Response)**:
```json
{
  "detail": "Rate limit exceeded: 1000/hour for Institution. Reset at 2025-10-26T14:00:00Z"
}
```

**Usage Tracking**:
- Resets every hour (UTC time boundaries)
- Tracked in-memory (no persistent storage)
- Check usage via `GET /api/v1/usage`

**Authentication Flow**:
1. Institution requests access → `research@donsystems.com`
2. DON Systems issues bearer token (256-bit random)
3. Token added to `AUTHORIZED_INSTITUTIONS` config
4. All API requests include: `Authorization: Bearer <token>`
5. Failed auth returns 401 with error message

### Audit Logging & Traceability

**Trace ID System** (every operation generates unique identifier):

**Format**: `{institution}_{date}_{operation}_{uuid}`  
**Example**: `tamu_20251026_compress_abc123xyz`

**Logged Metadata** (no actual data):
- Timestamp (ISO 8601 UTC)
- Institution name
- Endpoint called
- Operation type (compress, search, export, etc.)
- Input parameters (dimensions, seed, project_id)
- Output metrics (compression ratio, runtime, errors)

**Memory Endpoint** (retrieve traces):
```bash
GET /api/v1/bio/memory/{project_id}
```

**Use Cases**:
- Reproducibility: Track exact parameters for published results
- Debugging: Trace errors through multi-step workflows
- Compliance: Audit trail for institutional review
- Collaboration: Share trace IDs with co-investigators

### Deployment Status

**Current Deployment**: Render.com (US-East)  
**Service Type**: Web Service (Docker container)  
**Health Endpoint**: `https://don-research-api.onrender.com/api/v1/health`  
**Interactive Docs**: `https://don-research-api.onrender.com/docs`

**Production Configuration**:
```bash
# Environment Variables (Render.com)
PYTHON_VERSION=3.11
PORT=8080
DON_STACK_MODE=internal
DON_AUTHORIZED_INSTITUTIONS_JSON=<encrypted_json>
```

**Build Process**:
1. Git push to `main` branch
2. Render.com detects changes
3. `pip install -r requirements.txt` (2-3 minutes)
4. `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Health check passes → live traffic

**Zero-Downtime Deployments**:
- Render.com blue-green deployment pattern
- Old containers remain live until new containers healthy
- Automatic rollback on health check failure

**Monitoring & Alerts**:
- Uptime monitoring via Render.com dashboard
- Email alerts for service failures
- Performance metrics (response time, memory usage)
- Error rate tracking (500/429 responses)

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

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### 📘 [API Reference](docs/API_REFERENCE.md)
Complete endpoint documentation with request/response schemas, authentication, rate limiting, error handling, and performance characteristics.

**Coverage**: 22 endpoints across 4 modules (Main API, Bio Module, QAC, Genomics Router)  
**Validation**: 42/42 automated tests passing (100% test coverage)  
**Examples**: Python + cURL for every endpoint

### 🎓 [Texas A&M Integration Guide](docs/TAMU_INTEGRATION.md)
Step-by-step guide for single-cell genomics research using Scanpy pipelines.

**Target Audience**: Graduate students and postdocs in genomics laboratories  
**Coverage**: Complete Scanpy pipeline, cell type discovery, QC workflows, evolution tracking  
**Validation**: 15/15 automated tests passing  
**Includes**: Distance interpretation tables, contamination scoring, reproducibility best practices

### 🔒 [Data Policy & Security](docs/DATA_POLICY.md)
Comprehensive data handling, privacy, security, and IP protection policies.

**Coverage**: Data ownership, retention (24h-90d), GDPR/HIPAA compliance, IP protection  
**Sections**: 12 major sections + 3 appendices  
**Includes**: Patent status, trade secrets, acceptable use policy, incident response process

### 🚀 [Render.com Deployment Guide](docs/RENDER_DEPLOYMENT.md)
Production deployment guide for Render.com hosting platform.

**Coverage**: Environment variables, build configuration, health checks, monitoring  
**Status**: Coming soon

### Interactive Documentation

**OpenAPI/Swagger UI**: `https://your-deployment.onrender.com/docs`  
**HTML Help Page**: `https://your-deployment.onrender.com/help`  

## Support

- 📧 **Research Inquiries**: [research@donsystems.com](mailto:research@donsystems.com)
- 🐛 **Technical Issues**: [support@donsystems.com](mailto:support@donsystems.com)
- 📖 **Documentation**: See `docs/` directory or [API Docs](https://your-deployment.onrender.com/docs)
- 💬 **Collaboration**: [partnerships@donsystems.com](mailto:partnerships@donsystems.com)
- 🔐 **Security Issues**: [security@donsystems.com](mailto:security@donsystems.com)
- ⚖️ **Compliance & Legal**: [compliance@donsystems.com](mailto:compliance@donsystems.com)

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
