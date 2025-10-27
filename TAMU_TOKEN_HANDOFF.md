# Texas A&M University API Access - Secure Token Handoff

**Recipient**: Professor James J. Cai  
**Institution**: Texas A&M University - Cai Lab  
**Contact**: jcai@tamu.edu  
**Date**: October 27, 2025  
**Classification**: CONFIDENTIAL - Do Not Share

---

## API Token

Your unique API token for the DON Stack Research API has been provisioned:

```
tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc
```

**⚠️ SECURITY NOTICE:**
- This token is unique to Texas A&M University - Cai Lab
- Never commit this token to Git repositories
- Never share this token publicly or with unauthorized personnel
- Store in secure environment variables or secrets management system
- Report immediately if compromised: research@donsystems.com

---

## Account Details

| Parameter | Value |
|-----------|-------|
| **Institution** | Texas A&M University - Cai Lab |
| **Tier** | Academic |
| **Rate Limit** | 1,000 requests/hour |
| **API Base URL** | https://don-research.onrender.com |
| **API Version** | v1 |
| **Status** | ✅ Active |

---

## Quick Start

### 1. Test Your Token

```bash
# Test health endpoint (no auth required)
curl https://don-research.onrender.com/api/v1/health

# Test authenticated endpoint
curl -X POST https://don-research.onrender.com/api/v1/quantum/qac/fit \
  -H "Authorization: Bearer tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [[0.5, 0.5, 0.5, 0.5], [0.6, 0.4, 0.5, 0.5]],
    "params": {"k_nn": 3, "layers": 10},
    "sync": true
  }'
```

### 2. Set Up Environment

```bash
# Store token securely in environment variable
export DON_API_TOKEN="tamu_cai_lab_2025_HkRs17sgvbjnQax2KzD1iqYcHWbAs5xvZZ2ApKptWuc"

# Test with environment variable
curl -X POST https://don-research.onrender.com/api/v1/quantum/qac/fit \
  -H "Authorization: Bearer $DON_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"embedding": [[0.5, 0.5, 0.5, 0.5]], "sync": true}'
```

### 3. Python Example

```python
import os
import requests

# Load token from environment
TOKEN = os.environ.get("DON_API_TOKEN")
API_BASE_URL = "https://don-research.onrender.com"

# Set up headers
headers = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json"
}

# Example: QAC model fitting
response = requests.post(
    f"{API_BASE_URL}/api/v1/quantum/qac/fit",
    headers=headers,
    json={
        "embedding": [
            [0.5, 0.5, 0.5, 0.5],
            [0.6, 0.4, 0.5, 0.5],
            [0.4, 0.6, 0.5, 0.5]
        ],
        "params": {
            "k_nn": 15,
            "weight": "binary",
            "layers": 50,
            "engine": "real_qac"
        },
        "sync": True
    }
)

if response.status_code == 200:
    result = response.json()
    print(f"Model ID: {result['model_id']}")
    print(f"Status: {result['status']}")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

---

## Available Endpoints

### Genomics Processing

#### Build Vectors from .h5ad Files
```bash
curl -X POST https://don-research.onrender.com/api/v1/genomics/vectors/build \
  -H "Authorization: Bearer $DON_API_TOKEN" \
  -F "file=@your_dataset.h5ad" \
  -F "mode=cluster"
```

**Response:**
```json
{
  "ok": true,
  "mode": "cluster",
  "jsonl": "./data/vectors/your_dataset.cluster.jsonl",
  "count": 156,
  "preview": [...]
}
```

#### Search Vectors
```bash
curl -X POST https://don-research.onrender.com/api/v1/genomics/vectors/search \
  -H "Authorization: Bearer $DON_API_TOKEN" \
  -F "jsonl_path=./data/vectors/your_dataset.cluster.jsonl" \
  -F "k=10" \
  -F "psi=[0.5, 0.3, 0.2, ...]"
```

#### Generate Entropy Maps
```bash
curl -X POST https://don-research.onrender.com/api/v1/genomics/entropy-map \
  -H "Authorization: Bearer $DON_API_TOKEN" \
  -F "file=@your_dataset.h5ad" \
  -F "label_key=cell_type"
```

### Quantum QAC Endpoints

#### Fit QAC Model
```bash
curl -X POST https://don-research.onrender.com/api/v1/quantum/qac/fit \
  -H "Authorization: Bearer $DON_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "embedding": [[0.5, 0.5, 0.5, 0.5], ...],
    "params": {
      "k_nn": 15,
      "weight": "binary",
      "reinforce_rate": 0.05,
      "layers": 50,
      "engine": "real_qac"
    },
    "sync": true
  }'
```

**Response:**
```json
{
  "status": "succeeded",
  "model_id": "abc123-def456-...",
  "meta": {
    "n_cells": 1000,
    "k_nn": 15,
    "layers": 50,
    ...
  }
}
```

#### Apply QAC Model
```bash
curl -X POST https://don-research.onrender.com/api/v1/quantum/qac/apply \
  -H "Authorization: Bearer $DON_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "abc123-def456-...",
    "embedding": [[0.5, 0.5, 0.5, 0.5], ...],
    "sync": true
  }'
```

**Response:**
```json
{
  "status": "succeeded",
  "corrected": [[0.51, 0.49, 0.50, 0.50], ...],
  "stats": {
    "error_reduction": 0.23,
    "layers_applied": 50
  }
}
```

---

## Rate Limiting

Your academic tier includes:
- **1,000 requests per hour**
- Sliding window enforcement
- Rate limit headers in responses:
  - `X-RateLimit-Limit`: Your limit
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Window reset time

If you exceed the rate limit, you'll receive:
```json
{
  "detail": "Rate limit exceeded for institution: Texas A&M University - Cai Lab",
  "retry_after": 3600
}
```

---

## Request Tracing

All responses include a unique trace ID for debugging:
```
X-Trace-ID: tamu_20251027_abc123xyz
```

Include this trace ID when reporting issues for faster resolution.

---

## Best Practices

### 1. Token Management
```python
# ✅ GOOD: Load from environment
TOKEN = os.environ.get("DON_API_TOKEN")

# ❌ BAD: Hard-code in source
TOKEN = "tamu_cai_lab_2025_..."  # Never do this!
```

### 2. Error Handling
```python
try:
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 401:
        print("Authentication failed - check token")
    elif e.response.status_code == 429:
        print("Rate limit exceeded - wait before retry")
    else:
        print(f"HTTP error: {e}")
except Exception as e:
    print(f"Request failed: {e}")
```

### 3. Async vs Sync Execution

**Synchronous (blocks until complete):**
```python
response = requests.post(url, json={..., "sync": True})
result = response.json()  # Result available immediately
```

**Asynchronous (background job):**
```python
response = requests.post(url, json={..., "sync": False})
job = response.json()  # {"status": "queued", "job_id": "..."}

# Poll for completion
job_response = requests.get(f"{url}/jobs/{job['job_id']}")
```

---

## Support & Documentation

### Technical Support
- **Email**: research@donsystems.com
- **Subject Line**: Include `[TAMU]` for priority routing
- **Response Time**: Within 24 hours for technical issues

### Documentation
- **Full API Reference**: See `docs/API_REFERENCE.md` in repository
- **Integration Guide**: See `TEXAS_AM_LAB_GUIDE.md`
- **TAMU-Specific Guide**: See `docs/TAMU_INTEGRATION.md`

### Emergency Contact
For critical issues affecting production research:
- **Subject**: `[TAMU-URGENT] Brief description`
- **Include**: Trace ID, timestamp, and error details

---

## Production Validation Results

Your token has been validated in production:

| Test | Status | Details |
|------|--------|---------|
| Health Check | ✅ PASSED | Service operational |
| QAC Fit | ✅ PASSED | Model training successful |
| QAC Apply | ✅ PASSED | Error correction working |
| Invalid Token | ✅ PASSED | Security validation working |
| Rate Limiting | ✅ PASSED | 1000 req/hour configured |

**Validation Date**: October 27, 2025  
**API Version**: v1  
**Test Environment**: Production (https://don-research.onrender.com)

---

## Token Rotation

If you need to rotate your token (security best practice or compromise):

1. Contact: research@donsystems.com
2. Subject: `[TAMU] Token Rotation Request`
3. We will generate a new token within 24 hours
4. Old token will remain valid for 7 days transition period

---

## Usage Monitoring

We provide monthly usage reports including:
- Request volume and patterns
- Endpoint usage breakdown
- Performance metrics
- Rate limit utilization

Request your first report: research@donsystems.com

---

## Academic Collaboration

We welcome feedback on:
- Feature requests for genomics/quantum workflows
- Performance optimization suggestions
- Integration with your research tools
- Collaborative research opportunities

Contact Professor Donnie Van Mêtre: donnievanmetre@gmail.com

---

## Compliance & Data Policy

- **Data Retention**: 7 days for artifacts, 90 days for audit logs
- **No PII Storage**: Only genomics data and compute metrics
- **HIPAA Compliance**: Not applicable (no patient data)
- **Export Controls**: Algorithm IP protected, results exportable
- **Academic Use**: Approved for non-commercial research

Full policy: `docs/DATA_POLICY.md`

---

## Getting Started Checklist

- [ ] Store token in secure environment variable
- [ ] Test token with health endpoint
- [ ] Review `TEXAS_AM_LAB_GUIDE.md` documentation
- [ ] Run example QAC fit and apply workflow
- [ ] Upload test .h5ad file for genomics processing
- [ ] Set up error handling and retry logic
- [ ] Configure rate limit monitoring
- [ ] Save trace IDs for debugging
- [ ] Contact support with any questions

---

**Document Version**: 1.0  
**Last Updated**: October 27, 2025  
**Classification**: CONFIDENTIAL  
**Distribution**: Professor James J. Cai (jcai@tamu.edu) ONLY

---

*This token and documentation are provided under the terms of the research collaboration agreement between DON Systems LLC and Texas A&M University. Unauthorized use, reproduction, or distribution is prohibited.*

**© 2025 DON Systems LLC. All rights reserved.**
