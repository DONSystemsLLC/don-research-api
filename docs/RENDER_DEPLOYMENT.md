# DON Research API - Render.com Deployment Guide

**Target Platform**: Render.com  
**Service Type**: Web Service (Docker container)  
**Last Updated**: October 26, 2025  
**Guide Version**: 1.0

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Initial Deployment Setup](#initial-deployment-setup)
3. [Environment Configuration](#environment-configuration)
4. [Build & Start Commands](#build--start-commands)
5. [Health Check Configuration](#health-check-configuration)
6. [Scaling & Performance](#scaling--performance)
7. [Monitoring & Alerts](#monitoring--alerts)
8. [Troubleshooting](#troubleshooting)
9. [Security Hardening](#security-hardening)
10. [Cost Optimization](#cost-optimization)

---

## Prerequisites

### Repository Requirements

- ✅ Git repository hosted on GitHub (or GitLab/Bitbucket)
- ✅ `requirements.txt` in repository root
- ✅ `main.py` with FastAPI application
- ✅ Python 3.11+ compatibility

### Render.com Account

1. **Sign Up**: [https://render.com](https://render.com)
2. **Payment Method**: Credit card required (even for free tier)
3. **GitHub Integration**: Connect your GitHub account

### Local Testing

Before deploying, verify local functionality:

```bash
# Install dependencies
pip install -r requirements.txt

# Test server locally
uvicorn main:app --host 0.0.0.0 --port 8080

# Verify health endpoint
curl http://localhost:8080/api/v1/health
```

**Expected Response**:
```json
{
  "status": "healthy",
  "don_stack": {
    "mode": "internal",
    "adapter_loaded": true
  }
}
```

---

## Initial Deployment Setup

### Step 1: Create New Web Service

1. **Dashboard**: Navigate to [https://dashboard.render.com](https://dashboard.render.com)
2. **New Web Service**: Click "+ New" → "Web Service"
3. **Connect Repository**:
   - Select GitHub repository: `DONSystemsLLC/don-research-api`
   - Authorize Render to access repository
   - Select branch: `main` (or your production branch)

### Step 2: Configure Service Settings

**Service Name**: `don-research-api` (or custom name)  
**Region**: Oregon (US-West) or Virginia (US-East) - choose closest to users  
**Branch**: `main`  
**Root Directory**: Leave blank (project root)  
**Runtime**: Python 3  

### Step 3: Instance Type Selection

| Tier | vCPU | RAM | Price | Recommended For |
|------|------|-----|-------|-----------------|
| Free | 0.1 | 512 MB | $0 | Testing only (spins down after 15min inactivity) |
| Starter | 0.5 | 512 MB | $7/month | Small demos (<100 requests/hour) |
| Standard | 1 | 2 GB | $25/month | **Recommended** (academic research, <1000 req/hour) |
| Pro | 2 | 4 GB | $85/month | High-throughput (>1000 req/hour) |
| Pro Plus | 4 | 8 GB | $175/month | Enterprise deployments |

**Recommendation**: **Standard tier** for production academic research

### Step 4: Environment Variables

Configure in Render dashboard under "Environment" tab:

```env
# Required
PYTHON_VERSION=3.11
PORT=8080

# DON Stack Configuration
DON_STACK_MODE=internal

# Authentication (CRITICAL - Keep Secret!)
DON_AUTHORIZED_INSTITUTIONS_JSON={"demo_token":{"name":"Demo Access","contact":"demo@donsystems.com","rate_limit":100}}

# Optional: HTTP Mode (if using microservices)
# DON_GPU_ENDPOINT=http://don-gpu-service.onrender.com
# TACE_ENDPOINT=http://tace-service.onrender.com
```

⚠️ **IMPORTANT**: Never commit `DON_AUTHORIZED_INSTITUTIONS_JSON` to git. Use Render's encrypted environment variables.

### Step 5: Deploy

Click **"Create Web Service"** → Render automatically:
1. Clones repository
2. Installs dependencies (`pip install -r requirements.txt`)
3. Starts server (`uvicorn main:app --host 0.0.0.0 --port $PORT`)
4. Runs health checks
5. Routes traffic to service

**Deployment Time**: ~5-8 minutes for first deployment

---

## Environment Configuration

### Required Variables

```env
# Python Runtime
PYTHON_VERSION=3.11          # Required: Python version
PORT=8080                     # Required: HTTP port (Render sets automatically)
```

### DON Stack Configuration

```env
# Operating Mode
DON_STACK_MODE=internal       # Options: "internal" (default) or "http"

# Internal Mode (Default) - No additional variables needed
# HTTP Mode (Microservices) - Add these:
DON_GPU_ENDPOINT=http://don-gpu-service.onrender.com
TACE_ENDPOINT=http://tace-service.onrender.com
```

### Authentication Configuration

**Option 1: JSON String (Recommended for Production)**

```env
DON_AUTHORIZED_INSTITUTIONS_JSON={"token1":{"name":"Institution A","contact":"email@example.com","rate_limit":1000},"token2":{"name":"Institution B","contact":"other@example.com","rate_limit":100}}
```

**Option 2: File Path (Alternative)**

```env
DON_AUTHORIZED_INSTITUTIONS_FILE=/etc/secrets/institutions.json
```

Then add file via Render's "Secret Files" feature.

### Optional Debugging Variables

```env
# Logging
LOG_LEVEL=INFO                # Options: DEBUG, INFO, WARNING, ERROR
PYTHON_LOG_LEVEL=INFO

# Performance
WORKERS=1                     # Uvicorn worker count (default: 1)
TIMEOUT=300                   # Request timeout in seconds
```

### Adding Environment Variables

**Via Dashboard**:
1. Service → "Environment" tab
2. Click "Add Environment Variable"
3. Enter key and value
4. Click "Save Changes"
5. Service auto-redeploys

**Via Blueprint (Infrastructure as Code)**:

Create `render.yaml` in repository root:

```yaml
services:
  - type: web
    name: don-research-api
    env: python
    region: oregon
    plan: standard
    branch: main
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT
    envVars:
      - key: PYTHON_VERSION
        value: "3.11"
      - key: DON_STACK_MODE
        value: internal
      - key: DON_AUTHORIZED_INSTITUTIONS_JSON
        sync: false  # Must be set in dashboard (encrypted)
    healthCheckPath: /api/v1/health
```

---

## Build & Start Commands

### Build Command

**Default**: `pip install -r requirements.txt`

**Custom Build** (if needed):

```bash
# Install dependencies + custom setup
pip install -r requirements.txt && python setup_custom.py
```

**Build Optimization**:

```bash
# Use pip cache for faster rebuilds
pip install --upgrade pip && pip install -r requirements.txt
```

### Start Command

**Default**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

**Production Options**:

```bash
# With access logs
uvicorn main:app --host 0.0.0.0 --port $PORT --log-level info --access-log

# With worker tuning (for Standard+ tiers)
uvicorn main:app --host 0.0.0.0 --port $PORT --workers 2

# With timeout configuration
uvicorn main:app --host 0.0.0.0 --port $PORT --timeout-keep-alive 300
```

**Gunicorn Alternative** (for high concurrency):

```bash
# Install gunicorn
pip install gunicorn

# Start command
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

Add to `requirements.txt`:
```
gunicorn==21.2.0
```

---

## Health Check Configuration

Render automatically monitors your service health via the configured endpoint.

### Configure Health Check

**Dashboard**: Service → "Settings" → "Health & Alerts"

```yaml
Health Check Path: /api/v1/health
Health Check Interval: 60 seconds
Failure Threshold: 3 consecutive failures
Success Threshold: 2 consecutive successes
```

### Health Endpoint Requirements

Your `/api/v1/health` endpoint must:
1. Return HTTP 200 status code
2. Respond within 10 seconds
3. Return valid JSON (content doesn't matter, just 200 status)

**Current Implementation** (`main.py`):

```python
@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "don_stack": {
            "mode": os.getenv("DON_STACK_MODE", "internal"),
            "adapter_loaded": True
        }
    }
```

### Health Check Behavior

**Passing Health Checks**:
- Service marked as "healthy" (green status)
- Traffic routed to service
- No alerts sent

**Failing Health Checks** (3 consecutive failures):
- Service marked as "unhealthy" (red status)
- Render attempts automatic restart
- Email alert sent to account owner
- Traffic may be rerouted (if multiple instances)

**Restart Policy**:
- Automatic restart on health check failure
- Maximum 3 restart attempts per 10 minutes
- If all attempts fail → manual intervention required

---

## Scaling & Performance

### Vertical Scaling (Instance Size)

**Upgrade Instance Type**:
1. Service → "Settings" → "Instance Type"
2. Select higher tier (Starter → Standard → Pro)
3. Click "Save Changes"
4. Service redeploys with new resources

**When to Upgrade**:
- CPU usage consistently >70% → Upgrade vCPU
- Memory usage consistently >80% → Upgrade RAM
- Request latency >1 second → Upgrade to Pro tier

### Horizontal Scaling (Multiple Instances)

**Available on**: Standard tier and above

**Configure**:
1. Service → "Settings" → "Scaling"
2. Set "Number of Instances": 2-10
3. Traffic automatically load-balanced

**Cost**: Multiplies service cost (2 instances = 2× cost)

**Load Balancing**:
- Round-robin distribution
- Automatic failover on instance failure
- Session affinity not supported (stateless API only)

### Auto-Scaling (Enterprise)

**Render Scale** (requires Enterprise plan):
- Automatically scales based on CPU/memory usage
- Minimum and maximum instance counts
- Scale-up threshold: 70% resource usage
- Scale-down threshold: 30% resource usage

Contact Render sales for Enterprise pricing.

### Performance Optimization

**1. Enable HTTP/2**:
Enabled by default on Render (no configuration needed)

**2. Compression**:
Add middleware to `main.py`:

```python
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

**3. Connection Pooling**:
Already enabled in FastAPI (no changes needed)

**4. Caching**:
Implement Redis cache for frequently accessed data:

```bash
# Add Redis service on Render
# Update requirements.txt
redis==5.0.0

# Use in main.py
import redis
redis_client = redis.from_url(os.getenv("REDIS_URL"))
```

---

## Monitoring & Alerts

### Built-in Render Metrics

**Dashboard**: Service → "Metrics"

Available metrics (free on all tiers):
- **CPU Usage**: Percentage of allocated CPU
- **Memory Usage**: MB used / total MB
- **Request Count**: Requests per minute
- **Response Time**: Average latency (p50, p95, p99)
- **Error Rate**: 4xx and 5xx errors per minute

**Retention**: 7 days (free) or 30 days (paid plans)

### Email Alerts

**Configure**: Service → "Settings" → "Alerts"

**Available Alerts**:
- Health check failures (automatic)
- Deploy failures (automatic)
- High CPU usage (>80% for 5 minutes)
- High memory usage (>90% for 5 minutes)
- High error rate (>5% for 5 minutes)

**Recipients**: Account owner + team members

### External Monitoring (Recommended for Production)

**Uptime Monitoring**:
- [UptimeRobot](https://uptimerobot.com) (free tier: 50 monitors)
- [Pingdom](https://www.pingdom.com)
- [StatusCake](https://www.statuscake.com)

**Configuration**:
```
Monitor URL: https://your-service.onrender.com/api/v1/health
Interval: 60 seconds
Alert on: 3 consecutive failures
```

**Application Performance Monitoring (APM)**:
- [Sentry](https://sentry.io) (error tracking)
- [DataDog](https://www.datadoghq.com) (full APM)
- [New Relic](https://newrelic.com) (full APM)

**Sentry Integration** (recommended):

```bash
# Add to requirements.txt
sentry-sdk[fastapi]==1.40.0

# Add to main.py
import sentry_sdk

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    traces_sample_rate=0.1,  # 10% of requests
    environment="production"
)
```

Set `SENTRY_DSN` environment variable in Render dashboard.

---

## Troubleshooting

### Common Deployment Issues

#### 1. Build Failures

**Symptom**: Deployment fails during `pip install` step

**Causes & Solutions**:

```bash
# Missing dependencies in requirements.txt
→ Add missing packages to requirements.txt

# Incompatible Python version
→ Set PYTHON_VERSION=3.11 in environment variables

# Dependency conflicts
→ Use pip freeze > requirements.txt to capture exact versions

# Network timeout during package install
→ Retry deployment (Render auto-retries)
```

#### 2. Health Check Failures

**Symptom**: Service marked unhealthy immediately after deployment

**Debug Steps**:

```bash
# Check logs: Service → "Logs"
# Look for:
- Port binding errors: "Address already in use"
- Import errors: "ModuleNotFoundError"
- Configuration errors: "KeyError: DON_STACK_MODE"

# Common fixes:
→ Ensure uvicorn binds to $PORT (not hardcoded 8080)
→ Verify all imports in main.py resolve
→ Check environment variables are set correctly
```

**Manual Health Check Test**:

```bash
# Get service URL from dashboard
curl https://your-service.onrender.com/api/v1/health

# Should return 200 status code
```

#### 3. Authentication Failures

**Symptom**: All API requests return 401 errors

**Causes**:

```bash
# DON_AUTHORIZED_INSTITUTIONS_JSON not set
→ Set in Render environment variables (encrypted)

# Invalid JSON format
→ Validate JSON: https://jsonlint.com
→ Use proper escaping for quotes

# Token mismatch
→ Verify token in request matches token in config
```

#### 4. High Memory Usage / OOM Errors

**Symptom**: Service crashes with "Out of Memory" errors

**Solutions**:

```bash
# Upgrade instance tier
→ Standard (2 GB) → Pro (4 GB)

# Optimize memory usage
→ Reduce batch sizes in processing
→ Implement streaming for large files
→ Add memory limits to data structures

# Monitor memory in logs
→ Add logging: import psutil; print(psutil.virtual_memory())
```

#### 5. Slow Response Times

**Symptom**: Requests take >5 seconds to complete

**Debug**:

```bash
# Check Render metrics for bottleneck
→ CPU usage >90% → Upgrade CPU
→ Memory usage >80% → Upgrade RAM
→ Network latency → Enable HTTP/2 compression

# Profile application
→ Add timing logs to endpoints
→ Identify slow database/API calls
→ Implement caching for repeated queries
```

### Viewing Logs

**Real-time Logs**:
1. Dashboard → Service → "Logs"
2. Auto-refreshes every 5 seconds
3. Last 1000 lines visible

**Download Logs**:
```bash
# Install Render CLI
npm install -g @render-inc/cli

# Login
render login

# Download logs
render logs --service don-research-api --tail 10000 > logs.txt
```

**Log Retention**:
- Free tier: 7 days
- Paid plans: 30 days

### Restart Service

**Graceful Restart** (maintains uptime):
1. Dashboard → Service → "Manual Deploy"
2. Click "Clear build cache & deploy"

**Hard Restart** (brief downtime):
1. Dashboard → Service → "Settings"
2. Click "Suspend" → Wait 10 seconds → Click "Resume"

### Rollback to Previous Version

**Via Dashboard**:
1. Service → "Events" tab
2. Find successful previous deployment
3. Click "Rollback" button

**Via Git**:
```bash
# Revert to previous commit
git revert HEAD
git push origin main

# Render auto-deploys reverted version
```

---

## Security Hardening

### HTTPS/TLS Configuration

**Default**: HTTPS enabled automatically on `*.onrender.com` domains
- TLS 1.3 supported
- Free SSL certificate (Let's Encrypt)
- Auto-renewal (no manual intervention)

**Custom Domain** (requires paid plan):
1. Service → "Settings" → "Custom Domains"
2. Add your domain (e.g., `api.yourdomain.com`)
3. Configure DNS CNAME: `api.yourdomain.com` → `your-service.onrender.com`
4. SSL certificate auto-provisioned

### Environment Variable Security

**Best Practices**:
- ✅ Use Render's encrypted environment variables
- ✅ Never commit secrets to git
- ✅ Rotate tokens quarterly
- ❌ Don't log environment variable values
- ❌ Don't expose secrets in error messages

**Secret Files** (for large configs):
1. Service → "Settings" → "Secret Files"
2. Upload file (max 10 KB)
3. Reference via environment variable: `DON_AUTHORIZED_INSTITUTIONS_FILE=/etc/secrets/institutions.json`

### Network Security

**IP Whitelisting** (Enterprise only):
- Restrict access to specific IP ranges
- Useful for internal APIs
- Contact Render support

**DDoS Protection**:
- Built-in rate limiting at Render's edge
- Additional rate limiting in application (`main.py`)

**CORS Configuration** (if needed):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization"],
)
```

### Dependency Security

**Automated Updates**:
1. Enable Dependabot in GitHub repository
2. Auto-creates PRs for security updates
3. Review and merge PRs regularly

**Manual Audit**:
```bash
# Check for vulnerabilities
pip install safety
safety check -r requirements.txt

# Update outdated packages
pip list --outdated
```

---

## Cost Optimization

### Current Pricing (as of October 2025)

| Tier | Cost/Month | Best For |
|------|-----------|----------|
| Free | $0 | Testing (spins down after 15min) |
| Starter | $7 | Low-traffic demos |
| **Standard** | **$25** | **Academic research (recommended)** |
| Pro | $85 | High-throughput production |
| Pro Plus | $175 | Enterprise |

### Cost Reduction Strategies

**1. Right-size Instance**:
- Start with Standard tier
- Monitor CPU/memory usage
- Downgrade if usage <30%
- Only upgrade if bottlenecks occur

**2. Optimize Build Times**:
```bash
# Faster builds = less build minute usage
# Use pip cache
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

**3. Suspend Non-Production Services**:
- Suspend staging/dev services when not in use
- Resume for testing, then suspend again
- No charges while suspended

**4. Use Free Tier for Testing**:
- Deploy to free tier first
- Test functionality
- Migrate to paid tier for production

**5. Annual Billing**:
- Save 20% with annual commitment
- Standard: $240/year (vs $300/year monthly)

### Monitoring Costs

**Dashboard**: Billing → Usage

**Metrics Tracked**:
- Instance hours
- Bandwidth (free: 100 GB/month)
- Build minutes (free: 500/month)

**Alerts**:
Set up billing alerts at 50%, 80%, 100% of monthly budget.

---

## Additional Resources

### Render Documentation

- [Render Docs](https://render.com/docs)
- [Python on Render](https://render.com/docs/deploy-fastapi)
- [Environment Variables](https://render.com/docs/environment-variables)
- [Health Checks](https://render.com/docs/health-checks)

### DON Research API Documentation

- [API Reference](./API_REFERENCE.md)
- [Data Policy](./DATA_POLICY.md)
- [TAMU Integration Guide](./TAMU_INTEGRATION.md)

### Support Contacts

- **Render Support**: [https://render.com/support](https://render.com/support)
- **DON Systems Support**: [support@donsystems.com](mailto:support@donsystems.com)
- **Security Issues**: [security@donsystems.com](mailto:security@donsystems.com)

---

## Deployment Checklist

Before going to production:

- [ ] Environment variables configured (especially `DON_AUTHORIZED_INSTITUTIONS_JSON`)
- [ ] Health check endpoint returning 200 status
- [ ] Instance tier appropriate for expected traffic (Standard recommended)
- [ ] Monitoring alerts configured (email notifications)
- [ ] External uptime monitoring setup (UptimeRobot/Pingdom)
- [ ] Error tracking configured (Sentry recommended)
- [ ] Custom domain configured (optional, requires paid plan)
- [ ] SSL certificate verified (automatic on Render)
- [ ] Rate limiting tested (verify 429 responses)
- [ ] Authentication tested (verify 401 for invalid tokens)
- [ ] Documentation updated with production URL
- [ ] Team members added to Render account (for alerts)
- [ ] Backup/rollback plan documented
- [ ] Cost monitoring enabled (billing alerts)

---

**Document Version**: 1.0  
**Last Updated**: October 26, 2025  
**Maintained By**: DON Systems LLC

For questions or issues, contact [support@donsystems.com](mailto:support@donsystems.com)
