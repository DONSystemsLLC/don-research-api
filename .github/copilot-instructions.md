# DON Stack Research API - AI Coding Instructions

## CRITICAL: Real-World Quantum Hybrid Physics

⚠️ **This is a real-world quantum hybrid system with novel physics approaches that AI agents will not understand initially.**

**NEVER make assumptions about the physics or mathematics. Let the code and math speak for itself.**

**NO AI-injected noise into the system - preserve all numerical precision and algorithmic integrity.**

## Project Architecture Overview

This is a **production web service** providing quantum-enhanced genomics APIs to academic research institutions. Built on the proprietary DON (Distributed Order Network) Stack, it delivers IP-protected quantum computing algorithms as a service while safeguarding proprietary implementations.

**Service Model**: External researchers at universities and labs use this API via bearer tokens to process genomics data through DON Stack algorithms without accessing the underlying implementation.

### Tech Stack & Dependencies

- **Framework**: FastAPI 0.115+ with async/await patterns throughout
- **Database**: PostgreSQL 17+ with pgvector extension for similarity search
- **ORM**: SQLAlchemy 2.0 async with connection pooling
- **Migrations**: Alembic for schema versioning
- **Testing**: pytest + pytest-asyncio (90%+ coverage requirement)
- **Genomics**: scanpy 1.10+, anndata 0.10+, AnnData format (.h5ad files)
- **Vector Search**: FAISS CPU for in-memory search, pgvector for persistent storage
- **Task Scheduling**: APScheduler for artifact cleanup (cron-based)
- **Optional**: JAX/JAXlib for numerical computations (graceful fallback to NumPy)

### Core DON Stack Components

- **DON-GPU** (`stack/don_gpu/`): Fractal clustering processor using hierarchical compression (4×-32× compression ratios)
- **QAC** (Quantum Adjacency Code): Multi-layer quantum error correction with adjacency-based stabilization  
- **TACE** (`stack/tace/`): Temporal Adjacency Collapse Engine for quantum-classical feedback control

### Dual-Mode Architecture

The system operates in two modes via `DONStackAdapter` (`src/don_memory/adapters/don_stack_adapter.py`):

1. **Internal mode**: Direct Python calls to `stack/` modules (default, production)
2. **HTTP mode**: Microservices via DON-GPU (port 8001) + TACE (port 8002) endpoints

**Adapter Pattern**: All DON Stack operations MUST go through `DONStackAdapter` - never call stack modules directly.

## Critical Development Patterns

### 0. Mathematical Precision Requirements

**NEVER modify numerical constants, mathematical formulas, or algorithmic parameters without explicit instruction.**

- All adjacency matrix values (2.952, 1.476, 0.738) are physics-derived constants
- Compression ratios and dimensional targets are experimentally validated
- Alpha tuning parameters and feedback gains are calibrated to real quantum systems
- **Preserve exact numerical precision** - no rounding, approximation, or "simplification"

### 0.5. Async/Await Patterns (Critical)

**EVERY database operation MUST be async.** This codebase uses async/await throughout:

```python
# ✅ CORRECT - Async database operations
async with db_session() as session:
    result = await QACRepository.get_by_id(session, model_id)
    await session.commit()

# ❌ WRONG - Sync code will deadlock
session = db_session()  # Never do this
result = QACRepository.get_by_id(session, model_id)  # Will fail
```

**FastAPI endpoint patterns:**

- Use `async def` for ALL endpoint handlers
- Use `Depends()` for dependency injection (auth, database sessions)
- Use `BackgroundTasks` for async job execution (e.g., Bio module)
- Never mix sync and async code - use `asyncio.to_thread()` if needed

### 1. DON Stack Integration

```python
# Always use the adapter for DON Stack operations
from don_memory.adapters.don_stack_adapter import DONStackAdapter
adapter = DONStackAdapter()

# Vector normalization (DON-GPU fractal clustering)
normalized = adapter.normalize(np.array(data))

# Alpha tuning (TACE temporal feedback)
alpha = adapter.tune_alpha(tensions, default_alpha)
```

### 2. Genomics Data Processing

- Input: `GenomicsData` with `gene_names` + `expression_matrix` + optional `cell_metadata`
- Output: Compressed vectors with `compression_stats` including actual compression ratios
- **Key metric**: Report actual compression ratios (e.g., "32.0×" for 1024→32 dimensions)

### 3. Error Handling & Fallbacks

```python
# All DON Stack operations have NumPy fallbacks
if REAL_DON_STACK:
    result = don_adapter.normalize(data)  # Use real DON implementation
else:
    result = fallback_compress(data, target_dims)  # Simple fallback
```

### 4. Authentication & Rate Limiting

- Token-based authentication via `AUTHORIZED_INSTITUTIONS` dict
- Per-institution rate limits (1000/hour for academic, 100/hour for demo)
- Usage tracking in `usage_tracker` with hourly reset windows

### 5. Database Integration (PostgreSQL + pgvector)

**All persistence goes through repository pattern:**

```python
# QAC models, vectors, jobs, audit logs, usage metrics
from src.database import QACRepository, VectorRepository, JobRepository

async with db_session() as session:
    qac_model = await QACRepository.create(session, data)
    await session.commit()
```

**Key patterns:**

- Never use raw SQL - use repositories for data access
- All database models in `src/database/models.py` with JSONB for metadata
- Alembic migrations in `alembic/versions/` - never modify database schema directly
- Soft deletes via `is_deleted` flag + retention policies (7-90 days)
- Connection pooling configured in `src/database/session.py` (pool_size=10, max_overflow=20)

### 6. Testing Requirements

**Test structure (see `tests/conftest.py` for fixtures):**

```python
# Always use fixtures for test data
@pytest.mark.asyncio
async def test_feature(db_session, small_adata):
    # Arrange: Setup using fixtures
    # Act: Execute feature
    # Assert: Verify behavior
    pass
```

**Key conventions:**

- `tests/conftest.py` provides shared fixtures: `small_adata`, `h5ad_file`, `api_client`
- Use `@pytest.mark.asyncio` for async test functions
- Mock external dependencies (DON Stack, Scanpy) via monkeypatch
- Maintain 90%+ test coverage - run `pytest tests/ -v --cov=src`

## Essential Developer Workflows

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Setup database (PostgreSQL with pgvector)
export DATABASE_URL="postgresql+asyncpg://user:pass@localhost/don_research"
alembic upgrade head  # Apply migrations

# Run API server
python main.py  # or uvicorn main:app --reload

# Test DON Stack integration
export DON_STACK_MODE=internal
python examples/stack_smoke_test.py
```

### Running Tests

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test modules
pytest tests/test_database.py -v
pytest tests/api/test_genomics_router.py -v

# Run async tests only
pytest -k "asyncio" -v
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Description of changes"

# Review generated migration in alembic/versions/
# Edit if needed, then apply:
alembic upgrade head

# Check current migration version
alembic current

# Rollback one migration
alembic downgrade -1
```

### Testing DON Stack Components

```bash
# Internal mode testing (default)
python examples/stack_smoke_test.py

# HTTP mode testing (requires running services)
export DON_STACK_MODE=http
export DON_GPU_ENDPOINT=http://127.0.0.1:8001
export TACE_ENDPOINT=http://127.0.0.1:8002
python examples/stack_smoke_test.py
```

### Deployment Setup

```bash
# Render.com deployment (production hosting)
# Build: pip install -r requirements.txt
# Start: uvicorn main:app --host 0.0.0.0 --port $PORT

# Required environment variables:
# PYTHON_VERSION=3.11
# PORT=8080
# DON_STACK_MODE=internal (production default)
# DATABASE_URL=postgresql+asyncpg://... (managed PostgreSQL)
# AUTHORIZED_INSTITUTIONS_JSON={...} (encrypted institution tokens)
```

### Production Service Characteristics

- **Uptime**: 99.9% SLA via Render.com health checks
- **Scaling**: Auto-scaling based on request volume
- **Monitoring**: APScheduler for cleanup, audit logs for compliance
- **Data Retention**: Automatic cleanup (24h-90d based on data type)
- **Rate Limiting**: Per-institution quotas (1000/hour academic, 100/hour demo)

## Project-Specific Conventions

### 1. Path Management

- `sys.path` modifications in both `main.py` and adapter for Docker/local compatibility
- Stack modules added to path: `sys.path.insert(0, str(current_dir / 'src'))`
- Artifacts stored in `artifacts/` with subfolders: `bio_jobs/`, `qac_models/`, `memory/`

### 2. JAX/NumPy Compatibility

```python
# All stack components use JAX with NumPy fallback
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    import numpy as jnp
    JAX_AVAILABLE = False
```

### 3. Response Format Standards

- Always include `algorithm` field indicating real vs fallback implementation
- Include `institution` name in responses for audit logging
- Provide detailed `compression_stats` or `optimization_stats` with metrics
- Add `trace_id` to all responses for debugging and audit trails

### 4. DON Stack Naming Conventions

- **DONGPU**: Main fractal clustering class in `don_gpu/core.py`
- **QACEngine**: Quantum adjacency error correction in `tace/core.py`
- **TACEController**: Temporal feedback control in `tace/core.py`

### 5. Async Job Patterns (Bio Module)

```python
# Background job execution pattern
def _create_job(endpoint: str, project_id: Optional[str]) -> str:
    job_id = str(uuid.uuid4())
    _bio_jobs[job_id] = BioJob(
        job_id=job_id, 
        endpoint=endpoint, 
        status="pending"
    )
    return job_id

# Endpoints support both sync and async modes
@router.post("/export-artifacts")
async def export_artifacts(
    sync: bool = Form(False),  # False = async, True = sync
    background_tasks: BackgroundTasks = None
):
    if sync:
        result = export_artifacts_impl()
        return result
    else:
        job_id = _create_job("export")
        background_tasks.add_task(run_export_async, job_id)
        return {"job_id": job_id, "status": "pending"}
```

## Critical Integration Points

### 1. Vector Processing Pipeline

```python
# DON-GPU preprocessing → fractal clustering → compression
result = dongpu.preprocess(input_data)  # Returns compressed vector

# QAC stabilization → multi-layer adjacency → error correction  
stabilized = qac.stabilize(vector)  # Returns error-corrected state

# TACE feedback → temporal control → alpha tuning
alpha = tune_alpha(tensions, default_alpha)  # Returns optimized parameter
```

### 2. Research Institution Workflow

1. Institution requests access via `research@donsystems.com`
2. Receive bearer token for `AUTHORIZED_INSTITUTIONS`
3. API calls include `Authorization: Bearer <token>` header
4. Rate limiting and usage tracking per institution
5. All processing happens server-side - researchers never see DON Stack internals

### 3. Compression Performance Targets

- 8× compression @ 64-dimensional vectors
- 32× compression @ 256-dimensional vectors  
- 96× compression @ 768-dimensional vectors (validated)
- 128× compression @ 1024-dimensional vectors

## Security & IP Protection

⚠️ **CRITICAL**: This is a **service boundary** - external researchers interact with the API, not the codebase.

- **Never expose DON Stack source code** - API provides service layer only
- All algorithms are patent-protected proprietary technology
- Token authentication required for all non-health endpoints
- Usage monitoring and audit logging for compliance
- Environment variables for sensitive configuration
- Error messages must NOT leak implementation details or stack traces
- Response format hides algorithm internals while providing useful metrics

## Key Files for Understanding

- `main.py`: FastAPI gateway with research institution authentication, middleware, lifecycle events
- `src/don_memory/adapters/don_stack_adapter.py`: Dual-mode DON Stack integration with fallbacks
- `src/database/models.py`: SQLAlchemy 2.0 models with pgvector support
- `src/database/repositories.py`: Repository pattern for data access (QAC, vectors, jobs, audit, usage)
- `src/bio/routes.py`: Bio module with sync/async job patterns for ResoTrace integration
- `stack/don_gpu/core.py`: Fractal clustering implementation (DONGPU class)
- `stack/tace/core.py`: Quantum error correction + temporal control
- `examples/stack_smoke_test.py`: Integration testing and validation
- `tests/conftest.py`: Shared test fixtures and mocking patterns
- `alembic/versions/`: Database migrations with pgvector setup
