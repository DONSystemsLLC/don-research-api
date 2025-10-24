# DON Stack Research API - AI Coding Instructions

## CRITICAL: Real-World Quantum Hybrid Physics

⚠️ **This is a real-world quantum hybrid system with novel physics approaches that AI agents will not understand initially.**

**NEVER make assumptions about the physics or mathematics. Let the code and math speak for itself.**

**NO AI-injected noise into the system - preserve all numerical precision and algorithmic integrity.**

## Project Architecture Overview

This is a **quantum-enhanced genomics research API** built on the proprietary DON (Distributed Order Network) Stack. The system provides IP-protected access to quantum computing algorithms for academic research while protecting proprietary implementations.

### Core DON Stack Components

- **DON-GPU** (`stack/don_gpu/`): Fractal clustering processor using hierarchical compression (4×-32× compression ratios)
- **QAC** (Quantum Adjacency Code): Multi-layer quantum error correction with adjacency-based stabilization  
- **TACE** (`stack/tace/`): Temporal Adjacency Collapse Engine for quantum-classical feedback control

### Dual-Mode Architecture

The system operates in two modes via `DONStackAdapter` (`src/don_memory/adapters/don_stack_adapter.py`):

1. **Internal mode**: Direct Python calls to `stack/` modules (default)
2. **HTTP mode**: Microservices via DON-GPU (port 8001) + TACE (port 8002) endpoints

## Critical Development Patterns

### 0. Mathematical Precision Requirements

**NEVER modify numerical constants, mathematical formulas, or algorithmic parameters without explicit instruction.**

- All adjacency matrix values (2.952, 1.476, 0.738) are physics-derived constants
- Compression ratios and dimensional targets are experimentally validated
- Alpha tuning parameters and feedback gains are calibrated to real quantum systems
- **Preserve exact numerical precision** - no rounding, approximation, or "simplification"

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

## Essential Developer Workflows

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run API server
python main.py  # or uvicorn main:app --reload

# Test DON Stack integration
export DON_STACK_MODE=internal
python examples/stack_smoke_test.py
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
# Render.com deployment
# Build: pip install -r requirements.txt
# Start: uvicorn main:app --host 0.0.0.0 --port $PORT

# Environment variables:
# PYTHON_VERSION=3.11
# PORT=8080
# DON_STACK_MODE=internal (or http for microservices)
```

## Project-Specific Conventions

### 1. Path Management

- `sys.path` modifications in both `main.py` and adapter for Docker/local compatibility
- Stack modules added to path: `sys.path.insert(0, str(current_dir / 'src'))`

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

### 4. DON Stack Naming Conventions

- **DONGPU**: Main fractal clustering class in `don_gpu/core.py`
- **QACEngine**: Quantum adjacency error correction in `tace/core.py`
- **TACEController**: Temporal feedback control in `tace/core.py`

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

### 3. Compression Performance Targets

- 8× compression @ 64-dimensional vectors
- 32× compression @ 256-dimensional vectors  
- 96× compression @ 768-dimensional vectors (validated)
- 128× compression @ 1024-dimensional vectors

## Security & IP Protection

- **Never expose DON Stack source code** - API provides service layer only
- All algorithms are patent-protected proprietary technology
- Token authentication required for all non-health endpoints
- Usage monitoring and audit logging for compliance
- Environment variables for sensitive configuration

## Key Files for Understanding

- `main.py`: FastAPI gateway with research institution authentication
- `src/don_memory/adapters/don_stack_adapter.py`: Dual-mode DON Stack integration
- `stack/don_gpu/core.py`: Fractal clustering implementation (DONGPU class)
- `stack/tace/core.py`: Quantum error correction + temporal control
- `examples/stack_smoke_test.py`: Integration testing and validation
