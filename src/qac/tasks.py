from __future__ import annotations
import numpy as np, uuid, os
from typing import Dict, Any
from .core import build_laplacian, stabilize_embedding, SCIPY_OK
from .store import create_job, update_job, save_model, load_model

# Robust import of your real QAC engine
HAVE_REAL_QAC = False
RealQAC = None

# Try to import the real QAC engine from the DON Stack
try:
    import sys
    from pathlib import Path
    # Add stack directory to path for QAC engine import
    stack_dir = Path(__file__).parent.parent.parent / "stack"
    if str(stack_dir) not in sys.path:
        sys.path.insert(0, str(stack_dir))
    
    from tace.core import QACEngine as RealQAC
    HAVE_REAL_QAC = True
except Exception:
    # Also try alternative import paths
    _qac_paths = [
        'qac_engine',                 # repo-local module qac_engine.py
        'codex.qac_engine',           # package path if nested
        'tace.core',                  # DON Stack path
    ]
    for _mod in _qac_paths:
        try:
            RealQAC = __import__(_mod, fromlist=['QACEngine']).QACEngine
            HAVE_REAL_QAC = True
            break
        except Exception:
            continue

# Env knobs
DEFAULT_ENGINE = os.getenv('QAC_DEFAULT_ENGINE', 'real_qac').lower()  # 'real_qac' | 'laplace'
MAX_SYNC_CELLS = int(os.getenv('QAC_MAX_SYNC_CELLS', '10000'))


def _knn_idx_w(Z: np.ndarray, k_nn: int, weight: str, sigma):
    from .core import _pairwise_knn_indices, _weights
    idx = _pairwise_knn_indices(Z, k_nn)
    w   = _weights(Z, idx, mode=weight, sigma=sigma)
    return idx, w


def run_fit(payload: Dict[str, Any]) -> Dict[str, Any]:
    Z = np.asarray(payload['embedding'], dtype=float)
    p = payload.get('params', {}) or {}
    k_nn = int(p.get('k_nn', 15))
    weight = p.get('weight', 'binary')
    sigma  = p.get('sigma', None)

    idx, w = _knn_idx_w(Z, k_nn, weight, sigma)
    model_id = str(uuid.uuid4())
    meta = {
        'model_id': model_id,
        'n_cells': int(Z.shape[0]),
        'k_nn': k_nn,
        'weight': weight,
        'reinforce_rate': float(p.get('reinforce_rate', 0.05)),
        'layers': int(p.get('layers', 50)),
        'beta': float(p.get('beta', 0.7)),
        'lambda_entropy': float(p.get('lambda_entropy', 0.05)),
        'engine': p.get('engine', DEFAULT_ENGINE if HAVE_REAL_QAC else 'laplace'),
        'created_at': payload.get('created_at'),
        'version': 'qac-1'
    }
    save_model(model_id, meta, idx, w)
    return {'model_id': model_id, 'meta': meta}


def _apply_real_qac(Z0: np.ndarray, idx: np.ndarray, w: np.ndarray, meta: dict) -> np.ndarray:
    """Apply your QACEngine column-wise across samples using data-derived adjacency."""
    n, k = Z0.shape
    # Build symmetric W from (idx, w)
    if SCIPY_OK:
        from scipy import sparse
        rows = np.repeat(np.arange(n), idx.shape[1])
        cols = idx.reshape(-1)
        data = w.reshape(-1)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        W = (W + W.T).multiply(0.5).toarray()
    else:
        W = np.zeros((n, n), dtype=float)
        rows = np.repeat(np.arange(n), idx.shape[1])
        cols = idx.reshape(-1)
        W[rows, cols] = w.reshape(-1)
        W = 0.5 * (W + W.T)

    # Instantiate engine and inject adjacency
    eng = RealQAC(num_qubits=n,
                  reinforce_rate=float(meta.get('reinforce_rate', 0.05)),
                  layers=int(meta.get('layers', 50)))
    
    # QAC engine uses jnp; we set its base_adj directly
    try:
        # Try to import jnp compatibility layer
        import jax.numpy as jnp
        eng.base_adj = jnp.array(W, dtype=jnp.float32)
    except Exception:
        # If jnp not available, assign numpy (engine logs fallback internally)
        eng.base_adj = W

    Z = np.empty_like(Z0)
    for j in range(k):
        Z[:, j] = np.asarray(eng.stabilize(Z0[:, j].tolist()), dtype=float)
    return Z


def run_apply(payload: Dict[str, Any]) -> Dict[str, Any]:
    Z0 = np.asarray(payload['embedding'], dtype=float)
    m = load_model(payload['model_id'])
    idx, w, meta = m['idx'], m['w'], m['meta']
    n = Z0.shape[0]
    assert idx.shape[0] == n, 'Embedding row count must match fitted model n_cells'

    engine = (meta.get('engine') or DEFAULT_ENGINE).lower()
    used = engine
    fallback_reason = None

    if engine == 'real_qac' and HAVE_REAL_QAC:
        try:
            Z = _apply_real_qac(Z0, idx, w, meta)
        except Exception as e:
            # graceful fallback
            used = 'laplace'; fallback_reason = f"real_qac failed: {e}"
            L, _ = build_laplacian(Z0, k_nn=meta['k_nn'], weight=meta['weight'])
            Z, _ = stabilize_embedding(Z0, L, beta=float(meta['beta']), lam=float(meta['lambda_entropy']))
    else:
        L, _ = build_laplacian(Z0, k_nn=meta['k_nn'], weight=meta['weight'])
        Z, _ = stabilize_embedding(Z0, L, beta=float(meta['beta']), lam=float(meta['lambda_entropy']))
        if engine != 'laplace':
            used = 'laplace'; fallback_reason = 'real_qac not available'

    # Metrics (observational)
    Lm, _ = build_laplacian(Z0, k_nn=meta['k_nn'], weight=meta['weight'])
    if SCIPY_OK:
        from scipy import sparse
        e0 = float(((Lm @ Z0) * Z0).sum())
        e1 = float(((Lm @ Z)  * Z ).sum())
    else:
        e0 = float(np.sum(Z0 * (Lm @ Z0)))
        e1 = float(np.sum(Z  * (Lm @ Z )))
    psi = 1.0 - (np.linalg.norm(Z - Z0) / (np.linalg.norm(Z0) + 1e-9))
    gain = 0.0 if e0 == 0 else (e0 - e1) / e0

    out = {
        'model_id': meta['model_id'],
        'embedding_qac': Z.tolist(),
        'metrics': {'psi_fidelity': float(psi), 'coherence_gain': float(gain)},
        'n_cells': int(n),
        'engine_used': used
    }
    if fallback_reason:
        out['fallback_reason'] = fallback_reason
    return out


# ---------------- job wrappers ----------------

def enqueue_fit_job(body: Dict[str, Any]) -> str:
    jid = create_job('fit')
    update_job(jid, status='running', progress=0.05)
    out = run_fit(body)
    update_job(jid, status='succeeded', progress=1.0, result=out, model_id=out['model_id'])
    return jid


def enqueue_apply_job(body: Dict[str, Any]) -> str:
    jid = create_job('apply', model_id=body['model_id'])
    update_job(jid, status='running', progress=0.05)
    out = run_apply(body)
    update_job(jid, status='succeeded', progress=1.0, result=out)
    return jid