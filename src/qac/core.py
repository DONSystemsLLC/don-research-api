from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any

try:
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    SCIPY_OK = True
except Exception:
    sparse = None
    spsolve = None
    SCIPY_OK = False


def _pairwise_knn_indices(Z: np.ndarray, k: int) -> np.ndarray:
    """Return indices of k nearest neighbors for each row of Z (exclude self).
    Uses NumPy; if dataset is very large, upgrade to faiss/annoy.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    # Compute squared distances via (a-b)^2 = ||a||^2 + ||b||^2 - 2 a·b
    # Chunk to limit peak memory.
    chunk = max(1, 4000 * max(1, 16 // max(1, Z.shape[1] // 32)))
    idx_all = np.empty((n, k), dtype=int)
    norms = (Z * Z).sum(1)
    for start in range(0, n, chunk):
        end = min(n, start + chunk)
        dots = Z[start:end] @ Z.T
        d2 = norms[start:end, None] + norms[None, :] - 2.0 * dots
        np.fill_diagonal(d2, np.inf)  # no self
        idx = np.argpartition(d2, kth=k, axis=1)[:, :k]
        # sort the k neighbors by distance
        row = np.arange(end - start)[:, None]
        d2k = np.take_along_axis(d2, idx, axis=1)
        ordk = np.argsort(d2k, axis=1)
        idx_sorted = np.take_along_axis(idx, ordk, axis=1)
        idx_all[start:end] = idx_sorted
    return idx_all


def _weights(Z: np.ndarray, idx: np.ndarray, mode: str = 'binary', sigma: float | None = None) -> np.ndarray:
    """Compute neighbor weights for each (i, j) pair.
    Returns array shaped (n, k).
    """
    n, k = idx.shape
    W = np.ones((n, k), dtype=float)
    if mode == 'gaussian':
        # estimate sigma if not provided
        if sigma is None:
            # approximate local scale via median distance to kth neighbor
            dists = np.sqrt(((Z[:, None, :] - Z[idx, :]) ** 2).sum(-1))  # (n, k)
            sigma = float(np.median(dists[:, -1]) + 1e-9)
        dists = np.sqrt(((Z[:, None, :] - Z[idx, :]) ** 2).sum(-1))
        W = np.exp(- (dists ** 2) / (2.0 * sigma ** 2))
    return W


def build_laplacian(Z: np.ndarray, k_nn: int = 15, weight: str = 'binary', sigma: float | None = None,
                    symmetrize: bool = True) -> Tuple[Any, Dict[str, float]]:
    """Build graph Laplacian L. Returns (L, stats). If SciPy is present, L is CSR; else dense.
    """
    Z = np.asarray(Z, dtype=float)
    n = Z.shape[0]
    idx = _pairwise_knn_indices(Z, k_nn)
    Wvals = _weights(Z, idx, mode=weight, sigma=sigma)

    if SCIPY_OK:
        rows = np.repeat(np.arange(n), k_nn)
        cols = idx.reshape(-1)
        data = Wvals.reshape(-1)
        W = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
        if symmetrize:
            W = (W + W.T).multiply(0.5)
        d = np.array(W.sum(axis=1)).ravel()
        D = sparse.diags(d)
        L = D - W
        stats = {"edges": float(W.nnz), "avg_degree": float(d.mean() if n else 0.0)}
        return L, stats
    else:
        W = np.zeros((n, n), dtype=float)
        rows = np.repeat(np.arange(n), k_nn)
        cols = idx.reshape(-1)
        W[rows, cols] = Wvals.reshape(-1)
        if symmetrize:
            W = 0.5 * (W + W.T)
        d = W.sum(1)
        L = np.diag(d) - W
        stats = {"edges": float((W > 0).sum()), "avg_degree": float(d.mean() if n else 0.0)}
        return L, stats


def stabilize_embedding(Z0: np.ndarray, L: Any, beta: float = 0.7, lam: float = 0.05) -> Tuple[np.ndarray, Dict[str, float]]:
    """Solve z* = argmin ||z - Z0||^2 + beta * Tr(z^T L z) + lam * ||z||^2.
    Closed form: (I*(1+lam) + beta*L) z* = Z0. Uses sparse solve if available.
    Returns (Z*, metrics).
    """
    Z0 = np.asarray(Z0, dtype=float)
    n, k = Z0.shape
    if SCIPY_OK and sparse.isspmatrix(L):
        A = (sparse.eye(n) * (1.0 + lam) + L * beta).tocsr()
        # Solve per column (factorization would be faster; keep simple/robust)
        Z = np.empty_like(Z0)
        for j in range(k):
            Z[:, j] = spsolve(A, Z0[:, j])
    else:
        I = np.eye(n)
        A = I * (1.0 + lam) + (L * beta)
        Z = np.linalg.solve(A, Z0)

    # Metrics
    def dirichlet_energy(M, Z):
        if SCIPY_OK and sparse.isspmatrix(M):
            v = (M @ Z)
            return float((Z * v).sum())
        else:
            return float(np.sum(Z * (M @ Z)))

    e0 = dirichlet_energy(L, Z0)
    e1 = dirichlet_energy(L, Z)
    psi = 1.0 - (np.linalg.norm(Z - Z0) / (np.linalg.norm(Z0) + 1e-9))
    gain = 0.0 if e0 == 0 else (e0 - e1) / e0
    return Z, {"psi_fidelity": float(psi), "coherence_gain": float(gain)}