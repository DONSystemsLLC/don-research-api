from __future__ import annotations
from typing import Optional, Dict, Any, List
import os, time
import numpy as np

# The adapter supports two modes:
#   - internal: call Python modules under stack/ (same repo/process)
#   - http: call deployed services via HTTP (DON-GPU/TACE)
#
# Switch using env:
#   DON_STACK_MODE=internal|http
#   DON_GPU_ENDPOINT=...      (when http)
#   TACE_ENDPOINT=...         (when http)
#   DON_STACK_TOKEN=...       (optional Bearer token)

DEFAULT_TIMEOUT = 6.0
RETRY_BACKOFFS = (0.2, 0.5, 1.2)

class DONStackAdapter:
    def __init__(self):
        self.mode = os.getenv("DON_STACK_MODE", "internal").lower()
        self.gpu_url = (os.getenv("DON_GPU_ENDPOINT", "") or "").rstrip("/")
        self.tace_url = (os.getenv("TACE_ENDPOINT", "") or "").rstrip("/")
        self.token = os.getenv("DON_STACK_TOKEN", "")
        self._cb_open_until = 0.0  # simple circuit breaker for http mode

    # ---------------- public API ----------------
    def normalize(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if self.mode == "http":
            out = self._normalize_http(v)
        else:
            out = self._normalize_internal(v)
        # guarantee non-zero
        n = float(np.linalg.norm(out))
        return out / (n + 1e-12)

    def tune_alpha(self, tensions: List[float], default_alpha: float) -> float:
        tensions = list(map(float, tensions or []))[-16:]
        if self.mode == "http":
            return self._tune_http(tensions, default_alpha)
        return self._tune_internal(tensions, default_alpha)

    def health(self) -> Dict[str, Any]:
        if self.mode != "http":
            # try internal imports with proper path setup
            import sys
            from pathlib import Path
            stack_dir = Path(__file__).parent.parent.parent.parent / "stack"
            if str(stack_dir) not in sys.path:
                sys.path.insert(0, str(stack_dir))
            
            gpu_ok = self._try_import("don_gpu.core", "DONGPU")
            tace_ok = self._try_import("tace.core", "tune_alpha")
            return {"mode": "internal", "don_gpu": gpu_ok, "tace": tace_ok}
        # http mode health
        gpu = self._http("GET", f"{self.gpu_url}/health")
        tace = self._http("GET", f"{self.tace_url}/health")
        return {"mode": "http", "don_gpu": gpu, "tace": tace}

    # ---------------- internal mode ----------------
    def _normalize_internal(self, v: np.ndarray) -> np.ndarray:
        try:
            import sys
            from pathlib import Path
            # Add stack directory to path
            stack_dir = Path(__file__).parent.parent.parent.parent / "stack"
            if str(stack_dir) not in sys.path:
                sys.path.insert(0, str(stack_dir))
            
            from don_gpu.core import DONGPU
            dongpu = DONGPU(num_cores=64, cluster_size=8, depth=3)
            result = dongpu.preprocess(v.tolist())
            out = np.array(result, dtype=float)
            return out if np.isfinite(out).all() and len(out) > 0 else v
        except Exception as e:
            # Fallback to simple normalization
            norm = np.linalg.norm(v) + 1e-12
            return v / norm

    def _tune_internal(self, tensions: List[float], default_alpha: float) -> float:
        try:
            import sys
            from pathlib import Path
            # Add stack directory to path
            stack_dir = Path(__file__).parent.parent.parent.parent / "stack"
            if str(stack_dir) not in sys.path:
                sys.path.insert(0, str(stack_dir))
            
            from tace.core import tune_alpha
            return float(tune_alpha(tensions, default_alpha))
        except Exception as e:
            # Fallback to simple alpha tuning
            if not tensions:
                return float(default_alpha)
            t = np.asarray(tensions, dtype=float)
            mu = float(np.clip(t.mean(), 0.0, 1.0))
            a = float(default_alpha) * (1.0 - 0.3*mu) + 0.1*mu
            return float(np.clip(a, 0.2, 0.8))

    # ---------------- http mode ----------------
    def _headers(self) -> Dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _http(self, method: str, url: str, json: Dict[str, Any] | None = None) -> Optional[Dict[str, Any]]:
        import httpx
        now = time.time()
        if now < self._cb_open_until:
            return None
        for backoff in list(RETRY_BACKOFFS) + [None]:
            try:
                with httpx.Client(timeout=DEFAULT_TIMEOUT) as c:
                    r = c.request(method, url, headers=self._headers(), json=json)
                    if 200 <= r.status_code < 300:
                        return r.json() if r.content else {}
                    if r.status_code in (429, 502, 503, 504) and backoff is not None:
                        time.sleep(backoff); continue
                    break
            except Exception:
                if backoff is not None:
                    time.sleep(backoff); continue
        self._cb_open_until = time.time() + 10.0
        return None

    def _normalize_http(self, v: np.ndarray) -> np.ndarray:
        payload = {"vector": v.tolist(), "mode": "entropy_norm"}
        out = self._http("POST", f"{self.gpu_url}/v1/normalize", json=payload)
        try:
            arr = np.asarray(out["vector"], dtype=float)
            return arr if arr.size else v
        except Exception:
            return v

    def _tune_http(self, tensions: List[float], default_alpha: float) -> float:
        payload = {"tensions": tensions, "default_alpha": float(default_alpha)}
        out = self._http("POST", f"{self.tace_url}/v1/tune", json=payload)
        try:
            a = float(out.get("alpha", default_alpha)) if out else default_alpha
            return a if 0.05 <= a <= 1.5 else float(default_alpha)
        except Exception:
            return float(default_alpha)

    # ---------------- utils ----------------
    @staticmethod
    def _try_import(module: str, symbol: str) -> bool:
        try:
            mod = __import__(module, fromlist=[symbol])
            getattr(mod, symbol)
            return True
        except Exception:
            return False