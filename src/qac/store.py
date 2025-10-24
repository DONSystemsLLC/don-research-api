from __future__ import annotations
import json, os, uuid, time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np

ART_DIR = Path(os.getenv('ARTIFACTS_DIR', 'artifacts'))
MODEL_DIR = ART_DIR / 'qac_models'
JOB_PATH  = ART_DIR / 'qac_jobs.json'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

_jobs: Dict[str, Dict[str, Any]] = {}


def _now() -> float:
    return time.time()


def save_jobs() -> None:
    JOB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = {k: {**v, 'result': None} for k, v in _jobs.items()}  # keep disk light
    JOB_PATH.write_text(json.dumps(tmp, indent=2))


def create_job(job_type: str, model_id: Optional[str] = None) -> str:
    jid = str(uuid.uuid4())
    _jobs[jid] = {
        'id': jid,
        'type': job_type,
        'status': 'queued',
        'progress': 0.0,
        'model_id': model_id,
        'result': None,
        'error': None,
        'created_at': _now(),
        'updated_at': _now(),
    }
    save_jobs()
    return jid


def update_job(jid: str, **fields) -> None:
    if jid in _jobs:
        _jobs[jid].update(fields)
        _jobs[jid]['updated_at'] = _now()
        save_jobs()


def get_job(jid: str) -> Optional[Dict[str, Any]]:
    return _jobs.get(jid)


def save_model(model_id: str, L_meta: Dict[str, Any], idx, w) -> None:
    # Persist neighbor structure (idx, weights) and metadata as NPZ
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_DIR / f"{model_id}.npz", idx=idx, w=w)
    (MODEL_DIR / f"{model_id}.json").write_text(json.dumps(L_meta, indent=2))


def load_model(model_id: str) -> Dict[str, Any]:
    import numpy as np
    meta_path = MODEL_DIR / f"{model_id}.json"
    npz_path  = MODEL_DIR / f"{model_id}.npz"
    if not (meta_path.exists() and npz_path.exists()):
        raise FileNotFoundError(f"QAC model {model_id} not found")
    meta = json.loads(meta_path.read_text())
    npz = np.load(npz_path, allow_pickle=False)
    return {"meta": meta, "idx": npz['idx'], "w": npz['w']}