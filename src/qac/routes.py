from __future__ import annotations
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from datetime import datetime
from .schemas import QACFitRequest, QACApplyRequest
from .tasks import enqueue_fit_job, enqueue_apply_job, run_fit, run_apply
from .store import get_job, load_model

router = APIRouter(prefix="/api/v1/quantum/qac", tags=["quantum-qac"])

@router.post("/fit")
async def qac_fit(req: QACFitRequest):
    body = req.model_dump()
    body['created_at'] = datetime.utcnow().isoformat()
    if req.sync:
        try:
            out = run_fit(body)
            return {"status": "succeeded", **out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    jid = enqueue_fit_job(body)
    return {"status": "queued", "job_id": jid}

@router.post("/apply")
async def qac_apply(req: QACApplyRequest):
    body = req.model_dump()
    if req.sync:
        try:
            out = run_apply(body)
            return {"status": "succeeded", **out}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    jid = enqueue_apply_job(body)
    return {"status": "queued", "job_id": jid}

@router.get("/jobs/{job_id}")
async def qac_job(job_id: str):
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    return {
        "id": job['id'],
        "type": job['type'],
        "status": job['status'],
        "progress": job['progress'],
        "model_id": job.get('model_id'),
        "result": job.get('result'),
        "error": job.get('error'),
        "created_at": job['created_at'],
        "updated_at": job['updated_at']
    }

@router.get("/models/{model_id}")
async def qac_model(model_id: str):
    try:
        m = load_model(model_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="model not found")
    return {"meta": m['meta']}