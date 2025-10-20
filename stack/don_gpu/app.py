from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from core import entropy_normalize

app = FastAPI(title="DON-GPU (mock)", version="0.1.0")

class NormReq(BaseModel):
    vector: List[float]
    mode: str = "entropy_norm"

@app.get("/health")
def health():
    return {"ok": True, "service": "don-gpu-mock"}

@app.post("/v1/normalize")
def normalize(req: NormReq):
    v = np.asarray(req.vector, dtype=float)
    out = entropy_normalize(v)
    return {"vector": out.tolist()}