from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from core import tune_alpha

app = FastAPI(title="TACE (mock)", version="0.1.0")

class TuneReq(BaseModel):
    tensions: List[float] = []
    default_alpha: float = 0.42

@app.get("/health")
def health():
    return {"ok": True, "service": "tace-mock"}

@app.post("/v1/tune")
def tune(req: TuneReq):
    return {"alpha": tune_alpha(req.tensions, req.default_alpha)}