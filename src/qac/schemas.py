from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict
from datetime import datetime

class QACParams(BaseModel):
    # Graph
    k_nn: int = Field(15, ge=1, le=128)
    weight: Literal['binary','gaussian'] = 'binary'
    sigma: Optional[float] = None
    # Real QAC (default)
    reinforce_rate: float = Field(0.05, ge=0.0, le=1.0)
    layers: int = Field(50, ge=1, le=100000)
    # Fallback Laplacian (only used if engine='laplace' or real_qac unavailable)
    beta: float = Field(0.7, ge=0.0, le=10.0)
    lambda_entropy: float = Field(0.05, ge=0.0, le=10.0)
    # Engine selection
    engine: Literal['real_qac','laplace'] = 'real_qac'

class QACFitRequest(BaseModel):
    embedding: List[List[float]]  # (n_cells, k)
    params: Optional[QACParams] = None
    seed: Optional[int] = None
    sync: Optional[bool] = False

class QACApplyRequest(BaseModel):
    model_id: str
    embedding: List[List[float]]  # must match n_cells
    seed: Optional[int] = None
    sync: Optional[bool] = False

class QACJob(BaseModel):
    id: str
    type: Literal['fit','apply']
    status: Literal['queued','running','succeeded','failed']
    progress: float = 0.0
    model_id: Optional[str] = None
    result: Optional[Dict] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime

class QACModelMeta(BaseModel):
    model_id: str
    n_cells: int
    k_nn: int
    weight: str
    reinforce_rate: float
    layers: int
    # laplace params retained for fallback transparency
    beta: float
    lambda_entropy: float
    created_at: datetime
    version: str = 'qac-1'
    engine: Literal['real_qac','laplace']