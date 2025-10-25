"""
Pydantic schemas for bio endpoints
Following QAC pattern with sync flag, job system integration
"""

from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field


class BioExportRequest(BaseModel):
    """Request schema for export-artifacts endpoint"""
    cluster_key: str = Field(..., description="Key in adata.obs for cluster assignments")
    latent_key: str = Field(..., description="Key in adata.obsm for latent space (e.g., 'X_pca')")
    paga_key: Optional[str] = Field(None, description="Key in adata.uns for PAGA connectivity")
    sample_cells: Optional[int] = Field(None, description="Subsample to N cells (None = all cells)")
    sync: bool = Field(False, description="Execute synchronously or queue as job")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    project_id: Optional[str] = Field(None, description="Project identifier for memory logging")
    user_id: Optional[str] = Field(None, description="User identifier for audit")


class BioExportResponse(BaseModel):
    """Response schema for export-artifacts"""
    job_id: Optional[str] = Field(None, description="Job ID if async")
    nodes: int = Field(..., description="Number of cluster nodes")
    edges: int = Field(..., description="Number of edges in graph")
    vectors: int = Field(..., description="Number of cell vectors exported")
    artifacts: List[str] = Field(..., description="Paths to generated artifacts")
    status: str = Field(..., description="Job status: pending/completed/failed")
    message: Optional[str] = Field(None, description="Status message")


class BioSignalSyncRequest(BaseModel):
    """Request schema for signal-sync endpoint"""
    artifact1_path: str = Field(..., description="Path to first artifact (collapse_map.json)")
    artifact2_path: str = Field(..., description="Path to second artifact (collapse_map.json)")
    coherence_threshold: float = Field(0.8, description="Minimum coherence score (0-1)")
    sync: bool = Field(False, description="Execute synchronously or queue as job")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    project_id: Optional[str] = Field(None, description="Project identifier for memory logging")
    user_id: Optional[str] = Field(None, description="User identifier for audit")


class BioSignalSyncResponse(BaseModel):
    """Response schema for signal-sync"""
    job_id: Optional[str] = Field(None, description="Job ID if async")
    coherence_score: float = Field(..., description="Overall coherence (0-1)")
    node_overlap: float = Field(..., description="Node overlap ratio (0-1)")
    edge_consistency: float = Field(..., description="Edge consistency (0-1)")
    synchronized: bool = Field(..., description="Whether artifacts are coherent")
    report: Dict[str, Any] = Field(..., description="Detailed sync analysis")
    status: str = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Status message")


class BioParasiteDetectRequest(BaseModel):
    """Request schema for parasite-detect endpoint"""
    cluster_key: str = Field(..., description="Key in adata.obs for cluster assignments")
    batch_key: str = Field(..., description="Key in adata.obs for batch labels")
    ambient_threshold: float = Field(0.15, description="Ambient RNA threshold (0-1)")
    doublet_threshold: float = Field(0.25, description="Doublet enrichment threshold (0-1)")
    batch_threshold: float = Field(0.3, description="Batch contamination threshold (0-1)")
    sync: bool = Field(False, description="Execute synchronously or queue as job")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    project_id: Optional[str] = Field(None, description="Project identifier for memory logging")
    user_id: Optional[str] = Field(None, description="User identifier for audit")


class BioParasiteDetectResponse(BaseModel):
    """Response schema for parasite-detect"""
    job_id: Optional[str] = Field(None, description="Job ID if async")
    n_cells: int = Field(..., description="Total cells analyzed")
    n_flagged: int = Field(..., description="Cells flagged as contaminated")
    flags: List[bool] = Field(..., description="Per-cell contamination flags")
    parasite_score: float = Field(..., description="Overall contamination percentage")
    report: Dict[str, Any] = Field(..., description="Detailed QC report")
    thresholds: Dict[str, float] = Field(..., description="Applied thresholds")
    status: str = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Status message")


class BioEvolutionRequest(BaseModel):
    """Request schema for evolution/report endpoint"""
    run2_name: str = Field(..., description="Name/identifier for second run")
    cluster_key: str = Field(..., description="Key in adata.obs for cluster assignments")
    latent_key: str = Field(..., description="Key in adata.obsm for latent space")
    sync: bool = Field(False, description="Execute synchronously or queue as job")
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")
    project_id: Optional[str] = Field(None, description="Project identifier for memory logging")
    user_id: Optional[str] = Field(None, description="User identifier for audit")


class BioEvolutionResponse(BaseModel):
    """Response schema for evolution/report"""
    job_id: Optional[str] = Field(None, description="Job ID if async")
    run1_name: str = Field(..., description="First run identifier")
    run2_name: str = Field(..., description="Second run identifier")
    n_cells_run1: int = Field(..., description="Cell count in run 1")
    n_cells_run2: int = Field(..., description="Cell count in run 2")
    stability_score: float = Field(..., description="Overall stability (0-100)")
    delta_metrics: Dict[str, float] = Field(..., description="Delta metrics between runs")
    report: Dict[str, Any] = Field(..., description="Detailed evolution analysis")
    status: str = Field(..., description="Job status")
    message: Optional[str] = Field(None, description="Status message")


class BioJob(BaseModel):
    """Job status schema for bio endpoints"""
    job_id: str
    endpoint: str = Field(..., description="Which bio endpoint (export/signal-sync/parasite/evolution)")
    status: str = Field(..., description="pending/running/completed/failed")
    result: Optional[Dict[str, Any]] = Field(None, description="Job result if completed")
    error: Optional[str] = Field(None, description="Error message if failed")
    created_at: str
    completed_at: Optional[str] = None
    project_id: Optional[str] = None
    user_id: Optional[str] = None
