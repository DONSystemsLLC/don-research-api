"""
Bio module router
4 endpoints for ResoTrace integration: export, signal-sync, parasite-detect, evolution
Following genomics router pattern: UploadFile, Form, sync flag, job system
"""

import json
import tempfile
import uuid
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from src.bio.schemas import (
    BioExportRequest,
    BioExportResponse,
    BioSignalSyncRequest,
    BioSignalSyncResponse,
    BioParasiteDetectRequest,
    BioParasiteDetectResponse,
    BioEvolutionRequest,
    BioEvolutionResponse,
    BioJob
)
from src.bio.adapter_anndata import export_artifacts
from src.bio.parasite_detector import detect_parasites
from src.bio.evolution import compare_runs

import anndata as ad


# Router setup
router = APIRouter(prefix="/api/v1/bio", tags=["bio"])

# In-memory job store (replace with Redis/SQLite in production)
_bio_jobs: dict[str, BioJob] = {}


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def _create_job(endpoint: str, project_id: Optional[str], user_id: Optional[str]) -> str:
    """Create a new job and return job_id"""
    job_id = str(uuid.uuid4())
    job = BioJob(
        job_id=job_id,
        endpoint=endpoint,
        status="pending",
        created_at=datetime.utcnow().isoformat(),
        project_id=project_id,
        user_id=user_id
    )
    _bio_jobs[job_id] = job
    return job_id


def _update_job(job_id: str, status: str, result: Optional[dict] = None, error: Optional[str] = None):
    """Update job status"""
    if job_id not in _bio_jobs:
        return
    
    job = _bio_jobs[job_id]
    job.status = status
    if result:
        job.result = result
    if error:
        job.error = error
    if status in ["completed", "failed"]:
        job.completed_at = datetime.utcnow().isoformat()


def _log_memory_event(endpoint: str, project_id: Optional[str], user_id: Optional[str], data: dict):
    """Log event to memory system (JSONL/MD/SQLite)"""
    # TODO: Integrate with existing memory logging in don_memory/
    # For now, just print for visibility
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "project_id": project_id,
        "user_id": user_id,
        "data": data
    }
    print(f"[MEMORY LOG] {json.dumps(event)}")


# ------------------------------------------------------------------------------
# Endpoint 1: Export Artifacts (H5AD → collapse_map + vectors)
# ------------------------------------------------------------------------------

@router.post("/export-artifacts", response_model=BioExportResponse)
async def export_artifacts_endpoint(
    file: UploadFile = File(..., description="H5AD file to export"),
    cluster_key: str = Form(...),
    latent_key: str = Form(...),
    paga_key: Optional[str] = Form(None),
    sample_cells: Optional[int] = Form(None),
    sync: bool = Form(False),
    seed: int = Form(42),
    project_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Export H5AD to collapse_map.json + collapse_vectors.jsonl
    
    Sync mode: immediate execution
    Async mode: returns job_id for polling
    """
    
    # Validate file extension
    if not file.filename.endswith('.h5ad'):
        raise HTTPException(status_code=400, detail="File must be .h5ad format")
    
    # Create temp directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded file
        h5ad_path = Path(tmpdir) / file.filename
        with open(h5ad_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Artifacts output directory
        artifacts_dir = Path(tmpdir) / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        
        if sync:
            # Synchronous execution
            try:
                result = export_artifacts(
                    h5ad_path=str(h5ad_path),
                    cluster_key=cluster_key,
                    latent_key=latent_key,
                    paga_key=paga_key,
                    sample_cells=sample_cells,
                    output_dir=str(artifacts_dir),
                    seed=seed
                )
                
                response = BioExportResponse(
                    job_id=None,
                    nodes=result["nodes"],
                    edges=result["edges"],
                    vectors=result["vectors"],
                    artifacts=result["artifacts"],
                    status="completed",
                    message="Export completed successfully"
                )
                
                # Log to memory
                _log_memory_event("export-artifacts", project_id, user_id, {
                    "nodes": result["nodes"],
                    "vectors": result["vectors"],
                    "status": "completed"
                })
                
                return response
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
        
        else:
            # Asynchronous execution - create job
            job_id = _create_job("export-artifacts", project_id, user_id)
            
            # TODO: Queue job with Celery/background task
            # For now, just return pending status
            
            return BioExportResponse(
                job_id=job_id,
                nodes=0,
                edges=0,
                vectors=0,
                artifacts=[],
                status="pending",
                message="Job queued for processing"
            )


# ------------------------------------------------------------------------------
# Endpoint 2: Signal Sync (cross-artifact coherence)
# ------------------------------------------------------------------------------

@router.post("/signal-sync", response_model=BioSignalSyncResponse)
async def signal_sync_endpoint(
    artifact1: UploadFile = File(..., description="First collapse_map.json"),
    artifact2: UploadFile = File(..., description="Second collapse_map.json"),
    coherence_threshold: float = Form(0.8),
    sync: bool = Form(False),
    seed: int = Form(42),
    project_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Compute cross-artifact coherence and synchronization
    
    Checks node overlap, edge consistency, cluster alignment
    """
    
    # Read artifacts
    try:
        content1 = await artifact1.read()
        content2 = await artifact2.read()
        map1 = json.loads(content1)
        map2 = json.loads(content2)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid artifact format: {str(e)}")
    
    if sync:
        # Synchronous execution
        try:
            # Compute coherence metrics
            nodes1 = set(node["id"] for node in map1["nodes"])
            nodes2 = set(node["id"] for node in map2["nodes"])
            node_overlap = len(nodes1 & nodes2) / max(len(nodes1), len(nodes2))
            
            # Edge consistency (simplified)
            edges1 = {(e["source"], e["target"]) for e in map1.get("edges", [])}
            edges2 = {(e["source"], e["target"]) for e in map2.get("edges", [])}
            if edges1 or edges2:
                edge_consistency = len(edges1 & edges2) / max(len(edges1), len(edges2))
            else:
                edge_consistency = 1.0
            
            # Overall coherence
            coherence_score = (node_overlap + edge_consistency) / 2
            synchronized = coherence_score >= coherence_threshold
            
            report = {
                "nodes_artifact1": len(nodes1),
                "nodes_artifact2": len(nodes2),
                "shared_nodes": len(nodes1 & nodes2),
                "edges_artifact1": len(edges1),
                "edges_artifact2": len(edges2),
                "shared_edges": len(edges1 & edges2)
            }
            
            response = BioSignalSyncResponse(
                job_id=None,
                coherence_score=coherence_score,
                node_overlap=node_overlap,
                edge_consistency=edge_consistency,
                synchronized=synchronized,
                report=report,
                status="completed",
                message="Signal sync completed"
            )
            
            # Log to memory
            _log_memory_event("signal-sync", project_id, user_id, {
                "coherence_score": coherence_score,
                "synchronized": synchronized,
                "status": "completed"
            })
            
            return response
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Signal sync failed: {str(e)}")
    
    else:
        # Async mode
        job_id = _create_job("signal-sync", project_id, user_id)
        return BioSignalSyncResponse(
            job_id=job_id,
            coherence_score=0.0,
            node_overlap=0.0,
            edge_consistency=0.0,
            synchronized=False,
            report={},
            status="pending",
            message="Job queued"
        )


# ------------------------------------------------------------------------------
# Endpoint 3: Parasite Detector (QC contamination)
# ------------------------------------------------------------------------------

@router.post("/qc/parasite-detect", response_model=BioParasiteDetectResponse)
async def parasite_detect_endpoint(
    file: UploadFile = File(..., description="H5AD file for QC"),
    cluster_key: str = Form(...),
    batch_key: str = Form(...),
    ambient_threshold: float = Form(0.15),
    doublet_threshold: float = Form(0.25),
    batch_threshold: float = Form(0.3),
    sync: bool = Form(False),
    seed: int = Form(42),
    project_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Detect ambient RNA, doublets, batch contamination
    
    Returns per-cell flags and overall parasite score
    """
    
    if not file.filename.endswith('.h5ad'):
        raise HTTPException(status_code=400, detail="File must be .h5ad format")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        h5ad_path = Path(tmpdir) / file.filename
        with open(h5ad_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        if sync:
            try:
                # Load AnnData
                adata = ad.read_h5ad(h5ad_path)
                
                # Run detection
                result = detect_parasites(
                    adata=adata,
                    cluster_key=cluster_key,
                    batch_key=batch_key,
                    ambient_threshold=ambient_threshold,
                    doublet_threshold=doublet_threshold,
                    batch_threshold=batch_threshold,
                    seed=seed
                )
                
                response = BioParasiteDetectResponse(
                    job_id=None,
                    n_cells=result["n_cells"],
                    n_flagged=result["n_flagged"],
                    flags=result["flags"],
                    parasite_score=result["parasite_score"],
                    report=result["report"],
                    thresholds=result["thresholds"],
                    status="completed",
                    message="QC analysis completed"
                )
                
                # Log to memory
                _log_memory_event("parasite-detect", project_id, user_id, {
                    "n_cells": result["n_cells"],
                    "n_flagged": result["n_flagged"],
                    "parasite_score": result["parasite_score"],
                    "status": "completed"
                })
                
                return response
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Parasite detection failed: {str(e)}")
        
        else:
            job_id = _create_job("parasite-detect", project_id, user_id)
            return BioParasiteDetectResponse(
                job_id=job_id,
                n_cells=0,
                n_flagged=0,
                flags=[],
                parasite_score=0.0,
                report={},
                thresholds={},
                status="pending",
                message="Job queued"
            )


# ------------------------------------------------------------------------------
# Endpoint 4: Evolution Report (run-over-run stability)
# ------------------------------------------------------------------------------

@router.post("/evolution/report", response_model=BioEvolutionResponse)
async def evolution_report_endpoint(
    run1_file: UploadFile = File(..., description="First run H5AD"),
    run2_file: UploadFile = File(..., description="Second run H5AD"),
    run2_name: str = Form(...),
    cluster_key: str = Form(...),
    latent_key: str = Form(...),
    sync: bool = Form(False),
    seed: int = Form(42),
    project_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """
    Compare two pipeline runs for stability and drift
    
    Returns delta metrics and overall stability score
    """
    
    if not (run1_file.filename.endswith('.h5ad') and run2_file.filename.endswith('.h5ad')):
        raise HTTPException(status_code=400, detail="Both files must be .h5ad format")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        run1_path = Path(tmpdir) / run1_file.filename
        run2_path = Path(tmpdir) / run2_file.filename
        
        with open(run1_path, 'wb') as f:
            f.write(await run1_file.read())
        with open(run2_path, 'wb') as f:
            f.write(await run2_file.read())
        
        if sync:
            try:
                # Compare runs
                result = compare_runs(
                    run1_path=str(run1_path),
                    run2_path=str(run2_path),
                    cluster_key=cluster_key,
                    latent_key=latent_key,
                    seed=seed
                )
                
                response = BioEvolutionResponse(
                    job_id=None,
                    run1_name=run1_file.filename,
                    run2_name=run2_name,
                    n_cells_run1=result["n_cells_run1"],
                    n_cells_run2=result["n_cells_run2"],
                    stability_score=result["stability_score"],
                    delta_metrics=result["delta_metrics"],
                    report=result.get("report", {}),
                    status="completed",
                    message="Evolution analysis completed"
                )
                
                # Log to memory
                _log_memory_event("evolution-report", project_id, user_id, {
                    "run1": run1_file.filename,
                    "run2": run2_name,
                    "stability_score": result["stability_score"],
                    "status": "completed"
                })
                
                return response
                
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Evolution analysis failed: {str(e)}")
        
        else:
            job_id = _create_job("evolution-report", project_id, user_id)
            return BioEvolutionResponse(
                job_id=job_id,
                run1_name=run1_file.filename,
                run2_name=run2_name,
                n_cells_run1=0,
                n_cells_run2=0,
                stability_score=0.0,
                delta_metrics={},
                report={},
                status="pending",
                message="Job queued"
            )


# ------------------------------------------------------------------------------
# Job Status Endpoint
# ------------------------------------------------------------------------------

@router.get("/jobs/{job_id}", response_model=BioJob)
async def get_job_status(job_id: str):
    """Get status of async job"""
    if job_id not in _bio_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return _bio_jobs[job_id]
