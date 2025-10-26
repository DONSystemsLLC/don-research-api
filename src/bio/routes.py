"""
Bio module router
4 endpoints for ResoTrace integration: export, signal-sync, parasite-detect, evolution
Following genomics router pattern: UploadFile, Form, sync flag, job system
"""

import json
import logging
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile

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

from src.don_memory.dependencies import get_trace_storage
from src.don_memory.trace_storage import TraceStorage
from src.don_memory.system import get_system_health

import anndata as ad

logger = logging.getLogger(__name__)


# Router setup
router = APIRouter(prefix="/api/v1/bio", tags=["bio"])

# In-memory job store (replace with Redis/SQLite in production)
_bio_jobs: dict[str, BioJob] = {}
_ASYNC_ARTIFACT_ROOT = Path("artifacts") / "bio_jobs"


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
        created_at=_utc_now().isoformat(),
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
        job.completed_at = _utc_now().isoformat()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _health_snapshot() -> dict:
    return get_system_health()


def _log_memory_event(
    trace_storage: TraceStorage,
    event_type: str,
    project_id: Optional[str],
    user_id: Optional[str],
    metrics: Dict[str, Any],
    *,
    status: str = "completed",
    seed: Optional[int] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
    artifacts: Optional[Any] = None,
    parent_trace_id: Optional[str] = None,
) -> Optional[str]:
    """Persist bio event traces using TraceStorage."""

    if not trace_storage:
        return None

    trace_id = uuid.uuid4().hex
    started = started_at or _utc_now()
    finished = finished_at or started
    payload = {
        "id": trace_id,
        "project_id": project_id,
        "user_id": user_id,
        "event_type": event_type,
        "status": status,
        "metrics": metrics,
        "artifacts": artifacts or {},
        "engine_used": "bio_pipeline",
        "health": _health_snapshot(),
        "seed": seed,
        "started_at": started,
        "finished_at": finished,
    }

    try:
        trace_storage.store_trace(payload)
        if parent_trace_id:
            trace_storage.link(parent_trace_id, trace_id, "follows")
        return trace_id
    except Exception as exc:  # pragma: no cover - resilience guard
        logger.error("Failed to log bio memory event", exc_info=exc)
        return None


def _run_export_artifacts_job(
    job_id: str,
    h5ad_path: str,
    cluster_key: str,
    latent_key: str,
    paga_key: Optional[str],
    sample_cells: Optional[int],
    seed: int,
    project_id: Optional[str],
    user_id: Optional[str],
    output_dir: str,
    trace_storage: TraceStorage,
) -> None:
    """Process an export-artifacts job in the background."""

    _update_job(job_id, "running")
    started_at = _utc_now()

    try:
        result = export_artifacts(
            h5ad_path=h5ad_path,
            cluster_key=cluster_key,
            latent_key=latent_key,
            paga_key=paga_key,
            sample_cells=sample_cells,
            output_dir=output_dir,
            seed=seed,
        )
        finished_at = _utc_now()

        metrics = {
            "nodes": result["nodes"],
            "edges": result["edges"],
            "vectors": result["vectors"],
            "metadata": result.get("metadata", {}),
            "status": "completed",
        }

        trace_id = _log_memory_event(
            trace_storage,
            "export-artifacts",
            project_id,
            user_id,
            metrics,
            seed=seed,
            started_at=started_at,
            finished_at=finished_at,
            artifacts=result["artifacts"],
        )

        job_result: Dict[str, Any] = {
            "nodes": result["nodes"],
            "edges": result["edges"],
            "vectors": result["vectors"],
            "artifacts": result["artifacts"],
            "metadata": result.get("metadata", {}),
        }
        if trace_id:
            job_result["trace_id"] = trace_id

        _update_job(job_id, "completed", result=job_result)

    except Exception as exc:  # pragma: no cover - exercised via integration tests
        logger.exception("Async export job failed", exc_info=exc)
        failure_metrics = {
            "status": "failed",
            "error": str(exc),
        }
        _log_memory_event(
            trace_storage,
            "export-artifacts",
            project_id,
            user_id,
            failure_metrics,
            seed=seed,
            started_at=started_at,
            finished_at=_utc_now(),
            artifacts=None,
        )
        _update_job(job_id, "failed", error=str(exc))


# ------------------------------------------------------------------------------
# Endpoint 1: Export Artifacts (H5AD → collapse_map + vectors)
# ------------------------------------------------------------------------------

@router.post("/export-artifacts", response_model=BioExportResponse)
async def export_artifacts_endpoint(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="H5AD file to export"),
    cluster_key: str = Form(...),
    latent_key: str = Form(...),
    paga_key: Optional[str] = Form(None),
    sample_cells: Optional[int] = Form(None),
    sync: bool = Form(False),
    seed: int = Form(42),
    project_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    trace_storage: TraceStorage = Depends(get_trace_storage),
):
    """
    Export H5AD to collapse_map.json + collapse_vectors.jsonl
    
    Sync mode: immediate execution
    Async mode: returns job_id for polling
    """
    
    # Validate file extension
    if not file.filename.endswith('.h5ad'):
        raise HTTPException(status_code=400, detail="File must be .h5ad format")
    
    content = await file.read()

    if sync:
        # Create temp directory for processing
        with tempfile.TemporaryDirectory() as tmpdir:
            h5ad_path = Path(tmpdir) / file.filename
            with open(h5ad_path, 'wb') as f:
                f.write(content)

            artifacts_dir = Path(tmpdir) / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            try:
                started_at = _utc_now()
                result = export_artifacts(
                    h5ad_path=str(h5ad_path),
                    cluster_key=cluster_key,
                    latent_key=latent_key,
                    paga_key=paga_key,
                    sample_cells=sample_cells,
                    output_dir=str(artifacts_dir),
                    seed=seed
                )
                finished_at = _utc_now()

                response = BioExportResponse(
                    job_id=None,
                    nodes=result["nodes"],
                    edges=result["edges"],
                    vectors=result["vectors"],
                    artifacts=result["artifacts"],
                    status="completed",
                    message="Export completed successfully"
                )

                metrics = {
                    "nodes": result["nodes"],
                    "edges": result["edges"],
                    "vectors": result["vectors"],
                    "metadata": result.get("metadata", {}),
                    "status": "completed",
                }
                trace_id = _log_memory_event(
                    trace_storage,
                    "export-artifacts",
                    project_id,
                    user_id,
                    metrics,
                    seed=seed,
                    started_at=started_at,
                    finished_at=finished_at,
                    artifacts=result["artifacts"],
                )

                if trace_id:
                    response.message = f"Export completed successfully (trace: {trace_id})"

                return response

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")

    # Asynchronous execution - create job
    job_id = _create_job("export-artifacts", project_id, user_id)
    _ASYNC_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    job_dir = _ASYNC_ARTIFACT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = job_dir / file.filename
    with open(h5ad_path, 'wb') as f:
        f.write(content)

    artifacts_dir = job_dir / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    background_tasks.add_task(
        _run_export_artifacts_job,
        job_id,
        str(h5ad_path),
        cluster_key,
        latent_key,
        paga_key,
        sample_cells,
        seed,
        project_id,
        user_id,
        str(artifacts_dir),
        trace_storage,
    )

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
    user_id: Optional[str] = Form(None),
    trace_storage: TraceStorage = Depends(get_trace_storage),
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
            started_at = _utc_now()
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
            finished_at = _utc_now()
            metrics = {
                "coherence_score": coherence_score,
                "node_overlap": node_overlap,
                "edge_consistency": edge_consistency,
                "coherence_threshold": coherence_threshold,
                "synchronized": synchronized,
                "status": "completed",
            }
            trace_id = _log_memory_event(
                trace_storage,
                "signal-sync",
                project_id,
                user_id,
                metrics,
                seed=seed,
                started_at=started_at,
                finished_at=finished_at,
                artifacts={"report": report},
            )

            if trace_id:
                response.message = f"Signal sync completed (trace: {trace_id})"
            
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
    user_id: Optional[str] = Form(None),
    trace_storage: TraceStorage = Depends(get_trace_storage),
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
                started_at = _utc_now()
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
                finished_at = _utc_now()
                
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
                metrics = {
                    "n_cells": result["n_cells"],
                    "n_flagged": result["n_flagged"],
                    "parasite_score": result["parasite_score"],
                    "thresholds": result["thresholds"],
                    "status": "completed",
                }
                trace_id = _log_memory_event(
                    trace_storage,
                    "parasite-detect",
                    project_id,
                    user_id,
                    metrics,
                    seed=seed,
                    started_at=started_at,
                    finished_at=finished_at,
                    artifacts={"report": result["report"]},
                )

                if trace_id:
                    response.message = f"QC analysis completed (trace: {trace_id})"
                
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
    user_id: Optional[str] = Form(None),
    trace_storage: TraceStorage = Depends(get_trace_storage),
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
                started_at = _utc_now()
                # Compare runs
                result = compare_runs(
                    run1_path=str(run1_path),
                    run2_path=str(run2_path),
                    cluster_key=cluster_key,
                    latent_key=latent_key,
                    seed=seed
                )
                finished_at = _utc_now()
                
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
                metrics = {
                    "run1": run1_file.filename,
                    "run2": run2_name,
                    "n_cells_run1": result["n_cells_run1"],
                    "n_cells_run2": result["n_cells_run2"],
                    "stability_score": result["stability_score"],
                    "delta_metrics": result["delta_metrics"],
                    "status": "completed",
                }
                trace_id = _log_memory_event(
                    trace_storage,
                    "evolution-report",
                    project_id,
                    user_id,
                    metrics,
                    seed=seed,
                    started_at=started_at,
                    finished_at=finished_at,
                    artifacts={"report": result.get("report", {})},
                )

                if trace_id:
                    response.message = f"Evolution analysis completed (trace: {trace_id})"
                
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


# ------------------------------------------------------------------------------
# Memory Endpoint - Retrieve project traces
# ------------------------------------------------------------------------------

@router.get("/memory/{project_id}")
async def get_project_memory(
    project_id: str,
    limit: int = 100,
    trace_storage: TraceStorage = Depends(get_trace_storage),
):
    """
    Retrieve memory traces for a specific project.
    
    Returns traces ordered newest-first (by finished_at, then created_at).
    """
    traces = trace_storage.list_traces(project_id=project_id, limit=limit)
    
    return {
        "project_id": project_id,
        "count": len(traces),
        "traces": traces
    }
