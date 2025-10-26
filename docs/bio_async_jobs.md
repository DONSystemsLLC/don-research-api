# Bio Module Async Jobs

**Date:** 2025-10-26

## Export Artifacts Queue

- `POST /api/v1/bio/export-artifacts` now schedules a background job when `sync=false` is supplied.
- Artifacts and intermediate inputs are persisted under `artifacts/bio_jobs/<job_id>/` for downstream inspection.
- Job status is exposed through `GET /api/v1/bio/jobs/{job_id}` and includes vector counts plus trace identifiers.
- Trace telemetry continues to flow into the shared SQLite `TraceStorage`, preserving audit trails.
- Verified by test `tests/bio/test_routes.py::test_export_artifacts_async`.

## Operational Notes

- FastAPI `BackgroundTasks` handles execution in-process; Celery brokers can be plugged in later without changing the API contract.
- Failed jobs are marked as `failed`, record the error message, and emit a failed trace event for observability.
- Success paths attach generated artifact paths and metadata to the job record for quick reuse by downstream automations.
