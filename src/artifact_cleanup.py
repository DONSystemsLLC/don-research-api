"""
Artifact cleanup module for DON Stack Research API
Removes old temporary files to prevent disk exhaustion on persistent storage.

Key Design Decisions:
- Preserves artifacts/memory/ directory (audit logs, never delete)
- Configurable retention period via retention_days parameter
- Graceful error handling (continues on permission errors)
- Detailed statistics for monitoring

Production Deployment:
- Run via APScheduler CronTrigger (daily at 2 AM UTC)
- Triggered on startup for immediate cleanup
- Respects ARTIFACTS_DIR environment variable
"""

import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def cleanup_old_artifacts(
    artifacts_dir: Path = None,
    retention_days: int = 30
) -> Dict[str, Any]:
    """
    Remove artifacts older than retention_days to prevent disk exhaustion.
    
    CRITICAL POLICY: Never delete artifacts/memory/ (audit logs for compliance).
    
    Args:
        artifacts_dir: Path to artifacts directory (default: ARTIFACTS_DIR env var or 'artifacts/')
        retention_days: Files older than this many days are deleted (default: 30)
    
    Returns:
        Dictionary with cleanup statistics:
        - deleted_count: Number of files deleted
        - bytes_freed: Total bytes freed
        - retention_days: Retention period used
        - artifacts_dir: Directory cleaned
        - error: Error message if partial failure
    
    File Types Cleaned:
        - .h5ad files (uploaded single-cell data)
        - .json files (metadata, artifacts)
        - .jsonl files (NOT in memory/ subdirectory)
        - .npz files (QAC model weights)
    
    Preserved Files:
        - artifacts/memory/*.jsonl (event logs)
        - artifacts/memory/*.md (human-readable logs)
        - artifacts/memory/*.sqlite (queryable logs)
        - Any files younger than retention_days
    
    Example:
        >>> stats = cleanup_old_artifacts(retention_days=30)
        >>> print(f"Deleted {stats['deleted_count']} files, freed {stats['bytes_freed']} bytes")
    """
    
    # Determine artifacts directory
    if artifacts_dir is None:
        artifacts_dir = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
    else:
        artifacts_dir = Path(artifacts_dir)
    
    # Handle non-existent directory gracefully
    if not artifacts_dir.exists():
        logger.info(f"🗑️  Artifacts directory does not exist: {artifacts_dir}")
        return {
            "deleted_count": 0,
            "bytes_freed": 0,
            "retention_days": retention_days,
            "artifacts_dir": str(artifacts_dir),
            "error": None
        }
    
    # Calculate cutoff timestamp
    cutoff = datetime.now() - timedelta(days=retention_days)
    cutoff_timestamp = cutoff.timestamp()
    
    # Statistics
    deleted_count = 0
    total_size_freed = 0
    errors = []
    
    # File extensions to clean
    extensions_to_clean = [".h5ad", ".json", ".jsonl", ".npz"]
    
    try:
        for ext in extensions_to_clean:
            for file_path in artifacts_dir.rglob(f"*{ext}"):
                try:
                    # CRITICAL: Never delete memory logs
                    if "memory" in file_path.parts:
                        logger.debug(f"Preserving memory log: {file_path}")
                        continue
                    
                    # Check file age
                    file_mtime = file_path.stat().st_mtime
                    if file_mtime < cutoff_timestamp:
                        # File is older than retention period
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        
                        deleted_count += 1
                        total_size_freed += file_size
                        
                        logger.debug(f"Deleted old file: {file_path} ({file_size} bytes)")
                    
                except PermissionError as e:
                    error_msg = f"Permission denied: {file_path}"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                    continue
                    
                except Exception as e:
                    error_msg = f"Error deleting {file_path}: {e}"
                    errors.append(error_msg)
                    logger.error(error_msg)
                    continue
        
        size_mb = total_size_freed / (1024 * 1024)
        logger.info(
            f"🗑️  Cleanup complete: {deleted_count} files deleted, "
            f"{size_mb:.2f} MB freed (retention: {retention_days} days)"
        )
        
        return {
            "deleted_count": deleted_count,
            "bytes_freed": total_size_freed,
            "retention_days": retention_days,
            "artifacts_dir": str(artifacts_dir),
            "error": errors[0] if errors else None,
            "error_count": len(errors)
        }
        
    except Exception as e:
        logger.error(f"Artifact cleanup failed: {e}")
        return {
            "deleted_count": deleted_count,
            "bytes_freed": total_size_freed,
            "retention_days": retention_days,
            "artifacts_dir": str(artifacts_dir),
            "error": str(e),
            "error_count": len(errors) + 1
        }
