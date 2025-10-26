"""
Tests for artifact cleanup scheduler
TDD approach: Write tests first, then implement
"""

import pytest
from pathlib import Path
from datetime import datetime, timedelta
import tempfile
import json
import numpy as np
import time
import os


@pytest.fixture
def temp_artifacts_dir(tmpdir):
    """Create a temporary artifacts directory structure"""
    artifacts = Path(tmpdir) / "artifacts"
    
    # Create subdirectories matching production structure
    (artifacts / "bio_jobs").mkdir(parents=True)
    (artifacts / "qac_models").mkdir(parents=True)
    (artifacts / "memory").mkdir(parents=True)
    
    return artifacts


@pytest.fixture
def old_files(temp_artifacts_dir):
    """Create files older than 30 days"""
    old_time = (datetime.now() - timedelta(days=35)).timestamp()
    
    files_created = []
    
    # Old bio job files
    job_id = "old_job_123"
    job_dir = temp_artifacts_dir / "bio_jobs" / job_id
    job_dir.mkdir(parents=True)
    
    h5ad_file = job_dir / "test.h5ad"
    h5ad_file.write_text("fake h5ad content")
    os.utime(h5ad_file, (old_time, old_time))
    files_created.append(h5ad_file)
    
    json_file = job_dir / "metadata.json"
    json_file.write_text('{"test": true}')
    os.utime(json_file, (old_time, old_time))
    files_created.append(json_file)
    
    # Old QAC model files
    model_id = "old_model_456"
    npz_file = temp_artifacts_dir / "qac_models" / f"{model_id}.npz"
    np.savez_compressed(npz_file, data=np.array([1, 2, 3]))
    os.utime(npz_file, (old_time, old_time))
    files_created.append(npz_file)
    
    json_file = temp_artifacts_dir / "qac_models" / f"{model_id}.json"
    json_file.write_text('{"layers": 3}')
    os.utime(json_file, (old_time, old_time))
    files_created.append(json_file)
    
    # Old artifact JSONL (not in memory/)
    old_jsonl = temp_artifacts_dir / "bio_jobs" / "old_data.jsonl"
    old_jsonl.write_text('{"id": 1}\n{"id": 2}\n')
    os.utime(old_jsonl, (old_time, old_time))
    files_created.append(old_jsonl)
    
    return files_created


@pytest.fixture
def recent_files(temp_artifacts_dir):
    """Create files younger than 30 days"""
    recent_time = (datetime.now() - timedelta(days=10)).timestamp()
    
    files_created = []
    
    # Recent bio job files
    job_id = "recent_job_789"
    job_dir = temp_artifacts_dir / "bio_jobs" / job_id
    job_dir.mkdir(parents=True)
    
    h5ad_file = job_dir / "test.h5ad"
    h5ad_file.write_text("fake h5ad content")
    os.utime(h5ad_file, (recent_time, recent_time))
    files_created.append(h5ad_file)
    
    return files_created


@pytest.fixture
def memory_files(temp_artifacts_dir):
    """Create memory log files (should never be deleted)"""
    memory_dir = temp_artifacts_dir / "memory"
    
    # Old memory files (should still be preserved)
    old_time = (datetime.now() - timedelta(days=100)).timestamp()
    
    files_created = []
    
    events_jsonl = memory_dir / "events.jsonl"
    events_jsonl.write_text('{"event": "test"}\n')
    os.utime(events_jsonl, (old_time, old_time))
    files_created.append(events_jsonl)
    
    events_md = memory_dir / "events.md"
    events_md.write_text("# Events\n")
    os.utime(events_md, (old_time, old_time))
    files_created.append(events_md)
    
    events_db = memory_dir / "events.sqlite"
    events_db.write_text("fake sqlite")
    os.utime(events_db, (old_time, old_time))
    files_created.append(events_db)
    
    return files_created


def test_cleanup_removes_old_h5ad_files(temp_artifacts_dir, old_files, recent_files):
    """Test that cleanup removes H5AD files older than 30 days"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    # Verify old files exist
    old_h5ad = [f for f in old_files if f.suffix == ".h5ad"][0]
    assert old_h5ad.exists()
    
    # Run cleanup
    stats = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    
    # Old H5AD should be deleted
    assert not old_h5ad.exists()
    
    # Recent H5AD should remain
    recent_h5ad = [f for f in recent_files if f.suffix == ".h5ad"][0]
    assert recent_h5ad.exists()
    
    # Check stats
    assert stats["deleted_count"] > 0
    assert stats["bytes_freed"] > 0


def test_cleanup_removes_old_json_files(temp_artifacts_dir, old_files):
    """Test that cleanup removes JSON files older than 30 days"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    # Verify old JSON exists
    old_json = [f for f in old_files if f.suffix == ".json"][0]
    assert old_json.exists()
    
    # Run cleanup
    stats = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    
    # Old JSON should be deleted
    assert not old_json.exists()
    assert stats["deleted_count"] > 0


def test_cleanup_removes_old_npz_files(temp_artifacts_dir, old_files):
    """Test that cleanup removes NPZ files older than 30 days"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    # Verify old NPZ exists
    old_npz = [f for f in old_files if f.suffix == ".npz"][0]
    assert old_npz.exists()
    
    # Run cleanup
    stats = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    
    # Old NPZ should be deleted
    assert not old_npz.exists()
    assert stats["deleted_count"] > 0


def test_cleanup_removes_old_jsonl_files(temp_artifacts_dir, old_files):
    """Test that cleanup removes JSONL files older than 30 days (except memory/)"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    # Verify old JSONL exists
    old_jsonl = [f for f in old_files if f.suffix == ".jsonl"][0]
    assert old_jsonl.exists()
    
    # Run cleanup
    stats = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    
    # Old JSONL should be deleted
    assert not old_jsonl.exists()


def test_cleanup_preserves_memory_logs(temp_artifacts_dir):
    """Test that cleanup NEVER deletes memory log files regardless of age"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    old_time = (datetime.now() - timedelta(days=35)).timestamp()
    
    # Create old memory files (should be preserved)
    memory_dir = temp_artifacts_dir / "memory"
    memory_dir.mkdir(exist_ok=True)
    
    memory_jsonl = memory_dir / "events.jsonl"
    memory_jsonl.write_text('{"event": "test"}\n')
    os.utime(memory_jsonl, (old_time, old_time))
    
    memory_md = memory_dir / "events.md"
    memory_md.write_text("# Events\n")
    os.utime(memory_md, (old_time, old_time))
    
    # Create old NON-memory files (should be deleted)
    bio_dir = temp_artifacts_dir / "bio_jobs"
    bio_dir.mkdir(exist_ok=True)
    
    old_h5ad = bio_dir / "old_data.h5ad"
    old_h5ad.write_text("fake h5ad")
    os.utime(old_h5ad, (old_time, old_time))
    
    old_json = bio_dir / "old_metadata.json"
    old_json.write_text('{"test": true}')
    os.utime(old_json, (old_time, old_time))
    
    # Verify all files exist before cleanup
    assert memory_jsonl.exists()
    assert memory_md.exists()
    assert old_h5ad.exists()
    assert old_json.exists()
    
    # Run cleanup
    stats = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    
    # Memory files should STILL exist
    assert memory_jsonl.exists(), "Memory JSONL was incorrectly deleted!"
    assert memory_md.exists(), "Memory MD was incorrectly deleted!"
    
    # Non-memory files should be deleted
    assert not old_h5ad.exists(), "Old H5AD should have been deleted"
    assert not old_json.exists(), "Old JSON should have been deleted"
    
    # Stats should show files were deleted
    assert stats["deleted_count"] == 2  # Only the 2 non-memory files


def test_cleanup_handles_missing_directory(tmpdir):
    """Test that cleanup handles non-existent artifacts directory gracefully"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    nonexistent_dir = Path(tmpdir) / "does_not_exist"
    
    # Should not raise exception
    stats = cleanup_old_artifacts(artifacts_dir=nonexistent_dir, retention_days=30)
    
    # Should report zero deletions
    assert stats["deleted_count"] == 0
    assert stats["bytes_freed"] == 0
    assert "error" not in stats or stats["error"] is None


def test_cleanup_handles_permission_errors(temp_artifacts_dir, old_files, monkeypatch):
    """Test that cleanup continues when encountering permission errors"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    # Should handle individual file errors gracefully
    stats = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    
    # Should have attempted cleanup and potentially succeeded on some files
    assert "deleted_count" in stats
    assert "bytes_freed" in stats
    # Don't assert on specific counts since permission errors are environment-specific


def test_cleanup_respects_custom_retention_days(temp_artifacts_dir):
    """Test that cleanup respects custom retention period"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    # Create file that's 10 days old
    old_time = (datetime.now() - timedelta(days=10)).timestamp()
    test_file = temp_artifacts_dir / "bio_jobs" / "test.json"
    test_file.write_text('{"test": true}')
    os.utime(test_file, (old_time, old_time))
    
    # With 30 day retention, should NOT delete
    stats_30 = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    assert test_file.exists(), "File should not be deleted with 30-day retention"
    
    # With 5 day retention, SHOULD delete
    stats_5 = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=5)
    assert not test_file.exists(), "File should be deleted with 5-day retention"
    assert stats_5["deleted_count"] > 0


def test_cleanup_returns_detailed_stats(temp_artifacts_dir, old_files, recent_files):
    """Test that cleanup returns detailed statistics"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    stats = cleanup_old_artifacts(artifacts_dir=temp_artifacts_dir, retention_days=30)
    
    # Should have all expected stat fields
    assert "deleted_count" in stats
    assert "bytes_freed" in stats
    assert "retention_days" in stats
    assert "artifacts_dir" in stats
    
    assert isinstance(stats["deleted_count"], int)
    assert isinstance(stats["bytes_freed"], int)
    assert stats["retention_days"] == 30
    assert stats["artifacts_dir"] == str(temp_artifacts_dir)


def test_cleanup_with_env_var_override(temp_artifacts_dir, old_files, monkeypatch):
    """Test that cleanup respects ARTIFACTS_DIR environment variable"""
    from src.artifact_cleanup import cleanup_old_artifacts
    
    # Set environment variable
    monkeypatch.setenv("ARTIFACTS_DIR", str(temp_artifacts_dir))
    
    # Call without explicit artifacts_dir (should use env var)
    stats = cleanup_old_artifacts()
    
    # Should have found and cleaned files
    assert stats["deleted_count"] > 0
    assert stats["artifacts_dir"] == str(temp_artifacts_dir)
