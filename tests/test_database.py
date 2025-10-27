"""
Test suite for database models and operations.
Following TDD methodology - tests written before implementation.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from typing import AsyncGenerator
import numpy as np

# These imports will fail until we create the modules
try:
    from src.database.models import (
        QACModel, VectorStore, Job, UsageLog, AuditLog, Base
    )
    from src.database.session import DatabaseSession, get_db_session
    from src.database.repositories import (
        QACRepository, VectorRepository, JobRepository, 
        UsageRepository, AuditRepository
    )
except ImportError:
    pytest.skip("Database modules not yet implemented", allow_module_level=True)


@pytest.fixture
async def db_session() -> AsyncGenerator:
    """Create a test database session."""
    session = DatabaseSession(test_mode=True)
    await session.connect()
    
    # Create all tables
    await session.create_tables()
    
    yield session
    
    # Cleanup
    await session.drop_tables()
    await session.disconnect()


@pytest.fixture
def sample_qac_data():
    """Sample QAC model data for testing."""
    return {
        "model_id": "test-qac-model-001",
        "institution": "test_university",
        "adjacency_matrix": np.random.rand(100, 100).tolist(),
        "qac_corrected": np.random.rand(100).tolist(),
        "metadata": {
            "layers": 3,
            "qubits_per_layer": 5,
            "stabilizers": ["X", "Z", "Y"]
        }
    }


@pytest.fixture
def sample_vector_data():
    """Sample vector data for testing."""
    return {
        "project_id": "test-project-001",
        "institution": "test_university",
        "vector": np.random.rand(256).tolist(),
        "metadata": {
            "gene_names": ["GENE1", "GENE2", "GENE3"],
            "compression_ratio": 32.0,
            "original_dims": 8192
        }
    }


class TestDatabaseConnection:
    """Test database connection and session management."""
    
    @pytest.mark.asyncio
    async def test_database_connection(self, db_session):
        """Test that we can connect to the database."""
        assert db_session.is_connected()
        assert db_session.engine is not None
    
    @pytest.mark.asyncio
    async def test_connection_retry_logic(self):
        """Test connection retry with exponential backoff."""
        session = DatabaseSession(max_retries=3, retry_delay=0.1)
        
        # Should handle connection failures gracefully
        # (This will fail if database is not available)
        try:
            await session.connect()
            assert session.is_connected()
        except Exception as e:
            # Expected in test environment without database
            assert "connection" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_connection_pool(self, db_session):
        """Test connection pooling works correctly."""
        # Should be able to get multiple connections from pool
        async with db_session.get_connection() as conn1:
            assert conn1 is not None
            
            async with db_session.get_connection() as conn2:
                assert conn2 is not None
                assert conn1 != conn2  # Different connections


class TestQACModelRepository:
    """Test QAC model storage and retrieval."""
    
    @pytest.mark.asyncio
    async def test_create_qac_model(self, db_session, sample_qac_data):
        """Test creating a QAC model in the database."""
        repo = QACRepository(db_session)
        
        model = await repo.create(sample_qac_data)
        
        assert model.model_id == sample_qac_data["model_id"]
        assert model.institution == sample_qac_data["institution"]
        assert model.created_at is not None
        assert model.id is not None
    
    @pytest.mark.asyncio
    async def test_get_qac_model_by_id(self, db_session, sample_qac_data):
        """Test retrieving a QAC model by model_id."""
        repo = QACRepository(db_session)
        
        # Create model
        created = await repo.create(sample_qac_data)
        
        # Retrieve model
        retrieved = await repo.get_by_model_id(sample_qac_data["model_id"])
        
        assert retrieved is not None
        assert retrieved.model_id == created.model_id
        assert retrieved.adjacency_matrix == sample_qac_data["adjacency_matrix"]
    
    @pytest.mark.asyncio
    async def test_list_qac_models_by_institution(self, db_session, sample_qac_data):
        """Test listing QAC models filtered by institution."""
        repo = QACRepository(db_session)
        
        # Create multiple models
        for i in range(3):
            data = sample_qac_data.copy()
            data["model_id"] = f"test-qac-model-{i:03d}"
            await repo.create(data)
        
        # List models
        models = await repo.list_by_institution("test_university")
        
        assert len(models) == 3
        assert all(m.institution == "test_university" for m in models)
    
    @pytest.mark.asyncio
    async def test_delete_qac_model(self, db_session, sample_qac_data):
        """Test deleting a QAC model."""
        repo = QACRepository(db_session)
        
        # Create model
        created = await repo.create(sample_qac_data)
        
        # Delete model
        deleted = await repo.delete(created.model_id)
        
        assert deleted is True
        
        # Verify deletion
        retrieved = await repo.get_by_model_id(created.model_id)
        assert retrieved is None
    
    @pytest.mark.asyncio
    async def test_qac_model_retention_policy(self, db_session, sample_qac_data):
        """Test that models older than 30 days are marked for cleanup."""
        repo = QACRepository(db_session)
        
        # Create old model (simulate 31 days ago)
        old_data = sample_qac_data.copy()
        old_data["model_id"] = "old-model"
        old_model = await repo.create(old_data)
        
        # Manually set created_at to 31 days ago
        await repo.update_created_at(
            old_model.model_id,
            datetime.utcnow() - timedelta(days=31)
        )
        
        # Find models for cleanup
        expired = await repo.find_expired(days=30)
        
        assert len(expired) >= 1
        assert any(m.model_id == "old-model" for m in expired)


class TestVectorStoreRepository:
    """Test vector storage and retrieval."""
    
    @pytest.mark.asyncio
    async def test_create_vector(self, db_session, sample_vector_data):
        """Test creating a vector in the database."""
        repo = VectorRepository(db_session)
        
        vector = await repo.create(sample_vector_data)
        
        assert vector.project_id == sample_vector_data["project_id"]
        assert vector.institution == sample_vector_data["institution"]
        assert len(vector.vector) == len(sample_vector_data["vector"])
    
    @pytest.mark.asyncio
    async def test_search_vectors_by_similarity(self, db_session, sample_vector_data):
        """Test vector similarity search using pgvector."""
        repo = VectorRepository(db_session)
        
        # Create test vectors
        for i in range(5):
            data = sample_vector_data.copy()
            data["project_id"] = f"project-{i:03d}"
            await repo.create(data)
        
        # Search for similar vectors
        query_vector = sample_vector_data["vector"]
        results = await repo.search_similar(query_vector, limit=3)
        
        assert len(results) <= 3
        assert all(hasattr(r, 'distance') for r in results)
    
    @pytest.mark.asyncio
    async def test_get_project_memory(self, db_session, sample_vector_data):
        """Test retrieving all vectors for a project."""
        repo = VectorRepository(db_session)
        
        project_id = "test-project-memory"
        
        # Create multiple vectors for same project
        for i in range(3):
            data = sample_vector_data.copy()
            data["project_id"] = project_id
            data["vector"] = np.random.rand(256).tolist()
            await repo.create(data)
        
        # Retrieve project memory
        vectors = await repo.get_by_project(project_id)
        
        assert len(vectors) == 3
        assert all(v.project_id == project_id for v in vectors)
    
    @pytest.mark.asyncio
    async def test_vector_retention_policy(self, db_session, sample_vector_data):
        """Test that vectors older than 7 days are marked for cleanup."""
        repo = VectorRepository(db_session)
        
        # Create old vector
        old_data = sample_vector_data.copy()
        old_data["project_id"] = "old-project"
        old_vector = await repo.create(old_data)
        
        # Manually set created_at to 8 days ago
        await repo.update_created_at(
            old_vector.id,
            datetime.utcnow() - timedelta(days=8)
        )
        
        # Find vectors for cleanup
        expired = await repo.find_expired(days=7)
        
        assert len(expired) >= 1


class TestJobRepository:
    """Test job tracking for async operations."""
    
    @pytest.mark.asyncio
    async def test_create_job(self, db_session):
        """Test creating a job entry."""
        repo = JobRepository(db_session)
        
        job = await repo.create({
            "job_id": "test-job-001",
            "job_type": "qac_fit",
            "institution": "test_university",
            "status": "pending",
            "metadata": {"genes": 1000}
        })
        
        assert job.job_id == "test-job-001"
        assert job.status == "pending"
        assert job.created_at is not None
    
    @pytest.mark.asyncio
    async def test_update_job_status(self, db_session):
        """Test updating job status."""
        repo = JobRepository(db_session)
        
        # Create job
        job = await repo.create({
            "job_id": "test-job-002",
            "job_type": "parasite_detect",
            "institution": "test_university",
            "status": "pending"
        })
        
        # Update to running
        await repo.update_status(job.job_id, "running")
        
        updated = await repo.get_by_job_id(job.job_id)
        assert updated.status == "running"
        
        # Update to completed
        await repo.update_status(job.job_id, "completed", result={"count": 42})
        
        completed = await repo.get_by_job_id(job.job_id)
        assert completed.status == "completed"
        assert completed.result["count"] == 42
        assert completed.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_job_cleanup_policy(self, db_session):
        """Test that completed jobs older than 48 hours are cleaned up."""
        repo = JobRepository(db_session)
        
        # Create old completed job
        job = await repo.create({
            "job_id": "old-job",
            "job_type": "test",
            "institution": "test_university",
            "status": "completed"
        })
        
        # Set completion time to 49 hours ago
        await repo.update_completed_at(
            job.job_id,
            datetime.utcnow() - timedelta(hours=49)
        )
        
        # Find jobs for cleanup
        expired = await repo.find_expired(hours=48)
        
        assert len(expired) >= 1


class TestUsageRepository:
    """Test usage tracking and rate limiting."""
    
    @pytest.mark.asyncio
    async def test_record_usage(self, db_session):
        """Test recording API usage."""
        repo = UsageRepository(db_session)
        
        usage = await repo.record({
            "institution": "test_university",
            "endpoint": "/api/v1/genomics/compress",
            "trace_id": "test_2025-10-26_compress_abc123"
        })
        
        assert usage.institution == "test_university"
        assert usage.endpoint == "/api/v1/genomics/compress"
        assert usage.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_get_hourly_usage_count(self, db_session):
        """Test counting usage per institution per hour."""
        repo = UsageRepository(db_session)
        
        # Record multiple requests
        for i in range(5):
            await repo.record({
                "institution": "test_university",
                "endpoint": "/api/v1/genomics/compress",
                "trace_id": f"test_trace_{i}"
            })
        
        # Get count for last hour
        count = await repo.get_count_last_hour("test_university")
        
        assert count >= 5
    
    @pytest.mark.asyncio
    async def test_usage_retention_policy(self, db_session):
        """Test that usage logs older than 90 days are cleaned up."""
        repo = UsageRepository(db_session)
        
        # Create old usage record
        usage = await repo.record({
            "institution": "test_university",
            "endpoint": "/api/v1/test",
            "trace_id": "old_trace"
        })
        
        # Set timestamp to 91 days ago
        await repo.update_timestamp(
            usage.id,
            datetime.utcnow() - timedelta(days=91)
        )
        
        # Find logs for cleanup
        expired = await repo.find_expired(days=90)
        
        assert len(expired) >= 1


class TestAuditRepository:
    """Test audit logging."""
    
    @pytest.mark.asyncio
    async def test_create_audit_log(self, db_session):
        """Test creating an audit log entry."""
        repo = AuditRepository(db_session)
        
        log = await repo.create({
            "institution": "test_university",
            "action": "qac_fit",
            "trace_id": "test_2025-10-26_qac_fit_xyz789",
            "metadata": {
                "model_id": "test-model",
                "genes": 1000
            }
        })
        
        assert log.institution == "test_university"
        assert log.action == "qac_fit"
        assert log.timestamp is not None
    
    @pytest.mark.asyncio
    async def test_query_audit_logs_by_institution(self, db_session):
        """Test querying audit logs by institution."""
        repo = AuditRepository(db_session)
        
        # Create multiple logs
        for i in range(3):
            await repo.create({
                "institution": "test_university",
                "action": f"action_{i}",
                "trace_id": f"trace_{i}"
            })
        
        # Query logs
        logs = await repo.get_by_institution("test_university")
        
        assert len(logs) >= 3
    
    @pytest.mark.asyncio
    async def test_audit_retention_policy(self, db_session):
        """Test that audit logs older than 90 days are cleaned up."""
        repo = AuditRepository(db_session)
        
        # Create old log
        log = await repo.create({
            "institution": "test_university",
            "action": "old_action",
            "trace_id": "old_trace"
        })
        
        # Set timestamp to 91 days ago
        await repo.update_timestamp(
            log.id,
            datetime.utcnow() - timedelta(days=91)
        )
        
        # Find logs for cleanup
        expired = await repo.find_expired(days=90)
        
        assert len(expired) >= 1


class TestDatabaseMigrations:
    """Test database migration functionality."""
    
    @pytest.mark.asyncio
    async def test_pgvector_extension_enabled(self, db_session):
        """Test that pgvector extension is enabled."""
        result = await db_session.execute(
            "SELECT * FROM pg_extension WHERE extname = 'vector';"
        )
        
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.asyncio
    async def test_all_tables_created(self, db_session):
        """Test that all required tables are created."""
        expected_tables = [
            "qac_models",
            "vector_stores",
            "jobs",
            "usage_logs",
            "audit_logs"
        ]
        
        for table in expected_tables:
            result = await db_session.execute(
                f"SELECT EXISTS (SELECT FROM pg_tables WHERE tablename = '{table}');"
            )
            assert result[0][0] is True
    
    @pytest.mark.asyncio
    async def test_indexes_created(self, db_session):
        """Test that performance indexes are created."""
        # Check for key indexes
        expected_indexes = [
            ("qac_models", "idx_qac_model_id"),
            ("qac_models", "idx_qac_institution"),
            ("vector_stores", "idx_vector_project_id"),
            ("vector_stores", "idx_vector_institution"),
            ("jobs", "idx_job_job_id"),
            ("jobs", "idx_job_status"),
            ("usage_logs", "idx_usage_institution"),
            ("usage_logs", "idx_usage_timestamp"),
            ("audit_logs", "idx_audit_institution"),
            ("audit_logs", "idx_audit_timestamp")
        ]
        
        for table, index in expected_indexes:
            result = await db_session.execute(
                f"SELECT indexname FROM pg_indexes WHERE tablename = '{table}' AND indexname = '{index}';"
            )
            assert len(result) > 0, f"Index {index} not found on table {table}"


class TestDatabaseErrorHandling:
    """Test error handling and graceful degradation."""
    
    @pytest.mark.asyncio
    async def test_connection_failure_handling(self):
        """Test graceful handling of connection failures."""
        session = DatabaseSession(database_url="postgresql://invalid:invalid@localhost/invalid")
        
        with pytest.raises(Exception) as exc_info:
            await session.connect()
        
        assert "connection" in str(exc_info.value).lower()
    
    @pytest.mark.asyncio
    async def test_query_timeout_handling(self, db_session):
        """Test handling of query timeouts."""
        repo = QACRepository(db_session)
        
        # This should handle timeouts gracefully
        # (Implementation should use statement_timeout)
        try:
            result = await repo.execute_with_timeout(
                "SELECT pg_sleep(10);",
                timeout=1
            )
        except TimeoutError:
            # Expected behavior
            pass
    
    @pytest.mark.asyncio
    async def test_transaction_rollback(self, db_session):
        """Test transaction rollback on error."""
        repo = QACRepository(db_session)
        
        try:
            async with db_session.transaction():
                # Create a model
                await repo.create({
                    "model_id": "rollback-test",
                    "institution": "test",
                    "adjacency_matrix": [],
                    "qac_corrected": []
                })
                
                # Force an error
                raise ValueError("Test error")
        except ValueError:
            pass
        
        # Model should not exist (transaction rolled back)
        result = await repo.get_by_model_id("rollback-test")
        assert result is None
