"""
Repository pattern for database operations.
Provides clean abstraction layer over SQLAlchemy models.
"""
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from sqlalchemy import select, delete, update, and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession

from .models import QACModel, VectorStore, Job, UsageLog, AuditLog

logger = logging.getLogger(__name__)


class BaseRepository:
    """Base repository with common CRUD operations."""
    
    def __init__(self, session: AsyncSession, model_class):
        self.session = session
        self.model_class = model_class
    
    async def create(self, data: Dict[str, Any]):
        """Create a new record."""
        obj = self.model_class(**data)
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj
    
    async def get_by_id(self, id: int):
        """Get record by primary key."""
        result = await self.session.execute(
            select(self.model_class).where(self.model_class.id == id)
        )
        return result.scalar_one_or_none()
    
    async def list_all(self, limit: int = 100):
        """List all records with limit."""
        result = await self.session.execute(
            select(self.model_class)
            .where(self.model_class.is_deleted == False)
            .limit(limit)
        )
        return result.scalars().all()
    
    async def soft_delete(self, id: int) -> bool:
        """Soft delete a record."""
        await self.session.execute(
            update(self.model_class)
            .where(self.model_class.id == id)
            .values(is_deleted=True, deleted_at=datetime.utcnow())
        )
        return True
    
    async def hard_delete(self, id: int) -> bool:
        """Permanently delete a record."""
        await self.session.execute(
            delete(self.model_class).where(self.model_class.id == id)
        )
        return True


class QACRepository(BaseRepository):
    """Repository for QAC model operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, QACModel)
    
    async def get_by_model_id(self, model_id: str) -> Optional[QACModel]:
        """Get QAC model by model_id."""
        result = await self.session.execute(
            select(QACModel)
            .where(
                and_(
                    QACModel.model_id == model_id,
                    QACModel.is_deleted == False
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def list_by_institution(
        self, 
        institution: str, 
        limit: int = 100
    ) -> List[QACModel]:
        """List QAC models by institution."""
        result = await self.session.execute(
            select(QACModel)
            .where(
                and_(
                    QACModel.institution == institution,
                    QACModel.is_deleted == False
                )
            )
            .order_by(QACModel.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def delete(self, model_id: str) -> bool:
        """Soft delete QAC model."""
        result = await self.session.execute(
            update(QACModel)
            .where(QACModel.model_id == model_id)
            .values(is_deleted=True, deleted_at=datetime.utcnow())
        )
        return result.rowcount > 0
    
    async def find_expired(self, days: int = 30) -> List[QACModel]:
        """Find models older than specified days for cleanup."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(QACModel)
            .where(
                and_(
                    QACModel.created_at < cutoff_date,
                    QACModel.is_deleted == False
                )
            )
        )
        return result.scalars().all()
    
    async def update_created_at(self, model_id: str, created_at: datetime):
        """Update created_at timestamp (for testing)."""
        await self.session.execute(
            update(QACModel)
            .where(QACModel.model_id == model_id)
            .values(created_at=created_at)
        )
    
    async def execute_with_timeout(self, query: str, timeout: int = 30):
        """Execute query with timeout."""
        await self.session.execute(f"SET statement_timeout = {timeout * 1000};")
        try:
            result = await self.session.execute(query)
            return result
        finally:
            await self.session.execute("RESET statement_timeout;")


class VectorRepository(BaseRepository):
    """Repository for vector storage operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, VectorStore)
    
    async def get_by_project(
        self, 
        project_id: str,
        limit: int = 100
    ) -> List[VectorStore]:
        """Get all vectors for a project."""
        result = await self.session.execute(
            select(VectorStore)
            .where(
                and_(
                    VectorStore.project_id == project_id,
                    VectorStore.is_deleted == False
                )
            )
            .order_by(VectorStore.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def search_similar(
        self,
        query_vector: List[float],
        institution: Optional[str] = None,
        limit: int = 10
    ) -> List[tuple]:
        """
        Search for similar vectors using pgvector.
        
        Returns list of (VectorStore, distance) tuples.
        """
        # Build query with vector similarity
        query = select(
            VectorStore,
            VectorStore.vector.l2_distance(query_vector).label('distance')
        ).where(VectorStore.is_deleted == False)
        
        if institution:
            query = query.where(VectorStore.institution == institution)
        
        query = query.order_by('distance').limit(limit)
        
        result = await self.session.execute(query)
        return [(row[0], row[1]) for row in result.all()]
    
    async def list_by_institution(
        self,
        institution: str,
        limit: int = 100
    ) -> List[VectorStore]:
        """List vectors by institution."""
        result = await self.session.execute(
            select(VectorStore)
            .where(
                and_(
                    VectorStore.institution == institution,
                    VectorStore.is_deleted == False
                )
            )
            .order_by(VectorStore.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def find_expired(self, days: int = 7) -> List[VectorStore]:
        """Find vectors older than specified days for cleanup."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(VectorStore)
            .where(
                and_(
                    VectorStore.created_at < cutoff_date,
                    VectorStore.is_deleted == False
                )
            )
        )
        return result.scalars().all()
    
    async def update_created_at(self, id: int, created_at: datetime):
        """Update created_at timestamp (for testing)."""
        await self.session.execute(
            update(VectorStore)
            .where(VectorStore.id == id)
            .values(created_at=created_at)
        )


class JobRepository(BaseRepository):
    """Repository for job tracking operations."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, Job)
    
    async def get_by_job_id(self, job_id: str) -> Optional[Job]:
        """Get job by job_id."""
        result = await self.session.execute(
            select(Job)
            .where(
                and_(
                    Job.job_id == job_id,
                    Job.is_deleted == False
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def list_by_status(
        self,
        status: str,
        institution: Optional[str] = None,
        limit: int = 100
    ) -> List[Job]:
        """List jobs by status."""
        query = select(Job).where(
            and_(
                Job.status == status,
                Job.is_deleted == False
            )
        )
        
        if institution:
            query = query.where(Job.institution == institution)
        
        query = query.order_by(Job.created_at.desc()).limit(limit)
        
        result = await self.session.execute(query)
        return result.scalars().all()
    
    async def update_status(
        self,
        job_id: str,
        status: str,
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        progress: Optional[float] = None
    ) -> bool:
        """Update job status."""
        values = {"status": status, "updated_at": datetime.utcnow()}
        
        if status == "running" and not await self._has_started(job_id):
            values["started_at"] = datetime.utcnow()
        
        if status in ["completed", "failed", "cancelled"]:
            values["completed_at"] = datetime.utcnow()
        
        if result is not None:
            values["result"] = result
        
        if error is not None:
            values["error"] = error
        
        if progress is not None:
            values["progress"] = progress
        
        result = await self.session.execute(
            update(Job)
            .where(Job.job_id == job_id)
            .values(**values)
        )
        
        return result.rowcount > 0
    
    async def _has_started(self, job_id: str) -> bool:
        """Check if job has started_at timestamp."""
        result = await self.session.execute(
            select(Job.started_at)
            .where(Job.job_id == job_id)
        )
        started_at = result.scalar_one_or_none()
        return started_at is not None
    
    async def find_expired(self, hours: int = 48) -> List[Job]:
        """Find completed jobs older than specified hours."""
        cutoff_date = datetime.utcnow() - timedelta(hours=hours)
        result = await self.session.execute(
            select(Job)
            .where(
                and_(
                    Job.completed_at < cutoff_date,
                    Job.status.in_(["completed", "failed", "cancelled"]),
                    Job.is_deleted == False
                )
            )
        )
        return result.scalars().all()
    
    async def update_completed_at(self, job_id: str, completed_at: datetime):
        """Update completed_at timestamp (for testing)."""
        await self.session.execute(
            update(Job)
            .where(Job.job_id == job_id)
            .values(completed_at=completed_at)
        )


class UsageRepository(BaseRepository):
    """Repository for usage logging and rate limiting."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, UsageLog)
    
    async def record(self, data: Dict[str, Any]) -> UsageLog:
        """Record API usage."""
        return await self.create(data)
    
    async def get_count_last_hour(self, institution: str) -> int:
        """Get usage count for institution in last hour."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        
        result = await self.session.execute(
            select(func.count(UsageLog.id))
            .where(
                and_(
                    UsageLog.institution == institution,
                    UsageLog.timestamp >= one_hour_ago,
                    UsageLog.is_deleted == False
                )
            )
        )
        
        return result.scalar() or 0
    
    async def get_count_last_day(self, institution: str) -> int:
        """Get usage count for institution in last 24 hours."""
        one_day_ago = datetime.utcnow() - timedelta(days=1)
        
        result = await self.session.execute(
            select(func.count(UsageLog.id))
            .where(
                and_(
                    UsageLog.institution == institution,
                    UsageLog.timestamp >= one_day_ago,
                    UsageLog.is_deleted == False
                )
            )
        )
        
        return result.scalar() or 0
    
    async def find_expired(self, days: int = 90) -> List[UsageLog]:
        """Find usage logs older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(UsageLog)
            .where(
                and_(
                    UsageLog.timestamp < cutoff_date,
                    UsageLog.is_deleted == False
                )
            )
        )
        return result.scalars().all()
    
    async def update_timestamp(self, id: int, timestamp: datetime):
        """Update timestamp (for testing)."""
        await self.session.execute(
            update(UsageLog)
            .where(UsageLog.id == id)
            .values(timestamp=timestamp)
        )


class AuditRepository(BaseRepository):
    """Repository for audit logging."""
    
    def __init__(self, session: AsyncSession):
        super().__init__(session, AuditLog)
    
    async def get_by_institution(
        self,
        institution: str,
        limit: int = 100
    ) -> List[AuditLog]:
        """Get audit logs by institution."""
        result = await self.session.execute(
            select(AuditLog)
            .where(
                and_(
                    AuditLog.institution == institution,
                    AuditLog.is_deleted == False
                )
            )
            .order_by(AuditLog.timestamp.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def get_by_trace_id(self, trace_id: str) -> Optional[AuditLog]:
        """Get audit log by trace_id."""
        result = await self.session.execute(
            select(AuditLog)
            .where(
                and_(
                    AuditLog.trace_id == trace_id,
                    AuditLog.is_deleted == False
                )
            )
        )
        return result.scalar_one_or_none()
    
    async def find_expired(self, days: int = 90) -> List[AuditLog]:
        """Find audit logs older than specified days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        result = await self.session.execute(
            select(AuditLog)
            .where(
                and_(
                    AuditLog.timestamp < cutoff_date,
                    AuditLog.is_deleted == False
                )
            )
        )
        return result.scalars().all()
    
    async def update_timestamp(self, id: int, timestamp: datetime):
        """Update timestamp (for testing)."""
        await self.session.execute(
            update(AuditLog)
            .where(AuditLog.id == id)
            .values(timestamp=timestamp)
        )
