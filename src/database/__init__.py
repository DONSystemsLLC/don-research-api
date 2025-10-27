"""
Database package for DON Research API.
Provides async PostgreSQL access with SQLAlchemy and pgvector.
"""
from .models import (
    Base,
    QACModel,
    VectorStore,
    Job,
    UsageLog,
    AuditLog,
    get_all_tables,
    get_required_extensions
)

from .session import (
    DatabaseSession,
    get_database,
    get_db_session,
    db_session,
    init_database,
    close_database
)

from .repositories import (
    BaseRepository,
    QACRepository,
    VectorRepository,
    JobRepository,
    UsageRepository,
    AuditRepository
)

__all__ = [
    # Models
    "Base",
    "QACModel",
    "VectorStore",
    "Job",
    "UsageLog",
    "AuditLog",
    "get_all_tables",
    "get_required_extensions",
    
    # Session
    "DatabaseSession",
    "get_database",
    "get_db_session",
    "db_session",
    "init_database",
    "close_database",
    
    # Repositories
    "BaseRepository",
    "QACRepository",
    "VectorRepository",
    "JobRepository",
    "UsageRepository",
    "AuditRepository",
]
