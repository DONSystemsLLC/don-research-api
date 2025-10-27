"""
Database models for DON Research API.
Uses SQLAlchemy 2.0 async with PostgreSQL and pgvector extension.
"""
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import (
    Column, Integer, String, DateTime, Text, JSON, 
    Index, Boolean, Float, BigInteger
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import JSONB, ARRAY
from pgvector.sqlalchemy import Vector

Base = declarative_base()


class QACModel(Base):
    """
    QAC (Quantum Adjacency Code) model storage.
    Retention policy: 30 days
    """
    __tablename__ = "qac_models"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    model_id = Column(String(255), unique=True, nullable=False, index=True)
    institution = Column(String(255), nullable=False, index=True)
    
    # QAC data stored as JSONB for flexibility
    adjacency_matrix = Column(JSONB, nullable=False)
    qac_corrected = Column(JSONB, nullable=False)
    model_metadata = Column(JSONB, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete for retention policy
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    __table_args__ = (
        Index('idx_qac_model_id', 'model_id'),
        Index('idx_qac_institution', 'institution'),
        Index('idx_qac_created_at', 'created_at'),
        Index('idx_qac_deleted', 'is_deleted', 'deleted_at'),
    )
    
    def __repr__(self):
        return f"<QACModel(model_id='{self.model_id}', institution='{self.institution}')>"


class VectorStore(Base):
    """
    Vector storage for genomics data with pgvector for similarity search.
    Retention policy: 7 days
    """
    __tablename__ = "vector_stores"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    project_id = Column(String(255), nullable=False, index=True)
    institution = Column(String(255), nullable=False, index=True)
    
    # Vector data (using pgvector extension)
    # Default dimension is 256, adjust based on compression target
    vector = Column(Vector(256), nullable=False)
    
    # Metadata about the vector
    vector_metadata = Column(JSONB, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Original data reference (optional)
    original_dims = Column(Integer, nullable=True)
    compression_ratio = Column(Float, nullable=True)
    gene_names = Column(ARRAY(String), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    __table_args__ = (
        Index('idx_vector_project_id', 'project_id'),
        Index('idx_vector_institution', 'institution'),
        Index('idx_vector_created_at', 'created_at'),
        Index('idx_vector_deleted', 'is_deleted', 'deleted_at'),
        # pgvector index for similarity search (ivfflat or hnsw)
        Index('idx_vector_embedding', 'vector', postgresql_using='ivfflat'),
    )
    
    def __repr__(self):
        return f"<VectorStore(project_id='{self.project_id}', dims={self.original_dims})>"


class Job(Base):
    """
    Async job tracking for long-running operations.
    Retention policy: 48 hours after completion
    """
    __tablename__ = "jobs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    job_id = Column(String(255), unique=True, nullable=False, index=True)
    job_type = Column(String(100), nullable=False, index=True)
    institution = Column(String(255), nullable=False, index=True)
    
    # Job status: pending, running, completed, failed, cancelled
    status = Column(String(50), nullable=False, default='pending', index=True)
    
    # Job data
    input_data = Column(JSONB, nullable=True)
    result = Column(JSONB, nullable=True)
    error = Column(Text, nullable=True)
    job_metadata = Column(JSONB, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Progress tracking
    progress = Column(Float, default=0.0, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    __table_args__ = (
        Index('idx_job_job_id', 'job_id'),
        Index('idx_job_status', 'status'),
        Index('idx_job_institution', 'institution'),
        Index('idx_job_type', 'job_type'),
        Index('idx_job_completed_at', 'completed_at'),
        Index('idx_job_deleted', 'is_deleted', 'deleted_at'),
    )
    
    def __repr__(self):
        return f"<Job(job_id='{self.job_id}', type='{self.job_type}', status='{self.status}')>"


class UsageLog(Base):
    """
    API usage logging for rate limiting and analytics.
    Retention policy: 90 days
    """
    __tablename__ = "usage_logs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    institution = Column(String(255), nullable=False, index=True)
    endpoint = Column(String(500), nullable=False)
    trace_id = Column(String(255), nullable=False, index=True)
    
    # Request details
    method = Column(String(10), nullable=True)
    status_code = Column(Integer, nullable=True)
    response_time_ms = Column(Float, nullable=True)
    
    # Metadata
    usage_metadata = Column(JSONB, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    __table_args__ = (
        Index('idx_usage_institution', 'institution'),
        Index('idx_usage_timestamp', 'timestamp'),
        Index('idx_usage_trace_id', 'trace_id'),
        Index('idx_usage_endpoint', 'endpoint'),
        Index('idx_usage_institution_timestamp', 'institution', 'timestamp'),
        Index('idx_usage_deleted', 'is_deleted', 'deleted_at'),
    )
    
    def __repr__(self):
        return f"<UsageLog(institution='{self.institution}', endpoint='{self.endpoint}')>"


class AuditLog(Base):
    """
    Audit logging for compliance and security.
    Retention policy: 90 days
    """
    __tablename__ = "audit_logs"
    
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    institution = Column(String(255), nullable=False, index=True)
    action = Column(String(255), nullable=False, index=True)
    trace_id = Column(String(255), nullable=False, index=True)
    
    # Action details
    resource_type = Column(String(100), nullable=True)
    resource_id = Column(String(255), nullable=True)
    
    # User/IP information
    ip_address = Column(String(45), nullable=True)  # IPv6 max length
    user_agent = Column(Text, nullable=True)
    
    # Metadata
    audit_metadata = Column(JSONB, nullable=True)  # Renamed from 'metadata' to avoid SQLAlchemy conflict
    
    # Timestamp
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Soft delete
    deleted_at = Column(DateTime, nullable=True)
    is_deleted = Column(Boolean, default=False, index=True)
    
    __table_args__ = (
        Index('idx_audit_institution', 'institution'),
        Index('idx_audit_action', 'action'),
        Index('idx_audit_timestamp', 'timestamp'),
        Index('idx_audit_trace_id', 'trace_id'),
        Index('idx_audit_institution_timestamp', 'institution', 'timestamp'),
        Index('idx_audit_deleted', 'is_deleted', 'deleted_at'),
    )
    
    def __repr__(self):
        return f"<AuditLog(institution='{self.institution}', action='{self.action}')>"


# Migration helper functions
def get_all_tables():
    """Get list of all table names."""
    return [
        'qac_models',
        'vector_stores',
        'jobs',
        'usage_logs',
        'audit_logs'
    ]


def get_required_extensions():
    """Get list of required PostgreSQL extensions."""
    return ['vector']  # pgvector extension for similarity search
