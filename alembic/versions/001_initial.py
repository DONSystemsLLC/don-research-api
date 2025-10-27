"""Initial database schema with pgvector support

Revision ID: 001_initial
Revises: 
Create Date: 2025-10-26

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql
from pgvector.sqlalchemy import Vector

# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Enable pgvector extension
    op.execute('CREATE EXTENSION IF NOT EXISTS vector;')
    
    # Create qac_models table
    op.create_table(
        'qac_models',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('model_id', sa.String(length=255), nullable=False),
        sa.Column('institution', sa.String(length=255), nullable=False),
        sa.Column('adjacency_matrix', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('qac_corrected', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('model_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('model_id')
    )
    op.create_index('idx_qac_created_at', 'qac_models', ['created_at'])
    op.create_index('idx_qac_deleted', 'qac_models', ['is_deleted', 'deleted_at'])
    op.create_index('idx_qac_institution', 'qac_models', ['institution'])
    op.create_index('idx_qac_model_id', 'qac_models', ['model_id'])
    
    # Create vector_stores table
    op.create_table(
        'vector_stores',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('project_id', sa.String(length=255), nullable=False),
        sa.Column('institution', sa.String(length=255), nullable=False),
        sa.Column('vector', Vector(256), nullable=False),
        sa.Column('vector_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('original_dims', sa.Integer(), nullable=True),
        sa.Column('compression_ratio', sa.Float(), nullable=True),
        sa.Column('gene_names', postgresql.ARRAY(sa.String()), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_vector_created_at', 'vector_stores', ['created_at'])
    op.create_index('idx_vector_deleted', 'vector_stores', ['is_deleted', 'deleted_at'])
    op.create_index('idx_vector_institution', 'vector_stores', ['institution'])
    op.create_index('idx_vector_project_id', 'vector_stores', ['project_id'])
    # Create pgvector index for similarity search (using ivfflat)
    op.execute('CREATE INDEX idx_vector_embedding ON vector_stores USING ivfflat (vector);')
    
    # Create jobs table
    op.create_table(
        'jobs',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('job_id', sa.String(length=255), nullable=False),
        sa.Column('job_type', sa.String(length=100), nullable=False),
        sa.Column('institution', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='pending'),
        sa.Column('input_data', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('result', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('job_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('progress', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('job_id')
    )
    op.create_index('idx_job_completed_at', 'jobs', ['completed_at'])
    op.create_index('idx_job_deleted', 'jobs', ['is_deleted', 'deleted_at'])
    op.create_index('idx_job_institution', 'jobs', ['institution'])
    op.create_index('idx_job_job_id', 'jobs', ['job_id'])
    op.create_index('idx_job_status', 'jobs', ['status'])
    op.create_index('idx_job_type', 'jobs', ['job_type'])
    
    # Create usage_logs table
    op.create_table(
        'usage_logs',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('institution', sa.String(length=255), nullable=False),
        sa.Column('endpoint', sa.String(length=500), nullable=False),
        sa.Column('trace_id', sa.String(length=255), nullable=False),
        sa.Column('method', sa.String(length=10), nullable=True),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('response_time_ms', sa.Float(), nullable=True),
        sa.Column('usage_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_usage_deleted', 'usage_logs', ['is_deleted', 'deleted_at'])
    op.create_index('idx_usage_endpoint', 'usage_logs', ['endpoint'])
    op.create_index('idx_usage_institution', 'usage_logs', ['institution'])
    op.create_index('idx_usage_institution_timestamp', 'usage_logs', ['institution', 'timestamp'])
    op.create_index('idx_usage_timestamp', 'usage_logs', ['timestamp'])
    op.create_index('idx_usage_trace_id', 'usage_logs', ['trace_id'])
    
    # Create audit_logs table
    op.create_table(
        'audit_logs',
        sa.Column('id', sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column('institution', sa.String(length=255), nullable=False),
        sa.Column('action', sa.String(length=255), nullable=False),
        sa.Column('trace_id', sa.String(length=255), nullable=False),
        sa.Column('resource_type', sa.String(length=100), nullable=True),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('user_agent', sa.Text(), nullable=True),
        sa.Column('audit_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False),
        sa.Column('deleted_at', sa.DateTime(), nullable=True),
        sa.Column('is_deleted', sa.Boolean(), nullable=False, server_default='false'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_audit_action', 'audit_logs', ['action'])
    op.create_index('idx_audit_deleted', 'audit_logs', ['is_deleted', 'deleted_at'])
    op.create_index('idx_audit_institution', 'audit_logs', ['institution'])
    op.create_index('idx_audit_institution_timestamp', 'audit_logs', ['institution', 'timestamp'])
    op.create_index('idx_audit_timestamp', 'audit_logs', ['timestamp'])
    op.create_index('idx_audit_trace_id', 'audit_logs', ['trace_id'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('audit_logs')
    op.drop_table('usage_logs')
    op.drop_table('jobs')
    op.drop_table('vector_stores')
    op.drop_table('qac_models')
    op.execute('DROP EXTENSION IF EXISTS vector;')
