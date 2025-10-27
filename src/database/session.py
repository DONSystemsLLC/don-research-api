"""
Database session management with connection pooling and async support.
"""
import os
import asyncio
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager
import logging

from sqlalchemy.ext.asyncio import (
    create_async_engine, 
    AsyncSession,
    AsyncEngine,
    async_sessionmaker
)
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import text

from .models import Base

logger = logging.getLogger(__name__)


class DatabaseSession:
    """
    Async database session manager with connection pooling and retry logic.
    """
    
    def __init__(
        self,
        database_url: Optional[str] = None,
        test_mode: bool = False,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        pool_size: int = 10,
        max_overflow: int = 20
    ):
        """
        Initialize database session.
        
        Args:
            database_url: PostgreSQL connection string (defaults to env var DATABASE_URL)
            test_mode: If True, use in-memory SQLite for testing
            max_retries: Maximum connection retry attempts
            retry_delay: Delay between retry attempts (seconds)
            pool_size: Number of connections to keep in pool
            max_overflow: Maximum additional connections beyond pool_size
        """
        self.test_mode = test_mode
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Get database URL
        if database_url:
            self.database_url = database_url
        elif test_mode:
            self.database_url = "sqlite+aiosqlite:///:memory:"
        else:
            self.database_url = os.getenv(
                "DATABASE_URL",
                "postgresql+asyncpg://localhost/don_research"
            )
            
            # Convert postgres:// to postgresql:// if needed (Render compatibility)
            if self.database_url.startswith("postgres://"):
                self.database_url = self.database_url.replace("postgres://", "postgresql+asyncpg://", 1)
            elif self.database_url.startswith("postgresql://"):
                self.database_url = self.database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        
        # Engine configuration
        self.engine: Optional[AsyncEngine] = None
        self.session_maker: Optional[async_sessionmaker] = None
        
        # Connection pool settings
        if test_mode:
            self.pool_class = NullPool
            self.pool_size = 0
            self.max_overflow = 0
        else:
            self.pool_class = QueuePool
            self.pool_size = pool_size
            self.max_overflow = max_overflow
    
    async def connect(self) -> None:
        """
        Establish database connection with retry logic.
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Connecting to database (attempt {attempt + 1}/{self.max_retries})...")
                
                # Create engine
                self.engine = create_async_engine(
                    self.database_url,
                    poolclass=self.pool_class,
                    pool_size=self.pool_size,
                    max_overflow=self.max_overflow,
                    pool_pre_ping=True,  # Verify connections before using
                    echo=self.test_mode,  # Log SQL in test mode
                )
                
                # Create session maker
                self.session_maker = async_sessionmaker(
                    self.engine,
                    class_=AsyncSession,
                    expire_on_commit=False
                )
                
                # Test connection
                async with self.engine.begin() as conn:
                    await conn.execute(text("SELECT 1"))
                
                logger.info("Database connection established successfully")
                return
                
            except Exception as e:
                logger.error(f"Database connection failed: {e}")
                
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    raise ConnectionError(f"Failed to connect to database after {self.max_retries} attempts: {e}")
    
    async def disconnect(self) -> None:
        """Close database connection and cleanup resources."""
        if self.engine:
            logger.info("Closing database connections...")
            await self.engine.dispose()
            self.engine = None
            self.session_maker = None
            logger.info("Database connections closed")
    
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.engine is not None and self.session_maker is not None
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            async with db.get_session() as session:
                result = await session.execute(query)
        """
        if not self.is_connected():
            await self.connect()
        
        async with self.session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a raw database connection from the pool."""
        if not self.is_connected():
            await self.connect()
        
        async with self.engine.begin() as conn:
            yield conn
    
    @asynccontextmanager
    async def transaction(self):
        """
        Create a transaction context.
        
        Usage:
            async with db.transaction():
                # operations here
                # will be rolled back on exception
        """
        async with self.get_session() as session:
            async with session.begin():
                yield session
    
    async def execute(self, query: str, params: Optional[dict] = None):
        """
        Execute a raw SQL query.
        
        Args:
            query: SQL query string
            params: Query parameters
        
        Returns:
            Query result
        """
        async with self.get_connection() as conn:
            result = await conn.execute(text(query), params or {})
            return result.fetchall()
    
    async def create_tables(self) -> None:
        """Create all database tables."""
        logger.info("Creating database tables...")
        
        if not self.is_connected():
            await self.connect()
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables created successfully")
    
    async def drop_tables(self) -> None:
        """Drop all database tables (use with caution!)."""
        logger.warning("Dropping all database tables...")
        
        if not self.is_connected():
            await self.connect()
        
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("Database tables dropped")
    
    async def enable_extensions(self) -> None:
        """Enable required PostgreSQL extensions."""
        if self.test_mode:
            logger.info("Skipping extension setup in test mode")
            return
        
        logger.info("Enabling PostgreSQL extensions...")
        
        try:
            # Enable pgvector extension
            await self.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            logger.info("pgvector extension enabled")
        except Exception as e:
            logger.error(f"Failed to enable extensions: {e}")
            raise


# Global database instance
_db_instance: Optional[DatabaseSession] = None


def get_database() -> DatabaseSession:
    """Get or create global database instance."""
    global _db_instance
    
    if _db_instance is None:
        _db_instance = DatabaseSession()
    
    return _db_instance


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency for database sessions.
    
    Usage in FastAPI:
        @app.get("/endpoint")
        async def endpoint(db: AsyncSession = Depends(get_db_session)):
            result = await db.execute(query)
    """
    db = get_database()
    
    if not db.is_connected():
        await db.connect()
    
    async with db.get_session() as session:
        yield session


@asynccontextmanager
async def db_session():
    """
    Direct async context manager for database sessions.
    
    Usage:
        async with db_session() as session:
            result = await session.execute(query)
    """
    db = get_database()
    
    if not db.is_connected():
        await db.connect()
    
    async with db.get_session() as session:
        yield session


async def init_database() -> None:
    """
    Initialize database on application startup.
    Creates tables and enables extensions.
    """
    logger.info("Initializing database...")
    
    db = get_database()
    await db.connect()
    await db.enable_extensions()
    await db.create_tables()
    
    logger.info("Database initialization complete")


async def close_database() -> None:
    """Close database connection on application shutdown."""
    global _db_instance
    
    if _db_instance:
        await _db_instance.disconnect()
        _db_instance = None
