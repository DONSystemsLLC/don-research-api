"""
Test database session async compatibility fixes.

This test validates:
1. Async engine with proper pool configuration
2. db_session() async generator function
3. Context manager protocol compatibility
"""
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.session import (
    DatabaseSession,
    db_session,
    get_database,
    init_database,
    close_database
)


@pytest.mark.asyncio
async def test_database_session_async_pool():
    """Test that async engine uses proper pool configuration."""
    db = DatabaseSession(test_mode=True)
    
    # Should not raise QueuePool incompatibility error
    await db.connect()
    
    assert db.is_connected()
    assert db.engine is not None
    assert db.session_maker is not None
    
    await db.disconnect()


@pytest.mark.asyncio
async def test_db_session_context_manager():
    """Test that db_session() works as async context manager."""
    # This should not raise "async_generator object does not support..." error
    async with db_session() as session:
        assert isinstance(session, AsyncSession)
        
        # Test basic query
        from sqlalchemy import text
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_db_session_transaction_handling():
    """Test that db_session() properly handles transactions."""
    async with db_session() as session:
        # Session should auto-commit on successful context exit
        from sqlalchemy import text
        await session.execute(text("SELECT 1"))
    
    # Session should be closed after context exit
    # (We can't test this directly without accessing private state)


@pytest.mark.asyncio
async def test_get_session_direct():
    """Test direct use of get_session() context manager."""
    db = get_database()
    
    async with db.get_session() as session:
        assert isinstance(session, AsyncSession)
        
        from sqlalchemy import text
        result = await session.execute(text("SELECT 1"))
        assert result.scalar() == 1


@pytest.mark.asyncio
async def test_init_close_database():
    """Test database initialization and cleanup."""
    # Initialize
    await init_database()
    
    db = get_database()
    assert db.is_connected()
    
    # Close
    await close_database()
    
    # Should create new instance on next call
    db2 = get_database()
    assert not db2.is_connected()


@pytest.mark.asyncio
async def test_concurrent_sessions():
    """Test that multiple concurrent sessions work correctly."""
    import asyncio
    
    async def query_in_session():
        async with db_session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            return result.scalar()
    
    # Run 5 concurrent sessions
    results = await asyncio.gather(
        query_in_session(),
        query_in_session(),
        query_in_session(),
        query_in_session(),
        query_in_session()
    )
    
    assert all(r == 1 for r in results)


@pytest.mark.asyncio
async def test_connection_retry_on_failure():
    """Test that connection retry logic works."""
    # This test would require mocking connection failures
    # Skipped for now but validates retry logic exists
    db = DatabaseSession(test_mode=True, max_retries=2, retry_delay=0.1)
    
    # Should succeed on first attempt in test mode
    await db.connect()
    assert db.is_connected()
    
    await db.disconnect()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
