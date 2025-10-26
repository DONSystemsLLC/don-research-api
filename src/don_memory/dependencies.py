from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from .trace_storage import DEFAULT_DB_URL, TraceStorage


@lru_cache(maxsize=1)
def _storage() -> TraceStorage:
    db_url = os.getenv("DON_MEMORY_DB_URL")
    if not db_url:
        path_override = os.getenv("DON_MEMORY_DB_PATH")
        if path_override:
            db_url = f"sqlite:///{Path(path_override).resolve()}"
        else:
            db_url = DEFAULT_DB_URL
    return TraceStorage(db_url=db_url)


def get_trace_storage() -> TraceStorage:
    """FastAPI dependency that yields the singleton TraceStorage."""
    return _storage()
