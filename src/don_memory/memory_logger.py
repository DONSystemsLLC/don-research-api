"""Memory logging utilities for bio endpoints.

This module provides a simple multi-target logger that records events to
JSONL, Markdown, and SQLite stores under the artifacts directory. It is
intentionally lightweight so the API can call it without introducing heavy
runtime dependencies.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ARTIFACTS_ROOT = Path(os.getenv("ARTIFACTS_DIR", "artifacts"))
DEFAULT_LOG_DIR = ARTIFACTS_ROOT / "memory"


class MemoryLogger:
    """Persist events to JSONL, Markdown, and SQLite files."""

    def __init__(self, log_dir: Optional[Path | str] = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.jsonl_path = self.log_dir / "events.jsonl"
        self.md_path = self.log_dir / "events.md"
        self.sqlite_path = self.log_dir / "events.sqlite"

        self._lock = threading.Lock()
        self._ensure_markdown_header()
        self._ensure_sqlite()

    def log_event(
        self,
        *,
        endpoint: str,
        status: str,
        project_id: Optional[str],
        user_id: Optional[str],
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record an event across all supported backends."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "endpoint": endpoint,
            "status": status,
            "project_id": project_id,
            "user_id": user_id,
            "payload": payload or {},
        }

        with self._lock:
            self._append_jsonl(event)
            self._append_markdown(event)
            self._insert_sqlite(event)

        return event

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_markdown_header(self) -> None:
        if not self.md_path.exists() or self.md_path.stat().st_size == 0:
            header = (
                "# Bio Memory Events\n\n"
                "| Timestamp | Endpoint | Status | Project | User |\n"
                "| --- | --- | --- | --- | --- |\n"
            )
            self.md_path.write_text(header, encoding="utf-8")

    def _ensure_sqlite(self) -> None:
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bio_memory_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    status TEXT NOT NULL,
                    project_id TEXT,
                    user_id TEXT,
                    payload TEXT
                )
                """
            )
            conn.commit()

    def _append_jsonl(self, event: Dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, separators=(",", ":"), ensure_ascii=True))
            fh.write("\n")

    def _append_markdown(self, event: Dict[str, Any]) -> None:
        row = (
            f"| {event['timestamp']} | {event['endpoint']} | {event['status']} | "
            f"{event['project_id'] or '-'} | {event['user_id'] or '-'} |\n"
        )
        with self.md_path.open("a", encoding="utf-8") as fh:
            fh.write(row)

    def _insert_sqlite(self, event: Dict[str, Any]) -> None:
        payload_json = json.dumps(event["payload"], separators=(",", ":"), ensure_ascii=True)
        with sqlite3.connect(self.sqlite_path) as conn:
            conn.execute(
                """
                INSERT INTO bio_memory_events (timestamp, endpoint, status, project_id, user_id, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    event["timestamp"],
                    event["endpoint"],
                    event["status"],
                    event["project_id"],
                    event["user_id"],
                    payload_json,
                ),
            )
            conn.commit()


_global_logger: Optional[MemoryLogger] = None


def get_memory_logger() -> MemoryLogger:
    """Return a process-wide singleton logger."""
    global _global_logger
    if _global_logger is None:
        _global_logger = MemoryLogger()
    return _global_logger
