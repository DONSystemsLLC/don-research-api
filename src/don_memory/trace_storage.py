from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

ARTIFACT_DIR = Path("artifacts") / "memory"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_DB_URL = f"sqlite:///{(ARTIFACT_DIR / 'traces.db').resolve()}"


class Base(DeclarativeBase):
    pass


class Trace(Base):
    __tablename__ = "memory_traces"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    project_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    user_id: Mapped[Optional[str]] = mapped_column(String, index=True)
    event_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False)
    metrics: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    artifacts: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    engine_used: Mapped[Optional[str]] = mapped_column(String)
    health: Mapped[Dict[str, Any]] = mapped_column(JSON, default=dict)
    seed: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TraceEdge(Base):
    __tablename__ = "memory_trace_edges"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_id: Mapped[str] = mapped_column(String, ForeignKey("memory_traces.id", ondelete="CASCADE"), index=True)
    target_id: Mapped[str] = mapped_column(String, ForeignKey("memory_traces.id", ondelete="CASCADE"), index=True)
    relation: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class TraceStorage:
    """Manage persistent trace storage using SQLAlchemy."""

    def __init__(self, db_url: str = DEFAULT_DB_URL) -> None:
        connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
        self.engine = create_engine(db_url, future=True, echo=False, connect_args=connect_args)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(self.engine, expire_on_commit=False)

    def store_trace(self, payload: Dict[str, Any]) -> str:
        trace_id = payload.get("id")
        if not trace_id:
            raise ValueError("trace payload requires an 'id'")

        def _resolve_time(key: str) -> Optional[datetime]:
            value = payload.get(key)
            if not value:
                return None
            if isinstance(value, datetime):
                return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            if isinstance(value, (float, int)):
                return datetime.fromtimestamp(float(value), tz=timezone.utc)
            if isinstance(value, str):
                try:
                    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    raise ValueError(f"Invalid datetime for {key}: {value}")
            raise ValueError(f"Unsupported datetime type for {key}: {type(value)!r}")

        started_at = _resolve_time("started_at")
        finished_at = _resolve_time("finished_at")

        trace = Trace(
            id=trace_id,
            project_id=payload.get("project_id"),
            user_id=payload.get("user_id"),
            event_type=payload.get("event_type", "unknown"),
            status=payload.get("status", "unknown"),
            metrics=_ensure_json(payload.get("metrics", {})),
            artifacts=_ensure_json(payload.get("artifacts", {})),
            engine_used=payload.get("engine_used"),
            health=_ensure_json(payload.get("health", {})),
            seed=payload.get("seed"),
            started_at=started_at,
            finished_at=finished_at,
        )

        with self.SessionLocal() as session:
            session.merge(trace)
            session.commit()
        return trace_id

    def link(self, source_id: str, target_id: str, relation: str) -> None:
        edge = TraceEdge(source_id=source_id, target_id=target_id, relation=relation)
        with self.SessionLocal() as session:
            session.add(edge)
            session.commit()

    def list_traces(self, project_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        stmt = (
            select(Trace)
            .where(Trace.project_id == project_id)
            .order_by(Trace.finished_at.desc().nullslast(), Trace.created_at.desc())
            .limit(limit)
        )
        with self.SessionLocal() as session:
            rows = session.scalars(stmt).all()
        return [_to_dict(row) for row in rows]

    def list_edges(self, source_id: str) -> List[Dict[str, Any]]:
        stmt = select(TraceEdge).where(TraceEdge.source_id == source_id).order_by(TraceEdge.created_at.asc())
        with self.SessionLocal() as session:
            rows = session.scalars(stmt).all()
        return [
            {
                "source_id": row.source_id,
                "target_id": row.target_id,
                "relation": row.relation,
            }
            for row in rows
        ]

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        stmt = select(Trace).where(Trace.id == trace_id)
        with self.SessionLocal() as session:
            result = session.scalar(stmt)
        return _to_dict(result) if result else None


def _ensure_json(value: Any) -> Any:
    if isinstance(value, (dict, list, str, int, float)) or value is None:
        return value
    return json.loads(json.dumps(value, default=_json_default))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _to_dict(row: Trace) -> Dict[str, Any]:
    return {
        "id": row.id,
        "project_id": row.project_id,
        "user_id": row.user_id,
        "event_type": row.event_type,
        "status": row.status,
        "metrics": row.metrics,
        "artifacts": row.artifacts,
        "engine_used": row.engine_used,
        "health": row.health,
        "seed": row.seed,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "finished_at": row.finished_at.isoformat() if row.finished_at else None,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
