from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import JSON, DateTime, Enum, Float, ForeignKey, Integer, String, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from hexrl_platform.task_pipeline.schemas import TaskStatus


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class InferenceTask(Base):
    """Жизненный цикл задачи: её создаёт API или, если сообщение обогнало API, воркер."""

    __tablename__ = "inference_tasks"

    task_id: Mapped[UUID] = mapped_column(Uuid, primary_key=True)
    batch_id: Mapped[str | None] = mapped_column(String(128), index=True)
    status: Mapped[TaskStatus] = mapped_column(
        # native_enum=False: VARCHAR + CHECK одинаково работает в postgres и sqlite,
        # и новый статус не требует ALTER TYPE
        Enum(TaskStatus, native_enum=False, length=16),
        index=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    last_error: Mapped[str | None] = mapped_column(Text)
    submitted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class InferenceResult(Base):
    """Ровно одна строка на задачу: первичный ключ по task_id и есть ключ идемпотентности."""

    __tablename__ = "inference_results"

    task_id: Mapped[UUID] = mapped_column(
        Uuid, ForeignKey("inference_tasks.task_id", ondelete="CASCADE"), primary_key=True
    )
    action: Mapped[int] = mapped_column(Integer)
    # champion мог смениться между перезапусками: результат должен знать, кем он посчитан
    model_version: Mapped[str] = mapped_column(String(64))
    inference_ms: Mapped[float] = mapped_column(Float)
    delay_ms: Mapped[float] = mapped_column(Float)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
