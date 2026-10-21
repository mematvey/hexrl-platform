import argparse
import asyncio
import csv
import logging
import random
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

import psutil
from faststream.rabbit import RabbitBroker
from pydantic import ValidationError

from hexrl_platform.core.logging import configure_logging
from hexrl_platform.task_pipeline.config import PipelineSettings
from hexrl_platform.task_pipeline.errors import QueueOverloadedError
from hexrl_platform.task_pipeline.publisher import TaskPublisher
from hexrl_platform.task_pipeline.schemas import HexState, TaskMessage
from hexrl_platform.task_pipeline.topology import build_topology, declare_topology

logger = logging.getLogger(__name__)

METRICS_COLUMNS = (
    "ts",
    "t_s",
    "lines_read",
    "attempts",
    "published_ok",
    "rejected",
    "invalid_lines",
    "wait_s_total",
    "rss_mb",
)


class Publisher(Protocol):
    async def publish(self, message: TaskMessage) -> None: ...


@dataclass(slots=True)
class SourceStats:
    lines_read: int = 0
    published_ok: int = 0
    rejected: int = 0
    invalid_lines: int = 0
    wait_s_total: float = 0.0


@dataclass(frozen=True, slots=True)
class Backoff:
    base_s: float
    cap_s: float
    rng: random.Random

    def delay(self, attempt: int) -> float:
        # full jitter: равномерно от 0 до растущей границы; min(attempt, 30) ограничивает степень
        return self.rng.uniform(0, min(self.cap_s, self.base_s * 2 ** min(attempt, 30)))


def task_id_for(batch_id: str, line_no: int) -> UUID:
    return uuid5(NAMESPACE_URL, f"hexrl:{batch_id}:{line_no}")


def next_line(numbered: Iterator[tuple[int, str]]) -> tuple[int, str] | None:
    return next(numbered, None)


async def read_lines(path: Path) -> AsyncIterator[tuple[int, str]]:
    with path.open(encoding="utf-8") as file:
        numbered = enumerate(file, start=1)
        # чтение с диска уводим в поток: event loop остаётся свободным для confirm'ов и метрик
        while True:
            item = await asyncio.to_thread(next_line, numbered)
            if item is None:
                return
            yield item


async def run_batch(
    publisher: Publisher,
    lines: AsyncIterator[tuple[int, str]],
    batch_id: str,
    stats: SourceStats,
    backoff: Backoff,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> None:
    async for line_no, raw in lines:
        if not raw.strip():
            continue
        stats.lines_read += 1
        try:
            state = HexState.model_validate_json(raw)
        except ValidationError:
            stats.invalid_lines += 1
            logger.warning("invalid_line line=%d", line_no)
            continue

        message = TaskMessage(
            task_id=task_id_for(batch_id, line_no), batch_id=batch_id, payload=state
        )
        attempt = 0
        # следующую строку не читаем, пока эта не принята: в памяти всегда одна запись
        while True:
            try:
                await publisher.publish(message)
            except QueueOverloadedError:
                stats.rejected += 1
                delay = backoff.delay(attempt)
                attempt += 1
                stats.wait_s_total += delay
                await sleep(delay)
            else:
                stats.published_ok += 1
                break


class MetricsWriter:
    def __init__(self, path: Path, stats: SourceStats, interval_s: float) -> None:
        self._path = path
        self._stats = stats
        self._interval_s = interval_s
        self._process = psutil.Process()
        self._started = time.monotonic()

    async def run(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(self._write_header)
        try:
            while True:
                await asyncio.sleep(self._interval_s)
                await asyncio.to_thread(self._write_row)
        finally:
            await asyncio.to_thread(self._write_row)

    def _write_header(self) -> None:
        if not self._path.exists() or self._path.stat().st_size == 0:
            with self._path.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(METRICS_COLUMNS)

    def _write_row(self) -> None:
        stats = self._stats
        row = (
            datetime.now(UTC).isoformat(timespec="milliseconds"),
            f"{time.monotonic() - self._started:.3f}",
            stats.lines_read,
            stats.published_ok + stats.rejected,
            stats.published_ok,
            stats.rejected,
            stats.invalid_lines,
            f"{stats.wait_s_total:.3f}",
            f"{self._process.memory_info().rss / 1024 / 1024:.1f}",
        )
        with self._path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a JSONL file of hex states as tasks")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--metrics-csv", type=Path)
    parser.add_argument("--metrics-interval", type=float, default=0.5)
    parser.add_argument("--queue-prefix")
    parser.add_argument(
        "--unbounded",
        action="store_true",
        help="queue without length limit (baseline experiment only; use a separate --queue-prefix)",
    )
    return parser.parse_args()


async def amain(args: argparse.Namespace) -> SourceStats:
    settings = PipelineSettings()
    prefix = args.queue_prefix or settings.queue_prefix
    max_length = None if args.unbounded else settings.queue_max_length
    topology = build_topology(prefix, max_length, settings.retry_delay_ms)

    stats = SourceStats()
    backoff = Backoff(settings.backoff_base_s, settings.backoff_cap_s, random.Random())
    broker = RabbitBroker(settings.rabbitmq_url)
    await broker.start()
    metrics_task = None
    try:
        await declare_topology(broker, topology)
        publisher = TaskPublisher(broker, topology, settings.publish_confirm_timeout_s)
        if args.metrics_csv:
            writer = MetricsWriter(args.metrics_csv, stats, args.metrics_interval)
            metrics_task = asyncio.create_task(writer.run())
        logger.info(
            "batch_source_started batch=%s queue=%s", args.batch_id, topology.tasks_queue.name
        )
        await run_batch(publisher, read_lines(args.file), args.batch_id, stats, backoff)
    finally:
        if metrics_task:
            metrics_task.cancel()
            await asyncio.gather(metrics_task, return_exceptions=True)
        await broker.stop()
    return stats


def main() -> None:
    configure_logging("INFO")
    stats = asyncio.run(amain(parse_args()))
    logger.info(
        "batch_source_finished published=%d rejected=%d invalid=%d",
        stats.published_ok,
        stats.rejected,
        stats.invalid_lines,
    )
