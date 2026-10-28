import asyncio
import csv
import random
from pathlib import Path

import pytest

from hexrl_platform.task_pipeline.batch_source import (
    METRICS_COLUMNS,
    Backoff,
    MetricsWriter,
    SourceStats,
    read_lines,
    run_batch,
    task_id_for,
)
from hexrl_platform.task_pipeline.errors import QueueOverloadedError, TopologyError
from hexrl_platform.task_pipeline.schemas import TaskMessage

LINE = '{"q": 0, "r": 0, "goal_q": 2, "goal_r": -1, "steps_remaining": 20}'


class ScriptedPublisher:
    """Отказывает заданное число раз подряд перед каждым принятием; пишет все попытки."""

    def __init__(self, rejections_before_accept: dict[int, int] | None = None) -> None:
        self.rejections = dict(rejections_before_accept or {})
        self.attempts: list[TaskMessage] = []
        self.accepted: list[TaskMessage] = []
        self.error: Exception | None = None

    async def publish(self, message: TaskMessage) -> None:
        self.attempts.append(message)
        if self.error:
            raise self.error
        left = self.rejections.get(len(self.accepted), 0)
        if left > 0:
            self.rejections[len(self.accepted)] = left - 1
            raise QueueOverloadedError
        self.accepted.append(message)


async def numbered(*lines: str):
    for index, line in enumerate(lines, start=1):
        yield index, line


def make_backoff(base_s: float = 0.05, cap_s: float = 1.0) -> Backoff:
    return Backoff(base_s, cap_s, random.Random(7))


async def no_sleep(delay: float) -> None:
    return None


def test_backoff_delay_stays_within_growing_bound_and_cap():
    backoff = make_backoff(base_s=0.05, cap_s=1.0)
    for attempt in range(40):
        bound = min(1.0, 0.05 * 2 ** min(attempt, 30))
        for _ in range(50):
            assert 0 <= backoff.delay(attempt) <= bound


def test_backoff_is_jittered_not_constant():
    backoff = make_backoff()
    assert len({backoff.delay(5) for _ in range(20)}) > 1


def test_task_id_is_stable_per_batch_and_line():
    assert task_id_for("run-1", 3) == task_id_for("run-1", 3)
    assert task_id_for("run-1", 3) != task_id_for("run-1", 4)
    assert task_id_for("run-1", 3) != task_id_for("run-2", 3)


async def test_read_lines_numbers_from_one(tmp_path: Path):
    path = tmp_path / "data.jsonl"
    path.write_text("a\nb\nc\n", encoding="utf-8")
    assert [item async for item in read_lines(path)] == [(1, "a\n"), (2, "b\n"), (3, "c\n")]


async def test_all_records_published_in_order():
    publisher = ScriptedPublisher()
    stats = SourceStats()
    await run_batch(publisher, numbered(LINE, LINE, LINE), "b", stats, make_backoff(), no_sleep)
    assert [m.task_id for m in publisher.accepted] == [task_id_for("b", n) for n in (1, 2, 3)]
    assert (stats.lines_read, stats.published_ok, stats.rejected) == (3, 3, 0)


async def test_rejected_record_is_retried_and_next_one_waits():
    publisher = ScriptedPublisher({0: 3})
    stats = SourceStats()
    await run_batch(publisher, numbered(LINE, LINE), "b", stats, make_backoff(), no_sleep)
    first, second = task_id_for("b", 1), task_id_for("b", 2)
    assert [m.task_id for m in publisher.attempts] == [first, first, first, first, second]
    assert [m.task_id for m in publisher.accepted] == [first, second]
    assert stats.rejected == 3


async def test_backoff_grows_then_resets_after_success():
    publisher = ScriptedPublisher({0: 6, 1: 1})
    delays: list[float] = []

    async def record(delay: float) -> None:
        delays.append(delay)

    backoff = Backoff(base_s=0.05, cap_s=1.0, rng=random.Random(1))
    stats = SourceStats()
    await run_batch(publisher, numbered(LINE, LINE), "b", stats, backoff, record)
    assert len(delays) == 7
    assert all(delay <= min(1.0, 0.05 * 2**i) for i, delay in enumerate(delays[:6]))
    # после принятия первой записи счётчик попыток обнуляется: вторая начинает с base
    assert delays[6] <= 0.05
    assert stats.wait_s_total == pytest.approx(sum(delays))


async def test_invalid_lines_are_skipped_and_counted_without_stopping():
    publisher = ScriptedPublisher()
    stats = SourceStats()
    lines = numbered("not json", LINE, '{"q": 1}', "", LINE)
    await run_batch(publisher, lines, "b", stats, make_backoff(), no_sleep)
    assert [m.task_id for m in publisher.accepted] == [task_id_for("b", 2), task_id_for("b", 5)]
    assert (stats.lines_read, stats.invalid_lines) == (4, 2)


async def test_topology_error_stops_the_source():
    publisher = ScriptedPublisher()
    publisher.error = TopologyError("no route")
    stats = SourceStats()
    with pytest.raises(TopologyError):
        await run_batch(publisher, numbered(LINE, LINE), "b", stats, make_backoff(), no_sleep)
    assert len(publisher.attempts) == 1


async def test_metrics_writer_writes_header_and_final_row(tmp_path: Path):
    path = tmp_path / "out" / "source.csv"
    stats = SourceStats(lines_read=5, published_ok=3, rejected=2, wait_s_total=0.4)
    writer = MetricsWriter(path, stats, interval_s=0.02)
    task = asyncio.create_task(writer.run())
    await asyncio.sleep(0.1)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert tuple(rows[0]) == METRICS_COLUMNS
    assert len(rows) >= 2
    last = rows[-1]
    assert (last["lines_read"], last["attempts"], last["published_ok"]) == ("5", "5", "3")
    assert float(last["rss_mb"]) > 0
