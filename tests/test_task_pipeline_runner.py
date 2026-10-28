import asyncio
import threading
import time

import pytest

from hexrl_platform.task_pipeline.runner import InferenceRunner
from hexrl_platform.task_pipeline.schemas import HexState

STATE = HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)


class ConcurrencyProbe:
    def __init__(self, action: int = 3, work_s: float = 0.0) -> None:
        self.action = action
        self.work_s = work_s
        self.current = 0
        self.peak = 0
        self._lock = threading.Lock()

    def __call__(self, state: HexState) -> int:
        with self._lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        time.sleep(self.work_s)
        with self._lock:
            self.current -= 1
        return self.action


async def test_run_returns_action_and_splits_delay_from_inference():
    runner = InferenceRunner(ConcurrencyProbe(action=4, work_s=0.05), concurrency=1, delay_s=0.1)
    try:
        prediction = await runner.run(STATE)
    finally:
        runner.shutdown()
    assert prediction.action == 4
    assert prediction.delay_ms >= 100
    assert 50 <= prediction.inference_ms < prediction.delay_ms + 200


async def test_event_loop_stays_responsive_while_inference_runs():
    runner = InferenceRunner(ConcurrencyProbe(), concurrency=2, delay_s=0.3)
    gaps: list[float] = []
    stop = asyncio.Event()

    async def ticker() -> None:
        last = time.perf_counter()
        while not stop.is_set():
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    task = asyncio.create_task(ticker())
    try:
        await asyncio.gather(*(runner.run(STATE) for _ in range(4)))
    finally:
        stop.set()
        await task
        runner.shutdown()
    assert len(gaps) > 20
    assert max(gaps) < 0.1


async def test_never_more_than_concurrency_inferences_at_once():
    probe = ConcurrencyProbe(work_s=0.05)
    runner = InferenceRunner(probe, concurrency=2, delay_s=0.0)
    try:
        await asyncio.gather(*(runner.run(STATE) for _ in range(8)))
    finally:
        runner.shutdown()
    assert probe.peak == 2


async def test_throughput_scales_with_concurrency():
    runner = InferenceRunner(ConcurrencyProbe(), concurrency=2, delay_s=0.2)
    started = time.perf_counter()
    try:
        await asyncio.gather(*(runner.run(STATE) for _ in range(4)))
    finally:
        runner.shutdown()
    elapsed = time.perf_counter() - started
    assert 0.38 <= elapsed < 0.7


async def test_predict_error_propagates_and_runner_keeps_working():
    calls = 0

    def flaky(state: HexState) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ValueError("wall")
        return 1

    runner = InferenceRunner(flaky, concurrency=1, delay_s=0.0)
    try:
        with pytest.raises(ValueError, match="wall"):
            await runner.run(STATE)
        assert (await runner.run(STATE)).action == 1
    finally:
        runner.shutdown()


async def test_run_after_shutdown_fails():
    runner = InferenceRunner(ConcurrencyProbe(), concurrency=1, delay_s=0.0)
    runner.shutdown()
    with pytest.raises(RuntimeError):
        await runner.run(STATE)
