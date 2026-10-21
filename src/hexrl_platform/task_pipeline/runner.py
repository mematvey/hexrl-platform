import asyncio
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from hexrl_platform.task_pipeline.schemas import HexState, Prediction


class InferenceRunner:
    def __init__(
        self, predict: Callable[[HexState], int], concurrency: int, delay_s: float
    ) -> None:
        self._predict = predict
        self._delay_s = delay_s
        self._pool = ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="inference")

    async def run(self, state: HexState) -> Prediction:
        return await asyncio.get_running_loop().run_in_executor(self._pool, self._work, state)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True)

    def _work(self, state: HexState) -> Prediction:
        # sleep в потоке занимает слот, но не блокирует event loop
        started = time.perf_counter()
        time.sleep(self._delay_s)
        delayed = time.perf_counter()
        action = self._predict(state)
        finished = time.perf_counter()
        return Prediction(
            action=action,
            inference_ms=(finished - delayed) * 1000,
            delay_ms=(delayed - started) * 1000,
        )


def limit_torch_threads(count: int = 1) -> None:
    try:
        import torch
    except ImportError:
        return
    # иначе каждый параллельный predict займёт все ядра и они начнут мешать друг другу
    torch.set_num_threads(count)
