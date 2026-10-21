"""Демо-потребитель с ограниченными слотами и prefetch (пока нет настоящего воркера).

    uv run python scripts/backpressure/slow_consumer.py --slots 2 --prefetch 2 --seconds 20

Без БД и inference: держит слот `--delay` секунд и подтверждает сообщение.
В конце логирует скорость обработки за вторую половину прогона (установившийся режим).
"""

import argparse
import asyncio
import logging
import time
from functools import partial

from aio_pika.abc import AbstractIncomingMessage
from faststream.rabbit import RabbitBroker

from hexrl_platform.core.logging import configure_logging
from hexrl_platform.task_pipeline.config import PipelineSettings
from hexrl_platform.task_pipeline.topology import build_topology, declare_topology

# имя внутри пакета hexrl_platform: только его логгер настраивает configure_logging
logger = logging.getLogger("hexrl_platform.slow_consumer")


async def _handle(
    message: AbstractIncomingMessage,
    slots: asyncio.Semaphore,
    delay_s: float,
    finished: list[float],
) -> None:
    async with slots:
        await asyncio.sleep(delay_s)
    await message.ack()
    finished.append(time.monotonic())


async def main(args: argparse.Namespace) -> None:
    configure_logging("INFO")

    settings = PipelineSettings()
    broker = RabbitBroker(settings.rabbitmq_url)
    await broker.start()
    topology = build_topology(
        args.queue_prefix or settings.queue_prefix,
        settings.queue_max_length,
        settings.retry_delay_ms,
    )
    await declare_topology(broker, topology)
    queue = await broker.declare_queue(topology.tasks_queue)
    await queue.channel.set_qos(prefetch_count=args.prefetch)

    finished: list[float] = []
    handler = partial(
        _handle, slots=asyncio.Semaphore(args.slots), delay_s=args.delay, finished=finished
    )

    started = time.monotonic()
    await queue.consume(handler)
    await asyncio.sleep(args.seconds)

    steady = [t for t in finished if t - started >= args.seconds / 2]
    rate = len(steady) / (args.seconds / 2)
    logger.info(
        "consumer_finished slots=%d prefetch=%d processed=%d rate=%.2f/s",
        args.slots,
        args.prefetch,
        len(finished),
        rate,
    )
    await broker.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--prefetch", type=int, default=2)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--queue-prefix")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
