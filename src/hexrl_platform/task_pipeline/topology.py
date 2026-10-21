from dataclasses import dataclass

from faststream.rabbit import ExchangeType, RabbitBroker, RabbitExchange, RabbitQueue
from faststream.rabbit.schemas.queue import ClassicQueueArgs

TASKS_EXCHANGE = "hexrl.inference"
DLX_EXCHANGE = "hexrl.inference.dlx"


@dataclass(frozen=True, slots=True)
class Topology:
    tasks_exchange: RabbitExchange
    dlx: RabbitExchange
    tasks_queue: RabbitQueue
    retry_queue: RabbitQueue
    dlq: RabbitQueue

    @property
    def tasks_routing_key(self) -> str:
        return self.tasks_queue.routing()

    @property
    def retry_routing_key(self) -> str:
        return self.retry_queue.routing()

    @property
    def dead_routing_key(self) -> str:
        return self.dlq.routing()


def build_topology(queue_prefix: str, max_length: int | None, retry_delay_ms: int) -> Topology:
    tasks_exchange = RabbitExchange(TASKS_EXCHANGE, type=ExchangeType.DIRECT, durable=True)
    dlx = RabbitExchange(DLX_EXCHANGE, type=ExchangeType.DIRECT, durable=True)

    tasks_name = f"{queue_prefix}.tasks"
    retry_name = f"{queue_prefix}.retry"
    dlq_name = f"{queue_prefix}.dlq"

    tasks_arguments: ClassicQueueArgs = {
        "x-dead-letter-exchange": DLX_EXCHANGE,
        "x-dead-letter-routing-key": retry_name,
    }
    if max_length is not None:
        tasks_arguments["x-max-length"] = max_length
        # главный механизм backpressure: при заполнении брокер отвечает nack на публикацию
        # (drop-head молча выбросил бы старые задачи - это потеря данных)
        tasks_arguments["x-overflow"] = "reject-publish"

    tasks_queue = RabbitQueue(tasks_name, durable=True, arguments=tasks_arguments)
    retry_queue = RabbitQueue(
        retry_name,
        durable=True,
        arguments={
            # очередь без consumers: сообщение лежит TTL (это и есть задержка между
            # попытками) и отправляется обратно в основную очередь через dead-letter
            "x-message-ttl": retry_delay_ms,
            "x-dead-letter-exchange": TASKS_EXCHANGE,
            "x-dead-letter-routing-key": tasks_name,
        },
    )
    dlq = RabbitQueue(dlq_name, durable=True)

    return Topology(tasks_exchange, dlx, tasks_queue, retry_queue, dlq)


async def declare_topology(broker: RabbitBroker, topology: Topology) -> None:
    """Идемпотентно объявляет exchange, очереди и привязки

    Повторное объявление с теми же аргументами ничего не меняет, с другими - падает
    с PRECONDITION_FAILED (очередь надо удалить или назвать иначе).
    """
    tasks_exchange = await broker.declare_exchange(topology.tasks_exchange)
    dlx = await broker.declare_exchange(topology.dlx)

    tasks_queue = await broker.declare_queue(topology.tasks_queue)
    await tasks_queue.bind(tasks_exchange, routing_key=topology.tasks_routing_key)

    retry_queue = await broker.declare_queue(topology.retry_queue)
    await retry_queue.bind(dlx, routing_key=topology.retry_routing_key)

    dlq = await broker.declare_queue(topology.dlq)
    await dlq.bind(dlx, routing_key=topology.dead_routing_key)
