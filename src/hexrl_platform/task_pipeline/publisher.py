import logging
from typing import Any

from aiormq.abc import DeliveredMessage
from aiormq.exceptions import AMQPError, DeliveryError, PublishError
from faststream.rabbit import RabbitBroker, RabbitExchange
from pydantic import BaseModel

from hexrl_platform.task_pipeline.errors import QueueOverloadedError, TopologyError
from hexrl_platform.task_pipeline.schemas import TaskMessage
from hexrl_platform.task_pipeline.topology import Topology

logger = logging.getLogger(__name__)


async def publish_confirmed(
    broker: RabbitBroker,
    body: bytes | BaseModel,
    *,
    exchange: RabbitExchange,
    routing_key: str,
    timeout_s: float,
    headers: dict[str, Any] | None = None,
    message_id: str | None = None,
) -> None:
    try:
        result = await broker.publish(
            body,
            exchange=exchange,
            routing_key=routing_key,
            persist=True,
            # без mandatory брокер молча выбросит сообщение, если некуда маршрутизировать
            mandatory=True,
            timeout=timeout_s,
            headers=headers,
            message_id=message_id,
        )
    except PublishError as exc:
        raise TopologyError(f"no route for key {routing_key}") from exc
    except DeliveryError as exc:
        raise QueueOverloadedError(f"broker rejected publish to {routing_key}") from exc
    except TimeoutError as exc:
        logger.warning("publish_confirm_timeout", extra={"reason": routing_key})
        raise QueueOverloadedError(f"no confirm for {routing_key}") from exc
    except (AMQPError, ConnectionError, OSError) as exc:
        logger.warning("publish_broker_unavailable", extra={"reason": routing_key})
        raise QueueOverloadedError(f"broker unavailable for {routing_key}") from exc

    # aio-pika не бросает исключение на возврат mandatory-сообщения, а отдаёт его как результат
    if isinstance(result, DeliveredMessage):
        raise TopologyError(f"no route for key {routing_key}")


class TaskPublisher:
    def __init__(self, broker: RabbitBroker, topology: Topology, confirm_timeout_s: float) -> None:
        self._broker = broker
        self._topology = topology
        self._confirm_timeout_s = confirm_timeout_s

    async def publish(self, message: TaskMessage) -> None:
        await publish_confirmed(
            self._broker,
            message,
            exchange=self._topology.tasks_exchange,
            routing_key=self._topology.tasks_routing_key,
            timeout_s=self._confirm_timeout_s,
            message_id=str(message.task_id),
        )
