import logging

from aiormq.abc import DeliveredMessage
from aiormq.exceptions import AMQPError, DeliveryError, PublishError
from faststream.rabbit import RabbitBroker

from hexrl_platform.task_pipeline.errors import QueueOverloadedError, TopologyError
from hexrl_platform.task_pipeline.schemas import TaskMessage
from hexrl_platform.task_pipeline.topology import Topology

logger = logging.getLogger(__name__)


class TaskPublisher:
    def __init__(self, broker: RabbitBroker, topology: Topology, confirm_timeout_s: float) -> None:
        self._broker = broker
        self._topology = topology
        self._confirm_timeout_s = confirm_timeout_s

    async def publish(self, message: TaskMessage) -> None:
        try:
            result = await self._broker.publish(
                message,
                exchange=self._topology.tasks_exchange,
                routing_key=self._topology.tasks_routing_key,
                persist=True,
                # без mandatory брокер молча выбросит сообщение, если некуда маршрутизировать
                mandatory=True,
                timeout=self._confirm_timeout_s,
                message_id=str(message.task_id),
            )
        except PublishError as exc:
            raise TopologyError(f"no route for task {message.task_id}") from exc
        except DeliveryError as exc:
            raise QueueOverloadedError(f"broker rejected task {message.task_id}") from exc
        except TimeoutError as exc:
            logger.warning("publish_confirm_timeout", extra={"task_id": str(message.task_id)})
            raise QueueOverloadedError(f"no confirm for task {message.task_id}") from exc
        except (AMQPError, ConnectionError, OSError) as exc:
            logger.warning("publish_broker_unavailable", extra={"task_id": str(message.task_id)})
            raise QueueOverloadedError(f"broker unavailable for task {message.task_id}") from exc

        # aio-pika не бросает исключение на возврат mandatory-сообщения, а отдаёт его как результат
        if isinstance(result, DeliveredMessage):
            raise TopologyError(f"no route for task {message.task_id}")
