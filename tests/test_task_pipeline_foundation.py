import logging
import uuid
from typing import Any
from unittest.mock import AsyncMock, create_autospec

import pytest
from aiormq.abc import DeliveredMessage
from aiormq.exceptions import AMQPConnectionError, DeliveryError
from pamqp.commands import Basic
from pydantic import ValidationError

from hexrl_platform.task_pipeline.config import PipelineSettings
from hexrl_platform.task_pipeline.errors import QueueOverloadedError, TopologyError
from hexrl_platform.task_pipeline.publisher import TaskPublisher
from hexrl_platform.task_pipeline.schemas import HexState, TaskMessage
from hexrl_platform.task_pipeline.topology import build_topology

STATE = HexState(q=0, r=0, goal_q=2, goal_r=-1, steps_remaining=20)


def make_settings(**overrides: Any) -> PipelineSettings:
    return PipelineSettings(_env_file=None, rabbitmq_user="u", rabbitmq_password="p", **overrides)


def make_message() -> TaskMessage:
    return TaskMessage(task_id=uuid.uuid4(), payload=STATE)


def test_rabbitmq_url_escapes_credentials():
    settings = PipelineSettings(
        _env_file=None, rabbitmq_user="us er", rabbitmq_password="p@ss/word", rabbitmq_host="mq"
    )
    assert settings.rabbitmq_url == "amqp://us%20er:p%40ss%2Fword@mq:5672/"


def test_prefetch_above_concurrency_is_logged(caplog):
    with caplog.at_level(logging.WARNING):
        make_settings(worker_concurrency=2, prefetch_count=10)
    assert "prefetch_above_concurrency" in caplog.text


def test_prefetch_equal_to_concurrency_is_quiet(caplog):
    with caplog.at_level(logging.WARNING):
        make_settings(worker_concurrency=2, prefetch_count=2)
    assert "prefetch_above_concurrency" not in caplog.text


def test_non_positive_limits_rejected():
    with pytest.raises(ValidationError):
        make_settings(queue_max_length=0)


def test_hex_state_requires_positive_steps():
    with pytest.raises(ValidationError):
        HexState(q=0, r=0, goal_q=1, goal_r=1, steps_remaining=0)


def make_topology(prefix: str = "hexrl.inference", max_length: int | None = 20):
    return build_topology(prefix, max_length, retry_delay_ms=5000)


def test_topology_bounded_queue_rejects_publish():
    topology = make_topology(max_length=20)
    args = topology.tasks_queue.arguments
    assert args["x-max-length"] == 20
    assert args["x-overflow"] == "reject-publish"
    assert args["x-dead-letter-routing-key"] == topology.retry_routing_key


def test_topology_unbounded_queue_has_no_limit():
    args = make_topology(max_length=None).tasks_queue.arguments
    assert "x-max-length" not in args
    assert "x-overflow" not in args


def test_topology_default_names_match_contract():
    topology = make_topology()
    assert topology.tasks_queue.name == "hexrl.inference.tasks"
    assert topology.retry_queue.name == "hexrl.inference.retry"
    assert topology.dlq.name == "hexrl.inference.dlq"


def test_topology_separate_experiments_do_not_share_retry_or_dlq():
    a = make_topology("exp-a")
    b = make_topology("exp-b")
    assert {a.retry_queue.name, a.dlq.name}.isdisjoint({b.retry_queue.name, b.dlq.name})
    assert a.retry_routing_key != b.retry_routing_key


def test_topology_routing_keys_come_from_queue_names():
    topology = make_topology()
    assert topology.tasks_routing_key == "hexrl.inference.tasks"
    assert topology.retry_routing_key == "hexrl.inference.retry"
    assert topology.dead_routing_key == "hexrl.inference.dlq"


def make_publisher(broker: AsyncMock) -> TaskPublisher:
    return TaskPublisher(broker, make_topology(), confirm_timeout_s=0.5)


async def test_publish_success_returns_none():
    broker = AsyncMock()
    broker.publish.return_value = Basic.Ack(delivery_tag=1)
    await make_publisher(broker).publish(make_message())
    kwargs = broker.publish.call_args.kwargs
    assert kwargs["routing_key"] == "hexrl.inference.tasks"
    assert kwargs["exchange"].name == "hexrl.inference"
    assert kwargs["mandatory"] is True
    assert kwargs["persist"] is True


async def test_publish_nack_means_overloaded():
    broker = AsyncMock()
    broker.publish.side_effect = DeliveryError(None, Basic.Nack(delivery_tag=1))
    with pytest.raises(QueueOverloadedError):
        await make_publisher(broker).publish(make_message())


async def test_publish_confirm_timeout_means_overloaded():
    broker = AsyncMock()
    broker.publish.side_effect = TimeoutError
    with pytest.raises(QueueOverloadedError):
        await make_publisher(broker).publish(make_message())


async def test_publish_connection_loss_means_overloaded():
    broker = AsyncMock()
    broker.publish.side_effect = AMQPConnectionError()
    with pytest.raises(QueueOverloadedError):
        await make_publisher(broker).publish(make_message())


async def test_publish_returned_message_is_topology_error_not_success():
    broker = AsyncMock()
    broker.publish.return_value = create_autospec(DeliveredMessage, instance=True)
    with pytest.raises(TopologyError):
        await make_publisher(broker).publish(make_message())
