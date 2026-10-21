import json
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from functools import partial
from typing import Annotated

import uvicorn
from faststream import AckPolicy, Context
from faststream.annotations import ContextRepo
from faststream.asgi import AsgiFastStream, AsgiResponse, get, make_ping_asgi
from faststream.asgi.types import Scope
from faststream.rabbit import Channel, RabbitBroker
from faststream.rabbit.annotations import RabbitMessage
from pydantic import ValidationError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from hexrl_platform.core.config import get_settings
from hexrl_platform.core.logging import configure_logging
from hexrl_platform.db.schema import create_schema
from hexrl_platform.db.session import create_engine, create_session_factory
from hexrl_platform.inference.config import InferenceSettings, get_inference_settings
from hexrl_platform.inference.model_loader import LoadedModel, load_champion
from hexrl_platform.inference.predict import InvalidStateError, predict_action
from hexrl_platform.task_pipeline.config import get_pipeline_settings
from hexrl_platform.task_pipeline.repository import TaskRepository
from hexrl_platform.task_pipeline.retries import decide
from hexrl_platform.task_pipeline.runner import InferenceRunner, limit_torch_threads
from hexrl_platform.task_pipeline.schemas import TaskMessage, TaskStatus
from hexrl_platform.task_pipeline.topology import Topology, build_topology, declare_topology

logger = logging.getLogger(__name__)

WORKER_KEY = "worker"
DEAD_REASON_HEADER = "x-dead-reason"
# запас поверх самой долгой обработки, чтобы брокер успел дождаться незавершённых задач
GRACE_MARGIN_S = 10.0


@dataclass(frozen=True, slots=True)
class WorkerState:
    broker: RabbitBroker
    topology: Topology
    model: LoadedModel
    runner: InferenceRunner
    sessions: async_sessionmaker[AsyncSession]
    max_deliveries: int


Worker = Annotated[WorkerState, Context(WORKER_KEY)]
Loader = Callable[[InferenceSettings], LoadedModel]
EngineFactory = Callable[[], AsyncEngine]


async def _dead_letter(
    state: WorkerState, message: RabbitMessage, reason: str, attempt: int
) -> None:
    """Отправляет сообщение в DLQ и только потом подтверждает исходное.

    reject(requeue=False) отправил бы его в DLX основной очереди, то есть в
    retry-очередь, а оттуда обратно в работу - это путь повтора, а не отказа.
    """
    await state.broker.publish(
        message.body,
        exchange=state.topology.dlx,
        routing_key=state.topology.dead_routing_key,
        persist=True,
        headers={**message.headers, DEAD_REASON_HEADER: reason},
    )
    await message.ack()
    logger.warning("task_dead_lettered", extra={"error_type": reason, "status_code": attempt})


async def _handle_failure(
    state: WorkerState,
    message: RabbitMessage,
    task: TaskMessage,
    error: Exception,
    *,
    permanent: bool,
) -> None:
    decision = decide(
        headers=message.headers,
        queue_name=state.topology.tasks_queue.name,
        max_deliveries=state.max_deliveries,
        permanent=permanent,
    )
    async with state.sessions() as session:
        repository = TaskRepository(session)
        if decision.retry:
            await repository.record_attempt(task.task_id, decision.attempt, str(error))
        else:
            await repository.mark_failed(task.task_id, str(error), decision.attempt)
        await session.commit()

    if decision.retry:
        # в DLX основной очереди -> retry-очередь -> TTL -> обратно в работу
        await message.reject()
        logger.warning(
            "task_retry_scheduled",
            extra={"error_type": type(error).__name__, "status_code": decision.attempt},
        )
        return
    await _dead_letter(state, message, decision.reason, decision.attempt)


async def handle_task(message: RabbitMessage, state: Worker) -> None:
    try:
        task = TaskMessage.model_validate_json(bytes(message.body))
    except ValidationError:
        # невалидное сообщение не станет валидным от повтора
        await _dead_letter(state, message, "validation_error", attempt=1)
        return

    async with state.sessions() as session:
        repository = TaskRepository(session)
        await repository.ensure_task(task)
        if await repository.get_status(task.task_id) is TaskStatus.COMPLETED:
            # оптимизация, а не защита: защита - уникальный ключ в inference_results
            await session.commit()
            await message.ack()
            return
        await repository.mark_processing(task.task_id)
        await session.commit()

    try:
        prediction = await state.runner.run(task.payload)
    except InvalidStateError as error:
        await _handle_failure(state, message, task, error, permanent=True)
        return
    except Exception as error:  # noqa: BLE001 - воркер не должен падать из-за одной задачи
        await _handle_failure(state, message, task, error, permanent=False)
        return

    async with state.sessions() as session:
        await TaskRepository(session).save_result(task.task_id, prediction, state.model.version)
        await session.commit()

    # ack строго после commit: падение между ними даёт повторную доставку,
    # которую отобьёт уникальный ключ, а обратный порядок потерял бы задачу
    await message.ack()


def create_worker_app(
    loader: Loader = load_champion,
    engine_factory: EngineFactory | None = None,
) -> AsgiFastStream:
    """Фабрика приложения.

    loader и engine_factory подменяются в тестах: поднимать MLflow и Postgres не нужно.
    """
    core = get_settings()
    pipeline = get_pipeline_settings()
    inference = get_inference_settings()
    configure_logging(core.log_level)

    graceful_timeout = pipeline.prefetch_count * pipeline.inference_delay_s + GRACE_MARGIN_S
    broker = RabbitBroker(pipeline.rabbitmq_url, graceful_timeout=graceful_timeout)
    topology = build_topology(
        pipeline.queue_prefix, pipeline.queue_max_length, pipeline.retry_delay_ms
    )
    broker.subscriber(
        topology.tasks_queue,
        topology.tasks_exchange,
        channel=Channel(prefetch_count=pipeline.prefetch_count),
        # MANUAL: подтверждаем сами и строго после commit
        ack_policy=AckPolicy.MANUAL,
    )(handle_task)

    loaded: dict[str, LoadedModel] = {}

    @asynccontextmanager
    async def lifespan(context: ContextRepo) -> AsyncIterator[None]:
        await declare_topology(broker, topology)

        engine = engine_factory() if engine_factory is not None else create_engine(core.db_dsn)
        await create_schema(engine)

        limit_torch_threads()
        # один раз на процесс и до приёма сообщений: из обработчика грузить нельзя
        model = loader(inference)
        loaded["model"] = model
        logger.info(
            "model_loaded",
            extra={"component": model.name, "component_version": model.version},
        )

        runner = InferenceRunner(
            partial(predict_action, model.model),
            pipeline.worker_concurrency,
            pipeline.inference_delay_s,
        )
        context.set_global(
            WORKER_KEY,
            WorkerState(
                broker=broker,
                topology=topology,
                model=model,
                runner=runner,
                sessions=create_session_factory(engine),
                max_deliveries=pipeline.max_deliveries,
            ),
        )
        try:
            yield
        finally:
            # сюда попадаем после остановки брокера, то есть когда задачи доработали
            runner.shutdown()
            await engine.dispose()
            context.reset_global(WORKER_KEY)
            logger.info("worker_stopped")

    @get
    async def model_route(scope: Scope) -> AsgiResponse:
        """Какая версия модели сейчас обслуживается."""
        model = loaded.get("model")
        if model is None:
            return AsgiResponse(b'{"detail": "model is not loaded"}', status_code=503)
        body = json.dumps(
            {
                "name": model.name,
                "version": model.version,
                "alias": model.alias,
                "run_id": model.run_id,
                "max_steps": model.max_steps,
            }
        ).encode()
        return AsgiResponse(body, status_code=200, headers={"content-type": "application/json"})

    return AsgiFastStream(
        broker,
        asgi_routes=[
            # отвечает во время inference: сам inference живёт в потоке, event loop свободен
            ("/healthz", make_ping_asgi(broker, timeout=5.0)),
            ("/model", model_route),
        ],
        lifespan=lifespan,
    )


def main() -> None:
    inference = get_inference_settings()
    uvicorn.run(
        create_worker_app(),
        host=inference.inference_host,
        port=inference.inference_port,
        access_log=False,
    )
