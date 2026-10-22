import asyncio
import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
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
from hexrl_platform.inference.schemas import ModelInfo
from hexrl_platform.task_pipeline.config import get_pipeline_settings
from hexrl_platform.task_pipeline.publisher import publish_confirmed
from hexrl_platform.task_pipeline.repository import TaskRepository
from hexrl_platform.task_pipeline.retries import ATTEMPT_HEADER, decide
from hexrl_platform.task_pipeline.runner import InferenceRunner, limit_torch_threads
from hexrl_platform.task_pipeline.schemas import TaskMessage, TaskStatus
from hexrl_platform.task_pipeline.topology import Topology, build_topology, declare_topology

logger = logging.getLogger(__name__)

WORKER_KEY = "worker"
DEAD_REASON_HEADER = "x-dead-reason"
# запас поверх самой долгой обработки, чтобы брокер успел дождаться незавершённых задач
GRACE_MARGIN_S = 10.0
# пауза перед возвратом сообщения после сбоя обработчика: иначе недоступная БД крутит цикл вхолостую
CRASH_PAUSE_S = 1.0


@dataclass(frozen=True, slots=True)
class WorkerState:
    broker: RabbitBroker
    topology: Topology
    model: LoadedModel
    runner: InferenceRunner
    sessions: async_sessionmaker[AsyncSession]
    max_deliveries: int
    confirm_timeout_s: float


Worker = Annotated[WorkerState, Context(WORKER_KEY)]
Loader = Callable[[InferenceSettings], LoadedModel]
EngineFactory = Callable[[], AsyncEngine]


async def _dead_letter(
    state: WorkerState, message: RabbitMessage, reason: str, attempt: int
) -> None:
    # reject отправил бы сообщение в DLX основной очереди, то есть в retry; нужен именно DLQ
    await publish_confirmed(
        state.broker,
        message.body,
        exchange=state.topology.dlx,
        routing_key=state.topology.dead_routing_key,
        timeout_s=state.confirm_timeout_s,
        headers={**message.headers, DEAD_REASON_HEADER: reason},
    )
    await message.ack()
    logger.warning("task_dead_lettered", extra={"reason": reason, "attempt": attempt})


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
        # Classic DLX не подтверждает переход в retry. Сначала гарантируем приём
        # новой копии брокером, затем закрываем исходную доставку.
        await publish_confirmed(
            state.broker,
            message.body,
            exchange=state.topology.dlx,
            routing_key=state.topology.retry_routing_key,
            timeout_s=state.confirm_timeout_s,
            headers={**message.headers, ATTEMPT_HEADER: decision.attempt + 1},
            message_id=str(task.task_id),
        )
        await message.ack()
        logger.warning(
            "task_retry_scheduled",
            extra={
                "error_type": type(error).__name__,
                "task_id": str(task.task_id),
                "attempt": decision.attempt,
            },
        )
        return
    await _dead_letter(state, message, decision.reason, decision.attempt)


async def handle_task(message: RabbitMessage, state: Worker) -> None:
    try:
        await _process_task(message, state)
    except Exception as error:  # noqa: BLE001 - воркер не должен падать из-за одной задачи
        # при ручном ack исключение само ничего не подтверждает: без nack сообщение
        # осталось бы неподтверждённым и заняло слот prefetch до рестарта воркера
        logger.error(
            "task_handler_crashed", extra={"error_type": type(error).__name__, "error": str(error)}
        )
        await asyncio.sleep(CRASH_PAUSE_S)
        if message.committed is None:
            await message.nack(requeue=True)


async def _process_task(message: RabbitMessage, state: Worker) -> None:
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
        await TaskRepository(session).save_result(
            task.task_id, prediction, state.model.version, model_name=state.model.name
        )
        await session.commit()

    # ack строго после commit: падение между ними даёт повторную доставку,
    # которую отобьёт уникальный ключ, а обратный порядок потерял бы задачу
    await message.ack()


def create_worker_app(
    loader: Loader = load_champion,
    engine_factory: EngineFactory | None = None,
) -> AsgiFastStream:
    core = get_settings()
    pipeline = get_pipeline_settings()
    inference = get_inference_settings()
    configure_logging(core.log_level)

    graceful_timeout = pipeline.prefetch_count * pipeline.inference_delay_s + GRACE_MARGIN_S
    # Топология объявляется до subscriber: FastStream кэширует очередь вместе с
    # каналом первого объявления. Поэтому QoS нужен уже у default_channel.
    channel = Channel(prefetch_count=pipeline.prefetch_count)
    broker = RabbitBroker(
        pipeline.rabbitmq_url, graceful_timeout=graceful_timeout, default_channel=channel
    )
    topology = build_topology(
        pipeline.queue_prefix, pipeline.queue_max_length, pipeline.retry_delay_ms
    )
    broker.subscriber(
        topology.tasks_queue,
        topology.tasks_exchange,
        channel=channel,
        # MANUAL: подтверждаем сами и строго после commit
        ack_policy=AckPolicy.MANUAL,
    )(handle_task)

    loaded: dict[str, LoadedModel] = {}

    @asynccontextmanager
    async def lifespan(context: ContextRepo) -> AsyncIterator[None]:
        # разборка идёт в обратном порядке: контекст -> потоки -> соединения с БД,
        # и срабатывает даже если модель не загрузилась
        async with AsyncExitStack() as stack:
            # Lifespan вызывается раньше broker.start: для объявления очередей
            # соединение уже нужно, а принимать задачи до загрузки модели ещё нельзя.
            await stack.enter_async_context(broker)
            await declare_topology(broker, topology)
            engine = engine_factory() if engine_factory is not None else create_engine(core.db_dsn)
            stack.push_async_callback(engine.dispose)
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
            stack.callback(runner.shutdown)

            context.set_global(
                WORKER_KEY,
                WorkerState(
                    broker=broker,
                    topology=topology,
                    model=model,
                    runner=runner,
                    sessions=create_session_factory(engine),
                    max_deliveries=pipeline.max_deliveries,
                    confirm_timeout_s=pipeline.publish_confirm_timeout_s,
                ),
            )
            stack.callback(context.reset_global, WORKER_KEY)

            # сюда возвращаемся после остановки брокера, то есть когда задачи доработали
            yield
        logger.info("worker_stopped")

    @get
    async def model_route(scope: Scope) -> AsgiResponse:
        model = loaded.get("model")
        if model is None:
            return AsgiResponse(b'{"detail": "model is not loaded"}', status_code=503)
        info = ModelInfo(
            name=model.name,
            version=model.version,
            alias=model.alias,
            run_id=model.run_id,
            max_steps=model.max_steps,
        )
        return AsgiResponse(
            info.model_dump_json().encode(),
            status_code=200,
            headers={"content-type": "application/json"},
        )

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
