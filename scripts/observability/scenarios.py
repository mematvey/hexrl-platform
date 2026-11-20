import argparse
import asyncio
import csv
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from faststream.rabbit import RabbitBroker

from hexrl_platform.task_pipeline.publisher import publish_confirmed
from hexrl_platform.task_pipeline.topology import build_topology
from scripts.observability.analysis import (
    error_budget,
    first_time,
    good_fraction,
    histogram_series,
)
from scripts.observability.stand import (
    DEFAULT_PROJECT,
    ROOT,
    Grafana,
    Report,
    Stand,
    eventually,
    task_path,
)

PAYLOAD = {"q": 0, "r": 0, "goal_q": 2, "goal_r": -1, "steps_remaining": 20}
# Вне карты радиуса 4: модель отвергает состояние, повтор не поможет
OFF_MAP = {"q": 9, "r": 9, "goal_q": 0, "goal_r": 0, "steps_remaining": 5}
QUEUE_ALERT = "Queue near limit"
DLQ_ALERT = "DLQ contains messages"
TASKS = 'queue=~".*[.]tasks"'
COMPLETED = "hexrl_worker_completed_total"
PUBLISHED = "hexrl_source_published_total"
POSTED = 'hexrl_api_requests_total{route="/api/v1/tasks", method="POST"}'
REFUSED = 'hexrl_api_requests_total{route="/api/v1/tasks", status_code="503"}'
# Граница SLO обязана совпадать с bucket histogram task.duration
SLO_THRESHOLD_S = 10
SLO_OBJECTIVE = 0.99
TASK_BUCKETS = 13
SETTLE_S = 20
DRAIN_MARGIN_S = 10


def log_time(moment: float) -> str:
    """Формат timestamp JSON-логов: строки сравниваются с точностью до миллисекунды."""
    return (
        datetime.fromtimestamp(moment, UTC)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def dataset_rows(name: str) -> int:
    with (ROOT / "data/tasks" / f"{name}.jsonl").open("rb") as stream:
        return sum(1 for _ in stream)


async def submit(stand: Stand, payload: dict) -> str:
    async with stand.api() as api:
        response = await api.post("/api/v1/tasks", json=payload)
        response.raise_for_status()
        return response.json()["task_id"]


async def wait_task(stand: Stand, task_id: str, timeout_s: float = 90) -> dict:
    async with stand.api() as api:

        async def done() -> dict | None:
            body = (await api.get(f"/api/v1/tasks/{task_id}")).json()
            return body if body["status"] in ("COMPLETED", "FAILED") else None

        result = await eventually(done, timeout_s=timeout_s, interval_s=0.5)
    if result is None:
        raise TimeoutError(f"task {task_id} did not finish")
    return result


async def alert_state(stand: Stand, name: str) -> str:
    return (await stand.grafana().alert_states()).get(name, "missing")


async def wait_alert(stand: Stand, name: str, timeout_s: float = 150) -> float | None:
    async def firing() -> bool:
        return await alert_state(stand, name) == "firing"

    return time.time() if await eventually(firing, timeout_s=timeout_s, interval_s=5) else None


async def task_logs(stand: Stand, task_id: str, since: float) -> list[dict]:
    query = f'{{service=~"hexrl-.+"}} |= "{task_id}"'
    return task_path(await stand.grafana().loki(query, since - 60, time.time() + 5))


async def start_batch(stand: Stand, dataset: str) -> str:
    batch_id = f"{dataset}-{uuid4().hex[:8]}"
    batch = replace(
        stand, overrides={**stand.overrides, "BATCH_FILE": f"{dataset}.jsonl", "BATCH_ID": batch_id}
    )
    await batch.run("up", "-d", "--no-deps", "--force-recreate", "batch-source")
    return batch_id


async def source_state(stand: Stand) -> dict | None:
    items = [item for item in await stand.containers() if item["Service"] == "batch-source"]
    return items[0] if items else None


async def wait_batch(stand: Stand, batch_id: str, rows: int, timeout_s: float) -> dict[str, int]:
    deadline = time.monotonic() + timeout_s
    while True:
        statuses = await stand.statuses(f"batch_id='{batch_id}'")
        source = await source_state(stand)
        exited = source is not None and source["State"] == "exited"
        if exited and source.get("ExitCode") != 0:
            raise RuntimeError("batch source failed")
        if exited and statuses.get("COMPLETED", 0) + statuses.get("FAILED", 0) == rows:
            return statuses
        if time.monotonic() > deadline:
            raise TimeoutError(f"batch {batch_id} did not finish: {statuses}")
        await asyncio.sleep(5)


async def scale(stand: Stand, workers: int) -> None:
    await stand.run("up", "-d", "--no-deps", "--scale", f"worker={workers}", "worker")
    await stand.wait_ready(timeout_s=300, workers=workers)


async def restore_worker(stand: Stand) -> None:
    normal = replace(stand, faults=False)
    await normal.run("up", "-d", "--no-deps", "--force-recreate", "--scale", "worker=1", "worker")
    await normal.wait_ready(timeout_s=300)


async def scenario_load(stand: Stand, report: Report, args: argparse.Namespace) -> None:
    """Переполнение очереди, алерт, второй воркер, установившаяся скорость и SLO."""
    grafana = stand.grafana()
    await restore_worker(stand)
    rows = dataset_rows(args.dataset)
    limit = int(stand.environment["QUEUE_MAX_LENGTH"])
    started = time.time()
    batch_id = await start_batch(stand, args.dataset)
    report.facts.update({"dataset": args.dataset, "rows": rows, "batch_id": batch_id})

    alert_at = scaled_at = source_done = None
    memory: dict[str, float] = {}
    deadline = time.monotonic() + args.timeout
    while True:
        now = time.time()
        if alert_at is None and await alert_state(stand, QUEUE_ALERT) == "firing":
            alert_at = now
            print(f"queue alert firing after {now - started:.0f}s", flush=True)
        if args.scale and alert_at is not None and scaled_at is None:
            await stand.run("up", "-d", "--no-deps", "--scale", "worker=2", "worker")
            scaled_at = time.time()
            print("second worker requested", flush=True)
        if not memory and now - started > args.memory_after:
            memory = await stand.memory()
        source = await source_state(stand)
        if source_done is None and source is not None and source["State"] == "exited":
            source_done = now
            if source.get("ExitCode") != 0:
                raise RuntimeError("batch source failed")
        statuses = await stand.statuses(f"batch_id='{batch_id}'")
        terminal = statuses.get("COMPLETED", 0) + statuses.get("FAILED", 0)
        if source_done is not None and terminal == rows:
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"load did not finish: {statuses}")
        await asyncio.sleep(5)
    finished = time.time()
    await asyncio.sleep(SETTLE_S)

    queries = {
        "published": "sum(hexrl_source_published_total)",
        "completed": "sum(hexrl_worker_completed_total)",
        "ready": f"sum(rabbitmq_detailed_queue_messages_ready{{{TASKS}}})",
        "unacked": f"sum(rabbitmq_detailed_queue_messages_unacked{{{TASKS}}})",
        "workers": 'count(hexrl_process_running_ratio{job="hexrl-worker"} == 1)',
        "published_rate": "sum(rate(hexrl_source_published_total[30s]))",
        "completed_rate": "sum(rate(hexrl_worker_completed_total[30s]))",
    }
    window = (started - 30, finished + SETTLE_S)
    series = {
        name: await grafana.prom_range(expr, *window, step=5) for name, expr in queries.items()
    }
    write_series(args.output / f"load-{args.dataset}.csv", series)

    full_at = first_time(series["ready"], 0.9 * limit, after=started)
    # Конец источника - начало опустошения очереди; его и финальный экспорт не берём
    end = (source_done or finished) - DRAIN_MARGIN_S
    # Фаза 1: от заполнения очереди до второго воркера; фаза 2: после прогрева до конца
    # источника. Разгон до заполнения и финальное опустошение очереди не входят.
    phases = {"one_worker": (full_at or started, scaled_at or end)}
    if scaled_at is not None:
        phases["two_workers"] = (scaled_at + 30, end)
    rates = {
        name: {
            "window_s": round(stop - begin, 1),
            "completed_per_s": await window_rate(grafana, COMPLETED, begin, stop),
            "published_per_s": await window_rate(grafana, PUBLISHED, begin, stop),
        }
        for name, (begin, stop) in phases.items()
    }
    statuses = await stand.statuses(f"batch_id='{batch_id}'")
    results = await stand.sql(
        "SELECT count(*) FROM inference_results r JOIN inference_tasks t USING(task_id) "
        f"WHERE t.batch_id='{batch_id}'"
    )
    run = f"{int(finished - started) + SETTLE_S}s"
    at = finished + SETTLE_S
    rejections = await grafana.prom_value(
        f"sum(increase(hexrl_source_rejections_total[{run}]))", at
    )
    waited = await grafana.prom_value(f"sum(increase(hexrl_source_wait_seconds_total[{run}]))", at)
    waits = await grafana.search_traces('{name="source.wait"}', started, finished)
    report.facts.update(
        {
            "duration_s": round(finished - started, 1),
            "queue_full_after_s": None if full_at is None else round(full_at - started, 1),
            "alert_after_s": None if alert_at is None else round(alert_at - started, 1),
            "second_worker_after_s": None if scaled_at is None else round(scaled_at - started, 1),
            "statuses": statuses,
            "results": int(results[0][0]),
            "rates": rates,
            "source_rejections": rejections,
            "source_wait_s": waited,
            "memory_mib": memory,
        }
    )
    report.check("every accepted line has a terminal task", sum(statuses.values()) == rows)
    report.check("no task is lost or duplicated", int(results[0][0]) == statuses.get("COMPLETED"))
    report.check("queue reached 90% of the limit", full_at is not None, limit=limit)
    report.check(
        "source slowed down instead of flooding", (rejections or 0) > 0 and (waited or 0) > 0
    )
    report.check("source.wait is visible in traces", bool(waits), traces=len(waits))
    report.check("queue alert went Firing", alert_at is not None)
    if args.scale:
        span = f"{int(finished - started)}s"
        instances = await grafana.prom(
            f"sum by (instance) (increase(hexrl_worker_completed_total[{span}]))", at=finished
        )
        names = [item["metric"].get("instance") for item in instances]
        traced = {
            name: bool(
                await grafana.search_traces(
                    f'{{resource.service.instance.id="{name}" && name="inference"}}',
                    started,
                    finished,
                )
            )
            for name in names
        }
        report.facts["worker_instances"] = traced
        report.check(
            "two worker instances appear in traces", len(traced) == 2 and all(traced.values())
        )
        one, two = rates["one_worker"], rates.get("two_workers", {})
        one_done, two_done = one["completed_per_s"], two.get("completed_per_s")
        one_sent, two_sent = one["published_per_s"], two.get("published_per_s")
        if one_done and two_done:
            report.facts["speedup"] = round(two_done / one_done, 2)
        report.check(
            "second worker raised throughput and source speed",
            bool(one_done and two_done and two_done > one_done)
            and bool(one_sent and two_sent and two_sent > one_sent),
        )
    good, total = await slo_counts(stand, started, finished + SETTLE_S)
    failed = statuses.get("FAILED", 0)
    fraction = good_fraction(good, total + failed)
    report.facts["slo"] = {
        "threshold_s": SLO_THRESHOLD_S,
        "objective": SLO_OBJECTIVE,
        "fast_successes": good,
        "slow_successes": total - good,
        "terminal_errors": failed,
        "good_fraction": fraction,
        "error_budget_tasks": error_budget(total + failed, SLO_OBJECTIVE),
    }
    report.check("histogram covers every completed task", total == statuses.get("COMPLETED"))
    report.facts["cardinality"] = await cardinality(stand, rows)
    plot_load(series, args.output / f"load-{args.dataset}.png", limit, started, alert_at, scaled_at)


async def slo_counts(stand: Stand, start: float, end: float) -> tuple[float, float]:
    """Быстрые и все успешные задачи прогона по сырым histogram текущих воркеров.

    Сумма по всем сериям не годится: ряды прошлых процессов выпадают из lookback
    посреди окна. increase() тоже: первая точка новой серии теряется.
    """
    workers = [item["ID"][:12] for item in await stand.containers() if item["Service"] == "worker"]
    selector = f'instance=~"{"|".join(workers)}"'
    good = f'{selector}, le=~"{SLO_THRESHOLD_S}(\\\\.0)?"'
    grafana = stand.grafana()
    counts = []
    for metric, labels in (
        ("hexrl_worker_task_duration_seconds_bucket", good),
        ("hexrl_worker_task_duration_seconds_count", selector),
    ):
        after = await grafana.prom_value(f"sum({metric}{{{labels}}})", end) or 0.0
        before = await grafana.prom_value(f"sum({metric}{{{labels}}})", start) or 0.0
        counts.append(after - before)
    return counts[0], counts[1]


async def window_rate(grafana: Grafana, counter: str, begin: float, end: float) -> float | None:
    """Скорость за окно через increase по каждой серии: появление новой серии
    (второй воркер) и исчезновение устаревшей (прошлый источник) не ломают разность."""
    if end - begin < 10:
        return None
    increase = await grafana.prom_value(f"sum(increase({counter}[{int(end - begin)}s]))", end)
    return None if increase is None else increase / (end - begin)


async def cardinality(stand: Stand, tasks: int) -> dict:
    grafana = stand.grafana()
    by_metric = await grafana.prom('count by (__name__) ({__name__=~"hexrl_.+"})')
    counts = {item["metric"]["__name__"]: int(item["value"][1]) for item in by_metric}
    duration = histogram_series(TASK_BUCKETS, 1)
    return {
        "hexrl_series": sum(counts.values()),
        "by_metric": dict(sorted(counts.items())),
        # Гипотеза для защиты: тот же histogram с label task_id
        "task_duration_now": duration,
        "task_duration_with_task_id": histogram_series(TASK_BUCKETS, tasks),
    }


async def scenario_errors(stand: Stand, report: Report, _args: argparse.Namespace) -> None:
    """Нарушенная схема и доменная ошибка - сразу в DLQ; сбой inference - повторы, затем DLQ."""
    env = stand.environment
    max_deliveries = int(env.get("MAX_DELIVERIES", "3"))
    await restore_worker(stand)
    started = time.time()

    marker = uuid4().hex
    topology = build_topology(
        env["QUEUE_PREFIX"], int(env["QUEUE_MAX_LENGTH"]), int(env["RETRY_DELAY_MS"])
    )
    url = (
        f"amqp://{env['RABBITMQ_USER']}:{env['RABBITMQ_PASSWORD']}"
        f"@127.0.0.1:{env['RABBITMQ_PORT']}/"
    )
    async with RabbitBroker(url) as broker:
        await publish_confirmed(
            broker,
            json.dumps({"invalid": marker}).encode(),
            exchange=topology.tasks_exchange,
            routing_key=topology.tasks_routing_key,
            timeout_s=2,
        )

    async def schema_in_dlq() -> dict | None:
        return next(
            (item for item in await stand.peek("dlq", 500) if marker in item["payload"]), None
        )

    dead = await eventually(schema_in_dlq, timeout_s=30)
    headers = dead["properties"]["headers"] if dead else {}
    report.check(
        "broken schema goes to DLQ at once",
        headers.get("x-dead-reason") == "validation_error" and headers.get("x-attempt") == 1,
        headers={key: headers.get(key) for key in ("x-dead-reason", "x-attempt")},
    )

    domain_id = await submit(stand, OFF_MAP)
    domain = await wait_task(stand, domain_id)
    report.check(
        "domain error goes to DLQ without retries",
        domain["status"] == "FAILED" and domain.get("attempts", 1) == 1,
        task_id=domain_id,
        result=domain,
    )

    faulty = replace(stand, faults=True, overrides={"BENCHMARK_TRANSIENT_FAILURES": "1000"})
    await faulty.run("up", "-d", "--no-deps", "--force-recreate", "--scale", "worker=1", "worker")
    await faulty.wait_ready(timeout_s=300)
    try:
        task_id = await submit(stand, PAYLOAD)
        result = await wait_task(stand, task_id, timeout_s=120)
    finally:
        await restore_worker(stand)
    report.facts["transient_task_id"] = task_id
    report.check(
        "transient error is retried and then dead-lettered",
        result["status"] == "FAILED" and result.get("attempts") == max_deliveries,
        result=result,
    )

    async def logs() -> list[dict]:
        records = await task_logs(stand, task_id, started)
        return records if any(r.get("event") == "task_dead_lettered" for r in records) else []

    records = await eventually(logs, timeout_s=90)
    failures = [r for r in records if r.get("event") == "task_processing_failed"]
    trace_ids = {r["trace_id"] for r in records if r.get("trace_id")}
    report.facts["transient_logs"] = [
        {key: r.get(key) for key in ("timestamp", "service", "level", "event", "attempt", "reason")}
        for r in records
    ]
    report.check(
        "every attempt is logged as ERROR with its number",
        sorted(int(r.get("attempt") or 0) for r in failures) == list(range(1, max_deliveries + 1))
        and all(r.get("level") == "ERROR" for r in failures),
    )
    report.check(
        "all attempts share one trace_id", len(trace_ids) == 1, trace_ids=sorted(trace_ids)
    )
    if len(trace_ids) == 1:
        trace_id = next(iter(trace_ids))
        report.facts["transient_trace_id"] = trace_id

        async def traced() -> dict | None:
            summary = await stand.grafana().trace(trace_id)
            return summary if summary and summary["error_spans"] else None

        summary = await eventually(traced, timeout_s=90)
        report.check("trace marks the failure as ERROR", summary is not None, trace=summary)
    alert = await wait_alert(stand, DLQ_ALERT)
    report.check("DLQ alert went Firing", alert is not None)


async def scenario_lifecycle(stand: Stand, report: Report, args: argparse.Namespace) -> None:
    """SIGTERM доделывает взятые задачи; SIGKILL возвращает их брокеру."""
    grafana = stand.grafana()
    await restore_worker(stand)
    rows = dataset_rows("smoke")
    batch_id = await start_batch(stand, "smoke")
    where = f"batch_id='{batch_id}'"

    async def busy() -> int:
        return (await stand.statuses(where)).get("PROCESSING", 0)

    await eventually(busy, timeout_s=60, interval_s=0.2)
    container = next(item for item in await stand.containers() if item["Service"] == "worker")
    instance = container["ID"][:12]
    in_flight = await busy()
    signal_at = time.time()
    await stand.run("stop", "--timeout", "40", "worker")
    stopped_after = time.time() - signal_at
    after = await stand.statuses(where)
    exited = next(item for item in await stand.containers() if item["Service"] == "worker")
    queue = await stand.queue("tasks")
    logs = await stand.logs("worker")
    events = [record for record in logs if record.get("timestamp")]
    completed_after_signal = sum(
        1
        for record in events
        if record.get("event") == "task_completed" and record["timestamp"] >= log_time(signal_at)
    )
    names = [record.get("event") for record in events]
    report.facts["sigterm"] = {
        "in_flight_at_signal": in_flight,
        "stop_s": round(stopped_after, 2),
        "completed_after_signal": completed_after_signal,
        "exit_code": exited.get("ExitCode"),
        "unacked_after_stop": queue.get("messages_unacknowledged"),
    }
    report.check(
        "worker logged the start and end of shutdown",
        "worker_stopping" in names and "worker_stopped" in names,
    )
    report.check(
        "tasks taken before SIGTERM were finished and saved",
        in_flight > 0 and after.get("PROCESSING", 0) == 0 and completed_after_signal >= 1,
    )
    report.check(
        "messages were acked and the worker exited cleanly",
        queue.get("messages_unacknowledged") == 0 and exited.get("ExitCode") == 0,
    )

    async def final_export() -> bool:
        value = await grafana.prom_value(
            f'hexrl_process_running_ratio{{job="hexrl-worker", instance="{instance}"}}'
        )
        return value == 0

    report.check(
        "final telemetry export reported running=0", await eventually(final_export, timeout_s=30)
    )

    await stand.run("start", "worker")
    await stand.wait_ready(timeout_s=300)
    await eventually(busy, timeout_s=60, interval_s=0.2)
    redelivered_before = await grafana.prom_value("sum(hexrl_worker_redeliveries_total)") or 0
    await stand.run("kill", "-s", "SIGKILL", "worker")
    stuck = await busy()
    await stand.run("up", "-d", "--no-deps", "worker")
    await stand.wait_ready(timeout_s=300)
    statuses = await wait_batch(stand, batch_id, rows, args.timeout)
    results = await stand.sql(
        "SELECT count(*) FROM inference_results r JOIN inference_tasks t USING(task_id) "
        f"WHERE t.{where}"
    )
    await asyncio.sleep(SETTLE_S)
    redelivered = await grafana.prom_value("sum(hexrl_worker_redeliveries_total)") or 0
    report.facts["sigkill"] = {
        "processing_left_by_kill": stuck,
        "statuses": statuses,
        "results": int(results[0][0]),
        "redeliveries_seen_by_new_process": redelivered,
        "redeliveries_before_kill": redelivered_before,
    }
    report.check(
        "unacked tasks returned after SIGKILL and finished once",
        statuses.get("COMPLETED") == rows and int(results[0][0]) == rows and redelivered >= 1,
    )


async def scenario_overload(stand: Stand, report: Report, args: argparse.Namespace) -> None:
    """Очередь заполнена без потребителя: API честно отвечает 503 с Retry-After."""
    grafana = stand.grafana()
    await restore_worker(stand)
    refused_before = await grafana.prom_value(f"sum({REFUSED})") or 0.0
    posted_before = await grafana.prom_value(f"sum({POSTED})") or 0.0
    await stand.run("stop", "worker")
    started = time.time()
    responses = []
    try:
        async with stand.api() as api:
            for _ in range(args.requests):
                response = await api.post("/api/v1/tasks", json=PAYLOAD)
                responses.append(
                    {
                        "status": response.status_code,
                        "retry_after": response.headers.get("Retry-After"),
                        "task_id": response.json().get("task_id"),
                    }
                )
        pending = (await stand.statuses("status='PENDING'")).get("PENDING", 0)
    finally:
        await stand.run("start", "worker")
    await stand.wait_ready(timeout_s=300)
    accepted = [str(item["task_id"]) for item in responses if item["status"] == 202]
    rejected = [item for item in responses if item["status"] == 503]
    report.facts.update(
        {"accepted": len(accepted), "rejected": len(rejected), "pending_left": pending}
    )
    report.check("API answered 503 when the queue was full", bool(rejected))
    report.check(
        "every 503 carries Retry-After",
        all(str(item["retry_after"] or "").isdigit() for item in rejected),
        retry_after=sorted({item["retry_after"] for item in rejected}),
    )
    report.check("no PENDING rows left after refusals", pending == 0)
    finished = [await wait_task(stand, task_id) for task_id in accepted]
    report.check("accepted tasks still complete", all(t["status"] == "COMPLETED" for t in finished))

    async def refused() -> float:
        return await grafana.prom_value(f"sum({REFUSED})") or 0.0

    async def grew() -> bool:
        return await refused() - refused_before >= len(rejected)

    report.check("503 counter grew by every refusal", await eventually(grew, timeout_s=60))
    posted = await grafana.prom_value(f"sum({POSTED})") or 0.0
    report.facts["share_503"] = round(
        (await refused() - refused_before) / max(posted - posted_before, 1), 3
    )

    async def logged() -> list[dict]:
        return await grafana.loki(
            '{service="hexrl-api"} | json | status_code="503"', started - 5, time.time() + 5
        )

    lines = await eventually(logged, timeout_s=90)
    with_trace = [line for line in lines if line.get("trace_id") and line.get("task_id")]
    report.check("503 log lines carry trace_id and task_id", bool(with_trace), lines=len(lines))
    if with_trace:
        trace_id = with_trace[0]["trace_id"]

        async def traced() -> dict | None:
            summary = await grafana.trace(trace_id)
            return summary if summary and summary["error_spans"] else None

        summary = await eventually(traced, timeout_s=90)
        report.facts["rejected_trace"] = {"trace_id": trace_id, "trace": summary}
        report.check("503 is an ERROR trace with the refusal reason", summary is not None)


async def scenario_outage(stand: Stand, report: Report, args: argparse.Namespace) -> None:
    """Хранилища телеметрии падают по очереди, конвейер продолжает работать."""
    grafana = stand.grafana()
    await restore_worker(stand)
    for component in ("otel-collector", "loki"):
        before = await stand.memory()
        await stand.run("stop", component)
        stopped_at = time.time()
        latencies, ids = [], []
        try:
            async with stand.api() as api:
                for _ in range(args.requests):
                    began = time.perf_counter()
                    response = await api.post("/api/v1/tasks", json=PAYLOAD)
                    latencies.append(time.perf_counter() - began)
                    if response.status_code == 202:
                        ids.append(response.json()["task_id"])
                    await asyncio.sleep(0.3)
            results = [await wait_task(stand, task_id) for task_id in ids]
            during = await stand.memory()
        finally:
            await stand.run("start", component)
        restarted_at = time.time()
        await stand.wait_ready(timeout_s=300)
        growth = {
            service: round(during.get(service, 0) - before.get(service, 0), 1)
            for service in ("app", "worker")
        }
        fresh = await submit(stand, PAYLOAD)
        await wait_task(stand, fresh)
        if component == "otel-collector":

            async def recovered() -> bool:
                points = await grafana.search_traces(
                    '{resource.service.name="hexrl-worker"}', restarted_at, time.time()
                )
                sample = await grafana.prom_value("max(timestamp(hexrl_worker_completed_total))")
                return bool(points) and (sample or 0) > restarted_at

            lost = await grafana.search_traces(
                '{resource.service.name="hexrl-api"}', stopped_at, restarted_at - 1, limit=200
            )
            kept = {"traces_during_outage": len(lost)}
        else:

            async def recovered() -> bool:
                return bool(await task_logs(stand, fresh, restarted_at))

            found = 0
            for task_id in ids:
                found += bool(await task_logs(stand, task_id, stopped_at))
            kept = {"tasks_with_logs_from_outage": found}
        back = await eventually(recovered, timeout_s=120, interval_s=5)
        report.facts[component] = {
            "tasks": len(ids),
            "completed": sum(item["status"] == "COMPLETED" for item in results),
            "max_submit_ms": round(max(latencies) * 1000, 1),
            "memory_growth_mib": growth,
            "after_recovery": back,
            **kept,
        }
        report.check(
            f"tasks were accepted and completed without {component}",
            len(ids) == args.requests and all(t["status"] == "COMPLETED" for t in results),
        )
        report.check(
            f"app memory stayed bounded without {component}",
            all(value < 64 for value in growth.values()),
            growth=growth,
        )
        report.check(f"new {component} data appears after recovery", bool(back))


def write_series(path: Path, series: dict[str, list[tuple[float, float]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamps = sorted({ts for points in series.values() for ts, _ in points})
    lookup = {name: dict(points) for name, points in series.items()}
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ts", *series])
        for ts in stamps:
            writer.writerow([ts, *(lookup[name].get(ts, "") for name in series)])


def plot_load(
    series: dict[str, list[tuple[float, float]]],
    path: Path,
    limit: int,
    started: float,
    alert_at: float | None,
    scaled_at: float | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    def draw(axis: plt.Axes, name: str, label: str) -> None:
        points = series[name]
        axis.plot([ts - started for ts, _ in points], [value for _, value in points], label=label)

    draw(axes[0], "published_rate", "подтверждённые публикации источника, /с")
    draw(axes[0], "completed_rate", "завершённые задачи, /с")
    draw(axes[1], "ready", "ready")
    draw(axes[1], "unacked", "unacked")
    axes[1].axhline(limit, color="red", linestyle="--", label="лимит очереди")
    draw(axes[2], "workers", "работающие воркеры")
    for axis in axes:
        for moment, label, color, style in (
            (alert_at, "алерт Firing", "tab:red", ":"),
            (scaled_at, "второй воркер", "tab:green", "--"),
        ):
            if moment is not None:
                axis.axvline(moment - started, color=color, linestyle=style, label=label)
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=0.3)
    axes[2].set_xlabel("секунды от старта пакета")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


SCENARIOS: dict[str, Callable[[Stand, Report, argparse.Namespace], Awaitable[None]]] = {
    "load": scenario_load,
    "errors": scenario_errors,
    "lifecycle": scenario_lifecycle,
    "overload": scenario_overload,
    "outage": scenario_outage,
}


async def run(args: argparse.Namespace) -> Report:
    stand = Stand(args.env_file, args.project)
    report = Report(args.scenario if args.scenario != "load" else f"load-{args.dataset}")
    report.facts["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    try:
        await stand.wait_ready(timeout_s=120)
        await SCENARIOS[args.scenario](stand, report, args)
    except Exception as error:  # noqa: BLE001 - отчёт должен сохраниться при любом сбое
        report.check("scenario finished without errors", False, error=repr(error)[:500])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Defense scenarios on the observability stand")
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("--env-file", type=Path, default=Path(".env.obs"))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, default=Path("metrics/observability"))
    parser.add_argument("--dataset", choices=("smoke", "defense"), default="smoke")
    parser.add_argument("--scale", action="store_true", help="add a worker after the alert")
    parser.add_argument(
        "--memory-after", type=float, default=90, help="seconds before docker stats"
    )
    parser.add_argument("--requests", type=int, default=60)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()
    report = asyncio.run(run(args))
    print(f"report: {report.save(args.output)}")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
