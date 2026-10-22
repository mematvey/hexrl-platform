import argparse
import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import asyncpg
import httpx
from faststream.rabbit import RabbitBroker

from hexrl_platform.task_pipeline.publisher import publish_confirmed
from hexrl_platform.task_pipeline.topology import build_topology
from scripts.backpressure.collect_metrics import append_csv, database
from scripts.backpressure.run_experiments import source_manifest, wait_healthy
from scripts.backpressure.runtime import Runtime

PAYLOAD = {"q": 0, "r": 0, "goal_q": 2, "goal_r": -1, "steps_remaining": 20}
HEALTH_PROBE = """
import json, time, urllib.request
rows = []
for index in range(5):
    started = time.perf_counter()
    with urllib.request.urlopen('http://localhost:8001/healthz', timeout=1) as response:
        rows.append(dict(probe=index, status=response.status,
                         elapsed_ms=(time.perf_counter()-started)*1000))
    time.sleep(0.1)
print(json.dumps(rows))
"""


async def wait_api(client: httpx.AsyncClient) -> None:
    async with asyncio.timeout(60):
        while True:
            try:
                if (await client.get("/healthz")).status_code == 200:
                    return
            except httpx.TransportError:
                pass
            await asyncio.sleep(0.2)


async def task_rows(connection: asyncpg.Connection, ids: list[str]) -> list[dict]:
    rows = await connection.fetch(
        "SELECT t.task_id, t.status, t.attempts, r.action, r.model_name, r.model_version "
        "FROM inference_tasks t LEFT JOIN inference_results r USING(task_id) "
        "WHERE t.task_id=ANY($1::uuid[]) ORDER BY t.task_id",
        [UUID(item) for item in ids],
    )
    return [dict(row) for row in rows]


async def wait_status(
    connection: asyncpg.Connection, ids: list[str], status: str, timeout_s: float = 30
) -> list[dict]:
    async with asyncio.timeout(timeout_s):
        while True:
            rows = await task_rows(connection, ids)
            if len(rows) == len(ids) and all(row["status"] == status for row in rows):
                return rows
            await asyncio.sleep(0.05)


async def start_phase(runtime: Runtime) -> list[dict]:
    await runtime.run("stop", "worker")
    await runtime.run(
        "up", "-d", "--no-deps", "--force-recreate", "--scale", "worker=1", "app", "worker"
    )
    return await wait_healthy(runtime, 1)


async def check_graceful(args: argparse.Namespace) -> dict:
    runtime = Runtime(
        args.env_file,
        args.project,
        {
            "QUEUE_PREFIX": f"hexrl.defense.graceful.{uuid4().hex}",
            "QUEUE_MAX_LENGTH": "20",
            "INFERENCE_DELAY_S": "10",
            "WORKER_CONCURRENCY": "2",
            "PREFETCH_COUNT": "2",
            "BENCHMARK_TRANSIENT_FAILURES": "0",
        },
    )
    models = await start_phase(runtime)
    connection = await database(runtime)
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{runtime.environment.get('APP_PORT', '8000')}", timeout=5
        ) as client:
            await wait_api(client)
            ids = []
            for _ in range(2):
                response = await client.post("/api/v1/tasks", json=PAYLOAD)
                assert response.status_code == 202, response.text
                ids.append(response.json()["task_id"])
        before = await wait_status(connection, ids, "PROCESSING")
        probes = json.loads(await runtime.run("exec", "-T", "worker", "python", "-c", HEALTH_PROBE))
        after = await task_rows(connection, ids)
        assert all(row["status"] == "PROCESSING" for row in after)
        for probe in probes:
            assert probe["status"] == 204
            assert probe["elapsed_ms"] < 500
            append_csv(
                args.output / "health.csv",
                {
                    **probe,
                    "processing_before": len(before),
                    "processing_after": len(after),
                    "configured_inference_delay_s": 10,
                },
            )
        started = time.monotonic()
        await runtime.run("stop", "--timeout", "40", "worker")
        elapsed = time.monotonic() - started
        completed = await task_rows(connection, ids)
        assert len(completed) == 2
        assert all(row["status"] == "COMPLETED" and row["action"] is not None for row in completed)
        containers = [
            json.loads(line)
            for line in (
                await runtime.run("ps", "--all", "--format", "json", "worker")
            ).splitlines()
            if line.strip()
        ]
        assert len(containers) == 1 and containers[0]["ExitCode"] == 0
        log = await runtime.run("logs", "--no-color", "worker")
        assert log.count('"event": "model_loaded"') == 1
        (args.output / "graceful-worker.log").write_text(log.rstrip() + "\n", encoding="utf-8")
        return {
            "model": models[0],
            "processing_at_signal": 2,
            "completed_after_stop": completed,
            "sigterm_stop_s": elapsed,
            "worker_exit_code": 0,
            "model_load_events": 1,
            "max_health_ms": max(probe["elapsed_ms"] for probe in probes),
        }
    finally:
        await connection.close()


async def dlq_messages(runtime: Runtime) -> list[dict]:
    env = runtime.environment
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{env.get('RABBITMQ_MANAGEMENT_PORT', '15672')}",
        auth=(env["RABBITMQ_USER"], env["RABBITMQ_PASSWORD"]),
        timeout=5,
    ) as client:
        response = await client.post(
            f"/api/queues/%2F/{env['QUEUE_PREFIX']}.dlq/get",
            json={"count": 10, "ackmode": "ack_requeue_true", "encoding": "auto"},
        )
        response.raise_for_status()
        return response.json()


async def check_poison(args: argparse.Namespace) -> dict:
    runtime = Runtime(
        args.env_file,
        args.project,
        {
            "QUEUE_PREFIX": f"hexrl.defense.poison.{uuid4().hex}",
            "QUEUE_MAX_LENGTH": "20",
            "INFERENCE_DELAY_S": "0.05",
            "BENCHMARK_TRANSIENT_FAILURES": "100",
        },
    )
    models = await start_phase(runtime)
    connection = await database(runtime)
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{runtime.environment.get('APP_PORT', '8000')}", timeout=5
        ) as client:
            await wait_api(client)
            response = await client.post("/api/v1/tasks", json=PAYLOAD)
            assert response.status_code == 202
            task_id = response.json()["task_id"]
        failed = (await wait_status(connection, [task_id], "FAILED"))[0]
        settings = runtime.pipeline_settings
        assert failed["attempts"] == settings.max_deliveries
        assert failed["action"] is None
        invalid = {"invalid": "defense-check"}
        topology = build_topology(settings.queue_prefix, 20, settings.retry_delay_ms)
        async with RabbitBroker(settings.rabbitmq_url) as broker:
            await publish_confirmed(
                broker,
                json.dumps(invalid).encode(),
                exchange=topology.tasks_exchange,
                routing_key=topology.tasks_routing_key,
                timeout_s=1,
            )
        async with asyncio.timeout(15):
            while True:
                messages = await dlq_messages(runtime)
                if len(messages) == 2:
                    break
                await asyncio.sleep(0.1)
        by_reason = {item["properties"]["headers"]["x-dead-reason"]: item for item in messages}
        exhausted = by_reason["max_deliveries_reached"]
        assert json.loads(exhausted["payload"])["task_id"] == task_id
        assert exhausted["properties"]["headers"]["x-attempt"] == settings.max_deliveries
        assert json.loads(by_reason["validation_error"]["payload"]) == invalid
        log = await runtime.run("logs", "--no-color", "worker")
        retries = log.count('"event": "task_retry_scheduled"')
        assert retries == settings.max_deliveries - 1
        (args.output / "poison-worker.log").write_text(log.rstrip() + "\n", encoding="utf-8")
        for item in messages:
            append_csv(
                args.output / "dlq.csv",
                {
                    "payload": item["payload"],
                    "reason": item["properties"]["headers"]["x-dead-reason"],
                    "attempt": item["properties"]["headers"]["x-attempt"],
                },
            )
        return {
            "model": models[0],
            "failed": failed,
            "retry_events": retries,
            "max_deliveries": settings.max_deliveries,
            "malformed_message_in_dlq": True,
        }
    finally:
        await connection.close()


async def check(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=False)
    summary = {
        "started_at": datetime.now(UTC).isoformat(),
        "code": await asyncio.to_thread(source_manifest),
    }
    try:
        summary["graceful"] = await check_graceful(args)
        summary["poison"] = await check_poison(args)
        (args.output / "summary.json").write_text(
            json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        # Возвращаем обычный общий префикс из .env, чтобы ручные dc-команды не
        # переключали только воркер на другую очередь после автоматической проверки.
        runtime = Runtime(args.env_file, args.project, {"BENCHMARK_TRANSIENT_FAILURES": "0"})
        await runtime.run("stop", "worker")
        await runtime.run(
            "up", "-d", "--no-deps", "--force-recreate", "--scale", "worker=1", "app", "worker"
        )
        await wait_healthy(runtime, 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prove health responsiveness, graceful stop and DLQ"
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--project", default="hexrl-hw4")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(asyncio.run(check(parser.parse_args())), indent=2, default=str))
