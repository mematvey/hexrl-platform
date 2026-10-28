import asyncio
import csv
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import httpx

from scripts.backpressure.runtime import Runtime, container_rss


def append_csv(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


async def database(runtime: Runtime) -> asyncpg.Connection:
    env = runtime.environment
    return await asyncpg.connect(
        host="127.0.0.1",
        port=int(env.get("BENCHMARK_DB_PORT", "25432")),
        user=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        database=env["POSTGRES_DB"],
    )


async def snapshot(
    client: httpx.AsyncClient,
    connection: asyncpg.Connection,
    runtime: Runtime,
    prefix: str,
    batch_id: str,
    started: float,
) -> dict[str, Any]:
    responses = await asyncio.gather(
        *(client.get(f"/api/queues/%2F/{prefix}.{suffix}") for suffix in ("tasks", "retry", "dlq")),
        client.get("/api/nodes"),
    )
    for response in responses:
        response.raise_for_status()
    tasks, retry, dlq, nodes = (response.json() for response in responses)
    counts = await connection.fetch(
        "SELECT status, count(*) AS n FROM inference_tasks WHERE batch_id=$1 GROUP BY status",
        batch_id,
    )
    statuses = {record["status"]: record["n"] for record in counts}
    worker_ids = (await runtime.run("ps", "-q", "worker")).splitlines()
    rss = sum(await asyncio.gather(*(container_rss(item) for item in worker_ids)))
    stats = tasks.get("message_stats", {})
    return {
        "ts": datetime.now(UTC).isoformat(),
        "elapsed_s": round(time.monotonic() - started, 3),
        "queue_ready": tasks.get("messages_ready"),
        "queue_unacked": tasks.get("messages_unacknowledged"),
        "retry_messages": retry.get("messages"),
        "dlq_messages": dlq.get("messages"),
        "consumers": tasks.get("consumers"),
        "broker_publish_total": stats.get("publish"),
        "broker_ack_total": stats.get("ack"),
        "redelivered_total": stats.get("redeliver"),
        "broker_memory_bytes": sum(node["mem_used"] for node in nodes),
        "worker_rss_bytes": rss,
        "completed": statuses.get("COMPLETED", 0),
        "failed": statuses.get("FAILED", 0),
        "processing": statuses.get("PROCESSING", 0),
        "pending": statuses.get("PENDING", 0),
    }


async def collect(
    runtime: Runtime,
    prefix: str,
    batch_id: str,
    output: Path,
    stop: asyncio.Event,
    interval_s: float = 1.0,
) -> None:
    env = runtime.environment
    connection = await database(runtime)
    started = time.monotonic()
    seen = set()
    previous_poll = started
    try:
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{env.get('RABBITMQ_MANAGEMENT_PORT', '15672')}",
            auth=(env["RABBITMQ_USER"], env["RABBITMQ_PASSWORD"]),
            timeout=5.0,
        ) as client:
            while not stop.is_set():
                tick = time.monotonic()
                row = await snapshot(client, connection, runtime, prefix, batch_id, started)
                terminal = await connection.fetch(
                    "SELECT task_id FROM inference_tasks WHERE batch_id=$1 "
                    "AND status IN ('COMPLETED','FAILED')",
                    batch_id,
                )
                observed_at = datetime.now(UTC).isoformat()
                observed_elapsed = time.monotonic() - started
                poll_gap = time.monotonic() - previous_poll
                previous_poll = time.monotonic()
                for record in terminal:
                    task_id = str(record["task_id"])
                    if task_id not in seen:
                        await asyncio.to_thread(
                            append_csv,
                            output.with_name("observations.csv"),
                            {
                                "task_id": task_id,
                                "observed_at": observed_at,
                                "elapsed_s": observed_elapsed,
                                "poll_gap_s": poll_gap,
                            },
                        )
                        seen.add(task_id)
                await asyncio.to_thread(append_csv, output, row)
                try:
                    await asyncio.wait_for(
                        stop.wait(), timeout=max(0.01, interval_s - (time.monotonic() - tick))
                    )
                except TimeoutError:
                    pass
    finally:
        await connection.close()


async def export_tasks(runtime: Runtime, batch_id: str, path: Path) -> list[dict[str, Any]]:
    connection = await database(runtime)
    try:
        records = await connection.fetch(
            """SELECT t.task_id, t.status, t.attempts, t.submitted_at, t.created_at,
                      CASE WHEN r.task_id IS NOT NULL THEN r.created_at
                           WHEN t.status='FAILED' THEN t.updated_at END AS completed_at,
                      r.model_name, r.model_version, r.inference_ms, r.delay_ms, t.last_error
                 FROM inference_tasks t LEFT JOIN inference_results r USING(task_id)
                WHERE t.batch_id=$1 ORDER BY t.submitted_at, t.task_id""",
            batch_id,
        )
        rows = []
        for record in records:
            row = dict(record)
            completed_at = row["completed_at"]
            row["end_to_end_ms"] = (
                (completed_at - row["submitted_at"]).total_seconds() * 1000
                if completed_at is not None
                else None
            )
            rows.append(row)
            await asyncio.to_thread(append_csv, path, row)
        return rows
    finally:
        await connection.close()
