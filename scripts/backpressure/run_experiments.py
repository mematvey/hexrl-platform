import argparse
import asyncio
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import httpx
from faststream.rabbit import RabbitBroker

from hexrl_platform.task_pipeline.batch_source import task_id_for
from hexrl_platform.task_pipeline.config import PipelineSettings
from hexrl_platform.task_pipeline.topology import build_topology, declare_topology
from scripts.backpressure.collect_metrics import append_csv, collect, database, export_tasks
from scripts.backpressure.runtime import ROOT, Runtime


def source_manifest() -> dict:
    paths = sorted(
        [
            *ROOT.glob("src/**/*.py"),
            *ROOT.glob("scripts/backpressure/*.py"),
            ROOT / "uv.lock",
            ROOT / "pyproject.toml",
            ROOT / "docker-compose.yaml",
            ROOT / "scripts/backpressure/compose.benchmark.yaml",
            ROOT / "Dockerfile.inference",
        ]
    )
    checksums = {
        path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "base_commit": revision,
        "files_sha256": checksums,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


async def reconcile_dlq(runtime: Runtime, prefix: str, rows: list[dict], directory: Path) -> int:
    expected = {str(row["task_id"]) for row in rows if row["status"] == "FAILED"}
    env = runtime.environment
    async with httpx.AsyncClient(
        base_url=f"http://127.0.0.1:{env.get('RABBITMQ_MANAGEMENT_PORT', '15672')}",
        auth=(env["RABBITMQ_USER"], env["RABBITMQ_PASSWORD"]),
        timeout=10,
    ) as client:
        response = await client.post(
            f"/api/queues/%2F/{prefix}.dlq/get",
            json={
                "count": max(1, len(rows) * 3),
                "ackmode": "ack_requeue_true",
                "encoding": "auto",
            },
        )
        response.raise_for_status()
        messages = response.json()
    actual = set()
    for item in messages:
        task_id = json.loads(item["payload"])["task_id"]
        actual.add(task_id)
        headers = item["properties"].get("headers", {})
        append_csv(
            directory / "dlq.csv",
            {
                "task_id": task_id,
                "reason": headers.get("x-dead-reason"),
                "attempt": headers.get("x-attempt", 1),
            },
        )
    if actual != expected:
        raise AssertionError("DLQ task IDs do not reconcile with FAILED rows")
    return len(actual)


async def wait_healthy(runtime: Runtime, workers: int, timeout_s: float = 120) -> list[dict]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        status = await runtime.run("ps", "--format", "json", "worker")
        items = [json.loads(line) for line in status.splitlines() if line.strip()]
        if len(items) == workers and all(item.get("Health") == "healthy" for item in items):
            models = []
            for index in range(1, workers + 1):
                output = await runtime.run(
                    "exec",
                    "-T",
                    "--index",
                    str(index),
                    "worker",
                    "python",
                    "-c",
                    "import urllib.request as u; print(u.urlopen('http://localhost:8001/model').read().decode())",
                )
                models.append(json.loads(output))
            return models
        await asyncio.sleep(1)
    raise TimeoutError("Workers did not become healthy")


async def run_scenario(args: argparse.Namespace, label: str) -> dict:
    provenance = await asyncio.to_thread(source_manifest)
    directory = args.output / label
    directory.mkdir(parents=True, exist_ok=False)
    batch_id = f"hw4-{label}-{uuid4().hex[:12]}"
    prefix = f"hexrl.benchmark.{batch_id}"
    workers = 2 if label == "C" else 1
    runtime = Runtime(
        args.env_file,
        args.project,
        {
            "QUEUE_PREFIX": prefix,
            "QUEUE_MAX_LENGTH": "null" if label == "A" else "20",
            "ESTIMATED_WORKER_COUNT": str(workers),
            "BENCHMARK_TRANSIENT_FAILURES": "2" if label == "D" else "0",
        },
    )
    await runtime.run("stop", "worker")
    await runtime.run(
        "up", "-d", "--no-deps", "--force-recreate", "--scale", f"worker={workers}", "worker"
    )
    models = await wait_healthy(runtime, workers)
    if len({(model["name"], model["version"]) for model in models}) != 1:
        raise RuntimeError("Workers loaded different model versions")
    settings = PipelineSettings(
        _env_file=args.env_file, queue_prefix=prefix, queue_max_length=None if label == "A" else 20
    )
    broker = RabbitBroker(settings.rabbitmq_url)
    await broker.start()
    try:
        await declare_topology(
            broker, build_topology(prefix, settings.queue_max_length, settings.retry_delay_ms)
        )
    finally:
        await broker.stop()
    dataset = args.failure_dataset if label == "D" else args.dataset
    dataset_manifest = json.loads(dataset.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    expected = dataset_manifest["rows"]
    model_uri = f"models:/{models[0]['name']}/{models[0]['version']}"
    if dataset_manifest["model_uri"] != model_uri:
        raise RuntimeError("Dataset was generated for another model version")
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    source_log = (directory / "source.log").open("w", encoding="utf-8")
    command = [
        sys.executable,
        "-m",
        "hexrl_platform.task_pipeline.batch_source",
        "--file",
        str(dataset.resolve()),
        "--batch-id",
        batch_id,
        "--metrics-csv",
        str((directory / "source.csv").resolve()),
        "--queue-prefix",
        prefix,
    ]
    if label == "A":
        command += ["--unbounded", "--no-backpressure"]
    source = await asyncio.create_subprocess_exec(
        *command,
        cwd=ROOT,
        env=runtime.environment,
        stdout=source_log,
        stderr=asyncio.subprocess.STDOUT,
    )
    stop = asyncio.Event()
    collector = asyncio.create_task(
        collect(runtime, prefix, batch_id, directory / "metrics.csv", stop)
    )
    events = []

    async def failures() -> None:
        if label != "D":
            return
        await asyncio.sleep(5)
        await runtime.run("stop", "--timeout", "40", "worker")
        connection = await database(runtime)
        try:
            processing = await connection.fetchval(
                "SELECT count(*) FROM inference_tasks WHERE batch_id=$1 AND status='PROCESSING'",
                batch_id,
            )
        finally:
            await connection.close()
        events.append(
            {
                "event": "SIGTERM",
                "elapsed_s": time.monotonic() - started,
                "processing_after_stop": processing,
            }
        )
        if processing:
            raise AssertionError("Graceful shutdown left PROCESSING tasks")
        await runtime.run("start", "worker")
        await wait_healthy(runtime, workers)
        await asyncio.sleep(5)
        await runtime.run("kill", "-s", "SIGKILL", "worker")
        events.append({"event": "SIGKILL", "elapsed_s": time.monotonic() - started})
        await runtime.run("up", "-d", "--no-deps", "worker")
        await wait_healthy(runtime, workers)

    disruptions = asyncio.create_task(failures())
    connection = await database(runtime)
    try:
        async with asyncio.timeout(args.timeout):
            while True:
                if source.returncode not in (None, 0):
                    raise RuntimeError("Source failed; see source.log")
                if collector.done():
                    await collector
                if disruptions.done():
                    await disruptions
                terminal = await connection.fetchval(
                    "SELECT count(*) FROM inference_tasks WHERE batch_id=$1 "
                    "AND status IN ('COMPLETED','FAILED')",
                    batch_id,
                )
                if terminal == expected and source.returncode is not None:
                    if source.returncode:
                        raise RuntimeError("Source failed; see source.log")
                    break
                await asyncio.sleep(0.5)
            await disruptions
            await asyncio.sleep(2)
    finally:
        stop.set()
        disruptions.cancel()
        await asyncio.gather(disruptions, return_exceptions=True)
        if source.returncode is None:
            source.terminate()
        await source.wait()
        source_log.close()
        await connection.close()
        await collector
    rows = await export_tasks(runtime, batch_id, directory / "tasks.csv")
    ids = {str(row["task_id"]) for row in rows}
    expected_ids = {str(task_id_for(batch_id, line)) for line in range(1, expected + 1)}
    if ids != expected_ids:
        raise AssertionError("Task IDs do not reconcile with source lines")
    dlq_unique_tasks = await reconcile_dlq(runtime, prefix, rows, directory)
    manifest = {
        "scenario": label,
        "batch_id": batch_id,
        "queue_prefix": prefix,
        "started_at": started_at,
        "duration_s": time.monotonic() - started,
        "workers": workers,
        "concurrency": settings.worker_concurrency,
        "prefetch": settings.prefetch_count,
        "inference_delay_s": settings.inference_delay_s,
        "queue_limit": settings.queue_max_length,
        "publisher_confirms": label != "A",
        "dataset": dataset_manifest,
        "model": models[0],
        "completed": sum(row["status"] == "COMPLETED" for row in rows),
        "failed": sum(row["status"] == "FAILED" for row in rows),
        "reconciled_unique_tasks": len(ids),
        "dlq_unique_tasks": dlq_unique_tasks,
        "code": provenance,
        "events": events,
        "injected_transient_failures_per_process": 2 if label == "D" else 0,
    }
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "worker.log").write_text(
        await runtime.run("logs", "--no-color", "worker"), encoding="utf-8"
    )
    return manifest


async def main() -> None:
    parser = argparse.ArgumentParser(description="Measure finite backpressure workloads A-D")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--project", default="hexrl-hw4")
    parser.add_argument("--dataset", type=Path, default=Path("data/tasks/valid.jsonl"))
    parser.add_argument("--failure-dataset", type=Path, default=Path("data/tasks/failure.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenarios", nargs="+", choices=list("ABCD"), default=list("ABCD"))
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    for label in args.scenarios:
        print(f"Starting scenario {label}", flush=True)
        result = await run_scenario(args, label)
        print(
            json.dumps(
                {key: result[key] for key in ("scenario", "duration_s", "completed", "failed")}
            ),
            flush=True,
        )


if __name__ == "__main__":
    asyncio.run(main())
