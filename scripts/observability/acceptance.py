import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

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
DATASOURCES = ("metrics", "logs", "traces")
ALERT_RULES = ("queue_near_limit", "dlq_not_empty")
# Спаны, без которых не видно вклада очереди, inference и БД
TASK_SPANS = ("db.create_task", "inference", "db.save_result")
SIGNAL_TIMEOUT_S = 90


async def prepare(stand: Stand, report: Report, *, clean: bool, timeout_s: float) -> None:
    if clean:
        await stand.remove_volumes()
    started = time.monotonic()
    await stand.run("build")
    await stand.run("up", "-d", "--scale", "worker=1")
    await stand.wait_ready(timeout_s=timeout_s)
    report.facts["ready_after_s"] = round(time.monotonic() - started, 1)
    bootstrap = [item for item in await stand.logs("model-bootstrap") if "event" in item]
    ready = next((item for item in bootstrap if item["event"] == "bootstrap_model_ready"), None)
    report.check("bootstrap prepared the champion", ready is not None)
    if ready is not None:
        report.facts["model"] = {
            "name": ready.get("component"),
            "alias": ready.get("model_alias"),
            "version": ready.get("model_version"),
            "trained_now": ready.get("reason") == "trained",
        }
    worker_events = [item.get("event") for item in await stand.logs("worker")]
    report.check(
        "worker loaded the model before becoming healthy",
        "model_loaded" in worker_events,
    )


async def check_grafana(stand: Stand, report: Report) -> None:
    grafana = stand.grafana()
    report.check("grafana is up", await eventually(grafana.healthy, timeout_s=60))
    for uid in DATASOURCES:
        status = await eventually(lambda uid=uid: _ok(grafana, uid), timeout_s=60)
        report.check(f"datasource {uid} is provisioned and healthy", status == "OK", status=status)
    dashboard = await grafana.get("/api/dashboards/uid/pipeline")
    report.check(
        "pipeline dashboard is provisioned",
        dashboard["dashboard"]["uid"] == "pipeline",
        panels=len(dashboard["dashboard"].get("panels", [])),
    )
    rules = await grafana.alert_rules()
    report.check(
        "alert rules are provisioned",
        all(uid in rules for uid in ALERT_RULES),
        rules=sorted(rules),
    )


async def _ok(grafana: Grafana, uid: str) -> str:
    status = await grafana.datasource_health(uid)
    return status if status == "OK" else ""


async def check_task(stand: Stand, report: Report) -> None:
    started = time.time()
    async with stand.api() as api:
        response = await api.post("/api/v1/tasks", json=PAYLOAD)
        report.check("task accepted", response.status_code == 202, status=response.status_code)
        task_id = response.json()["task_id"]

        async def finished() -> dict | None:
            body = (await api.get(f"/api/v1/tasks/{task_id}")).json()
            return body if body["status"] in ("COMPLETED", "FAILED") else None

        result = await eventually(finished, timeout_s=60, interval_s=0.5)
    report.facts["task_id"] = task_id
    report.check(
        "task result is available",
        result is not None and result["status"] == "COMPLETED",
        result=result,
    )

    grafana = stand.grafana()
    query = f'{{service=~"hexrl-api|hexrl-worker"}} |= "{task_id}"'

    async def logs() -> list[dict]:
        records = await grafana.loki(query, started - 60, time.time() + 5)
        records = task_path(records)
        services = {record.get("service") for record in records}
        return records if {"hexrl-api", "hexrl-worker"} <= services else []

    records = await eventually(logs, timeout_s=SIGNAL_TIMEOUT_S)
    events = sorted({str(record.get("event")) for record in records})
    trace_ids = {record["trace_id"] for record in records if record.get("trace_id")}
    report.check("logs of both services are found by task_id", bool(records), events=events)
    report.check(
        "all task logs share one trace_id", len(trace_ids) == 1, trace_ids=sorted(trace_ids)
    )
    if len(trace_ids) != 1:
        return
    trace_id = trace_ids.pop()
    report.facts["trace_id"] = trace_id

    async def trace() -> dict | None:
        summary = await grafana.trace(trace_id)
        return summary if summary and set(TASK_SPANS) <= set(summary["spans"]) else None

    summary = await eventually(trace, timeout_s=SIGNAL_TIMEOUT_S)
    report.check(
        "trace links API, broker, inference and DB",
        summary is not None and {"hexrl-api", "hexrl-worker"} <= set(summary["services"]),
        trace=summary,
    )
    back = await grafana.loki(
        f'{{service=~".+"}} | json | trace_id="{trace_id}"', started - 60, time.time() + 5
    )
    report.check("trace_id leads back to the same logs", len(back) >= len(records), lines=len(back))

    async def metrics() -> dict | None:
        values = {
            "completed": await grafana.prom_value("sum(hexrl_worker_completed_total)"),
            "queue_limit": await grafana.prom_value("max(hexrl_queue_limit_ratio)"),
            "ready": await grafana.prom_value(
                'sum(rabbitmq_detailed_queue_messages_ready{queue=~".*[.]tasks"})'
            ),
            "workers": await grafana.prom_value(
                'count(hexrl_process_running_ratio{job="hexrl-worker"} == 1)'
            ),
        }
        return values if all(value is not None for value in values.values()) else None

    values = await eventually(metrics, timeout_s=SIGNAL_TIMEOUT_S)
    report.check(
        "metrics of app, worker and broker reach Prometheus",
        values is not None and values["completed"] >= 1,
        values=values,
    )


async def commit() -> str:
    process = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "--short", "HEAD", cwd=ROOT, stdout=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    return stdout.decode().strip()


async def accept(args: argparse.Namespace) -> Report:
    stand = Stand(args.env_file, args.project)
    report = Report("acceptance")
    report.facts["started_at"] = datetime.now(UTC).isoformat()
    report.facts["commit"] = await commit()
    report.facts["clean"] = args.clean
    try:
        await prepare(stand, report, clean=args.clean, timeout_s=args.timeout)
        await check_grafana(stand, report)
        await check_task(stand, report)
    except Exception as error:  # noqa: BLE001 - отчёт должен сохраниться при любом сбое
        report.check("acceptance finished without errors", False, error=repr(error)[:500])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean start and acceptance of the stand")
    parser.add_argument("--env-file", type=Path, default=Path(".env.obs"))
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--output", type=Path, default=Path("metrics/observability"))
    parser.add_argument(
        "--clean", action="store_true", help="remove volumes of this project before start"
    )
    parser.add_argument("--timeout", type=float, default=1200, help="seconds to become ready")
    args = parser.parse_args()
    report = asyncio.run(accept(args))
    print(f"report: {report.save(args.output)}")
    sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
