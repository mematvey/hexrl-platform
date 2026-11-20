import asyncio
import json
import os
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT = "hexrl-obs"
# Без этих сервисов демонстрация неполна; одноразовые проверяются по коду выхода
REQUIRED_HEALTHY = (
    "db",
    "rabbitmq",
    "mlflow",
    "app",
    "worker",
    "grafana",
    "prometheus",
    "loki",
    "tempo",
)
REQUIRED_RUNNING = ("alloy", "otel-collector")
ONE_SHOT = ("artifact-bucket", "model-bootstrap")


def parse_compose_ps(output: str) -> list[dict[str, Any]]:
    """Compose печатает либо JSON-массив, либо по объекту на строку - зависит от версии."""
    output = output.strip()
    if not output:
        return []
    if output.startswith("["):
        return json.loads(output)
    return [json.loads(line) for line in output.splitlines() if line.strip()]


def readiness(
    containers: Iterable[dict[str, Any]], workers: int = 1
) -> tuple[bool, list[str], list[str]]:
    """Возвращает (готов, что ещё не готово, что сломано окончательно)."""
    by_service: dict[str, list[dict[str, Any]]] = {}
    for item in containers:
        by_service.setdefault(item["Service"], []).append(item)
    waiting, failed = [], []
    for name in ONE_SHOT:
        items = by_service.get(name, [])
        if any(item["State"] == "exited" and item.get("ExitCode", 0) != 0 for item in items):
            failed.append(f"{name} exited with an error")
        elif not any(item["State"] == "exited" for item in items):
            waiting.append(f"{name} has not finished")
    for name in REQUIRED_HEALTHY:
        items = [item for item in by_service.get(name, []) if item["State"] == "running"]
        expected = workers if name == "worker" else 1
        if sum(item.get("Health") == "healthy" for item in items) < expected:
            waiting.append(f"{name} is not healthy")
    for name in REQUIRED_RUNNING:
        if not any(item["State"] == "running" for item in by_service.get(name, [])):
            waiting.append(f"{name} is not running")
    return not waiting and not failed, waiting, failed


async def eventually[T](
    probe: Callable[[], Awaitable[T]], *, timeout_s: float, interval_s: float = 2.0
) -> T:
    """Повторяет probe, пока он не вернёт истинное значение; иначе отдаёт последнее."""
    deadline = time.monotonic() + timeout_s
    while True:
        value = await probe()
        if value or time.monotonic() >= deadline:
            return value
        await asyncio.sleep(interval_s)


@dataclass
class Stand:
    env_file: Path
    project: str = DEFAULT_PROJECT
    # benchmark-override подменяет воркер на fault_worker с искусственными сбоями
    faults: bool = False
    overrides: dict[str, str] = field(default_factory=dict)

    @property
    def environment(self) -> dict[str, str]:
        values = {
            key: value for key, value in dotenv_values(self.env_file).items() if value is not None
        }
        return {
            **os.environ,
            **values,
            **self.overrides,
            "HEXRL_ENV_FILE": str(self.env_file.resolve()),
        }

    def command(self, *args: str) -> list[str]:
        files = ["-f", str(ROOT / "docker-compose.yaml")]
        if self.faults:
            files += ["-f", str(ROOT / "scripts/backpressure/compose.benchmark.yaml")]
        return [
            "docker",
            "compose",
            "-p",
            self.project,
            "--env-file",
            str(self.env_file.resolve()),
            *files,
            "--profile",
            "observability",
            *self._batch_profile(args),
            *args,
        ]

    @staticmethod
    def _batch_profile(args: tuple[str, ...]) -> list[str]:
        # up без явного batch-source не должен запускать источник с файлом по умолчанию
        if args and args[0] == "up" and "batch-source" not in args:
            return []
        return ["--profile", "batch"]

    async def run(self, *args: str, check: bool = True) -> str:
        process = await asyncio.create_subprocess_exec(
            *self.command(*args),
            cwd=ROOT,
            env=self.environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if check and process.returncode:
            raise RuntimeError(
                f"compose {' '.join(args[:2])} failed: {stderr.decode(errors='replace')[-2000:]}"
            )
        return stdout.decode(errors="replace").strip()

    async def remove_volumes(self) -> None:
        # Очистка разрешена только изолированному стенду: тома MLflow других проектов не трогаем
        if not self.project.startswith(DEFAULT_PROJECT):
            raise ValueError(f"refusing to delete volumes of project {self.project!r}")
        await self.run("down", "--volumes", "--remove-orphans")

    async def containers(self) -> list[dict[str, Any]]:
        return parse_compose_ps(await self.run("ps", "--all", "--format", "json"))

    async def wait_ready(self, *, timeout_s: float, workers: int = 1) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            ready, waiting, failed = readiness(await self.containers(), workers)
            if failed:
                raise RuntimeError("; ".join(failed))
            if ready:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError("stand is not ready: " + "; ".join(waiting))
            await asyncio.sleep(3)

    async def sql(self, query: str) -> list[list[str]]:
        env = self.environment
        output = await self.run(
            "exec",
            "-T",
            "db",
            "psql",
            "-U",
            env["POSTGRES_USER"],
            "-d",
            env["POSTGRES_DB"],
            "-tA",
            "-F",
            "|",
            "-c",
            query,
        )
        return [line.split("|") for line in output.splitlines() if line]

    async def statuses(self, where: str) -> dict[str, int]:
        rows = await self.sql(
            f"SELECT status, count(*) FROM inference_tasks WHERE {where} GROUP BY 1"
        )
        return {status: int(count) for status, count in rows}

    async def logs(self, service: str) -> list[dict[str, Any]]:
        output = await self.run("logs", "--no-color", "--no-log-prefix", service)
        return parse_json_lines(output.splitlines())

    def rabbit(self) -> httpx.AsyncClient:
        env = self.environment
        return httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{env.get('RABBITMQ_MANAGEMENT_PORT', '15672')}",
            auth=(env["RABBITMQ_USER"], env["RABBITMQ_PASSWORD"]),
            timeout=10,
        )

    async def queue(self, suffix: str) -> dict[str, Any]:
        async with self.rabbit() as client:
            prefix = self.environment.get("QUEUE_PREFIX", "hexrl.inference")
            response = await client.get(f"/api/queues/%2F/{prefix}.{suffix}")
            response.raise_for_status()
            return response.json()

    async def peek(self, suffix: str, count: int = 100) -> list[dict[str, Any]]:
        """Читает сообщения и возвращает их обратно в очередь."""
        async with self.rabbit() as client:
            prefix = self.environment.get("QUEUE_PREFIX", "hexrl.inference")
            response = await client.post(
                f"/api/queues/%2F/{prefix}.{suffix}/get",
                json={"count": count, "ackmode": "ack_requeue_true", "encoding": "auto"},
            )
            response.raise_for_status()
            return response.json()

    async def memory(self) -> dict[str, float]:
        """Фактическое потребление памяти контейнерами стенда в MiB."""
        names = {
            item["Name"]: item["Service"]
            for item in await self.containers()
            if item["State"] == "running"
        }
        process = await asyncio.create_subprocess_exec(
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{json .}}",
            *names,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await process.communicate()
        usage: dict[str, float] = {}
        for row in parse_json_lines(stdout.decode().splitlines()):
            service = names.get(row["Name"], row["Name"])
            usage[service] = usage.get(service, 0.0) + mebibytes(row["MemUsage"].split("/")[0])
        return usage

    def api(self) -> httpx.AsyncClient:
        port = self.environment.get("APP_PORT", "8000")
        return httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=10)

    def grafana(self) -> "Grafana":
        env = self.environment
        return Grafana(
            f"http://127.0.0.1:{env.get('GRAFANA_PORT', '3000')}",
            (env.get("GRAFANA_USER", "admin"), env.get("GRAFANA_PASSWORD", "admin")),
        )


def parse_json_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


UNITS = {
    "B": 1 / 2**20,
    "KiB": 1 / 2**10,
    "MiB": 1,
    "GiB": 2**10,
    "kB": 1e3 / 2**20,
    "MB": 1e6 / 2**20,
    "GB": 1e9 / 2**20,
}


def mebibytes(text: str) -> float:
    text = text.strip()
    for unit in sorted(UNITS, key=len, reverse=True):
        if text.endswith(unit):
            return float(text[: -len(unit)]) * UNITS[unit]
    raise ValueError(f"unknown memory unit in {text!r}")


def task_path(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Логи пути задачи: опросы GET /tasks/{id} - отдельные запросы со своими трейсами."""
    return [r for r in records if r.get("method") != "GET" and r.get("trace_id")]


def trace_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """Tempo отдаёт OTLP JSON: v1 - batches, v2 - trace.resourceSpans."""
    batches = payload.get("batches") or payload.get("trace", {}).get("resourceSpans") or []
    services, instances, names, errors = set(), set(), set(), set()
    for batch in batches:
        attributes = {
            item["key"]: next(iter(item.get("value", {}).values()), None)
            for item in batch.get("resource", {}).get("attributes", [])
        }
        services.add(attributes.get("service.name"))
        instances.add(attributes.get("service.instance.id"))
        for scope in batch.get("scopeSpans") or batch.get("instrumentationLibrarySpans") or []:
            for span in scope.get("spans", []):
                names.add(span.get("name"))
                if span.get("status", {}).get("code") in ("STATUS_CODE_ERROR", 2):
                    errors.add(span.get("name"))
    return {
        "services": sorted(item for item in services if item),
        "instances": sorted(item for item in instances if item),
        "spans": sorted(item for item in names if item),
        "error_spans": sorted(item for item in errors if item),
    }


@dataclass
class Grafana:
    base_url: str
    auth: tuple[str, str]

    async def _get(self, path: str, params: dict[str, str | float]) -> httpx.Response:
        async with httpx.AsyncClient(base_url=self.base_url, auth=self.auth, timeout=15) as client:
            response = await client.get(path, params=params, headers={"Accept": "application/json"})
            response.raise_for_status()
            return response

    async def get(self, path: str, **params: str | float) -> dict[str, Any]:
        return (await self._get(path, params)).json()

    async def get_list(self, path: str) -> list[dict[str, Any]]:
        return (await self._get(path, {})).json()

    async def healthy(self) -> bool:
        try:
            return (await self.get("/api/health")).get("database") == "ok"
        except httpx.HTTPError:
            return False

    async def datasource_health(self, uid: str) -> str:
        try:
            return (await self.get(f"/api/datasources/uid/{uid}/health")).get("status", "")
        except httpx.HTTPStatusError as error:
            if error.response.status_code != 404:
                return f"HTTP {error.response.status_code}"
        # Плагин Tempo не реализует health API: проверяем сам backend через прокси Grafana
        try:
            await self._get(f"/api/datasources/proxy/uid/{uid}/api/echo", {})
        except httpx.HTTPError as error:
            return type(error).__name__
        return "OK"

    async def alert_rules(self) -> dict[str, str]:
        return {
            rule["uid"]: rule["title"]
            for rule in await self.get_list("/api/v1/provisioning/alert-rules")
        }

    async def alert_states(self) -> dict[str, str]:
        data = await self.get("/api/prometheus/grafana/api/v1/rules")
        return {
            rule["name"]: rule.get("state", "")
            for group in data["data"]["groups"]
            for rule in group["rules"]
        }

    async def prom(self, expr: str, at: float | None = None) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"query": expr}
        if at is not None:
            params["time"] = at
        data = await self.get("/api/datasources/proxy/uid/metrics/api/v1/query", **params)
        return data["data"]["result"]

    async def prom_value(self, expr: str, at: float | None = None) -> float | None:
        result = await self.prom(expr, at)
        return float(result[0]["value"][1]) if result else None

    async def prom_range(
        self, expr: str, start: float, end: float, step: float = 5
    ) -> list[tuple[float, float]]:
        data = await self.get(
            "/api/datasources/proxy/uid/metrics/api/v1/query_range",
            query=expr,
            start=start,
            end=end,
            step=step,
        )
        result = data["data"]["result"]
        return [(float(ts), float(value)) for ts, value in result[0]["values"]] if result else []

    async def loki(self, query: str, start: float, end: float, limit: int = 1000) -> list[dict]:
        data = await self.get(
            "/api/datasources/proxy/uid/logs/loki/api/v1/query_range",
            query=query,
            start=int(start * 1e9),
            end=int(end * 1e9),
            limit=limit,
            direction="forward",
        )
        return parse_json_lines(
            line for stream in data["data"]["result"] for _, line in stream["values"]
        )

    async def trace(self, trace_id: str) -> dict[str, Any] | None:
        try:
            return trace_summary(
                await self.get(f"/api/datasources/proxy/uid/traces/api/traces/{trace_id}")
            )
        except httpx.HTTPStatusError:
            return None

    async def search_traces(self, traceql: str, start: float, end: float, limit: int = 20) -> list:
        data = await self.get(
            "/api/datasources/proxy/uid/traces/api/search",
            q=traceql,
            start=int(start),
            end=int(end) + 1,
            limit=limit,
        )
        return data.get("traces", [])


@dataclass
class Report:
    name: str
    checks: list[dict[str, Any]] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, ok: bool, **detail: object) -> bool:
        self.checks.append({"check": name, "ok": bool(ok), **detail})
        print(f"[{'ok' if ok else 'FAIL'}] {self.name}: {name}", flush=True)
        return bool(ok)

    @property
    def ok(self) -> bool:
        return bool(self.checks) and all(item["ok"] for item in self.checks)

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        payload = {"scenario": self.name, "ok": self.ok, "facts": self.facts, "checks": self.checks}
        path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
        return path
