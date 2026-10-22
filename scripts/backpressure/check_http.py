import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

from scripts.backpressure.collect_metrics import append_csv, database
from scripts.backpressure.run_experiments import wait_healthy
from scripts.backpressure.runtime import Runtime


async def check(args: argparse.Namespace) -> dict:
    args.output.mkdir(parents=True, exist_ok=False)
    runtime = Runtime(
        args.env_file,
        args.project,
        {
            "QUEUE_PREFIX": f"hexrl.http-check.{uuid4().hex}",
            "QUEUE_MAX_LENGTH": "20",
            "BENCHMARK_TRANSIENT_FAILURES": "0",
        },
    )
    await runtime.run("stop", "worker")
    await runtime.run("up", "-d", "--no-deps", "--force-recreate", "app")
    port = runtime.environment.get("APP_PORT", "8000")
    accepted = []
    rejected_ms = []
    payload = {"q": 0, "r": 0, "goal_q": 2, "goal_r": -1, "steps_remaining": 20}
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=5) as client:
        async with asyncio.timeout(60):
            while True:
                try:
                    response = await client.get("/healthz")
                    if response.status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.2)
        for index in range(25):
            started = time.perf_counter()
            response = await client.post("/api/v1/tasks", json=payload)
            elapsed_ms = (time.perf_counter() - started) * 1000
            append_csv(
                args.output / "requests.csv",
                {
                    "request": index,
                    "status": response.status_code,
                    "elapsed_ms": elapsed_ms,
                    "retry_after": response.headers.get("Retry-After"),
                },
            )
            if index < 20:
                assert response.status_code == 202, response.text
                accepted.append(response.json()["status_url"])
            else:
                assert response.status_code == 503, response.text
                assert response.headers["Retry-After"] == "1"
                assert elapsed_ms < 1000
                rejected_ms.append(elapsed_ms)
        connection = await database(runtime)
        try:
            pending = await connection.fetchval(
                "SELECT count(*) FROM inference_tasks WHERE status='PENDING'"
            )
        finally:
            await connection.close()
        assert pending == 0
        await runtime.run(
            "up", "-d", "--no-deps", "--force-recreate", "--scale", "worker=1", "worker"
        )
        models = await wait_healthy(runtime, 1)
        details = []
        async with asyncio.timeout(60):
            for location in accepted:
                while True:
                    detail = (await client.get(location)).json()
                    if detail["status"] in ("COMPLETED", "FAILED"):
                        assert detail["status"] == "COMPLETED", detail
                        assert detail["result"]["model_version"] == models[0]["version"]
                        details.append(detail)
                        break
                    await asyncio.sleep(0.1)
        invalid = await client.post("/api/v1/tasks", json={**payload, "q": 999})
        assert invalid.status_code == 202
        async with asyncio.timeout(10):
            while True:
                failed = (await client.get(invalid.json()["status_url"])).json()
                if failed["status"] == "FAILED":
                    assert failed["last_error"]
                    break
                await asyncio.sleep(0.1)
    summary = {
        "accepted": len(accepted),
        "rejected": len(rejected_ms),
        "max_503_ms": max(rejected_ms),
        "pending_after_rejections": pending,
        "completed_after_restart": len(details),
        "model": models[0],
        "domain_invalid": failed,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.output / "completed.json").write_text(
        json.dumps(details, indent=2) + "\n", encoding="utf-8"
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure real HTTP overload and verify recovery")
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--project", default="hexrl-hw4")
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(asyncio.run(check(parser.parse_args())), indent=2))
