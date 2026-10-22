import asyncio
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from hexrl_platform.task_pipeline.config import PipelineSettings

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Runtime:
    env_file: Path
    project: str
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

    @property
    def command(self) -> list[str]:
        return [
            "docker",
            "compose",
            "-p",
            self.project,
            "--env-file",
            str(self.env_file.resolve()),
            "-f",
            str(ROOT / "docker-compose.yaml"),
            "-f",
            str(ROOT / "scripts/backpressure/compose.benchmark.yaml"),
        ]

    @property
    def pipeline_settings(self) -> PipelineSettings:
        values: dict[str, Any] = {
            name: self.environment[name.upper()]
            for name in PipelineSettings.model_fields
            if name.upper() in self.environment
        }
        if values.get("queue_max_length") == "null":
            values["queue_max_length"] = None
        return PipelineSettings(_env_file=None, **values)

    async def run(self, *args: str) -> str:
        process = await asyncio.create_subprocess_exec(
            *self.command,
            *args,
            cwd=ROOT,
            env=self.environment,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        if process.returncode:
            raise RuntimeError(
                f"Compose {args[0]} failed: {stderr.decode(errors='replace')[-3000:]}"
            )
        return stdout.decode(errors="replace").strip()


async def container_rss(container_id: str) -> int:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "exec",
        container_id,
        "python",
        "-c",
        "import psutil; print(psutil.Process(1).memory_info().rss)",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await process.communicate()
    return int(stdout.strip()) if process.returncode == 0 else 0
