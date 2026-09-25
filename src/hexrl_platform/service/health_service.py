import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from hexrl_platform.schemas.health import ComponentHealth, HealthResponse

logger = logging.getLogger(__name__)


async def _postgres_version(session: AsyncSession) -> str:
    result = await session.execute(text("SELECT version()"))
    return str(result.scalar_one())


async def _check(
    name: str,
    probe: Callable[[], Awaitable[str]],
    timeout: float,
) -> ComponentHealth:
    started = time.perf_counter()
    try:
        component_version = await asyncio.wait_for(probe(), timeout=timeout)
    except Exception as exc:  # noqa: BLE001 - health-check обязан пережить любую ошибку
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.warning("component check failed: %s (%s)", name, type(exc).__name__)
        return ComponentHealth(
            name=name,
            status="down",
            latency_ms=round(elapsed_ms, 2),
            error=type(exc).__name__,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ComponentHealth(
        name=name,
        status="up",
        version=component_version,
        latency_ms=round(elapsed_ms, 2),
    )


async def collect_health(session: AsyncSession, timeout: float) -> HealthResponse:
    probes: dict[str, Callable[[], Awaitable[str]]] = {
        "postgres": lambda: _postgres_version(session),
    }
    components = await asyncio.gather(
        *(_check(name, probe, timeout) for name, probe in probes.items())
    )
    overall = "ok" if all(c.status == "up" for c in components) else "degraded"
    return HealthResponse(status=overall, components=list(components))
