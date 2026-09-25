import asyncio

from hexrl_platform.core.config import APP_VERSION


async def test_healthz(client):
    resp = await client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_version(client):
    resp = await client.get("/api/v1/version")
    assert resp.status_code == 200
    body = resp.json()
    assert body["version"] == APP_VERSION
    assert body["name"]


async def test_health_ok(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    (component,) = body["components"]
    assert component["name"] == "postgres"
    assert component["status"] == "up"
    assert component["version"] == "PostgreSQL 16.0"
    assert component["latency_ms"] >= 0


async def test_health_db_down(client, session):
    session.error = ConnectionRefusedError("db is down")
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 503
    body = resp.json()
    assert body["status"] == "degraded"
    component = body["components"][0]
    assert component["status"] == "down"
    assert component["version"] is None
    assert component["error"] == "ConnectionRefusedError"


async def test_health_db_timeout(client, session, monkeypatch):
    from hexrl_platform.core.config import get_settings

    monkeypatch.setattr(get_settings(), "health_timeout_seconds", 0.05)
    session.error = None

    async def slow(*_a, **_k):
        await asyncio.sleep(1)

    session.execute = slow
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 503
    assert resp.json()["components"][0]["error"] == "TimeoutError"


async def test_unknown_route_404(client):
    resp = await client.get("/api/v1/nope")
    assert resp.status_code == 404


async def test_healthz_not_under_api_prefix(client):
    assert (await client.get("/api/v1/healthz")).status_code == 404
