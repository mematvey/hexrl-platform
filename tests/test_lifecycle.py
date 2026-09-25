from asgi_lifespan import LifespanManager

from hexrl_platform import main as main_module


async def test_lifespan_sets_state_and_disposes(app):
    async with LifespanManager(app):
        assert app.state.engine is not None
        assert app.state.session_factory is not None


def test_main_runs_uvicorn(monkeypatch):
    calls = {}
    monkeypatch.setattr(main_module.uvicorn, "run", lambda *a, **k: calls.update(k))
    main_module.main()
    assert calls["factory"] is True
