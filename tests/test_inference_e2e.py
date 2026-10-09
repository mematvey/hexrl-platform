import mlflow
import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from mlflow.tracking import MlflowClient

from hexrl_platform.inference.config import InferenceSettings
from hexrl_platform.inference.main import create_app
from hexrl_platform.inference.model_loader import load_champion
from hexrl_platform.rl.experiment import run_study
from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.rl.registry import RegistryConfig, register_study
from hexrl_platform.rl.replay import ReplayGenerationConfig, generate_replays
from hexrl_platform.rl.study import StudyConfig
from hexrl_platform.rl.training import DQNConfig

MODEL_NAME = "nav-e2e"
TINY_DQN = DQNConfig(
    total_timesteps=200,
    learning_starts=50,
    buffer_size=500,
    batch_size=16,
    target_update_interval=50,
    eval_freq=100,
    net_arch=(16,),
)


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    """Real trained policies registered in a local sqlite Model Registry."""
    tmp_path = tmp_path_factory.mktemp("inference")
    previous_uri = mlflow.get_tracking_uri()
    with pytest.MonkeyPatch.context() as monkeypatch:
        # With a sqlite backend MLflow stores artifacts in ./mlruns of the working directory
        monkeypatch.chdir(tmp_path)
        replay_path = tmp_path / "replays.jsonl"
        generate_replays(replay_path, ReplayGenerationConfig(episodes=300, seed=5, max_steps=12))
        study = StudyConfig(
            name="inference-e2e",
            experiment="inference-e2e",
            replays=replay_path,
            seeds=(1,),
            variants=(("dqn-a", TINY_DQN), ("dqn-b", TINY_DQN)),
        )
        tracking_uri = f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
        try:
            run_study(study, tracking_uri=tracking_uri, output_dir=tmp_path / "out")
            result = register_study(
                study, RegistryConfig(model_name=MODEL_NAME), tracking_uri=tracking_uri
            )
            yield tracking_uri, result
        finally:
            mlflow.set_tracking_uri(previous_uri)


def make_app(tracking_uri: str) -> FastAPI:
    settings = InferenceSettings(mlflow_tracking_uri=tracking_uri, model_name=MODEL_NAME)
    return create_app(loader=lambda _: load_champion(settings))


def client_for(app: FastAPI) -> AsyncClient:
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def test_service_serves_the_registered_champion(registry):
    tracking_uri, result = registry
    app = make_app(tracking_uri)
    async with LifespanManager(app), client_for(app) as client:
        model = (await client.get("/model")).json()
        assert model["name"] == MODEL_NAME
        assert model["alias"] == "champion"
        assert model["version"] == result.champion_version
        assert model["run_id"] == result.champion.run_id
        assert model["max_steps"] == 12

        response = await client.post(
            "/process", json={"q": 2, "r": 0, "goal_q": 0, "goal_r": 0, "steps_remaining": 5}
        )
        assert response.status_code == 200
        body = response.json()
        direction = DIRECTIONS[body["action"]]
        assert body["direction"] == {"dq": direction.q, "dr": direction.r}
        assert body["model"]["version"] == result.champion_version


@pytest.mark.parametrize(
    "payload",
    [
        # (0, 1) is a wall on the default map
        {"q": 0, "r": 1, "goal_q": 2, "goal_r": 0, "steps_remaining": 5},
        {"q": 9, "r": 9, "goal_q": 0, "goal_r": 0, "steps_remaining": 5},
        # Above max_steps of the loaded model, which the request schema cannot know
        {"q": 2, "r": 0, "goal_q": 0, "goal_r": 0, "steps_remaining": 13},
    ],
)
async def test_states_outside_the_model_map_are_rejected(registry, payload):
    tracking_uri, _ = registry
    app = make_app(tracking_uri)
    async with LifespanManager(app), client_for(app) as client:
        assert (await client.post("/process", json=payload)).status_code == 422


async def test_alias_switch_reaches_only_a_restarted_service(registry):
    tracking_uri, result = registry
    registry_client = MlflowClient(tracking_uri=tracking_uri)
    champion = result.champion_version
    (other,) = set(result.versions.values()) - {champion}

    running = make_app(tracking_uri)
    async with LifespanManager(running), client_for(running) as client:
        registry_client.set_registered_model_alias(MODEL_NAME, "champion", other)
        try:
            # The model is loaded once at startup, so a running process keeps the old version
            assert (await client.get("/model")).json()["version"] == champion

            restarted = make_app(tracking_uri)
            async with LifespanManager(restarted), client_for(restarted) as fresh:
                assert (await fresh.get("/model")).json()["version"] == other
        finally:
            registry_client.set_registered_model_alias(MODEL_NAME, "champion", champion)
