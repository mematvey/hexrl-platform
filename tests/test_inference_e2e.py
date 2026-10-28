import mlflow
import pytest
from mlflow.tracking import MlflowClient

from hexrl_platform.inference.config import InferenceSettings
from hexrl_platform.inference.model_loader import LoadedModel, load_champion
from hexrl_platform.inference.predict import InvalidStateError, predict_action
from hexrl_platform.rl.experiment import run_study
from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.rl.registry import RegistryConfig, register_study
from hexrl_platform.rl.replay import ReplayGenerationConfig, generate_replays
from hexrl_platform.rl.study import StudyConfig
from hexrl_platform.rl.training import DQNConfig
from hexrl_platform.task_pipeline.schemas import HexState

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


def load(tracking_uri: str) -> LoadedModel:
    return load_champion(InferenceSettings(mlflow_tracking_uri=tracking_uri, model_name=MODEL_NAME))


def test_champion_is_loaded_by_alias(registry):
    tracking_uri, result = registry
    loaded = load(tracking_uri)

    assert loaded.name == MODEL_NAME
    assert loaded.alias == "champion"
    assert loaded.version == result.champion_version
    assert loaded.run_id == result.champion.run_id
    assert loaded.max_steps == 12


def test_loaded_model_predicts_a_valid_direction(registry):
    tracking_uri, _ = registry
    loaded = load(tracking_uri)

    action = predict_action(loaded.model, HexState(q=2, r=0, goal_q=0, goal_r=0, steps_remaining=5))

    assert 0 <= action < len(DIRECTIONS)


@pytest.mark.parametrize(
    "state",
    [
        # (0, 1) is a wall on the default map
        HexState(q=0, r=1, goal_q=2, goal_r=0, steps_remaining=5),
        HexState(q=9, r=9, goal_q=0, goal_r=0, steps_remaining=5),
        # Above max_steps of the loaded model, which the message schema cannot know
        HexState(q=2, r=0, goal_q=0, goal_r=0, steps_remaining=13),
    ],
)
def test_states_outside_the_model_map_are_rejected(registry, state):
    tracking_uri, _ = registry
    loaded = load(tracking_uri)

    with pytest.raises(InvalidStateError):
        predict_action(loaded.model, state)


def test_alias_switch_reaches_only_a_reloaded_model(registry):
    tracking_uri, result = registry
    registry_client = MlflowClient(tracking_uri=tracking_uri)
    champion = result.champion_version
    (other,) = set(result.versions.values()) - {champion}

    running = load(tracking_uri)
    registry_client.set_registered_model_alias(MODEL_NAME, "champion", other)
    try:
        # The model is loaded once at startup, so a running worker keeps the old version
        assert running.version == champion
        assert load(tracking_uri).version == other
    finally:
        registry_client.set_registered_model_alias(MODEL_NAME, "champion", champion)
