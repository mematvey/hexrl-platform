import json
import shutil
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from hexrl_platform.rl.bootstrap import bootstrap, ensure_champion
from hexrl_platform.rl.errors import BootstrapError
from hexrl_platform.rl.inputs import InputSpec, ensure_inputs
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.registry import RegistryConfig
from hexrl_platform.rl.replay import ReplayGenerationConfig, default_map
from hexrl_platform.rl.study import StudyConfig
from hexrl_platform.rl.training import DQNConfig

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
TINY_DQN = DQNConfig(
    total_timesteps=200,
    learning_starts=50,
    buffer_size=500,
    batch_size=16,
    target_update_interval=50,
    eval_freq=100,
    net_arch=(16,),
)
SPECS = (InputSpec("smoke", 30, 1), InputSpec("invalid", 5, 2, invalid_fraction=1.0))


def test_inputs_are_generated_once_and_rebuilt_when_stale(tmp_path):
    encoder = ObservationEncoder(default_map(), 20)
    first = ensure_inputs(encoder, tmp_path, "models:/m/1", SPECS)
    second = ensure_inputs(encoder, tmp_path, "models:/m/1", SPECS)
    assert first == {"smoke": True, "invalid": True}
    assert second == {"smoke": False, "invalid": False}

    (tmp_path / "smoke.jsonl").write_text("{}\n", encoding="utf-8")
    assert ensure_inputs(encoder, tmp_path, "models:/m/1", SPECS)["smoke"] is True
    # Новая версия модели - новое кодирование: входы пересоздаются под неё
    assert ensure_inputs(encoder, tmp_path, "models:/m/2", SPECS) == {
        "smoke": True,
        "invalid": True,
    }
    invalid = [json.loads(line) for line in (tmp_path / "invalid.jsonl").read_text().splitlines()]
    assert all(row["q"] > encoder.hex_map.radius for row in invalid)


def test_bootstrap_refuses_to_train_a_model_the_worker_will_not_load(tmp_path):
    with pytest.raises(BootstrapError, match="worker loads"):
        bootstrap(
            CONFIGS / "navigation-dqn-smoke.toml",
            model_name="hex-navigation-default",
            alias="champion",
            tracking_uri="http://127.0.0.1:1",
            work_dir=tmp_path,
            inputs_dir=None,
        )


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    previous = mlflow.get_tracking_uri()
    yield f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}"
    mlflow.set_tracking_uri(previous)


def test_champion_is_trained_once_and_then_only_verified(tmp_path, tracking):
    study = StudyConfig(
        name="bootstrap-test",
        experiment="bootstrap-test",
        replays=Path("unused.jsonl"),
        seeds=(1,),
        variants=(("dqn", TINY_DQN),),
    )
    registry = RegistryConfig(model_name="nav-bootstrap")
    replays = ReplayGenerationConfig(episodes=200, seed=5, max_steps=12)

    first = ensure_champion(
        study, registry, tracking_uri=tracking, work_dir=tmp_path / "w", replays=replays
    )
    second = ensure_champion(
        study, registry, tracking_uri=tracking, work_dir=tmp_path / "w", replays=replays
    )
    client = MlflowClient(tracking_uri=tracking)
    assert first.trained and not second.trained
    assert first.version == second.version
    assert len(client.search_model_versions("name = 'nav-bootstrap'")) == 1
    assert (
        str(client.get_model_version_by_alias("nav-bootstrap", "champion").version) == first.version
    )

    # alias остался, а артефакта нет: это ошибка стенда, а не повод тихо обучить заново
    shutil.rmtree(tmp_path / "mlruns")
    # MLflow бросает разные типы для разных хранилищ артефактов
    with pytest.raises(Exception):
        ensure_champion(
            study, registry, tracking_uri=tracking, work_dir=tmp_path / "w", replays=replays
        )
    assert len(client.search_model_versions("name = 'nav-bootstrap'")) == 1
