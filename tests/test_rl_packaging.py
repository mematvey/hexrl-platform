import mlflow
import pandas as pd
import pytest

from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.navigation import NavigationObservation, NavigationTask
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.packaging import INPUT_COLUMNS, OUTPUT_COLUMN, log_policy_model
from hexrl_platform.rl.replay import default_map
from hexrl_platform.rl.training import DQNConfig, load_policy, train_dqn

TINY_DQN = DQNConfig(
    total_timesteps=200,
    learning_starts=50,
    buffer_size=500,
    batch_size=16,
    target_update_interval=50,
    eval_freq=100,
    net_arch=(16,),
)


@pytest.fixture
def logged_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hex_map = default_map()
    tasks = [NavigationTask(hex_map, HexCoord(-2, 0), HexCoord(0, 0), 10)]
    result = train_dqn(
        TINY_DQN,
        ObservationEncoder(hex_map, 10),
        tasks,
        lambda policy: {"success_rate": 0.0, "mean_return": 0.0},
        seed=1,
        output_dir=tmp_path / "policy",
    )
    previous_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    try:
        mlflow.set_experiment("packaging-test")
        with mlflow.start_run():
            model_info = log_policy_model(result.best_policy_dir)
        yield mlflow.pyfunc.load_model(model_info.model_uri), result.best_policy_dir
    finally:
        mlflow.set_tracking_uri(previous_uri)


def test_logged_model_has_explicit_signature(logged_model):
    model, _ = logged_model
    signature = model.metadata.signature
    assert signature.inputs.input_names() == list(INPUT_COLUMNS)
    assert signature.outputs.input_names() == [OUTPUT_COLUMN]
    assert model.metadata.metadata["encoder"]["version"] == "nav-obs-v1"


def test_logged_model_acts_like_the_saved_policy(logged_model):
    model, policy_dir = logged_model
    policy = load_policy(policy_dir)
    goal = HexCoord(0, 0)
    starts = [cell for cell in policy.encoder.hex_map.cells if cell != goal]
    frame = pd.DataFrame(
        [
            {"q": c.q, "r": c.r, "goal_q": goal.q, "goal_r": goal.r, "steps_remaining": 7}
            for c in starts
        ]
    )
    prediction = model.predict(frame)
    assert isinstance(prediction, pd.DataFrame)
    predicted = prediction[OUTPUT_COLUMN].tolist()
    expected = [policy.act(NavigationObservation(cell, goal, 7)) for cell in starts]
    assert predicted == expected


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"q": 9, "r": 0, "goal_q": 0, "goal_r": 0, "steps_remaining": 5}, "walkable"),
        ({"q": 0, "r": 1, "goal_q": 2, "goal_r": 0, "steps_remaining": 5}, "walkable"),
        ({"q": 2, "r": 0, "goal_q": 0, "goal_r": 0, "steps_remaining": 0}, "steps_remaining"),
        ({"q": 2, "r": 0, "goal_q": 0, "goal_r": 0, "steps_remaining": 11}, "steps_remaining"),
    ],
)
def test_logged_model_rejects_states_outside_its_map(logged_model, row, message):
    model, _ = logged_model
    with pytest.raises(Exception, match=message):
        model.predict(pd.DataFrame([row]))


def test_logging_requires_a_saved_policy(tmp_path):
    with pytest.raises(FileNotFoundError, match="No saved policy"):
        log_policy_model(tmp_path)
