from importlib.metadata import version
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.models import ModelSignature
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import PythonModel, PythonModelContext
from mlflow.types import ColSpec, Schema

from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.navigation import NavigationObservation
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.training import POLICY_FILE, SB3Policy, load_policy

MODEL_ARTIFACT = "model"
POLICY_ARTIFACT = "policy"
INPUT_COLUMNS = ("q", "r", "goal_q", "goal_r", "steps_remaining")
OUTPUT_COLUMN = "action"

SIGNATURE = ModelSignature(
    inputs=Schema([ColSpec("long", column) for column in INPUT_COLUMNS]),
    outputs=Schema([ColSpec("long", OUTPUT_COLUMN)]),
)


class NavigationPolicyModel(PythonModel):
    """Raw navigation state in, action index out; encoder and action order travel inside."""

    def load_context(self, context: PythonModelContext) -> None:
        self._policy = load_policy(Path(context.artifacts[POLICY_ARTIFACT]))

    @property
    def policy(self) -> SB3Policy:
        return self._policy

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        encoder = self.policy.encoder
        actions = []
        for row in model_input.to_dict("records"):
            position = HexCoord(int(row["q"]), int(row["r"]))
            goal = HexCoord(int(row["goal_q"]), int(row["goal_r"]))
            steps_remaining = int(row["steps_remaining"])
            if not encoder.hex_map.is_walkable(position) or not encoder.hex_map.is_walkable(goal):
                raise ValueError("Position and goal must be walkable cells of the model map")
            if not 1 <= steps_remaining <= encoder.max_steps:
                raise ValueError(f"steps_remaining must be between 1 and {encoder.max_steps}")
            observation = NavigationObservation(position, goal, steps_remaining)
            actions.append(self.policy.act(observation))
        return pd.DataFrame({OUTPUT_COLUMN: pd.Series(actions, dtype="int64")})


def example_input(encoder: ObservationEncoder) -> pd.DataFrame:
    start, goal = encoder.hex_map.cells[0], encoder.hex_map.cells[-1]
    return pd.DataFrame(
        [
            {
                "q": start.q,
                "r": start.r,
                "goal_q": goal.q,
                "goal_r": goal.r,
                "steps_remaining": encoder.max_steps,
            }
        ]
    )


def model_requirements() -> list[str]:
    # torch is resolved by the serving image (CPU or CUDA build), not pinned per model
    packages = ("mlflow", "stable-baselines3", "gymnasium", "numpy", "pandas")
    return [f"{package}=={version(package)}" for package in packages] + ["torch"]


def log_policy_model(policy_dir: Path) -> ModelInfo:
    """Logs a saved policy as a pyfunc model into the active MLflow run."""
    if not (policy_dir / POLICY_FILE).exists():
        raise FileNotFoundError(f"No saved policy in {policy_dir}")
    encoder = load_policy(policy_dir).encoder
    return mlflow.pyfunc.log_model(
        name=MODEL_ARTIFACT,
        python_model=NavigationPolicyModel(),
        artifacts={POLICY_ARTIFACT: str(policy_dir)},
        signature=SIGNATURE,
        input_example=example_input(encoder),
        pip_requirements=model_requirements(),
        metadata={"encoder": encoder.to_config()},
    )
