import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from hexrl_platform.rl.gym_env import HexNavigationEnv
from hexrl_platform.rl.hex_grid import DIRECTIONS
from hexrl_platform.rl.navigation import NavigationObservation, NavigationTask
from hexrl_platform.rl.observation import ObservationEncoder
from hexrl_platform.rl.policies import Policy
from hexrl_platform.rl.values import Probability

POLICY_FILE = "policy.json"
MODEL_FILE = "model.zip"

Metrics = dict[str, float]
EvaluateFn = Callable[[Policy], Metrics]
EvaluationListener = Callable[[int, Metrics], object]


@dataclass(frozen=True, slots=True)
class DQNConfig:
    total_timesteps: int = 100_000
    learning_rate: float = 5e-4
    buffer_size: int = 50_000
    learning_starts: int = 1_000
    batch_size: int = 64
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 1_000
    exploration_fraction: float = 0.3
    exploration_initial_eps: Probability = Probability(1.0)
    exploration_final_eps: Probability = Probability(0.05)
    net_arch: tuple[int, ...] = (64, 64)
    eval_freq: int = 5_000
    device: str = "cpu"

    def __post_init__(self) -> None:
        positive = {
            "total_timesteps": self.total_timesteps,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "eval_freq": self.eval_freq,
        }
        for name, value in positive.items():
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.learning_starts < 0:
            raise ValueError("learning_starts must not be negative")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 < self.exploration_fraction <= 1:
            raise ValueError("exploration_fraction must be in (0, 1]")
        if not self.net_arch or any(size < 1 for size in self.net_arch):
            raise ValueError("net_arch must contain positive layer sizes")
        if self.device not in {"auto", "cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError("device must be auto, cpu, cuda or cuda:<index>")

    def to_params(self) -> dict[str, Any]:
        return {
            "total_timesteps": self.total_timesteps,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "train_freq": self.train_freq,
            "gradient_steps": self.gradient_steps,
            "target_update_interval": self.target_update_interval,
            "exploration_fraction": self.exploration_fraction,
            "exploration_initial_eps": self.exploration_initial_eps.value,
            "exploration_final_eps": self.exploration_final_eps.value,
            "net_arch": ",".join(map(str, self.net_arch)),
            "eval_freq": self.eval_freq,
            "device": self.device,
        }


class SB3Policy(Policy):
    """Adapts a trained SB3 model to the domain Policy protocol."""

    def __init__(self, model: DQN, encoder: ObservationEncoder) -> None:
        self.model = model
        self.encoder = encoder

    def act(self, observation: NavigationObservation) -> int:
        action, _ = self.model.predict(self.encoder.encode(observation), deterministic=True)
        return int(action)


@dataclass(frozen=True, slots=True)
class TrainingResult:
    best_step: int
    best_metrics: Metrics
    best_policy_dir: Path
    final_policy_dir: Path
    device: str
    curve: tuple[tuple[int, Metrics], ...] = field(default_factory=tuple)


def _is_better(candidate: Metrics, best: Metrics | None) -> bool:
    if best is None:
        return True
    return (candidate["success_rate"], candidate["mean_return"]) > (
        best["success_rate"],
        best["mean_return"],
    )


class PeriodicEvaluation(BaseCallback):
    def __init__(
        self,
        evaluate: EvaluateFn,
        encoder: ObservationEncoder,
        eval_freq: int,
        best_dir: Path,
        on_evaluation: EvaluationListener | None = None,
    ) -> None:
        super().__init__()
        self.evaluate = evaluate
        self.encoder = encoder
        self.eval_freq = eval_freq
        self.best_dir = best_dir
        self.on_evaluation = on_evaluation
        self.curve: list[tuple[int, Metrics]] = []
        self.best_step = 0
        self.best_metrics: Metrics | None = None
        self._last_evaluated_step = -1

    def _run_evaluation(self) -> None:
        if self.num_timesteps == self._last_evaluated_step:
            return
        self._last_evaluated_step = self.num_timesteps
        model = self.model
        if not isinstance(model, DQN):
            raise TypeError("PeriodicEvaluation supports DQN models only")
        metrics = self.evaluate(SB3Policy(model, self.encoder))
        self.curve.append((self.num_timesteps, metrics))
        if self.on_evaluation is not None:
            self.on_evaluation(self.num_timesteps, metrics)
        if _is_better(metrics, self.best_metrics):
            self.best_metrics = metrics
            self.best_step = self.num_timesteps
            save_policy(model, self.encoder, self.best_dir)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0:
            self._run_evaluation()
        return True

    def _on_training_end(self) -> None:
        self._run_evaluation()


def train_dqn(
    config: DQNConfig,
    encoder: ObservationEncoder,
    train_tasks: Sequence[NavigationTask],
    evaluate: EvaluateFn,
    *,
    seed: int,
    output_dir: Path,
    on_evaluation: EvaluationListener | None = None,
) -> TrainingResult:
    model = DQN(
        "MultiInputPolicy",
        HexNavigationEnv(train_tasks, encoder),
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps.value,
        exploration_final_eps=config.exploration_final_eps.value,
        policy_kwargs={"net_arch": list(config.net_arch)},
        seed=seed,
        device=config.device,
        verbose=0,
    )
    callback = PeriodicEvaluation(
        evaluate, encoder, config.eval_freq, output_dir / "best", on_evaluation
    )
    model.learn(total_timesteps=config.total_timesteps, callback=callback)
    final_dir = output_dir / "final"
    save_policy(model, encoder, final_dir)
    if callback.best_metrics is None:
        raise RuntimeError("Training finished without a single evaluation")
    return TrainingResult(
        best_step=callback.best_step,
        best_metrics=callback.best_metrics,
        best_policy_dir=output_dir / "best",
        final_policy_dir=final_dir,
        # SB3 silently falls back to CPU when CUDA is requested but unavailable
        device=str(model.device),
        curve=tuple(callback.curve),
    )


def save_policy(model: DQN, encoder: ObservationEncoder, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    model.save(directory / MODEL_FILE)
    description = {
        "algorithm": "DQN",
        "encoder": encoder.to_config(),
        "actions": [{"q": direction.q, "r": direction.r} for direction in DIRECTIONS],
    }
    (directory / POLICY_FILE).write_text(
        json.dumps(description, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_policy(directory: Path) -> SB3Policy:
    description = json.loads((directory / POLICY_FILE).read_text(encoding="utf-8"))
    actions = [(action["q"], action["r"]) for action in description["actions"]]
    if actions != [(direction.q, direction.r) for direction in DIRECTIONS]:
        raise ValueError("Saved action order does not match the simulator")
    encoder = ObservationEncoder.from_config(description["encoder"])
    model = DQN.load(directory / MODEL_FILE, device="cpu")
    return SB3Policy(model, encoder)
