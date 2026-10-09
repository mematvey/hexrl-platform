from pathlib import Path

import pytest

from hexrl_platform.rl.study import StudyConfig, dqn_config, load_study_config
from hexrl_platform.rl.training import DQNConfig
from hexrl_platform.rl.values import Probability

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_full_study_config_differs_only_in_exploration_fraction():
    study = load_study_config(CONFIGS / "navigation-dqn.toml")
    assert study.seeds == (1, 2, 3, 4, 5)
    assert study.experiment == "hex-navigation-dqn"
    (label_a, config_a), (label_b, config_b) = study.variants
    assert (label_a, label_b) == ("dqn-explore-0.1", "dqn-explore-0.5")
    params_a, params_b = config_a.to_params(), config_b.to_params()
    differing = {name for name in params_a if params_a[name] != params_b[name]}
    assert differing == {"exploration_fraction"}
    assert config_a.exploration_final_eps == Probability(0.05)
    assert config_a.net_arch == (64, 64)


def test_smoke_study_config_loads():
    study = load_study_config(CONFIGS / "navigation-dqn-smoke.toml")
    assert all(config.total_timesteps <= 5_000 for _, config in study.variants)


def test_variant_overrides_shared_parameters(tmp_path):
    path = tmp_path / "study.toml"
    path.write_text(
        """
[study]
name = "s"
experiment = "e"
seeds = [7]

[dqn]
total_timesteps = 2_000
exploration_fraction = 0.2

[variants.a]

[variants.b]
exploration_fraction = 0.6
""",
        encoding="utf-8",
    )
    study = load_study_config(path)
    configs = dict(study.variants)
    assert configs["a"].exploration_fraction == 0.2
    assert configs["b"].exploration_fraction == 0.6
    assert configs["b"].total_timesteps == 2_000


def test_config_errors_are_reported(tmp_path):
    with pytest.raises(ValueError, match="Unknown DQN parameters: learnin_rate"):
        dqn_config({"learnin_rate": 0.1})
    path = tmp_path / "broken.toml"
    path.write_text('[study]\nname = "s"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        load_study_config(path)


def test_study_config_validation():
    variant = (("a", DQNConfig()),)
    with pytest.raises(ValueError, match="seeds"):
        StudyConfig(name="s", experiment="e", seeds=(), variants=variant)
    with pytest.raises(ValueError, match="unique labels"):
        StudyConfig(name="s", experiment="e", seeds=(1,), variants=variant * 2)
    with pytest.raises(ValueError, match="must not use"):
        StudyConfig(name="s", experiment="e", seeds=(1,), variants=(("random", DQNConfig()),))
    with pytest.raises(ValueError, match="must not be empty"):
        StudyConfig(name="", experiment="e", seeds=(1,), variants=variant)
