import numpy as np
from matplotlib.figure import Figure

from hexrl_platform.rl import plots
from hexrl_platform.rl.hex_grid import HexCoord
from hexrl_platform.rl.replay import default_map


def test_learning_curves_draw_seed_lines_means_and_references():
    figure = plots.learning_curves(
        {
            "dqn-a": [[(100, 0.1), (200, 0.4)], [(100, 0.3), (200, 0.6)]],
            "dqn-b": [[(100, 0.2), (200, 0.2)]],
        },
        "success_rate",
        references={"random": 0.15},
    )
    assert isinstance(figure, Figure)
    (axis,) = figure.axes
    mean_line = next(line for line in axis.lines if line.get_label() == "dqn-a (mean)")
    assert np.asarray(mean_line.get_ydata()).tolist() == [0.2, 0.5]
    assert any(line.get_label() == "random" for line in axis.lines)


def test_other_figures_render_without_pyplot():
    hex_map = default_map()
    path = [HexCoord(-2, 0), HexCoord(-1, 0), HexCoord(0, 0)]
    figures = [
        plots.variant_comparison({"random": [0.1, 0.2], "dqn": [0.7, 0.8, 0.9]}, "success_rate"),
        plots.episode_lengths({"random": [60, 60, 12], "dqn": [4, 5, 6]}, max_steps=60),
        plots.trajectory(hex_map, path[0], HexCoord(2, 0), path, "failed episode"),
    ]
    assert all(isinstance(figure, Figure) and figure.axes for figure in figures)
