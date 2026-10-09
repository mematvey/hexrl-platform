import math
from collections.abc import Mapping, Sequence

from matplotlib.figure import Figure

from hexrl_platform.rl.hex_grid import HexCoord, HexMap

Curve = Sequence[tuple[int, float]]


def _to_xy(cell: HexCoord) -> tuple[float, float]:
    return cell.q + cell.r / 2, cell.r * math.sqrt(3) / 2


def learning_curves(
    curves: Mapping[str, Sequence[Curve]],
    metric: str,
    references: Mapping[str, float] | None = None,
) -> Figure:
    """One thin line per seed and a thick mean line per variant."""
    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    for label, seed_curves in sorted(curves.items()):
        by_step: dict[int, list[float]] = {}
        for curve in seed_curves:
            steps, values = zip(*curve, strict=True) if curve else ((), ())
            line = axis.plot(steps, values, alpha=0.25, linewidth=1)
            color = line[0].get_color()
            for step, value in curve:
                by_step.setdefault(step, []).append(value)
        steps = sorted(by_step)
        means = [sum(by_step[step]) / len(by_step[step]) for step in steps]
        axis.plot(steps, means, linewidth=2.5, label=f"{label} (mean)", color=color)
    for label, value in sorted((references or {}).items()):
        axis.axhline(value, linestyle="--", linewidth=1.5, label=label, color="0.4")
    axis.set(title=f"Validation {metric} during training", xlabel="Timesteps", ylabel=metric)
    axis.legend()
    figure.tight_layout()
    return figure


def variant_comparison(values: Mapping[str, Sequence[float]], metric: str) -> Figure:
    """Every seed as a point, the mean as a bar: few seeds make spread visible."""
    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    labels = sorted(values)
    for index, label in enumerate(labels):
        points = values[label]
        axis.bar(index, sum(points) / len(points), alpha=0.35)
        axis.scatter([index] * len(points), points, zorder=3, color="black", s=20)
    axis.set_xticks(range(len(labels)), labels, rotation=15)
    axis.set(title=f"Test {metric} by variant", ylabel=metric)
    figure.tight_layout()
    return figure


def episode_lengths(lengths: Mapping[str, Sequence[int]], max_steps: int) -> Figure:
    figure = Figure(figsize=(8, 5))
    axis = figure.subplots()
    bins = range(0, max_steps + 2, max(1, max_steps // 30))
    for label, values in sorted(lengths.items()):
        axis.hist(values, bins=bins, alpha=0.5, label=label)
    axis.set(title="Test episode lengths", xlabel="Steps", ylabel="Episodes")
    axis.legend()
    figure.tight_layout()
    return figure


def trajectory(
    hex_map: HexMap, start: HexCoord, goal: HexCoord, path: Sequence[HexCoord], title: str
) -> Figure:
    figure = Figure(figsize=(6, 6))
    axis = figure.subplots()
    walkable = [_to_xy(cell) for cell in hex_map.cells]
    blocked = [_to_xy(cell) for cell in hex_map.blocked]
    axis.scatter(*zip(*walkable, strict=True), s=260, marker="h", color="0.9", edgecolor="0.7")
    if blocked:
        axis.scatter(*zip(*blocked, strict=True), s=260, marker="h", color="0.25")
    if path:
        xs, ys = zip(*(_to_xy(cell) for cell in path), strict=True)
        axis.plot(xs, ys, marker="o", markersize=4, linewidth=1.5, alpha=0.8)
    axis.scatter(*_to_xy(start), s=120, color="tab:blue", label="start", zorder=3)
    axis.scatter(*_to_xy(goal), s=160, marker="*", color="tab:red", label="goal", zorder=3)
    axis.set(title=title, aspect="equal")
    axis.axis("off")
    axis.legend(loc="upper right")
    figure.tight_layout()
    return figure
