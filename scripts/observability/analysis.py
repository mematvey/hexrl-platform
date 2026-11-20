from collections.abc import Sequence

Points = Sequence[tuple[float, float]]


def first_time(points: Points, threshold: float, *, after: float = float("-inf")) -> float | None:
    """Первый момент, когда значение достигло порога: начало установившегося заполнения."""
    return next((ts for ts, value in points if ts >= after and value >= threshold), None)


def good_fraction(good: float, total: float) -> float | None:
    return None if total <= 0 else good / total


def error_budget(total: float, objective: float) -> float:
    """Сколько плохих исходов допускает SLO на том же множестве задач."""
    if not 0 < objective < 1:
        raise ValueError("objective must be between 0 and 1")
    return total * (1 - objective)


def histogram_series(buckets: int, combinations: int) -> int:
    """Классическая Prometheus histogram: K границ + Inf + sum + count на сочетание labels."""
    return (buckets + 3) * combinations


def label_cardinality(*values_per_label: int) -> int:
    combinations = 1
    for count in values_per_label:
        combinations *= count
    return combinations
