from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Probability:
    value: float

    def __post_init__(self) -> None:
        if not 0 <= self.value <= 1:
            raise ValueError("Probability value must be in [0,1]")
