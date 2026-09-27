from collections import deque
from dataclasses import dataclass, field


@dataclass(frozen=True, order=True, slots=True)
class HexCoord:
    q: int
    r: int

    def distance_to(self, other: "HexCoord") -> int:
        dq = self.q - other.q
        dr = self.r - other.r
        return max(abs(dq), abs(dr), abs(dq + dr))


DIRECTIONS: tuple[HexCoord, ...] = (
    HexCoord(1, 0),
    HexCoord(1, -1),
    HexCoord(0, -1),
    HexCoord(-1, 0),
    HexCoord(-1, 1),
    HexCoord(0, 1),
)


@dataclass(frozen=True, slots=True)
class HexMap:
    radius: int
    blocked: frozenset[HexCoord] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        if self.radius < 1:
            raise ValueError("Map radius must be positive")
        if any(not self.contains(cell) for cell in self.blocked):
            raise ValueError("Blocked cells must be inside the map")
        if len(self.cells) < 2:
            raise ValueError("Map must contain at least two walkable cells")

    def contains(self, cell: HexCoord) -> bool:
        return cell.distance_to(HexCoord(0, 0)) <= self.radius

    def is_walkable(self, cell: HexCoord) -> bool:
        return self.contains(cell) and cell not in self.blocked

    @property
    def cells(self) -> tuple[HexCoord, ...]:
        return tuple(
            HexCoord(q, r)
            for q in range(-self.radius, self.radius + 1)
            for r in range(-self.radius, self.radius + 1)
            if self.is_walkable(HexCoord(q, r))
        )

    def neighbors(self, cell: HexCoord) -> tuple[HexCoord, ...]:
        if not self.is_walkable(cell):
            raise ValueError("Cell is not walkable")
        return tuple(
            neighbor
            for direction in DIRECTIONS
            if self.is_walkable(neighbor := HexCoord(cell.q + direction.q, cell.r + direction.r))
        )

    def distances_from(self, start: HexCoord) -> dict[HexCoord, int]:
        if not self.is_walkable(start):
            raise ValueError("Start cell is not walkable")
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def shortest_distance(self, start: HexCoord, goal: HexCoord) -> int | None:
        return self.distances_from(start).get(goal)
