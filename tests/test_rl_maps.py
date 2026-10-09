import pytest

from hexrl_platform.rl.hex_grid import HexCoord, hex_ring
from hexrl_platform.rl.replay import MAPS, maze_map

CENTER = HexCoord(0, 0)


@pytest.mark.parametrize("distance", [1, 2, 4])
def test_hex_ring_contains_every_cell_at_that_distance_once(distance):
    ring = hex_ring(CENTER, distance)
    assert len(ring) == len(set(ring)) == 6 * distance
    assert all(cell.distance_to(CENTER) == distance for cell in ring)
    # Consecutive cells are neighbours, so the ring is walked in order
    assert all(a.distance_to(b) == 1 for a, b in zip(ring, ring[1:] + ring[:1], strict=True))


def test_hex_ring_rejects_non_positive_distance():
    with pytest.raises(ValueError, match="positive"):
        hex_ring(CENTER, 0)


def test_maze_is_connected_and_forces_detours():
    hex_map = maze_map()
    distances = hex_map.distances_from(CENTER)
    assert set(distances) == set(hex_map.cells)
    outside = HexCoord(5, 0)
    assert outside.distance_to(CENTER) == 5
    assert distances[outside] > 3 * outside.distance_to(CENTER)


def test_maze_gaps_are_on_opposite_sides():
    hex_map = maze_map()
    assert hex_map.is_walkable(HexCoord(2, 0))
    assert hex_map.is_walkable(HexCoord(-4, 0))
    assert not hex_map.is_walkable(HexCoord(-2, 0))
    assert not hex_map.is_walkable(HexCoord(4, 0))


def test_maze_rejects_too_small_radius_and_presets_are_registered():
    with pytest.raises(ValueError, match="radius"):
        maze_map(4)
    assert set(MAPS) == {"default", "maze"}
    assert MAPS["default"]().radius == 4
    assert MAPS["maze"]().radius == 6
