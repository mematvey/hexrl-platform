from hexrl_platform.rl.hex_grid import DIRECTIONS, HexCoord
from hexrl_platform.rl.navigation import NavigationEnv, NavigationObservation, NavigationTask
from hexrl_platform.rl.policies import RandomPolicy, ShortestPathPolicy
from hexrl_platform.rl.replay import default_map


def test_random_policy_is_reproducible_and_uses_all_actions():
    observation = NavigationObservation(HexCoord(0, 0), HexCoord(2, 0), 10)
    policy_a, policy_b = RandomPolicy(seed=5), RandomPolicy(seed=5)
    actions_a = [policy_a.act(observation) for _ in range(200)]
    actions_b = [policy_b.act(observation) for _ in range(200)]
    assert actions_a == actions_b
    assert set(actions_a) == set(range(len(DIRECTIONS)))


def test_shortest_path_policy_reaches_goal_in_optimal_steps_around_obstacles():
    hex_map = default_map()
    policy = ShortestPathPolicy(hex_map)
    for start, goal in [(HexCoord(0, 2), HexCoord(0, -1)), (HexCoord(-3, 1), HexCoord(3, -1))]:
        environment = NavigationEnv(NavigationTask(hex_map, start, goal, max_steps=30))
        result = None
        for _ in range(30):
            result = environment.step(policy.act(environment.observe()))
            assert not result.blocked_move
            if result.terminated or result.truncated:
                break
        assert result is not None and result.terminated
        assert environment.steps_taken == hex_map.shortest_distance(start, goal)
