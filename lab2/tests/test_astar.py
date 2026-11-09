"""A* 搜索的单元测试。

覆盖：最短路径正确性、路径合法性、无解、起点=终点、启发函数可采纳性、
加权地形代价规则、路径恢复、统计量一致性。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.astar import (  # noqa: E402
    COST_UNIFORM,
    COST_WEIGHTED,
    HEURISTIC_MANHATTAN,
    HEURISTIC_WEIGHTED,
    HEURISTIC_ZERO,
    astar_search,
    get_move_cost,
    get_neighbors,
    heuristic,
    min_terrain_cost,
    reconstruct_path,
)
from problems.maze import (  # noqa: E402
    MAZE_SAMPLE,
    MAZE_SAMPLE_EXPECTED,
    MAZE_SAMPLE_GOAL,
    MAZE_SAMPLE_START,
    format_maze_result,
    load_maze,
    solve_maze,
    validate_path,
)
from problems.weighted_terrain import (  # noqa: E402
    WEIGHTED_SAMPLE,
    WEIGHTED_SAMPLE_EXPECTED,
    WEIGHTED_SAMPLE_GOAL,
    calculate_path_cost,
    load_weighted_map,
    solve_weighted_terrain,
    verify_admissible,
)


class TestHeuristic(unittest.TestCase):
    def test_zero_heuristic(self) -> None:
        self.assertEqual(heuristic((0, 0), (3, 4), HEURISTIC_ZERO), 0)

    def test_manhattan_distance(self) -> None:
        self.assertEqual(heuristic((0, 0), (3, 4), HEURISTIC_MANHATTAN), 7)
        self.assertEqual(heuristic((2, 2), (2, 2), HEURISTIC_MANHATTAN), 0)

    def test_weighted_heuristic_never_negative(self) -> None:
        grid = WEIGHTED_SAMPLE
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                self.assertGreaterEqual(
                    heuristic((r, c), WEIGHTED_SAMPLE_GOAL, HEURISTIC_WEIGHTED, grid), 0
                )

    def test_min_terrain_cost(self) -> None:
        # 题目二地图中最小进入代价为 1（E 的 0 不计入）
        self.assertEqual(min_terrain_cost(WEIGHTED_SAMPLE), 1.0)

    def test_weighted_heuristic_is_admissible(self) -> None:
        report = verify_admissible(WEIGHTED_SAMPLE, WEIGHTED_SAMPLE_GOAL, HEURISTIC_WEIGHTED)
        self.assertTrue(report["admissible"], msg=f"违反可采纳性: {report['violations']}")
        self.assertEqual(report["violation_count"], 0)


class TestNeighborsAndCost(unittest.TestCase):
    def test_neighbor_order_is_deterministic(self) -> None:
        neighbors = get_neighbors((1, 2), MAZE_SAMPLE)
        # 上(0,2) 下(2,2) 左(1,1)=X 右(1,3)
        self.assertEqual(neighbors, [(0, 2), (2, 2), (1, 3)])

    def test_neighbors_skip_blocked_and_out_of_bounds(self) -> None:
        for neighbor in get_neighbors((0, 0), MAZE_SAMPLE):
            self.assertNotEqual(MAZE_SAMPLE[neighbor[0]][neighbor[1]], "X")
            self.assertTrue(0 <= neighbor[0] < len(MAZE_SAMPLE))
            self.assertTrue(0 <= neighbor[1] < len(MAZE_SAMPLE[0]))

    def test_uniform_cost_is_one(self) -> None:
        self.assertEqual(get_move_cost((0, 0), (0, 1), MAZE_SAMPLE, COST_UNIFORM), 1)

    def test_weighted_cost_uses_entry_cell(self) -> None:
        # (1,1) -> (1,2) 进入权值为 3 的沼泽
        self.assertEqual(get_move_cost((1, 1), (1, 2), WEIGHTED_SAMPLE, COST_WEIGHTED), 3)

    def test_goal_entry_cost_is_zero(self) -> None:
        self.assertEqual(get_move_cost((4, 3), (4, 4), WEIGHTED_SAMPLE, COST_WEIGHTED), 0)


class TestReconstructPath(unittest.TestCase):
    def test_reconstruct_returns_start_first(self) -> None:
        came_from = {(1, 0): (0, 0), (2, 0): (1, 0)}
        self.assertEqual(reconstruct_path(came_from, (2, 0)), [(0, 0), (1, 0), (2, 0)])

    def test_reconstruct_single_node(self) -> None:
        self.assertEqual(reconstruct_path({}, (0, 0)), [(0, 0)])


class TestMazeProblem(unittest.TestCase):
    def test_load_maze_detects_start_and_goal(self) -> None:
        start, goal, grid = load_maze()
        self.assertEqual(start, MAZE_SAMPLE_START)
        self.assertEqual(goal, MAZE_SAMPLE_GOAL)
        self.assertEqual(len(grid), len(MAZE_SAMPLE))

    def test_expected_path(self) -> None:
        start, goal, grid = load_maze()
        path = solve_maze(start, goal, grid)
        self.assertEqual(format_maze_result(path), format_maze_result(MAZE_SAMPLE_EXPECTED))

    def test_path_is_legal_and_shortest(self) -> None:
        start, goal, grid = load_maze()
        path = solve_maze(start, goal, grid)
        ok, message = validate_path(path, start, goal, grid)
        self.assertTrue(ok, msg=message)
        # 曼哈顿距离是单位代价地图上的下界，路径长度等于它即为最短
        self.assertEqual(len(path) - 1, 7)

    def test_unreachable_returns_empty_list(self) -> None:
        grid = [
            ["S", "X", "."],
            ["X", "X", "."],
            [".", ".", "E"],
        ]
        self.assertEqual(solve_maze((0, 0), (2, 2), grid), [])

    def test_start_equals_goal(self) -> None:
        result = astar_search((0, 0), (0, 0), MAZE_SAMPLE, HEURISTIC_MANHATTAN, COST_UNIFORM)
        self.assertEqual(result.path, [(0, 0)])
        self.assertEqual(result.cost, 0)

    def test_zero_heuristic_keeps_optimality(self) -> None:
        start, goal, grid = load_maze()
        zero = astar_search(start, goal, grid, HEURISTIC_ZERO, COST_UNIFORM)
        manhattan = astar_search(start, goal, grid, HEURISTIC_MANHATTAN, COST_UNIFORM)
        self.assertEqual(zero.cost, manhattan.cost)
        # 曼哈顿启发式应当扩展不多于 Dijkstra 的节点
        self.assertLessEqual(manhattan.stats.expanded_nodes, zero.stats.expanded_nodes)

    def test_stats_are_consistent(self) -> None:
        start, goal, grid = load_maze()
        result = astar_search(start, goal, grid, HEURISTIC_MANHATTAN, COST_UNIFORM)
        stats = result.stats
        self.assertEqual(stats.path_length, len(result.path) - 1)
        self.assertEqual(stats.total_cost, result.cost)
        self.assertGreater(stats.expanded_nodes, 0)
        self.assertGreaterEqual(stats.visited_nodes, stats.expanded_nodes)
        self.assertGreaterEqual(stats.generated_nodes, stats.visited_nodes)


class TestWeightedTerrain(unittest.TestCase):
    def test_expected_cost(self) -> None:
        start, goal, grid = load_weighted_map()
        self.assertEqual(solve_weighted_terrain(start, goal, grid), WEIGHTED_SAMPLE_EXPECTED)

    def test_path_cost_matches_solver(self) -> None:
        start, goal, grid = load_weighted_map()
        result = astar_search(start, goal, grid, HEURISTIC_WEIGHTED, COST_WEIGHTED)
        self.assertEqual(calculate_path_cost(result.path, grid), result.cost)

    def test_weighted_heuristic_keeps_optimality(self) -> None:
        start, goal, grid = load_weighted_map()
        zero = astar_search(start, goal, grid, HEURISTIC_ZERO, COST_WEIGHTED)
        weighted = astar_search(start, goal, grid, HEURISTIC_WEIGHTED, COST_WEIGHTED)
        self.assertEqual(zero.cost, weighted.cost)
        self.assertLessEqual(weighted.stats.expanded_nodes, zero.stats.expanded_nodes)

    def test_shortest_steps_not_shortest_cost(self) -> None:
        """题目二的关键点：步数最少不等于代价最低。"""
        start, goal, grid = load_weighted_map()
        result = astar_search(start, goal, grid, HEURISTIC_WEIGHTED, COST_WEIGHTED)
        # 曼哈顿距离为 8 步，实际路径也是 8 步；但若贪心按步数走"对角线"会穿过 8 的沼泽
        naive = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 3), (3, 3), (3, 4), (4, 4)]
        self.assertGreater(calculate_path_cost(naive, grid), result.cost)

    def test_unreachable_returns_minus_one(self) -> None:
        grid = [
            ["S", "X", "1"],
            ["X", "X", "1"],
            ["1", "1", "E"],
        ]
        self.assertEqual(solve_weighted_terrain((0, 0), (2, 2), grid), -1)

    def test_start_equals_goal_returns_zero(self) -> None:
        start, goal, grid = load_weighted_map()
        self.assertEqual(solve_weighted_terrain(start, start, grid), 0)


if __name__ == "__main__":
    unittest.main()
