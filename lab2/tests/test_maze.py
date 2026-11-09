"""随机迷宫生成的单元测试。

覆盖：连通性校验、可复现性、障碍比例、起终点位置、非法参数、
以及生成出来的地图能被 A* 正常求解。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.astar import HEURISTIC_MANHATTAN, astar_search  # noqa: E402
from problems.maze import generate_maze, is_reachable  # noqa: E402


class TestIsReachable(unittest.TestCase):
    def test_open_grid_is_reachable(self) -> None:
        grid = [["S", ".", "."], [".", ".", "."], [".", ".", "E"]]
        self.assertTrue(is_reachable(grid, (0, 0), (2, 2)))

    def test_walled_off_goal_is_unreachable(self) -> None:
        grid = [["S", ".", "X"], [".", ".", "X"], ["X", "X", "E"]]
        self.assertFalse(is_reachable(grid, (0, 0), (2, 2)))

    def test_same_cell_is_reachable(self) -> None:
        grid = [["S", "X"], ["X", "E"]]
        self.assertTrue(is_reachable(grid, (0, 0), (0, 0)))

    def test_diagonal_gap_does_not_count(self) -> None:
        """只允许上下左右移动，对角相邻的缝隙不算连通。"""
        grid = [["S", "X"], ["X", "E"]]
        self.assertFalse(is_reachable(grid, (0, 0), (1, 1)))


class TestGenerateMaze(unittest.TestCase):
    def test_shape_and_endpoints(self) -> None:
        start, goal, grid = generate_maze(12, seed=3)
        self.assertEqual(len(grid), 12)
        self.assertTrue(all(len(row) == 12 for row in grid))
        self.assertEqual(start, (0, 0))
        self.assertEqual(goal, (11, 11))
        self.assertEqual(grid[start[0]][start[1]], "S")
        self.assertEqual(grid[goal[0]][goal[1]], "E")

    def test_generated_maze_is_connected(self) -> None:
        for size in (10, 20, 30):
            with self.subTest(size=size):
                start, goal, grid = generate_maze(size, seed=size)
                self.assertTrue(is_reachable(grid, start, goal))

    def test_same_seed_gives_same_map(self) -> None:
        first = generate_maze(15, seed=42)
        second = generate_maze(15, seed=42)
        self.assertEqual(first, second)

    def test_different_seeds_differ(self) -> None:
        _s1, _g1, grid_a = generate_maze(15, seed=1)
        _s2, _g2, grid_b = generate_maze(15, seed=2)
        self.assertNotEqual(grid_a, grid_b)

    def test_obstacle_ratio_is_respected(self) -> None:
        size = 30
        _start, _goal, grid = generate_maze(size, obstacle_ratio=0.30, seed=5)
        blocked = sum(row.count("X") for row in grid)
        ratio = blocked / (size * size)
        # 起点与终点不会被占用，比例略低于设定值
        self.assertGreater(ratio, 0.15)
        self.assertLess(ratio, 0.32)

    def test_size_too_small_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_maze(2)

    def test_generated_maze_is_solvable_by_astar(self) -> None:
        start, goal, grid = generate_maze(25, seed=11)
        result = astar_search(start, goal, grid, HEURISTIC_MANHATTAN)
        self.assertTrue(result.path)
        self.assertEqual(result.path[0], start)
        self.assertEqual(result.path[-1], goal)
        self.assertEqual(len(result.path) - 1, result.cost)


if __name__ == "__main__":
    unittest.main()
