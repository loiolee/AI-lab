"""实验编排模块的单元测试。

``utils/experiments.py`` 是 CLI 与报告图表的唯一数据来源，它出错会让
"报告里的数字"和"代码跑出来的数字"不一致，因此单独覆盖。

覆盖：实验编号与行数、同一题目在不同启发函数下的最优代价必须相同、
启发式不增加扩展节点数、四种访问顺序不改变博弈值、耗时字段非负。
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
    HEURISTIC_MANHATTAN,
    HEURISTIC_ZERO,
    astar_search,
)
from problems.maze import generate_maze  # noqa: E402
from utils.experiments import (  # noqa: E402
    astar_experiments,
    bench_astar,
    game_experiments,
    tree_order_experiments,
)


class TestAstarExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows = astar_experiments()

    def test_four_rows_with_expected_tags(self) -> None:
        self.assertEqual([row["experiment"] for row in self.rows], ["A1", "A2", "A3", "A4"])

    def test_required_fields_present(self) -> None:
        required = {"expanded_nodes", "visited_nodes", "path_length", "total_cost", "time"}
        for row in self.rows:
            with self.subTest(tag=row["experiment"]):
                self.assertTrue(required.issubset(row.keys()))

    def test_same_problem_same_optimal_cost(self) -> None:
        """换启发函数只能改变搜索量，不能改变最优代价。"""
        by_problem = {}
        for row in self.rows:
            by_problem.setdefault(row["problem"], []).append(row["total_cost"])
        for problem, costs in by_problem.items():
            with self.subTest(problem=problem):
                self.assertEqual(len(set(costs)), 1)

    def test_heuristic_does_not_expand_more(self) -> None:
        maze = [r for r in self.rows if r["problem"] == "problem1_maze"]
        zero = next(r for r in maze if r["heuristic"] == "zero")
        manhattan = next(r for r in maze if r["heuristic"] == "manhattan")
        self.assertLessEqual(manhattan["expanded_nodes"], zero["expanded_nodes"])

    def test_time_is_positive(self) -> None:
        for row in self.rows:
            with self.subTest(tag=row["experiment"]):
                self.assertGreater(row["time"], 0.0)


class TestBenchAstar(unittest.TestCase):
    def test_same_map_same_cost(self) -> None:
        start, goal, grid = generate_maze(12, seed=21)
        zero = bench_astar(start, goal, grid, HEURISTIC_ZERO, COST_UNIFORM, "test", repeats=3)
        manh = bench_astar(start, goal, grid, HEURISTIC_MANHATTAN, COST_UNIFORM, "test", repeats=3)
        self.assertEqual(zero.cost, manh.cost)
        self.assertLessEqual(manh.stats.expanded_nodes, zero.stats.expanded_nodes)

    def test_repeats_argument_does_not_change_result(self) -> None:
        start, goal, grid = generate_maze(10, seed=8)
        once = astar_search(start, goal, grid, HEURISTIC_MANHATTAN)
        many = bench_astar(start, goal, grid, HEURISTIC_MANHATTAN, COST_UNIFORM, "test", repeats=5)
        self.assertEqual(once.cost, many.cost)
        self.assertEqual(once.path, many.path)


class TestGameExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.minimax_rows, cls.alpha_rows = game_experiments()

    def test_all_seven_experiments_present(self) -> None:
        tags = [row["experiment"] for row in self.minimax_rows + self.alpha_rows]
        self.assertEqual(sorted(tags), ["B1", "B2", "B3", "B4", "B5", "B6", "B7"])

    def test_expected_values(self) -> None:
        values = {row["experiment"]: row["best_value"] for row in self.minimax_rows + self.alpha_rows}
        self.assertEqual(values["B1"], -1)   # H=2 Minimax
        self.assertEqual(values["B2"], -1)   # H=2 Alpha-Beta
        self.assertEqual(values["B3"], 3)    # H=4 Minimax
        # 三种访问顺序都不改变 H=4 的博弈值
        for tag in ("B4", "B5", "B6", "B7"):
            with self.subTest(tag=tag):
                self.assertEqual(values[tag], 3)

    def test_alpha_beta_never_visits_more_nodes(self) -> None:
        pairs = (("B1", "B2"), ("B3", "B4"))
        rows = {row["experiment"]: row for row in self.minimax_rows + self.alpha_rows}
        for mm_tag, ab_tag in pairs:
            with self.subTest(depth=mm_tag):
                self.assertLess(rows[ab_tag]["total_nodes"], rows[mm_tag]["total_nodes"])

    def test_single_layer_orders_sit_between_baseline_and_both(self) -> None:
        """单层排序应优于基线、劣于双向排序，否则说明排序没生效。"""
        rows = {row["experiment"]: row for row in self.alpha_rows}
        base = rows["B4"]["total_nodes"]
        max_high = rows["B5"]["total_nodes"]
        min_low = rows["B6"]["total_nodes"]
        both = rows["B7"]["total_nodes"]
        self.assertLess(max_high, base)
        self.assertLess(min_low, base)
        self.assertLess(both, max_high)
        self.assertLess(both, min_low)


class TestTreeOrderExperiments(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = tree_order_experiments()

    def test_both_trees_have_four_rows(self) -> None:
        self.assertEqual(len(self.data["pdf_tree"]), 4)
        self.assertEqual(len(self.data["random_tree"]), 4)

    def test_order_does_not_change_value(self) -> None:
        for tree, rows in self.data.items():
            values = {row["best_value"] for row in rows}
            with self.subTest(tree=tree):
                self.assertEqual(len(values), 1)

    def test_heuristic_order_is_cheapest_on_random_tree(self) -> None:
        rows = self.data["random_tree"]
        baseline = next(r for r in rows if r["order"] == "left_to_right")
        heuristic = next(r for r in rows if r["order"] == "heuristic")
        self.assertLess(heuristic["total_nodes"], baseline["total_nodes"])

    def test_node_counts_are_consistent(self) -> None:
        """总访问节点数不能少于叶子节点数。"""
        for rows in self.data.values():
            for row in rows:
                with self.subTest(order=row["order"]):
                    self.assertGreaterEqual(row["total_nodes"], row["leaf_nodes"])


if __name__ == "__main__":
    unittest.main()
