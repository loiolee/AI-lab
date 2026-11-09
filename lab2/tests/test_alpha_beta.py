"""Alpha-Beta 剪枝的单元测试。

覆盖：题目三预期输出、与 Minimax 结果一致性（随机树）、剪枝确实发生、
访问顺序不改变结果、题目四 H=2 / H=4 的预期结果、剪枝统计口径。
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.alpha_beta import (  # noqa: E402
    ORDER_HEURISTIC,
    ORDER_LEFT_TO_RIGHT,
    ORDER_MAX_HIGH,
    ORDER_MIN_LOW,
    alpha_beta_max,
    alpha_beta_tree,
    order_children,
    timed_alpha_beta,
    timed_alpha_beta_tree,
)
from algorithms.minimax import minimax_tree, timed_minimax  # noqa: E402
from problems.game_tree import build_sample_tree  # noqa: E402
from problems.treasure_game import default_game  # noqa: E402
from utils.data_structures import (  # noqa: E402
    TreeNode,
    build_random_tree,
    count_nodes,
)


def random_tree(rng: random.Random, depth: int, branching: int = 3) -> TreeNode:
    """生成随机博弈树（不设显式标记，由层数奇偶决定 MAX/MIN）。"""
    if depth == 0:
        return TreeNode(value=rng.randint(-10, 10))
    children = [random_tree(rng, depth - 1, branching) for _ in range(rng.randint(1, branching))]
    return TreeNode(children=children)


class TestProblem3(unittest.TestCase):
    def test_expected_value_is_five(self) -> None:
        self.assertEqual(alpha_beta_tree(build_sample_tree()), 5)

    def test_matches_minimax(self) -> None:
        root = build_sample_tree()
        self.assertEqual(alpha_beta_tree(root), minimax_tree(root))

    def test_pruning_happens(self) -> None:
        value, stats = timed_alpha_beta_tree(build_sample_tree())
        self.assertEqual(value, 5)
        self.assertEqual(stats.pruning_count, 1)
        # 被剪掉的是 max2 的第二个叶子（值 9），树结构已知可精确统计
        self.assertEqual(stats.pruned_nodes, 1)
        self.assertEqual(stats.leaf_nodes, 7)
        self.assertLess(stats.total_nodes, count_nodes(build_sample_tree()))

    def test_order_does_not_change_value(self) -> None:
        root = build_sample_tree()
        self.assertEqual(
            alpha_beta_tree(root, order=ORDER_LEFT_TO_RIGHT),
            alpha_beta_tree(root, order=ORDER_HEURISTIC),
        )


class TestProblem3PureFunction(unittest.TestCase):
    """题目三要求「一个接收根节点、返回 MAX 最佳得分的函数」，这里验证
    :func:`alpha_beta_max` 满足该契约，且与完整实现、Minimax 结果一致。"""

    def test_single_argument_returns_five(self) -> None:
        self.assertEqual(alpha_beta_max(build_sample_tree()), 5)

    def test_agrees_with_full_implementation(self) -> None:
        root = build_sample_tree()
        self.assertEqual(alpha_beta_max(root), alpha_beta_tree(root))

    def test_agrees_on_random_trees(self) -> None:
        for seed in range(1, 31):
            tree = build_random_tree(branching=3, depth=4, seed=seed)
            with self.subTest(seed=seed):
                self.assertEqual(alpha_beta_max(tree), minimax_tree(tree))

    def test_min_and_max_single_child(self) -> None:
        # 两个 MIN 节点各有一个叶子：MAX(min(3), min(7)) = MAX(3, 7) = 7
        tree = TreeNode(children=[
            TreeNode(children=[TreeNode(value=3)]),
            TreeNode(children=[TreeNode(value=7)]),
        ])
        self.assertEqual(alpha_beta_max(tree), 7)

    def test_negative_values(self) -> None:
        # MAX(min(-5, -2), min(-9, -1)) = MAX(-5, -9) = -5
        tree = TreeNode(children=[
            TreeNode(children=[TreeNode(value=-5), TreeNode(value=-2)]),
            TreeNode(children=[TreeNode(value=-9), TreeNode(value=-1)]),
        ])
        self.assertEqual(alpha_beta_max(tree), -5)


class TestAlphaBetaEqualsMinimax(unittest.TestCase):
    def test_random_trees(self) -> None:
        rng = random.Random(20261109)
        for case in range(40):
            root = random_tree(rng, depth=rng.randint(1, 4), branching=3)
            expected = minimax_tree(root)
            for order in (ORDER_LEFT_TO_RIGHT, ORDER_HEURISTIC):
                with self.subTest(case=case, order=order):
                    self.assertEqual(alpha_beta_tree(root, order=order), expected)

    def test_pruning_never_increases_nodes(self) -> None:
        rng = random.Random(7)
        for _ in range(20):
            root = random_tree(rng, depth=4, branching=3)
            from algorithms.minimax import GameSearchStats
            mm_stats = GameSearchStats()
            minimax_tree(root, stats=mm_stats)
            ab_value, ab_stats = timed_alpha_beta_tree(root)
            self.assertEqual(ab_value, minimax_tree(root))
            self.assertLessEqual(ab_stats.total_nodes, mm_stats.total_nodes)


class TestOrderChildren(unittest.TestCase):
    def test_left_to_right_keeps_input_order(self) -> None:
        items = [("a", 3), ("b", 9), ("c", 1)]
        self.assertEqual(order_children(items, True, ORDER_LEFT_TO_RIGHT), items)

    def test_max_prefers_high_scores(self) -> None:
        items = [("a", 3), ("b", 9), ("c", 1)]
        self.assertEqual([x[0] for x in order_children(items, True, ORDER_HEURISTIC)], ["b", "a", "c"])

    def test_min_prefers_low_scores(self) -> None:
        items = [("a", 3), ("b", 9), ("c", 1)]
        self.assertEqual([x[0] for x in order_children(items, False, ORDER_HEURISTIC)], ["c", "a", "b"])

    def test_max_high_only_sorts_max_layer(self) -> None:
        """C2 只在 MAX 层排序，MIN 层保持原顺序。"""
        items = [("a", 3), ("b", 9), ("c", 1)]
        self.assertEqual([x[0] for x in order_children(items, True, ORDER_MAX_HIGH)], ["b", "a", "c"])
        self.assertEqual(order_children(items, False, ORDER_MAX_HIGH), items)

    def test_min_low_only_sorts_min_layer(self) -> None:
        """C3 只在 MIN 层排序，MAX 层保持原顺序。"""
        items = [("a", 3), ("b", 9), ("c", 1)]
        self.assertEqual([x[0] for x in order_children(items, False, ORDER_MIN_LOW)], ["c", "a", "b"])
        self.assertEqual(order_children(items, True, ORDER_MIN_LOW), items)

    def test_unknown_order_raises(self) -> None:
        with self.assertRaises(ValueError):
            order_children([("a", 1)], True, "unknown")


class TestFourOrdersAgree(unittest.TestCase):
    """四种顺序只影响访问次序，不能影响博弈值。"""

    def test_pdf_tree_all_orders(self) -> None:
        for order in (ORDER_LEFT_TO_RIGHT, ORDER_MAX_HIGH, ORDER_MIN_LOW, ORDER_HEURISTIC):
            with self.subTest(order=order):
                self.assertEqual(alpha_beta_tree(build_sample_tree(), order=order), 5)

    def test_random_tree_all_orders(self) -> None:
        root = build_random_tree(branching=3, depth=4, seed=1)
        expected = minimax_tree(root)
        for order in (ORDER_LEFT_TO_RIGHT, ORDER_MAX_HIGH, ORDER_MIN_LOW, ORDER_HEURISTIC):
            with self.subTest(order=order):
                self.assertEqual(alpha_beta_tree(root, order=order), expected)


class TestProblem4(unittest.TestCase):
    def setUp(self) -> None:
        self.game = default_game()
        self.state = self.game.initial_state()

    def test_depth_two(self) -> None:
        value, action, stats = timed_alpha_beta(self.state, self.game, 2)
        self.assertEqual(action, "T2")
        self.assertEqual(value, -1)
        self.assertGreater(stats.pruning_count, 0)

    def test_depth_four(self) -> None:
        value, action, _stats = timed_alpha_beta(self.state, self.game, 4)
        self.assertEqual(action, "T3")
        self.assertEqual(value, 3)

    def test_matches_minimax_at_both_depths(self) -> None:
        for depth in (2, 4):
            with self.subTest(depth=depth):
                mm_value, mm_action, _ = timed_minimax(self.state, self.game, depth)
                ab_value, ab_action, _ = timed_alpha_beta(self.state, self.game, depth)
                self.assertEqual(mm_value, ab_value)
                self.assertEqual(mm_action, ab_action)

    def test_heuristic_order_keeps_result(self) -> None:
        for depth in (2, 4):
            with self.subTest(depth=depth):
                base_value, base_action, _ = timed_alpha_beta(self.state, self.game, depth)
                heur_value, heur_action, _ = timed_alpha_beta(
                    self.state, self.game, depth, order=ORDER_HEURISTIC
                )
                self.assertEqual(base_value, heur_value)
                self.assertEqual(base_action, heur_action)

    def test_pruned_branches_are_not_guessed(self) -> None:
        """题目四无法预知被剪分支规模，pruned_nodes 必须保持 0。"""
        _value, _action, stats = timed_alpha_beta(self.state, self.game, 4)
        self.assertEqual(stats.pruned_nodes, 0)
        self.assertGreater(stats.pruned_branches, 0)


if __name__ == "__main__":
    unittest.main()
