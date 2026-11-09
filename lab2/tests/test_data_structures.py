"""数据结构与随机博弈树构造的单元测试。

覆盖：``count_nodes`` 的计数口径、``build_random_tree`` 的形状与可复现性、
叶子取值范围，以及随机树上 Minimax 与 Alpha-Beta 结果一致。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.alpha_beta import ORDER_HEURISTIC, alpha_beta_tree  # noqa: E402
from algorithms.minimax import minimax_tree  # noqa: E402
from problems.game_tree import build_sample_tree  # noqa: E402
from utils.data_structures import (  # noqa: E402
    TreeNode,
    build_random_tree,
    count_nodes,
)


def collect_leaves(node: TreeNode):
    if node.is_leaf:
        yield node
    for child in node.children:
        yield from collect_leaves(child)


class TestCountNodes(unittest.TestCase):
    def test_single_leaf(self) -> None:
        self.assertEqual(count_nodes(TreeNode(value=3)), 1)

    def test_sample_tree_total(self) -> None:
        # 1 个根 + 2 个 MIN + 4 个 MAX + 8 个叶子
        self.assertEqual(count_nodes(build_sample_tree()), 15)


class TestBuildRandomTree(unittest.TestCase):
    def test_balanced_shape(self) -> None:
        root = build_random_tree(branching=3, depth=4)
        self.assertEqual(count_nodes(root), sum(3 ** k for k in range(5)))
        self.assertEqual(len(list(collect_leaves(root))), 3 ** 4)

    def test_depth_counts_by_level(self) -> None:
        root = build_random_tree(branching=2, depth=3)
        self.assertEqual(count_nodes(root), 1 + 2 + 4 + 8)

    def test_same_seed_gives_same_values(self) -> None:
        first = [leaf.value for leaf in collect_leaves(build_random_tree(seed=7))]
        second = [leaf.value for leaf in collect_leaves(build_random_tree(seed=7))]
        self.assertEqual(first, second)

    def test_different_seeds_differ(self) -> None:
        first = [leaf.value for leaf in collect_leaves(build_random_tree(seed=1))]
        second = [leaf.value for leaf in collect_leaves(build_random_tree(seed=2))]
        self.assertNotEqual(first, second)

    def test_leaf_values_within_range(self) -> None:
        root = build_random_tree(seed=3, low=-5, high=5)
        for leaf in collect_leaves(root):
            self.assertGreaterEqual(leaf.value, -5)
            self.assertLessEqual(leaf.value, 5)

    def test_depth_zero_is_leaf(self) -> None:
        root = build_random_tree(depth=0)
        self.assertTrue(root.is_leaf)

    def test_search_algorithms_agree_on_random_tree(self) -> None:
        root = build_random_tree(branching=3, depth=4, seed=1)
        expected = minimax_tree(root)
        self.assertEqual(alpha_beta_tree(root, order=ORDER_HEURISTIC), expected)


if __name__ == "__main__":
    unittest.main()
