"""题目三（problems 层）的单元测试。

覆盖：给定树的结构、解题入口、与 Minimax 基准的一致性、剪枝记录、
镜像树的等价性，以及字符画渲染与预期值校验。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from problems.game_tree import (  # noqa: E402
    EXPECTED_VALUE,
    build_mirrored_tree,
    build_sample_tree,
    check_expected,
    format_tree,
    solve_tree,
    solve_tree_baseline,
)
from utils.data_structures import count_nodes  # noqa: E402


def collect_leaves(node):
    if node.is_leaf:
        return [node.value]
    values = []
    for child in node.children:
        values.extend(collect_leaves(child))
    return values


class TestSampleTree(unittest.TestCase):
    def test_node_count(self) -> None:
        # 1 个根 + 2 个 MIN + 4 个 MAX + 8 个叶子；max2 被复用，故节点对象为 15 个
        self.assertEqual(count_nodes(build_sample_tree()), 15)

    def test_leaf_values(self) -> None:
        self.assertEqual(sorted(collect_leaves(build_sample_tree())),
                         [1, 2, 3, 5, 6, 6, 9, 9])

    def test_root_is_max(self) -> None:
        self.assertTrue(build_sample_tree().is_maximizing_player)

    def test_max2_is_shared(self) -> None:
        """PDF 原代码里 min2 复用 max2，这里照原样保留。"""
        root = build_sample_tree()
        min1, min2 = root.children
        self.assertIs(min1.children[1], min2.children[0])


class TestMirroredTree(unittest.TestCase):
    def test_same_value(self) -> None:
        self.assertEqual(solve_tree(build_mirrored_tree()), EXPECTED_VALUE)

    def test_nodes_not_shared(self) -> None:
        root = build_mirrored_tree()
        min1, min2 = root.children
        self.assertIsNot(min1.children[1], min2.children[0])

    def test_same_leaf_multiset(self) -> None:
        self.assertEqual(sorted(collect_leaves(build_mirrored_tree())),
                         sorted(collect_leaves(build_sample_tree())))


class TestSolveEntry(unittest.TestCase):
    def test_solve_returns_expected(self) -> None:
        self.assertEqual(solve_tree(), EXPECTED_VALUE)

    def test_baseline_agrees(self) -> None:
        self.assertEqual(solve_tree_baseline(), solve_tree())

    def test_return_stats(self) -> None:
        value, stats = solve_tree(return_stats=True)
        self.assertEqual(value, EXPECTED_VALUE)
        self.assertEqual(stats.total_nodes, 14)
        self.assertEqual(stats.leaf_nodes, 7)
        self.assertEqual(stats.pruning_count, 1)

    def test_prune_log_filled(self) -> None:
        log = []
        solve_tree(prune_log=log)
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["depth"], 2)
        self.assertEqual(log[0]["node_type"], "MAX")
        self.assertEqual(log[0]["skipped_nodes"], 1)

    def test_custom_root(self) -> None:
        # MAX(MIN(1, 2), MIN(3, 4)) = MAX(1, 3) = 3
        from utils.data_structures import TreeNode
        tree = TreeNode(children=[
            TreeNode(children=[TreeNode(value=1), TreeNode(value=2)]),
            TreeNode(children=[TreeNode(value=3), TreeNode(value=4)]),
        ])
        self.assertEqual(solve_tree(tree), 3)


class TestFormatAndCheck(unittest.TestCase):
    def test_format_tree_lines(self) -> None:
        lines = format_tree(build_sample_tree())
        self.assertEqual(len(lines), 15)          # 15 个节点各占一行
        self.assertEqual(lines[0], "MAX")
        self.assertIn("[3]", "\n".join(lines))

    def test_check_expected_ok(self) -> None:
        ok, message = check_expected(5)
        self.assertTrue(ok)
        self.assertIn("一致", message)

    def test_check_expected_fail(self) -> None:
        ok, message = check_expected(4)
        self.assertFalse(ok)
        self.assertIn("不一致", message)


if __name__ == "__main__":
    unittest.main()
