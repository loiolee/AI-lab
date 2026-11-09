"""Minimax 的单元测试。

覆盖：显式博弈树求解、MAX/MIN 交替、深度截断、根节点最优行动记录、
题目四 H=2 的预期结果。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.minimax import minimax, minimax_tree, timed_minimax  # noqa: E402
from problems.game_tree import build_sample_tree  # noqa: E402
from problems.treasure_game import default_game  # noqa: E402
from utils.data_structures import TreeNode  # noqa: E402


class TestMinimaxOnTree(unittest.TestCase):
    def test_sample_tree_value(self) -> None:
        self.assertEqual(minimax_tree(build_sample_tree()), 5)

    def test_simple_max_root(self) -> None:
        root = TreeNode(children=[TreeNode(value=1), TreeNode(value=9), TreeNode(value=4)])
        self.assertEqual(minimax_tree(root), 9)

    def test_simple_min_root(self) -> None:
        root = TreeNode(children=[TreeNode(value=1), TreeNode(value=9)], is_maximizing_player=False)
        self.assertEqual(minimax_tree(root), 1)

    def test_single_leaf(self) -> None:
        self.assertEqual(minimax_tree(TreeNode(value=7)), 7)

    def test_deeper_alternation(self) -> None:
        # MAX(root) -> MIN -> MAX(叶子)
        # min1 = min(max(3,5)=5, max(6,9)=9) = 5
        # min2 = min(max(1,2)=2, max(4,8)=8) = 2
        # root = max(5, 2) = 5
        def max_node(values):
            return TreeNode(children=[TreeNode(value=v) for v in values])

        min1 = TreeNode(children=[max_node([3, 5]), max_node([6, 9])])
        min2 = TreeNode(children=[max_node([1, 2]), max_node([4, 8])])
        root = TreeNode(children=[min1, min2])
        self.assertEqual(minimax_tree(root), 5)


class TestMinimaxOnTreasureGame(unittest.TestCase):
    def setUp(self) -> None:
        self.game = default_game()
        self.state = self.game.initial_state()

    def test_depth_two_expected_result(self) -> None:
        value, action, stats = timed_minimax(self.state, self.game, 2)
        self.assertEqual(action, "T2")
        self.assertEqual(value, -1)
        self.assertGreater(stats.total_nodes, 0)
        self.assertEqual(stats.leaf_nodes, 12)

    def test_root_is_max_player(self) -> None:
        self.assertTrue(self.game.is_max_turn(self.state))
        self.assertEqual(self.state.current_player, "A")

    def test_players_alternate(self) -> None:
        after_a = self.game.apply(self.state, "T1")
        self.assertEqual(after_a.current_player, "B")
        self.assertFalse(self.game.is_max_turn(after_a))
        after_b = self.game.apply(after_a, "T2")
        self.assertEqual(after_b.current_player, "A")

    def test_depth_limited_search_returns_action_at_root(self) -> None:
        value, action = minimax(self.state, self.game, 0, 1)
        # 深度 1 只搜索 A 的一步，直接按即时收益选最大值（T2 的 gain=2）
        self.assertEqual(action, "T2")
        self.assertEqual(value, 2)

    def test_terminal_state_evaluates_current_utility(self) -> None:
        state = self.state
        for action in ("T1", "T2", "T3", "T4"):
            state = self.game.apply(state, action)
        self.assertTrue(self.game.is_terminal(state))
        value, action = minimax(state, self.game, 0, 4)
        self.assertIsNone(action)
        self.assertEqual(value, state.utility())

    def test_state_is_immutable(self) -> None:
        before = self.state
        self.game.apply(before, "T3")
        self.assertEqual(before.remaining, ("T1", "T2", "T3", "T4"))
        self.assertEqual(before.score_a, 0)
        self.assertEqual(before.depth, 0)


if __name__ == "__main__":
    unittest.main()
