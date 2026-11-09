"""题目四综合博弈的单元测试。

覆盖：距离矩阵（对称性、对角线、已知数值）、状态转移、收益公式、
宝藏移除、玩家交替、终止条件，以及 PDF 给出的两组预期输出。
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.astar import astar_search, COST_WEIGHTED, HEURISTIC_WEIGHTED  # noqa: E402
from problems.treasure_game import (  # noqa: E402
    EXPECTED_RESULTS,
    PLAYER_START,
    TREASURE_MAP,
    TREASURE_VALUES,
    TreasureGame,
    apply_action,
    check_expected,
    default_game,
    evaluate_state,
    generate_actions,
    play_best_line,
    solve_treasure_game,
)


class TestDistanceMatrix(unittest.TestCase):
    def setUp(self) -> None:
        self.game = default_game()

    def test_labels_and_positions(self) -> None:
        self.assertEqual(self.game.labels, ["A", "B", "T1", "T2", "T3", "T4"])
        self.assertEqual(self.game.positions["A"], PLAYER_START["A"])
        self.assertEqual(self.game.positions["B"], PLAYER_START["B"])
        self.assertEqual(self.game.positions["T1"], (0, 6))
        self.assertEqual(self.game.positions["T2"], (2, 2))
        self.assertEqual(self.game.positions["T3"], (4, 4))
        self.assertEqual(self.game.positions["T4"], (6, 0))

    def test_diagonal_is_zero(self) -> None:
        for label in self.game.labels:
            self.assertEqual(self.game.distance_matrix[(label, label)], 0)

    def test_all_reachable(self) -> None:
        for key, value in self.game.distance_matrix.items():
            self.assertFalse(math.isinf(value), msg=f"{key} 不可达")

    def test_known_distances(self) -> None:
        # 手工推算：A->T2 沿 (1,0)(2,0)(2,1)(2,2) 共 4；B->T3 共 4
        self.assertEqual(self.game.distance_matrix[("A", "T2")], 4)
        self.assertEqual(self.game.distance_matrix[("B", "T3")], 4)
        self.assertEqual(self.game.distance_matrix[("A", "T3")], 8)
        self.assertEqual(self.game.distance_matrix[("A", "T1")], 10)

    def test_matrix_is_symmetric_for_unit_entry_cells(self) -> None:
        """六个关键格的进入代价都是 1，因此距离矩阵应当对称。"""
        for src in self.game.labels:
            for dst in self.game.labels:
                self.assertEqual(
                    self.game.distance_matrix[(src, dst)],
                    self.game.distance_matrix[(dst, src)],
                    msg=f"{src}->{dst} 与反向距离不一致",
                )

    def test_matrix_matches_direct_astar(self) -> None:
        for src in self.game.labels:
            for dst in self.game.labels:
                if src == dst:
                    continue
                result = astar_search(
                    self.game.positions[src], self.game.positions[dst],
                    self.game.grid, HEURISTIC_WEIGHTED, COST_WEIGHTED,
                )
                self.assertEqual(result.cost, self.game.distance_matrix[(src, dst)])

    def test_matrix_shape(self) -> None:
        rows = self.game.matrix_rows()
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(len(row) == 6 for row in rows))


class TestGameRules(unittest.TestCase):
    def setUp(self) -> None:
        self.game = default_game()
        self.state = self.game.initial_state()

    def test_initial_state(self) -> None:
        self.assertEqual(self.state.pos_a, (0, 0))
        self.assertEqual(self.state.pos_b, (6, 6))
        self.assertEqual(self.state.remaining, ("T1", "T2", "T3", "T4"))
        self.assertEqual(self.state.score_a, 0)
        self.assertEqual(self.state.score_b, 0)
        self.assertEqual(self.state.current_player, "A")
        self.assertEqual(self.state.depth, 0)

    def test_gain_formula(self) -> None:
        # A 到 T2 距离 4，价值 6 => gain = 2
        self.assertEqual(self.game.gain((0, 0), "T2"), 2)
        # A 到 T1 距离 10，价值 9 => gain = -1（收益可以为负）
        self.assertEqual(self.game.gain((0, 0), "T1"), -1)

    def test_apply_updates_position_score_and_remaining(self) -> None:
        new_state = self.game.apply(self.state, "T2")
        self.assertEqual(new_state.pos_a, (2, 2))
        self.assertEqual(new_state.score_a, 2)
        self.assertEqual(new_state.pos_b, (6, 6))
        self.assertEqual(new_state.remaining, ("T1", "T3", "T4"))
        self.assertEqual(new_state.last_action, "T2")
        self.assertEqual(new_state.depth, 1)
        self.assertEqual(new_state.current_player, "B")

    def test_apply_for_b_player(self) -> None:
        after_a = self.game.apply(self.state, "T2")
        after_b = self.game.apply(after_a, "T3")
        self.assertEqual(after_b.pos_b, (4, 4))
        self.assertEqual(after_b.score_b, 3)   # 7 - 4
        self.assertEqual(after_b.current_player, "A")
        self.assertEqual(after_b.remaining, ("T1", "T4"))

    def test_cannot_pick_taken_treasure(self) -> None:
        after_a = self.game.apply(self.state, "T2")
        with self.assertRaises(ValueError):
            self.game.apply(after_a, "T2")

    def test_actions_shrink(self) -> None:
        state = self.state
        self.assertEqual(generate_actions(state, self.game), ["T1", "T2", "T3", "T4"])
        state = apply_action(state, "T1", self.game)
        self.assertEqual(generate_actions(state, self.game), ["T2", "T3", "T4"])

    def test_evaluate_is_utility_difference(self) -> None:
        after_a = self.game.apply(self.state, "T2")
        after_b = self.game.apply(after_a, "T3")
        self.assertEqual(evaluate_state(after_b), after_b.score_a - after_b.score_b)
        self.assertEqual(evaluate_state(after_b), -1)

    def test_terminal_after_four_moves(self) -> None:
        state = self.state
        for action in ("T1", "T2", "T3", "T4"):
            self.assertFalse(self.game.is_terminal(state))
            state = self.game.apply(state, action)
        self.assertTrue(self.game.is_terminal(state))
        self.assertEqual(self.game.actions(state), [])

    def test_passing_over_treasure_does_not_collect_it(self) -> None:
        """经过其他宝藏不自动收集：只有被选择的宝藏才会离开剩余集合。"""
        # A 从 (0,0) 到 T1(0,6) 的最优路径会经过 T2(2,2)
        path = astar_search(
            (0, 0), self.game.positions["T1"],
            self.game.grid, HEURISTIC_WEIGHTED, COST_WEIGHTED,
        ).path
        self.assertIn((2, 2), path)
        state = self.game.apply(self.state, "T1")
        self.assertIn("T2", state.remaining)          # T2 并未被收集
        self.assertEqual(state.pos_a, (0, 6))
        self.assertEqual(state.score_a, -1)           # 9 - 10

    def test_custom_map_and_values(self) -> None:
        grid = [
            ["A", "1", "T1"],
            ["1", "1", "1"],
            ["T2", "1", "B"],
        ]
        game = TreasureGame(grid=grid, values={"T1": 5, "T2": 2},
                            players={"A": (0, 0), "B": (2, 2)})
        self.assertEqual(game.distance_matrix[("A", "T1")], 2)
        self.assertEqual(game.distance_matrix[("A", "T2")], 2)
        self.assertEqual(game.gain((0, 0), "T1"), 3)


class TestExpectedOutputs(unittest.TestCase):
    def test_expected_table_matches_pdf(self) -> None:
        self.assertEqual(EXPECTED_RESULTS[2], {"best_target": "T2", "best_value": -1})
        self.assertEqual(EXPECTED_RESULTS[4], {"best_target": "T3", "best_value": 3})

    def test_check_expected_helper(self) -> None:
        ok, _ = check_expected(2, "T2", -1)
        self.assertTrue(ok)
        bad, message = check_expected(4, "T2", 3)
        self.assertFalse(bad)
        self.assertIn("不一致", message)

    def test_treasure_values(self) -> None:
        self.assertEqual(TREASURE_VALUES, {"T1": 9, "T2": 6, "T3": 7, "T4": 7})
        self.assertEqual(len(TREASURE_MAP), 7)
        self.assertTrue(all(len(row) == 7 for row in TREASURE_MAP))


class TestSolveEntry(unittest.TestCase):
    """problems 层的解题入口：求解、算法切换、主变例回放。"""

    def setUp(self) -> None:
        self.game = default_game()

    def test_alpha_beta_expected_results(self) -> None:
        self.assertEqual(solve_treasure_game(self.game, 2), (-1, "T2"))
        self.assertEqual(solve_treasure_game(self.game, 4), (3, "T3"))

    def test_minimax_agrees_with_alpha_beta(self) -> None:
        for depth in (2, 4):
            with self.subTest(depth=depth):
                self.assertEqual(
                    solve_treasure_game(self.game, depth, "Minimax"),
                    solve_treasure_game(self.game, depth, "Alpha-Beta"),
                )

    def test_return_stats(self) -> None:
        value, action, stats = solve_treasure_game(self.game, 4, return_stats=True)
        self.assertEqual((value, action), (3, "T3"))
        self.assertEqual(stats.depth, 4)
        self.assertGreater(stats.total_nodes, 0)

    def test_unknown_algorithm_raises(self) -> None:
        with self.assertRaises(ValueError):
            solve_treasure_game(self.game, 2, "Negamax")

    def test_play_best_line_depth_2(self) -> None:
        line, final = play_best_line(self.game, 2)
        self.assertEqual(len(line), 2)
        self.assertEqual([row["player"] for row in line], ["A", "B"])
        self.assertEqual(line[0]["target"], "T2")
        self.assertEqual(final.utility(), -1)

    def test_play_best_line_depth_4(self) -> None:
        line, final = play_best_line(self.game, 4)
        self.assertEqual(len(line), 4)
        self.assertEqual([row["target"] for row in line], ["T3", "T1", "T2", "T4"])
        self.assertEqual(final.utility(), 3)
        # 终局分数与报告中的主变例一致：A=-1、B=-4
        self.assertEqual((final.score_a, final.score_b), (-1, -4))


if __name__ == "__main__":
    unittest.main()
