"""Minimax

本模块**不依赖任何具体题目**，只面向下面这组"博弈模型"接口（鸭子类型）::

    game.actions(state)       -> Sequence[action]   当前玩家的全部合法行动
    game.apply(state, action) -> state              执行行动后的新状态（不可变）
    game.evaluate(state)      -> float              静态评价值
    game.is_terminal(state)   -> bool               是否终止
    game.is_max_turn(state)   -> bool               当前是否 MAX 玩家行动

题目四的 :class:`problems.treasure_game.TreasureGame` 实现该接口，
因此 Minimax 与 Alpha-Beta 共享完全相同的行动生成 / 状态转移 / 评价逻辑，
两者唯一的差别只在于"搜索方式"，保证实验对比是公平的。

评价值约定
----------
``U = Score_A - Score_B``。A 是 MAX（希望 U 最大），B 是 MIN（希望 U 最小）。
"""

from __future__ import annotations

import math
import time
from typing import Any, Optional, Tuple

from utils.data_structures import TreeNode
from utils.metrics import GameSearchStats

INF = math.inf


# ---------------------------------------------------------------------------
# 博弈状态上的 Minimax
# ---------------------------------------------------------------------------
def minimax(
    state: Any,
    game: Any,
    depth: int = 0,
    max_depth: int = 2,
    stats: Optional[GameSearchStats] = None,
) -> Tuple[float, Optional[Any]]:
    """Minimax 递归搜索。

    参数
    ----
    state     : 当前博弈状态
    game      : 博弈模型（提供 actions / apply / evaluate / is_terminal / is_max_turn）
    depth     : 当前层数（根为 0）
    max_depth : 最大搜索深度 H，``H=2`` 表示搜索两层行动
    stats     : 统计对象，若为 ``None`` 则内部创建

    返回
    ----
    ``(最优评价值, 最优行动)``。根节点返回的行动即"最优首步"。
    """
    if stats is None:
        stats = GameSearchStats(algorithm="Minimax", depth=max_depth)
    return _minimax(state, game, depth, max_depth, stats)


def _minimax(
    state: Any,
    game: Any,
    depth: int,
    max_depth: int,
    stats: GameSearchStats,
) -> Tuple[float, Optional[Any]]:
    stats.total_nodes += 1
    if depth > stats.max_depth:
        stats.max_depth = depth

    # ---- 终止条件 1：到达搜索深度；条件 2：没有剩余宝藏 ----
    if depth >= max_depth or game.is_terminal(state):
        stats.leaf_nodes += 1
        return float(game.evaluate(state)), None

    actions = list(game.actions(state))
    if not actions:
        stats.leaf_nodes += 1
        return float(game.evaluate(state)), None

    stats.expanded_nodes += 1
    maximizing = game.is_max_turn(state)

    best_value = -INF if maximizing else INF
    best_action: Optional[Any] = None

    for action in actions:
        child = game.apply(state, action)
        value, _ = _minimax(child, game, depth + 1, max_depth, stats)
        if maximizing:
            if value > best_value:
                best_value, best_action = value, action
        else:
            if value < best_value:
                best_value, best_action = value, action

    return best_value, best_action


# ---------------------------------------------------------------------------
# 显式博弈树（题目三）上的 Minimax
# ---------------------------------------------------------------------------
def minimax_tree(
    root: TreeNode,
    stats: Optional[GameSearchStats] = None,
) -> float:
    """对显式的 :class:`TreeNode` 博弈树求根节点的 Minimax 值。

    用作题目三 Alpha-Beta 结果的正确性基准。
    """
    if stats is None:
        stats = GameSearchStats(algorithm="Minimax", depth=0)
    value, _ = _minimax_tree(root, 0, stats)
    return value


def _minimax_tree(
    node: TreeNode,
    depth: int,
    stats: GameSearchStats,
) -> Tuple[float, Optional[TreeNode]]:
    stats.total_nodes += 1
    if depth > stats.max_depth:
        stats.max_depth = depth

    if node.is_leaf:
        stats.leaf_nodes += 1
        return float(node.value if node.value is not None else 0.0), None

    stats.expanded_nodes += 1
    maximizing = _is_max_node(node, depth)
    best_value = -INF if maximizing else INF
    best_child: Optional[TreeNode] = None

    for child in node.children:
        value, _ = _minimax_tree(child, depth + 1, stats)
        if maximizing:
            if value > best_value:
                best_value, best_child = value, child
        else:
            if value < best_value:
                best_value, best_child = value, child

    return best_value, best_child


def _is_max_node(node: TreeNode, depth: int) -> bool:
    """判断节点是 MAX 还是 MIN。

    优先使用节点自带的 ``is_maximizing_player``（JSON 输入样例中有该字段），
    否则按层数奇偶推断：根节点（depth=0）为 MAX，逐层交替。
    """
    if node.is_maximizing_player is not None:
        return bool(node.is_maximizing_player)
    return depth % 2 == 0


def timed_minimax(
    state: Any,
    game: Any,
    max_depth: int,
    order: str = "left_to_right",
    problem: str = "",
) -> Tuple[float, Optional[Any], GameSearchStats]:
    """带计时与统计的 Minimax 调用封装，返回 ``(值, 首步, 统计)``。"""
    stats = GameSearchStats(
        algorithm="Minimax",
        problem=problem,
        depth=max_depth,
        order=order,
    )
    t0 = time.perf_counter()
    value, action = minimax(state, game, 0, max_depth, stats)
    stats.execution_time = time.perf_counter() - t0
    stats.best_value = value
    stats.best_target = "" if action is None else str(action)
    return value, action, stats
