"""Alpha-Beta 剪枝。

Alpha-Beta 是 Minimax 的等价加速：在**相同的博弈树、相同的搜索深度、
相同的子节点访问顺序**下，它返回的最优值与 Minimax 完全一致，
只是跳过了那些不可能影响最终决策的分支。

两个边界
--------
* ``alpha`` —— MAX 目前能够保证的**下界**（"我已经能拿到这么多"）
* ``beta``  —— MIN 目前能够保证的**上界**（"我最多让你拿到这么多"）

剪枝条件 ``alpha >= beta``：当前节点的搜索结果已经不可能被上层采用，
剩余子节点可以全部跳过。

剪枝只能改变**搜索效率**，绝不能改变**最终结果**。
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.data_structures import TreeNode, count_nodes
from utils.metrics import GameSearchStats

from algorithms.minimax import _is_max_node

INF = math.inf

ORDER_LEFT_TO_RIGHT = "left_to_right"   # C1：严格按输入顺序，题目三的正式要求
ORDER_MAX_HIGH = "max_high"             # C2：仅 MAX 节点按估值降序，MIN 保持原序
ORDER_MIN_LOW = "min_low"               # C3：仅 MIN 节点按估值升序，MAX 保持原序
ORDER_HEURISTIC = "heuristic"           # C2 与 C3 同时生效（B5/B6 使用）
ORDER_MODES = (ORDER_LEFT_TO_RIGHT, ORDER_MAX_HIGH, ORDER_MIN_LOW, ORDER_HEURISTIC)

ORDER_LABELS = {
    ORDER_LEFT_TO_RIGHT: "从左到右",
    ORDER_MAX_HIGH: "MAX 优先高估值",
    ORDER_MIN_LOW: "MIN 优先低估值",
    ORDER_HEURISTIC: "双向启发式排序",
}


# ---------------------------------------------------------------------------
# 子节点访问顺序
# ---------------------------------------------------------------------------
def order_children(
    items: Sequence[Tuple[Any, float]],
    is_maximizing: bool,
    order: str = ORDER_LEFT_TO_RIGHT,
) -> List[Tuple[Any, float]]:
    """按指定策略排列子节点。

    参数
    ----
    items        : ``[(payload, score), ...]``，payload 为行动或子节点，
                   score 为该子节点的静态估值（用于排序）
    is_maximizing: 当前节点是否为 MAX
    order        : 四种模式

                   * ``left_to_right``：保持输入顺序，题目三的正式要求
                   * ``max_high``     ：仅在 MAX 节点按估值降序排列
                   * ``min_low``      ：仅在 MIN 节点按估值升序排列
                   * ``heuristic``    ：两者同时生效

    后三种模式使更可能触发剪枝的分支先被搜索。排序只影响**效率**，
    不改变 Minimax 值——这一点由单元测试对随机树逐一比对验证。
    """
    if order == ORDER_LEFT_TO_RIGHT:
        return list(items)
    if order == ORDER_HEURISTIC:
        return sorted(items, key=lambda item: item[1], reverse=is_maximizing)
    if order == ORDER_MAX_HIGH:
        # MIN 节点保持输入顺序，只有 MAX 节点改变访问次序
        return sorted(items, key=lambda item: item[1], reverse=True) if is_maximizing else list(items)
    if order == ORDER_MIN_LOW:
        return sorted(items, key=lambda item: item[1]) if not is_maximizing else list(items)
    raise ValueError(f"未知访问顺序: {order!r}，可选 {ORDER_MODES}")


def record_pruning(
    prune_log: Optional[List[Dict[str, Any]]],
    depth: int,
    is_maximizing: bool,
    alpha: float,
    beta: float,
    skipped_children: int,
    skipped_nodes: int = 0,
    note: str = "",
) -> None:
    """记录一次剪枝事件（写入 ``prune_log``，可为 ``None`` 表示不记录）。"""
    if prune_log is None:
        return
    prune_log.append({
        "depth": depth,
        "node_type": "MAX" if is_maximizing else "MIN",
        "alpha": alpha,
        "beta": beta,
        "skipped_children": skipped_children,
        "skipped_nodes": skipped_nodes,
        "note": note,
    })


# ---------------------------------------------------------------------------
# 博弈状态上的 Alpha-Beta
# ---------------------------------------------------------------------------
def alpha_beta(
    state: Any,
    game: Any,
    depth: int = 0,
    max_depth: int = 2,
    alpha: float = -INF,
    beta: float = INF,
    stats: Optional[GameSearchStats] = None,
    order: str = ORDER_LEFT_TO_RIGHT,
    prune_log: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, Optional[Any]]:
    """Alpha-Beta 递归搜索，返回 ``(最优评价值, 最优行动)``。

    与 :func:`algorithms.minimax.minimax` 接口一致，便于直接替换做对照实验。
    """
    if stats is None:
        stats = GameSearchStats(algorithm="Alpha-Beta", depth=max_depth, order=order)
    return _alpha_beta(state, game, depth, max_depth, alpha, beta, stats, order, prune_log)


def _alpha_beta(
    state: Any,
    game: Any,
    depth: int,
    max_depth: int,
    alpha: float,
    beta: float,
    stats: GameSearchStats,
    order: str,
    prune_log: Optional[List[Dict[str, Any]]],
) -> Tuple[float, Optional[Any]]:
    stats.total_nodes += 1
    if depth > stats.max_depth:
        stats.max_depth = depth

    # ---- 终止条件 ----
    if depth >= max_depth or game.is_terminal(state):
        stats.leaf_nodes += 1
        return float(game.evaluate(state)), None

    actions = list(game.actions(state))
    if not actions:
        stats.leaf_nodes += 1
        return float(game.evaluate(state)), None

    stats.expanded_nodes += 1
    maximizing = game.is_max_turn(state)

    # 生成子节点；除"从左到右"外的三种顺序都用子状态的静态估值作为排序依据，
    # 具体在哪一层排序由 order_children() 决定（MAX 层 / MIN 层 / 两层都排）
    children: List[Tuple[Any, Any]] = [(action, game.apply(state, action)) for action in actions]
    if order != ORDER_LEFT_TO_RIGHT:
        scored = [(idx, float(game.evaluate(child))) for idx, (_action, child) in enumerate(children)]
        ordered_index = [idx for idx, _score in order_children(scored, maximizing, order)]
        children = [children[idx] for idx in ordered_index]

    best_value = -INF if maximizing else INF
    best_action: Optional[Any] = None

    for index, (action, child) in enumerate(children):
        value, _ = _alpha_beta(child, game, depth + 1, max_depth, alpha, beta, stats, order, prune_log)

        if maximizing:
            if value > best_value:
                best_value, best_action = value, action
            if best_value > alpha:
                alpha = best_value
        else:
            if value < best_value:
                best_value, best_action = value, action
            if best_value < beta:
                beta = best_value

        # ---- 剪枝判定：只有当确实还有子节点没搜索时才构成一次剪枝 ----
        remaining = len(children) - index - 1
        if alpha >= beta and remaining > 0:
            stats.pruning_count += 1
            stats.pruned_branches += remaining
            record_pruning(prune_log, depth, maximizing, alpha, beta, remaining,
                           note="alpha >= beta，跳过剩余子节点")
            break

    return best_value, best_action


# ---------------------------------------------------------------------------
# 显式博弈树（题目三）上的 Alpha-Beta
# ---------------------------------------------------------------------------
def alpha_beta_max(root: TreeNode) -> float:
    """题目三的标准解：接收博弈树根节点，返回 MAX 玩家能保证的最佳得分。

    与下面的 :func:`alpha_beta_tree` 返回同一个值，区别在于这一个是
    **自包含的单一函数**——不依赖排序、统计或轨迹记录等任何辅助设施，
    因此可以直接作为题目要求的那个函数提交。

    规则
    ----
    * 根节点为 MAX，逐层交替，偶数层 MAX、奇数层 MIN
    * 子节点严格从左到右访问
    * 叶子节点取其 ``value``
    """
    def search(node: TreeNode, depth: int, alpha: float, beta: float) -> float:
        if not node.children:                       # 叶子：直接返回估值
            return float(node.value)

        if depth % 2 == 0:                          # 偶数层为 MAX
            best = -INF
            for child in node.children:             # 从左到右
                best = max(best, search(child, depth + 1, alpha, beta))
                alpha = max(alpha, best)
                if alpha >= beta:                   # 剪枝
                    break
            return best

        best = INF                                  # 奇数层为 MIN
        for child in node.children:
            best = min(best, search(child, depth + 1, alpha, beta))
            beta = min(beta, best)
            if alpha >= beta:
                break
        return best

    return search(root, 0, -INF, INF)


def alpha_beta_tree(
    root: TreeNode,
    order: str = ORDER_LEFT_TO_RIGHT,
    stats: Optional[GameSearchStats] = None,
    prune_log: Optional[List[Dict[str, Any]]] = None,
    trace: Optional[Dict[int, Dict[str, Any]]] = None,
) -> float:
    """对显式的 :class:`TreeNode` 博弈树求根节点的 Alpha-Beta 值。

    题目三要求**从左到右**访问子节点，因此默认 ``order="left_to_right"``。

    ``trace`` 用于记录逐节点搜索轨迹（键为 ``id(node)``），内容包括进入该节点时的
    ``alpha``、``beta``、返回的 ``value`` 以及被剪掉的子节点列表，
    供实验报告绘制剪枝示意图使用。
    """
    if stats is None:
        stats = GameSearchStats(algorithm="Alpha-Beta", depth=0, order=order)
    value, _ = _alpha_beta_tree(root, 0, -INF, INF, stats, order, prune_log, trace)
    return value


def _alpha_beta_tree(
    node: TreeNode,
    depth: int,
    alpha: float,
    beta: float,
    stats: GameSearchStats,
    order: str,
    prune_log: Optional[List[Dict[str, Any]]],
    trace: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[float, Optional[TreeNode]]:
    stats.total_nodes += 1
    if depth > stats.max_depth:
        stats.max_depth = depth

    maximizing = _is_max_node(node, depth)
    info: Dict[str, Any] = {
        "depth": depth,
        "node_type": "MAX" if maximizing else "MIN",
        "alpha_in": alpha,
        "beta_in": beta,
        "visited": True,
        "pruned_children": [],
        "value": None,
    }
    if trace is not None:
        trace[id(node)] = info

    if node.is_leaf:
        stats.leaf_nodes += 1
        value = float(node.value if node.value is not None else 0.0)
        info["value"] = value
        return value, None

    stats.expanded_nodes += 1

    items = [(child, estimate_node(child, depth + 1)) for child in node.children]
    ordered = order_children(items, maximizing, order)
    info["order"] = [id(child) for child, _score in ordered]

    best_value = -INF if maximizing else INF
    best_child: Optional[TreeNode] = None

    for index, (child, _score) in enumerate(ordered):
        value, _ = _alpha_beta_tree(child, depth + 1, alpha, beta, stats, order, prune_log, trace)

        if maximizing:
            if value > best_value:
                best_value, best_child = value, child
            if best_value > alpha:
                alpha = best_value
        else:
            if value < best_value:
                best_value, best_child = value, child
            if best_value < beta:
                beta = best_value

        # 树结构已知，可以精确统计被剪掉的节点数
        skipped = ordered[index + 1:]
        if alpha >= beta and skipped:
            skipped_nodes = sum(count_nodes(item[0]) for item in skipped)
            stats.pruning_count += 1
            stats.pruned_branches += len(skipped)
            stats.pruned_nodes += skipped_nodes
            info["pruned_children"] = [id(item[0]) for item in skipped]
            info["alpha_at_prune"] = alpha
            info["beta_at_prune"] = beta
            record_pruning(prune_log, depth, maximizing, alpha, beta,
                           len(skipped), skipped_nodes,
                           note="alpha >= beta，剪掉整棵子树")
            break

    info["value"] = best_value
    info["alpha_out"] = alpha
    info["beta_out"] = beta
    return best_value, best_child


def estimate_node(node: TreeNode, depth: int = 0, max_plies: int = 2) -> float:
    """子节点静态估值，仅用于**排序**（不影响最终结果）。

    叶子直接返回其 ``value``；内部节点做深度受限的极大极小估计。
    """
    if node.is_leaf:
        return float(node.value if node.value is not None else 0.0)
    if max_plies <= 0:
        leaves = _collect_leaf_values(node, limit=8)
        return float(sum(leaves) / len(leaves)) if leaves else 0.0

    maximizing = _is_max_node(node, depth)
    values = [estimate_node(child, depth + 1, max_plies - 1) for child in node.children]
    if not values:
        return 0.0
    return max(values) if maximizing else min(values)


def _collect_leaf_values(node: TreeNode, limit: int = 8) -> List[float]:
    values: List[float] = []
    stack = [node]
    while stack and len(values) < limit:
        current = stack.pop()
        if current.is_leaf:
            values.append(float(current.value if current.value is not None else 0.0))
        else:
            stack.extend(current.children)
    return values


# ---------------------------------------------------------------------------
# 计时封装
# ---------------------------------------------------------------------------
def timed_alpha_beta(
    state: Any,
    game: Any,
    max_depth: int,
    order: str = ORDER_LEFT_TO_RIGHT,
    problem: str = "",
    prune_log: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[float, Optional[Any], GameSearchStats]:
    """带计时与统计的 Alpha-Beta 调用封装，返回 ``(值, 首步, 统计)``。"""
    stats = GameSearchStats(
        algorithm="Alpha-Beta",
        problem=problem,
        depth=max_depth,
        order=order,
    )
    t0 = time.perf_counter()
    value, action = alpha_beta(
        state, game, 0, max_depth, -INF, INF, stats, order, prune_log
    )
    stats.execution_time = time.perf_counter() - t0
    stats.best_value = value
    stats.best_target = "" if action is None else str(action)
    return value, action, stats


def timed_alpha_beta_tree(
    root: TreeNode,
    order: str = ORDER_LEFT_TO_RIGHT,
    problem: str = "problem3_tree",
    prune_log: Optional[List[Dict[str, Any]]] = None,
    trace: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Tuple[float, GameSearchStats]:
    """题目三用：带计时与统计的树搜索。"""
    stats = GameSearchStats(algorithm="Alpha-Beta", problem=problem, order=order)
    t0 = time.perf_counter()
    value = alpha_beta_tree(root, order=order, stats=stats, prune_log=prune_log, trace=trace)
    stats.execution_time = time.perf_counter() - t0
    stats.best_value = value
    return value, stats
