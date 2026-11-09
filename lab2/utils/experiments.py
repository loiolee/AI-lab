"""统一的实验定义与基准测试。

把"跑哪些实验、用什么参数"集中到一处，命令行（``main.py``）与
报告图表（``utils.report_figures``）都从这里取数据，保证
**报告里的数字与 CSV 里的数字来自同一段代码**。

实验编号与报告中的表格一致：

=====  ==================================================
编号    内容
=====  ==================================================
A1~A4  题目一 / 题目二 的 A* 启发函数对照
B1~B6  题目四 的 Minimax 与 Alpha-Beta 对照
C1~C4  题目三 的四种子节点访问顺序（PDF 树 + 随机树）
D1~D8  拓展实验：不同地图规模下的 A* 效率
=====  ==================================================
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Sequence, Tuple

from algorithms.alpha_beta import (
    ORDER_HEURISTIC,
    ORDER_LEFT_TO_RIGHT,
    ORDER_MAX_HIGH,
    ORDER_MIN_LOW,
    timed_alpha_beta,
    timed_alpha_beta_tree,
)
from algorithms.astar import (
    COST_UNIFORM,
    COST_WEIGHTED,
    HEURISTIC_MANHATTAN,
    HEURISTIC_WEIGHTED,
    HEURISTIC_ZERO,
    astar_search,
)
from algorithms.minimax import timed_minimax
from problems.game_tree import build_sample_tree
from problems.maze import generate_maze, load_maze
from problems.treasure_game import TreasureGame, default_game
from problems.weighted_terrain import load_weighted_map
from utils.data_structures import build_random_tree

# 单次搜索只有几十微秒，重复取均值才能得到稳定的时间数据
ASTAR_REPEATS = 200
GAME_REPEATS = 100
TREE_REPEATS = 300

# 拓展实验使用的地图规模
SCALING_SIZES: Tuple[int, ...] = (10, 20, 30, 40, 50)
SCALING_OBSTACLE_RATIO = 0.25


# ---------------------------------------------------------------------------
# 基准封装
# ---------------------------------------------------------------------------
def bench_astar(
    start: Tuple[int, int],
    goal: Tuple[int, int],
    grid: Sequence[Sequence[object]],
    heuristic_type: str,
    cost_mode: str,
    problem: str,
    repeats: int = ASTAR_REPEATS,
):
    """重复运行 A* 取平均耗时，返回最后一次的完整结果。"""
    t0 = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = astar_search(start, goal, grid, heuristic_type, cost_mode, problem=problem)
    result.stats.execution_time = (time.perf_counter() - t0) / max(1, repeats)
    return result


def bench_game(
    game: TreasureGame,
    depth: int,
    algorithm: str,
    order: str = ORDER_LEFT_TO_RIGHT,
    repeats: int = GAME_REPEATS,
):
    """重复运行博弈搜索取平均耗时，返回 ``(值, 首步, 统计)``。"""
    state = game.initial_state()
    t0 = time.perf_counter()
    outcome = None
    for _ in range(repeats):
        if algorithm == "Minimax":
            outcome = timed_minimax(state, game, depth, order=order, problem="problem4")
        else:
            outcome = timed_alpha_beta(state, game, depth, order=order, problem="problem4")
    value, action, stats = outcome
    stats.execution_time = (time.perf_counter() - t0) / max(1, repeats)
    return value, action, stats


def bench_tree(root, order: str, repeats: int = TREE_REPEATS):
    """重复运行题目三的树搜索取平均耗时，返回 ``(值, 统计)``。"""
    t0 = time.perf_counter()
    value, stats = None, None
    for _ in range(repeats):
        value, stats = timed_alpha_beta_tree(root, order=order)
    stats.execution_time = (time.perf_counter() - t0) / max(1, repeats)
    return value, stats


# ---------------------------------------------------------------------------
# A1~A4：A* 启发函数对照
# ---------------------------------------------------------------------------
def astar_experiments() -> List[Dict[str, Any]]:
    """A1~A4：题目一与题目二的 h=0 / 启发式对照。"""
    maze_start, maze_goal, maze_grid = load_maze()
    w_start, w_goal, w_grid = load_weighted_map()

    definitions = [
        ("A1", "题目一 迷宫", "problem1_maze", maze_start, maze_goal, maze_grid,
         HEURISTIC_ZERO, COST_UNIFORM),
        ("A2", "题目一 迷宫", "problem1_maze", maze_start, maze_goal, maze_grid,
         HEURISTIC_MANHATTAN, COST_UNIFORM),
        ("A3", "题目二 加权地形", "problem2_weighted", w_start, w_goal, w_grid,
         HEURISTIC_WEIGHTED, COST_WEIGHTED),
        ("A4", "题目二 加权地形", "problem2_weighted", w_start, w_goal, w_grid,
         HEURISTIC_ZERO, COST_WEIGHTED),
    ]

    rows: List[Dict[str, Any]] = []
    for tag, label, problem, start, goal, grid, hid, cost_mode in definitions:
        result = bench_astar(start, goal, grid, hid, cost_mode, problem)
        row = result.stats.to_row()
        row["experiment"] = tag
        row["problem_label"] = label
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# D1~D8：不同地图规模下的 A* 效率（拓展实验一）
# ---------------------------------------------------------------------------
def astar_scaling_experiments(
    sizes: Sequence[int] = SCALING_SIZES,
    obstacle_ratio: float = SCALING_OBSTACLE_RATIO,
) -> List[Dict[str, Any]]:
    """在同一随机种子的不同规模地图上，比较 h=0 与 Manhattan。

    每张地图只用一种障碍密度与一个固定种子，保证 h=0 与 Manhattan
    跑的是**同一张地图**，对比才有意义。
    """
    rows: List[Dict[str, Any]] = []
    for index, size in enumerate(sizes, start=1):
        start, goal, grid = generate_maze(size, obstacle_ratio, seed=1000 + size)
        for hid, hid_label in ((HEURISTIC_ZERO, "h=0"), (HEURISTIC_MANHATTAN, "Manhattan")):
            result = bench_astar(start, goal, grid, hid, COST_UNIFORM, "scaling")
            rows.append({
                "experiment": f"D{index * 2 - (1 if hid == HEURISTIC_ZERO else 0)}",
                "map_size": f"{size}x{size}",
                "size": size,
                "obstacle_ratio": obstacle_ratio,
                "heuristic": hid_label,
                "heuristic_key": hid,
                "expanded_nodes": result.stats.expanded_nodes,
                "visited_nodes": result.stats.visited_nodes,
                "path_length": result.stats.path_length,
                "total_cost": result.stats.total_cost,
                "time": round(result.stats.execution_time, 6),
            })
    return rows


# ---------------------------------------------------------------------------
# B1~B6：题目四 Minimax 与 Alpha-Beta
# ---------------------------------------------------------------------------
GAME_DEFINITIONS = [
    ("B1", "Minimax", 2, ORDER_LEFT_TO_RIGHT),
    ("B2", "Alpha-Beta", 2, ORDER_LEFT_TO_RIGHT),
    ("B3", "Minimax", 4, ORDER_LEFT_TO_RIGHT),
    ("B4", "Alpha-Beta", 4, ORDER_LEFT_TO_RIGHT),
    # B5/B6 对应框架里的两种单层排序：只在 MAX 层按估值从高到低、
    # 只在 MIN 层按估值从低到高；B7 是两者同时启用的双向排序。
    ("B5", "Alpha-Beta", 4, ORDER_MAX_HIGH),
    ("B6", "Alpha-Beta", 4, ORDER_MIN_LOW),
    ("B7", "Alpha-Beta", 4, ORDER_HEURISTIC),
]


def game_experiments() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """返回 ``(minimax_rows, alpha_beta_rows)``，对应 B1~B7。"""
    game = default_game()
    minimax_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []

    for tag, algorithm, depth, order in GAME_DEFINITIONS:
        value, action, stats = bench_game(game, depth, algorithm, order)
        row = stats.to_row()
        row["experiment"] = tag
        (minimax_rows if algorithm == "Minimax" else alpha_rows).append(row)
    return minimax_rows, alpha_rows


# ---------------------------------------------------------------------------
# C1~C4：题目三的节点访问顺序（拓展实验二）
# ---------------------------------------------------------------------------
def tree_order_experiments() -> Dict[str, List[Dict[str, Any]]]:
    """在 PDF 给定的博弈树与一棵随机平衡树上分别比较四种访问顺序。

    返回 ``{"pdf_tree": rows, "random_tree": rows}``。
    """
    orders = [
        (ORDER_LEFT_TO_RIGHT, "C1", "从左到右"),
        (ORDER_MAX_HIGH, "C2", "MAX 优先高估值"),
        (ORDER_MIN_LOW, "C3", "MIN 优先低估值"),
        (ORDER_HEURISTIC, "C4", "双向启发式排序"),
    ]

    pdf_root = build_sample_tree()
    random_root = build_random_tree(branching=3, depth=4, seed=1)

    result: Dict[str, List[Dict[str, Any]]] = {"pdf_tree": [], "random_tree": []}
    for key, root in (("pdf_tree", pdf_root), ("random_tree", random_root)):
        for order, tag, label in orders:
            value, stats = bench_tree(root, order)
            result[key].append({
                "experiment": tag,
                "order": order,
                "order_label": label,
                "best_value": value,
                "total_nodes": stats.total_nodes,
                "leaf_nodes": stats.leaf_nodes,
                "pruning_count": stats.pruning_count,
                "pruned_nodes": stats.pruned_nodes,
                "time": stats.execution_time,
            })
    return result
