"""题目二：加权地形。

地图由代表进入代价的数字组成（1=平原，3=沼泽，8=山……），``X`` 为障碍。
从 A 到相邻格 B 的代价等于**进入 B** 的地形权值；终点 ``E`` 的进入代价为 0。
返回最小总代价，无解返回 -1。

启发函数可采纳性
----------------
此时普通曼哈顿距离不再可采纳（每步代价可能远大于 1）。本模块提供

    h(n) = (d - 1) * c_min + cost(goal)

其中 ``c_min`` 由当前地图**实际计算**（本图为 1），``cost(E) = 0``，
因此本题退化为 ``(d-1) * 1``，与框架建议的 ``max(0, Manhattan-1) * 1`` 一致。

:func:`verify_admissible` 会遍历地图上每个格子，用 A* 求出真实剩余代价，
逐个比较 ``h(n) <= 真实代价``，从**实证**角度验证可采纳性。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from algorithms.astar import (
    COST_WEIGHTED,
    HEURISTIC_WEIGHTED,
    HEURISTIC_ZERO,
    AStarResult,
    astar_search,
    cell_entry_cost,
    heuristic,
    min_terrain_cost,
    normalize_grid,
)

Coord = Tuple[int, int]

# PDF 题目二的测试用例
WEIGHTED_SAMPLE: List[List[str]] = [
    ["S", "1", "1", "8", "1"],
    ["1", "8", "3", "1", "1"],
    ["1", "8", "1", "8", "1"],
    ["1", "1", "1", "8", "1"],
    ["8", "8", "1", "1", "E"],
]
WEIGHTED_SAMPLE_START: Coord = (0, 0)
WEIGHTED_SAMPLE_GOAL: Coord = (4, 4)
WEIGHTED_SAMPLE_EXPECTED: int = 7
WEIGHTED_SAMPLE_EXPECTED_PATH: List[Coord] = [
    (0, 0), (1, 0), (2, 0), (3, 0), (3, 1),
    (3, 2), (4, 2), (4, 3), (4, 4),
]


def load_weighted_map(
    grid: Optional[Sequence[Sequence[object]]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
) -> Tuple[Coord, Coord, List[List[str]]]:
    """读取并校验加权地图，返回 ``(start, goal, grid)``。"""
    grid = normalize_grid(grid if grid is not None else WEIGHTED_SAMPLE)

    if start is None or goal is None:
        marks: Dict[str, Coord] = {}
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                text = str(cell).strip().upper()
                if text in ("S", "E"):
                    marks[text] = (r, c)
        start = start if start is not None else marks.get("S")
        goal = goal if goal is not None else marks.get("E")

    if start is None:
        raise ValueError("未能确定起点：请显式传入 start 或在地图中标注 'S'")
    if goal is None:
        raise ValueError("未能确定终点：请显式传入 goal 或在地图中标注 'E'")

    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))
    height, width = len(grid), len(grid[0])
    for name, pos in (("起点", start), ("终点", goal)):
        if not (0 <= pos[0] < height and 0 <= pos[1] < width):
            raise ValueError(f"{name} {pos} 超出地图范围 {height}x{width}")
        if cell_entry_cost(grid[pos[0]][pos[1]]) is None:
            raise ValueError(f"{name} {pos} 落在障碍物上")
    return start, goal, grid


def calculate_path_cost(
    path: Sequence[Coord],
    grid: Sequence[Sequence[object]],
) -> float:
    """按"进入格的权值"累计路径实际总代价（起点不计代价）。"""
    grid = normalize_grid(grid)
    if len(path) < 2:
        return 0.0
    total = 0.0
    for r, c in path[1:]:
        cost = cell_entry_cost(grid[r][c])
        if cost is None:
            raise ValueError(f"路径经过障碍物 {(r, c)}")
        total += cost
    return total


def solve_weighted_terrain(
    start: Coord,
    goal: Coord,
    grid: Sequence[Sequence[object]],
    heuristic_type: str = HEURISTIC_WEIGHTED,
    return_result: bool = False,
):
    """调用 A* 求最小移动代价。

    返回 ``int``（无解为 ``-1``）；``return_result=True`` 时返回完整
    :class:`AStarResult`，其中 ``cost`` 为 ``inf`` 表示无解。
    """
    result = astar_search(
        start, goal, grid,
        heuristic_type=heuristic_type,
        cost_mode=COST_WEIGHTED,
        problem="problem2_weighted",
    )
    if return_result:
        return result
    return int(result.cost) if result.found else -1


def verify_admissible(
    grid: Sequence[Sequence[object]],
    goal: Coord,
    heuristic_type: str = HEURISTIC_WEIGHTED,
) -> Dict[str, object]:
    """遍历所有可通行格，实证检验启发函数是否可采纳（h(n) <= h*(n)）。

    返回字典包含：

    * ``min_terrain_cost`` : 地图最小进入代价 c_min
    * ``goal_entry_cost``  : 终点进入代价
    * ``checked``          : 检查的格子数
    * ``violations``       : 违反可采纳性的格子（最多列出 10 个）
    * ``admissible``       : 是否全部通过
    * ``max_slack``        : max(h*(n) - h(n))，衡量启发函数强度
    """
    grid = normalize_grid(grid)
    c_min = min_terrain_cost(grid)
    goal_cost = cell_entry_cost(grid[goal[0]][goal[1]]) or 0.0

    violations: List[Dict[str, object]] = []
    checked = 0
    max_slack = -math.inf

    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if cell_entry_cost(cell) is None:
                continue
            pos = (r, c)
            if pos == goal:
                continue
            # 真实剩余代价 h*(n)：从该格到终点的最优代价
            truth = astar_search(
                pos, goal, grid,
                heuristic_type=HEURISTIC_ZERO,   # 用 Dijkstra 得到真值
                cost_mode=COST_WEIGHTED,
            )
            if not truth.found:
                continue  # 不可达的格子不参与比较
            checked += 1
            h_value = heuristic(pos, goal, heuristic_type, grid, c_min)
            slack = truth.cost - h_value
            max_slack = max(max_slack, slack)
            if h_value > truth.cost + 1e-9:
                violations.append({
                    "pos": pos,
                    "h": h_value,
                    "true_cost": truth.cost,
                    "over_estimate": h_value - truth.cost,
                })

    return {
        "heuristic": heuristic_type,
        "min_terrain_cost": c_min,
        "goal_entry_cost": goal_cost,
        "checked": checked,
        "violations": violations[:10],
        "violation_count": len(violations),
        "admissible": not violations,
        "max_slack": max_slack if checked else 0.0,
    }


def compare_heuristics(
    start: Coord,
    goal: Coord,
    grid: Sequence[Sequence[object]],
) -> List[AStarResult]:
    """h=0 与加权曼哈顿距离的对照实验。"""
    return [
        astar_search(start, goal, grid, HEURISTIC_ZERO, COST_WEIGHTED, problem="problem2_weighted"),
        astar_search(start, goal, grid, HEURISTIC_WEIGHTED, COST_WEIGHTED, problem="problem2_weighted"),
    ]
