"""通用 A* 搜索算法。

一个实现同时服务三个题目，通过两个参数控制行为：

============  ==========================  ==================================
参数           取值                        适用题目
============  ==========================  ==================================
cost_mode     ``"uniform"``               题目一：每移动一格代价为 1
              ``"weighted"``              题目二 / 题目四：代价 = 进入格的权值
heuristic      ``"zero"``                  等价于 Dijkstra，作为对照基准
              ``"manhattan"``             题目一
              ``"weighted_manhattan"``    题目二 / 题目四（需可采纳性验证）
============  ==========================  ==================================

设计约定
--------
1. 邻居访问顺序固定为 上 → 下 → 左 → 右，保证实验可复现。
2. 优先队列排序键为 ``(f, -g, 入队序号)``：
   先比较 f；f 相同时优先扩展 g 更大（更靠近终点）的节点。
   这一 tie-break 不改变最优性，但能显著减少无效扩展，
   并且使题目一的输出恰好与 PDF 给出的示例路径一致。
3. 允许"重新打开"已扩展节点：若之后发现更小的 g，则该节点会再次被扩展，
   以保证在启发函数仅满足可采纳性（不保证一致性）时依然得到最优解。
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from utils.metrics import AStarStats

Coord = Tuple[int, int]

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
BLOCKED_CELLS = {"X", "#", "W"}      # 不可通过的格子
GOAL_CELL = "E"
START_CELL = "S"

# 固定的邻居访问顺序：上、下、左、右
NEIGHBOR_OFFSETS: Tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

HEURISTIC_ZERO = "zero"
HEURISTIC_MANHATTAN = "manhattan"
HEURISTIC_WEIGHTED = "weighted_manhattan"
HEURISTIC_TYPES = (HEURISTIC_ZERO, HEURISTIC_MANHATTAN, HEURISTIC_WEIGHTED)

COST_UNIFORM = "uniform"
COST_WEIGHTED = "weighted"
COST_MODES = (COST_UNIFORM, COST_WEIGHTED)


# ---------------------------------------------------------------------------
# 地图与代价
# ---------------------------------------------------------------------------
def normalize_grid(grid: Sequence[Sequence[object]]) -> List[List[str]]:
    """把地图统一成 ``List[List[str]]``，同时校验矩形性。

    支持 ``[["S", ".", ...], ...]`` 与 ``["S..X.", ...]`` 两种写法。
    """
    if not grid:
        raise ValueError("地图不能为空")
    rows: List[List[str]] = []
    for row in grid:
        if isinstance(row, str):
            rows.append(list(row))
        else:
            rows.append([str(cell).strip() for cell in row])
    width = len(rows[0])
    for idx, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"地图第 {idx} 行长度为 {len(row)}，与第 0 行的 {width} 不一致")
    return rows


def cell_entry_cost(cell: object) -> Optional[float]:
    """返回"进入该格子"需要付出的代价；障碍物返回 ``None``。

    规则（对应题目二、题目四）：

    * ``X``        —— 障碍，不可进入，返回 ``None``
    * ``E``        —— 终点，进入代价为 0（题目二明确规定）
    * ``S``/``A``/``B``/``T1``..``Tn`` —— 起点、玩家、宝藏所在格，代价为 1
    * ``.``        —— 普通通路，代价为 1
    * 数字         —— 地形权值
    """
    if cell is None:
        return None
    text = str(cell).strip().upper()
    if text in BLOCKED_CELLS:
        return None
    if text == GOAL_CELL:
        return 0.0
    if text in (".", START_CELL, "A", "B"):
        return 1.0
    # 形如 T1、T2 的宝藏格
    if text.startswith("T") and text[1:].isdigit():
        return 1.0
    try:
        return float(text)
    except ValueError:
        # 未知符号按普通通路处理，保证不因地图标注差异而崩溃
        return 1.0


def min_terrain_cost(grid: Sequence[Sequence[object]]) -> float:
    """地图中可通行格子的最小进入代价（忽略代价为 0 的终点）。

    该值是加权启发函数的构造基础：任何一步移动的代价都不小于它。
    """
    costs = []
    for row in grid:
        for cell in row:
            cost = cell_entry_cost(cell)
            if cost is not None and cost > 0:
                costs.append(cost)
    return min(costs) if costs else 1.0


def manhattan(a: Coord, b: Coord) -> int:
    """曼哈顿距离 h = |r1-r2| + |c1-c2|。"""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(pos: Coord, grid: Sequence[Sequence[object]]) -> bool:
    return 0 <= pos[0] < len(grid) and 0 <= pos[1] < len(grid[0])


def is_passable(pos: Coord, grid: Sequence[Sequence[object]]) -> bool:
    return in_bounds(pos, grid) and cell_entry_cost(grid[pos[0]][pos[1]]) is not None


# ---------------------------------------------------------------------------
# 启发函数
# ---------------------------------------------------------------------------
def heuristic(
    pos: Coord,
    goal: Coord,
    heuristic_type: str = HEURISTIC_MANHATTAN,
    grid: Optional[Sequence[Sequence[object]]] = None,
    min_cost: Optional[float] = None,
) -> float:
    """计算节点 ``pos`` 到 ``goal`` 的启发式估计值 h(n)。

    参数
    ----
    heuristic_type
        * ``"zero"``               : h = 0，退化为 Dijkstra（保证最优，用作基准）
        * ``"manhattan"``          : h = 曼哈顿距离（单位代价地图下可采纳且一致）
        * ``"weighted_manhattan"`` : h = (d-1)*c_min + cost(goal)

          推导：从 n 到 goal 至少要走 d 步，沿途进入的 d 个格子中，
          除终点外每一步的代价都 >= c_min；终点代价单独计入。
          因此 ``(d-1)*c_min + cost(goal)`` 一定不超过真实剩余代价，
          在 **c_min 由当前地图实际计算** 的前提下是可采纳的。

          题目二的终点是 ``E``（代价 0），公式退化为 ``(d-1)*c_min``，
          与框架建议的 ``max(0, Manhattan-1) * 1`` 一致。
    grid
        ``weighted_manhattan`` 需要地图来计算 c_min 与终点进入代价。
    min_cost
        预先算好的 c_min，避免每次调用都扫描地图。
    """
    if heuristic_type == HEURISTIC_ZERO:
        return 0.0

    d = manhattan(pos, goal)
    if heuristic_type == HEURISTIC_MANHATTAN:
        return float(d)

    if heuristic_type == HEURISTIC_WEIGHTED:
        if grid is None:
            raise ValueError("weighted_manhattan 启发函数需要传入 grid")
        c_min = min_cost if min_cost is not None else min_terrain_cost(grid)
        if d == 0:
            return 0.0
        goal_cost = cell_entry_cost(grid[goal[0]][goal[1]]) or 0.0
        return float((d - 1) * c_min + goal_cost)

    raise ValueError(f"未知启发函数类型: {heuristic_type!r}，可选 {HEURISTIC_TYPES}")


# ---------------------------------------------------------------------------
# 邻居与移动代价
# ---------------------------------------------------------------------------
def get_neighbors(
    pos: Coord,
    grid: Sequence[Sequence[object]],
) -> List[Coord]:
    """返回 ``pos`` 所有合法相邻格子（上、下、左、右，顺序固定）。

    不越界、不进入障碍物。注意：**不**排除已经访问过的节点，
    是否值得再次扩展由 A* 内部根据 g 值判断。
    """
    neighbors: List[Coord] = []
    for d_row, d_col in NEIGHBOR_OFFSETS:
        nxt = (pos[0] + d_row, pos[1] + d_col)
        if is_passable(nxt, grid):
            neighbors.append(nxt)
    return neighbors


def get_move_cost(
    current: Coord,
    neighbor: Coord,
    grid: Sequence[Sequence[object]],
    cost_mode: str = COST_UNIFORM,
) -> float:
    """返回从 ``current`` 移动到 ``neighbor`` 的实际代价。

    * ``uniform``  : 固定为 1（题目一）
    * ``weighted`` : 等于**进入** ``neighbor`` 的地形权值（题目二 / 题目四）
    """
    if cost_mode == COST_UNIFORM:
        return 1.0
    if cost_mode == COST_WEIGHTED:
        cost = cell_entry_cost(grid[neighbor[0]][neighbor[1]])
        if cost is None:
            raise ValueError(f"试图进入障碍物: {neighbor}")
        return cost
    raise ValueError(f"未知代价模式: {cost_mode!r}，可选 {COST_MODES}")


# ---------------------------------------------------------------------------
# 路径恢复
# ---------------------------------------------------------------------------
def reconstruct_path(came_from: Dict[Coord, Coord], goal: Coord) -> List[Coord]:
    """根据前驱表恢复从起点到终点的完整路径（起点在前）。"""
    if goal not in came_from:
        return [goal]
    path = [goal]
    node = goal
    while node in came_from:
        node = came_from[node]
        path.append(node)
    path.reverse()
    return path


# ---------------------------------------------------------------------------
# 搜索结果
# ---------------------------------------------------------------------------
@dataclass
class AStarResult:
    """A* 的完整返回结果。"""

    path: List[Coord] = field(default_factory=list)
    cost: float = math.inf
    stats: AStarStats = field(default_factory=AStarStats)
    # 实际被扩展的节点集合（按扩展顺序），供实验报告绘制搜索区域使用
    expanded: List[Coord] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.path)

    def __bool__(self) -> bool:  # 允许 `if result:` 判断是否有解
        return self.found


# ---------------------------------------------------------------------------
# 主算法
# ---------------------------------------------------------------------------
def astar_search(
    start: Coord,
    goal: Coord,
    grid: Sequence[Sequence[object]],
    heuristic_type: str = HEURISTIC_MANHATTAN,
    cost_mode: str = COST_UNIFORM,
    stats: Optional[AStarStats] = None,
    problem: str = "",
) -> AStarResult:
    """A* 搜索。

    参数
    ----
    start, goal   : 起点 / 终点坐标 ``(row, col)``
    grid          : 二维地图
    heuristic_type: 见 :func:`heuristic`
    cost_mode     : ``"uniform"`` 或 ``"weighted"``
    stats         : 可选的外部统计对象（便于批量实验复用）
    problem       : 仅用于统计标签

    返回
    ----
    :class:`AStarResult`，包含路径、总代价与统计信息。
    无解时 ``path == []``、``cost == inf``。

    实现要点
    --------
    * ``g_score`` 记录起点到各节点的已知最小实际代价；
    * ``came_from`` 记录前驱，用于恢复路径；
    * Open 表使用 ``heapq``，键为 ``(f, -g, seq)``；
    * 出队时若该条记录的 g 大于当前最优 g，说明是**过期节点**，直接丢弃；
    * 允许重新打开：只要发现更小的 g，就重新入队并可能再次扩展。
    """
    grid = normalize_grid(grid)
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    if stats is None:
        stats = AStarStats()
    stats.algorithm = "A*"
    stats.problem = problem
    stats.heuristic = heuristic_type
    stats.cost_mode = cost_mode

    if not is_passable(start, grid) or not is_passable(goal, grid):
        stats.execution_time = 0.0
        stats.found = False
        stats.total_cost = math.inf
        return AStarResult(path=[], cost=math.inf, stats=stats, expanded=[])

    min_cost = min_terrain_cost(grid) if heuristic_type == HEURISTIC_WEIGHTED else None
    h_start = heuristic(start, goal, heuristic_type, grid, min_cost)

    t0 = time.perf_counter()

    g_score: Dict[Coord, float] = {start: 0.0}
    came_from: Dict[Coord, Coord] = {}
    expanded_g: Dict[Coord, float] = {}          # 记录每个节点被扩展时的 g，用于判断是否值得重开

    # 堆元素: (f, -g, seq, node)
    open_heap: List[Tuple[float, float, int, Coord]] = [(h_start, 0.0, 0, start)]
    seq = 1

    stats.visited_nodes = 1
    stats.expanded_nodes = 0
    stats.generated_nodes = 0
    stats.found = False

    final_cost = math.inf
    path: List[Coord] = []
    expanded_order: List[Coord] = []

    while open_heap:
        f_value, neg_g, _, node = heapq.heappop(open_heap)
        g_value = -neg_g

        # 1) 过期节点：同一节点在堆中可能存在多条记录，只保留 g 最小的那条
        if g_value > g_score.get(node, math.inf):
            continue
        # 2) 已经以不大于当前 g 的代价扩展过，跳过（否则允许重新打开）
        if node in expanded_g and g_value >= expanded_g[node]:
            continue

        expanded_g[node] = g_value
        stats.expanded_nodes += 1
        expanded_order.append(node)

        if node == goal:
            final_cost = g_value
            path = reconstruct_path(came_from, goal)
            stats.found = True
            break

        for neighbor in get_neighbors(node, grid):
            step_cost = get_move_cost(node, neighbor, grid, cost_mode)
            new_g = g_value + step_cost
            stats.generated_nodes += 1

            if new_g < g_score.get(neighbor, math.inf):
                g_score[neighbor] = new_g
                came_from[neighbor] = node
                if neighbor not in expanded_g:
                    stats.visited_nodes += 1
                h_value = heuristic(neighbor, goal, heuristic_type, grid, min_cost)
                heapq.heappush(open_heap, (new_g + h_value, -new_g, seq, neighbor))
                seq += 1

    stats.execution_time = time.perf_counter() - t0
    stats.path_length = max(0, len(path) - 1)
    stats.total_cost = final_cost if stats.found else math.inf

    return AStarResult(path=path, cost=final_cost, stats=stats, expanded=expanded_order)
