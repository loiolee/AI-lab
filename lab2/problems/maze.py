"""题目一：迷宫寻路。

在二维网格中用 A* 找起点 S 到终点 E 的最短路径，只能上下左右移动，
每移动一格代价为 1，``X`` 为障碍物。无解时返回空列表。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from algorithms.astar import (
    COST_UNIFORM,
    HEURISTIC_MANHATTAN,
    HEURISTIC_ZERO,
    AStarResult,
    astar_search,
    normalize_grid,
)

Coord = Tuple[int, int]

# PDF 题目一的测试用例
MAZE_SAMPLE: List[List[str]] = [
    ["S", ".", ".", "X", "."],
    [".", "X", ".", ".", "."],
    [".", "X", ".", "X", "."],
    [".", ".", ".", "X", "E"],
    ["X", "X", ".", ".", "."],
]
MAZE_SAMPLE_START: Coord = (0, 0)
MAZE_SAMPLE_GOAL: Coord = (3, 4)
MAZE_SAMPLE_EXPECTED: List[Coord] = [
    (0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4),
]


def find_cells(grid: Sequence[Sequence[object]]) -> Dict[str, Coord]:
    """扫描地图，返回 ``{"S": (r,c), "E": (r,c)}``（缺失的键不出现）。"""
    grid = normalize_grid(grid)
    found: Dict[str, Coord] = {}
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            text = str(cell).strip().upper()
            if text in ("S", "E"):
                found[text] = (r, c)
    return found


def load_maze(
    grid: Optional[Sequence[Sequence[object]]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
) -> Tuple[Coord, Coord, List[List[str]]]:
    """读取并校验迷宫输入。

    起点/终点既可以显式给出，也可以从地图中的 ``S`` / ``E`` 自动识别。
    返回 ``(start, goal, grid)``。
    """
    grid = normalize_grid(grid if grid is not None else MAZE_SAMPLE)

    if start is None or goal is None:
        marks = find_cells(grid)
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
    return start, goal, grid


def solve_maze(
    start: Coord,
    goal: Coord,
    grid: Sequence[Sequence[object]],
    heuristic_type: str = HEURISTIC_MANHATTAN,
    return_result: bool = False,
):
    """调用 A* 求解迷宫，返回最短路径的坐标列表（无解返回 ``[]``）。

    ``return_result=True`` 时返回完整的 :class:`AStarResult`（含统计信息）。
    """
    result = astar_search(
        start, goal, grid,
        heuristic_type=heuristic_type,
        cost_mode=COST_UNIFORM,
        problem="problem1_maze",
    )
    return result if return_result else result.path


def format_maze_result(path: Sequence[Coord]) -> str:
    """按 PDF 要求格式化输出：``[(0,0), (0,1), ...]``。"""
    if not path:
        return "[]"
    return "[" + ", ".join(f"({r},{c})" for r, c in path) + "]"


def render_maze(
    grid: Sequence[Sequence[object]],
    path: Optional[Sequence[Coord]] = None,
    path_mark: str = "*",
) -> str:
    """把迷宫与路径渲染成字符画，便于检查路径是否合法。"""
    grid = normalize_grid(grid)
    cells = [row[:] for row in grid]
    if path:
        for r, c in path:
            if cells[r][c] not in ("S", "E"):
                cells[r][c] = path_mark
    width = max(len(str(len(grid))) , 1)
    header = " " * (width + 1) + "".join(
        str(c).rjust(2) for c in range(len(grid[0]))
    )
    lines = [header]
    for r, row in enumerate(cells):
        lines.append(str(r).rjust(width) + " " + "".join(cell.rjust(2) for cell in row))
    return "\n".join(lines)


def compare_heuristics(
    start: Coord,
    goal: Coord,
    grid: Sequence[Sequence[object]],
) -> List[AStarResult]:
    """在同一张地图上分别运行 h=0 与 h=Manhattan，用于对照实验。"""
    return [
        astar_search(start, goal, grid, HEURISTIC_ZERO, COST_UNIFORM, problem="problem1_maze"),
        astar_search(start, goal, grid, HEURISTIC_MANHATTAN, COST_UNIFORM, problem="problem1_maze"),
    ]


def validate_path(
    path: Sequence[Coord],
    start: Coord,
    goal: Coord,
    grid: Sequence[Sequence[object]],
) -> Tuple[bool, str]:
    """校验路径合法性：连续相邻、不越界、不穿墙、首尾正确。"""
    grid = normalize_grid(grid)
    if not path:
        return False, "路径为空"
    if tuple(path[0]) != tuple(start):
        return False, f"路径起点 {path[0]} 与给定起点 {start} 不符"
    if tuple(path[-1]) != tuple(goal):
        return False, f"路径终点 {path[-1]} 与给定终点 {goal} 不符"
    for r, c in path:
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])):
            return False, f"坐标 {(r, c)} 越界"
        if str(grid[r][c]).strip().upper() in ("X", "#"):
            return False, f"坐标 {(r, c)} 是障碍物"
    for (r1, c1), (r2, c2) in zip(path, path[1:]):
        if abs(r1 - r2) + abs(c1 - c2) != 1:
            return False, f"{(r1, c1)} -> {(r2, c2)} 不是合法的一步移动"
    return True, "ok"


# ---------------------------------------------------------------------------
# 拓展实验用的迷宫生成
# ---------------------------------------------------------------------------
def is_reachable(
    grid: Sequence[Sequence[object]],
    start: Coord,
    goal: Coord,
) -> bool:
    """用 BFS 判断起点到终点是否连通（只走上下左右，不穿障碍）。"""
    grid = normalize_grid(grid)
    height, width = len(grid), len(grid[0])
    blocked = lambda r, c: str(grid[r][c]).strip().upper() in ("X", "#")

    if blocked(*start) or blocked(*goal):
        return False

    seen = {start}
    queue = [start]
    while queue:
        r, c = queue.pop()
        if (r, c) == goal:
            return True
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in seen and not blocked(nr, nc):
                seen.add((nr, nc))
                queue.append((nr, nc))
    return False


def generate_maze(
    size: int,
    obstacle_ratio: float = 0.25,
    seed: int = 0,
    max_attempts: int = 400,
) -> Tuple[Coord, Coord, List[List[str]]]:
    """生成一张"起点到终点一定连通"的随机迷宫。

    参数
    ----
    size           : 地图边长，生成 ``size x size`` 的方格
    obstacle_ratio : 障碍物比例
    seed           : 随机种子，保证结果可复现
    max_attempts   : 连通性校验失败时的最大重试次数

    返回 ``(start, goal, grid)``，起点为 ``(0, 0)``，终点为 ``(size-1, size-1)``。

    做法：按比例随机撒障碍，用 BFS 校验连通性；不连通则换下一个种子重试。
    这样得到的迷宫既有随机障碍，又保证 A* 一定有解，避免对照实验出现空结果。
    """
    import random

    if size < 3:
        raise ValueError("地图边长至少为 3")
    start: Coord = (0, 0)
    goal: Coord = (size - 1, size - 1)

    for attempt in range(max_attempts):
        rng = random.Random(seed + attempt)
        grid = [["." for _ in range(size)] for _ in range(size)]
        for r in range(size):
            for c in range(size):
                if (r, c) in (start, goal):
                    continue
                if rng.random() < obstacle_ratio:
                    grid[r][c] = "X"
        grid[start[0]][start[1]] = "S"
        grid[goal[0]][goal[1]] = "E"
        if is_reachable(grid, start, goal):
            return start, goal, grid

    raise RuntimeError(f"在 {max_attempts} 次尝试内未能生成连通的 {size}x{size} 迷宫")
