"""实验报告图表生成。

本模块只做一件事：把算法跑出来的**真实数据**画成可以直接放进报告的中文图表。
所有图表的编号与报告正文一致（图 1-1 ~ 图 5-2）。

数据来源
--------
* 数据类图表（柱状图、折线图、热力图）调用 :mod:`utils.experiments` 与
  各题目的求解函数，与 ``main.py experiments`` 写出的 CSV 使用同一段代码。
* 示意图（流程图、博弈树、剪枝过程）由本模块按真实搜索结果绘制，
  其中剪枝位置、alpha/beta 数值直接取自 :func:`algorithms.alpha_beta.alpha_beta_tree`
  的 ``trace`` 记录，不是手工编造。

依赖 ``matplotlib``；中文字体由 :func:`utils.visualization.setup_chinese_font` 配置。
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from algorithms.alpha_beta import (
    ORDER_LEFT_TO_RIGHT,
    alpha_beta_tree,
)
from algorithms.astar import (
    COST_UNIFORM,
    COST_WEIGHTED,
    HEURISTIC_MANHATTAN,
    HEURISTIC_WEIGHTED,
    HEURISTIC_ZERO,
    astar_search,
    cell_entry_cost,
)
from problems.game_tree import build_mirrored_tree
from problems.maze import load_maze
from problems.treasure_game import default_game
from problems.weighted_terrain import load_weighted_map
from utils.data_structures import TreeNode
from utils.experiments import (
    astar_experiments,
    astar_scaling_experiments,
    game_experiments,
    tree_order_experiments,
)
from utils.visualization import _pyplot, setup_chinese_font

# 统一配色
C_MAX = "#E8913A"        # MAX 节点
C_MIN = "#4A90D9"        # MIN 节点
C_PATH = "#3FA45B"       # 路径
C_VISITED = "#C9DDF3"    # 已扩展区域
C_BLOCK = "#3A3F47"      # 障碍
C_GOAL = "#C0392B"       # 终点
C_TEXT = "#1A1A1A"
C_GRAY = "#8A8F98"
C_PRUNE = "#B0B4BA"

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "results" / "report_figures"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


# ---------------------------------------------------------------------------
# 基础设施
# ---------------------------------------------------------------------------
def _plt():
    plt = _pyplot()
    if plt is None:
        raise RuntimeError("绘图需要 matplotlib，请先执行 pip install matplotlib")
    setup_chinese_font()
    return plt


def _save(fig, path: Path, tight: bool = True) -> Path:
    plt = _plt()
    path.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return path


def _new_figure(width: float = 7.0, height: float = 4.0):
    plt = _plt()
    fig, ax = plt.subplots(figsize=(width, height))
    return fig, ax


def _box(
    ax,
    cx: float,
    cy: float,
    w: float,
    h: float,
    text: str,
    fc: str = "#EAF1FB",
    ec: str = "#2E6FD9",
    fs: float = 8.5,
    tc: str = C_TEXT,
    bold: bool = False,
) -> None:
    """画一个圆角矩形流程框。"""
    from matplotlib.patches import FancyBboxPatch

    patch = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.015,rounding_size=0.10",
        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=3, linespacing=1.45)


def _arrow(ax, p0, p1, color: str = "#5A6068", lw: float = 1.2,
           label: Optional[str] = None, label_dx: float = 0.0, label_dy: float = 0.0,
           fs: float = 7.5) -> None:
    """直线箭头，可在中点标注文字。"""
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=1.5, shrinkB=1.5, mutation_scale=11),
                zorder=1)
    if label:
        mx = (p0[0] + p1[0]) / 2 + label_dx
        my = (p0[1] + p1[1]) / 2 + label_dy
        ax.text(mx, my, label, fontsize=fs, color=color, ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=4)


def _elbow(ax, points: Sequence[Tuple[float, float]], color: str = "#5A6068",
           lw: float = 1.2, label: Optional[str] = None,
           label_xy: Optional[Tuple[float, float]] = None, fs: float = 7.5) -> None:
    """折线箭头（用于回边）。"""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if len(points) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, zorder=1,
                solid_capstyle="round", solid_joinstyle="round")
    ax.annotate("", xy=points[-1], xytext=points[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=1.0, shrinkB=1.5, mutation_scale=11),
                zorder=1)
    if label and label_xy:
        ax.text(label_xy[0], label_xy[1], label, fontsize=fs, color=color,
                ha="center", va="center",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2), zorder=4)


def _clean_axis(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)


# ---------------------------------------------------------------------------
# 图 1-1：A* 算法流程图
# ---------------------------------------------------------------------------
def fig_1_1_astar_flowchart(path: Path) -> Path:
    fig, ax = _new_figure(6.4, 5.6)
    _clean_axis(ax)

    w, h, cx = 4.2, 0.62, 4.0
    rows = [
        (9.55, "开始", "#DCEBFB", "#2E6FD9"),
        (8.80, "初始化：g(S) = 0，Open = {S}，Closed 为空", "#EAF1FB", "#2E6FD9"),
        (8.05, "Open 为空？", "#FDF1E3", "#D98324"),
        (7.30, "取 f 值最小的节点 n，移入 Closed", "#EAF1FB", "#2E6FD9"),
        (6.55, "n 是终点？", "#FDF1E3", "#D98324"),
        (5.60, "取一个邻居 m：\n tentative_g = g(n) + cost(n, m)", "#EAF1FB", "#2E6FD9"),
        (4.65, "tentative_g < g(m)？", "#FDF1E3", "#D98324"),
        (3.70, "更新 g(m)、f(m) 与前驱，m 入 Open", "#EAF1FB", "#2E6FD9"),
        (2.75, "还有未处理的邻居？", "#FDF1E3", "#D98324"),
    ]
    for y, text, fc, ec in rows:
        _box(ax, cx, y, w, h, text, fc=fc, ec=ec)

    for i in range(len(rows) - 1):
        _arrow(ax, (cx, rows[i][0] - h / 2), (cx, rows[i + 1][0] + h / 2))

    # 右分支：无解
    _box(ax, 8.45, 8.05, 2.5, 0.66, "返回空列表（无解）", fc="#F6E3E1", ec=C_GOAL, fs=7.8)
    _arrow(ax, (cx + w / 2, 8.05), (7.20, 8.05), label="是", label_dy=0.20)

    # 右分支：找到终点
    _box(ax, 8.45, 6.55, 2.5, 0.66, "由 came_from 回溯\n并输出路径", fc="#E3F2E7", ec=C_PATH, fs=7.8)
    _arrow(ax, (cx + w / 2, 6.55), (7.20, 6.55), label="是", label_dy=0.20)

    # 右分支：代价未变优
    _box(ax, 8.45, 4.65, 2.5, 0.66, "跳过该邻居", fc="#EFEFEF", ec=C_GRAY, fs=7.8)
    _arrow(ax, (cx + w / 2, 4.65), (7.20, 4.65), label="否", label_dy=0.20)

    # 内层回边：还有邻居 -> 继续处理邻居
    _elbow(ax, [(cx - w / 2, 2.75), (1.05, 2.75), (1.05, 5.60), (cx - w / 2, 5.60)],
           label="是", label_xy=(1.05, 4.18))

    # 外层回边：邻居处理完 -> 回到 Open 判空
    _elbow(ax, [(cx, 2.75 - h / 2), (cx, 1.85), (0.30, 1.85), (0.30, 8.05), (cx - w / 2, 8.05)],
           color="#2E6FD9", label="否", label_xy=(0.30, 5.20))

    # 跳过该邻居 -> 回到"还有未处理的邻居"
    _elbow(ax, [(8.45, 4.65 - 0.33), (8.45, 2.75), (cx + w / 2, 2.75)], color=C_GRAY)

    ax.text(cx, 1.35, "循环执行，直到找到终点或 Open 为空", fontsize=7.5,
            color=C_GRAY, ha="center")
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 1-2：Minimax 与 Alpha-Beta 递归流程图
# ---------------------------------------------------------------------------
def fig_1_2_search_flowchart(path: Path) -> Path:
    fig, ax = _new_figure(6.6, 4.8)
    _clean_axis(ax)

    w, h, cx = 3.6, 0.62, 4.85
    _box(ax, cx, 9.55, w, h, "进入节点 n（深度 d）", fc="#DCEBFB", ec="#2E6FD9", bold=True)
    _box(ax, cx, 8.70, w, h, "d = H 或 已无剩余宝藏？", fc="#FDF1E3", ec="#D98324")
    _box(ax, 8.60, 8.70, 2.4, 0.66, "返回评价值 U", fc="#E3F2E7", ec=C_PATH, fs=7.8)
    _box(ax, cx, 7.85, w, h, "生成全部合法行动并按估值排序", fc="#EAF1FB", ec="#2E6FD9")
    _box(ax, cx, 7.00, w, h, "当前是 MAX 玩家？", fc="#FDF1E3", ec="#D98324")

    _box(ax, 1.85, 5.95, 3.3, 0.86, "v = max(v, 子节点值)\nα = max(α, v)", fc="#FBEBDD", ec=C_MAX, fs=7.8)
    _box(ax, 7.85, 5.95, 3.3, 0.86, "v = min(v, 子节点值)\nβ = min(β, v)", fc="#E4EEF9", ec=C_MIN, fs=7.8)

    _box(ax, cx, 4.95, w, h, "α ≥ β ？", fc="#FDF1E3", ec="#D98324")
    _box(ax, cx, 4.05, w, h, "返回 v 给父节点", fc="#E3F2E7", ec=C_PATH)
    _box(ax, 8.60, 4.05, 2.4, 0.66, "剪掉剩余子节点", fc="#F6E3E1", ec=C_GOAL, fs=7.8)

    _arrow(ax, (cx, 9.24), (cx, 9.01))
    _arrow(ax, (cx + w / 2, 8.70), (7.40, 8.70), label="是", label_dy=0.20)
    _arrow(ax, (cx, 8.39), (cx, 8.16), label="否", label_dx=0.26)
    _arrow(ax, (cx, 7.54), (cx, 7.31))
    _arrow(ax, (cx - 0.50, 6.69), (1.85, 6.38), label="是", label_dx=-0.24, label_dy=0.10)
    _arrow(ax, (cx + 0.50, 6.69), (7.85, 6.38), label="否", label_dx=0.24, label_dy=0.10)
    _arrow(ax, (1.85, 5.52), (4.05, 5.26))
    _arrow(ax, (7.85, 5.52), (5.65, 5.26))
    _elbow(ax, [(cx + w / 2, 4.95), (8.60, 4.95), (8.60, 4.38)],
           label="是", label_xy=(7.95, 5.12))
    _arrow(ax, (cx, 4.64), (cx, 4.36), label="否", label_dx=0.26)
    _arrow(ax, (7.40, 4.05), (cx + w / 2, 4.05), color=C_GOAL)

    ax.text(cx, 3.10,
            "对每个子节点重复上述过程（递归调用），全部处理完后返回 v",
            fontsize=7.5, color=C_GRAY, ha="center")
    ax.text(cx, 2.55,
            "Minimax 与 Alpha-Beta 的差别只在于是否维护 α、β\n以及是否执行 α ≥ β 的剪枝判断",
            fontsize=7.5, color=C_GRAY, ha="center")
    ax.set_ylim(2.20, 10.0)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 2-1：迷宫最短路径与扩展区域
# ---------------------------------------------------------------------------
def fig_2_1_maze_path(path: Path) -> Path:
    from matplotlib.patches import Patch

    start, goal, grid = load_maze()
    result = astar_search(start, goal, grid, HEURISTIC_MANHATTAN, COST_UNIFORM)
    expanded = set(result.expanded)
    on_path = set(result.path)

    height, width = len(grid), len(grid[0])
    fig, ax = _new_figure(4.2, 4.2)

    for r in range(height):
        for c in range(width):
            cell = str(grid[r][c])
            if cell == "X":
                fc = C_BLOCK
            elif (r, c) in on_path:
                fc = C_PATH
            elif (r, c) in expanded:
                fc = C_VISITED
            else:
                fc = "#FFFFFF"
            ax.add_patch(plt_rect(c, r, fc))

    # 路径连线
    xs = [c + 0.5 for r, c in result.path]
    ys = [r + 0.5 for r, c in result.path]
    ax.plot(xs, ys, color="#1F6B36", lw=1.8, zorder=4, alpha=0.85)

    for (r1, c1), (r2, c2) in zip(result.path, result.path[1:]):
        ax.annotate("", xy=(c2 + 0.5, r2 + 0.5), xytext=(c1 + 0.5, r1 + 0.5),
                    arrowprops=dict(arrowstyle="-|>", color="#1F6B36", lw=1.4,
                                    mutation_scale=9), zorder=4)

    import matplotlib.patheffects as pe
    for (r, c), letter in ((start, "S"), (goal, "E")):
        ax.text(c + 0.5, r + 0.5, letter, ha="center", va="center",
                fontsize=12, color="white", fontweight="bold", zorder=7,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="#1F6B36")])

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([i + 0.5 for i in range(width)])
    ax.set_xticklabels(range(width), fontsize=8)
    ax.set_yticks([i + 0.5 for i in range(height)])
    ax.set_yticklabels(range(height), fontsize=8)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#CCCCCC")

    legend = [
        Patch(facecolor=C_BLOCK, label="障碍物 X"),
        Patch(facecolor=C_VISITED, label="已扩展节点"),
        Patch(facecolor=C_PATH, label="最短路径"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.06),
              ncol=3, fontsize=8, frameon=False)
    ax.set_title(f"最短路径 {len(result.path) - 1} 步，扩展 {result.stats.expanded_nodes} 个节点",
                 fontsize=9.5, color=C_TEXT)
    return _save(fig, path)


def plt_rect(col: int, row: int, color: str):
    """生成一个方格矩形（供网格图使用）。"""
    from matplotlib.patches import Rectangle

    return Rectangle((col, row), 1, 1, facecolor=color, edgecolor="#D5D8DD", linewidth=0.8)


# ---------------------------------------------------------------------------
# 图 2-2：A* 启发函数对照（节点数 + 运行时间）
# ---------------------------------------------------------------------------
def fig_2_2_astar_heuristic(path: Path, rows: Optional[List[Dict[str, Any]]] = None) -> Path:
    plt = _plt()
    rows = rows or astar_experiments()

    labels = ["A1\n迷宫\nh=0", "A2\n迷宫\nManhattan", "A3\n加权\n加权启发", "A4\n加权\nh=0"]
    nodes = [row["expanded_nodes"] for row in rows]
    visited = [row["visited_nodes"] for row in rows]
    colors = ["#9BB7D4", "#2E6FD9", "#9BB7D4", "#2E6FD9"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    for ax, values, ylabel, title, fmt in (
        (axes[0], nodes, "扩展节点数", "(a) 扩展节点数", "{:.0f}"),
        (axes[1], visited, "生成节点数", "(b) 生成节点数", "{:.0f}"),
    ):
        bars = ax.bar(range(len(values)), values, color=colors, width=0.6)
        ax.bar_label(bars, labels=[fmt.format(v) for v in values], fontsize=8, padding=2)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(values) * 1.22)

    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 2-3：加权地形与最优路径
# ---------------------------------------------------------------------------
def fig_2_3_weighted_terrain(path: Path) -> Path:
    start, goal, grid = load_weighted_map()
    result = astar_search(start, goal, grid, HEURISTIC_WEIGHTED, COST_WEIGHTED)

    height, width = len(grid), len(grid[0])
    costs = []
    for row in grid:
        line = []
        for cell in row:
            value = cell_entry_cost(cell)
            line.append(math.nan if value is None else value)
        costs.append(line)

    fig, ax = _new_figure(4.8, 4.1)
    masked = [[0 if v != v else v for v in line] for line in costs]
    im = ax.imshow(masked, cmap="YlOrRd", vmin=0, vmax=8)

    # 数字放在格子左上角，路径沿格心连线，两者不会互相遮挡；
    # 深色格用白字、浅色格用深字，保证任何底色上都读得出来。
    for r in range(height):
        for c in range(width):
            if math.isnan(costs[r][c]):
                ax.add_patch(plt_rect(c - 0.5, r - 0.5, C_BLOCK))
                ax.text(c, r, "X", ha="center", va="center", color="white", fontsize=10)
            else:
                value = costs[r][c]
                ax.text(c - 0.40, r - 0.36, f"{value:.0f}", ha="left", va="top",
                        fontsize=8, color="white" if value >= 4 else "#3A3A3A")

    # 单元格描边，避免同色相邻格连成一片
    for r in range(1, height):
        ax.axhline(r - 0.5, color="white", lw=0.8, alpha=0.85, zorder=2)
    for c in range(1, width):
        ax.axvline(c - 0.5, color="white", lw=0.8, alpha=0.85, zorder=2)

    xs = [c for r, c in result.path]
    ys = [r for r, c in result.path]
    ax.plot(xs, ys, color="#1F4E9C", lw=2.2, zorder=4, solid_capstyle="round")

    # 起点/终点做成实心圆牌，深色底 + 白字，放在格心压在路径之上
    for (r, c), letter, color in ((start, "S", "#1F4E9C"), (goal, "E", C_GOAL)):
        ax.scatter([c], [r], s=210, color=color, edgecolor="white", linewidth=1.2, zorder=6)
        ax.text(c, r, letter, ha="center", va="center", fontsize=9.5,
                color="white", fontweight="bold", zorder=7)

    ax.set_xticks(range(width))
    ax.set_xticklabels(range(width), fontsize=8)
    ax.set_yticks(range(height))
    ax.set_yticklabels(range(height), fontsize=8)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("进入代价", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax.set_title(f"最优路径总代价 {result.cost:.0f}（终点 E 进入代价为 0）", fontsize=9.5)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 3-1：Alpha-Beta 剪枝过程
# ---------------------------------------------------------------------------
def _build_display_tree() -> TreeNode:
    """报告图 3-1 使用的博弈树：结构、取值与给定树相同，但**不复用**子树对象。

    复用会导致同一节点对象出现在树的两个位置，``id(node)`` 做键的逐节点
    轨迹会互相覆盖。具体构造见 :func:`problems.game_tree.build_mirrored_tree`。
    """
    return build_mirrored_tree()


def _layout_tree(root: TreeNode):
    """计算树形布局：叶子等间距，内部节点取子节点横坐标均值。

    返回 ``(positions, depths)``，键均为 ``id(node)``。
    """
    positions: Dict[int, Tuple[float, float]] = {}
    depths: Dict[int, int] = {}
    counter = [0.0]

    def walk(node: TreeNode, depth: int) -> float:
        if node.is_leaf:
            x = counter[0]
            counter[0] += 1.0
        else:
            xs = [walk(child, depth + 1) for child in node.children]
            x = sum(xs) / len(xs)
        positions[id(node)] = (x, -float(depth))
        depths[id(node)] = depth
        return x

    walk(root, 0)
    return positions, depths


def _iter_nodes(node: TreeNode):
    yield node
    for child in node.children:
        yield from _iter_nodes(child)


def _draw_tree(ax, root: TreeNode, positions, depths, trace,
               node_radius: float = 0.28) -> None:
    """绘制博弈树：MAX/MIN 用圆点区分，被剪枝的叶子用虚线灰框表示。"""
    from matplotlib.patches import Circle

    pruned_ids = set()
    for info in trace.values():
        pruned_ids.update(info.get("pruned_children", []))

    def edges(node: TreeNode):
        for child in node.children:
            yield node, child
            yield from edges(child)

    for parent, child in edges(root):
        x0, y0 = positions[id(parent)]
        x1, y1 = positions[id(child)]
        pruned = id(child) in pruned_ids
        ax.plot([x0, x1], [y0, y1],
                color=C_PRUNE if pruned else "#7A8088",
                lw=1.2, ls="--" if pruned else "-", zorder=1)

    for node in _iter_nodes(root):
        x, y = positions[id(node)]
        pruned = id(node) in pruned_ids
        if node.is_leaf:
            color = "#FFFFFF" if pruned else "#EFF3F8"
            edge = C_PRUNE if pruned else "#5A6068"
            ax.add_patch(plt_rect_square(x, y, 0.52, color, edge, pruned))
            ax.text(x, y, f"{node.value:.0f}", ha="center", va="center",
                    fontsize=9, color=C_PRUNE if pruned else C_TEXT, fontweight="bold")
            if pruned:
                ax.text(x, y - 0.48, "被剪枝", ha="center", va="center",
                        fontsize=6.8, color=C_GOAL)
        else:
            is_max = (depths[id(node)] % 2 == 0)
            ax.add_patch(Circle((x, y), node_radius,
                                facecolor=C_MAX if is_max else C_MIN,
                                edgecolor="white", linewidth=1.2, zorder=3))
            ax.text(x, y, "MAX" if is_max else "MIN", ha="center", va="center",
                    fontsize=6.8, color="white", fontweight="bold", zorder=4)
            info = trace.get(id(node), {})
            value = info.get("value")
            if value is not None:
                ax.text(x, y - node_radius - 0.16, f"v={value:.0f}", ha="center",
                        va="top", fontsize=8, color=C_TEXT)


def plt_rect_square(cx: float, cy: float, size: float, fc: str, ec: str, dashed: bool = False):
    from matplotlib.patches import Rectangle

    return Rectangle((cx - size / 2, cy - size / 2), size, size,
                     facecolor=fc, edgecolor=ec, linewidth=1.1, zorder=3,
                     linestyle="--" if dashed else "-")


def fig_3_1_alpha_beta_tree(path: Path) -> Path:
    from matplotlib.patches import Patch

    root = _build_display_tree()
    trace: Dict[int, Dict[str, Any]] = {}
    value = alpha_beta_tree(root, order=ORDER_LEFT_TO_RIGHT, trace=trace)

    positions, depths = _layout_tree(root)
    fig, ax = _new_figure(6.8, 3.9)
    ax.set_axis_off()

    _draw_tree(ax, root, positions, depths, trace)

    # 在发生剪枝的节点上标注剪枝那一刻的 alpha / beta
    notes = []
    for node in _iter_nodes(root):
        info = trace.get(id(node))
        if not info or node.is_leaf:
            continue
        if info.get("pruned_children"):
            a_prune = info.get("alpha_at_prune", info["alpha_in"])
            b_prune = info.get("beta_at_prune", info["beta_in"])
            x, y = positions[id(node)]
            notes.append((x, y, f"α={a_prune:.0f} ≥ β={b_prune:.0f}\n触发剪枝"))
    for x, y, text in notes:
        ax.annotate(text, xy=(x, y), xytext=(x + 0.55, y + 0.62),
                    fontsize=7.5, color=C_GOAL, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color=C_GOAL, lw=0.8,
                                    connectionstyle="arc3,rad=0.2"))

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.6)
    ax.set_ylim(min(ys) - 1.05, max(ys) + 0.85)

    # 层级标注
    for depth, label in ((0, "MAX"), (1, "MIN"), (2, "MAX"), (3, "叶子")):
        ax.text(min(xs) - 0.95, -float(depth), label, fontsize=8,
                color=C_GRAY, ha="right", va="center")

    legend = [
        Patch(facecolor=C_MAX, label="MAX 节点"),
        Patch(facecolor=C_MIN, label="MIN 节点"),
        Patch(facecolor="#EFF3F8", edgecolor="#5A6068", label="叶子（已访问）"),
        Patch(facecolor="#FFFFFF", edgecolor=C_PRUNE, linestyle="--", label="叶子（被剪枝）"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=7.5, frameon=False, ncol=2)
    ax.set_title(f"Alpha-Beta 剪枝过程：根节点最优值 v = {value:.0f}", fontsize=10)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 3-2：不同访问顺序的搜索开销
# ---------------------------------------------------------------------------
def fig_3_2_tree_order(path: Path, order_data: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Path:
    from matplotlib.ticker import MaxNLocator

    plt = _plt()
    order_data = order_data or tree_order_experiments()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    panels = [
        (axes[0], order_data["pdf_tree"], "(a) 题三给定博弈树（8 个叶子）"),
        (axes[1], order_data["random_tree"], "(b) 3 叉 4 层随机博弈树"),
    ]

    labels = [row["order_label"].replace(" ", "\n", 1) for row in order_data["pdf_tree"]]
    for ax, rows, title in panels:
        x = range(len(rows))
        total = [row["total_nodes"] for row in rows]
        leaf = [row["leaf_nodes"] for row in rows]
        width = 0.36
        b1 = ax.bar([i - width / 2 for i in x], total, width, label="访问节点总数", color="#2E6FD9")
        b2 = ax.bar([i + width / 2 for i in x], leaf, width, label="其中叶子节点", color="#E8913A")
        ax.bar_label(b1, fontsize=7, padding=1)
        ax.bar_label(b2, fontsize=7, padding=1)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("节点数", fontsize=9)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(total) * 1.25)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 4-1：六个关键位置的距离矩阵
# ---------------------------------------------------------------------------
def fig_4_1_distance_heatmap(path: Path) -> Path:
    game = default_game()
    labels = game.labels
    matrix = game.matrix_rows()

    fig, ax = _new_figure(4.6, 3.8)
    data = [[0 if v == math.inf else v for v in row] for row in matrix]
    im = ax.imshow(data, cmap="Blues", vmin=0)

    for i, row in enumerate(data):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if value > max(max(r) for r in data) * 0.55 else "#1A1A1A")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("终点", fontsize=9)
    ax.set_ylabel("起点", fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("最小移动代价", fontsize=8.5)
    cbar.ax.tick_params(labelsize=7.5)
    ax.set_title("六个关键位置的最小移动代价", fontsize=9.5)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 4-2：宝藏分配的博弈树（H=2）
# ---------------------------------------------------------------------------
def fig_4_2_treasure_tree(path: Path) -> Path:
    from matplotlib.patches import FancyBboxPatch

    game = default_game()
    state = game.initial_state()
    actions = game.actions(state)

    best_target, best_value = None, -math.inf
    branches = []
    for action in actions:
        after_a = game.apply(state, action)
        gain_a = game.gain(state.pos_a, action)
        best_reply, best_u = None, math.inf
        for reply in game.actions(after_a):
            after_b = game.apply(after_a, reply)
            u = game.evaluate(after_b)
            if u < best_u:
                best_u, best_reply = u, reply
        branches.append((action, gain_a, best_reply, best_u))
        if best_u > best_value:
            best_value, best_target = best_u, action

    fig, ax = _new_figure(6.8, 3.7)
    ax.set_axis_off()
    ax.set_xlim(-0.8, 5.6)
    ax.set_ylim(-1.75, 1.60)

    root_xy = (2.2, 1.28)
    ax.add_patch(FancyBboxPatch((root_xy[0] - 0.62, root_xy[1] - 0.20), 1.24, 0.40,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=C_MAX, edgecolor="white", lw=1.2, zorder=3))
    ax.text(*root_xy, "A 选择宝藏", ha="center", va="center", fontsize=8.5,
            color="white", fontweight="bold", zorder=4)

    xs = [0.25, 1.55, 2.85, 4.15]
    for x, (action, gain_a, reply, u) in zip(xs, branches):
        # A 的决策节点：净收益 = 宝藏价值 - 移动代价
        ax.plot([root_xy[0], x], [root_xy[1] - 0.20, 0.62], color="#7A8088", lw=1.1, zorder=1)
        ax.add_patch(FancyBboxPatch((x - 0.50, 0.36), 1.00, 0.50,
                                    boxstyle="round,pad=0.02,rounding_size=0.07",
                                    facecolor="#F7D9BE" if action == best_target else "#FDF1E3",
                                    edgecolor=C_MAX, lw=1.0, zorder=3))
        ax.text(x, 0.72, f"取 {action}", ha="center", va="center",
                fontsize=8, color=C_TEXT, zorder=4)
        ax.text(x, 0.49, f"净收益 {gain_a:+.0f}", ha="center", va="center",
                fontsize=7.5, color="#8A5A20", zorder=4)

        # B 的应对
        ax.plot([x, x], [0.36, -0.12], color="#7A8088", lw=1.1, zorder=1)
        ax.add_patch(FancyBboxPatch((x - 0.50, -0.38), 1.00, 0.50,
                                    boxstyle="round,pad=0.02,rounding_size=0.07",
                                    facecolor="#E4EEF9", edgecolor=C_MIN, lw=1.0, zorder=3))
        ax.text(x, -0.02, f"B 取 {reply}", ha="center", va="center",
                fontsize=8, color=C_TEXT, zorder=4)
        ax.text(x, -0.25, f"净收益 {game.gain(state.pos_b, reply):+.0f}",
                ha="center", va="center", fontsize=7.5, color="#2B5C8A", zorder=4)

        # 终局 U
        color = C_GOAL if action == best_target else "#5A6068"
        weight = "bold" if action == best_target else "normal"
        ax.plot([x, x], [-0.38, -0.78], color="#7A8088", lw=1.1, zorder=1)
        ax.text(x, -0.94, f"U = {u:+.0f}", ha="center", va="center",
                fontsize=9, color=color, fontweight=weight, zorder=4)

    ax.text(2.2, -1.45,
            f"H=2 时 A 的最优首步为 {best_target}，终局 U = {best_value:+.0f}；"
            f"其余三个首步的终局 U 都更低",
            ha="center", va="center", fontsize=8, color=C_GRAY)
    ax.set_title("宝藏分配博弈树（A 为 MAX，B 为 MIN，深度 H=2）", fontsize=10)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 4-3：Minimax 与 Alpha-Beta 节点数对比
# ---------------------------------------------------------------------------
def fig_4_3_minimax_vs_ab(
    path: Path,
    minimax_rows: Optional[List[Dict[str, Any]]] = None,
    alpha_rows: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    plt = _plt()
    if minimax_rows is None or alpha_rows is None:
        minimax_rows, alpha_rows = game_experiments()

    depths = ["H=2", "H=4"]
    mm = [next(r for r in minimax_rows if r["depth"] == d)["total_nodes"] for d in (2, 4)]
    ab = [next(r for r in alpha_rows if r["depth"] == d and r["order"] == ORDER_LEFT_TO_RIGHT)["total_nodes"]
          for d in (2, 4)]

    fig, ax = _new_figure(5.4, 3.3)
    x = range(len(depths))
    width = 0.30
    b1 = ax.bar([i - width / 2 for i in x], mm, width, label="Minimax", color="#7D5BA6")
    b2 = ax.bar([i + width / 2 for i in x], ab, width, label="Alpha-Beta", color="#2E6FD9")
    ax.bar_label(b1, fontsize=8, padding=2)
    ax.bar_label(b2, fontsize=8, padding=2)

    # 降幅标注统一放在图顶同一水平线上，与柱顶数值彻底分开
    top_line = max(mm) * 1.17
    for i, (a, b) in enumerate(zip(mm, ab)):
        ax.annotate(f"减少 {(a - b) / a * 100:.0f}%", xy=(i, top_line),
                    ha="center", fontsize=8, color=C_GOAL)

    ax.set_xticks(list(x))
    ax.set_xticklabels(depths, fontsize=9)
    ax.set_ylabel("访问节点数", fontsize=9)
    ax.set_ylim(0, max(mm) * 1.32)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("相同深度下两种算法的访问节点数", fontsize=9.5)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 5-1：不同地图规模下 A* 的扩展节点数
# ---------------------------------------------------------------------------
def fig_5_1_scaling(path: Path, rows: Optional[List[Dict[str, Any]]] = None) -> Path:
    plt = _plt()
    rows = rows or astar_scaling_experiments()

    sizes = sorted({row["size"] for row in rows})

    def series(key: str, field: str) -> List[float]:
        return [next(r for r in rows if r["size"] == s and r["heuristic_key"] == key)[field]
                for s in sizes]

    nodes_zero = series(HEURISTIC_ZERO, "expanded_nodes")
    nodes_manh = series(HEURISTIC_MANHATTAN, "expanded_nodes")
    time_zero = [v * 1000 for v in series(HEURISTIC_ZERO, "time")]
    time_manh = [v * 1000 for v in series(HEURISTIC_MANHATTAN, "time")]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    panels = (
        (axes[0], nodes_zero, nodes_manh, "扩展节点数", "(a) 扩展节点数", "{:.0f}"),
        (axes[1], time_zero, time_manh, "单次运行时间 / ms", "(b) 单次运行时间", "{:.2f}"),
    )
    for ax, a_vals, b_vals, ylabel, title, fmt in panels:
        ax.plot(sizes, a_vals, marker="o", color="#C0392B", lw=1.7, ms=5,
                label="h = 0（等价 Dijkstra）")
        ax.plot(sizes, b_vals, marker="s", color="#2E6FD9", lw=1.7, ms=5,
                label="h = Manhattan")
        top = max(max(a_vals), max(b_vals))
        for i, (s, a_val, b_val) in enumerate(zip(sizes, a_vals, b_vals)):
            # 首个数据点紧贴左边界，把红标签右移，避免压住图例
            if i == 0:
                ax.annotate(fmt.format(a_val), (s, a_val), textcoords="offset points",
                            xytext=(9, 6), ha="left", fontsize=7, color="#C0392B")
            else:
                ax.annotate(fmt.format(a_val), (s, a_val), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=7, color="#C0392B")
            ax.annotate(fmt.format(b_val), (s, b_val), textcoords="offset points",
                        xytext=(0, -15), ha="center", fontsize=7, color="#2E6FD9")
        ax.set_xlabel("地图边长（障碍密度 25%）", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xticks(sizes)
        ax.set_xlim(sizes[0] - 3, sizes[-1] + 3)
        ax.set_ylim(bottom=-top * 0.17, top=top * 1.14)
        ax.grid(linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_title(title, fontsize=9.5)

    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle("地图规模对 A* 搜索开销的影响", fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 图 5-2：四种访问顺序在两棵树上的节点数
# ---------------------------------------------------------------------------
def fig_5_2_order_summary(path: Path, order_data: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> Path:
    """四种访问顺序相对“从左到右”的节点数降幅。

    两棵树的规模相差一个数量级（14 个节点与 121 个节点），直接画绝对节点数
    会让小题的柱子被压平，因此改画相对基线的降幅，两棵树放在同一尺度上比较。
    """
    plt = _plt()
    order_data = order_data or tree_order_experiments()

    labels = [row["order_label"].replace(" ", "\n", 1) for row in order_data["pdf_tree"]]

    def reduction(rows: List[Dict[str, Any]]) -> List[float]:
        base = rows[0]["total_nodes"]
        return [(base - row["total_nodes"]) / base * 100 for row in rows]

    pdf = reduction(order_data["pdf_tree"])
    rnd = reduction(order_data["random_tree"])

    fig, ax = _new_figure(6.0, 3.3)
    x = list(range(len(labels)))
    width = 0.34
    b1 = ax.bar([i - width / 2 for i in x], pdf, width,
                label="题三给定博弈树（共 14 个节点）", color="#9BB7D4")
    b2 = ax.bar([i + width / 2 for i in x], rnd, width,
                label="3 叉 4 层随机博弈树（共 121 个节点）", color="#2E6FD9")
    ax.bar_label(b1, labels=[f"{v:.0f}%" for v in pdf], fontsize=7.5, padding=2)
    ax.bar_label(b2, labels=[f"{v:.0f}%" for v in rnd], fontsize=7.5, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("相对“从左到右”的节点数降幅 / %", fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, max(rnd) * 1.30)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.set_title("四种访问顺序的剪枝效果（基线为从左到右）", fontsize=9.5)
    return _save(fig, path)


# ---------------------------------------------------------------------------
# 统一入口
# ---------------------------------------------------------------------------
def _load_rows(csv_name: str) -> Optional[List[Dict[str, Any]]]:
    """读取 ``results/`` 下的实验 CSV，把能转成数字的列转成数字。

    图表与报告正文共用同一份 CSV，保证"报告里的数字 = 图上的数字"。
    """
    path = RESULTS_DIR / csv_name
    if not path.exists():
        return None
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle):
            row: Dict[str, Any] = {}
            for key, value in raw.items():
                text = (value or "").strip()
                try:
                    row[key] = int(text)
                except ValueError:
                    try:
                        row[key] = float(text)
                    except ValueError:
                        row[key] = text
            rows.append(row)
    return rows or None


def generate_all(output_dir: Optional[Path] = None) -> List[Path]:
    """生成全部报告图表，返回文件路径列表。

    数据类图表**优先读取** ``results/*.csv``。``main.py experiments`` 先写 CSV、
    再调用本函数，因此报告里引用的数字与图中标注的数字来自同一次测量，
    不会因为重新跑一遍实验而产生偏差。CSV 不存在时才现场跑实验。
    """
    out = Path(output_dir) if output_dir else DEFAULT_DIR
    out.mkdir(parents=True, exist_ok=True)

    astar_rows = _load_rows("astar_results.csv") or astar_experiments()
    scaling_rows = _load_rows("astar_scaling_results.csv") or astar_scaling_experiments()

    minimax_rows = _load_rows("minimax_results.csv")
    alpha_rows = _load_rows("alpha_beta_results.csv")
    if not minimax_rows or not alpha_rows:
        minimax_rows, alpha_rows = game_experiments()

    order_rows = _load_rows("tree_order_results.csv")
    if order_rows:
        order_data = {
            "pdf_tree": [r for r in order_rows if r["tree"] == "pdf_tree"],
            "random_tree": [r for r in order_rows if r["tree"] == "random_tree"],
        }
    else:
        order_data = tree_order_experiments()

    outputs = [
        fig_1_1_astar_flowchart(out / "fig1_1_astar_flowchart.png"),
        fig_1_2_search_flowchart(out / "fig1_2_search_flowchart.png"),
        fig_2_1_maze_path(out / "fig2_1_maze_path.png"),
        fig_2_2_astar_heuristic(out / "fig2_2_astar_heuristic.png", astar_rows),
        fig_2_3_weighted_terrain(out / "fig2_3_weighted_terrain.png"),
        fig_3_1_alpha_beta_tree(out / "fig3_1_alpha_beta_tree.png"),
        fig_3_2_tree_order(out / "fig3_2_tree_order.png", order_data),
        fig_4_1_distance_heatmap(out / "fig4_1_distance_heatmap.png"),
        fig_4_2_treasure_tree(out / "fig4_2_treasure_tree.png"),
        fig_4_3_minimax_vs_ab(out / "fig4_3_minimax_vs_ab.png", minimax_rows, alpha_rows),
        fig_5_1_scaling(out / "fig5_1_scaling.png", scaling_rows),
        fig_5_2_order_summary(out / "fig5_2_order_summary.png", order_data),
    ]
    return outputs
