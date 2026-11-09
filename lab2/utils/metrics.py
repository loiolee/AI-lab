"""统一的搜索性能统计接口。

设计要点
--------
1. 统计量必须由算法**内部**在真正访问/扩展节点时同步累加，绝不在算法结束后估算。
2. 所有时间统一使用秒，由 :class:`Timer` 或 ``time.perf_counter`` 提供。
3. 所有 CSV 列名固定，方便不同实验之间直接拼接与绘图。

关于 ``pruned_nodes`` 的说明
----------------------------
``pruned_nodes`` 指"因为剪枝而没有被实际搜索的节点数"。
只有在博弈树结构完全已知时（题目三的 ``TreeNode``）才能精确统计；
在题目四这种按需生成子节点的场景下无法事先知道被跳过分支的规模，
此时只统计 ``pruned_branches``（被剪掉的子分支个数）与 ``pruning_count``，
**不做任何估算**，避免出现凭空构造的节点数。
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# A* 统计
# ---------------------------------------------------------------------------
@dataclass
class AStarStats:
    """A* 搜索的统计信息。"""

    algorithm: str = "A*"
    problem: str = ""
    heuristic: str = ""
    cost_mode: str = ""
    expanded_nodes: int = 0      # 实际从 Open 表中取出并扩展的节点数
    visited_nodes: int = 0       # 曾经被发现（进入过 Open 表）的节点数
    generated_nodes: int = 0     # 由扩展产生的后继节点总数
    path_length: int = 0         # 路径包含的移动步数
    total_cost: float = 0.0      # 路径总代价
    execution_time: float = 0.0  # 运行时间（秒）
    found: bool = True           # 是否找到解

    def to_row(self) -> Dict[str, Any]:
        """转换为 CSV 行（列顺序与 README 中的说明一致）。"""
        return {
            "algorithm": self.algorithm,
            "problem": self.problem,
            "heuristic": self.heuristic,
            "expanded_nodes": self.expanded_nodes,
            "visited_nodes": self.visited_nodes,
            "path_length": self.path_length,
            "total_cost": _num(self.total_cost),
            "time": round(self.execution_time, 6),
            "cost_mode": self.cost_mode,
            "generated_nodes": self.generated_nodes,
            "found": int(self.found),
        }


# ---------------------------------------------------------------------------
# 博弈搜索统计（Minimax / Alpha-Beta 共用）
# ---------------------------------------------------------------------------
@dataclass
class GameSearchStats:
    """Minimax / Alpha-Beta 的统计信息。"""

    algorithm: str = "Minimax"
    problem: str = ""
    depth: int = 0
    order: str = "left_to_right"
    total_nodes: int = 0         # 递归访问到的节点总数
    expanded_nodes: int = 0      # 内部节点（真正展开子节点）的数量
    leaf_nodes: int = 0          # 执行评价值计算的叶子节点数
    pruning_count: int = 0       # 剪枝发生的次数
    pruned_nodes: int = 0        # 被剪掉的节点数（仅树结构可精确统计）
    pruned_branches: int = 0     # 被剪掉的子分支个数
    max_depth: int = 0           # 实际到达的最大层数
    best_target: str = ""        # 根节点选出的最优行动
    best_value: float = 0.0      # 根节点的最优评价值
    execution_time: float = 0.0  # 运行时间（秒）

    def to_row(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "depth": self.depth,
            "total_nodes": self.total_nodes,
            "leaf_nodes": self.leaf_nodes,
            "pruning_count": self.pruning_count,
            "pruned_nodes": self.pruned_nodes,
            "best_target": self.best_target,
            "best_value": _num(self.best_value),
            "time": round(self.execution_time, 6),
            "order": self.order,
            "expanded_nodes": self.expanded_nodes,
            "pruned_branches": self.pruned_branches,
            "max_depth": self.max_depth,
            "problem": self.problem,
        }


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def _num(value: Any) -> Any:
    """整数值的浮点数转成 int，避免 CSV 里出现 7.0 这种写法。"""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def format_number(value: Any, none: str = "-") -> str:
    """把数值格式化成终端输出用的字符串。

    整数值的浮点数去掉小数点（``5.0`` -> ``5``），其余浮点保留 4 位小数，
    ``None`` 返回 ``none`` 指定的占位符。终端打印、字符画渲染、校验消息
    统一走这一个函数，避免各处格式不一致。
    """
    if value is None:
        return none
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}"
    return str(value)


class Timer:
    """上下文管理器，用于测量代码块耗时（秒）。::

        with Timer() as t:
            ...
        print(t.elapsed)
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.elapsed = time.perf_counter() - self.start


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> Path:
    """把字典列表写成 CSV，返回实际写入的路径。

    列顺序取自 ``rows[0]`` 的键顺序；空列表时只创建一个空文件。
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8-sig")
        return out_path

    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def format_table(rows: Sequence[Dict[str, Any]], columns: Optional[Sequence[str]] = None) -> str:
    """把字典列表渲染成等宽文本表格，便于在终端查看实验数据。"""
    if not rows:
        return "(空)"
    columns = list(columns) if columns else list(rows[0].keys())

    body: List[List[str]] = []
    for row in rows:
        body.append([str(_num(row.get(col, ""))) for col in columns])

    widths = [len(str(col)) for col in columns]
    for line in body:
        for idx, cell in enumerate(line):
            widths[idx] = max(widths[idx], len(cell))

    header = "  ".join(str(col).ljust(widths[i]) for i, col in enumerate(columns))
    sep = "  ".join("-" * widths[i] for i in range(len(columns)))
    lines = [header, sep]
    for line in body:
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)))
    return "\n".join(lines)
