"""题目三：极大极小博弈树的最佳得分。

本模块只描述**题目本身**——树的结构、预期输出与展示方式，
不含任何搜索逻辑。求解由 :mod:`algorithms.alpha_beta` 提供。

树的形态（与 PDF 给定的构建代码一致）::

    root(MAX)
    ├── min1(MIN)
    │   ├── max1(MAX): 3, 5
    │   └── max2(MAX): 6, 9
    └── min2(MIN)
        ├── max2(MAX): 6, 9   <- 与 min1 的子节点是同一棵子树
        └── max3(MAX): 1, 2

``min2`` 复用 ``max2`` 是 PDF 原代码的写法，这里照原样保留。它带来一个
副作用：同一个节点对象出现在树的两个位置，因此用 ``id(node)`` 做键的
逐节点轨迹（报告图 3-1 用）会互相覆盖。绘图代码另建了一棵不复用子树的
等价树来规避，算法结果不受影响——两处层数相同，MAX/MIN 判定一致。

规则要点
--------
* 根节点为 MAX，逐层交替；
* MAX 最大化得分，MIN 最小化得分；
* 叶子节点取 ``value``；
* 子节点**从左到右**访问。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from algorithms.alpha_beta import (
    ORDER_LEFT_TO_RIGHT,
    timed_alpha_beta_tree,
)
from algorithms.minimax import minimax_tree
from utils.data_structures import TreeNode
from utils.metrics import format_number

# PDF 给出的预期输出
EXPECTED_VALUE = 5.0


def build_sample_tree() -> TreeNode:
    """构建 PDF 题目三的测试用例（预期输出 5）。"""
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])

    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])

    return TreeNode(children=[min1, min2], is_maximizing_player=True)


def build_mirrored_tree() -> TreeNode:
    """构建一棵与给定树**取值相同但节点不复用**的等价树。

    与 :func:`build_sample_tree` 的唯一区别是 ``max2`` 被复制成两个独立对象，
    因此每个位置有唯一的 ``id``。报告图 3-1 需要逐节点标注 alpha/beta，
    用这棵树可以避免轨迹互相覆盖。
    """
    def leaves(*values: float) -> TreeNode:
        return TreeNode(children=[TreeNode(value=v) for v in values])

    max1 = leaves(3, 5)
    max2a = leaves(6, 9)
    max2b = leaves(6, 9)
    max3 = leaves(1, 2)
    min1 = TreeNode(children=[max1, max2a])
    min2 = TreeNode(children=[max2b, max3])
    return TreeNode(children=[min1, min2], is_maximizing_player=True)


def solve_tree(
    root: Optional[TreeNode] = None,
    order: str = ORDER_LEFT_TO_RIGHT,
    prune_log: Optional[List[Dict[str, Any]]] = None,
    return_stats: bool = False,
) -> Any:
    """调用 Alpha-Beta 求博弈树的最佳得分（MAX 玩家的视角）。

    ``root`` 为空时使用 PDF 给定的博弈树。默认从左到右访问子节点，
    与题目要求一致。

    ``prune_log`` 传入一个列表时，剪枝事件会逐条追加进去，用于展示
    "在哪个节点、以什么 alpha/beta 值跳过了多少节点"。
    ``return_stats=True`` 时返回 ``(value, stats)``，其中 ``stats`` 为
    :class:`utils.metrics.GameSearchStats`，含节点数与剪枝次数。
    """
    tree = root if root is not None else build_sample_tree()
    value, stats = timed_alpha_beta_tree(tree, order=order, prune_log=prune_log)
    return (value, stats) if return_stats else value


def solve_tree_baseline(root: Optional[TreeNode] = None) -> float:
    """用 Minimax 求同一个博弈树的得分，作为 Alpha-Beta 的对照基准。

    两者必须相等——剪枝只改变搜索规模，不改变最终决策。
    """
    tree = root if root is not None else build_sample_tree()
    return minimax_tree(tree)


def format_tree(node: TreeNode, prefix: str = "", is_last: bool = True,
                depth: int = 0) -> List[str]:
    """把博弈树渲染成字符画，返回每一行的文本。"""
    connector = "" if depth == 0 else ("└── " if is_last else "├── ")

    if node.is_leaf:
        return [f"{prefix}{connector}[{format_number(node.value)}]"]

    if node.is_maximizing_player is not None:
        kind = "MAX" if node.is_maximizing_player else "MIN"
    else:
        kind = "MAX" if depth % 2 == 0 else "MIN"

    lines = [f"{prefix}{connector}{kind}"]
    child_prefix = prefix + ("" if depth == 0 else ("    " if is_last else "│   "))
    for index, child in enumerate(node.children):
        lines.extend(format_tree(child, child_prefix,
                                 index == len(node.children) - 1, depth + 1))
    return lines


def print_tree(node: TreeNode) -> None:
    """打印博弈树的字符画。"""
    for line in format_tree(node):
        print(line)


def check_expected(value: float) -> Tuple[bool, str]:
    """对照 PDF 预期输出校验结果。"""
    if abs(float(value) - EXPECTED_VALUE) < 1e-9:
        return True, f"与 PDF 预期一致：{format_number(value)}"
    return False, f"与 PDF 预期不一致：期望 {format_number(EXPECTED_VALUE)}，实际 {format_number(value)}"
