"""共享数据结构定义。

本模块只描述"数据"与通用构造，不含任何搜索逻辑，也不含任何具体题目的用例：

* :class:`TreeNode`  —— 极大极小博弈树节点（PDF 给定的结构，
  额外支持显式的 ``is_maximizing_player`` 标记，以兼容 JSON 输入样例）。
* :class:`GameState` —— 题目四的宝藏博弈状态，**不可变**。
* :func:`build_random_tree` —— 通用随机博弈树，用于算法压力测试。

题目三的具体用例（PDF 给定的那棵树及其预期输出）在
:mod:`problems.game_tree`，与其他题目的输入数据放在一起。

``GameState`` 通过 ``dataclasses.replace`` 产生新状态，因此不同搜索分支
永远不会共享并修改同一份状态，满足"状态复制 / 不可变更新"的要求。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Tuple

# 地图坐标统一使用 (row, col)
Coord = Tuple[int, int]


# ---------------------------------------------------------------------------
# 题目三：极大极小博弈树节点
# ---------------------------------------------------------------------------
@dataclass
class TreeNode:
    """博弈树节点。

    PDF 给出的定义只包含 ``value`` 与 ``children``：

        class TreeNode:
            def __init__(self, value=None, children=None):
                self.value = value
                self.children = children if children is not None else []

    这里额外增加 ``is_maximizing_player``，用于承载 JSON 输入样例中的
    显式玩家标记；若为 ``None``，则由搜索时的层数奇偶推断（根为 MAX）。
    """

    value: Optional[float] = None
    children: List["TreeNode"] = field(default_factory=list)
    is_maximizing_player: Optional[bool] = None

    def __post_init__(self) -> None:
        # 允许 TreeNode(value=3, children=None) 这种写法
        if self.children is None:
            self.children = []

    # -- 便捷属性 ---------------------------------------------------------
    @property
    def is_leaf(self) -> bool:
        """叶子节点：没有子节点，只有评价值。"""
        return not self.children

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        if self.is_leaf:
            return f"TreeNode(value={self.value})"
        flag = "" if self.is_maximizing_player is None else (
            "MAX" if self.is_maximizing_player else "MIN"
        )
        return f"TreeNode({flag}, children={len(self.children)})"


def count_nodes(node: Optional[TreeNode]) -> int:
    """统计以 ``node`` 为根的子树节点总数（用于精确统计被剪掉的节点数）。"""
    if node is None:
        return 0
    total = 1
    for child in node.children:
        total += count_nodes(child)
    return total


def build_tree_from_dict(data: Any) -> TreeNode:
    """由 JSON 样例结构构建博弈树。

    支持两种输入形式::

        {"tree_structure": {...}}      # PDF 给出的完整输入样例
        {"is_maximizing_player": ...}  # 直接给出根节点

    节点字典中 ``value`` 表示叶子，``children`` 表示非叶子。
    """
    if isinstance(data, dict) and "tree_structure" in data:
        data = data["tree_structure"]
    if not isinstance(data, dict):
        raise ValueError("博弈树输入必须是 dict 结构")

    value = data.get("value")
    raw_children = data.get("children") or []
    children = [build_tree_from_dict(child) for child in raw_children]
    return TreeNode(
        value=value,
        children=children,
        is_maximizing_player=data.get("is_maximizing_player"),
    )


def build_random_tree(
    branching: int = 3,
    depth: int = 4,
    seed: int = 1,
    low: int = 0,
    high: int = 9,
) -> TreeNode:
    """生成一棵平衡的随机博弈树（拓展实验用）。

    MAX / MIN 由层数奇偶决定（根为 MAX），叶子取值在 ``[low, high]`` 内均匀分布。
    固定 ``seed`` 保证每次生成完全相同的树，实验可复现。
    """
    import random

    rng = random.Random(seed)

    def build(remaining: int) -> TreeNode:
        if remaining == 0:
            return TreeNode(value=rng.randint(low, high))
        return TreeNode(children=[build(remaining - 1) for _ in range(branching)])

    return build(depth)


# ---------------------------------------------------------------------------
# 题目四：宝藏博弈状态
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GameState:
    """不可变的博弈状态。

    只保存"局面"，不含规则。规则（距离矩阵、宝藏价值）由
    :class:`problems.treasure_game.TreasureGame` 提供，避免状态与地图耦合。

    属性
    ----
    pos_a, pos_b        : 双方当前位置
    remaining           : 尚未被领取的宝藏名（有序元组，保证确定性）
    score_a, score_b    : 双方累计收益
    current_player      : 当前行动玩家，"A"(MAX) 或 "B"(MIN)
    last_action         : 上一回合选择的宝藏，便于调试与回溯首步
    depth               : 已经走过的行动层数
    """

    pos_a: Coord
    pos_b: Coord
    remaining: Tuple[str, ...]
    score_a: float = 0
    score_b: float = 0
    current_player: str = "A"
    last_action: Optional[str] = None
    depth: int = 0

    # -- 只读派生信息 -----------------------------------------------------
    def is_max_turn(self) -> bool:
        """当前是否为 MAX 玩家（A）行动。"""
        return self.current_player == "A"

    def position_of(self, player: str) -> Coord:
        """返回指定玩家的当前位置。"""
        if player == "A":
            return self.pos_a
        if player == "B":
            return self.pos_b
        raise ValueError(f"未知玩家: {player!r}")

    def utility(self) -> float:
        """当前状态的评价值 U = Score_A - Score_B。"""
        return self.score_a - self.score_b

    def replace(self, **changes: Any) -> "GameState":
        """返回修改了部分字段的新状态（原状态保持不变）。"""
        return replace(self, **changes)

    def describe(self) -> str:  # pragma: no cover - 仅用于调试输出
        return (
            f"depth={self.depth} player={self.current_player} "
            f"A@{self.pos_a}({self.score_a}) B@{self.pos_b}({self.score_b}) "
            f"remaining={list(self.remaining)}"
        )
