"""题目四：竞争环境下的宝藏分配（综合题）。

四个阶段
--------
1. **距离矩阵**：调用 A* 计算 A、B、T1、T2、T3、T4 两两之间的最小移动代价，
   得到 6x6 矩阵。矩阵只算一次，之后博弈搜索直接查表，避免反复运行 A*。
2. **博弈状态**：:class:`utils.data_structures.GameState` 记录双方位置、
   剩余宝藏、累计收益与当前玩家；状态不可变，分支之间互不影响。
3. **博弈搜索**：Minimax 与 Alpha-Beta 共享本模块的
   ``actions`` / ``apply`` / ``evaluate``，只替换搜索方式。
4. **结果校验**：``H=2 -> T2 / -1``，``H=4 -> T3 / 3``。

规则要点
--------
* ``X`` 为障碍；数字为进入代价；``A``/``B``/``T*`` 所在格代价为 1；
* 只能上下左右移动；
* 每回合选一个尚未领取的宝藏，沿最小代价路径移动过去；
* 经过其他宝藏不自动收集；已领取宝藏的位置仍可通行；另一玩家不阻挡寻路；
* ``Gain(p, Ti) = Vi - d(p, Ti)``，``U = Score_A - Score_B``。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from algorithms.astar import (
    COST_WEIGHTED,
    HEURISTIC_WEIGHTED,
    astar_search,
    normalize_grid,
)
from algorithms.alpha_beta import timed_alpha_beta
from algorithms.minimax import timed_minimax
from utils.data_structures import GameState

Coord = Tuple[int, int]

# ---------------------------------------------------------------------------
# PDF 给定地图与宝藏价值
# ---------------------------------------------------------------------------
TREASURE_MAP: List[List[str]] = [
    ["A",  "1", "1", "X", "2", "1", "T1"],
    ["1",  "X", "6", "X", "1", "X", "1"],
    ["1",  "1", "T2", "1", "1", "1", "1"],
    ["X",  "1", "X", "8", "X", "1", "X"],
    ["1",  "1", "1", "1", "T3", "1", "1"],
    ["1",  "X", "9", "X", "1", "X", "1"],
    ["T4", "1", "1", "X", "1", "1", "B"],
]

TREASURE_VALUES: Dict[str, int] = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}
PLAYER_START: Dict[str, Coord] = {"A": (0, 0), "B": (6, 6)}

# 地图不可达时的收益惩罚（本题地图全部可达，仅作兜底）
UNREACHABLE_PENALTY = -1e9

# PDF 给出的预期输出
EXPECTED_RESULTS: Dict[int, Dict[str, Any]] = {
    2: {"best_target": "T2", "best_value": -1},
    4: {"best_target": "T3", "best_value": 3},
}


# ---------------------------------------------------------------------------
# 地图解析
# ---------------------------------------------------------------------------
def find_label(grid: Sequence[Sequence[object]], label: str) -> Coord:
    """在地图中定位某个标注（``A`` / ``B`` / ``T1`` 等）。"""
    target = label.strip().upper()
    for r, row in enumerate(grid):
        for c, cell in enumerate(row):
            if str(cell).strip().upper() == target:
                return (r, c)
    raise ValueError(f"地图中找不到标注 {label!r}")


# ---------------------------------------------------------------------------
# 博弈模型
# ---------------------------------------------------------------------------
class TreasureGame:
    """题目四的博弈模型：持有地图、宝藏价值与距离矩阵，并定义博弈规则。

    该对象实现 :mod:`algorithms.minimax` 文档中约定的博弈模型接口::

        actions(state) / apply(state, action) / evaluate(state)
        is_terminal(state) / is_max_turn(state)
    """

    def __init__(
        self,
        grid: Optional[Sequence[Sequence[object]]] = None,
        values: Optional[Dict[str, int]] = None,
        players: Optional[Dict[str, Coord]] = None,
        heuristic_type: str = HEURISTIC_WEIGHTED,
    ) -> None:
        self.grid: List[List[str]] = normalize_grid(grid if grid is not None else TREASURE_MAP)
        self.values: Dict[str, int] = dict(values if values is not None else TREASURE_VALUES)
        self.players: Dict[str, Coord] = dict(players if players is not None else PLAYER_START)
        self.heuristic_type = heuristic_type

        self.treasure_names: List[str] = sorted(self.values)      # T1, T2, T3, T4
        self.player_names: List[str] = sorted(self.players)       # A, B

        # 六个关键位置：A、B、T1..T4
        self.positions: Dict[str, Coord] = {}
        for name in self.player_names:
            self.positions[name] = self.players[name]
        for name in self.treasure_names:
            self.positions[name] = find_label(self.grid, name)

        self.labels: List[str] = self.player_names + self.treasure_names
        self.label_at: Dict[Coord, str] = {pos: name for name, pos in self.positions.items()}

        # 阶段一：一次性算出 6x6 距离矩阵
        self.distance_matrix: Dict[Tuple[str, str], float] = self._build_distance_matrix()

    # -- 阶段一：距离矩阵 -------------------------------------------------
    def _build_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """对所有**有序**位置对运行 A*，得到最小移动代价。

        对角线为 0；不可达记为 ``math.inf``。
        """
        matrix: Dict[Tuple[str, str], float] = {}
        for src in self.labels:
            for dst in self.labels:
                if src == dst:
                    matrix[(src, dst)] = 0.0
                    continue
                result = astar_search(
                    self.positions[src],
                    self.positions[dst],
                    self.grid,
                    heuristic_type=self.heuristic_type,
                    cost_mode=COST_WEIGHTED,
                    problem="problem4_distance",
                )
                matrix[(src, dst)] = result.cost if result.found else math.inf
        return matrix

    def distance_between_positions(self, src: Coord, dst: Coord) -> float:
        """按坐标查距离矩阵；坐标不在六个关键位置时回退到现场 A*。"""
        if src == dst:
            return 0.0
        src_label = self.label_at.get(tuple(src))
        dst_label = self.label_at.get(tuple(dst))
        if src_label is not None and dst_label is not None:
            return self.distance_matrix[(src_label, dst_label)]
        result = astar_search(
            tuple(src), tuple(dst), self.grid,
            heuristic_type=self.heuristic_type,
            cost_mode=COST_WEIGHTED,
            problem="problem4_distance",
        )
        return result.cost if result.found else math.inf

    def matrix_rows(self) -> List[List[Any]]:
        """返回 6x6 距离矩阵（二维列表，行/列顺序见 ``self.labels``）。"""
        return [
            [self.distance_matrix[(src, dst)] for dst in self.labels]
            for src in self.labels
        ]

    def format_matrix(self) -> str:
        """把距离矩阵渲染成对齐的文本表格。"""
        def cell(value: Any) -> str:
            if value == math.inf:
                return "inf"
            if isinstance(value, float) and value.is_integer():
                return str(int(value))
            return str(value)

        width = max(len(cell(v)) for row in self.matrix_rows() for v in row)
        width = max(width, 3)
        header = "".ljust(5) + "".join(label.rjust(width + 2) for label in self.labels)
        lines = [header]
        for src, row in zip(self.labels, self.matrix_rows()):
            lines.append(src.ljust(5) + "".join(cell(v).rjust(width + 2) for v in row))
        return "\n".join(lines)

    # -- 阶段二：状态与规则 -----------------------------------------------
    def initial_state(self) -> GameState:
        """初始状态：A 在起点、B 在起点、四个宝藏全部剩余、当前玩家 A。"""
        return GameState(
            pos_a=self.players["A"],
            pos_b=self.players["B"],
            remaining=tuple(self.treasure_names),
            score_a=0,
            score_b=0,
            current_player="A",
            last_action=None,
            depth=0,
        )

    def actions(self, state: GameState) -> List[str]:
        """当前玩家可选的宝藏：所有尚未被领取的宝藏，顺序固定为 T1..T4。"""
        return [name for name in self.treasure_names if name in state.remaining]

    def gain(self, position: Coord, treasure: str) -> float:
        """``Gain(p, Ti) = Vi - d(p, Ti)``。"""
        dist = self.distance_between_positions(position, self.positions[treasure])
        if dist == math.inf:
            return UNREACHABLE_PENALTY
        return float(self.values[treasure] - dist)

    def apply(self, state: GameState, action: str) -> GameState:
        """执行行动，返回**新的**状态（原状态不变）。

        步骤：算代价 → 算收益 → 更新位置与累计得分 → 移除宝藏 → 换手 → 层数 +1。
        """
        if action not in state.remaining:
            raise ValueError(f"宝藏 {action!r} 已被领取或不存在")

        player = state.current_player
        current_pos = state.position_of(player)
        value = self.gain(current_pos, action)
        new_pos = self.positions[action]
        remaining = tuple(name for name in state.remaining if name != action)

        if player == "A":
            return state.replace(
                pos_a=new_pos,
                score_a=state.score_a + value,
                remaining=remaining,
                current_player="B",
                last_action=action,
                depth=state.depth + 1,
            )
        return state.replace(
            pos_b=new_pos,
            score_b=state.score_b + value,
            remaining=remaining,
            current_player="A",
            last_action=action,
            depth=state.depth + 1,
        )

    def evaluate(self, state: GameState) -> float:
        """``U = Score_A - Score_B``。"""
        return state.utility()

    def is_terminal(self, state: GameState) -> bool:
        """宝藏被取完即终止。"""
        return not state.remaining

    def is_max_turn(self, state: GameState) -> bool:
        """A 是 MAX，B 是 MIN。"""
        return state.is_max_turn()


# ---------------------------------------------------------------------------
# 模块级规则函数（便于单独调用 / 单元测试）
# ---------------------------------------------------------------------------
def generate_actions(state: GameState, game: TreasureGame) -> List[str]:
    """当前状态下的全部合法行动。"""
    return game.actions(state)


def apply_action(state: GameState, action: str, game: TreasureGame) -> GameState:
    """执行行动并返回新状态。"""
    return game.apply(state, action)


def evaluate_state(state: GameState) -> float:
    """状态评价值 U = Score_A - Score_B。"""
    return state.utility()


# ---------------------------------------------------------------------------
# 求解入口
# ---------------------------------------------------------------------------
SEARCH_ALGORITHMS = ("Minimax", "Alpha-Beta")


def solve_treasure_game(
    game: Optional[TreasureGame] = None,
    depth: int = 2,
    algorithm: str = "Alpha-Beta",
    return_stats: bool = False,
) -> Any:
    """在给定搜索深度下求 A 的最优首步。

    ``game`` 为空时使用 PDF 给定的地图与宝藏价值。返回 ``(值, 首步)``；
    ``return_stats=True`` 时返回 ``(值, 首步, 统计)``，``统计`` 中含节点数、
    剪枝次数与耗时。``algorithm`` 取 ``"Minimax"`` 或 ``"Alpha-Beta"``。
    """
    game = game if game is not None else default_game()
    state = game.initial_state()

    if algorithm == "Minimax":
        value, action, stats = timed_minimax(
            state, game, depth, problem="problem4_treasure"
        )
    elif algorithm == "Alpha-Beta":
        value, action, stats = timed_alpha_beta(
            state, game, depth, problem="problem4_treasure"
        )
    else:
        raise ValueError(f"未知算法 {algorithm!r}，可选 {SEARCH_ALGORITHMS}")

    return (value, action, stats) if return_stats else (value, action)


# ---------------------------------------------------------------------------
# 分析辅助
# ---------------------------------------------------------------------------
def play_best_line(
    game: TreasureGame,
    depth: int,
) -> Tuple[List[Dict[str, Any]], GameState]:
    """沿最优走法回放主变例（principal variation）。

    每步都以**剩余层数**为深度重做一次 Minimax，得到双方的最优应对。
    这条线用来解释"为什么 H=2 选 T2、H=4 改选 T3"——深度改变的不是
    评价值本身，而是双方在剩余回合里的相对位置。
    """
    state = game.initial_state()
    remaining_depth = depth
    line: List[Dict[str, Any]] = []

    while remaining_depth > 0 and not game.is_terminal(state):
        _value, action, _stats = timed_minimax(state, game, remaining_depth)
        player = state.current_player
        gain = game.gain(state.position_of(player), action)
        line.append({
            "step": state.depth + 1,
            "player": player,
            "target": action,
            "value": game.values[action],
            "distance": game.distance_between_positions(
                state.position_of(player), game.positions[action]
            ),
            "gain": gain,
            "score_a": (state.score_a + gain) if player == "A" else state.score_a,
            "score_b": (state.score_b + gain) if player == "B" else state.score_b,
        })
        state = game.apply(state, action)
        remaining_depth -= 1

    return line, state


def describe_decision_path(
    game: TreasureGame,
    state: GameState,
    actions: Sequence[str],
    max_depth: int,
) -> List[Dict[str, Any]]:
    """列出根节点各候选首步的收益与即时 U，便于分析 H=2 的决策依据。"""
    rows: List[Dict[str, Any]] = []
    for action in actions:
        gain = game.gain(state.pos_a, action)
        rows.append({
            "target": action,
            "value": game.values[action],
            "distance": game.distance_between_positions(state.pos_a, game.positions[action]),
            "gain": gain,
            "u_after_move": state.score_a + gain - state.score_b,
            "depth": max_depth,
        })
    return rows


def default_game() -> TreasureGame:
    """返回题目四的默认博弈（PDF 给定地图与价值）。"""
    return TreasureGame()


def check_expected(depth: int, best_target: str, best_value: float) -> Tuple[bool, str]:
    """对照 PDF 预期输出校验结果。"""
    expected = EXPECTED_RESULTS.get(depth)
    if expected is None:
        return True, f"深度 H={depth} 无给定预期输出"
    ok_target = expected["best_target"] == best_target
    ok_value = abs(float(expected["best_value"]) - float(best_value)) < 1e-9
    if ok_target and ok_value:
        return True, f"H={depth} 与预期一致：{best_target} / {best_value}"
    return False, (
        f"H={depth} 与预期不一致：期望 {expected['best_target']}/{expected['best_value']}，"
        f"实际 {best_target}/{best_value}"
    )
