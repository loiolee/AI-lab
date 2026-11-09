# 实验二：搜索算法（A\* / Minimax / Alpha-Beta）

人工智能实验课 · 实验二。项目用 Python 实现三种搜索算法，并用它们完成四个题目：

| 题目 | 内容 | 用到的算法 |
| --- | --- | --- |
| 题目一 | 迷宫寻路（单位代价） | A\*（h=0 / Manhattan） |
| 题目二 | 加权地形最小移动代价 | A\*（加权曼哈顿） |
| 题目三 | 极大极小博弈树的最佳得分 | Alpha-Beta（Minimax 作基准） |
| 题目四 | 竞争环境下的宝藏分配 | A\*（距离矩阵）+ Minimax + Alpha-Beta |

四道题的输出都与 PDF 给出的预期结果逐项核对过；除四道基本题外，另做了四组拓展实验
（地图规模、访问顺序、单层/双向排序、可采纳性验证），共 29 条对照实验记录
（A1~A4 四条、B1~B7 七条、C1~C4 在两棵树上共八条、D1~D10 十条），
全部数据由 `python main.py experiments` 一次跑出。

---

## 1. 环境要求

* **Python 3.8 及以上**（开发与验证环境为 Python 3.13，Windows 11）
* **核心功能零第三方依赖**：算法、CLI、单元测试只用标准库
  （`heapq`、`dataclasses`、`csv`、`argparse`、`unittest`、`time`、`random`）
* **绘图依赖 `matplotlib`**：只有 `main.py figures` 与 `experiments` 的绘图环节
  需要它；未安装时求解、实验、CSV 与单元测试照常运行，只是不产出图片

```bash
pip install -r requirements.txt   # 只装 matplotlib
```

---

## 2. 项目结构

```
experiment_2_search_algorithms/
├── main.py                     # 入口：单题求解 / 对照实验 / 图表 / 测试
├── requirements.txt
│
├── algorithms/                 # 三个算法模块，互不依赖，也不 import problems/
│   ├── astar.py                #   通用 A*（启发函数、代价规则参数化）
│   ├── minimax.py              #   Minimax（面向通用博弈模型接口）
│   └── alpha_beta.py           #   Alpha-Beta（剪枝记录、四种访问顺序、trace）
│
├── problems/                   # 题目层：定义输入、调用算法、格式化输出
│   ├── maze.py                 #   题目一 solve_maze() + 随机迷宫生成
│   ├── weighted_terrain.py     #   题目二 solve_weighted_terrain() + 可采纳性验证
│   ├── game_tree.py            #   题目三 solve_tree() + 博弈树定义与渲染
│   └── treasure_game.py        #   题目四 solve_treasure_game() + 6x6 距离矩阵
│
├── utils/
│   ├── data_structures.py      # TreeNode、GameState（不可变）、随机博弈树
│   ├── metrics.py              # 统计类、计时、CSV 读写、表格与数值格式化
│   ├── experiments.py          # A1~A4 / B1~B7 / C1~C4 / D1~D10 实验定义
│   ├── visualization.py        # matplotlib 后端加载与中文字体配置
│   └── report_figures.py       # 报告用 12 张中文图表（图 1-1 ~ 图 5-2）
│
├── tests/                      # 标准库 unittest，136 个用例
│   ├── test_astar.py           ├── test_minimax.py
│   ├── test_alpha_beta.py      ├── test_game_tree.py
│   ├── test_maze.py            ├── test_treasure_game.py
│   ├── test_data_structures.py └── test_experiments.py
│
├── report/                     # 实验报告（5 页 A4）
│   ├── report.html             #   报告源文件（打印即得 PDF）
│   └── 实验二报告_李麒_10245501423.pdf
│
└── results/
    ├── astar_results.csv           # A1~A4
    ├── astar_scaling_results.csv   # D1~D10
    ├── tree_order_results.csv      # C1~C4（两棵树）
    ├── minimax_results.csv         # B1、B3
    ├── alpha_beta_results.csv      # B2、B4~B7
    ├── treasure_results.csv        # 题目四单次求解输出
    └── report_figures/             # 报告用 12 张 PNG
```

### 四道题的解题入口

`main.py` 只负责终端展示，求解本身都在 `problems/` 里，四道题各有一个 `solve_*`
函数，签名风格统一（返回简洁结果，`return_result` / `return_stats` 取完整信息）：

| 题目 | 入口 | 返回 |
| --- | --- | --- |
| 一 | `maze.solve_maze(start, goal, grid, heuristic)` | 路径坐标列表 / `AStarResult` |
| 二 | `weighted_terrain.solve_weighted_terrain(start, goal, grid, heuristic)` | 最小总代价 / `AStarResult` |
| 三 | `game_tree.solve_tree(root, order, prune_log)` | MAX 的最佳得分 / `(值, 统计)` |
| 四 | `treasure_game.solve_treasure_game(game, depth, algorithm)` | `(值, 首步)` / `(值, 首步, 统计)` |

题目三另有 `solve_tree_baseline()` 返回 Minimax 结果作对照；题目四另有
`play_best_line()` 回放主变例，逐层解释双方的连续应对。

### 设计约束

1. **A\* 只有一份实现**。题目一、二、四共用 `astar_search()`，靠
   `heuristic_type`（`zero` / `manhattan` / `weighted_manhattan`）和
   `cost_mode`（`uniform` / `weighted`）两个参数切换行为。
2. **Minimax 与 Alpha-Beta 共享全部博弈逻辑**。两者只依赖模型提供的
   `actions / apply / evaluate / is_terminal / is_max_turn`，差别仅在递归时
   是否维护 `alpha`、`beta`。行动生成、收益计算、状态更新都只有一份代码，
   所以对照实验比较的是搜索方式本身。
3. **依赖方向单一**。`algorithms/*` 只依赖 `utils`，`problems/*` 调用算法层，
   算法层从不 import 题目层，任一模块都可单独测试。
4. **解题逻辑在 `problems/`，`main.py` 只做展示**。四道题各有一个 `solve_*`
   入口，输入定义、算法调用与结果校验都在题目层完成；`main.py` 负责打印表格、
   渲染字符画和拼接分析文字，不直接调用算法函数。
5. **统计量在算法内部产生**。`expanded_nodes` 在节点真正出堆扩展时自增，
   `visited_nodes` 在首次入队时自增，`pruning_count` 在 `alpha >= beta`
   且确实还有子节点未搜索时自增，没有任何事后估算。

---

## 3. 运行方式

```bash
python main.py p1            # 题目一：迷宫寻路（含 h=0 对照）
python main.py p2            # 题目二：加权地形（含可采纳性逐格验证）
python main.py p3            # 题目三：博弈树最佳得分（含四种访问顺序对照）
python main.py p4            # 题目四：距离矩阵 + H=2 / H=4 主变例
python main.py matrix        # 只打印 6x6 距离矩阵
python main.py experiments   # A/B/C/D 全部对照实验 → CSV + 12 张图表
python main.py figures       # 只重画图表，复用已有 CSV
python main.py all           # 依次运行 p1~p4 与 experiments
python main.py test          # 运行 136 个单元测试
```

常用参数：

```bash
python main.py p4 --depth 2                # 只跑 H=2
python main.py experiments --no-charts     # 只写 CSV，不画图
python main.py p4 --no-save                # 不写 treasure_results.csv
```

也可以直接用标准库跑测试：

```bash
python -m unittest discover -s tests -t .
```

---

## 4. 算法要点

### 4.1 A\*

* 状态是网格坐标 `(row, col)`；`f(n) = g(n) + h(n)`
* Open 表用 `heapq`，排序键 `(f, -g, 入队序号)`
  * `f` 相同时优先扩展 **g 更大**（离终点更近）的节点。这个 tie-break 不改变
    最优性，但能减少无效扩展，并使题目一的输出与 PDF 示例路径逐点一致
* Closed 表记录每个节点被扩展时的 `g`；发现更小的 `g` 时**允许重新打开**，
  保证启发函数只满足可采纳、不满足一致性时依然最优
* 出堆时若该记录的 `g` 大于当前最优 `g`，直接丢弃（过期节点）
* 邻居顺序固定为 上 → 下 → 左 → 右，保证结果可复现

**三种启发函数**

| 类型 | 公式 | 可采纳性 |
| --- | --- | --- |
| `zero` | `h = 0` | 恒可采纳，等价 Dijkstra |
| `manhattan` | `h = \|Δr\| + \|Δc\|` | 只在单位代价地图上可采纳 |
| `weighted_manhattan` | `h = (d-1)·c_min + cost(goal)` | 可采纳，`c_min` 由当前地图实算 |

推导：从 n 到终点至少走 d 步，除终点外每一步的进入代价都不小于 `c_min`，
终点代价单独计入，因此 `(d-1)·c_min + cost(goal)` 不会超过真实剩余代价。
题目二的终点 `E` 进入代价为 0，公式退化为 `max(0, Manhattan-1) × c_min`。
`problems/weighted_terrain.py` 的 `verify_admissible()` 遍历全部 24 个可通行格，
用 Dijkstra 求真值逐格比对，结果 0 处违反，最大松弛量 2。

### 4.2 Minimax 与 Alpha-Beta

* 评价值 `U = Score_A - Score_B`，A 为 MAX，B 为 MIN
* 终止条件：`depth >= H` 或宝藏取完，此时返回当前 `U`
* `H=2` 为 A、B 各行动一次，`H=4` 为 A→B→A→B；根节点返回的 action 即最优首步
* 剪枝条件 `alpha >= beta`，且**只在确实还有子节点未搜索时**记一次剪枝
  （否则会把"最后一个子节点搜完"误记成剪枝）

**题目三的三个入口**（分层：`problems` 调用 `algorithms`）

| 层次 | 函数 | 签名 | 用途 |
| --- | --- | --- | --- |
| 题目层 | `game_tree.solve_tree(root, order, prune_log, return_stats)` | 可选参数，默认用 PDF 给定的树 | 题目三的解题入口，`main.py p3` 走这里 |
| 算法层 | `alpha_beta_max(root)` | 单参数，返回 `float` | 题目要求的那一个函数：自包含、从左到右、不依赖任何辅助设施，可直接提交 |
| 算法层 | `alpha_beta_tree(root, order, stats, prune_log, trace)` | 带可选参数 | 实验与报告用：额外记录节点统计、剪枝明细与逐节点 α/β 轨迹 |

三者返回值完全相同：`tests/test_alpha_beta.py` 的 `TestProblem3PureFunction`
在给定树与 30 棵随机树上比对 `alpha_beta_max` 与完整实现，
`tests/test_game_tree.py` 比对 `solve_tree` 与 `solve_tree_baseline`。
`alpha_beta_max` 刻意不做子节点排序——题目三要求从左到右，先算一遍静态估值
再排回原序是多余开销。

**四种子节点访问顺序**（`algorithms/alpha_beta.py`）

| 常量 | 含义 | 作用层 |
| --- | --- | --- |
| `ORDER_LEFT_TO_RIGHT` | 按行动生成顺序，从左到右 | — |
| `ORDER_MAX_HIGH` | 估值从高到低 | 仅 MAX 层 |
| `ORDER_MIN_LOW` | 估值从低到高 | 仅 MIN 层 |
| `ORDER_HEURISTIC` | 双向排序：MAX 高优先、MIN 低优先 | 两层 |

排序只改变访问次序，不改变 Minimax 值，这一点由 `tests/test_alpha_beta.py`
用 40 棵随机树逐一比对验证。

**`pruned_nodes` 的统计口径**：题目三的树结构完全已知，被剪掉的节点数可以精确
统计（`count_nodes()` 累加子树规模）；题目四按需生成子节点，无法预知被跳过分支的
规模，因此只记 `pruned_branches`（被剪掉的子分支个数），`pruned_nodes` 保持 0，
不做估算。

### 4.3 题目四的状态建模

```python
GameState(pos_a, pos_b, remaining, score_a, score_b,
          current_player, last_action, depth)   # frozen dataclass
```

* 不可变：`apply()` 用 `dataclasses.replace()` 生成新状态，不同搜索分支不共享状态
* `Gain(p, Ti) = Vi - d(p, Ti)`，收益可以为负（A 到 T1：`9 - 10 = -1`）
* 经过其他宝藏不自动收集；已领取宝藏的格子仍可通行；对手不阻挡寻路
* 6x6 距离矩阵只算一次（36 个有序对），博弈搜索全程查表，不再调用 A\*

### 4.4 随机迷宫生成

拓展实验需要不同规模的地图，`problems/maze.py` 的 `generate_maze()` 按障碍比例
随机撒点，再用 BFS 校验起点到终点是否连通，不连通就换种子重试。这样得到的迷宫
既有随机障碍，又保证 A\* 一定有解，避免对照实验出现空结果。固定种子即可复现。

---

## 5. 四道题的结果与核对

| 题目 | 输入 | PDF 预期 | 实际输出 | 结果 |
| --- | --- | --- | --- | --- |
| 一 | `(0,0) → (3,4)`，5×5 迷宫 | 8 个坐标点的路径 | 逐点一致，7 步 / 代价 7 | ✅ |
| 二 | `(0,0) → (4,4)`，5×5 加权图 | `7` | `7`，路径与 PDF 一致 | ✅ |
| 三 | 给定博弈树 | `5` | `5`（Minimax 基准同为 5） | ✅ |
| 四 | H=2 | `T2 / -1` | `T2 / -1`（两种算法一致） | ✅ |
| 四 | H=4 | `T3 / 3` | `T3 / 3`（两种算法一致） | ✅ |

题目二的路径代价累计（`main.py p2` 逐行打印）：

```
(0,0)→(1,0) +1 =1  …  (4,3)→(4,4) +0 =7
```

最后一步进入 `E` 的代价为 0，这是总代价为 7 而不是 8 的原因。

---

## 6. 题目四：6×6 距离矩阵与主变例

A\*（加权模式）算出的距离矩阵，行是起点、列是终点：

|      | A  | B  | T1 | T2 | T3 | T4 |
| ---- | -- | -- | -- | -- | -- | -- |
| A    | 0  | 12 | 10 | 4  | 8  | 8  |
| B    | 12 | 0  | 8  | 8  | 4  | 10 |
| T1   | 10 | 8  | 0  | 6  | 6  | 12 |
| T2   | 4  | 8  | 6  | 0  | 6  | 6  |
| T3   | 8  | 4  | 6  | 6  | 0  | 6  |
| T4   | 8  | 10 | 12 | 6  | 6  | 0  |

六个关键格的进入代价都是 1，矩阵因此对称；`tests/test_treasure_game.py`
对 36 个元素逐一验证，并与直接调用 A\* 的结果交叉核对。

**H=2 为什么是 T2 / -1**

| A 的首步 | A 的净收益 | B 的最优应对 | 终局 U |
| --- | --- | --- | --- |
| T1 | -1（9-10） | T3（+3） | -4 |
| **T2** | **+2（6-4）** | **T3（+3）** | **-1** |
| T3 | -1（7-8） | T1（+1） | -2 |
| T4 | -1（7-8） | T3（+3） | -4 |

A 取 T2 时自己拿 +2，B 仍能拿 +3，`U = 2 - 3 = -1`，是四个首步里最好的。

**H=4 为什么变成 T3 / 3**

主变例是 `A:T3(-1) → B:T1(+1) → A:T2(0) → B:T4(-5)`，终局
`Score_A = -1`、`Score_B = -4`，`U = 3`。

关键在于位置塑造：A 先占 T3(4,4)，B 从 (6,6) 出发只剩 T1(0,6) 可选；A 再取走
T2(2,2)，最后留给 B 的是 T4(6,0)，而 B 从 T1 到 T4 要走 12 步，净收益
`7 - 12 = -5`，把 B 的累计收益拖成负数。A 自己首步亏 1，却把"最后一手必须长途
奔袭"的代价转给了对手。

若 A 仍按 H=2 的选择先取 T2，B 会立刻拿走附近的 T3（+3），A 之后只能取 T1，
终局 `U = 1`，不如 3。这说明加深搜索改变了决策：H=2 看不到"谁最后被留下远距离
宝藏"这一后果。

---

## 7. 对照实验数据

`python main.py experiments` 写出 `results/` 下的 CSV 和 `results/report_figures/`
下的图表。为压低计时噪声，每个实验重复执行取均值：A\* 200 次、博弈搜索 100 次、
树搜索 300 次。节点数与剪枝次数完全确定，时间仅作佐证。

### 7.1 A\* 启发函数对照（A1~A4）

| 编号 | 题目 | 启发函数 | 扩展节点 | 生成节点 | 路径步数 | 总代价 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | 题目一 | `zero` | 17 | 18 | 7 | 7 | 0.048 ms |
| A2 | 题目一 | `manhattan` | 14 | 17 | 7 | 7 | 0.048 ms |
| A3 | 题目二 | `weighted_manhattan` | 9 | 16 | 8 | 7 | 0.065 ms |
| A4 | 题目二 | `zero` | 15 | 24 | 8 | 7 | 0.079 ms |

* A1→A2：扩展节点 17→14（减少 17.6%），代价不变，说明启发信息只省搜索、不改最优性
* A4→A3：扩展节点 15→9（减少 40%），加权地图上启发信息的作用更明显
* 四组总代价都是 7，说明加权启发函数确实可采纳
* 题目二里步数最短不等于代价最低：一条 8 步、穿越 8 的沼泽的路径代价为 23

### 7.2 博弈搜索对照（B1~B7）

| 编号 | 算法 | H | 访问顺序 | 访问节点 | 叶子 | 剪枝次数 | 跳过分支 | 首步 | 收益差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B1 | Minimax | 2 | 从左到右 | 17 | 12 | – | – | T2 | -1 |
| B2 | Alpha-Beta | 2 | 从左到右 | 13 | 8 | 2 | 4 | T2 | -1 |
| B3 | Minimax | 4 | 从左到右 | 65 | 24 | – | – | T3 | 3 |
| B4 | Alpha-Beta | 4 | 从左到右 | 47 | 16 | 5 | 6 | T3 | 3 |
| B5 | Alpha-Beta | 4 | MAX 优先高估值 | 39 | 13 | 5 | 7 | T3 | 3 |
| B6 | Alpha-Beta | 4 | MIN 优先低估值 | 43 | 14 | 7 | 8 | T3 | 3 |
| B7 | Alpha-Beta | 4 | 双向启发式 | 37 | 12 | 6 | 8 | T3 | 3 |

* B1→B2：H=2 时省掉 4 个节点（-23.5%）；B3→B4：H=4 时省掉 18 个（-27.7%）
* 深度越深，剪枝省得越多：H=2 的剪枝机会只有 2 处，H=4 增加到 5 处
* B4→B5（只排 MAX 层）-17.0%，B4→B6（只排 MIN 层）-8.5%，B7（两层都排）-21.3%
* MAX 层排序比 MIN 层更有效：根节点是 MAX，先试最像好棋的分支能立刻抬高 alpha，
  对手层的 beta 随即被压低，剪枝来得更早
* **B1~B7 的最优首步与收益差完全一致**，剪枝和排序只改变开销，不改变结果

### 7.3 题目三的访问顺序（C1~C4）

题三给定的树共 15 个节点、8 个叶子；随机树为 3 叉 4 层（121 个节点、81 个叶子）。

| 顺序 | 给定树节点 | 给定树剪枝 | 随机树节点 | 随机树剪枝 | 博弈值 |
| --- | --- | --- | --- | --- | --- |
| 从左到右 | 14 | 1 次 / 1 个 | 77 | 14 次 / 44 个 | 5 / 3 |
| MAX 优先高估值 | 14 | 1 次 / 1 个 | 68 | 14 次 / 53 个 | 5 / 3 |
| MIN 优先低估值 | 11 | 2 次 / 4 个 | 53 | 13 次 / 68 个 | 5 / 3 |
| 双向启发式 | 11 | 2 次 / 4 个 | 49 | 12 次 / 72 个 | 5 / 3 |

* 给定树上"MAX 优先"与基线完全相同：该树的剪枝点在 MAX 节点且只剩一个叶子，
  叶子怎么排都只能剪掉这一个，排序改变不了结果
* 随机树上四种顺序是 77 → 68 → 53 → 49，双向排序比基线少 36.4% 的节点
* 两棵树上"MIN 优先"都比"MAX 优先"更有效，与 B5/B6 的结论相反。
  原因在于剪枝发生的位置：树搜索的剪枝点在 MIN 节点，MIN 的孩子按估值升序访问
  能最快压低 beta；而题目四的根是 MAX，先搜好分支才能抬高 alpha
* 四种顺序的博弈值始终相同（5 与 3），说明排序不影响正确性

### 7.4 地图规模的影响（D1~D10）

同一张随机地图（障碍密度 25%）上比较 h=0 与 Manhattan：

| 边长 | h=0 扩展 | Manhattan 扩展 | 扩展降幅 | h=0 耗时 | Manhattan 耗时 |
| --- | --- | --- | --- | --- | --- |
| 10×10 | 79 | 23 | 70.9% | 0.24 ms | 0.09 ms |
| 20×20 | 287 | 61 | 78.7% | 0.89 ms | 0.23 ms |
| 30×30 | 662 | 141 | 78.7% | 2.09 ms | 0.55 ms |
| 40×40 | 1206 | 145 | 88.0% | 3.86 ms | 0.60 ms |
| 50×50 | 1878 | 175 | 90.7% | 6.01 ms | 0.75 ms |

* h=0 退化为 Dijkstra，扩展节点数随可通行格数量线性增长，50×50 时几乎铺满整张图
* Manhattan 的增长明显更平缓：40×40 之后扩展数基本饱和（145 → 175），
  因为搜索被限制在起点与终点之间的带状区域
* 规模越大，启发信息的相对收益越高：10×10 省 71%，50×50 省 91%
* 两者求出的路径长度与总代价完全相同，只是搜索范围不同

---

## 8. 报告图表

`results/report_figures/` 下 12 张 PNG（300 dpi，全中文标注）：

| 文件 | 内容 |
| --- | --- |
| `fig1_1_astar_flowchart` | A\* 主循环流程（Open/Closed 表、重新打开、路径回溯） |
| `fig1_2_search_flowchart` | Minimax 与 Alpha-Beta 共用的搜索流程与剪枝分支 |
| `fig2_1_maze_path` | 题目一的最短路径与已扩展区域 |
| `fig2_2_astar_heuristic` | A1~A4 的扩展节点数与生成节点数 |
| `fig2_3_weighted_terrain` | 题目二地形权值、最优路径与总代价 |
| `fig3_1_alpha_beta_tree` | 题三博弈树上的剪枝位置与 alpha/beta 数值 |
| `fig3_2_tree_order` | 四种访问顺序在两棵树上的节点数 |
| `fig4_1_distance_heatmap` | 6×6 最小移动代价矩阵 |
| `fig4_2_treasure_tree` | H=2 的博弈树、净收益与终局 U |
| `fig4_3_minimax_vs_ab` | 同深度下 Minimax 与 Alpha-Beta 的节点数 |
| `fig5_1_scaling` | 地图规模对扩展节点数与耗时的影响 |
| `fig5_2_order_summary` | 四种访问顺序相对基线的节点降幅 |

图表数字与 CSV 同源：`main.py experiments` 先写 CSV，绘图时优先读 CSV
（`_load_rows()`），所以报告里的数字、图上的标注、CSV 的记录三者一致。
单独执行 `main.py figures` 也复用已有 CSV，不会因为重跑而改数。

文件名按所在章节编号（`fig2_3` 即第 2 节第 3 张）。实验报告
`report/实验二报告_李麒_10245501423.pdf`（5 页 A4）按正文出现顺序把它们编为图 1~图 6，
收录了其中 6 张（`fig1_1`、`fig2_1`、`fig2_3`、`fig3_1`、`fig4_2`、`fig5_1`）。
其余 6 张留在本目录备查，报告里对应的信息由正文与表格承担，避免同义图表重复占版面：
`fig1_2`（搜索流程）见 1.1 与 4.2 节文字，`fig2_2`（A1~A4 节点数）对应表 1，
`fig3_2`（顺序节点数）与 `fig5_2`（顺序降幅）对应表 4，`fig4_1`（距离矩阵）对应表 5，
`fig4_3`（Minimax 与 Alpha-Beta 对比）对应表 6。报告源文件为
`report/report.html`，用 Chrome 打印即可重新生成 PDF：

```bash
chrome --headless --print-to-pdf=report/实验二报告_李麒_10245501423.pdf report/report.html
```

---

## 9. 复现说明

| 项目 | 取值 |
| --- | --- |
| 输入数据 | 四道题的用例写在 `problems/*.py` 顶部常量中，与 PDF 一致 |
| A\* 启发函数 | 题一 `manhattan`；题二 `weighted_manhattan`；题四距离矩阵 `weighted_manhattan` |
| A\* 代价规则 | 题一 `uniform`；题二、四 `weighted`（进入格权值，`E` 为 0） |
| 搜索深度 | 题四 `H=2` 与 `H=4` |
| 访问顺序 | 正式结果用 `left_to_right`，其余三种仅用于对照 |
| tie-break | `(f, -g, 入队序号)`，邻居顺序 上→下→左→右 |
| 随机性 | 只在随机迷宫与随机博弈树中出现，均固定种子（迷宫 `1000+边长`，树 `seed=1`） |
| 计时 | `time.perf_counter()`，A\* 200 次、博弈 100 次、树搜索 300 次取均值 |
| CSV 编码 | `utf-8-sig`，Excel 直接打开不乱码 |

绝对时间依赖机器状态，每次运行会有波动；节点数、剪枝次数、博弈值完全确定，
分析结论以这三类指标为主。

---

## 10. 已知局限

* **计时精度**：题目一、二的地图很小，单次运行在几十微秒量级，A1 与 A2 的耗时
  差异（0.048 ms 与 0.048 ms）落在测量噪声内，不能据此判断启发函数更慢。
  真正能体现差距的是 D1~D10 的规模实验。
* **题目四的估值函数**：`evaluate()` 用 `Score_A - Score_B`，没有考虑剩余宝藏的
  潜在归属，H=2 与 H=4 的差异完全来自搜索深度。若加入"剩余宝藏价值估计"，
  浅层搜索也可能做出接近深层的判断，但会牺牲可解释性，本实验未采用。
* **距离矩阵的静态性**：博弈过程中双方位置会变，但矩阵按初始位置算一次。
  题目四的规则里"已领取宝藏的格子仍可通行"，因此从当前位置查表即可，
  不需要重算；若规则改成"领取后变障碍"，矩阵就必须在线更新。
* **随机迷宫只保证连通**：不保证存在多条不同代价的路径，因此在某些规模下
  Manhattan 与 h=0 的路径长度完全相同，对照的是搜索范围而非路径质量。
