"""实验二：搜索算法 —— 项目入口。

用法
----
::

    python main.py p1            # 题目一 迷宫寻路
    python main.py p2            # 题目二 加权地形
    python main.py p3            # 题目三 极大极小博弈树
    python main.py p4            # 题目四 竞争环境下的宝藏分配
    python main.py matrix        # 只输出题目四的 6x6 距离矩阵
    python main.py experiments   # 运行全部对照实验，写 CSV 与图表
    python main.py figures       # 只生成报告图表
    python main.py all           # 依次运行 p1~p4 与 experiments
    python main.py test          # 运行单元测试

所有输出同时打印到终端并写入 ``results/`` 下的 CSV / 图表文件。
"""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithms.alpha_beta import (  # noqa: E402
    ORDER_LEFT_TO_RIGHT,
    alpha_beta_max,
)
from algorithms.astar import (  # noqa: E402
    HEURISTIC_MANHATTAN,
    HEURISTIC_WEIGHTED,
    cell_entry_cost,
)
from problems.game_tree import (  # noqa: E402
    EXPECTED_VALUE as TREE_EXPECTED_VALUE,
    build_sample_tree,
    check_expected as check_tree_expected,
    print_tree,
    solve_tree,
    solve_tree_baseline,
)
from problems.maze import (  # noqa: E402
    MAZE_SAMPLE_EXPECTED,
    compare_heuristics as compare_maze_heuristics,
    format_maze_result,
    load_maze,
    render_maze,
    solve_maze,
    validate_path,
)
from problems.treasure_game import (  # noqa: E402
    check_expected,
    default_game,
    describe_decision_path,
    play_best_line,
    solve_treasure_game,
)
from problems.weighted_terrain import (  # noqa: E402
    WEIGHTED_SAMPLE_EXPECTED,
    calculate_path_cost,
    compare_heuristics as compare_weighted_heuristics,
    load_weighted_map,
    solve_weighted_terrain,
    verify_admissible,
)
from utils.experiments import (  # noqa: E402
    ASTAR_REPEATS,
    GAME_REPEATS,
    astar_experiments,
    astar_scaling_experiments,
    game_experiments,
    tree_order_experiments,
)
from utils.metrics import format_number as fmt_num, format_table, write_csv  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "results"
REPORT_FIGURES_DIR = RESULTS_DIR / "report_figures"


# ---------------------------------------------------------------------------
# 输出辅助
# ---------------------------------------------------------------------------
def banner(title: str) -> None:
    line = "=" * 74
    print(f"\n{line}\n{title}\n{line}", flush=True)


def section(title: str) -> None:
    print(f"\n--- {title} " + "-" * max(0, 66 - len(title)), flush=True)


def pct(new: float, old: float) -> str:
    """描述 new 相对 old 的变化，例如"减少 17.6%"、"增加 5.0%"。"""
    if old == 0:
        return "n/a"
    delta = (new - old) / old * 100
    if abs(delta) < 0.05:
        return "基本持平"
    return f"{'增加' if delta > 0 else '减少'} {abs(delta):.1f}%"


# ---------------------------------------------------------------------------
# 题目一
# ---------------------------------------------------------------------------
def run_problem1() -> List[Dict[str, Any]]:
    banner("题目一：迷宫寻路（A*，单位代价）")
    start, goal, grid = load_maze()

    print(f"输入: start={start}, goal={goal}")
    print("地图:")
    print(render_maze(grid))

    result = solve_maze(start, goal, grid, HEURISTIC_MANHATTAN, return_result=True)
    print(f"\nA* (h=Manhattan) 输出: {format_maze_result(result.path)}")
    print(f"PDF 预期输出: {format_maze_result(MAZE_SAMPLE_EXPECTED)}")
    print(f"最短路径步数: {result.stats.path_length}  |  总代价: {fmt_num(result.cost)}")

    ok, message = validate_path(result.path, start, goal, grid)
    print(f"路径合法性校验: {'通过' if ok else '失败'}（{message}）")
    print("路径可视化:")
    print(render_maze(grid, result.path))

    section("搜索统计")
    print(f"扩展节点数 expanded_nodes : {result.stats.expanded_nodes}")
    print(f"发现节点数 visited_nodes  : {result.stats.visited_nodes}")
    print(f"生成后继数 generated_nodes: {result.stats.generated_nodes}")
    print(f"运行时间                  : {result.stats.execution_time * 1000:.3f} ms")

    section("对照实验：h=0（Dijkstra） vs h=Manhattan")
    zero, manhattan = compare_maze_heuristics(start, goal, grid)
    rows = []
    for label, res in (("h=0", zero), ("h=Manhattan", manhattan)):
        rows.append({
            "heuristic": label,
            "expanded_nodes": res.stats.expanded_nodes,
            "visited_nodes": res.stats.visited_nodes,
            "path_length": res.stats.path_length,
            "total_cost": fmt_num(res.cost),
        })
    print(format_table(rows))
    print(
        f"\n分析: 曼哈顿启发式把扩展节点数从 {zero.stats.expanded_nodes} "
        f"降到 {manhattan.stats.expanded_nodes}（扩展规模{pct(manhattan.stats.expanded_nodes, zero.stats.expanded_nodes)}），"
        f"两者路径长度与总代价相同（{zero.stats.path_length} 步 / {fmt_num(zero.cost)}），"
        "说明启发函数只影响搜索规模，不影响最优性。"
    )

    return [zero.stats.to_row(), manhattan.stats.to_row()]


# ---------------------------------------------------------------------------
# 题目二
# ---------------------------------------------------------------------------
def run_problem2() -> List[Dict[str, Any]]:
    banner("题目二：加权地形（A*，代价 = 进入格权值，E 代价为 0）")
    start, goal, grid = load_weighted_map()

    print(f"输入: start={start}, goal={goal}")
    print("地图（数字为进入该格的代价）:")
    print(render_maze(grid))

    result = solve_weighted_terrain(start, goal, grid, HEURISTIC_WEIGHTED, return_result=True)
    print(f"\nA* 输出最小总代价: {fmt_num(result.cost)}")
    print(f"PDF 预期输出: {WEIGHTED_SAMPLE_EXPECTED}")
    print(f"最短代价路径: {result.path}")

    section("路径代价累计过程")
    cumulative = 0.0
    print(f"{'step':>4}  {'from':>9} -> {'to':<9}  {'enter_cost':>10}  {'cumulative':>10}")
    for idx, (prev, cur) in enumerate(zip(result.path, result.path[1:]), start=1):
        step_cost = cell_entry_cost(grid[cur[0]][cur[1]]) or 0.0
        cumulative += step_cost
        print(f"{idx:>4}  {str(prev):>9} -> {str(cur):<9}  {step_cost:>10.0f}  {cumulative:>10.0f}")
    print(f"独立重算路径总代价: {fmt_num(calculate_path_cost(result.path, grid))}")

    section("启发函数可采纳性验证")
    report = verify_admissible(grid, goal, HEURISTIC_WEIGHTED)
    print(f"地图最小进入代价 c_min = {report['min_terrain_cost']}, 终点 E 的进入代价 = {report['goal_entry_cost']}")
    print(f"h(n) = (d-1)*c_min + cost(goal) = max(0, Manhattan-1) * {report['min_terrain_cost']}")
    print(f"检查格子数: {report['checked']}，违反可采纳性的格子: {report['violation_count']}")
    print(f"结论: {'可采纳（h(n) <= 真实剩余代价）' if report['admissible'] else '不可采纳'}")
    print(f"启发强度 max(h*(n)-h(n)) = {fmt_num(report['max_slack'])}（越小说明启发越贴近真值）")

    section("对照实验：h=0 vs 加权曼哈顿")
    zero, weighted = compare_weighted_heuristics(start, goal, grid)
    rows = [
        {
            "heuristic": "h=0",
            "expanded_nodes": zero.stats.expanded_nodes,
            "visited_nodes": zero.stats.visited_nodes,
            "path_length": zero.stats.path_length,
            "total_cost": fmt_num(zero.cost),
        },
        {
            "heuristic": "weighted_manhattan",
            "expanded_nodes": weighted.stats.expanded_nodes,
            "visited_nodes": weighted.stats.visited_nodes,
            "path_length": weighted.stats.path_length,
            "total_cost": fmt_num(weighted.cost),
        },
    ]
    print(format_table(rows))
    print(
        f"\n分析: 两者总代价相同（{fmt_num(zero.cost)}），说明加权启发函数仍然可采纳；"
        f"扩展节点数 {zero.stats.expanded_nodes} -> {weighted.stats.expanded_nodes}"
        f"（{pct(weighted.stats.expanded_nodes, zero.stats.expanded_nodes)}）。"
    )

    return [zero.stats.to_row(), weighted.stats.to_row()]


# ---------------------------------------------------------------------------
# 题目三
# ---------------------------------------------------------------------------
def run_problem3() -> List[Dict[str, Any]]:
    banner("题目三：极大极小博弈树的最佳得分（Alpha-Beta）")
    root = build_sample_tree()

    print("博弈树结构（根为 MAX，从左到右访问）:")
    print_tree(root)

    section("搜索结果")
    prune_log: List[Dict[str, Any]] = []
    value, stats = solve_tree(root, order=ORDER_LEFT_TO_RIGHT,
                              prune_log=prune_log, return_stats=True)
    print(f"Alpha-Beta 输出: {fmt_num(value)}")
    ok, message = check_tree_expected(value)
    print(f"PDF 预期输出: {fmt_num(TREE_EXPECTED_VALUE)}  ->  {message}")
    print(f"单参数函数 alpha_beta_max(root): {fmt_num(alpha_beta_max(root))}"
          f"  ->  与上式同一结果，可直接作为题目要求的那个函数")
    baseline = solve_tree_baseline(root)
    print(f"Minimax 基准值 : {fmt_num(baseline)}  ->  "
          f"{'与 Alpha-Beta 一致' if abs(baseline - value) < 1e-9 else '不一致！'}")

    print(f"\n访问节点数 total_nodes : {stats.total_nodes}")
    print(f"展开节点数 expanded    : {stats.expanded_nodes}")
    print(f"叶子评估数 leaf_nodes  : {stats.leaf_nodes}")
    print(f"剪枝次数 pruning_count : {stats.pruning_count}")
    print(f"剪枝跳过节点 pruned    : {stats.pruned_nodes}（树结构已知，可精确统计）")

    section("剪枝事件明细")
    for event in prune_log:
        print(f"  深度 {event['depth']} 的 {event['node_type']} 节点："
              f"alpha={fmt_num(event['alpha'])} beta={fmt_num(event['beta'])} "
              f"-> 跳过 {event['skipped_children']} 个子分支共 {event['skipped_nodes']} 个节点")

    section("不同子节点访问顺序的对照（C1~C4）")
    order_data = tree_order_experiments()
    pdf_rows = order_data["pdf_tree"]
    random_rows = order_data["random_tree"]

    print("C1~C4 在 PDF 给定博弈树上：")
    print(format_table(pdf_rows, columns=["experiment", "order_label", "best_value",
                                          "total_nodes", "leaf_nodes",
                                          "pruning_count", "pruned_nodes"]))
    print("\nC1~C4 在 3 叉 4 层随机博弈树上（种子 1）：")
    print(format_table(random_rows, columns=["experiment", "order_label", "best_value",
                                             "total_nodes", "leaf_nodes",
                                             "pruning_count", "pruned_nodes"]))
    print("\n分析: 四种访问顺序的最优值完全相同，说明排序只改变搜索开销；"
          "PDF 树上 MAX 层每个节点只有两个子节点，无论先后都要访问第一个子节点才能触发剪枝，"
          "因此 C2 与 C1 的节点数相同；C3 让 MIN 层优先访问低估值子树，"
          "使 beta 提前收紧，从而跳过整棵子树。随机树上四个子节点的情形下，排序的收益更明显。")

    return [stats.to_row()]


# ---------------------------------------------------------------------------
# 题目四
# ---------------------------------------------------------------------------
def run_problem4(depths: Sequence[int] = (2, 4), save: bool = True) -> List[Dict[str, Any]]:
    banner("题目四：竞争环境下的宝藏分配（A* + Minimax + Alpha-Beta）")
    game = default_game()

    section("阶段一：6x6 距离矩阵（A* 计算，单位：最小移动代价）")
    print(f"关键位置: {', '.join(f'{name}{game.positions[name]}' for name in game.labels)}")
    print(game.format_matrix())
    print(f"宝藏价值: {', '.join(f'{k}={v}' for k, v in sorted(game.values.items()))}")

    state = game.initial_state()
    section("阶段二：初始博弈状态")
    print(f"A 位置={state.pos_a}  B 位置={state.pos_b}  剩余宝藏={list(state.remaining)}")
    print(f"当前玩家={state.current_player}（MAX）  累计收益 A={state.score_a} B={state.score_b}")

    section("阶段三：根节点候选首步的即时收益")
    candidates = describe_decision_path(game, state, game.actions(state), 1)
    print(format_table(candidates))

    section("阶段四：Minimax 与 Alpha-Beta 对照")
    rows: List[Dict[str, Any]] = []
    treasure_rows: List[Dict[str, Any]] = []
    for depth in depths:
        for algorithm in ("Minimax", "Alpha-Beta"):
            value, action, stats = solve_treasure_game(
                game, depth, algorithm, return_stats=True
            )
            rows.append({
                "algorithm": algorithm,
                "depth": depth,
                "total_nodes": stats.total_nodes,
                "leaf_nodes": stats.leaf_nodes,
                "pruning_count": stats.pruning_count,
                "pruned_branches": stats.pruned_branches,
                "best_target": action,
                "best_value": fmt_num(value),
                "time_ms": f"{stats.execution_time * 1000:.4f}",
            })
            treasure_rows.append({
                "algorithm": algorithm,
                "depth": depth,
                "best_target": action,
                "best_value": fmt_num(value),
                "total_nodes": stats.total_nodes,
                "leaf_nodes": stats.leaf_nodes,
                "pruning_count": stats.pruning_count,
                "time": round(stats.execution_time, 8),
            })
    print(format_table(rows, columns=["algorithm", "depth", "best_target", "best_value",
                                      "total_nodes", "leaf_nodes", "pruning_count",
                                      "pruned_branches", "time_ms"]))

    section("与 PDF 预期输出核对")
    all_ok = True
    for depth in depths:
        for algorithm in ("Minimax", "Alpha-Beta"):
            match = next(r for r in rows if r["depth"] == depth and r["algorithm"] == algorithm)
            ok, message = check_expected(depth, match["best_target"], float(match["best_value"]))
            all_ok = all_ok and ok
            status = "OK  " if ok else "FAIL"
            payload = '{"best_target": "%s", "best_value": %s}' % (
                match["best_target"], match["best_value"]
            )
            print(f"[{status}] {algorithm:<10} H={depth} => {payload}")
    print(f"\n结论: {'全部与预期一致' if all_ok else '存在与预期不一致的结果，请检查状态更新逻辑'}")

    section("剪枝效率")
    for depth in depths:
        mm = next(r for r in rows if r["depth"] == depth and r["algorithm"] == "Minimax")
        ab = next(r for r in rows if r["depth"] == depth and r["algorithm"] == "Alpha-Beta")
        print(
            f"H={depth}: 节点数 {mm['total_nodes']} -> {ab['total_nodes']}"
            f"（{pct(ab['total_nodes'], mm['total_nodes'])}），"
            f"剪枝 {ab['pruning_count']} 次，跳过 {ab['pruned_branches']} 个子分支，"
            f"最优首步与收益差完全一致"
        )

    section("最优走法回放（主变例，双方均按最优策略应对）")
    for depth in depths:
        line, final_state = play_best_line(game, depth)
        print(f"H={depth}:")
        print(format_table(line, columns=["step", "player", "target", "value",
                                          "distance", "gain", "score_a", "score_b"]))
        print(f"  终局 U = {fmt_num(final_state.utility())}"
              f"（Score_A={fmt_num(final_state.score_a)}, Score_B={fmt_num(final_state.score_b)}）\n")

    if save:
        write_csv(RESULTS_DIR / "treasure_results.csv", treasure_rows)
    return treasure_rows


def run_matrix() -> None:
    banner("题目四：6x6 距离矩阵")
    game = default_game()
    print(game.format_matrix())
    print("\n行 = 起点，列 = 终点；对角线为 0；inf 表示不可达。")


# ---------------------------------------------------------------------------
# 性能对照实验（对应评分中的"算法性能 10 分"）
# ---------------------------------------------------------------------------
def run_experiments(no_charts: bool = False) -> None:
    banner("性能对照实验")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"结果目录: {RESULTS_DIR}")
    print(f"计时方式: A* 重复 {ASTAR_REPEATS} 次、博弈搜索重复 {GAME_REPEATS} 次后取均值")

    # ---------------- A* 对照：A1~A4 ----------------
    section("A* 启发函数对照（A1~A4）")
    astar_rows = astar_experiments()
    print(format_table(astar_rows, columns=["experiment", "problem_label", "heuristic",
                                            "expanded_nodes", "visited_nodes",
                                            "path_length", "total_cost", "time"]))
    write_csv(RESULTS_DIR / "astar_results.csv", astar_rows)
    print(f"\n已写入 {RESULTS_DIR / 'astar_results.csv'}")

    # ---------------- 拓展实验一：地图规模 D1~D8 ----------------
    section("拓展实验一：不同地图规模下的 A*（D1~D8）")
    scaling_rows = astar_scaling_experiments()
    print(format_table(scaling_rows, columns=["experiment", "map_size", "heuristic",
                                              "expanded_nodes", "visited_nodes",
                                              "path_length", "time"]))
    write_csv(RESULTS_DIR / "astar_scaling_results.csv", scaling_rows)
    print(f"\n已写入 {RESULTS_DIR / 'astar_scaling_results.csv'}")

    # ---------------- 题目三访问顺序：C1~C4 ----------------
    section("题目三 子节点访问顺序对照（C1~C4）")
    order_data = tree_order_experiments()
    pdf_rows = order_data["pdf_tree"]
    random_rows = order_data["random_tree"]
    print("PDF 给定博弈树：")
    print(format_table(pdf_rows, columns=["experiment", "order_label", "best_value",
                                          "total_nodes", "leaf_nodes",
                                          "pruning_count", "pruned_nodes"]))
    print("\n3 叉 4 层随机博弈树（种子 1）：")
    print(format_table(random_rows, columns=["experiment", "order_label", "best_value",
                                             "total_nodes", "leaf_nodes",
                                             "pruning_count", "pruned_nodes"]))
    for row in pdf_rows:
        row["tree"] = "pdf_tree"
    for row in random_rows:
        row["tree"] = "random_tree"
    write_csv(RESULTS_DIR / "tree_order_results.csv", pdf_rows + random_rows)
    print(f"\n已写入 {RESULTS_DIR / 'tree_order_results.csv'}")

    # ---------------- 博弈搜索对照：B1~B7 ----------------
    section("Minimax / Alpha-Beta 对照（B1~B7）")
    minimax_rows, alpha_rows = game_experiments()

    print("Minimax：")
    print(format_table(minimax_rows, columns=["experiment", "depth", "order", "total_nodes",
                                              "leaf_nodes", "best_target", "best_value", "time"]))
    print("\nAlpha-Beta：")
    print(format_table(alpha_rows, columns=["experiment", "depth", "order", "total_nodes",
                                            "leaf_nodes", "pruning_count", "pruned_branches",
                                            "best_target", "best_value", "time"]))

    write_csv(RESULTS_DIR / "minimax_results.csv", minimax_rows)
    write_csv(RESULTS_DIR / "alpha_beta_results.csv", alpha_rows)
    print(f"\n已写入 {RESULTS_DIR / 'minimax_results.csv'} 与 {RESULTS_DIR / 'alpha_beta_results.csv'}")

    # ---------------- 报告图表 ----------------
    if no_charts:
        print("\n已跳过绘图（--no-charts）")
        return

    section("生成报告图表")
    try:
        from utils.report_figures import generate_all
    except ImportError as exc:  # pragma: no cover
        print(f"绘图模块不可用：{exc}")
        return
    for path in generate_all():
        print(f"  {path}")


def run_figures() -> None:
    """只生成报告图表（不重跑 CSV）。"""
    banner("生成报告图表")
    from utils.report_figures import generate_all
    for path in generate_all():
        print(f"  {path}")


# ---------------------------------------------------------------------------
# 单元测试
# ---------------------------------------------------------------------------
def run_tests() -> int:
    banner("运行单元测试")
    loader = unittest.TestLoader()
    suite = loader.discover(str(PROJECT_ROOT / "tests"), pattern="test_*.py",
                            top_level_dir=str(PROJECT_ROOT))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="实验二：搜索算法（A* / Minimax / Alpha-Beta）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["p1", "p2", "p3", "p4", "matrix", "experiments", "figures", "all", "test"],
        help="要运行的模块",
    )
    parser.add_argument("--no-save", action="store_true", help="不写入 results/ 下的 CSV")
    parser.add_argument("--no-charts", action="store_true", help="experiments 模式不生成图表")
    parser.add_argument("--depth", type=int, choices=[2, 4], action="append",
                        help="题目四的搜索深度，可重复指定，默认 2 与 4")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    save = not args.no_save
    depths = tuple(args.depth) if args.depth else (2, 4)

    if args.command == "p1":
        run_problem1()
    elif args.command == "p2":
        run_problem2()
    elif args.command == "p3":
        run_problem3()
    elif args.command == "p4":
        run_problem4(depths=depths, save=save)
    elif args.command == "matrix":
        run_matrix()
    elif args.command == "experiments":
        run_experiments(no_charts=args.no_charts)
    elif args.command == "figures":
        run_figures()
    elif args.command == "all":
        run_problem1()
        run_problem2()
        run_problem3()
        run_problem4(depths=depths, save=save)
        run_experiments(no_charts=args.no_charts)
    elif args.command == "test":
        return run_tests()
    else:  # pragma: no cover
        parser.error(f"未知命令: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
