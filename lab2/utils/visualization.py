"""绘图基础设施。

本模块只负责两件事：加载 matplotlib 后端、配置中文字体。具体图表的内容
与样式由 :mod:`utils.report_figures` 负责。

早期版本曾内置一套纯标准库的 SVG 渲染后端，用于在缺少 matplotlib 的环境下
出图。报告图表改为统一由 matplotlib 输出 300 dpi PNG 后，那套后端已无调用
者，故移除，避免留下无法验证的死代码。
"""

from __future__ import annotations

from typing import Optional

# 依次尝试的中文字体；命中第一个已安装的即可
CJK_FONT_CANDIDATES = (
    "Microsoft YaHei",
    "SimHei",
    "Source Han Sans SC",
    "Noto Sans CJK SC",
    "DengXian",
    "SimSun",
)


def _pyplot():
    """导入 matplotlib 的 pyplot 并切到无界面后端，失败返回 ``None``。"""
    try:  # pragma: no cover - 取决于运行环境
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception:  # pragma: no cover
        return None


def setup_chinese_font() -> Optional[str]:
    """为 matplotlib 配置可用的中文字体，返回选中的字体名（无则返回 ``None``）。

    同时把 ``axes.unicode_minus`` 设为 ``False``，避免负号显示成方块。
    """
    plt = _pyplot()
    if plt is None:
        return None
    from matplotlib import font_manager, rcParams

    available = {font.name for font in font_manager.fontManager.ttflist}
    chosen = next((name for name in CJK_FONT_CANDIDATES if name in available), None)

    rcParams["font.sans-serif"] = ([chosen] if chosen else []) + [
        "DejaVu Sans", "Arial", "sans-serif",
    ]
    rcParams["axes.unicode_minus"] = False
    rcParams["figure.facecolor"] = "white"
    rcParams["savefig.facecolor"] = "white"
    return chosen
