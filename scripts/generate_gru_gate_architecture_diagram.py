"""生成项目默认 GRU 时序编码器及门控计算架构图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import FancyArrowPatch
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
FIGURE_DIR = ROOT / "docs" / "figures"
PNG_PATH = FIGURE_DIR / "algorithm-gru-gate-architecture.png"
SVG_PATH = FIGURE_DIR / "algorithm-gru-gate-architecture.svg"

BG = "#0b0f14"
FG = "#f1f5f9"
MUTED = "#aeb8c6"
GRID = "#334155"
CYAN = "#58c7df"
TEAL = "#55d6be"
GREEN = "#79d98c"
ORANGE = "#f2aa63"
BLUE = "#70a7ff"
PURPLE = "#bd91ff"


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    edge: str,
    face: str = BG,
    fontsize: float = 13,
    linewidth: float = 2.0,
    radius: float = 0.015,
    text_color: str = FG,
    zorder: int = 4,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        color=text_color,
        fontsize=fontsize,
        zorder=zorder + 1,
    )
    return patch


def circle(
    ax,
    x: float,
    y: float,
    label: str,
    *,
    edge: str,
    radius: float = 0.020,
    fontsize: float = 13,
    face: str = BG,
    zorder: int = 5,
) -> Circle:
    patch = Circle(
        (x, y),
        radius,
        edgecolor=edge,
        facecolor=face,
        linewidth=2.2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x,
        y,
        label,
        ha="center",
        va="center",
        color=FG,
        fontsize=fontsize,
        zorder=zorder + 1,
    )
    return patch


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    width: float = 1.7,
    connectionstyle: str = "arc3,rad=0",
    zorder: int = 2,
    mutation_scale: float = 13,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=width,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def line(ax, start, end, *, color=MUTED, width=1.5, zorder=1) -> None:
    ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color=color,
        linewidth=width,
        zorder=zorder,
    )


def label(ax, x, y, text, *, color=FG, size=12, weight="normal", ha="center") -> None:
    ax.text(
        x,
        y,
        text,
        color=color,
        fontsize=size,
        fontweight=weight,
        ha=ha,
        va="center",
    )


def build() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "mathtext.fontset": "dejavusans",
            "svg.fonttype": "none",
        }
    )

    fig, ax = plt.subplots(figsize=(16, 10), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    label(ax, 0.5, 0.965, "GRU 时序编码器与门控结构", size=25, weight="bold")
    label(
        ax,
        0.5,
        0.930,
        "项目默认配置：历史长度 20  ·  单帧观测 28 维  ·  单层 GRU  ·  隐藏宽度 64",
        color=MUTED,
        size=13,
    )

    # 上部：20 个时间步展开和最终投影。
    label(ax, 0.035, 0.892, "一、时间展开", color=CYAN, size=16, weight="bold", ha="left")
    label(ax, 0.035, 0.858, "所有时间步共享同一组 GRU 参数", color=MUTED, size=11, ha="left")

    time_x = [0.13, 0.30, 0.52, 0.68]
    time_names = ["1", "2", "…", "20"]
    for x, name in zip(time_x, time_names):
        if name == "…":
            label(ax, x, 0.785, "···", color=MUTED, size=25)
            label(ax, x, 0.710, "···", color=MUTED, size=25)
            continue
        box(
            ax,
            x,
            0.785,
            0.105,
            0.060,
            rf"GRU 单元  $t={name}$",
            edge=TEAL,
            fontsize=12,
        )
        box(
            ax,
            x,
            0.690,
            0.090,
            0.052,
            rf"$x_{{{name}}}$" + "\n" + r"$B\times28$",
            edge=CYAN,
            fontsize=11,
        )
        arrow(ax, (x, 0.717), (x, 0.752), color=CYAN)

    box(ax, 0.045, 0.785, 0.060, 0.052, "$h_{0}$" + "\n" + r"$B\times64$", edge=BLUE, fontsize=10)
    label(ax, 0.045, 0.740, "全零初始化", color=MUTED, size=9)
    arrow(ax, (0.075, 0.785), (time_x[0] - 0.055, 0.785), color=BLUE)

    arrow(ax, (time_x[0] + 0.055, 0.785), (time_x[1] - 0.055, 0.785), color=BLUE)
    arrow(ax, (time_x[1] + 0.055, 0.785), (0.475, 0.785), color=BLUE)
    arrow(ax, (0.565, 0.785), (time_x[3] - 0.055, 0.785), color=BLUE)
    label(ax, 0.215, 0.825, r"$h_{1}$：$B\times64$", color=BLUE, size=10)
    label(ax, 0.400, 0.825, "$h_{2}$", color=BLUE, size=10)
    label(ax, 0.615, 0.825, "$h_{19}$", color=BLUE, size=10)

    box(
        ax,
        0.825,
        0.785,
        0.165,
        0.070,
        "特征投影\nLinear 64 → 64 + GELU",
        edge=PURPLE,
        fontsize=12,
    )
    arrow(ax, (time_x[3] + 0.055, 0.785), (0.742, 0.785), color=BLUE)
    label(ax, 0.735, 0.825, r"$h_{20}$：$B\times64$", color=BLUE, size=10)
    box(
        ax,
        0.955,
        0.785,
        0.075,
        0.060,
        "$f$" + "\n" + r"$B\times64$",
        edge=PURPLE,
        fontsize=12,
    )
    arrow(ax, (0.908, 0.785), (0.915, 0.785), color=PURPLE)
    label(ax, 0.955, 0.735, "共享给 Actor 和 Critic", color=MUTED, size=10)

    line(ax, (0.03, 0.640), (0.97, 0.640), color=GRID, width=1.2)
    label(ax, 0.035, 0.610, "二、单个 GRU 单元的门控计算", color=CYAN, size=16, weight="bold", ha="left")
    label(
        ax,
        0.965,
        0.610,
        "以下所有门和状态均为 64 维",
        color=MUTED,
        size=11,
        ha="right",
    )

    # 左侧输入与分流总线。
    box(ax, 0.075, 0.515, 0.090, 0.060, "$x_t$" + "\n" + r"$B\times28$", edge=CYAN, fontsize=12)
    box(ax, 0.075, 0.330, 0.105, 0.060, "$h_{t-1}$" + "\n" + r"$B\times64$", edge=BLUE, fontsize=12)
    line(ax, (0.125, 0.515), (0.165, 0.515), color=CYAN, width=2.0)
    line(ax, (0.165, 0.515), (0.165, 0.215), color=CYAN, width=1.5)
    line(ax, (0.130, 0.330), (0.205, 0.330), color=BLUE, width=2.0)
    line(ax, (0.205, 0.330), (0.205, 0.185), color=BLUE, width=1.5)
    line(ax, (0.205, 0.330), (0.205, 0.555), color=BLUE, width=1.5)

    gate_rows = [
        (0.535, TEAL, "重置门", "$r_t$", "$\sigma$"),
        (0.405, GREEN, "更新门", "$z_t$", "$\sigma$"),
    ]
    gate_outputs: dict[str, tuple[float, float]] = {}
    for y, color, name, symbol, activation in gate_rows:
        box(ax, 0.295, y, 0.150, 0.058, "输入仿射 + 隐藏仿射", edge=color, fontsize=10)
        circle(ax, 0.405, y, "+", edge=color, radius=0.017, fontsize=11)
        box(ax, 0.480, y, 0.070, 0.052, activation, edge=color, fontsize=14)
        box(ax, 0.570, y, 0.068, 0.052, symbol, edge=color, fontsize=14)
        label(ax, 0.570, y + 0.045, name, color=color, size=11, weight="bold")
        arrow(ax, (0.165, y), (0.218, y), color=CYAN, width=1.4)
        arrow(ax, (0.205, y - 0.020), (0.218, y - 0.010), color=BLUE, width=1.4)
        arrow(ax, (0.370, y), (0.387, y), color=color)
        arrow(ax, (0.422, y), (0.445, y), color=color)
        arrow(ax, (0.515, y), (0.536, y), color=color)
        gate_outputs[name] = (0.604, y)

    # 候选状态：输入分支与被重置门调制的隐藏分支。
    candidate_y = 0.255
    box(ax, 0.285, candidate_y + 0.030, 0.130, 0.050, "输入仿射", edge=ORANGE, fontsize=10)
    box(ax, 0.285, candidate_y - 0.055, 0.130, 0.050, "隐藏仿射", edge=ORANGE, fontsize=10)
    circle(ax, 0.405, candidate_y - 0.055, "$\odot$", edge=TEAL, radius=0.018, fontsize=12)
    circle(ax, 0.475, candidate_y, "+", edge=ORANGE, radius=0.017, fontsize=11)
    box(ax, 0.555, candidate_y, 0.075, 0.052, "tanh", edge=ORANGE, fontsize=12)
    box(ax, 0.650, candidate_y, 0.070, 0.052, "$n_t$", edge=ORANGE, fontsize=14)
    arrow(ax, (0.165, candidate_y + 0.030), (0.218, candidate_y + 0.030), color=CYAN, width=1.4)
    arrow(ax, (0.205, candidate_y - 0.055), (0.218, candidate_y - 0.055), color=BLUE, width=1.4)
    arrow(ax, (0.350, candidate_y - 0.055), (0.387, candidate_y - 0.055), color=ORANGE)
    line(ax, gate_outputs["重置门"], (0.610, gate_outputs["重置门"][1]), color=TEAL, width=1.6)
    line(ax, (0.610, gate_outputs["重置门"][1]), (0.610, 0.205), color=TEAL, width=1.6)
    arrow(ax, (0.610, 0.205), (0.405, candidate_y - 0.035), color=TEAL, connectionstyle="arc3,rad=0.10")
    arrow(ax, (0.423, candidate_y - 0.055), (0.458, candidate_y - 0.010), color=ORANGE)
    arrow(ax, (0.350, candidate_y + 0.030), (0.458, candidate_y + 0.010), color=ORANGE)
    arrow(ax, (0.492, candidate_y), (0.517, candidate_y), color=ORANGE)
    arrow(ax, (0.593, candidate_y), (0.615, candidate_y), color=ORANGE)

    # 隐藏状态融合：旧状态保留支路 + 新候选写入支路。
    circle(ax, 0.750, 0.405, "$\odot$", edge=GREEN, radius=0.019, fontsize=12)
    circle(ax, 0.750, 0.255, "$\odot$", edge=ORANGE, radius=0.019, fontsize=12)
    circle(ax, 0.840, 0.330, "+", edge=PURPLE, radius=0.020, fontsize=12)
    box(ax, 0.925, 0.330, 0.080, 0.060, "$h_t$" + "\n" + r"$B\times64$", edge=PURPLE, fontsize=13)

    arrow(ax, gate_outputs["更新门"], (0.731, 0.405), color=GREEN)
    arrow(ax, (0.205, 0.350), (0.731, 0.420), color=BLUE, connectionstyle="arc3,rad=-0.12")
    label(ax, 0.690, 0.445, "$z_t\odot h_{t-1}$", color=GREEN, size=11)

    box(ax, 0.675, 0.330, 0.060, 0.045, "$1-z_t$", edge=GREEN, fontsize=11)
    arrow(ax, gate_outputs["更新门"], (0.645, 0.350), color=GREEN, connectionstyle="arc3,rad=0.12")
    arrow(ax, (0.675, 0.307), (0.733, 0.270), color=GREEN)
    arrow(ax, (0.685, candidate_y), (0.731, candidate_y), color=ORANGE)
    label(ax, 0.700, 0.215, "$(1-z_t)\odot n_t$", color=ORANGE, size=11)

    arrow(ax, (0.769, 0.405), (0.820, 0.345), color=GREEN)
    arrow(ax, (0.769, 0.255), (0.820, 0.315), color=ORANGE)
    arrow(ax, (0.860, 0.330), (0.885, 0.330), color=PURPLE)
    arrow(ax, (0.965, 0.330), (0.990, 0.330), color=PURPLE)
    label(ax, 0.990, 0.290, "传给下一时间步", color=MUTED, size=10, ha="right")

    # 四个与 PyTorch GRU 实现一致的公式。
    formula_x = 0.035
    formula_y = [0.155, 0.115, 0.075, 0.035]
    formulas = [
        (TEAL, "重置门", r"$r_t=\sigma(W_{ir}x_t+b_{ir}+W_{hr}h_{t-1}+b_{hr})$"),
        (GREEN, "更新门", r"$z_t=\sigma(W_{iz}x_t+b_{iz}+W_{hz}h_{t-1}+b_{hz})$"),
        (ORANGE, "候选状态", r"$n_t=\tanh(W_{in}x_t+b_{in}+r_t\odot(W_{hn}h_{t-1}+b_{hn}))$"),
        (PURPLE, "状态融合", r"$h_t=(1-z_t)\odot n_t+z_t\odot h_{t-1}$"),
    ]
    for y, (color, name, formula) in zip(formula_y, formulas):
        label(ax, formula_x, y, name, color=color, size=11, weight="bold", ha="left")
        label(ax, 0.125, y, formula, color=FG, size=11, ha="left")

    label(
        ax,
        0.965,
        0.155,
        r"$W_{ih}:192\times28$   $W_{hh}:192\times64$",
        color=MUTED,
        size=10.5,
        ha="right",
    )
    label(
        ax,
        0.965,
        0.115,
        "$b_{ih}:192$   $b_{hh}:192$   GRU参数：18,048",
        color=MUTED,
        size=10.5,
        ha="right",
    )
    label(
        ax,
        0.965,
        0.075,
        "投影层参数：4,160   时序编码器合计：22,208",
        color=MUTED,
        size=10.5,
        ha="right",
    )
    label(
        ax,
        0.965,
        0.035,
        "单层时 dropout 为 0；无障碍时每帧输入改为16维，其余门控结构不变",
        color=MUTED,
        size=10.5,
        ha="right",
    )

    fig.savefig(PNG_PATH, dpi=200, facecolor=BG, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(SVG_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    print(PNG_PATH)
    print(SVG_PATH)


if __name__ == "__main__":
    build()
