"""绘制双触须移动嗅觉系统的总体闭环架构图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import (
    Circle,
    Ellipse,
    FancyArrowPatch,
    FancyBboxPatch,
    PathPatch,
    Rectangle,
)
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = ROOT / "results" / "figures" / "paper" / "fig_2_1_system_architecture"

WIDTH_MM = 159.0
HEIGHT_MM = 86.0

COLORS = {
    "ink": "#263543",
    "muted": "#677684",
    "blue": "#376FA8",
    "blue_fill": "#EAF1FA",
    "orange": "#D4772D",
    "orange_fill": "#FCEDDF",
    "green": "#4F8871",
    "green_fill": "#EAF4EE",
    "grey_fill": "#F4F6F7",
    "light_line": "#A9B4BE",
    "plume": "#ABCBC0",
    "white": "#FFFFFF",
}


def select_font() -> str:
    """按既定优先级选择可用中文字体。"""
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in (
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
    ):
        if candidate in available:
            return candidate
    raise RuntimeError(
        "未找到可用中文字体；请安装 Noto Sans CJK SC、思源黑体或 Microsoft YaHei。"
    )


CJK_FONT = select_font()
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [CJK_FONT, "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
        "savefig.facecolor": COLORS["white"],
    }
)


def add_text(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 6.4,
    color: str = COLORS["ink"],
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    zorder: int = 8,
) -> None:
    """添加保持可编辑的文字。"""
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=ha,
        va=va,
        zorder=zorder,
    )


def rounded_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str,
    edge: str,
    radius: float = 1.5,
    linewidth: float = 0.9,
    zorder: int = 2,
) -> FancyBboxPatch:
    """绘制圆角矩形。"""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.18,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["blue"],
    width: float = 1.2,
    zorder: int = 5,
) -> None:
    """绘制直线箭头。"""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=width,
            color=color,
            shrinkA=0.7,
            shrinkB=0.7,
            zorder=zorder,
        )
    )


def path_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str = COLORS["blue"],
    width: float = 1.2,
    zorder: int = 4,
) -> None:
    """沿折线路径绘制箭头。"""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=8.0,
            linewidth=width,
            color=color,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )
    )


def data_callout(
    ax: plt.Axes,
    center_x: float,
    width: float,
    title: str,
    detail_1: str,
    detail_2: str,
    *,
    face: str,
    edge: str,
) -> None:
    """在数据箭头上方集中说明传输变量及其组成。"""
    x = center_x - width / 2
    rounded_rect(
        ax,
        x,
        68.5,
        width,
        14.5,
        face=face,
        edge=edge,
        radius=1.0,
        linewidth=0.8,
        zorder=6,
    )
    add_text(ax, center_x, 79.5, title, size=5.7, color=edge, weight="bold")
    add_text(ax, center_x, 75.3, detail_1, size=5.2, color=COLORS["ink"])
    add_text(ax, center_x, 71.6, detail_2, size=5.2, color=COLORS["muted"])
    ax.plot(
        [center_x, center_x],
        [68.3, 54.4],
        color=edge,
        linewidth=0.7,
        zorder=4,
    )


def draw_robot(ax: plt.Axes) -> None:
    """绘制带双触须的差速移动机器人。"""
    rounded_rect(ax, 3.0, 39.5, 24.0, 25.5, face=COLORS["grey_fill"], edge=COLORS["light_line"], radius=1.2)
    rounded_rect(ax, 8.0, 45.5, 14.0, 10.0, face=COLORS["white"], edge=COLORS["green"], radius=1.6, linewidth=1.0, zorder=5)
    for x in (6.8, 21.7):
        ax.add_patch(Rectangle((x, 46.2), 2.2, 3.0, facecolor="#4F5B65", edgecolor=COLORS["ink"], linewidth=0.5, zorder=5))
        ax.add_patch(Rectangle((x, 51.7), 2.2, 3.0, facecolor="#4F5B65", edgecolor=COLORS["ink"], linewidth=0.5, zorder=5))
    ax.plot([11.2, 7.8], [55.0, 60.3], color=COLORS["green"], linewidth=1.5, solid_capstyle="round", zorder=5)
    ax.plot([18.8, 22.2], [55.0, 60.3], color=COLORS["green"], linewidth=1.5, solid_capstyle="round", zorder=5)
    for x in (7.3, 22.7):
        ax.add_patch(Circle((x, 61.0), 1.45, facecolor="#F1C77C", edgecolor="#98631E", linewidth=0.7, zorder=6))
    arrow(ax, (15.0, 47.2), (15.0, 53.5), color=COLORS["green"], width=0.85, zorder=6)
    add_text(ax, 15.0, 35.3, "移动机器人", size=7.0, weight="bold")


def draw_sensor_readings(ax: plt.Axes) -> None:
    """绘制项目所需的高层传感器读数。"""
    rounded_rect(ax, 37.0, 39.5, 29.0, 25.5, face=COLORS["blue_fill"], edge=COLORS["blue"], radius=1.6, linewidth=1.0)

    # 双触须气味信号。
    ax.plot([42.3, 45.4], [57.6, 60.2], color=COLORS["green"], linewidth=1.2, zorder=5)
    ax.plot([42.3, 45.4], [57.6, 55.0], color=COLORS["green"], linewidth=1.2, zorder=5)
    ax.add_patch(Circle((46.2, 60.8), 1.25, facecolor="#F1C77C", edgecolor="#98631E", linewidth=0.6, zorder=6))
    ax.add_patch(Circle((46.2, 54.4), 1.25, facecolor="#F1C77C", edgecolor="#98631E", linewidth=0.6, zorder=6))
    add_text(ax, 56.0, 57.6, "双触须气味信号", size=5.8, weight="bold")

    # 机器人状态，仅作系统级逻辑表达。
    ax.add_patch(Circle((45.6, 46.7), 3.0, facecolor=COLORS["white"], edgecolor=COLORS["blue"], linewidth=0.8, zorder=5))
    arrow(ax, (45.6, 46.7), (45.6, 49.0), color=COLORS["blue"], width=0.7, zorder=6)
    add_text(ax, 56.0, 46.7, "机器人状态", size=5.8, weight="bold")
    add_text(ax, 51.5, 35.3, "传感器读数", size=7.0, weight="bold")


def draw_ppo_network(ax: plt.Axes) -> None:
    """以简洁神经网络图标表示 PPO 策略模块。"""
    rounded_rect(ax, 77.0, 39.5, 31.0, 25.5, face=COLORS["orange_fill"], edge=COLORS["orange"], radius=1.6, linewidth=1.0)
    input_nodes = [(83.5, y) for y in (47.0, 52.2, 57.4)]
    hidden_nodes = [(93.5, y) for y in (45.5, 49.9, 54.3, 58.7)]
    output_nodes = [(103.5, y) for y in (48.6, 55.6)]
    for p1 in input_nodes:
        for p2 in hidden_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#D8B08F", linewidth=0.45, zorder=3)
    for p1 in hidden_nodes:
        for p2 in output_nodes:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#D8B08F", linewidth=0.45, zorder=3)
    for x, y in input_nodes:
        ax.add_patch(Circle((x, y), 1.15, facecolor=COLORS["white"], edgecolor=COLORS["orange"], linewidth=0.7, zorder=5))
    for x, y in hidden_nodes:
        ax.add_patch(Circle((x, y), 1.15, facecolor="#F6C99F", edgecolor=COLORS["orange"], linewidth=0.7, zorder=5))
    for x, y in output_nodes:
        ax.add_patch(Circle((x, y), 1.15, facecolor=COLORS["white"], edgecolor=COLORS["orange"], linewidth=0.7, zorder=5))
    add_text(ax, 92.5, 61.8, "PPO", size=7.0, weight="bold", color=COLORS["orange"])
    add_text(ax, 92.5, 35.3, "PPO 策略网络", size=7.0, weight="bold")


def draw_joint_command(ax: plt.Axes) -> None:
    """绘制小车移动与触须摆动的联合动作输出。"""
    rounded_rect(ax, 119.0, 39.5, 36.0, 25.5, face=COLORS["grey_fill"], edge=COLORS["light_line"], radius=1.2)

    # 小车移动。
    rounded_rect(ax, 123.0, 55.0, 8.4, 5.0, face=COLORS["white"], edge=COLORS["orange"], radius=1.0, linewidth=0.8, zorder=5)
    ax.add_patch(Circle((125.0, 54.7), 0.9, facecolor="#4F5B65", edgecolor="none", zorder=6))
    ax.add_patch(Circle((129.4, 54.7), 0.9, facecolor="#4F5B65", edgecolor="none", zorder=6))
    arrow(ax, (132.1, 57.5), (136.4, 57.5), color=COLORS["orange"], width=0.85, zorder=6)
    add_text(ax, 145.2, 57.5, "小车移动", size=5.8, weight="bold")

    # 双触须摆动。
    ax.plot([127.2, 124.0], [46.5, 50.4], color=COLORS["green"], linewidth=1.25, zorder=5)
    ax.plot([127.2, 130.4], [46.5, 50.4], color=COLORS["green"], linewidth=1.25, zorder=5)
    ax.add_patch(Circle((123.6, 50.9), 1.0, facecolor="#F1C77C", edgecolor="#98631E", linewidth=0.5, zorder=6))
    ax.add_patch(Circle((130.8, 50.9), 1.0, facecolor="#F1C77C", edgecolor="#98631E", linewidth=0.5, zorder=6))
    add_text(ax, 145.2, 48.5, "触须摆动", size=5.8, weight="bold")
    add_text(ax, 137.0, 35.3, "联合控制指令", size=7.0, weight="bold")


def draw_search_environment(ax: plt.Axes) -> None:
    """绘制闭环底部的仿真搜索环境。"""
    rounded_rect(ax, 49.0, 4.0, 62.0, 22.5, face=COLORS["white"], edge=COLORS["blue"], radius=0.8, linewidth=0.9, zorder=2)
    ax.add_patch(Circle((55.0, 15.2), 1.4, facecolor="#8B5E3C", edgecolor=COLORS["ink"], linewidth=0.55, zorder=5))
    for y0, alpha in ((15.0, 0.55), (18.3, 0.38), (11.9, 0.3)):
        vertices = [(56.4, y0), (66.0, y0 + 2.0), (78.0, y0 - 2.2), (97.0, y0 + 0.5)]
        ax.add_patch(
            PathPatch(
                MplPath(vertices, [MplPath.MOVETO] + [MplPath.CURVE4] * 3),
                facecolor="none",
                edgecolor=COLORS["plume"],
                linewidth=2.5,
                alpha=alpha,
                capstyle="round",
                zorder=3,
            )
        )
    for cx, cy, w, h in ((65.0, 16.7, 3.4, 1.5), (75.0, 13.5, 4.0, 1.8), (87.0, 17.2, 3.5, 1.6)):
        ax.add_patch(Ellipse((cx, cy), w, h, facecolor=COLORS["plume"], edgecolor="none", alpha=0.48, zorder=4))
    rounded_rect(ax, 98.0, 11.8, 7.5, 5.5, face=COLORS["green_fill"], edge=COLORS["green"], radius=1.0, linewidth=0.7, zorder=5)
    ax.add_patch(Circle((99.5, 11.6), 0.7, facecolor="#4F5B65", edgecolor="none", zorder=6))
    ax.add_patch(Circle((104.0, 11.6), 0.7, facecolor="#4F5B65", edgecolor="none", zorder=6))
    arrow(ax, (101.7, 14.6), (97.2, 14.6), color=COLORS["green"], width=0.7, zorder=6)
    add_text(ax, 80.0, 1.2, "仿真搜索环境", size=6.8, weight="bold")


def build_figure() -> plt.Figure:
    """构建参考论文式的单一闭环总图。"""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0.0, WIDTH_MM)
    ax.set_ylim(0.0, HEIGHT_MM)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    draw_robot(ax)
    draw_sensor_readings(ax)
    draw_ppo_network(ax)
    draw_joint_command(ax)
    draw_search_environment(ax)

    # 顶部感知—决策—控制链。
    arrow(ax, (27.2, 52.3), (36.8, 52.3), color=COLORS["blue"])
    arrow(ax, (66.2, 52.3), (76.8, 52.3), color=COLORS["blue"])
    arrow(ax, (108.2, 52.3), (118.8, 52.3), color=COLORS["orange"])

    data_callout(
        ax,
        32.0,
        38.0,
        "原始采样数据 s(t)",
        "左右气味读数 gL(t), gR(t)",
        "触须角度/扇区 + 机器人航向",
        face=COLORS["blue_fill"],
        edge=COLORS["blue"],
    )
    data_callout(
        ax,
        71.5,
        38.0,
        "策略输入 o(t)（16维）",
        "气味与左右差分 8维 + 触须状态 4维",
        "航向/上一移动/空白时长 4维",
        face=COLORS["blue_fill"],
        edge=COLORS["blue"],
    )
    data_callout(
        ax,
        113.5,
        44.0,
        "联合动作 a(t)",
        "a(t) = [移动分量, kL, kR]",
        "移动6类；左右触须各10扇区",
        face=COLORS["orange_fill"],
        edge=COLORS["orange"],
    )

    # 联合动作作用于环境，环境变化形成下一时刻输入。
    path_arrow(ax, [(155.2, 52.3), (157.0, 52.3), (157.0, 15.3), (111.2, 15.3)], color=COLORS["orange"])
    add_text(ax, 134.0, 20.8, "移动分量 → 底盘移动", size=5.2, color=COLORS["orange"], weight="bold")
    add_text(ax, 134.0, 17.8, "kL, kR → 左右触须摆动", size=5.2, color=COLORS["orange"], weight="bold")
    path_arrow(ax, [(48.8, 15.3), (2.0, 15.3), (2.0, 52.3), (3.8, 52.3)], color=COLORS["blue"])
    add_text(ax, 25.5, 20.8, "环境反馈：机器人位姿 q(t+1)", size=5.2, color=COLORS["blue"], weight="bold")
    add_text(ax, 25.5, 17.8, "新采样位置 → 下一帧左右气味读数", size=5.2, color=COLORS["blue"], weight="bold")

    return fig


def save_figure(fig: plt.Figure) -> None:
    """导出 SVG、PDF、600 dpi TIFF 与 300 dpi PNG。"""
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "双触须移动嗅觉系统总体架构"}
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), metadata=metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), metadata=metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    fig = build_figure()
    save_figure(fig)
    plt.close(fig)
    print(f"saved={OUTPUT_STEM}")
    print(f"font={CJK_FONT}")
    print(f"size_mm={WIDTH_MM}x{HEIGHT_MM}")


if __name__ == "__main__":
    main()
