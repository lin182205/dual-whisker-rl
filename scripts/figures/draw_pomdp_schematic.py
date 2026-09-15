"""绘制POMDP建模、观测设计与动作空间之间的项目逻辑图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = ROOT / "results" / "figures" / "paper" / "fig_2_3_pomdp_schematic"

WIDTH_MM = 159.0
HEIGHT_MM = 105.0

COLORS = {
    "ink": "#263543",
    "muted": "#687783",
    "line": "#A9B2BA",
    "blue": "#376FA8",
    "blue_fill": "#EAF1FA",
    "orange": "#D4772D",
    "orange_fill": "#FCEDDF",
    "green": "#4F8871",
    "green_fill": "#EAF4EE",
    "grey": "#6D7882",
    "grey_fill": "#F1F4F6",
    "soft": "#F7F9FA",
    "white": "#FFFFFF",
}


def select_font() -> str:
    """选择适合论文中文标注的字体。"""
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
    zorder: int = 10,
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
    radius: float = 1.2,
    linewidth: float = 0.9,
    zorder: int = 3,
) -> FancyBboxPatch:
    """绘制圆角矩形。"""
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.16,rounding_size={radius}",
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
    color: str,
    width: float = 1.1,
    linestyle: str = "-",
    connection: str = "arc3,rad=0",
    zorder: int = 6,
) -> None:
    """绘制因果关系箭头。"""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.8,
            linewidth=width,
            linestyle=linestyle,
            color=color,
            connectionstyle=connection,
            shrinkA=0.7,
            shrinkB=0.7,
            zorder=zorder,
        )
    )


def path_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    width: float = 1.0,
    linestyle: str = "-",
    zorder: int = 5,
) -> None:
    """沿折线路径绘制箭头。"""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=7.8,
            linewidth=width,
            linestyle=linestyle,
            color=color,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )
    )


def lane(
    ax: plt.Axes,
    y: float,
    height: float,
    label: str,
    *,
    face: str,
    color: str,
) -> None:
    """绘制POMDP的观测、隐藏状态和决策层。"""
    rounded_rect(ax, 3.0, y, 153.0, height, face=face, edge="none", radius=1.4, linewidth=0.0, zorder=1)
    ax.plot([20.0, 20.0], [y + 2.0, y + height - 2.0], color=color, linewidth=0.75, alpha=0.6, zorder=2)
    add_text(ax, 11.2, y + height / 2, label, size=5.8, color=color, weight="bold")


def circle_node(
    ax: plt.Axes,
    x: float,
    y: float,
    radius: float,
    label: str,
    *,
    face: str,
    edge: str,
    size: float = 7.0,
) -> None:
    """绘制状态、观测或奖励节点。"""
    ax.add_patch(Circle((x, y), radius, facecolor=face, edgecolor=edge, linewidth=1.05, zorder=5))
    add_text(ax, x, y, label, size=size, color=edge, weight="bold")


def decision_node(
    ax: plt.Axes,
    x: float,
    width: float,
    title: str,
    detail: str,
) -> None:
    """绘制历史、策略和动作节点。"""
    rounded_rect(ax, x, 15.0, width, 13.0, face=COLORS["white"], edge=COLORS["orange"], radius=1.0, linewidth=0.85, zorder=4)
    add_text(ax, x + width / 2, 23.4, title, size=5.8, color=COLORS["orange"], weight="bold")
    add_text(ax, x + width / 2, 18.6, detail, size=5.1, color=COLORS["muted"])


def design_panel(
    ax: plt.Axes,
    x: float,
    width: float,
    number: str,
    title: str,
    *,
    face: str,
    color: str,
) -> None:
    """绘制带编号的一级设计模块。"""
    rounded_rect(ax, x, 27.0, width, 75.0, face=face, edge=color, radius=1.6, linewidth=0.9, zorder=1)
    ax.add_patch(Circle((x + 5.0, 95.2), 2.7, facecolor=color, edgecolor="none", zorder=4))
    add_text(ax, x + 5.0, 95.2, number, size=6.1, color=COLORS["white"], weight="bold")
    add_text(ax, x + 9.0, 95.2, title, size=6.8, color=color, weight="bold", ha="left")
    ax.plot(
        [x + 4.0, x + width - 4.0],
        [90.5, 90.5],
        color=color,
        linewidth=0.65,
        alpha=0.55,
        zorder=2,
    )


def two_line_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    edge: str,
    face: str = COLORS["white"],
    title_size: float = 5.8,
    detail_size: float = 5.1,
) -> None:
    """绘制标题与说明两层模块。"""
    rounded_rect(ax, x, y, width, height, face=face, edge=edge, radius=0.9, linewidth=0.8, zorder=4)
    add_text(ax, x + width / 2, y + height * 0.67, title, size=title_size, color=edge, weight="bold")
    add_text(ax, x + width / 2, y + height * 0.31, detail, size=detail_size, color=COLORS["muted"])


def observation_row(
    ax: plt.Axes,
    y: float,
    label: str,
    dims: str,
    detail: str,
) -> None:
    """绘制16维观测中的一个功能分组。"""
    rounded_rect(ax, 59.0, y, 41.0, 7.7, face=COLORS["white"], edge=COLORS["blue"], radius=0.7, linewidth=0.65, zorder=4)
    rounded_rect(ax, 60.0, y + 1.0, 8.2, 5.7, face=COLORS["blue_fill"], edge="none", radius=0.5, linewidth=0.0, zorder=5)
    add_text(ax, 64.1, y + 3.85, dims, size=5.25, color=COLORS["blue"], weight="bold")
    add_text(ax, 70.0, y + 5.25, label, size=5.45, color=COLORS["ink"], weight="bold", ha="left")
    add_text(ax, 70.0, y + 2.4, detail, size=5.0, color=COLORS["muted"], ha="left")


def build_figure() -> plt.Figure:
    """构建双时间步POMDP因果与决策关系图。"""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0.0, WIDTH_MM)
    ax.set_ylim(0.0, HEIGHT_MM)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    lane(ax, 63.0, 22.0, "观测与奖励", face=COLORS["blue_fill"], color=COLORS["blue"])
    lane(ax, 38.0, 22.0, "隐藏状态", face=COLORS["grey_fill"], color=COLORS["grey"])
    lane(ax, 11.0, 23.0, "智能体决策", face=COLORS["orange_fill"], color=COLORS["orange"])

    # 两个相邻时刻的隐藏状态。
    circle_node(ax, 48.0, 49.0, 7.0, "s(t)", face=COLORS["white"], edge=COLORS["grey"], size=7.4)
    circle_node(ax, 121.0, 49.0, 7.0, "s(t+1)", face=COLORS["white"], edge=COLORS["grey"], size=7.0)
    arrow(ax, (55.2, 49.0), (113.8, 49.0), color=COLORS["grey"], width=1.2)
    add_text(ax, 84.5, 43.5, "状态转移  P[s(t+1) | s(t), a(t)]", size=5.5, color=COLORS["grey"], weight="bold")
    add_text(ax, 48.0, 40.8, "时刻 t", size=5.2, color=COLORS["muted"])
    add_text(ax, 121.0, 40.8, "时刻 t+1", size=5.2, color=COLORS["muted"])

    # 隐藏状态生成可观测量；转移同时产生即时奖励。
    circle_node(ax, 48.0, 73.5, 5.8, "o(t)", face=COLORS["white"], edge=COLORS["blue"], size=6.8)
    circle_node(ax, 121.0, 73.5, 5.8, "o(t+1)", face=COLORS["white"], edge=COLORS["blue"], size=6.5)
    circle_node(ax, 84.5, 73.5, 5.8, "r(t)", face=COLORS["green_fill"], edge=COLORS["green"], size=6.8)

    arrow(ax, (48.0, 56.2), (48.0, 67.5), color=COLORS["blue"])
    arrow(ax, (121.0, 56.2), (121.0, 67.5), color=COLORS["blue"])
    arrow(ax, (84.5, 49.8), (84.5, 67.5), color=COLORS["green"], width=1.0)

    add_text(ax, 36.0, 62.1, "O[o(t) | s(t)]", size=5.2, color=COLORS["blue"], weight="bold")
    add_text(ax, 133.8, 62.1, "O[o(t+1) | s(t+1)]", size=5.2, color=COLORS["blue"], weight="bold")
    add_text(ax, 84.5, 82.0, "即时奖励  R[s(t), a(t), s(t+1)]", size=5.25, color=COLORS["green"], weight="bold")

    # 观测历史经PPO策略生成联合动作；动作参与下一状态转移。
    decision_node(ax, 28.0, 27.0, "历史信息 h(t)", "o(0:t), a(0:t-1)")
    decision_node(ax, 62.0, 25.0, "PPO 策略", "基于历史信息决策")
    decision_node(ax, 94.0, 24.0, "联合动作 a(t)", "[移动, kL, kR]")

    path_arrow(
        ax,
        [(43.0, 68.7), (23.0, 61.0), (23.0, 21.5), (27.8, 21.5)],
        color=COLORS["blue"],
        width=1.05,
    )
    add_text(ax, 29.5, 33.5, "观测及历史", size=5.2, color=COLORS["blue"], weight="bold")
    arrow(ax, (55.2, 21.5), (61.8, 21.5), color=COLORS["orange"])
    arrow(ax, (87.2, 21.5), (93.8, 21.5), color=COLORS["orange"])
    path_arrow(
        ax,
        [(118.2, 21.5), (132.0, 21.5), (132.0, 40.5), (126.4, 44.8)],
        color=COLORS["orange"],
        width=1.1,
    )
    add_text(ax, 140.0, 35.2, "动作作用", size=5.2, color=COLORS["orange"], weight="bold")

    # 下一观测进入后续决策历史。
    arrow(ax, (127.0, 73.5), (151.5, 73.5), color=COLORS["blue"], width=0.95, linestyle="--")
    add_text(ax, 141.0, 79.8, "更新 h(t+1)", size=5.2, color=COLORS["blue"], weight="bold")

    # 项目语义映射，避免把不可观测真实状态误当成策略输入。
    rounded_rect(ax, 20.0, 2.0, 136.0, 6.5, face="#F7F9FA", edge="none", radius=0.7, linewidth=0.0, zorder=2)
    add_text(
        ax,
        88.0,
        5.3,
        "项目映射：s(t)=羽流/气源/机器人/传感器真实状态；o(t)=16维硬件可得观测；a(t)=[移动, kL, kR]",
        size=5.2,
        color=COLORS["muted"],
        weight="bold",
    )

    return fig


def build_logic_figure() -> plt.Figure:
    """构建POMDP建模、观测设计、动作设计及其闭环关系。"""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0.0, WIDTH_MM)
    ax.set_ylim(0.0, HEIGHT_MM)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    design_panel(
        ax, 3.0, 46.0, "1", "部分可观测决策建模",
        face=COLORS["grey_fill"], color=COLORS["grey"],
    )
    design_panel(
        ax, 55.0, 49.0, "2", "观测状态设计",
        face=COLORS["blue_fill"], color=COLORS["blue"],
    )
    design_panel(
        ax, 110.0, 46.0, "3", "动作空间设计",
        face=COLORS["orange_fill"], color=COLORS["orange"],
    )

    # 1. POMDP定义真实状态、可见信息、转移与奖励的边界。
    add_text(
        ax, 26.0, 87.5,
        "POMDP = (S, A, T, R, Ω, O, gamma)",
        size=5.4, color=COLORS["grey"], weight="bold",
    )
    two_line_box(
        ax, 8.0, 72.0, 36.0, 14.0,
        "隐藏真实状态 s(t)",
        "动态羽流/气源 · 机器人位姿 · 传感器动态",
        edge=COLORS["grey"], detail_size=5.1,
    )
    rounded_rect(
        ax, 13.0, 63.4, 26.0, 5.8,
        face=COLORS["white"], edge=COLORS["blue"],
        radius=0.7, linewidth=0.75, zorder=4,
    )
    add_text(
        ax, 26.0, 66.3, "观测模型  O[o(t) | s(t)]",
        size=5.15, color=COLORS["blue"], weight="bold",
    )
    arrow(ax, (26.0, 71.8), (26.0, 69.4), color=COLORS["blue"], width=0.9)
    two_line_box(
        ax, 8.0, 49.0, 36.0, 11.0,
        "可观测信息 o(t)",
        "策略可访问；真实状态不可直接访问",
        edge=COLORS["blue"], detail_size=5.1,
    )
    arrow(ax, (26.0, 63.2), (26.0, 60.2), color=COLORS["blue"], width=0.9)
    two_line_box(
        ax, 8.0, 31.0, 36.0, 13.0,
        "状态转移与奖励",
        "P[s(t+1)|s(t),a(t)]；r(t)=R(s(t),a(t),s(t+1))",
        edge=COLORS["green"], face=COLORS["green_fill"], detail_size=5.1,
    )
    path_arrow(
        ax,
        [(8.0, 37.5), (5.4, 37.5), (5.4, 79.0), (7.8, 79.0)],
        color=COLORS["grey"], width=0.85, linestyle="--", zorder=3,
    )
    # 2. 16维硬件可得观测按功能分组，再堆叠20帧交给PPO。
    rounded_rect(
        ax, 61.5, 82.5, 36.0, 6.2,
        face=COLORS["white"], edge=COLORS["blue"],
        radius=0.8, linewidth=0.8, zorder=4,
    )
    add_text(
        ax, 79.5, 85.6, "单帧观测 o(t)：16维硬件可得向量",
        size=5.4, color=COLORS["blue"], weight="bold",
    )
    observation_row(ax, 73.0, "气味响应", "6维", "左右信号、平滑值、变化趋势")
    observation_row(ax, 63.9, "左右差分", "2维", "平滑差与趋势差")
    observation_row(ax, 54.8, "触须位姿", "4维", "左右角度与扇区位置")
    observation_row(ax, 45.7, "本体与搜索", "4维", "航向、上步移动、空白时长")
    two_line_box(
        ax, 59.0, 30.0, 41.0, 11.5,
        "20帧历史 h(t) → PPO策略 πθ(a|h)",
        "[o(t−19), …, o(t)]：16 × 20 = 320维",
        edge=COLORS["blue"], title_size=5.35, detail_size=5.1,
    )
    arrow(ax, (79.5, 45.5), (79.5, 41.7), color=COLORS["blue"], width=0.95)

    # 3. PPO联合选择底盘动作和两个触须扇区。
    rounded_rect(
        ax, 115.0, 81.0, 36.0, 7.5,
        face=COLORS["white"], edge=COLORS["orange"],
        radius=0.8, linewidth=0.8, zorder=4,
    )
    add_text(ax, 133.0, 85.8, "联合动作空间", size=5.6, color=COLORS["orange"], weight="bold")
    add_text(ax, 133.0, 82.7, "MultiDiscrete([6, 10, 10])", size=5.15, color=COLORS["muted"], weight="bold")
    two_line_box(
        ax, 115.0, 63.0, 36.0, 13.0,
        "底盘动作 a_m(t)：6类",
        "前进/左右转 · 原地左右旋转 · 停止",
        edge=COLORS["orange"], detail_size=5.1,
    )
    two_line_box(
        ax, 115.0, 47.5, 17.0, 11.0,
        "左触须 kL", "扇区 0–9", edge=COLORS["orange"],
    )
    two_line_box(
        ax, 134.0, 47.5, 17.0, 11.0,
        "右触须 kR", "扇区 0–9", edge=COLORS["orange"],
    )
    two_line_box(
        ax, 115.0, 31.0, 36.0, 11.0,
        "a(t) = [a_m(t), kL, kR]",
        "小车移动与双触须摆动同步决策",
        edge=COLORS["orange"], title_size=5.5, detail_size=5.1,
    )
    path_arrow(
        ax, [(133.0, 80.8), (133.0, 78.5), (133.0, 76.2)],
        color=COLORS["orange"], width=0.9,
    )
    arrow(ax, (133.0, 62.8), (133.0, 42.2), color=COLORS["orange"], width=0.8)
    arrow(ax, (123.5, 47.3), (127.5, 42.2), color=COLORS["orange"], width=0.8)
    arrow(ax, (142.5, 47.3), (138.5, 42.2), color=COLORS["orange"], width=0.8)

    # 跨模块数据流：观测用于决策，奖励用于训练，PPO输出联合动作。
    path_arrow(
        ax,
        [(44.2, 54.5), (51.5, 54.5), (51.5, 85.6), (61.3, 85.6)],
        color=COLORS["blue"], width=1.15,
    )
    arrow(ax, (44.2, 36.3), (58.8, 36.3), color=COLORS["green"], width=1.0)
    arrow(ax, (100.2, 35.8), (114.8, 35.8), color=COLORS["orange"], width=1.15)

    # 底部闭环：动作改变采样条件，环境推进并产生下一观测和奖励。
    rounded_rect(
        ax, 3.0, 3.0, 153.0, 20.0,
        face=COLORS["soft"], edge=COLORS["line"],
        radius=1.4, linewidth=0.7, zorder=1,
    )
    two_line_box(
        ax, 112.0, 7.0, 39.0, 11.0,
        "执行联合动作", "底盘运动 + 双触须摆动",
        edge=COLORS["orange"], face=COLORS["orange_fill"],
    )
    two_line_box(
        ax, 61.0, 7.0, 40.0, 11.0,
        "改变采样条件", "机器人位姿 + 左右传感器采样点",
        edge=COLORS["grey"], detail_size=5.1,
    )
    two_line_box(
        ax, 12.0, 7.0, 38.0, 11.0,
        "环境推进与反馈", "生成 s(t+1)、o(t+1) 与 r(t)",
        edge=COLORS["blue"], face=COLORS["blue_fill"], detail_size=5.1,
    )
    path_arrow(
        ax, [(133.0, 30.8), (133.0, 23.0), (131.5, 18.2)],
        color=COLORS["orange"], width=1.1,
    )
    arrow(ax, (111.8, 12.5), (101.2, 12.5), color=COLORS["orange"], width=1.05)
    arrow(ax, (60.8, 12.5), (50.2, 12.5), color=COLORS["grey"], width=1.05)
    path_arrow(
        ax, [(31.0, 18.2), (31.0, 24.5), (26.0, 30.8)],
        color=COLORS["blue"], width=1.05,
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    """导出 SVG、PDF、600 dpi TIFF 与 300 dpi PNG。"""
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "POMDP建模、观测状态与动作空间逻辑图"}
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), metadata=metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), metadata=metadata)
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(
        OUTPUT_STEM.with_suffix(".tiff"),
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )


def main() -> None:
    fig = build_logic_figure()
    save_figure(fig)
    plt.close(fig)
    print(f"saved={OUTPUT_STEM}")
    print(f"font={CJK_FONT}")
    print(f"size_mm={WIDTH_MM}x{HEIGHT_MM}")


if __name__ == "__main__":
    main()
