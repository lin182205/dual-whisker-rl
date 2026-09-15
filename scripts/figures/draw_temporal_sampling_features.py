"""绘制历史窗口、时序采样与GRU特征提取过程图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = ROOT / "results" / "figures" / "paper" / "fig_2_4_temporal_sampling_features"

WIDTH_MM = 159.0
HEIGHT_MM = 96.0

COLORS = {
    "ink": "#263543",
    "muted": "#687783",
    "line": "#A8B2BA",
    "grey": "#6D7882",
    "grey_fill": "#F1F4F6",
    "blue": "#376FA8",
    "blue_fill": "#EAF1FA",
    "teal": "#4F8871",
    "teal_fill": "#EAF4EE",
    "orange": "#D4772D",
    "orange_fill": "#FCEDDF",
    "white": "#FFFFFF",
    "soft": "#F7F9FA",
}


def select_font() -> str:
    """选择支持中文的论文绘图字体。"""
    available = {entry.name for entry in font_manager.fontManager.ttflist}
    for candidate in (
        "Noto Sans CJK SC",
        "Source Han Sans CN",
        "Microsoft YaHei",
        "SimHei",
    ):
        if candidate in available:
            return candidate
    raise RuntimeError("未找到支持中文的字体。")


CJK_FONT = select_font()
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [CJK_FONT, "Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 6.2,
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
    size: float = 6.2,
    color: str = COLORS["ink"],
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    zorder: int = 10,
) -> None:
    """添加可编辑文字。"""
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
    radius: float = 1.0,
    linewidth: float = 0.8,
    zorder: int = 3,
) -> FancyBboxPatch:
    """绘制圆角模块。"""
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
    width: float = 1.0,
    linestyle: str = "-",
    connection: str = "arc3,rad=0",
    zorder: int = 6,
) -> None:
    """绘制数据或控制箭头。"""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.5,
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
    zorder: int = 6,
) -> None:
    """沿折线绘制闭环箭头。"""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=7.5,
            linewidth=width,
            linestyle=linestyle,
            color=color,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )
    )


def stage_panel(
    ax: plt.Axes,
    x: float,
    width: float,
    number: str,
    title: str,
    *,
    color: str,
    face: str,
) -> None:
    """绘制一级流程阶段。"""
    rounded_rect(ax, x, 25.0, width, 68.0, face=face, edge=color, radius=1.5, linewidth=0.9, zorder=1)
    ax.add_patch(Circle((x + 4.4, 87.0), 2.5, facecolor=color, edgecolor="none", zorder=4))
    add_text(ax, x + 4.4, 87.0, number, size=5.9, color=COLORS["white"], weight="bold")
    add_text(ax, x + 8.0, 87.0, title, size=6.35, color=color, weight="bold", ha="left")
    ax.plot([x + 3.5, x + width - 3.5], [82.6, 82.6], color=color, linewidth=0.65, alpha=0.55, zorder=2)


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
    title_size: float = 5.55,
    detail_size: float = 5.1,
) -> None:
    """绘制标题与细节两层模块。"""
    rounded_rect(ax, x, y, width, height, face=face, edge=edge, radius=0.8, linewidth=0.75, zorder=4)
    add_text(ax, x + width / 2, y + height * 0.67, title, size=title_size, color=edge, weight="bold")
    add_text(ax, x + width / 2, y + height * 0.30, detail, size=detail_size, color=COLORS["muted"])


def feature_row(
    ax: plt.Axes,
    y: float,
    dims: str,
    label: str,
    detail: str,
) -> None:
    """绘制16维单帧观测中的一个功能分组。"""
    rounded_rect(ax, 43.0, y, 42.0, 7.0, face=COLORS["white"], edge=COLORS["teal"], radius=0.6, linewidth=0.6, zorder=4)
    rounded_rect(ax, 44.0, y + 0.8, 7.0, 5.4, face=COLORS["teal_fill"], edge="none", radius=0.45, linewidth=0.0, zorder=5)
    add_text(ax, 47.5, y + 3.5, dims, size=5.15, color=COLORS["teal"], weight="bold")
    add_text(ax, 52.5, y + 4.8, label, size=5.25, color=COLORS["ink"], weight="bold", ha="left")
    add_text(ax, 52.5, y + 2.1, detail, size=5.1, color=COLORS["muted"], ha="left")


def build_figure() -> plt.Figure:
    """构建时序采样、历史窗口和GRU特征提取闭环图。"""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0.0, WIDTH_MM)
    ax.set_ylim(0.0, HEIGHT_MM)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    stage_panel(ax, 3.0, 34.0, "1", "时序数据采样", color=COLORS["grey"], face=COLORS["grey_fill"])
    stage_panel(ax, 40.0, 48.0, "2", "单帧特征构造", color=COLORS["teal"], face=COLORS["teal_fill"])
    stage_panel(ax, 91.0, 31.0, "3", "20帧历史窗口", color=COLORS["blue"], face=COLORS["blue_fill"])
    stage_panel(ax, 125.0, 31.0, "4", "GRU特征提取", color=COLORS["orange"], face=COLORS["orange_fill"])

    # 1. 上一动作改变机器人、羽流与触须状态，随后在时刻t采样。
    two_line_box(ax, 7.0, 70.5, 26.0, 8.5, "上一联合动作 a(t-1)", "移动 + 左触须 + 右触须", edge=COLORS["grey"])
    two_line_box(ax, 7.0, 58.0, 26.0, 8.5, "物理状态推进", "底盘运动 · 羽流推进 · 触须转动", edge=COLORS["grey"])
    two_line_box(
        ax, 7.0, 42.5, 26.0, 12.0,
        "时刻 t 原始采样",
        "左右气味 · 触须角度/扇区\n航向 · 上步移动 · blank_age",
        edge=COLORS["blue"], face=COLORS["blue_fill"], detail_size=5.0,
    )
    arrow(ax, (20.0, 70.3), (20.0, 66.7), color=COLORS["grey"], width=0.9)
    arrow(ax, (20.0, 57.8), (20.0, 54.2), color=COLORS["blue"], width=0.95)
    rounded_rect(ax, 7.0, 29.0, 26.0, 10.0, face=COLORS["white"], edge=COLORS["grey"], radius=0.7, linewidth=0.7, zorder=4)
    add_text(ax, 20.0, 36.2, "等间隔采样", size=5.4, color=COLORS["grey"], weight="bold")
    add_text(ax, 20.0, 32.5, "dt = 0.2 s；20帧约4 s", size=5.1, color=COLORS["muted"])

    # 2. 左右气味流先做在线预处理，再与其余可测状态组成16维观测。
    add_text(ax, 64.0, 79.2, "左右气味流的在线预处理", size=5.45, color=COLORS["teal"], weight="bold")
    preprocess_boxes = [
        (42.5, 70.0, 9.2, "基线\n扣除"),
        (53.0, 70.0, 9.2, "EMA平滑\ntau=2 s"),
        (63.5, 70.0, 9.2, "8帧\n趋势"),
        (74.0, 70.0, 11.5, "左右差分\n归一化/裁剪"),
    ]
    for x, y, width, label in preprocess_boxes:
        rounded_rect(ax, x, y, width, 7.0, face=COLORS["white"], edge=COLORS["teal"], radius=0.55, linewidth=0.65, zorder=4)
        add_text(ax, x + width / 2, y + 3.5, label, size=5.1, color=COLORS["teal"], weight="bold")
    arrow(ax, (51.8, 73.5), (52.8, 73.5), color=COLORS["teal"], width=0.75)
    arrow(ax, (62.3, 73.5), (63.3, 73.5), color=COLORS["teal"], width=0.75)
    arrow(ax, (72.8, 73.5), (73.8, 73.5), color=COLORS["teal"], width=0.75)
    rounded_rect(ax, 46.0, 63.0, 36.0, 4.8, face=COLORS["white"], edge=COLORS["teal"], radius=0.55, linewidth=0.7, zorder=4)
    add_text(ax, 64.0, 65.4, "单帧硬件可得观测 o(t)：16维", size=5.4, color=COLORS["teal"], weight="bold")
    arrow(ax, (79.75, 69.8), (79.75, 68.0), color=COLORS["teal"], width=0.8)
    feature_row(ax, 53.5, "6维", "气味响应", "左右信号、平滑值、趋势")
    feature_row(ax, 45.5, "2维", "左右差分", "平滑差、趋势差")
    feature_row(ax, 37.5, "4维", "触须位姿", "左右角度、扇区")
    feature_row(ax, 29.5, "4维", "本体与搜索", "航向cos/sin、上步移动、blank_age")

    # 3. 历史窗口保持最旧到最新的顺序，并按步滑动更新。
    add_text(ax, 106.5, 79.3, "时间顺序：最旧 → 最新", size=5.3, color=COLORS["blue"], weight="bold")
    labels = ("t-19", "...", "t-2", "t-1", "t")
    cell_x = (94.0, 99.3, 104.6, 109.9, 115.2)
    fills = ("#F4F7FA", "#F4F7FA", "#EDF3F9", "#E5EEF7", "#DCEAF6")
    for x, label, fill in zip(cell_x, labels, fills, strict=True):
        rounded_rect(ax, x, 68.5, 4.7, 8.0, face=fill, edge=COLORS["blue"], radius=0.4, linewidth=0.6, zorder=4)
        add_text(ax, x + 2.35, 72.5, label, size=5.1, color=COLORS["blue"], weight="bold")
    two_line_box(ax, 95.0, 53.0, 23.0, 11.0, "H(t) = [o(t-19), ..., o(t)]", "20 x 16 = 320维扁平向量", edge=COLORS["blue"], title_size=5.15)
    two_line_box(ax, 95.0, 39.5, 23.0, 9.0, "滑动更新", "移除最旧帧 · 追加最新帧", edge=COLORS["blue"], detail_size=5.0)
    two_line_box(ax, 95.0, 28.5, 23.0, 7.5, "episode初始化", "首帧 o(0) 复制20次", edge=COLORS["grey"], title_size=5.2, detail_size=5.0)
    arrow(ax, (106.5, 68.3), (106.5, 64.2), color=COLORS["blue"], width=0.9)
    arrow(ax, (106.5, 52.8), (106.5, 48.7), color=COLORS["blue"], width=0.9)

    # 4. SB3扁平接口在提取器内部恢复时间维，再用默认单层GRU编码。
    two_line_box(ax, 129.0, 72.0, 23.0, 8.0, "SB3输入：320维", "扁平Box观测", edge=COLORS["orange"])
    two_line_box(ax, 129.0, 61.0, 23.0, 7.0, "恢复时间维", "reshape → [20, 16]", edge=COLORS["orange"])
    two_line_box(
        ax, 129.0, 46.5, 23.0, 10.5,
        "单层GRU递推",
        "input=16 · hidden=64\n按最旧→最新顺序递推",
        edge=COLORS["orange"], face=COLORS["white"], detail_size=5.0,
    )
    two_line_box(ax, 129.0, 34.5, 23.0, 8.0, "最后隐藏状态 h(t)", "取最后时刻、最后一层", edge=COLORS["orange"], detail_size=5.0)
    two_line_box(
        ax, 129.0, 26.0, 23.0, 6.5,
        "Linear + GELU", "z(t)：64维时序特征",
        edge=COLORS["orange"], title_size=5.1, detail_size=5.0,
    )
    arrow(ax, (140.5, 71.8), (140.5, 68.2), color=COLORS["orange"], width=0.9)
    arrow(ax, (140.5, 60.8), (140.5, 57.2), color=COLORS["orange"], width=0.9)
    arrow(ax, (140.5, 46.3), (140.5, 42.7), color=COLORS["orange"], width=0.9)
    arrow(ax, (140.5, 34.3), (140.5, 32.7), color=COLORS["orange"], width=0.9)

    # 主数据链在各阶段之间直接连接。
    path_arrow(ax, [(33.2, 48.7), (38.5, 48.7), (38.5, 73.5), (42.3, 73.5)], color=COLORS["blue"], width=1.1)
    arrow(ax, (82.2, 65.4), (94.8, 58.5), color=COLORS["teal"], width=1.05, connection="arc3,rad=0.08")
    arrow(ax, (118.2, 58.5), (128.8, 76.0), color=COLORS["blue"], width=1.05, connection="arc3,rad=-0.08")

    # 底部闭环：时序特征进入PPO，联合动作改变下一时刻采样条件。
    rounded_rect(ax, 3.0, 3.0, 153.0, 18.0, face=COLORS["soft"], edge=COLORS["line"], radius=1.2, linewidth=0.7, zorder=1)
    two_line_box(ax, 126.0, 6.5, 27.0, 10.5, "PPO策略/价值网络", "z(t) → 联合动作 a(t)", edge=COLORS["orange"], face=COLORS["orange_fill"])
    two_line_box(ax, 70.0, 6.5, 43.0, 10.5, "改变下一时刻采样条件", "底盘位姿 + 左右触须采样点", edge=COLORS["grey"])
    two_line_box(ax, 13.0, 6.5, 43.0, 10.5, "生成下一帧观测", "执行 a(t) 后获得 o(t+1)", edge=COLORS["blue"], face=COLORS["blue_fill"])
    path_arrow(ax, [(140.5, 26.3), (140.5, 21.5), (139.5, 17.2)], color=COLORS["orange"], width=1.05)
    arrow(ax, (125.8, 11.75), (113.2, 11.75), color=COLORS["orange"], width=1.05)
    arrow(ax, (69.8, 11.75), (56.2, 11.75), color=COLORS["grey"], width=1.05)
    path_arrow(ax, [(34.5, 17.2), (34.5, 22.5), (20.0, 29.0)], color=COLORS["blue"], width=1.05)

    return fig


def save_figure(fig: plt.Figure) -> None:
    """导出SVG、PDF、600 dpi TIFF及PNG预览。"""
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "历史窗口、时序采样与GRU特征提取过程"}
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
