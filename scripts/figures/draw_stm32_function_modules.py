"""绘制 STM32 双触须下位机功能模块与数据流示意图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.path import Path as MplPath


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_STEM = ROOT / "results" / "figures" / "paper" / "fig_2_2_stm32_function_modules"

WIDTH_MM = 159.0
HEIGHT_MM = 93.0

COLORS = {
    "ink": "#263543",
    "muted": "#677684",
    "blue": "#376FA8",
    "blue_fill": "#EAF1FA",
    "orange": "#D4772D",
    "orange_fill": "#FCEDDF",
    "green": "#4F8871",
    "green_fill": "#EAF4EE",
    "grey": "#6D7882",
    "grey_fill": "#F3F5F6",
    "boundary": "#DDE5EC",
    "white": "#FFFFFF",
}


def select_font() -> str:
    """选择可用于论文中文标注的字体。"""
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
        "font.size": 6.3,
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
    size: float = 6.3,
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
    zorder: int = 2,
) -> FancyBboxPatch:
    """绘制统一样式的圆角矩形。"""
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
    width: float = 1.05,
    linestyle: str = "-",
    zorder: int = 6,
) -> None:
    """绘制直线数据箭头。"""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=7.6,
            linewidth=width,
            linestyle=linestyle,
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
    color: str,
    width: float = 1.0,
    linestyle: str = "-",
    zorder: int = 5,
) -> None:
    """沿折线路径绘制模块关系箭头。"""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=7.6,
            linewidth=width,
            linestyle=linestyle,
            color=color,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )
    )


def module(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    detail: str,
    *,
    face: str,
    edge: str,
) -> None:
    """绘制功能模块及其职责说明。"""
    rounded_rect(ax, x, y, width, height, face=face, edge=edge, radius=1.0, linewidth=0.85, zorder=4)
    add_text(ax, x + width / 2, y + height * 0.65, title, size=5.9, weight="bold", color=edge)
    add_text(ax, x + width / 2, y + height * 0.29, detail, size=5.15, color=COLORS["muted"])


def flow_label(
    ax: plt.Axes,
    x: float,
    y: float,
    value: str,
    *,
    color: str,
) -> None:
    """直接标注箭头传输的数据。"""
    add_text(ax, x, y, value, size=5.2, color=color, weight="bold")


def draw_pc(ax: plt.Axes) -> None:
    """绘制外部 PC 上位机接口。"""
    rounded_rect(ax, 2.0, 20.0, 19.0, 55.0, face=COLORS["grey_fill"], edge=COLORS["grey"], radius=1.4, linewidth=0.9)
    rounded_rect(ax, 5.0, 44.0, 13.0, 12.0, face=COLORS["white"], edge=COLORS["blue"], radius=0.8, linewidth=0.8, zorder=5)
    ax.add_patch(Rectangle((9.0, 41.8), 5.0, 2.2, facecolor="#D9E0E5", edgecolor=COLORS["grey"], linewidth=0.55, zorder=5))
    ax.plot([7.5, 15.5], [41.7, 41.7], color=COLORS["grey"], linewidth=0.8, zorder=5)
    add_text(ax, 11.5, 61.8, "PC 上位机", size=6.7, weight="bold")
    add_text(ax, 11.5, 36.5, "串口决策与记录", size=5.2, color=COLORS["muted"])
    add_text(ax, 11.5, 70.4, "发送 STEP kL kR", size=5.2, color=COLORS["orange"], weight="bold")
    add_text(ax, 11.5, 24.5, "接收 JSON 数据", size=5.2, color=COLORS["blue"], weight="bold")


def draw_servos(ax: plt.Axes) -> None:
    """绘制外部双舵机执行机构。"""
    rounded_rect(ax, 137.0, 56.0, 19.0, 23.0, face=COLORS["orange_fill"], edge=COLORS["orange"], radius=1.2, linewidth=0.9)
    for cx, direction in ((142.1, -1.0), (150.9, 1.0)):
        rounded_rect(ax, cx - 2.4, 62.0, 4.8, 6.0, face=COLORS["white"], edge=COLORS["orange"], radius=0.7, linewidth=0.7, zorder=5)
        ax.plot([cx, cx + direction * 3.1], [68.0, 73.0], color=COLORS["green"], linewidth=1.25, solid_capstyle="round", zorder=6)
        ax.add_patch(Circle((cx + direction * 3.4, 73.5), 0.9, facecolor="#F1C77C", edgecolor="#98631E", linewidth=0.5, zorder=7))
    add_text(ax, 146.5, 59.0, "左右触须舵机", size=5.5, weight="bold")


def draw_sensors(ax: plt.Axes) -> None:
    """绘制外部双 MQ-3 模拟传感器。"""
    rounded_rect(ax, 137.0, 15.0, 19.0, 24.0, face=COLORS["green_fill"], edge=COLORS["green"], radius=1.2, linewidth=0.9)
    for cx, label in ((142.2, "L"), (150.8, "R")):
        ax.add_patch(Circle((cx, 29.9), 2.3, facecolor="#F1C77C", edgecolor="#98631E", linewidth=0.7, zorder=6))
        ax.add_patch(Rectangle((cx - 1.2, 26.9), 2.4, 1.1, facecolor="#64717C", edgecolor=COLORS["ink"], linewidth=0.45, zorder=5))
        add_text(ax, cx, 30.0, label, size=5.0, color="#754B13", weight="bold")
    add_text(ax, 146.5, 20.5, "双 MQ-3", size=5.6, weight="bold")
    add_text(ax, 146.5, 17.2, "模拟电压 VL, VR", size=5.0, color=COLORS["muted"])


def build_figure() -> plt.Figure:
    """构建命令控制、传感采集与回传闭环。"""
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0.0, WIDTH_MM)
    ax.set_ylim(0.0, HEIGHT_MM)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    draw_pc(ax)
    draw_servos(ax)
    draw_sensors(ax)

    # STM32 功能边界。
    rounded_rect(ax, 25.0, 7.0, 108.0, 79.0, face="#FAFCFE", edge=COLORS["boundary"], radius=1.5, linewidth=1.0, zorder=1)
    ax.add_patch(Rectangle((25.0, 80.0), 108.0, 6.0, facecolor="#DCE7F0", edgecolor="none", zorder=2))
    add_text(ax, 79.0, 83.0, "STM32F103C8T6 下位机｜运行时控制—采样—回传闭环", size=7.0, weight="bold", color=COLORS["ink"])

    # 上方：控制命令链。
    module(ax, 29.0, 60.0, 18.0, 14.0, "USART1 接收", "115200｜行缓冲", face=COLORS["orange_fill"], edge=COLORS["orange"])
    module(ax, 52.0, 60.0, 25.0, 14.0, "命令解析与校验", "STEP kL kR｜扇区 0–9", face=COLORS["orange_fill"], edge=COLORS["orange"])
    module(ax, 82.0, 60.0, 23.0, 14.0, "扇区映射与校准", "9°–171°｜反向与微调", face=COLORS["orange_fill"], edge=COLORS["orange"])
    module(ax, 110.0, 60.0, 19.0, 14.0, "TIM2 PWM 输出", "CH1 / CH2｜50 Hz", face=COLORS["orange_fill"], edge=COLORS["orange"])

    arrow(ax, (21.2, 67.0), (28.8, 67.0), color=COLORS["orange"], width=1.15)
    arrow(ax, (47.2, 67.0), (51.8, 67.0), color=COLORS["orange"])
    arrow(ax, (77.2, 67.0), (81.8, 67.0), color=COLORS["orange"])
    arrow(ax, (105.2, 67.0), (109.8, 67.0), color=COLORS["orange"])
    arrow(ax, (129.2, 67.0), (136.8, 67.0), color=COLORS["orange"], width=1.15)

    flow_label(ax, 49.5, 77.0, "字符缓冲", color=COLORS["orange"])
    flow_label(ax, 79.5, 77.0, "kL, kR", color=COLORS["orange"])
    flow_label(ax, 107.5, 77.0, "角度/比较值", color=COLORS["orange"])

    # 下方：传感采集与数据回传链，方向由右向左。
    module(ax, 107.0, 22.0, 22.0, 14.0, "ADC1 双通道采样", "IN2 / IN3｜软件触发", face=COLORS["blue_fill"], edge=COLORS["blue"])
    module(ax, 85.0, 22.0, 17.0, 14.0, "采样平均", "每通道 10 次", face=COLORS["blue_fill"], edge=COLORS["blue"])
    module(ax, 55.0, 22.0, 25.0, 14.0, "响应数据封装", "扇区 + 左右 ADC", face=COLORS["blue_fill"], edge=COLORS["blue"])
    module(ax, 30.0, 22.0, 19.0, 14.0, "USART1 发送", "JSON + CRLF", face=COLORS["blue_fill"], edge=COLORS["blue"])

    arrow(ax, (136.8, 29.0), (129.2, 29.0), color=COLORS["blue"], width=1.15)
    arrow(ax, (106.8, 29.0), (102.2, 29.0), color=COLORS["blue"])
    arrow(ax, (84.8, 29.0), (80.2, 29.0), color=COLORS["blue"])
    arrow(ax, (54.8, 29.0), (49.2, 29.0), color=COLORS["blue"])
    arrow(ax, (29.8, 29.0), (21.2, 29.0), color=COLORS["blue"], width=1.15)

    flow_label(ax, 104.5, 39.0, "ADC 序列", color=COLORS["blue"])
    flow_label(ax, 82.5, 39.0, "均值", color=COLORS["blue"])
    flow_label(ax, 52.0, 39.0, "JSON 字符串", color=COLORS["blue"])

    # 舵机动作完成后再触发传感器采样。
    path_arrow(
        ax,
        [(119.5, 59.8), (131.0, 59.8), (131.0, 45.0), (118.0, 45.0), (118.0, 36.2)],
        color=COLORS["grey"],
        width=0.95,
        linestyle="--",
        zorder=5,
    )
    add_text(ax, 119.0, 48.5, "舵机稳定 500 ms 后触发采样", size=5.2, color=COLORS["grey"], weight="bold")

    # 非法格式或越界命令直接形成错误响应，不驱动舵机与 ADC。
    path_arrow(
        ax,
        [(64.5, 59.8), (64.5, 51.0), (42.0, 51.0), (42.0, 36.2)],
        color=COLORS["grey"],
        width=0.9,
        linestyle="--",
        zorder=5,
    )
    add_text(ax, 48.0, 55.2, "格式/范围错误 → error JSON", size=5.2, color=COLORS["grey"], weight="bold")

    # 明确回传字段，放在下位机边界底部，不遮挡主链。
    rounded_rect(ax, 46.0, 10.0, 67.0, 7.5, face=COLORS["blue_fill"], edge="none", radius=0.8, linewidth=0.0, zorder=3)
    add_text(
        ax,
        79.5,
        13.8,
        "回传字段：{left_sector, right_sector, left_adc, right_adc}",
        size=5.25,
        color=COLORS["blue"],
        weight="bold",
    )

    return fig


def save_figure(fig: plt.Figure) -> None:
    """导出 SVG、PDF、600 dpi TIFF 与 300 dpi PNG。"""
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"Title": "STM32 双触须下位机功能模块与数据流"}
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
