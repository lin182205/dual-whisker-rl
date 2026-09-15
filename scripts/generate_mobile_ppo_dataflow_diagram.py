"""生成移动双触须 PPO 闭环数据流论文流程图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_STEM = ROOT / "results" / "figures" / "mobile_ppo_dataflow"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans", "sans-serif"],
        "font.size": 6.4,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.linewidth": 0.8,
    }
)


COLORS = {
    "ink": "#243447",
    "muted": "#637487",
    "line": "#8A9AAA",
    "env": "#DCEAF5",
    "env_edge": "#4F7FA5",
    "obs": "#DDF1EC",
    "obs_edge": "#3B8B7C",
    "policy": "#EAE3F3",
    "policy_edge": "#76569A",
    "reward": "#F8E7D2",
    "reward_edge": "#C67A2B",
    "buffer": "#EEF1F4",
    "white": "#FFFFFF",
}


def rounded_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    detail: str,
    *,
    face: str,
    edge: str,
    title_size: float = 7.0,
    detail_size: float = 5.8,
    linewidth: float = 1.0,
) -> dict[str, tuple[float, float]]:
    """绘制带标题和说明的圆角节点，并返回连接锚点。"""
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.28,rounding_size=0.9",
        linewidth=linewidth,
        edgecolor=edge,
        facecolor=face,
        zorder=3,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        color=COLORS["ink"],
        fontsize=title_size,
        fontweight="bold",
        linespacing=1.12,
        zorder=4,
    )
    ax.text(
        x + w / 2,
        y + h * 0.29,
        detail,
        ha="center",
        va="center",
        color=COLORS["muted"],
        fontsize=detail_size,
        linespacing=1.12,
        zorder=4,
    )
    return {
        "left": (x, y + h / 2),
        "right": (x + w, y + h / 2),
        "top": (x + w / 2, y + h),
        "bottom": (x + w / 2, y),
        "center": (x + w / 2, y + h / 2),
    }


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["line"],
    width: float = 1.15,
    style: str = "-",
    connection: str = "arc3,rad=0",
    zorder: int = 2,
) -> None:
    """绘制统一风格的单向连接箭头。"""
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=width,
            linestyle=style,
            color=color,
            connectionstyle=connection,
            shrinkA=1.2,
            shrinkB=1.2,
            zorder=zorder,
        )
    )


def path_arrow(
    ax: plt.Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    width: float = 1.2,
    style: str = "-",
    zorder: int = 2,
) -> None:
    """沿折线路径绘制反馈箭头，避免穿过节点与文字。"""
    codes = [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1)
    path = MplPath(points, codes)
    ax.add_patch(
        FancyArrowPatch(
            path=path,
            arrowstyle="-|>",
            mutation_scale=8.5,
            linewidth=width,
            linestyle=style,
            color=color,
            joinstyle="round",
            capstyle="round",
            zorder=zorder,
        )
    )


def band(
    ax: plt.Axes,
    y: float,
    h: float,
    label: str,
    color: str,
    *,
    label_x: float = 2.0,
) -> None:
    """绘制流程阶段背景带。"""
    ax.add_patch(Rectangle((1.0, y), 98.0, h, facecolor=color, edgecolor="none", alpha=0.30))
    ax.text(
        label_x,
        y + h - 1.0,
        label,
        ha="left",
        va="top",
        fontsize=6.0,
        fontweight="bold",
        color=COLORS["muted"],
    )


def build_figure() -> plt.Figure:
    """构建三阶段闭环流程图。"""
    width_mm = 183.0
    height_mm = 118.0
    width_in = width_mm / 25.4
    height_in = height_mm / 25.4
    fig, ax = plt.subplots(figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    fig.patch.set_facecolor(COLORS["white"])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 72)
    ax.axis("off")

    ax.text(
        2.0,
        70.0,
        "移动双触须 PPO：从动态羽流到 rollout buffer 的闭环数据流",
        ha="left",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color=COLORS["ink"],
    )
    ax.text(
        98.0,
        70.0,
        "每个并行环境独立推进",
        ha="right",
        va="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )

    band(ax, 51.0, 16.0, "A  环境与真实传感链", COLORS["env"])
    band(ax, 31.0, 16.0, "B  硬件可部署观测与时间上下文", COLORS["obs"])
    band(
        ax,
        5.0,
        22.0,
        "C  PPO 决策、反馈与采样存储",
        COLORS["policy"],
        label_x=62.0,
    )

    scene = rounded_box(
        ax,
        3.0,
        54.0,
        19.0,
        9.5,
        "场景初始化",
        "气源 / 风场 / 机器人位姿",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
    )
    dynamics = rounded_box(
        ax,
        27.0,
        54.0,
        20.0,
        9.5,
        "状态推进",
        "机器人 + 动态羽流 + 双触须",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
    )
    concentration = rounded_box(
        ax,
        52.0,
        54.0,
        19.0,
        9.5,
        "双端点真实浓度",
        "c_L(t), c_R(t)",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
    )
    sensor = rounded_box(
        ax,
        76.0,
        54.0,
        21.0,
        9.5,
        "非对称慢响应传感器",
        "快速响应 · 缓慢恢复 · 噪声",
        face=COLORS["env"],
        edge=COLORS["env_edge"],
    )

    preprocess = rounded_box(
        ax,
        73.0,
        34.0,
        24.0,
        9.5,
        "在线预处理",
        "基线扣除 / EMA / 趋势 / 左右差分",
        face=COLORS["obs"],
        edge=COLORS["obs_edge"],
    )
    obs = rounded_box(
        ax,
        49.0,
        34.0,
        19.0,
        9.5,
        "16 维单帧观测",
        "12 气味-触须 + 3 本体\n+ blank age",
        face=COLORS["obs"],
        edge=COLORS["obs_edge"],
    )
    history = rounded_box(
        ax,
        27.0,
        34.0,
        17.0,
        9.5,
        "20 帧历史堆叠",
        "16 × 20 = 320 维",
        face=COLORS["obs"],
        edge=COLORS["obs_edge"],
    )
    encoder = rounded_box(
        ax,
        3.0,
        34.0,
        19.0,
        9.5,
        "时序编码器",
        "GRU / Transformer\nMLP 基线",
        face=COLORS["obs"],
        edge=COLORS["obs_edge"],
    )

    latent = rounded_box(
        ax,
        4.0,
        13.0,
        16.0,
        8.0,
        "时序表征 h_t",
        "共享特征",
        face=COLORS["policy"],
        edge=COLORS["policy_edge"],
    )
    actor = rounded_box(
        ax,
        26.0,
        17.5,
        16.0,
        7.0,
        "Actor",
        "πθ(a_t | h_t), log π_t",
        face=COLORS["policy"],
        edge=COLORS["policy_edge"],
        detail_size=5.5,
    )
    critic = rounded_box(
        ax,
        26.0,
        8.0,
        16.0,
        7.0,
        "Critic",
        "Vφ(h_t)",
        face=COLORS["policy"],
        edge=COLORS["policy_edge"],
    )
    outputs = rounded_box(
        ax,
        49.0,
        12.5,
        20.0,
        9.0,
        "联合动作与状态价值",
        "a_t: [移动, 左扇区, 右扇区]\nV_t: 状态价值",
        face=COLORS["policy"],
        edge=COLORS["policy_edge"],
        detail_size=5.4,
    )
    transition = rounded_box(
        ax,
        76.0,
        13.0,
        21.0,
        8.0,
        "环境反馈",
        "奖励 r_t  +  下一观测 o_(t+1)",
        face=COLORS["reward"],
        edge=COLORS["reward_edge"],
    )
    buffer_box = rounded_box(
        ax,
        62.0,
        5.8,
        35.0,
        5.1,
        "rollout buffer",
        "(o_t, a_t, log π_t, V_t, r_t, o_(t+1), done_t)",
        face=COLORS["buffer"],
        edge=COLORS["line"],
        title_size=6.4,
        detail_size=5.1,
        linewidth=0.9,
    )

    arrow(ax, scene["right"], dynamics["left"], color=COLORS["env_edge"])
    arrow(ax, dynamics["right"], concentration["left"], color=COLORS["env_edge"])
    arrow(ax, concentration["right"], sensor["left"], color=COLORS["env_edge"])
    arrow(
        ax,
        sensor["bottom"],
        preprocess["top"],
        color=COLORS["obs_edge"],
        connection="arc3,rad=0.08",
    )
    arrow(ax, preprocess["left"], obs["right"], color=COLORS["obs_edge"])
    arrow(ax, obs["left"], history["right"], color=COLORS["obs_edge"])
    arrow(ax, history["left"], encoder["right"], color=COLORS["obs_edge"])
    arrow(ax, encoder["bottom"], latent["top"], color=COLORS["policy_edge"])

    arrow(ax, latent["right"], actor["left"], color=COLORS["policy_edge"], connection="arc3,rad=-0.12")
    arrow(ax, latent["right"], critic["left"], color=COLORS["policy_edge"], connection="arc3,rad=0.12")
    arrow(ax, actor["right"], outputs["left"], color=COLORS["policy_edge"], connection="arc3,rad=-0.08")
    arrow(ax, critic["right"], outputs["left"], color=COLORS["policy_edge"], connection="arc3,rad=0.08")
    arrow(ax, outputs["right"], transition["left"], color=COLORS["reward_edge"])
    arrow(ax, transition["bottom"], buffer_box["top"], color=COLORS["line"], style="--", width=1.0)

    # 动作反馈沿节点间隙回到环境状态推进，不穿过观测节点。
    path_arrow(
        ax,
        [(59.0, 21.5), (59.0, 28.6), (23.8, 28.6), (23.8, 50.0), (37.0, 54.0)],
        color=COLORS["policy_edge"],
        width=1.35,
        zorder=1,
    )
    ax.text(
        42.0,
        29.2,
        "联合动作 a_t",
        ha="center",
        va="bottom",
        fontsize=5.7,
        fontweight="bold",
        color=COLORS["policy_edge"],
        zorder=5,
    )

    # 下一观测沿右侧回到在线预处理，构成下一个决策时刻。
    path_arrow(
        ax,
        [transition["top"], (98.2, 24.5), (98.2, 31.5), (85.0, 34.0)],
        color=COLORS["obs_edge"],
        width=1.25,
        zorder=1,
    )

    ax.text(
        2.0,
        1.9,
        "闭环含义：动作改变机器人与触须采样位置；环境反馈更新历史窗口，并与价值估计共同形成 PPO rollout。",
        ha="left",
        va="center",
        fontsize=5.6,
        color=COLORS["muted"],
    )
    return fig


def save_figure(fig: plt.Figure) -> None:
    """导出可编辑矢量图与高分辨率栅格图。"""
    OUTPUT_STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"))
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"))
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300)
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600)


def main() -> None:
    fig = build_figure()
    save_figure(fig)
    plt.close(fig)
    print(f"saved: {OUTPUT_STEM}")


if __name__ == "__main__":
    main()
