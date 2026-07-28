"""生成 sensor_preprocess.py 数据处理流程的论文用示意图。

输出：
- figures/sensor_preprocess_pipeline.pdf （矢量，投稿首选）
- figures/sensor_preprocess_pipeline.png （300 dpi 预览）

图中参数取自 SensorPreprocessConfig 默认值。
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.font_manager import FontProperties

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.paths import project_path

# ---- 中文字体 ----
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 配色（柔和、印刷友好）
C_INPUT = "#E8EEF6"
C_INPUT_E = "#3B6EA5"
C_PROC = "#FBF0DC"
C_PROC_E = "#C9952B"
C_FEAT = "#E6F2E6"
C_FEAT_E = "#4E8C53"
C_GATE = "#F6E3E3"
C_GATE_E = "#B5524F"
C_OUT = "#E7E4F2"
C_OUT_E = "#6A55A0"
C_ARROW = "#444444"

TITLE_FP = FontProperties(family="Microsoft YaHei", size=15, weight="bold")


def box(ax, x, y, w, h, text, fc, ec, fs=10.5, weight="normal", lw=1.6):
    p = FancyBboxPatch(
        (x - w / 2, y - h / 2),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=2,
    )
    ax.add_patch(p)
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fs,
        weight=weight,
        zorder=3,
        linespacing=1.35,
    )
    return (x, y, w, h)


def arrow(ax, p0, p1, style="-|>", color=C_ARROW, lw=1.7, ls="-", rad=0.0):
    a = FancyArrowPatch(
        p0,
        p1,
        arrowstyle=style,
        mutation_scale=14,
        linewidth=lw,
        color=color,
        linestyle=ls,
        connectionstyle=f"arc3,rad={rad}",
        zorder=1,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(a)


def right(b):
    return (b[0] + b[2] / 2, b[1])


def left(b):
    return (b[0] - b[2] / 2, b[1])


def top(b):
    return (b[0], b[1] + b[3] / 2)


def bottom(b):
    return (b[0], b[1] - b[3] / 2)


fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(13.2, 9.6), gridspec_kw={"height_ratios": [1.55, 1.0]}
)

# ============================================================
# 面板 A：单个 MQ-3 传感器在线预处理链
# ============================================================
ax = axA
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")
ax.text(
    1, 57.5, "(a) 单个 MQ-3 传感器在线预处理流程", fontproperties=TITLE_FP, ha="left"
)

# 主链 y
yM = 38
bw, bh = 15, 9

raw = box(ax, 9, yM, 13, 8,
          "原始 ADC\n$x_t$", C_INPUT, C_INPUT_E, fs=11.5, weight="bold")

sig = box(ax, 30, yM, bw, bh,
          "去基线\n$s_t = x_t - b_t$", C_PROC, C_PROC_E)

smooth = box(ax, 51, yM, bw, bh,
             "EMA 平滑\n$\\tilde{s}_t$\n$\\tau_s=2\\,$s", C_PROC, C_PROC_E)

# 平滑后分两路
trend = box(ax, 72, yM + 11, bw, bh,
            "趋势估计\n$d_t$ (窗口=8)\n每步平均斜率", C_PROC, C_PROC_E, fs=10)
est = box(ax, 72, yM - 11, bw, bh,
          "一阶响应反推\n$\\hat{u}_t$\n$\\tau_r=2\\,$s", C_PROC, C_PROC_E, fs=10)

# 归一化
norm = box(ax, 90, yM, 14, 30,
           "归一化\n+ 裁剪\n\n$\\div$ scale=200\nclip $\\pm5$",
           C_FEAT, C_FEAT_E, fs=10.5, weight="bold")

# 主链箭头
arrow(ax, right(raw), left(sig))
arrow(ax, right(sig), left(smooth))
# smooth -> trend / est (分叉)
arrow(ax, right(smooth), left(trend), rad=-0.18)
arrow(ax, right(smooth), left(est), rad=0.18)
# signal 也直接进入归一化（旁路）
arrow(ax, top(sig), (30, 56), rad=0.0)
arrow(ax, (30, 56), (norm[0], 56), rad=0.0)
arrow(ax, (norm[0], 56), top(norm))
ax.text(60, 57.3, "$s_t$ 直接送入归一化", ha="center", fontsize=9, color="#666")

# smooth/trend/est -> norm
arrow(ax, right(trend), (norm[0] - norm[2] / 2, norm[1] + 9), rad=-0.12)
arrow(ax, right(est), (norm[0] - norm[2] / 2, norm[1] - 9), rad=0.12)
arrow(ax, top(smooth), (51, 48), rad=0)
arrow(ax, (51, 48), (norm[0] - norm[2] / 2, norm[1] + 3), rad=-0.05)

# 归一化输出特征向量
feat = box(ax, 90, 9, 16, 8,
           "单传感器特征 (4 维)\n$[\\,\\bar{s},\\ \\bar{\\tilde{s}},\\ \\bar{d},\\ \\bar{\\hat{u}}\\,]$",
           C_FEAT, C_FEAT_E, fs=10, weight="bold")
arrow(ax, bottom(norm), top(feat))

# 基线自适应更新（门控反馈）
gate = box(ax, 30, 14, 30, 11,
           "基线更新门控（条件冻结）\n仅当 $|\\bar{\\tilde{s}}|\\leq0.2$ 且 $|\\bar{d}|\\leq0.05$\n"
           "$b_{t+1}=\\alpha b_t+(1-\\alpha)x_t,\\ \\tau_b=180\\,$s",
           C_GATE, C_GATE_E, fs=9.5)
# smooth & trend 提供判据
arrow(ax, bottom(smooth), (gate[0] + 8, top(gate)[1]), color=C_GATE_E, ls=(0, (4, 3)), rad=0.12)
# gate -> baseline（更新 b_t，反馈到去基线节点）
arrow(ax, top(gate), bottom(sig), color=C_GATE_E, ls=(0, (4, 3)), rad=0.0)
ax.text(46, 20.5, "气味刺激时\n自动冻结基线", ha="left", fontsize=8.3, color=C_GATE_E,
        linespacing=1.2)

# ============================================================
# 面板 B：左右传感器融合 -> PPO 观测
# ============================================================
ax = axB
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.axis("off")
ax.text(1, 37, "(b) 左右双传感器融合与 PPO 观测构造",
        fontproperties=TITLE_FP, ha="left")

leftS = box(ax, 14, 26, 22, 9,
            "左 MQ-3\n预处理链 (a)\n$\\to$ 特征 (4 维)", C_INPUT, C_INPUT_E, fs=10)
rightS = box(ax, 14, 8, 22, 9,
             "右 MQ-3\n预处理链 (a)\n$\\to$ 特征 (4 维)", C_INPUT, C_INPUT_E, fs=10)

diff = box(ax, 48, 17, 22, 11,
           "差分特征 (4 维)\n$\\Delta\\bar{s},\\ \\Delta\\bar{\\tilde{s}},$\n"
           "$\\Delta\\bar{d},\\ \\Delta\\bar{\\hat{u}}$\n(左 $-$ 右)",
           C_PROC, C_PROC_E, fs=9.5)

obs = box(ax, 82, 17, 26, 14,
          "PPO 观测向量 (12 维)\n左特征 (4) + 右特征 (4)\n+ 差分特征 (4)",
          C_OUT, C_OUT_E, fs=10.5, weight="bold")

arrow(ax, right(leftS), left(diff), rad=-0.12)
arrow(ax, right(rightS), left(diff), rad=0.12)
arrow(ax, right(diff), left(obs))
# 左右特征也直接进观测（旁路，浅色虚线）
arrow(ax, top(leftS), (obs[0] - obs[2] / 2, obs[1] + obs[3] / 2 - 2.5),
      rad=-0.18, color="#9A9A9A", ls=(0, (5, 3)), lw=1.4)
arrow(ax, bottom(rightS), (obs[0] - obs[2] / 2, obs[1] - obs[3] / 2 + 2.5),
      rad=0.18, color="#9A9A9A", ls=(0, (5, 3)), lw=1.4)
ax.text(50, 33, "左/右特征旁路直连观测", ha="center", fontsize=8.5, color="#777")

fig.tight_layout(pad=1.2)

out_dir = project_path("figures")
assert out_dir is not None
out_dir.mkdir(parents=True, exist_ok=True)
pdf = out_dir / "sensor_preprocess_pipeline.pdf"
png = out_dir / "sensor_preprocess_pipeline.png"
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=300, bbox_inches="tight")
print("saved:", pdf)
print("saved:", png)
