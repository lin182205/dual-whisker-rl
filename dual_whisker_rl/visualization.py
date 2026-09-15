"""Trajectory visualization utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from matplotlib import animation
import matplotlib.pyplot as plt

from dual_whisker_rl.plotting import draw_odor_field


def save_matplotlib_animation(
    anim: animation.Animation,
    path: Path,
    *,
    fps: float,
    dpi: float | None = None,
) -> None:
    """按输出扩展名保存 Matplotlib 动画，默认推荐 H.264 MP4。"""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "导出 MP4 需要 FFmpeg。请安装 FFmpeg、将 ffmpeg 加入 PATH，"
                "或显式传入 .gif 输出路径。"
            )
        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            extra_args=[
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-crf",
                "21",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
    elif suffix == ".gif":
        writer = animation.PillowWriter(fps=fps)
    else:
        raise ValueError(
            f"不支持的动画格式 {suffix or '<无扩展名>'}；只支持 .mp4 和 .gif。"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"writer": writer}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    anim.save(path, **save_kwargs)


def save_trajectory_animation(
    env: Any,
    trajectory: list[dict[str, Any]],
    path: Path,
    *,
    label: str,
    stride: int = 1,
    fps: int = 4,
) -> None:
    """按路径扩展名保存机器人轨迹动画。"""

    if not trajectory:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    frames = trajectory[:: max(1, stride)]
    if frames[-1] is not trajectory[-1]:
        frames.append(trajectory[-1])

    xx, yy, zz = env.concentration_grid(resolution=120)
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_odor_field(ax, xx, yy, zz, alpha=0.9)
    sx, sy = env.plume.params.source
    ax.scatter([sx], [sy], marker="*", s=180, c="red", label="source")
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(label)

    (path_line,) = ax.plot([], [], color="white", linewidth=1.5, label="trajectory")
    robot_dot = ax.scatter([], [], c="black", s=45, zorder=5)
    heading_line, = ax.plot([], [], color="black", linewidth=1.8, zorder=5)
    left_whisker, = ax.plot([], [], color="#4c78a8", linewidth=1.6, zorder=5)
    right_whisker, = ax.plot([], [], color="#f58518", linewidth=1.6, zorder=5)
    step_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="white",
        bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
    )
    ax.legend(loc="lower right")

    xs: list[float] = []
    ys: list[float] = []

    def update(frame_index: int):
        row = frames[frame_index]
        x = float(row["x"])
        y = float(row["y"])
        heading = float(row["heading"])
        left_angle = float(row["left_angle"])
        right_angle = float(row["right_angle"])
        length = float(env.whiskers.length)

        xs.append(x)
        ys.append(y)
        path_line.set_data(xs, ys)
        robot_dot.set_offsets([[x, y]])

        heading_len = 0.25
        heading_line.set_data(
            [x, x + heading_len * math.cos(heading)],
            [y, y + heading_len * math.sin(heading)],
        )
        left_whisker.set_data(
            [x, x + length * math.cos(heading + left_angle)],
            [y, y + length * math.sin(heading + left_angle)],
        )
        right_whisker.set_data(
            [x, x + length * math.cos(heading + right_angle)],
            [y, y + length * math.sin(heading + right_angle)],
        )
        step = min(frame_index * max(1, stride), len(trajectory) - 1)
        left_sector = row.get("left_sector", "-")
        right_sector = row.get("right_sector", "-")
        step_text.set_text(
            f"step {step}\n"
            f"L {float(row['left']):.3f}  R {float(row['right']):.3f}\n"
            f"sector L{left_sector} R{right_sector}"
        )
        return path_line, robot_dot, heading_line, left_whisker, right_whisker, step_text

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    save_matplotlib_animation(anim, path, fps=fps)
    plt.close(fig)
