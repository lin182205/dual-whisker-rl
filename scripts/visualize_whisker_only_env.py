"""可视化触须独立训练环境，图像风格参考 TD3 puff plume demo。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
from matplotlib import colors
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs.whisker_only_env import WhiskerOnlyPuffEnv
from dual_whisker_rl.envs.whisker_only_env import WORLD_MAX
from dual_whisker_rl.envs.whisker_only_env import WORLD_MIN
from dual_whisker_rl.paths import resolve_path_args
from dual_whisker_rl.visualization import save_matplotlib_animation


def build_plume_colormap() -> colors.LinearSegmentedColormap:
    """参考图风格：低浓度融入奶白背景，羽流随浓度加深为蓝，峰值跳到黄色。

    低端锚定到与 facecolor 一致的奶白，使空白区（浓度≈0）与背景无缝衔接；
    浓度升高时蓝色逐渐加深至深海军蓝（羽流核心），最高浓度过渡到黄色高亮。
    """
    return colors.LinearSegmentedColormap.from_list(
        "puff_demo",
        [
            (0.00, "#f7f4ef"),
            (0.15, "#cfe2ec"),
            (0.35, "#93c1e0"),
            (0.55, "#4f93cc"),
            (0.75, "#22619f"),
            (0.90, "#123b6b"),
            (0.97, "#0c2747"),
            (1.00, "#ffe27a"),
        ],
        N=256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a SimpleGasEnv-style puff animation for the whisker-only environment."
    )
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--png-path", type=Path, default=Path("results/figures/whisker_only_env.png"))
    parser.add_argument(
        "--animation-path",
        "--gif-path",
        dest="animation_path",
        type=Path,
        default=Path("results/figures/whisker_only_env.mp4"),
        help="动画输出路径；扩展名支持 .mp4 或 .gif，--gif-path 为兼容别名。",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="手动指定动画帧率；默认根据环境 dt 和 realtime-speed 自动计算。",
    )
    parser.add_argument(
        "--realtime-speed",
        type=float,
        default=1.0,
        help="播放速度倍率1.0 表示动画以 1 倍仿真时间播放。",
    )
    parser.add_argument(
        "--animation-stride",
        type=int,
        default=1,
        help="每隔多少个仿真步取一个关键帧；增大可减小文件，但会损失细节。",
    )
    parser.add_argument(
        "--interpolation-frames",
        type=int,
        default=1,
        help="相邻关键帧之间插入的过渡帧数；越大越丝滑但文件越大。",
    )
    parser.add_argument("--title", type=str, default="Whisker-only Puff Sampling")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/models/whisker_only_ppo.zip"),
        help="给定则用训练好的 PPO 策略驱动触须；不给则用随机动作。",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="策略采样动作（默认 deterministic，更能看清学到的稳定偏好）。",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="可视化环境也开域随机化（与训练一致的分布）。",
    )
    return resolve_path_args(parser.parse_args(), "png_path", "animation_path", "model_path")


def capture_rollout(
    env: WhiskerOnlyPuffEnv,
    max_steps: int,
    grid_resolution: int,
    seed: int,
    model: object | None = None,
    history_length: int = 1,
    deterministic: bool = True,
) -> dict[str, object]:
    """采样一段 rollout 并缓存每帧气体场用于动画。

    `model` 为 None 时用随机触须动作；给定 PPO 模型时，维护一个与训练
    `ObservationHistoryWrapper` 完全一致的历史缓冲（首帧复制填满、每步 append、
    从旧到新拼接），用堆叠观测驱动策略，使可视化反映真实学到的行为。
    """
    from collections import deque

    rng = np.random.default_rng(seed + 123)
    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    history: deque[np.ndarray] = deque(maxlen=max(1, history_length))
    for _ in range(history.maxlen):
        history.append(obs.copy())
    map_metadata = env.get_map_metadata()
    xs, ys, initial_grid = env.concentration_grid(resolution=grid_resolution)

    robot_state = map_metadata["robot_position"]
    trajectory = [robot_state]
    robot_headings = [float(map_metadata["robot_heading"])]
    left_points = []
    right_points = []
    left_sectors = [0]
    right_sectors = [0]
    sensor_values = [(0.0, 0.0)]
    concentration_frames = [initial_grid.copy()]
    wind_directions = [float(env.field.wind_direction)]
    wind_speeds = [float(env.field.wind_speed)]
    frame_steps = [0]
    frame_rewards = [0.0]
    cumulative_rewards = [0.0]
    total_reward = 0.0

    initial_whisker = env._observe()[1]["whisker_state"]
    left_points.append(initial_whisker.left_point)
    right_points.append(initial_whisker.right_point)

    for step_idx in range(max_steps):
        if model is None:
            action = [
                int(rng.integers(0, env.whiskers.sector_count)),
                int(rng.integers(0, env.whiskers.sector_count)),
            ]
        else:
            stacked = np.concatenate(tuple(history)).astype(np.float32)
            action, _ = model.predict(stacked, deterministic=deterministic)
            action = [int(action[0]), int(action[1])]
        obs, reward, _, _, info = env.step(action)
        history.append(np.asarray(obs, dtype=np.float32))
        total_reward += reward
        obs_info = env._observe()[1]
        whisker_state = obs_info["whisker_state"]
        trajectory.append((env.robot_state.x, env.robot_state.y))
        robot_headings.append(float(env.robot_state.heading))
        left_points.append(whisker_state.left_point)
        right_points.append(whisker_state.right_point)
        left_sectors.append(int(info["left_sector"]))
        right_sectors.append(int(info["right_sector"]))
        sensor_values.append((float(info["left"]), float(info["right"])))
        _, _, frame_grid = env.concentration_grid(resolution=grid_resolution)
        concentration_frames.append(frame_grid.copy())
        wind_directions.append(float(env.field.wind_direction))
        wind_speeds.append(float(env.field.wind_speed))
        frame_steps.append(step_idx + 1)
        frame_rewards.append(float(reward))
        cumulative_rewards.append(float(total_reward))

    return {
        "trajectory": np.asarray(trajectory, dtype=np.float32),
        "robot_headings": np.asarray(robot_headings, dtype=np.float32),
        "left_points": np.asarray(left_points, dtype=np.float32),
        "right_points": np.asarray(right_points, dtype=np.float32),
        "whisker_length": float(env.unwrapped.whiskers.length),
        "left_sectors": np.asarray(left_sectors, dtype=np.int32),
        "right_sectors": np.asarray(right_sectors, dtype=np.int32),
        "sensor_values": np.asarray(sensor_values, dtype=np.float32),
        "xs": xs,
        "ys": ys,
        "concentration_frames": np.asarray(concentration_frames, dtype=np.float32),
        "wind_directions": np.asarray(wind_directions, dtype=np.float32),
        "wind_speeds": np.asarray(wind_speeds, dtype=np.float32),
        "frame_steps": np.asarray(frame_steps, dtype=np.int32),
        "frame_rewards": np.asarray(frame_rewards, dtype=np.float32),
        "cumulative_rewards": np.asarray(cumulative_rewards, dtype=np.float32),
        "reward": float(total_reward),
        "steps": max_steps,
        "dt": float(env.unwrapped.plume.dt),
        "termination": "timeout",
        "map_metadata": map_metadata,
    }


def draw_whisker_artists(
    ax,
    trajectory: np.ndarray,
    left_points: np.ndarray,
    right_points: np.ndarray,
    frame_idx: int,
):
    robot_xy = trajectory[frame_idx]
    left_xy = left_points[frame_idx]
    right_xy = right_points[frame_idx]
    left_line, = ax.plot(
        [robot_xy[0], left_xy[0]],
        [robot_xy[1], left_xy[1]],
        color="#4ea5ff",
        linewidth=2.4,
        alpha=0.95,
        zorder=6,
    )
    right_line, = ax.plot(
        [robot_xy[0], right_xy[0]],
        [robot_xy[1], right_xy[1]],
        color="#ff9f43",
        linewidth=2.4,
        alpha=0.95,
        zorder=6,
    )
    left_tip = ax.scatter(
        left_xy[0],
        left_xy[1],
        s=34,
        color="#4ea5ff",
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )
    right_tip = ax.scatter(
        right_xy[0],
        right_xy[1],
        s=34,
        color="#ff9f43",
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )
    return left_line, right_line, left_tip, right_tip


def robot_heading_endpoint(
    robot_xy: np.ndarray,
    heading: float,
    length: float = 0.07,
) -> tuple[float, float]:
    """计算机器人朝向箭头终点。"""
    return (
        float(robot_xy[0] + length * math.cos(heading)),
        float(robot_xy[1] + length * math.sin(heading)),
    )


def draw_robot_heading_marker(
    ax,
    robot_xy: np.ndarray,
    heading: float,
    length: float = 0.07,
):
    """在机器人中心绘制一个短箭头，用于表示当前车体朝向。"""
    start = (float(robot_xy[0]), float(robot_xy[1]))
    end = robot_heading_endpoint(robot_xy, heading, length=length)
    shadow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=5.0,
        color="white",
        alpha=0.95,
        shrinkA=2.0,
        shrinkB=0.0,
        zorder=8,
    )
    arrow = patches.FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=2.4,
        color="#111111",
        shrinkA=2.0,
        shrinkB=0.0,
        zorder=9,
    )
    ax.add_patch(shadow)
    ax.add_patch(arrow)
    return shadow, arrow


def interpolate_angle(angle0: float, angle1: float, alpha: float) -> float:
    """沿最短角度方向插值，避免跨越 -pi/pi 时突然反向旋转。"""
    delta = math.atan2(math.sin(angle1 - angle0), math.cos(angle1 - angle0))
    return angle0 + alpha * delta


def whisker_point_from_angle(
    robot_xy: np.ndarray,
    angle: float,
    length: float,
) -> np.ndarray:
    """按固定长度从角度重新计算触须端点。"""
    return np.asarray(
        [
            robot_xy[0] + length * math.cos(angle),
            robot_xy[1] + length * math.sin(angle),
        ],
        dtype=np.float32,
    )


def interpolate_whisker_point(
    robot_xy: np.ndarray,
    robot0: np.ndarray,
    point0: np.ndarray,
    robot1: np.ndarray,
    point1: np.ndarray,
    alpha: float,
    length: float,
) -> np.ndarray:
    """插帧触须端点：插值角度，再用固定长度重建端点。

    直接对端点坐标做线性插值会让端点走直线，视觉上触须会变短。
    这里改为角度插值，使端点沿圆弧运动，长度始终保持不变。
    """
    angle0 = math.atan2(point0[1] - robot0[1], point0[0] - robot0[0])
    angle1 = math.atan2(point1[1] - robot1[1], point1[0] - robot1[0])
    angle = interpolate_angle(angle0, angle1, alpha)
    return whisker_point_from_angle(robot_xy, angle, length)


def draw_obstacles(ax, obstacles: np.ndarray) -> None:
    for xmin, xmax, ymin, ymax in obstacles:
        ax.add_patch(
            patches.Rectangle(
                (float(xmin), float(ymin)),
                float(xmax - xmin),
                float(ymax - ymin),
                facecolor="#333333",
                edgecolor="#f2f2f2",
                linewidth=1.8,
                alpha=0.95,
                zorder=4,
            )
        )


def draw_source_marker(ax, map_metadata: dict[str, object]) -> None:
    source_x, source_y = map_metadata["source_position"]
    clipped_x = float(np.clip(source_x, WORLD_MIN + 0.02, WORLD_MAX - 0.02))
    clipped_y = float(np.clip(source_y, WORLD_MIN + 0.02, WORLD_MAX - 0.02))
    ax.scatter(
        clipped_x,
        clipped_y,
        s=150,
        color="#f0d43a",
        marker="*",
        edgecolor="#2b2b2b",
        linewidth=1.1,
        label="Upwind source",
        zorder=7,
    )
    ax.annotate(
        "",
        xy=(clipped_x, clipped_y),
        xytext=(float(np.clip(source_x, WORLD_MIN - 0.12, WORLD_MAX + 0.12)), float(np.clip(source_y, WORLD_MIN - 0.12, WORLD_MAX + 0.12))),
        arrowprops=dict(arrowstyle="->", color="#2b2b2b", linewidth=1.2),
        zorder=7,
    )


def render_static(episode_data: dict[str, object], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = build_plume_colormap()
    map_metadata = episode_data["map_metadata"]
    trajectory = np.asarray(episode_data["trajectory"], dtype=np.float32)
    robot_headings = np.asarray(
        episode_data.get(
            "robot_headings",
            np.full(trajectory.shape[0], float(map_metadata.get("robot_heading", 0.0))),
        ),
        dtype=np.float32,
    )
    xs = np.asarray(episode_data["xs"], dtype=np.float32)
    ys = np.asarray(episode_data["ys"], dtype=np.float32)
    concentration_frames = np.asarray(episode_data["concentration_frames"], dtype=np.float32)
    left_points = np.asarray(episode_data["left_points"], dtype=np.float32)
    right_points = np.asarray(episode_data["right_points"], dtype=np.float32)
    sensor_values = np.asarray(episode_data["sensor_values"], dtype=np.float32)
    wind_directions = np.asarray(episode_data["wind_directions"], dtype=np.float32)
    wind_speeds = np.asarray(episode_data["wind_speeds"], dtype=np.float32)
    frame_idx = concentration_frames.shape[0] - 1
    vmax = max(0.05, float(np.percentile(concentration_frames, 99.2)))

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.set_facecolor("#f7f4ef")
    heatmap = ax.imshow(
        concentration_frames[frame_idx],
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap=cmap,
        alpha=0.88,
        interpolation="bilinear",
        aspect="equal",
        vmin=0.0,
        vmax=vmax,
    )
    draw_obstacles(ax, np.asarray(map_metadata["obstacles"], dtype=np.float32))
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        color="#212121",
        linewidth=2.0,
        alpha=0.88,
        zorder=5,
        label="Robot trace",
    )
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=90,
        color="#4ea5ff",
        edgecolor="white",
        linewidth=1.2,
        label="Start",
        zorder=6,
    )
    ax.scatter(
        trajectory[frame_idx, 0],
        trajectory[frame_idx, 1],
        s=90,
        color="#101010",
        edgecolor="white",
        linewidth=1.2,
        label="Robot",
        zorder=6,
    )
    draw_source_marker(ax, map_metadata)
    draw_whisker_artists(ax, trajectory, left_points, right_points, frame_idx)
    draw_robot_heading_marker(
        ax,
        trajectory[frame_idx],
        float(robot_headings[frame_idx]),
    )

    wind_direction_deg = math.degrees(float(wind_directions[frame_idx])) % 360.0
    left_value, right_value = sensor_values[frame_idx]
    info_box = ax.text(
        0.02,
        0.97,
        "\n".join(
            [
                "mode=wide_plume_whisker",
                "frame=%d/%d | termination=%s"
                % (frame_idx, concentration_frames.shape[0] - 1, episode_data["termination"]),
                "wind=%.0fdeg | speed=%.2f"
                % (wind_direction_deg, float(wind_speeds[frame_idx])),
                "left=%.3f | right=%.3f" % (left_value, right_value),
                "cum_reward=%.2f | final_steps=%d"
                % (float(episode_data["reward"]), int(episode_data["steps"])),
            ]
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=(0.15, 0.18, 0.24, 0.82),
            edgecolor="none",
        ),
        zorder=8,
    )

    ax.set_xlim(WORLD_MIN, WORLD_MAX)
    ax.set_ylim(WORLD_MIN, WORLD_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_title(title, fontsize=18, pad=12)
    ax.grid(alpha=0.14, linewidth=0.7)
    ax.legend(loc="lower right", framealpha=0.88)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Gas concentration (a.u.)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_animation(
    episode_data: dict[str, object],
    output_path: Path,
    title: str,
    fps: float | None,
    stride: int,
    realtime_speed: float,
    interpolation_frames: int,
) -> None:
    """按参考样式渲染动画：气味场、源点、机器人、左右触须和信息框。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stride = max(1, int(stride))
    interpolation_frames = max(0, int(interpolation_frames))
    subframes_per_segment = interpolation_frames + 1
    dt = float(episode_data.get("dt", 0.2))
    realtime_speed = max(float(realtime_speed), 1e-6)
    if fps is None:
        effective_frame_dt = dt * stride / subframes_per_segment
        fps = realtime_speed / effective_frame_dt
    fps = max(float(fps), 1e-6)
    cmap = build_plume_colormap()
    map_metadata = episode_data["map_metadata"]
    trajectory = np.asarray(episode_data["trajectory"], dtype=np.float32)
    robot_headings = np.asarray(
        episode_data.get(
            "robot_headings",
            np.full(trajectory.shape[0], float(map_metadata.get("robot_heading", 0.0))),
        ),
        dtype=np.float32,
    )
    left_points = np.asarray(episode_data["left_points"], dtype=np.float32)
    right_points = np.asarray(episode_data["right_points"], dtype=np.float32)
    whisker_length = float(episode_data["whisker_length"])
    left_sectors = np.asarray(episode_data["left_sectors"], dtype=np.int32)
    right_sectors = np.asarray(episode_data["right_sectors"], dtype=np.int32)
    sensor_values = np.asarray(episode_data["sensor_values"], dtype=np.float32)
    xs = np.asarray(episode_data["xs"], dtype=np.float32)
    ys = np.asarray(episode_data["ys"], dtype=np.float32)
    concentration_frames = np.asarray(episode_data["concentration_frames"], dtype=np.float32)
    wind_directions = np.asarray(episode_data["wind_directions"], dtype=np.float32)
    wind_speeds = np.asarray(episode_data["wind_speeds"], dtype=np.float32)
    frame_steps = np.asarray(episode_data["frame_steps"], dtype=np.int32)
    frame_rewards = np.asarray(episode_data["frame_rewards"], dtype=np.float32)
    cumulative_rewards = np.asarray(episode_data["cumulative_rewards"], dtype=np.float32)
    vmax = max(0.05, float(np.percentile(concentration_frames, 99.2)))

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    ax.set_facecolor("#f7f4ef")
    heatmap = ax.imshow(
        concentration_frames[0],
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap=cmap,
        alpha=0.88,
        interpolation="bilinear",
        aspect="equal",
        animated=True,
        vmin=0.0,
        vmax=vmax,
    )
    line, = ax.plot([], [], color="#212121", linewidth=2.0, alpha=0.88, zorder=5)
    start_marker = ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=90,
        color="#4ea5ff",
        edgecolor="white",
        linewidth=1.2,
        label="Start",
        zorder=6,
    )
    robot_marker = ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        s=90,
        color="#101010",
        edgecolor="white",
        linewidth=1.2,
        label="Robot",
        zorder=6,
    )
    draw_obstacles(ax, np.asarray(map_metadata["obstacles"], dtype=np.float32))
    draw_source_marker(ax, map_metadata)
    left_line, right_line, left_tip, right_tip = draw_whisker_artists(
        ax,
        trajectory,
        left_points,
        right_points,
        0,
    )
    heading_shadow, heading_arrow = draw_robot_heading_marker(
        ax,
        trajectory[0],
        float(robot_headings[0]),
    )
    info_box = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.35",
            facecolor=(0.15, 0.18, 0.24, 0.82),
            edgecolor="none",
        ),
        zorder=8,
    )
    ax.set_xlim(WORLD_MIN, WORLD_MAX)
    ax.set_ylim(WORLD_MIN, WORLD_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Grid X")
    ax.set_ylabel("Grid Y")
    ax.set_title(title, fontsize=18, pad=12)
    ax.grid(alpha=0.14, linewidth=0.7)
    ax.legend(loc="lower right", framealpha=0.88)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Gas concentration (a.u.)", fontsize=12)
    total_source_frames = concentration_frames.shape[0]
    frame_indices = np.arange(0, total_source_frames, stride, dtype=np.int32)
    if frame_indices[-1] != total_source_frames - 1:
        frame_indices = np.append(frame_indices, total_source_frames - 1)
    frame_specs: list[tuple[int, int, float]] = []
    for idx in range(len(frame_indices) - 1):
        start_idx = int(frame_indices[idx])
        end_idx = int(frame_indices[idx + 1])
        for sub_idx in range(subframes_per_segment):
            alpha = sub_idx / subframes_per_segment
            frame_specs.append((start_idx, end_idx, float(alpha)))
    final_idx = int(frame_indices[-1])
    frame_specs.append((final_idx, final_idx, 0.0))
    total_frames = len(frame_specs)

    def update(animation_idx: int):
        frame_idx, next_idx, alpha = frame_specs[animation_idx]
        concentration_frame = (
            (1.0 - alpha) * concentration_frames[frame_idx]
            + alpha * concentration_frames[next_idx]
        )
        robot_xy = (1.0 - alpha) * trajectory[frame_idx] + alpha * trajectory[next_idx]
        left_xy = interpolate_whisker_point(
            robot_xy,
            trajectory[frame_idx],
            left_points[frame_idx],
            trajectory[next_idx],
            left_points[next_idx],
            alpha,
            whisker_length,
        )
        right_xy = interpolate_whisker_point(
            robot_xy,
            trajectory[frame_idx],
            right_points[frame_idx],
            trajectory[next_idx],
            right_points[next_idx],
            alpha,
            whisker_length,
        )
        left_value, right_value = (
            (1.0 - alpha) * sensor_values[frame_idx]
            + alpha * sensor_values[next_idx]
        )
        wind_direction = (1.0 - alpha) * wind_directions[frame_idx] + alpha * wind_directions[next_idx]
        wind_speed = (1.0 - alpha) * wind_speeds[frame_idx] + alpha * wind_speeds[next_idx]
        robot_heading = interpolate_angle(
            float(robot_headings[frame_idx]),
            float(robot_headings[next_idx]),
            alpha,
        )
        reward = (1.0 - alpha) * frame_rewards[frame_idx] + alpha * frame_rewards[next_idx]
        cumulative_reward = (
            (1.0 - alpha) * cumulative_rewards[frame_idx]
            + alpha * cumulative_rewards[next_idx]
        )
        sim_step = (1.0 - alpha) * frame_steps[frame_idx] + alpha * frame_steps[next_idx]

        heatmap.set_data(concentration_frame)
        line.set_data(trajectory[: frame_idx + 1, 0], trajectory[: frame_idx + 1, 1])
        robot_marker.set_offsets(robot_xy)
        heading_end = robot_heading_endpoint(robot_xy, robot_heading)
        heading_shadow.set_positions(
            (float(robot_xy[0]), float(robot_xy[1])),
            heading_end,
        )
        heading_arrow.set_positions(
            (float(robot_xy[0]), float(robot_xy[1])),
            heading_end,
        )
        left_line.set_data([robot_xy[0], left_xy[0]], [robot_xy[1], left_xy[1]])
        right_line.set_data([robot_xy[0], right_xy[0]], [robot_xy[1], right_xy[1]])
        left_tip.set_offsets(left_xy)
        right_tip.set_offsets(right_xy)
        wind_direction_deg = math.degrees(float(wind_direction)) % 360.0
        info_box.set_text(
            "\n".join(
                [
                    "mode=wide_plume_whisker",
                    "frame=%d/%d | interp=%d"
                    % (animation_idx, total_frames - 1, interpolation_frames),
                    "wind=%.0fdeg | speed=%.2f"
                    % (wind_direction_deg, float(wind_speed)),
                    "sim_time=%.2fs | playback=%.2fx"
                    % (float(sim_step) * dt, realtime_speed),
                    "step=%d | reward=%.2f"
                    % (int(round(sim_step)), float(reward)),
                    "left_sector=%d | right_sector=%d"
                    % (int(left_sectors[frame_idx]), int(right_sectors[frame_idx])),
                    "left=%.3f | right=%.3f" % (left_value, right_value),
                    "cum_reward=%.2f | final_steps=%d"
                    % (
                        float(cumulative_reward),
                        int(episode_data["steps"]),
                    ),
                ]
            )
        )
        return (
            heatmap,
            line,
            robot_marker,
            heading_shadow,
            heading_arrow,
            info_box,
            start_marker,
            left_line,
            right_line,
            left_tip,
            right_tip,
        )

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=max(1, int(1000 / fps)),
        blit=False,
    )
    save_matplotlib_animation(anim, output_path, fps=fps, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    env = WhiskerOnlyPuffEnv({"domain_randomization": args.domain_randomization})

    model = None
    history_length = 1
    if args.model_path is not None:
        if not args.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {args.model_path}")
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model_path))
        # 从模型堆叠观测维度反推 history 长度，无需手动指定。
        base_dim = int(env.observation_space.shape[0])
        stacked_dim = int(model.observation_space.shape[0])
        if stacked_dim % base_dim != 0:
            raise ValueError(
                f"模型观测维度 {stacked_dim} 不是环境观测维度 {base_dim} 的整数倍，"
                "可能观测格式不匹配（模型是否用当前环境训练的？）。"
            )
        history_length = stacked_dim // base_dim
        print(
            f"loaded_model={args.model_path} base_obs={base_dim} "
            f"stacked_obs={stacked_dim} history_length={history_length} "
            f"deterministic={not args.stochastic}"
        )

    episode_data = capture_rollout(
        env,
        max_steps=args.steps,
        grid_resolution=args.resolution,
        seed=args.seed,
        model=model,
        history_length=history_length,
        deterministic=not args.stochastic,
    )
    render_static(episode_data, args.png_path, args.title)
    render_animation(
        episode_data,
        args.animation_path,
        args.title,
        fps=args.fps,
        stride=args.animation_stride,
        realtime_speed=args.realtime_speed,
        interpolation_frames=args.interpolation_frames,
    )
    print(f"saved_png={args.png_path}")
    print(f"saved_animation={args.animation_path}")


if __name__ == "__main__":
    main()
