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
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs.whisker_only_env import WhiskerOnlyPuffEnv
from dual_whisker_rl.envs.whisker_only_env import WORLD_MAX
from dual_whisker_rl.envs.whisker_only_env import WORLD_MIN


def build_plume_colormap() -> colors.LinearSegmentedColormap:
    """参考 TD3 示例的气味场配色：低浓度浅色，高浓度黄色。"""
    return colors.LinearSegmentedColormap.from_list(
        "puff_demo",
        [
            "#f7f4ef",
            "#d8e2f1",
            "#8ab5ef",
            "#4f8edf",
            "#2f68b2",
            "#27447d",
            "#ffe27a",
        ],
        N=256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a SimpleGasEnv-style puff animation for the whisker-only environment."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--resolution", type=int, default=120)
    parser.add_argument("--png-path", type=Path, default=ROOT / "results" / "figures" / "whisker_only_env.png")
    parser.add_argument("--gif-path", type=Path, default=ROOT / "results" / "figures" / "whisker_only_env.gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--title", type=str, default="Whisker-only Puff Sampling Demo")
    return parser.parse_args()


def capture_rollout(
    env: WhiskerOnlyPuffEnv,
    max_steps: int,
    grid_resolution: int,
    seed: int,
) -> dict[str, object]:
    """随机触须动作采样一段 rollout，并缓存每一帧气体场用于动画。"""
    rng = np.random.default_rng(seed + 123)
    env.reset(seed=seed)
    map_metadata = env.get_map_metadata()
    xs, ys, initial_grid = env.concentration_grid(resolution=grid_resolution)

    robot_state = map_metadata["robot_position"]
    trajectory = [robot_state]
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
        action = [
            int(rng.integers(0, env.whiskers.sector_count)),
            int(rng.integers(0, env.whiskers.sector_count)),
        ]
        _, reward, _, _, info = env.step(action)
        total_reward += reward
        obs_info = env._observe()[1]
        whisker_state = obs_info["whisker_state"]
        trajectory.append((env.robot_state.x, env.robot_state.y))
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
        "left_points": np.asarray(left_points, dtype=np.float32),
        "right_points": np.asarray(right_points, dtype=np.float32),
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


def render_static(episode_data: dict[str, object], output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = build_plume_colormap()
    map_metadata = episode_data["map_metadata"]
    trajectory = np.asarray(episode_data["trajectory"], dtype=np.float32)
    xs = np.asarray(episode_data["xs"], dtype=np.float32)
    ys = np.asarray(episode_data["ys"], dtype=np.float32)
    concentration_frames = np.asarray(episode_data["concentration_frames"], dtype=np.float32)
    left_points = np.asarray(episode_data["left_points"], dtype=np.float32)
    right_points = np.asarray(episode_data["right_points"], dtype=np.float32)
    sensor_values = np.asarray(episode_data["sensor_values"], dtype=np.float32)
    wind_directions = np.asarray(episode_data["wind_directions"], dtype=np.float32)
    wind_speeds = np.asarray(episode_data["wind_speeds"], dtype=np.float32)
    frame_idx = concentration_frames.shape[0] - 1

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    heatmap = ax.imshow(
        concentration_frames[frame_idx],
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap=cmap,
        alpha=0.95,
        interpolation="bilinear",
        aspect="equal",
    )
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
    ax.scatter(
        map_metadata["source_position"][0],
        map_metadata["source_position"][1],
        s=150,
        color="#f0d43a",
        marker="*",
        edgecolor="#2b2b2b",
        linewidth=1.1,
        label="Source",
        zorder=7,
    )
    draw_whisker_artists(ax, trajectory, left_points, right_points, frame_idx)

    wind_direction_deg = math.degrees(float(wind_directions[frame_idx])) % 360.0
    left_value, right_value = sensor_values[frame_idx]
    info_box = ax.text(
        0.02,
        0.97,
        "\n".join(
            [
                "mode=whisker_only_puff",
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
    ax.grid(alpha=0.08, linewidth=0.7)
    ax.legend(loc="lower right", framealpha=0.88)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Gas concentration (a.u.)", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_animation(episode_data: dict[str, object], output_path: Path, title: str, fps: int) -> None:
    """按参考动画样式渲染 GIF：气味场、源点、机器人、左右触须和信息框。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmap = build_plume_colormap()
    map_metadata = episode_data["map_metadata"]
    trajectory = np.asarray(episode_data["trajectory"], dtype=np.float32)
    left_points = np.asarray(episode_data["left_points"], dtype=np.float32)
    right_points = np.asarray(episode_data["right_points"], dtype=np.float32)
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

    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    heatmap = ax.imshow(
        concentration_frames[0],
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap=cmap,
        alpha=0.95,
        interpolation="bilinear",
        aspect="equal",
        animated=True,
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
    ax.scatter(
        map_metadata["source_position"][0],
        map_metadata["source_position"][1],
        s=150,
        color="#f0d43a",
        marker="*",
        edgecolor="#2b2b2b",
        linewidth=1.1,
        label="Source",
        zorder=7,
    )
    left_line, right_line, left_tip, right_tip = draw_whisker_artists(
        ax,
        trajectory,
        left_points,
        right_points,
        0,
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
    ax.grid(alpha=0.08, linewidth=0.7)
    ax.legend(loc="lower right", framealpha=0.88)
    colorbar = fig.colorbar(heatmap, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Gas concentration (a.u.)", fontsize=12)
    total_frames = concentration_frames.shape[0]

    def update(frame_idx: int):
        heatmap.set_data(concentration_frames[frame_idx])
        line.set_data(trajectory[: frame_idx + 1, 0], trajectory[: frame_idx + 1, 1])
        robot_marker.set_offsets(trajectory[frame_idx])
        robot_xy = trajectory[frame_idx]
        left_xy = left_points[frame_idx]
        right_xy = right_points[frame_idx]
        left_line.set_data([robot_xy[0], left_xy[0]], [robot_xy[1], left_xy[1]])
        right_line.set_data([robot_xy[0], right_xy[0]], [robot_xy[1], right_xy[1]])
        left_tip.set_offsets(left_xy)
        right_tip.set_offsets(right_xy)
        wind_direction_deg = math.degrees(float(wind_directions[frame_idx])) % 360.0
        left_value, right_value = sensor_values[frame_idx]
        info_box.set_text(
            "\n".join(
                [
                    "mode=whisker_only_puff",
                    "frame=%d/%d | termination=%s"
                    % (frame_idx, total_frames - 1, episode_data["termination"]),
                    "wind=%.0fdeg | speed=%.2f"
                    % (wind_direction_deg, float(wind_speeds[frame_idx])),
                    "step=%d | reward=%.2f"
                    % (int(frame_steps[frame_idx]), float(frame_rewards[frame_idx])),
                    "left_sector=%d | right_sector=%d"
                    % (int(left_sectors[frame_idx]), int(right_sectors[frame_idx])),
                    "left=%.3f | right=%.3f" % (left_value, right_value),
                    "cum_reward=%.2f | final_steps=%d"
                    % (
                        float(cumulative_rewards[frame_idx]),
                        int(episode_data["steps"]),
                    ),
                ]
            )
        )
        return heatmap, line, robot_marker, info_box, start_marker, left_line, right_line, left_tip, right_tip

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=total_frames,
        interval=max(1, int(1000 / fps)),
        blit=False,
    )
    anim.save(output_path, writer=animation.PillowWriter(fps=fps), dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    env = WhiskerOnlyPuffEnv()
    episode_data = capture_rollout(
        env,
        max_steps=args.steps,
        grid_resolution=args.resolution,
        seed=args.seed,
    )
    render_static(episode_data, args.png_path, args.title)
    render_animation(episode_data, args.gif_path, args.title, fps=args.fps)
    print(f"saved_png={args.png_path}")
    print(f"saved_gif={args.gif_path}")


if __name__ == "__main__":
    main()
