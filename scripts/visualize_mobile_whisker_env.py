"""移动机器人 + 双触须气源搜索可视化（静态 PNG + GIF）。

复用 `visualize_whisker_only_env` 的羽流配色与触须/机器人绘制助手；本脚本负责
3 元动作 rollout（[移动, 左扇区, 右扇区]）、到达/出界即停、并绘制 goal-radius 圆。
给定 --model-path 时用 PPO 策略驱动（历史堆叠长度从模型反推），否则随机动作。
"""

from __future__ import annotations

import argparse
from collections import deque
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
from matplotlib import patches
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.whisker_only_env import WORLD_MAX
from dual_whisker_rl.envs.whisker_only_env import WORLD_MIN
from visualize_whisker_only_env import build_plume_colormap
from visualize_whisker_only_env import draw_robot_heading_marker
from visualize_whisker_only_env import draw_source_marker
from visualize_whisker_only_env import draw_whisker_artists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "results" / "models" / "mobile_whisker_ppo.zip")
    parser.add_argument("--no-model", action="store_true", help="忽略模型、用随机动作。")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--resolution", type=int, default=100)
    parser.add_argument("--stride", type=int, default=2, help="GIF 每隔多少步取一帧。")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--png-path", type=Path, default=ROOT / "results" / "figures" / "mobile_whisker_env.png")
    parser.add_argument("--gif-path", type=Path, default=ROOT / "results" / "figures" / "mobile_whisker_env.gif")
    parser.add_argument("--title", type=str, default="Mobile Whisker Source Search")
    return parser.parse_args()


def capture_rollout(env, model, history_length, steps, resolution, seed, deterministic):
    """3 元动作 rollout；到达/出界即停。缓存逐帧气体场用于动画。"""
    rng = np.random.default_rng(seed + 123)
    obs, info = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)
    history: deque[np.ndarray] = deque([obs.copy()] * max(1, history_length), maxlen=max(1, history_length))

    xs, ys, grid0 = env.concentration_grid(resolution=resolution)
    sc = env.whiskers.sector_count
    ws0 = env._observe()[1]["whisker_state"]

    traj = [(env.robot_state.x, env.robot_state.y)]
    headings = [float(env.robot_state.heading)]
    left_pts = [ws0.left_point]
    right_pts = [ws0.right_point]
    sensors = [(0.0, 0.0)]
    frames = [grid0.copy()]
    dists = [float(info["distance_to_source"])]
    total_reward = 0.0
    termination = "timeout"

    for _ in range(steps):
        if model is None:
            action = [int(rng.integers(6)), int(rng.integers(sc)), int(rng.integers(sc))]
        else:
            stacked = np.concatenate(tuple(history)).astype(np.float32)
            a, _ = model.predict(stacked, deterministic=deterministic)
            action = [int(a[0]), int(a[1]), int(a[2])]
        obs, reward, terminated, truncated, info = env.step(action)
        history.append(np.asarray(obs, dtype=np.float32))
        total_reward += float(reward)
        ws = env._observe()[1]["whisker_state"]
        traj.append((env.robot_state.x, env.robot_state.y))
        headings.append(float(env.robot_state.heading))
        left_pts.append(ws.left_point)
        right_pts.append(ws.right_point)
        sensors.append((float(info["left"]), float(info["right"])))
        _, _, g = env.concentration_grid(resolution=resolution)
        frames.append(g.copy())
        dists.append(float(info["distance_to_source"]))
        if terminated or truncated:
            reached = info["distance_to_source"] <= env.goal_radius
            termination = "reached_goal" if reached else ("out_of_bounds" if info.get("out_of_bounds") else "timeout")
            break

    return {
        "xs": xs,
        "ys": ys,
        "frames": np.asarray(frames, dtype=np.float32),
        "trajectory": np.asarray(traj, dtype=np.float32),
        "headings": np.asarray(headings, dtype=np.float32),
        "left_points": np.asarray(left_pts, dtype=np.float32),
        "right_points": np.asarray(right_pts, dtype=np.float32),
        "sensor_values": np.asarray(sensors, dtype=np.float32),
        "distances": np.asarray(dists, dtype=np.float32),
        "reward": total_reward,
        "termination": termination,
        "source": (env.plume.source_x, env.plume.source_y),
        "goal_radius": float(env.goal_radius),
        "map_metadata": env.get_map_metadata(),
    }


def _draw_goal(ax, source, goal_radius):
    ax.add_patch(
        patches.Circle(
            (float(source[0]), float(source[1])),
            goal_radius,
            fill=False,
            edgecolor="#e45756",
            linewidth=1.4,
            linestyle="--",
            zorder=7,
            label="Goal radius",
        )
    )


def _setup_axes(ax, data, cmap, frame_idx, title):
    xs, ys, frames = data["xs"], data["ys"], data["frames"]
    vmax = max(0.05, float(np.percentile(frames, 99.2)))
    ax.set_facecolor("#f7f4ef")
    heatmap = ax.imshow(
        frames[frame_idx],
        extent=[xs.min(), xs.max(), ys.min(), ys.max()],
        origin="lower",
        cmap=cmap,
        alpha=0.9,
        interpolation="bilinear",
        aspect="equal",
        vmin=0.0,
        vmax=vmax,
        animated=True,
    )
    draw_source_marker(ax, data["map_metadata"])
    _draw_goal(ax, data["source"], data["goal_radius"])
    ax.set_xlim(WORLD_MIN, WORLD_MAX)
    ax.set_ylim(WORLD_MIN, WORLD_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=15, pad=10)
    return heatmap, vmax


def render_static(data, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = build_plume_colormap()
    n = data["trajectory"].shape[0]
    idx = n - 1
    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    heatmap, _ = _setup_axes(ax, data, cmap, idx, title)
    traj = data["trajectory"]
    ax.plot(traj[:, 0], traj[:, 1], color="#54524e", lw=1.8, alpha=0.85, zorder=5, label="Robot trace")
    ax.scatter(traj[0, 0], traj[0, 1], s=85, facecolor="white", edgecolor="#7a756c", linewidth=1.4, zorder=6, label="Start")
    ax.scatter(traj[idx, 0], traj[idx, 1], s=95, color="#2b2b2b", edgecolor="white", linewidth=1.3, zorder=6, label="Robot")
    draw_whisker_artists(ax, traj, data["left_points"], data["right_points"], idx)
    draw_robot_heading_marker(ax, traj[idx], float(data["headings"][idx]))
    left_v, right_v = data["sensor_values"][idx]
    ax.text(
        0.02, 0.98,
        "\n".join([
            f"termination = {data['termination']}",
            f"steps = {idx}   dist = {data['distances'][idx]:.3f} m",
            f"sensor  L {left_v:.3f}   R {right_v:.3f}",
            f"cum_reward = {data['reward']:.2f}",
        ]),
        transform=ax.transAxes, ha="left", va="top", fontsize=9.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.72, edgecolor="#d8d2c7"),
        zorder=8,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.colorbar(heatmap, ax=ax, fraction=0.043, pad=0.03).set_label("Gas concentration (a.u.)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_animation(data, path, title, stride, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = build_plume_colormap()
    traj = data["trajectory"]
    frames = data["frames"]
    n = traj.shape[0]
    frame_ids = list(range(0, n, max(1, stride)))
    if frame_ids[-1] != n - 1:
        frame_ids.append(n - 1)

    fig, ax = plt.subplots(figsize=(9.5, 8.5))
    heatmap, _ = _setup_axes(ax, data, cmap, 0, title)
    line, = ax.plot([], [], color="#54524e", lw=1.8, alpha=0.85, zorder=5, label="Robot trace")
    robot = ax.scatter([traj[0, 0]], [traj[0, 1]], s=95, color="#2b2b2b", edgecolor="white", linewidth=1.3, zorder=6, label="Robot")
    left_line, right_line, left_tip, right_tip = draw_whisker_artists(ax, traj, data["left_points"], data["right_points"], 0)
    hs, ha = draw_robot_heading_marker(ax, traj[0], float(data["headings"][0]))
    txt = ax.text(0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
                  family="monospace", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.72, edgecolor="#d8d2c7"), zorder=8)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    fig.colorbar(heatmap, ax=ax, fraction=0.043, pad=0.03).set_label("Gas concentration (a.u.)")

    def update(fi):
        heatmap.set_data(frames[fi])
        line.set_data(traj[: fi + 1, 0], traj[: fi + 1, 1])
        robot.set_offsets(traj[fi])
        end = (traj[fi, 0] + 0.07 * math.cos(data["headings"][fi]), traj[fi, 1] + 0.07 * math.sin(data["headings"][fi]))
        hs.set_positions((float(traj[fi, 0]), float(traj[fi, 1])), end)
        ha.set_positions((float(traj[fi, 0]), float(traj[fi, 1])), end)
        lp, rp = data["left_points"][fi], data["right_points"][fi]
        left_line.set_data([traj[fi, 0], lp[0]], [traj[fi, 1], lp[1]])
        right_line.set_data([traj[fi, 0], rp[0]], [traj[fi, 1], rp[1]])
        left_tip.set_offsets(lp)
        right_tip.set_offsets(rp)
        lv, rv = data["sensor_values"][fi]
        txt.set_text(f"step {fi}   dist {data['distances'][fi]:.3f}\nsensor L {lv:.3f} R {rv:.3f}")
        return heatmap, line, robot, hs, ha, left_line, right_line, left_tip, right_tip, txt

    anim = animation.FuncAnimation(fig, update, frames=frame_ids, interval=int(1000 / fps), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = {"domain_randomization": args.domain_randomization} if args.domain_randomization else {}
    env = MobileWhiskerPuffEnv(config)

    model = None
    history_length = 1
    if not args.no_model and args.model_path.exists():
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model_path))
        base = int(env.observation_space.shape[0])
        history_length = int(model.observation_space.shape[0]) // base
        print(f"loaded_model={args.model_path} history_length={history_length}")
    else:
        print("no model -> random actions")

    data = capture_rollout(env, model, history_length, args.steps, args.resolution, args.seed, not args.stochastic)
    render_static(data, args.png_path, args.title)
    render_animation(data, args.gif_path, args.title, args.stride, fps=6)
    print(f"termination={data['termination']} steps={data['trajectory'].shape[0]-1} final_dist={data['distances'][-1]:.3f}")
    print(f"saved_png={args.png_path}")
    print(f"saved_gif={args.gif_path}")


if __name__ == "__main__":
    main()
