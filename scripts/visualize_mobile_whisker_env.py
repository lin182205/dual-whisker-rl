"""移动机器人 + 双触须气源搜索可视化（静态 PNG + GIF）。

复用 `visualize_whisker_only_env` 的羽流配色与触须/机器人绘制助手；本脚本负责
3 元动作 rollout（[移动, 左扇区, 右扇区]）、到达/出界/碰撞即停，并在信息框显示左右
传感器浓度、L-R 差值及 mean|L-R|。给定 --model-path 时用 PPO 策略驱动（历史堆叠
长度从模型反推），否则随机动作。
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
from matplotlib.patches import Rectangle
from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.paths import resolve_path_args
from train_whisker_only_ppo import load_config
from visualize_whisker_only_env import build_plume_colormap
from visualize_whisker_only_env import draw_robot_heading_marker
from visualize_whisker_only_env import draw_whisker_artists


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/models/mobile_whisker_transformer_ppo.zip"),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--scenario-mode",
        choices=("randomized", "fixed"),
        default=None,
        help="覆盖配置中的场景初始化模式；默认使用 randomized。",
    )
    parser.add_argument("--no-model", action="store_true", help="忽略模型、用随机动作。")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument(
        "--resolution",
        type=int,
        default=200,
        help="气味热图分辨率；与物理 LBM 风场网格相互独立。",
    )
    parser.add_argument("--stride", type=int, default=1, help="GIF 每隔多少步取一帧。")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--png-path", type=Path, default=Path("results/figures/mobile_whisker_env.png"))
    parser.add_argument("--gif-path", type=Path, default=Path("results/figures/mobile_whisker_env.gif"))
    parser.add_argument("--title", type=str, default="Mobile Whisker Source Search")
    return resolve_path_args(
        parser.parse_args(),
        "model_path",
        "config",
        "png_path",
        "gif_path",
    )


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
    lidar_ranges = [np.asarray(info["lidar_ranges_m"], dtype=np.float32)]
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
        lidar_ranges.append(np.asarray(info["lidar_ranges_m"], dtype=np.float32))
        _, _, g = env.concentration_grid(resolution=resolution)
        frames.append(g.copy())
        dists.append(float(info["distance_to_source"]))
        if terminated or truncated:
            termination = str(info.get("termination_reason") or "timeout")
            break

    sensor_values = np.asarray(sensors, dtype=np.float32)
    sensor_differences = sensor_values[:, 0] - sensor_values[:, 1]
    return {
        "xs": xs,
        "ys": ys,
        "frames": np.asarray(frames, dtype=np.float32),
        "trajectory": np.asarray(traj, dtype=np.float32),
        "headings": np.asarray(headings, dtype=np.float32),
        "left_points": np.asarray(left_pts, dtype=np.float32),
        "right_points": np.asarray(right_pts, dtype=np.float32),
        "sensor_values": sensor_values,
        "sensor_differences": sensor_differences,
        "lidar_ranges_m": np.asarray(lidar_ranges, dtype=np.float32),
        "distances": np.asarray(dists, dtype=np.float32),
        "reward": total_reward,
        "termination": termination,
        "source": (env.plume.source_x, env.plume.source_y),
        "goal_radius": float(env.goal_radius),
        "map_metadata": env.get_map_metadata(),
    }


def summarize_sensor_differences(differences):
    """汇总真实 step 的 L-R；跳过仅用于绘图的 reset 初始帧。"""
    values = np.asarray(differences, dtype=np.float32).reshape(-1)
    if values.size > 1:
        values = values[1:]
    if values.size == 0:
        return 0.0, 0.0
    return float(np.mean(values)), float(np.mean(np.abs(values)))


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
        interpolation="bicubic",
        aspect="equal",
        vmin=0.0,
        vmax=vmax,
        animated=True,
    )
    # 用星标记目标气源位置。
    sx, sy = data["source"]
    ax.scatter(
        [float(sx)], [float(sy)],
        s=200, marker="*", color="#f2c53d", edgecolor="#2b2b2b",
        linewidth=1.0, zorder=7, label="Source",
    )
    obstacles = np.asarray(data["map_metadata"]["obstacles"], dtype=np.float32)
    for obstacle_index, (xmin, xmax, ymin, ymax) in enumerate(obstacles):
        ax.add_patch(
            Rectangle(
                (float(xmin), float(ymin)),
                float(xmax - xmin),
                float(ymax - ymin),
                facecolor="#656565",
                edgecolor="#252525",
                linewidth=1.2,
                alpha=0.82,
                zorder=4,
                label="Obstacle" if obstacle_index == 0 else None,
            )
        )
    wmin = float(data["map_metadata"]["world_min"])
    wmax = float(data["map_metadata"]["world_max"])
    ax.set_xlim(wmin, wmax)
    ax.set_ylim(wmin, wmax)
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
    difference = float(data["sensor_differences"][idx])
    mean_difference, mean_abs_difference = summarize_sensor_differences(
        data["sensor_differences"]
    )
    ax.text(
        0.02, 0.98,
        "\n".join([
            f"termination = {data['termination']}",
            f"steps = {idx}   dist = {data['distances'][idx]:.3f} m",
            f"min lidar = {float(np.min(data['lidar_ranges_m'][idx])):.3f} m",
            f"sensor  L {left_v:.3f}   R {right_v:.3f}",
            f"delta(L-R) = {difference:+.3f}",
            f"mean delta = {mean_difference:+.3f}   mean|delta| = {mean_abs_difference:.3f}",
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
        difference = float(data["sensor_differences"][fi])
        _, running_mean_abs_difference = summarize_sensor_differences(
            data["sensor_differences"][: fi + 1]
        )
        txt.set_text(
            f"step {fi}   dist {data['distances'][fi]:.3f}\n"
            f"min lidar {float(np.min(data['lidar_ranges_m'][fi])):.3f} m\n"
            f"sensor L {lv:.3f} R {rv:.3f}\n"
            f"delta(L-R) {difference:+.3f}   "
            f"mean|delta| {running_mean_abs_difference:.3f}"
        )
        return heatmap, line, robot, hs, ha, left_line, right_line, left_tip, right_tip, txt

    anim = animation.FuncAnimation(fig, update, frames=frame_ids, interval=int(1000 / fps), blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=120)
    plt.close(fig)


def render_static_rollouts(rollouts, path, title):
    """在同一 PNG 中绘制一个或两个评估 episode。"""
    if not 1 <= len(rollouts) <= 2:
        raise ValueError("render_static_rollouts expects one or two rollouts")
    if len(rollouts) == 1:
        label, data = rollouts[0]
        render_static(data, path, f"{title}\n{label}")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = build_plume_colormap()
    fig, axes = plt.subplots(1, len(rollouts), figsize=(9.0 * len(rollouts), 8.2))
    axes = np.atleast_1d(axes)
    for ax, (label, data) in zip(axes, rollouts):
        idx = data["trajectory"].shape[0] - 1
        heatmap, _ = _setup_axes(ax, data, cmap, idx, label)
        traj = data["trajectory"]
        ax.plot(
            traj[:, 0],
            traj[:, 1],
            color="#54524e",
            lw=1.8,
            alpha=0.85,
            zorder=5,
            label="Robot trace",
        )
        ax.scatter(
            traj[0, 0],
            traj[0, 1],
            s=85,
            facecolor="white",
            edgecolor="#7a756c",
            linewidth=1.4,
            zorder=6,
            label="Start",
        )
        ax.scatter(
            traj[idx, 0],
            traj[idx, 1],
            s=95,
            color="#2b2b2b",
            edgecolor="white",
            linewidth=1.3,
            zorder=6,
            label="Robot",
        )
        draw_whisker_artists(
            ax,
            traj,
            data["left_points"],
            data["right_points"],
            idx,
        )
        draw_robot_heading_marker(ax, traj[idx], float(data["headings"][idx]))
        left_v, right_v = data["sensor_values"][idx]
        difference = float(data["sensor_differences"][idx])
        mean_difference, mean_abs_difference = summarize_sensor_differences(
            data["sensor_differences"]
        )
        ax.text(
            0.02,
            0.98,
            "\n".join(
                [
                    f"termination = {data['termination']}",
                    f"steps = {idx}   dist = {data['distances'][idx]:.3f} m",
                    f"min lidar = {float(np.min(data['lidar_ranges_m'][idx])):.3f} m",
                    f"sensor  L {left_v:.3f}   R {right_v:.3f}",
                    f"delta(L-R) = {difference:+.3f}",
                    f"mean delta = {mean_difference:+.3f}",
                    f"mean|delta| = {mean_abs_difference:.3f}",
                    f"cum_reward = {data['reward']:.2f}",
                ]
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.72,
                edgecolor="#d8d2c7",
            ),
            zorder=8,
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        fig.colorbar(heatmap, ax=ax, fraction=0.043, pad=0.03).set_label(
            "Gas concentration (a.u.)"
        )
    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(path, dpi=170)
    plt.close(fig)


def render_animation_rollouts(rollouts, path, title, stride, fps):
    """在同一 GIF 中并排播放一个或两个评估 episode。"""
    if not 1 <= len(rollouts) <= 2:
        raise ValueError("render_animation_rollouts expects one or two rollouts")
    if len(rollouts) == 1:
        label, data = rollouts[0]
        render_animation(data, path, f"{title}\n{label}", stride, fps)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    cmap = build_plume_colormap()
    fig, axes = plt.subplots(1, len(rollouts), figsize=(9.0 * len(rollouts), 8.2))
    axes = np.atleast_1d(axes)
    artists = []
    frame_sequences = []

    for ax, (label, data) in zip(axes, rollouts):
        traj = data["trajectory"]
        n = traj.shape[0]
        frame_ids = list(range(0, n, max(1, stride)))
        if frame_ids[-1] != n - 1:
            frame_ids.append(n - 1)
        frame_sequences.append(frame_ids)

        heatmap, _ = _setup_axes(ax, data, cmap, 0, label)
        (line,) = ax.plot(
            [], [], color="#54524e", lw=1.8, alpha=0.85, zorder=5,
            label="Robot trace",
        )
        robot = ax.scatter(
            [traj[0, 0]],
            [traj[0, 1]],
            s=95,
            color="#2b2b2b",
            edgecolor="white",
            linewidth=1.3,
            zorder=6,
            label="Robot",
        )
        left_line, right_line, left_tip, right_tip = draw_whisker_artists(
            ax, traj, data["left_points"], data["right_points"], 0
        )
        hs, ha = draw_robot_heading_marker(
            ax, traj[0], float(data["headings"][0])
        )
        txt = ax.text(
            0.02,
            0.98,
            "",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.5,
            family="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="white",
                alpha=0.72,
                edgecolor="#d8d2c7",
            ),
            zorder=8,
        )
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
        fig.colorbar(heatmap, ax=ax, fraction=0.043, pad=0.03).set_label(
            "Gas concentration (a.u.)"
        )
        artists.append(
            {
                "data": data,
                "heatmap": heatmap,
                "line": line,
                "robot": robot,
                "heading_shadow": hs,
                "heading_arrow": ha,
                "left_line": left_line,
                "right_line": right_line,
                "left_tip": left_tip,
                "right_tip": right_tip,
                "text": txt,
            }
        )

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    def update(frame_position):
        updated = []
        for state, frame_ids in zip(artists, frame_sequences):
            data = state["data"]
            traj = data["trajectory"]
            fi = frame_ids[min(frame_position, len(frame_ids) - 1)]
            state["heatmap"].set_data(data["frames"][fi])
            state["line"].set_data(traj[: fi + 1, 0], traj[: fi + 1, 1])
            state["robot"].set_offsets([traj[fi]])
            end = (
                traj[fi, 0] + 0.07 * math.cos(data["headings"][fi]),
                traj[fi, 1] + 0.07 * math.sin(data["headings"][fi]),
            )
            start = (float(traj[fi, 0]), float(traj[fi, 1]))
            state["heading_shadow"].set_positions(start, end)
            state["heading_arrow"].set_positions(start, end)
            left_point = data["left_points"][fi]
            right_point = data["right_points"][fi]
            state["left_line"].set_data(
                [traj[fi, 0], left_point[0]], [traj[fi, 1], left_point[1]]
            )
            state["right_line"].set_data(
                [traj[fi, 0], right_point[0]], [traj[fi, 1], right_point[1]]
            )
            state["left_tip"].set_offsets([left_point])
            state["right_tip"].set_offsets([right_point])
            left_value, right_value = data["sensor_values"][fi]
            difference = float(data["sensor_differences"][fi])
            _, mean_abs_difference = summarize_sensor_differences(
                data["sensor_differences"][: fi + 1]
            )
            state["text"].set_text(
                f"step {fi}   dist {data['distances'][fi]:.3f}\n"
                f"min lidar {float(np.min(data['lidar_ranges_m'][fi])):.3f} m\n"
                f"sensor L {left_value:.3f} R {right_value:.3f}\n"
                f"delta(L-R) {difference:+.3f}\n"
                f"mean|delta| {mean_abs_difference:.3f}"
            )
            updated.extend(
                [
                    state["heatmap"],
                    state["line"],
                    state["robot"],
                    state["heading_shadow"],
                    state["heading_arrow"],
                    state["left_line"],
                    state["right_line"],
                    state["left_tip"],
                    state["right_tip"],
                    state["text"],
                ]
            )
        return updated

    frame_count = max(len(frame_ids) for frame_ids in frame_sequences)
    anim = animation.FuncAnimation(
        fig,
        update,
        frames=range(frame_count),
        interval=int(1000 / fps),
        blit=False,
    )
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.scenario_mode is not None:
        config["scenario_mode"] = args.scenario_mode
    if args.domain_randomization:
        config["domain_randomization"] = True
    env = MobileWhiskerPuffEnv(config)

    model = None
    history_length = 1
    if not args.no_model and args.model_path.exists():
        from stable_baselines3 import PPO
        from dual_whisker_rl.agents import TransformerHistoryExtractor

        _ = TransformerHistoryExtractor
        model = PPO.load(str(args.model_path))
        base = int(env.observation_space.shape[0])
        stacked = int(model.observation_space.shape[0])
        if stacked % base != 0:
            raise ValueError(
                f"model observation dim {stacked} is not divisible by environment "
                f"base dim {base}; check include_blank_age_observation and "
                "include_lidar_observation in --config"
            )
        history_length = stacked // base
        print(f"loaded_model={args.model_path} history_length={history_length}")
    else:
        print("no model -> random actions")

    data = capture_rollout(env, model, history_length, args.steps, args.resolution, args.seed, not args.stochastic)
    render_static(data, args.png_path, args.title)
    render_animation(data, args.gif_path, args.title, args.stride, fps=6)
    mean_difference, mean_abs_difference = summarize_sensor_differences(
        data["sensor_differences"]
    )
    print(f"termination={data['termination']} steps={data['trajectory'].shape[0]-1} final_dist={data['distances'][-1]:.3f}")
    print(
        "mean_left_right_difference="
        f"{mean_difference:+.6f}"
    )
    print(
        "mean_abs_left_right_difference="
        f"{mean_abs_difference:.6f}"
    )
    print(f"saved_png={args.png_path}")
    print(f"saved_gif={args.gif_path}")


if __name__ == "__main__":
    main()
