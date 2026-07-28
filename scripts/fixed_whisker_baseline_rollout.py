"""Run the fixed-whisker baseline environment with random movement actions."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import FixedWhiskerPlumeEnv
from dual_whisker_rl.paths import project_path
from dual_whisker_rl.plotting import draw_odor_field


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config_path = project_path("configs/default.yaml")
    out_dir = project_path("results/figures")
    assert config_path is not None and out_dir is not None
    config = load_config(config_path)
    env = FixedWhiskerPlumeEnv(config)
    obs, info = env.reset(seed=11)

    terminated = False
    truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    out_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory(env, out_dir / "fixed_whisker_random_trajectory.png")
    save_sensor_response(env, out_dir / "fixed_whisker_sensor_response.png")
    save_whisker_angles(env, out_dir / "fixed_whisker_angles.png")
    print(
        f"baseline=fixed_whisker_random steps={len(env.trajectory)} "
        f"total_reward={total_reward:.3f} distance={info['distance_to_source']:.3f}"
    )


def save_trajectory(env: FixedWhiskerPlumeEnv, path: Path) -> None:
    xs = [p["x"] for p in env.trajectory]
    ys = [p["y"] for p in env.trajectory]
    xx, yy, zz = env.concentration_grid(resolution=140)
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_odor_field(ax, xx, yy, zz, alpha=0.9)
    ax.plot(xs, ys, color="white", linewidth=1.5, label="fixed whisker baseline")
    if xs and ys:
        ax.scatter([xs[0]], [ys[0]], c="cyan", s=50, label="start")
        ax.scatter([xs[-1]], [ys[-1]], c="black", s=50, label="end")
    sx, sy = env.plume.params.source
    ax.scatter([sx], [sy], marker="*", s=180, c="red", label="source")
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sensor_response(env: FixedWhiskerPlumeEnv, path: Path) -> None:
    left = np.array([p["left"] for p in env.trajectory])
    right = np.array([p["right"] for p in env.trajectory])
    steps = np.arange(len(left))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, left, color="#4c78a8", label="left sensor")
    plt.plot(steps, right, color="#f58518", label="right sensor")
    plt.xlabel("step")
    plt.ylabel("concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_whisker_angles(env: FixedWhiskerPlumeEnv, path: Path) -> None:
    left = np.rad2deg([p["left_angle"] for p in env.trajectory])
    right = np.rad2deg([p["right_angle"] for p in env.trajectory])
    steps = np.arange(len(left))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, left, color="#4c78a8", label="left angle")
    plt.plot(steps, right, color="#f58518", label="right angle")
    plt.xlabel("step")
    plt.ylabel("angle (deg)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
