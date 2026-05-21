"""Run a random-policy rollout and save first validation figures."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import PlumeEnv


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config(ROOT / "configs" / "default.yaml")
    env = PlumeEnv(config)
    obs, info = env.reset(seed=7)

    terminated = False
    truncated = False
    total_reward = 0.0
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    out_dir = ROOT / "results" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_plume(env, out_dir / "plume_field.png")
    save_trajectory(env, out_dir / "random_trajectory.png")
    save_sensor_response(env, out_dir / "sensor_response.png")
    print(
        f"steps={len(env.trajectory)} total_reward={total_reward:.3f} "
        f"distance={info['distance_to_source']:.3f}"
    )


def save_plume(env: PlumeEnv, path: Path) -> None:
    xx, yy, zz = env.concentration_grid(resolution=140)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=40, cmap="viridis")
    plt.colorbar(label="odor concentration")
    sx, sy = env.plume.params.source
    plt.scatter([sx], [sy], marker="*", s=180, c="red", label="source")
    plt.xlim(0, env.width)
    plt.ylim(0, env.height)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_trajectory(env: PlumeEnv, path: Path) -> None:
    xs = [p["x"] for p in env.trajectory]
    ys = [p["y"] for p in env.trajectory]
    xx, yy, zz = env.concentration_grid(resolution=140)
    plt.figure(figsize=(6, 5))
    plt.contourf(xx, yy, zz, levels=40, cmap="viridis", alpha=0.85)
    plt.plot(xs, ys, color="white", linewidth=1.5, label="random policy")
    if xs and ys:
        plt.scatter([xs[0]], [ys[0]], c="cyan", s=50, label="start")
        plt.scatter([xs[-1]], [ys[-1]], c="black", s=50, label="end")
    sx, sy = env.plume.params.source
    plt.scatter([sx], [sy], marker="*", s=180, c="red", label="source")
    plt.xlim(0, env.width)
    plt.ylim(0, env.height)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_sensor_response(env: PlumeEnv, path: Path) -> None:
    left = np.array([p["left"] for p in env.trajectory])
    right = np.array([p["right"] for p in env.trajectory])
    raw_left = np.array([p["raw_left"] for p in env.trajectory])
    raw_right = np.array([p["raw_right"] for p in env.trajectory])
    steps = np.arange(len(left))

    plt.figure(figsize=(8, 4))
    plt.plot(steps, raw_left, "--", color="#4c78a8", alpha=0.5, label="left raw")
    plt.plot(steps, raw_right, "--", color="#f58518", alpha=0.5, label="right raw")
    plt.plot(steps, left, color="#4c78a8", label="left sensor")
    plt.plot(steps, right, color="#f58518", label="right sensor")
    plt.xlabel("step")
    plt.ylabel("concentration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


if __name__ == "__main__":
    main()
