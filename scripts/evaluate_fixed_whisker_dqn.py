"""Evaluate a trained fixed-whisker DQN model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from stable_baselines3 import DQN
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import FixedWhiskerPlumeEnv
from dual_whisker_rl.evaluation import evaluate_policy
from dual_whisker_rl.plotting import draw_odor_field
from dual_whisker_rl.plotting import save_sensor_response
from dual_whisker_rl.visualization import save_trajectory_animation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--model-path", type=Path, default=ROOT / "results" / "models" / "fixed_whisker_dqn.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--metrics-path", type=Path, default=ROOT / "results" / "logs" / "fixed_whisker_dqn" / "eval_metrics.json")
    parser.add_argument("--figure-path", type=Path, default=ROOT / "results" / "figures" / "fixed_whisker_dqn_eval_trajectory.png")
    parser.add_argument("--sensor-response-path", type=Path, default=ROOT / "results" / "figures" / "fixed_whisker_dqn_sensor_response.png")
    parser.add_argument("--animation-path", type=Path, default=ROOT / "results" / "figures" / "fixed_whisker_dqn_eval_animation.gif")
    parser.add_argument("--animation-stride", type=int, default=5)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_eval_trajectory(
    env: FixedWhiskerPlumeEnv,
    trajectory: list[dict],
    path: Path,
) -> None:
    xx, yy, zz = env.concentration_grid(resolution=140)
    xs = [row["x"] for row in trajectory]
    ys = [row["y"] for row in trajectory]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    draw_odor_field(ax, xx, yy, zz, alpha=0.9)
    ax.plot(xs, ys, color="white", linewidth=1.5, label="fixed whisker DQN")
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


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    model = DQN.load(args.model_path)

    metrics, trajectories = evaluate_policy(
        model,
        env_factory=lambda: FixedWhiskerPlumeEnv(config),
        episodes=args.episodes,
        seed=args.seed,
        deterministic=True,
    )

    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    if trajectories:
        plot_env = FixedWhiskerPlumeEnv(config)
        plot_env.reset(seed=args.seed)
        save_eval_trajectory(plot_env, trajectories[0], args.figure_path)
        save_sensor_response(trajectories[0], args.sensor_response_path)
        save_trajectory_animation(
            plot_env,
            trajectories[0],
            args.animation_path,
            label="fixed whisker DQN",
            stride=args.animation_stride,
        )
        plot_env.close()

    print(f"loaded_model={args.model_path}")
    print(f"saved_metrics={args.metrics_path}")
    print(f"saved_figure={args.figure_path}")
    print(f"saved_sensor_response={args.sensor_response_path}")
    print(f"saved_animation={args.animation_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
