"""Deprecated: DQN is not compatible with the current MultiDiscrete joint action space."""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--model-path", type=Path, default=Path("results/models/joint_dqn.zip"))
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--metrics-path", type=Path, default=Path("results/logs/joint_dqn/eval_metrics.json"))
    parser.add_argument("--figure-path", type=Path, default=Path("results/figures/joint_dqn_eval_trajectory.png"))
    parser.add_argument("--animation-path", type=Path, default=Path("results/figures/joint_dqn_eval_animation.gif"))
    parser.add_argument("--animation-stride", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(
        "evaluate_joint_dqn.py is deprecated because PlumeEnv now uses "
        "MultiDiscrete([move, left_sector, right_sector]). Use "
        "scripts\\evaluate_joint_ppo.py instead."
    )


if __name__ == "__main__":
    main()
