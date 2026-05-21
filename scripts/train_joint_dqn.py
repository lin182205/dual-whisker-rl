"""Train DQN on the joint movement-and-whisker control environment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import PlumeEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "default.yaml")
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=Path, default=ROOT / "results" / "models" / "joint_dqn.zip")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results" / "logs" / "joint_dqn")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_env(config: dict, seed: int, monitor_dir: Path | None = None) -> Monitor:
    env = PlumeEnv(config)
    env.reset(seed=seed)
    monitor_file = None
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        monitor_file = str(monitor_dir / "monitor.csv")
    return Monitor(env, filename=monitor_file)


def train(args: argparse.Namespace) -> DQN:
    config = load_config(args.config)
    set_random_seed(args.seed)
    env = make_env(config, args.seed, args.log_dir)

    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=80_000,
        learning_starts=2_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.45,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [160, 160]},
        tensorboard_log=str(args.log_dir),
        seed=args.seed,
        verbose=1,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    env.close()
    return model


def main() -> None:
    args = parse_args()
    train(args)
    print(f"saved_model={args.model_path}")


if __name__ == "__main__":
    main()
