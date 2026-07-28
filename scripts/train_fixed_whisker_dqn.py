"""训练 Fixed-whisker DQN baseline：策略只控制机器人，触须按固定周期扫描。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import FixedWhiskerPlumeEnv
from dual_whisker_rl.paths import portable_path
from dual_whisker_rl.paths import resolve_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model-path", type=Path, default=Path("results/models/fixed_whisker_dqn.zip"))
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs/fixed_whisker_dqn"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("results/tensorboard"))
    parser.add_argument("--run-name", type=str, default="fixed_whisker_dqn")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    return resolve_path_args(
        parser.parse_args(),
        "config",
        "model_path",
        "log_dir",
        "tensorboard_dir",
    )


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_seed(seed: int | None) -> int:
    """未指定 seed 时随机生成，减少单一随机种子导致的偶然性。"""
    if seed is not None:
        return int(seed)
    return secrets.randbelow(2**31 - 1)


def make_env(config: dict, seed: int, monitor_dir: Path | None = None) -> Monitor:
    """创建固定触须 baseline 环境，并用 Monitor 记录训练曲线。"""
    env = FixedWhiskerPlumeEnv(config)
    env.reset(seed=seed)
    monitor_file = None
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        monitor_file = str(monitor_dir / "monitor.csv")
    return Monitor(env, filename=monitor_file)


def train(args: argparse.Namespace) -> DQN:
    config = load_config(args.config)
    args.seed = resolve_seed(args.seed)
    set_random_seed(args.seed)
    env = make_env(config, args.seed, args.log_dir)
    eval_env = make_env(config, args.seed + 100_000, args.log_dir / "eval")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # FixedWhiskerPlumeEnv 的动作空间仍是 Discrete(6)，因此 DQN 仍然适用。
    model = DQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1_000,
        exploration_fraction=0.35,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs={"net_arch": [128, 128]},
        tensorboard_log=str(args.tensorboard_dir),
        seed=args.seed,
        verbose=1,
    )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False,
        callback=eval_callback,
        tb_log_name=args.run_name,
        log_interval=args.log_interval,
    )
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    env.close()
    eval_env.close()
    save_run_metadata(args, config)
    return model


def save_run_metadata(args: argparse.Namespace, config: dict) -> None:
    """保存训练配置，便于后续和 Joint PPO 做公平对比。"""
    args.log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": args.seed,
        "train_env_seed": args.seed,
        "eval_seed": args.seed + 100_000,
        "timesteps": args.timesteps,
        "config_path": portable_path(args.config),
        "model_path": portable_path(args.model_path),
        "tensorboard_dir": portable_path(args.tensorboard_dir),
        "run_name": args.run_name,
        "config": config,
    }
    with (args.log_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    args = parse_args()
    train(args)
    print(f"saved_model={args.model_path}")
    print(f"seed={args.seed}")
    print(f"tensorboard_dir={args.tensorboard_dir}")
    print(f"tensorboard_command=tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
