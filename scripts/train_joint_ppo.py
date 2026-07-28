"""训练 Joint PPO：一个策略同时输出机器人动作和左右触须扇区。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import secrets
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import PlumeEnv
from dual_whisker_rl.paths import portable_path
from dual_whisker_rl.paths import resolve_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--model-path", type=Path, default=Path("results/models/joint_ppo.zip"))
    parser.add_argument("--log-dir", type=Path, default=Path("results/logs/joint_ppo"))
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("results/tensorboard"))
    parser.add_argument("--run-name", type=str, default="joint_ppo")
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
    """未指定 seed 时随机生成，避免策略只适配单一随机序列。"""
    if seed is not None:
        return int(seed)
    return secrets.randbelow(2**31 - 1)


def make_env(
    config: dict,
    seed: int,
    monitor_dir: Path | None = None,
    monitor_name: str = "monitor",
) -> Monitor:
    """创建带 Monitor 的环境，用于记录 episode reward/length 到日志。"""
    env = PlumeEnv(config)
    env.reset(seed=seed)
    monitor_file = None
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        monitor_file = str(monitor_dir / f"{monitor_name}.csv")
    return Monitor(env, filename=monitor_file)


def train(args: argparse.Namespace) -> PPO:
    config = load_config(args.config)
    args.seed = resolve_seed(args.seed)
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")

    set_random_seed(args.seed)
    # 多环境采样：PPO 每次从多个独立环境收集 rollout，提高样本多样性。
    env = DummyVecEnv(
        [
            (
                lambda rank=rank: make_env(
                    config,
                    args.seed + rank,
                    args.log_dir / "train",
                    f"monitor_{rank}",
                )
            )
            for rank in range(args.n_envs)
        ]
    )
    # 评估环境使用独立 seed，避免训练采样和评估采样完全重合。
    eval_env = make_env(config, args.seed + 100_000, args.log_dir / "eval")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # PPO 支持 MultiDiscrete 动作空间，适合当前 [move, left_sector, right_sector]。
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs={"net_arch": [160, 160]},
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
    """保存本次训练的关键参数，方便后续复现实验。"""
    args.log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": args.seed,
        "train_env_seeds": [args.seed + idx for idx in range(args.n_envs)],
        "eval_seed": args.seed + 100_000,
        "n_envs": args.n_envs,
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
    print(f"n_envs={args.n_envs}")
    print(f"tensorboard_dir={args.tensorboard_dir}")
    print(f"tensorboard_command=tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
