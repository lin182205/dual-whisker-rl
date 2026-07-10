"""训练移动机器人 + 双触须气源搜索 PPO。

动作 `MultiDiscrete([6, 10, 10])` = [移动, 左扇区, 右扇区]。观测为 15 维硬件可部署
特征（12 维气味特征 + 朝向 cos/sin + 上一步移动），经历史堆叠喂给 PPO（MQ-3 慢响应
需要短时历史）。复用 `train_whisker_only_ppo` 的历史包装器与工具函数。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from train_whisker_only_ppo import ObservationHistoryWrapper
from train_whisker_only_ppo import load_config
from train_whisker_only_ppo import resolve_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument("--model-path", type=Path, default=ROOT / "results" / "models" / "mobile_whisker_ppo.zip")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results" / "logs" / "mobile_whisker_ppo")
    parser.add_argument("--tensorboard-dir", type=Path, default=ROOT / "results" / "tensorboard")
    parser.add_argument("--run-name", type=str, default="mobile_whisker_ppo")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="开启羽流/传感器 episode 级域随机化，提高 sim-to-real 鲁棒性。",
    )
    return parser.parse_args()


def make_env(config, seed, history_length, monitor_dir=None, monitor_name="monitor") -> Monitor:
    env = MobileWhiskerPuffEnv(config)
    env = ObservationHistoryWrapper(env, history_length=history_length)
    env.reset(seed=seed)
    monitor_file = None
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
        monitor_file = str(monitor_dir / f"{monitor_name}.csv")
    return Monitor(env, filename=monitor_file)


def train(args: argparse.Namespace) -> PPO:
    config = load_config(args.config)
    if args.domain_randomization:
        config["domain_randomization"] = True
    args.seed = resolve_seed(args.seed)
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.history_length < 1:
        raise ValueError("--history-length must be at least 1")

    set_random_seed(args.seed)
    env = DummyVecEnv(
        [
            (
                lambda rank=rank: make_env(
                    config,
                    args.seed + rank,
                    args.history_length,
                    args.log_dir / "train",
                    f"monitor_{rank}",
                )
            )
            for rank in range(args.n_envs)
        ]
    )
    eval_env = make_env(config, args.seed + 100_000, args.history_length, args.log_dir / "eval")
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # PPO 原生支持 MultiDiscrete([6, 10, 10])。
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
    preview_policy_command(model, config, args)
    env.close()
    eval_env.close()
    save_run_metadata(args, config)
    return model


def preview_policy_command(model: PPO, config: dict, args: argparse.Namespace) -> None:
    """训练结束后打印一个示例动作，方便和硬件命令对齐。"""
    env = make_env(config, args.seed + 200_000, args.history_length, monitor_dir=None)
    obs, _ = env.reset(seed=args.seed + 200_000)
    action, _ = model.predict(obs, deterministic=True)
    move, left, right = np.asarray(action, dtype=np.int64).reshape(-1)[:3]
    print(f"policy_action=[move={int(move)}, left={int(left)}, right={int(right)}]")
    print(f"hardware_like_command=MOVE {int(move)} | STEP {int(left)} {int(right)}")
    env.close()


def save_run_metadata(args: argparse.Namespace, config: dict) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    metadata_env = MobileWhiskerPuffEnv(config)
    base_obs_dim = int(metadata_env.observation_space.shape[0])
    metadata = {
        "seed": args.seed,
        "train_env_seeds": [args.seed + idx for idx in range(args.n_envs)],
        "eval_seed": args.seed + 100_000,
        "n_envs": args.n_envs,
        "timesteps": args.timesteps,
        "history_length": args.history_length,
        "base_observation_dim": base_obs_dim,
        "stacked_observation_dim": base_obs_dim * args.history_length,
        "observation_mode": metadata_env.observation_mode,
        "observation_field_names": metadata_env.observation_field_names,
        "sim_sensor_scale": metadata_env.observation_builder.config.scale,
        "init_pose_mode": metadata_env.init_pose_mode,
        "action_space": "[move_action, left_sector, right_sector]",
        "goal_radius": metadata_env.goal_radius,
        "source_position": (metadata_env.plume.source_x, metadata_env.plume.source_y),
        "model_path": str(args.model_path),
        "tensorboard_dir": str(args.tensorboard_dir),
        "run_name": args.run_name,
        "config_path": str(args.config) if args.config is not None else None,
        "config": config,
    }
    metadata_env.close()
    with (args.log_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    train(args)
    print(f"saved_model={args.model_path}")
    print(f"seed={args.seed}")


if __name__ == "__main__":
    main()
