"""训练触须独立控制 PPO：策略只输出左右触须扇区命令。"""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import secrets
import sys
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs import WhiskerOnlyPuffEnv


class ObservationHistoryWrapper(gym.Wrapper):
    """把最近 N 帧观测拼接起来，给策略提供短时历史信息。

    MQ-3 这类气体传感器存在明显滞后，单帧读数很难判断气味是在上升、
    下降还是刚刚经过。因此这里把最近 `history_length` 帧连续观测拼接为
    一个长向量，让 MLP 策略也能看到短期趋势。
    """

    def __init__(self, env: gym.Env, history_length: int = 6) -> None:
        super().__init__(env)
        if history_length < 1:
            raise ValueError("history_length must be at least 1")
        self.history_length = int(history_length)
        self.history: deque[np.ndarray] = deque(maxlen=self.history_length)

        base_space = env.observation_space
        if not isinstance(base_space, spaces.Box):
            raise TypeError("ObservationHistoryWrapper requires a Box observation space")
        if len(base_space.shape) != 1:
            raise ValueError("Only flat vector observations are supported")

        low = np.tile(base_space.low, self.history_length).astype(np.float32)
        high = np.tile(base_space.high, self.history_length).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = np.asarray(obs, dtype=np.float32)
        self.history.clear()
        for _ in range(self.history_length):
            self.history.append(obs.copy())
        return self._stack_history(), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.history.append(np.asarray(obs, dtype=np.float32))
        return self._stack_history(), reward, terminated, truncated, info

    def _stack_history(self) -> np.ndarray:
        return np.concatenate(tuple(self.history)).astype(np.float32, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--history-length", type=int, default=6)
    parser.add_argument("--model-path", type=Path, default=ROOT / "results" / "models" / "whisker_only_ppo.zip")
    parser.add_argument("--log-dir", type=Path, default=ROOT / "results" / "logs" / "whisker_only_ppo")
    parser.add_argument("--tensorboard-dir", type=Path, default=ROOT / "results" / "tensorboard")
    parser.add_argument("--run-name", type=str, default="whisker_only_ppo")
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=1_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="开启羽流物理和传感器特性的 episode 级域随机化，提高 sim-to-real 鲁棒性。",
    )
    return parser.parse_args()


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def resolve_seed(seed: int | None) -> int:
    """未显式指定 seed 时随机生成，避免训练只适配单一气体场。"""
    if seed is not None:
        return int(seed)
    return secrets.randbelow(2**31 - 1)


def make_env(
    config: dict,
    seed: int,
    history_length: int,
    monitor_dir: Path | None = None,
    monitor_name: str = "monitor",
) -> Monitor:
    """创建触须环境。

    环境动作含义为 `[left_sector, right_sector]`。每一步中，环境会：
    1. 接收左右触须扇区；
    2. 将触须转到对应扇区；
    3. 在新的触须采样点读取气味浓度；
    4. 更新慢响应传感器；
    5. 返回带历史信息的观测。
    """
    env = WhiskerOnlyPuffEnv(config)
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
    # 多环境并行采样：每个环境有独立随机气体羽流，提高触须策略泛化性。
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
    eval_env = make_env(
        config,
        args.seed + 100_000,
        args.history_length,
        args.log_dir / "eval",
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(args.log_dir / "best_model"),
        log_path=str(args.log_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )

    # PPO 支持 MultiDiscrete([10, 10])，可以直接输出左右触须扇区。
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
    """训练结束后打印一个示例触须命令，方便和硬件 STEP 协议对齐。"""
    env = make_env(
        config,
        args.seed + 200_000,
        args.history_length,
        monitor_dir=None,
    )
    obs, _ = env.reset(seed=args.seed + 200_000)
    action, _ = model.predict(obs, deterministic=True)
    left_sector, right_sector = np.asarray(action, dtype=np.int64).reshape(-1)[:2]
    print(f"policy_action=[{int(left_sector)}, {int(right_sector)}]")
    print(f"hardware_like_command=STEP {int(left_sector)} {int(right_sector)}")
    env.close()


def save_run_metadata(args: argparse.Namespace, config: dict) -> None:
    """保存训练配置，后续复现实验或部署时可以直接查看。"""
    args.log_dir.mkdir(parents=True, exist_ok=True)
    metadata_env = WhiskerOnlyPuffEnv(config)
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
        "action_space": "[left_sector, right_sector]",
        "servo_60deg_time_s": metadata_env.whiskers.servo_60deg_time_s,
        "servo_angular_speed_deg_s": float(
            np.degrees(metadata_env.whiskers.angular_speed_rad_s)
        ),
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
    print(f"n_envs={args.n_envs}")
    print(f"history_length={args.history_length}")
    print(f"tensorboard_dir={args.tensorboard_dir}")
    print(f"tensorboard_command=tensorboard --logdir {args.tensorboard_dir}")


if __name__ == "__main__":
    main()
