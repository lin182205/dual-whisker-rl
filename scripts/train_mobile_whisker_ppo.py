"""训练移动机器人 + 双触须气源搜索 PPO。

动作 `MultiDiscrete([6, 10, 10])` = [移动, 左扇区, 右扇区]。默认观测为 16 维硬件可部署
特征（12 维气味特征 + 朝向 cos/sin + 上一步移动 + blank_age）。历史仍由包装器堆叠为扁平向量，
可选择直接交给 MLP，或先通过 GRU / Transformer 编码时间依赖后再交给 PPO。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.agents import GRUHistoryExtractor
from dual_whisker_rl.agents import TransformerHistoryExtractor
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.paths import portable_path
from dual_whisker_rl.paths import resolve_path_args
from train_whisker_only_ppo import ObservationHistoryWrapper
from train_whisker_only_ppo import load_config
from train_whisker_only_ppo import resolve_seed


class RewardComponentsTensorboardCallback(BaseCallback):
    """记录奖励分量以及 blank/重捕获行为诊断。"""

    _NON_TENSORBOARD_OUTPUTS = ("stdout", "log", "json", "csv")

    def __init__(self, component_names: tuple[str, ...], verbose: int = 0) -> None:
        super().__init__(verbose=verbose)
        self.component_names = component_names
        self._episode_sums: list[dict[str, float]] = []
        self._rollout_sums: dict[str, float] = {}
        self._rollout_total = 0.0
        self._rollout_count = 0
        self._completed_episode_sums: dict[str, list[float]] = {}
        self._completed_episode_totals: list[float] = []
        self._diagnostic_step_count = 0
        self._blank_step_count = 0
        self._blank_body_spin_count = 0
        self._blank_body_motion_count = 0
        self._blank_whisker_motion_count = 0
        self._blank_age_sum = 0.0
        self._blank_age_max = 0
        self._qualified_reacquisition_events = 0
        self._whisker_only_reacquisition_events = 0
        self._body_assisted_reacquisition_events = 0
        self._passive_reacquisition_events = 0
        self._whisker_reacquisition_reward_events = 0

    def _on_training_start(self) -> None:
        self._episode_sums = [
            {name: 0.0 for name in self.component_names}
            for _ in range(self.training_env.num_envs)
        ]

    def _on_rollout_start(self) -> None:
        self._rollout_sums = {name: 0.0 for name in self.component_names}
        self._rollout_total = 0.0
        self._rollout_count = 0
        self._completed_episode_sums = {
            name: [] for name in self.component_names
        }
        self._completed_episode_totals = []
        self._diagnostic_step_count = 0
        self._blank_step_count = 0
        self._blank_body_spin_count = 0
        self._blank_body_motion_count = 0
        self._blank_whisker_motion_count = 0
        self._blank_age_sum = 0.0
        self._blank_age_max = 0
        self._qualified_reacquisition_events = 0
        self._whisker_only_reacquisition_events = 0
        self._body_assisted_reacquisition_events = 0
        self._passive_reacquisition_events = 0
        self._whisker_reacquisition_reward_events = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", ())
        dones = np.asarray(self.locals.get("dones", ()), dtype=bool).reshape(-1)
        for env_index, info in enumerate(infos):
            if "odor_hit" in info:
                self._diagnostic_step_count += 1
                blank_age_steps = int(info.get("blank_age_steps", 0))
                self._blank_age_sum += blank_age_steps
                self._blank_age_max = max(self._blank_age_max, blank_age_steps)
                if not bool(info["odor_hit"]):
                    self._blank_step_count += 1
                    if str(info.get("move_action", "")) in {
                        "spin_left",
                        "spin_right",
                    }:
                        self._blank_body_spin_count += 1
                    self._blank_body_motion_count += int(
                        bool(info.get("body_moved", False))
                    )
                    self._blank_whisker_motion_count += int(
                        bool(info.get("whisker_moved", False))
                    )
                self._qualified_reacquisition_events += int(
                    bool(info.get("reacquisition_event", False))
                )
                self._whisker_only_reacquisition_events += int(
                    bool(info.get("whisker_only_reacquisition", False))
                )
                self._body_assisted_reacquisition_events += int(
                    bool(info.get("body_assisted_reacquisition", False))
                )
                self._passive_reacquisition_events += int(
                    bool(info.get("passive_reacquisition", False))
                )
                self._whisker_reacquisition_reward_events += int(
                    bool(info.get("whisker_reacquisition_rewarded", False))
                )
            components = info.get("reward_components")
            if not isinstance(components, dict):
                continue
            self._rollout_count += 1
            step_total = 0.0
            for name in self.component_names:
                value = float(components.get(name, 0.0))
                self._rollout_sums[name] += value
                self._episode_sums[env_index][name] += value
                step_total += value
            self._rollout_total += step_total
            if env_index < dones.size and bool(dones[env_index]):
                episode_total = 0.0
                for name in self.component_names:
                    component_sum = self._episode_sums[env_index][name]
                    self._completed_episode_sums[name].append(component_sum)
                    episode_total += component_sum
                    self._episode_sums[env_index][name] = 0.0
                self._completed_episode_totals.append(episode_total)
        return True

    def _on_rollout_end(self) -> None:
        if self._rollout_count > 0:
            for name in self.component_names:
                self.logger.record(
                    f"reward_components/{name}_step_mean",
                    self._rollout_sums[name] / self._rollout_count,
                    exclude=self._NON_TENSORBOARD_OUTPUTS,
                )
            self.logger.record(
                "reward_components/total_step_mean",
                self._rollout_total / self._rollout_count,
                exclude=self._NON_TENSORBOARD_OUTPUTS,
            )
        for name in self.component_names:
            completed_values = self._completed_episode_sums[name]
            if completed_values:
                self.logger.record(
                    f"reward_components/{name}_episode_sum_mean",
                    float(np.mean(completed_values)),
                    exclude=self._NON_TENSORBOARD_OUTPUTS,
                )
        if self._completed_episode_totals:
            self.logger.record(
                "reward_components/total_episode_sum_mean",
                float(np.mean(self._completed_episode_totals)),
                exclude=self._NON_TENSORBOARD_OUTPUTS,
            )
        if self._diagnostic_step_count > 0:
            blank_denominator = max(self._blank_step_count, 1)
            reacquisition_denominator = max(
                self._qualified_reacquisition_events,
                1,
            )
            diagnostics = {
                "blank_body_spin_ratio": (
                    self._blank_body_spin_count / blank_denominator
                ),
                "blank_body_motion_ratio": (
                    self._blank_body_motion_count / blank_denominator
                ),
                "blank_whisker_motion_ratio": (
                    self._blank_whisker_motion_count / blank_denominator
                ),
                "mean_blank_age_steps": (
                    self._blank_age_sum / self._diagnostic_step_count
                ),
                "max_blank_age_steps": float(self._blank_age_max),
                "qualified_reacquisition_events_per_1000_steps": (
                    1000.0
                    * self._qualified_reacquisition_events
                    / self._diagnostic_step_count
                ),
                "qualified_reacquisition_events": float(
                    self._qualified_reacquisition_events
                ),
                "whisker_only_reacquisition_events": float(
                    self._whisker_only_reacquisition_events
                ),
                "body_assisted_reacquisition_events": float(
                    self._body_assisted_reacquisition_events
                ),
                "passive_reacquisition_events": float(
                    self._passive_reacquisition_events
                ),
                "whisker_only_reacquisition_rate": (
                    self._whisker_only_reacquisition_events
                    / reacquisition_denominator
                ),
                "body_assisted_reacquisition_rate": (
                    self._body_assisted_reacquisition_events
                    / reacquisition_denominator
                ),
                "passive_reacquisition_rate": (
                    self._passive_reacquisition_events
                    / reacquisition_denominator
                ),
                "whisker_reacquisition_reward_events": float(
                    self._whisker_reacquisition_reward_events
                ),
            }
            for name, value in diagnostics.items():
                self.logger.record(
                    f"diagnostics/{name}",
                    value,
                    exclude=self._NON_TENSORBOARD_OUTPUTS,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--scenario-mode",
        choices=("randomized", "fixed"),
        default=None,
        help="覆盖配置中的移动场景初始化模式；默认使用 randomized。",
    )
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--history-length", type=int, default=20)
    parser.add_argument(
        "--temporal-encoder",
        choices=("transformer", "gru", "mlp"),
        default="gru",
        help="历史信息编码方式：Transformer 时序编码、单层 GRU 时序编码，或原始扁平 MLP 基线。",
    )
    parser.add_argument("--transformer-d-model", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-ff-dim", type=int, default=128)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-features-dim", type=int, default=64)
    parser.add_argument("--gru-hidden-size", type=int, default=64)
    parser.add_argument("--gru-layers", type=int, default=1)
    parser.add_argument("--gru-dropout", type=float, default=0.0)
    parser.add_argument("--gru-features-dim", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=5)
    parser.add_argument("--ent-coef", type=float, default=0.003)
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="PPO 近似 KL 超过该值时提前停止当前轮更新，抑制 Transformer 过大更新。",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="默认按 temporal encoder 写入独立模型文件。",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="默认按 temporal encoder 写入独立日志目录。",
    )
    parser.add_argument("--tensorboard-dir", type=Path, default=Path("results/tensorboard"))
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="默认使用 mobile_whisker_<encoder>_ppo。",
    )
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=10_000,
        help="每隔多少个全局环境步保存一次 checkpoint。",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="默认保存到 <log-dir>/checkpoints。",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="从指定 PPO checkpoint 继续训练；--timesteps 表示额外训练步数。",
    )
    parser.add_argument(
        "--domain-randomization",
        action="store_true",
        help="开启羽流/传感器 episode 级域随机化，提高 sim-to-real 鲁棒性。",
    )
    args = parser.parse_args()
    resolve_output_paths(args)
    return args


def resolve_output_paths(args: argparse.Namespace) -> None:
    """按编码器生成默认相对路径，再统一按仓库根目录解析。"""
    experiment_name = f"mobile_whisker_{args.temporal_encoder}_ppo"
    if args.model_path is None:
        args.model_path = Path("results/models") / f"{experiment_name}.zip"
    if args.log_dir is None:
        args.log_dir = Path("results/logs") / experiment_name
    if args.run_name is None:
        args.run_name = experiment_name
    if args.checkpoint_dir is None:
        args.checkpoint_dir = args.log_dir / "checkpoints"
    resolve_path_args(
        args,
        "config",
        "model_path",
        "log_dir",
        "tensorboard_dir",
        "checkpoint_dir",
        "resume_from",
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.timesteps < 1:
        raise ValueError("--timesteps must be at least 1")
    if args.n_envs < 1:
        raise ValueError("--n-envs must be at least 1")
    if args.history_length < 1:
        raise ValueError("--history-length must be at least 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if args.n_steps < 1:
        raise ValueError("--n-steps must be at least 1")
    if args.batch_size < 2:
        raise ValueError("--batch-size must be at least 2")
    if args.n_epochs < 1:
        raise ValueError("--n-epochs must be at least 1")
    if args.ent_coef < 0.0:
        raise ValueError("--ent-coef must be non-negative")
    if args.target_kl <= 0.0:
        raise ValueError("--target-kl must be positive")
    if args.eval_freq < 1:
        raise ValueError("--eval-freq must be at least 1")
    if args.eval_episodes < 1:
        raise ValueError("--eval-episodes must be at least 1")
    if args.checkpoint_freq < 1:
        raise ValueError("--checkpoint-freq must be at least 1")
    if args.resume_from is not None and not args.resume_from.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {args.resume_from}")
    rollout_size = args.n_steps * args.n_envs
    if args.batch_size > rollout_size or rollout_size % args.batch_size != 0:
        raise ValueError(
            "--batch-size must not exceed n_steps * n_envs and must divide it exactly "
            f"(got batch_size={args.batch_size}, rollout_size={rollout_size})"
        )

    positive_transformer_args = {
        "--transformer-d-model": args.transformer_d_model,
        "--transformer-heads": args.transformer_heads,
        "--transformer-layers": args.transformer_layers,
        "--transformer-ff-dim": args.transformer_ff_dim,
        "--transformer-features-dim": args.transformer_features_dim,
    }
    for name, value in positive_transformer_args.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if args.transformer_d_model % args.transformer_heads != 0:
        raise ValueError("--transformer-d-model must be divisible by --transformer-heads")
    if not 0.0 <= args.transformer_dropout < 1.0:
        raise ValueError("--transformer-dropout must satisfy 0 <= dropout < 1")

    positive_gru_args = {
        "--gru-hidden-size": args.gru_hidden_size,
        "--gru-layers": args.gru_layers,
        "--gru-features-dim": args.gru_features_dim,
    }
    for name, value in positive_gru_args.items():
        if value < 1:
            raise ValueError(f"{name} must be at least 1")
    if not 0.0 <= args.gru_dropout < 1.0:
        raise ValueError("--gru-dropout must satisfy 0 <= dropout < 1")


def build_policy_kwargs(args: argparse.Namespace, base_observation_dim: int) -> dict:
    """构造 MLP 基线、Transformer 或 GRU 历史编码策略参数。"""
    policy_kwargs: dict = {
        "net_arch": [160, 160],
        "share_features_extractor": True,
    }
    if args.temporal_encoder == "transformer":
        policy_kwargs.update(
            {
                "features_extractor_class": TransformerHistoryExtractor,
                "features_extractor_kwargs": {
                    "history_length": args.history_length,
                    "base_observation_dim": base_observation_dim,
                    "d_model": args.transformer_d_model,
                    "n_heads": args.transformer_heads,
                    "n_layers": args.transformer_layers,
                    "dim_feedforward": args.transformer_ff_dim,
                    "dropout": args.transformer_dropout,
                    "features_dim": args.transformer_features_dim,
                },
            }
        )
    elif args.temporal_encoder == "gru":
        policy_kwargs.update(
            {
                "features_extractor_class": GRUHistoryExtractor,
                "features_extractor_kwargs": {
                    "history_length": args.history_length,
                    "base_observation_dim": base_observation_dim,
                    "hidden_size": args.gru_hidden_size,
                    "n_layers": args.gru_layers,
                    "dropout": args.gru_dropout,
                    "features_dim": args.gru_features_dim,
                },
            }
        )
    return policy_kwargs


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
    resolve_output_paths(args)
    validate_args(args)
    config = load_config(args.config)
    if args.scenario_mode is not None:
        config["scenario_mode"] = args.scenario_mode
    if args.domain_randomization:
        config["domain_randomization"] = True
    args.seed = resolve_seed(args.seed)

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
    base_observation_dim = int(env.envs[0].unwrapped.observation_space.shape[0])
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
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(args.checkpoint_dir),
        name_prefix=args.run_name,
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=2,
    )
    reward_components_callback = RewardComponentsTensorboardCallback(
        MobileWhiskerPuffEnv.REWARD_COMPONENT_NAMES
    )
    callbacks = CallbackList(
        [reward_components_callback, eval_callback, checkpoint_callback]
    )

    # PPO 原生支持 MultiDiscrete([6, 10, 10])。
    if args.resume_from is None:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            target_kl=args.target_kl,
            policy_kwargs=build_policy_kwargs(args, base_observation_dim),
            tensorboard_log=str(args.tensorboard_dir),
            seed=args.seed,
            verbose=1,
        )
        starting_num_timesteps = 0
    else:
        model = PPO.load(str(args.resume_from), device="auto")
        checkpoint_shape = tuple(model.observation_space.shape)
        environment_shape = tuple(env.observation_space.shape)
        if checkpoint_shape != environment_shape:
            env.close()
            eval_env.close()
            raise ValueError(
                "checkpoint observation shape does not match the current mobile "
                f"environment: checkpoint={checkpoint_shape}, "
                f"environment={environment_shape}, base_observation_dim="
                f"{base_observation_dim}, include_blank_age_observation="
                f"{env.envs[0].unwrapped.include_blank_age_observation}, "
                "include_lidar_observation="
                f"{env.envs[0].unwrapped.include_lidar_observation}. Use the same "
                "observation config as the checkpoint or start a new run."
            )
        model.set_env(env, force_reset=True)
        starting_num_timesteps = int(model.num_timesteps)
        model.tensorboard_log = str(args.tensorboard_dir)
        expected_extractor_cls = {
            "transformer": TransformerHistoryExtractor,
            "gru": GRUHistoryExtractor,
        }.get(args.temporal_encoder)
        checkpoint_extractor = model.policy.features_extractor
        # mlp 基线用 SB3 默认 flatten 提取器；transformer/gru 各自校验类型是否匹配。
        matches = (
            not isinstance(checkpoint_extractor, (TransformerHistoryExtractor, GRUHistoryExtractor))
            if expected_extractor_cls is None
            else isinstance(checkpoint_extractor, expected_extractor_cls)
        )
        if not matches:
            raise ValueError(
                "--temporal-encoder does not match checkpoint features extractor: "
                f"requested={args.temporal_encoder}, "
                f"checkpoint={type(checkpoint_extractor).__name__}"
            )
        print(f"resumed_from={portable_path(args.resume_from)}")
        print(f"starting_num_timesteps={starting_num_timesteps}")
        print(
            "resume_note=checkpoint PPO/model/optimizer parameters are preserved; "
            "--timesteps is additional training"
        )
    model.learn(
        total_timesteps=args.timesteps,
        progress_bar=False,
        callback=callbacks,
        tb_log_name=args.run_name,
        log_interval=args.log_interval,
        reset_num_timesteps=args.resume_from is None,
    )

    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    preview_policy_command(model, config, args)
    env.close()
    eval_env.close()
    save_run_metadata(args, config, model, starting_num_timesteps)
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


def save_run_metadata(
    args: argparse.Namespace,
    config: dict,
    model: PPO,
    starting_num_timesteps: int,
) -> None:
    args.log_dir.mkdir(parents=True, exist_ok=True)
    metadata_env = MobileWhiskerPuffEnv(config)
    base_obs_dim = int(metadata_env.observation_space.shape[0])
    extractor = model.policy.features_extractor
    transformer_config = None
    gru_config = None
    if args.temporal_encoder == "transformer":
        transformer_config = {
            "d_model": args.transformer_d_model,
            "n_heads": args.transformer_heads,
            "n_layers": args.transformer_layers,
            "dim_feedforward": args.transformer_ff_dim,
            "dropout": args.transformer_dropout,
            "features_dim": args.transformer_features_dim,
        }
    elif args.temporal_encoder == "gru":
        gru_config = {
            "hidden_size": args.gru_hidden_size,
            "n_layers": args.gru_layers,
            "dropout": args.gru_dropout,
            "features_dim": args.gru_features_dim,
        }
    metadata = {
        "seed": args.seed,
        "train_env_seeds": [args.seed + idx for idx in range(args.n_envs)],
        "eval_seed": args.seed + 100_000,
        "n_envs": args.n_envs,
        "timesteps": args.timesteps,
        "starting_num_timesteps": starting_num_timesteps,
        "final_num_timesteps": int(model.num_timesteps),
        "resume_from": portable_path(args.resume_from),
        "checkpoint_freq": args.checkpoint_freq,
        "checkpoint_dir": portable_path(args.checkpoint_dir),
        "ppo": {
            "learning_rate": float(model.lr_schedule(1.0)),
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs),
            "ent_coef": float(model.ent_coef),
            "target_kl": (
                float(model.target_kl) if model.target_kl is not None else None
            ),
        },
        "history_length": args.history_length,
        "base_observation_dim": base_obs_dim,
        "stacked_observation_dim": base_obs_dim * args.history_length,
        "temporal_encoder": args.temporal_encoder,
        "transformer": transformer_config,
        "gru": gru_config,
        "policy_features_dim": int(extractor.features_dim),
        "features_extractor_class": type(extractor).__name__,
        "features_extractor_parameters": sum(
            parameter.numel() for parameter in extractor.parameters()
        ),
        "whisker_motion_epsilon_rad": metadata_env.whisker_motion_epsilon_rad,
        "observation_mode": metadata_env.observation_mode,
        "include_blank_age_observation": (
            metadata_env.include_blank_age_observation
        ),
        "include_lidar_observation": metadata_env.include_lidar_observation,
        "blank_age_clip_s": metadata_env.blank_age_clip_s,
        "observation_field_names": metadata_env.observation_field_names,
        "lidar": {
            "num_beams": metadata_env.lidar.num_beams,
            "fov_deg": metadata_env.lidar.fov_deg,
            "max_range": metadata_env.lidar.max_range,
            "noise_std": metadata_env.lidar.noise_std,
            "robot_radius": metadata_env.robot_radius,
            "obstacles": metadata_env.plume.obstacles.tolist(),
        },
        "sim_sensor_scale": metadata_env.observation_builder.config.scale,
        "init_pose_mode": metadata_env.init_pose_mode,
        "scenario": metadata_env.scenario_metadata(),
        "action_space": "[move_action, left_sector, right_sector]",
        "goal_radius": metadata_env.goal_radius,
        "reward": {
            "progress_reward_scale": metadata_env.progress_reward_scale,
            "goal_bonus": metadata_env.goal_bonus,
            "oob_penalty": metadata_env.oob_penalty,
            "collision_penalty": metadata_env.collision_penalty,
            "best_concentration_reward_scale": (
                metadata_env.best_concentration_reward_scale
            ),
            "best_concentration_clip": metadata_env.best_concentration_clip,
            "reacquisition_min_blank_s": (
                metadata_env.reacquisition_min_blank_s
            ),
            "reacquisition_min_blank_steps": (
                metadata_env.reacquisition_min_blank_steps
            ),
            "reacquisition_credit_window_s": (
                metadata_env.reacquisition_credit_window_s
            ),
            "reacquisition_credit_window_steps": (
                metadata_env.reacquisition_credit_window_steps
            ),
            "whisker_reacquisition_bonus": (
                metadata_env.whisker_reacquisition_bonus
            ),
            "max_whisker_reacquisition_rewards": (
                metadata_env.max_whisker_reacquisition_rewards
            ),
            "concentration_progress_epsilon": (
                metadata_env.concentration_progress_epsilon
            ),
            "position_progress_epsilon": metadata_env.position_progress_epsilon,
            "distance_progress_epsilon": metadata_env.distance_progress_epsilon,
            "stagnation_window": metadata_env.stagnation_window,
            "stagnation_penalty": metadata_env.stagnation_penalty,
            "mobile_time_penalty": metadata_env.mobile_time_penalty,
            "positive_auxiliary_reward_upper_bound": (
                metadata_env.positive_auxiliary_reward_upper_bound
            ),
            "max_episode_time_cost": metadata_env.max_episode_time_cost,
            "minimum_oob_penalty": metadata_env.minimum_oob_penalty,
            "component_names": list(metadata_env.REWARD_COMPONENT_NAMES),
        },
        "source_position": (
            list(metadata_env.plume.source_position_override)
            if metadata_env.plume.source_position_override is not None
            else None
        ),
        "model_path": portable_path(args.model_path),
        "tensorboard_dir": portable_path(args.tensorboard_dir),
        "run_name": args.run_name,
        "config_path": portable_path(args.config),
        "config": config,
    }
    metadata_env.close()
    with (args.log_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    model = train(args)
    print(f"saved_model={portable_path(args.model_path)}")
    print(f"checkpoint_dir={portable_path(args.checkpoint_dir)}")
    print(f"final_num_timesteps={model.num_timesteps}")
    print(f"seed={args.seed}")
    print(f"temporal_encoder={args.temporal_encoder}")


if __name__ == "__main__":
    main()
