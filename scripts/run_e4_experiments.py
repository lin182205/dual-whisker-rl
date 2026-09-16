"""统一运行 E4 仿真主对比实验。

示例：
  python scripts/run_e4_experiments.py prepare --profile smoke
  python scripts/run_e4_experiments.py run --profile smoke --methods main,b0_random
  python scripts/run_e4_experiments.py evaluate --profile formal --resume
  python scripts/run_e4_experiments.py status --profile smoke
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.e4_experiments import (  # noqa: E402
    E4Scenario,
    METHOD_ORDER,
    RULE_METHODS,
    TRAINABLE_METHODS,
    RandomController,
    ReactiveController,
    aggregate,
    build_scenarios,
    evaluate_one,
    make_env,
    method_metadata,
    scenario_hash,
    sha256_file,
    choose_mlp_width,
)
from dual_whisker_rl.paths import project_path, portable_path
from dual_whisker_rl.envs.world_bounds import resolve_world_bounds, world_bounds_config  # noqa: E402


VERSION = "e4-v1"
DEFAULT_OUTPUT = Path("results/e4")
PPO_EPOCHS = 5
ETA_WINDOW_ROLLOUTS = 5


def configure_output_encoding() -> None:
    """重定向日志统一写 UTF-8；交互式 Windows 终端沿用其本地编码。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure) and not stream.isatty():
            reconfigure(encoding="utf-8", errors="replace")


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".tmp", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
        temp = Path(f.name)
    temp.replace(path)


def load_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    import yaml
    with path.open("r", encoding="utf-8-sig") as f:
        value = yaml.safe_load(f)
    return dict(value or {})


def parse_list(value: str | None, *, cast=str) -> list[Any] | None:
    if value is None or not value.strip():
        return None
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def make_rollout_eta_callback(method: str, seed: int, target_timesteps: int, rollout_size: int):
    """按完整 rollout 加 PPO 更新周期估算剩余训练分钟数。"""
    from stable_baselines3.common.callbacks import BaseCallback

    class RolloutEtaCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.cycle_started_at: float | None = None
            self.cycle_seconds: list[float] = []
            self.completed_rollouts = 0

        def _report_cycle(self, finished_at: float) -> None:
            if self.cycle_started_at is None:
                return
            elapsed = max(0.0, finished_at - self.cycle_started_at)
            self.cycle_seconds.append(elapsed)
            self.cycle_seconds = self.cycle_seconds[-ETA_WINDOW_ROLLOUTS:]
            self.completed_rollouts += 1
            average_seconds = float(np.mean(self.cycle_seconds))
            completed_steps = int(self.model.num_timesteps)
            remaining_steps = max(0, int(target_timesteps) - completed_steps)
            remaining_rollouts = (remaining_steps + int(rollout_size) - 1) // int(rollout_size)
            remaining_minutes = average_seconds * remaining_rollouts / 60.0
            print(
                f"[训练剩余时间] 方法={method} 随机种子={seed} "
                f"已完成采样轮次={self.completed_rollouts} "
                f"本轮采样及{PPO_EPOCHS}轮更新耗时={elapsed:.2f}秒 "
                f"最近{ETA_WINDOW_ROLLOUTS}轮平均耗时={average_seconds:.2f}秒 "
                f"训练步数={completed_steps}/{target_timesteps} "
                f"预计剩余={remaining_minutes:.2f}分钟",
                flush=True,
            )

        def _on_rollout_start(self) -> None:
            now = time.perf_counter()
            self._report_cycle(now)
            self.cycle_started_at = now

        def _on_step(self) -> bool:
            return True

        def _on_training_end(self) -> None:
            self._report_cycle(time.perf_counter())
            self.cycle_started_at = None

    return RolloutEtaCallback()


def profile_defaults(profile: str) -> dict[str, Any]:
    if profile == "smoke":
        return {"timesteps": 4096, "n_envs": 2, "seeds": [1], "scenario_profile": "smoke", "checkpoint_rollouts": 2, "vec_env_backend": "dummy"}
    if profile == "formal":
        return {"timesteps": 1_000_000, "n_envs": 8, "seeds": [1, 2, 3, 4, 5], "scenario_profile": "formal", "checkpoint_rollouts": 25, "vec_env_backend": "subproc"}
    raise ValueError("profile must be smoke or formal")


def resolve_args(args: argparse.Namespace) -> None:
    defaults = profile_defaults(args.profile)
    args.output_dir = project_path(args.output_dir)
    args.config = project_path(args.config) if args.config else None
    args.methods = parse_list(args.methods) or list(METHOD_ORDER)
    args.seeds = parse_list(args.seeds, cast=int) or defaults["seeds"]
    args.timesteps = int(args.timesteps or defaults["timesteps"])
    args.n_envs = int(args.n_envs or defaults["n_envs"])
    args.checkpoint_rollouts = int(getattr(args, "checkpoint_rollouts", None) or defaults["checkpoint_rollouts"])
    config_backend = load_yaml(args.config).get("vec_env_backend") if args.config else None
    requested_backend = str(getattr(args, "vec_env_backend", None) or config_backend or defaults["vec_env_backend"])
    if requested_backend == "auto":
        args.vec_env_backend = "subproc" if args.n_envs > 1 else "dummy"
    else:
        args.vec_env_backend = requested_backend
    if args.eval_limit is None:
        args.eval_limit = 12 if args.profile == "smoke" else None
    unknown = [name for name in args.methods if name not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"unknown E4 methods: {', '.join(unknown)}")
    if args.vec_env_backend not in ("dummy", "subproc"):
        raise ValueError("vec-env-backend must be auto, dummy or subproc")
    if args.timesteps < 1 or args.n_envs < 1 or args.checkpoint_rollouts < 1:
        raise ValueError("timesteps, n-envs and checkpoint-rollouts must be positive")

def seeds_for_method(args: argparse.Namespace, method: str) -> list[int]:
    """返回方法实际需要运行的训练种子。

    B1 是确定性的规则控制器，没有训练过程；只保留请求列表中的第一个种子作为
    规范目录和统计键，避免 formal 的 5 个 seed 被重复计入。
    """
    if method == "b1_reactive":
        return [int(args.seeds[0])]
    return [int(seed) for seed in args.seeds]


def base_config(args: argparse.Namespace) -> dict[str, Any]:
    cfg = load_yaml(args.config)
    cfg["vec_env_backend"] = args.vec_env_backend
    cfg.setdefault("plume_model", "dynamic")
    cfg.setdefault("scenario_mode", "randomized")
    world_min, world_max, world_half = resolve_world_bounds(
        cfg.get("world_bounds"), cfg.get("world_half"), default_half=2.5
    )
    cfg["world_bounds"] = world_bounds_config(world_min, world_max)
    cfg.setdefault("domain_randomization", False)
    # E4 统一使用现有动态 puff 的慢风分层，禁止纸面羽流配置混入。
    if str(cfg.get("plume_model")) != "dynamic":
        raise ValueError("E4 requires plume_model=dynamic")
    cfg["wind_speed_range"] = (0.12, 0.20)
    # 将大场羽流的尺度相关默认显式写回 manifest，保证运行可追溯。
    plume_overrides = dict(cfg.get("plume_overrides") or {})
    wind_min = max(float(cfg["wind_speed_range"][0]), 1e-3)
    plume_overrides.setdefault("max_puff_age", max(16.0, 2.2 * world_half / wind_min))
    plume_overrides.setdefault("puff_bounds_margin", max(1.1, 0.20 * world_half))
    cfg["plume_overrides"] = plume_overrides
    cfg.setdefault("progress_reward_scale", 3.0 / max(world_half, 1e-6))
    cfg.setdefault("b1_hit_threshold", 0.08)
    cfg.setdefault("b1_trend_threshold", -0.02)
    cfg.setdefault("b1_contrast_threshold", 0.03)
    cfg.setdefault("b1_search_period", 8)
    cfg.setdefault("b1_scan_period", 2)
    if float(cfg["b1_hit_threshold"]) < 0.0 or float(cfg["b1_contrast_threshold"]) < 0.0:
        raise ValueError("B1 hit/contrast thresholds must be non-negative")
    if int(cfg["b1_search_period"]) < 1 or int(cfg["b1_scan_period"]) < 1:
        raise ValueError("B1 search/scan periods must be positive")
    return cfg


def experiment_manifest(args: argparse.Namespace, scenarios: dict[str, list[E4Scenario]]) -> dict[str, Any]:
    cfg = base_config(args)
    code_files = [Path(__file__).resolve(), ROOT / "dual_whisker_rl" / "e4_experiments.py", ROOT / "dual_whisker_rl" / "envs" / "mobile_whisker_env.py", ROOT / "dual_whisker_rl" / "envs" / "whisker_only_env.py", ROOT / "dual_whisker_rl" / "envs" / "world_bounds.py"]
    return {
        "version": VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "profile": args.profile,
        "methods": [method_metadata(m) for m in args.methods],
        "seeds": args.seeds,
        "method_seeds": {method: seeds_for_method(args, method) for method in args.methods},
        "timesteps": args.timesteps,
        "n_envs": args.n_envs,
        "vec_env_backend": args.vec_env_backend,
        "checkpoint_rollouts": args.checkpoint_rollouts,
        "config": cfg,
        "scenario_counts": {k: len(v) for k, v in scenarios.items()},
        "scenario_hashes": {k: scenario_hash(v) for k, v in scenarios.items()},
        "code_hashes": {portable_path(p): sha256_file(p) for p in code_files},
        "evaluation": {
            "primary_metric": "success_rate",
            "validation_not_used_for_test_selection": True,
            "deterministic_policy": True,
            "animation": False,
        },
    }


def prepare(args: argparse.Namespace) -> Path:
    cfg = base_config(args)
    scenarios = {
        "validation": build_scenarios("validation", args.profile, world_bounds=cfg["world_bounds"]),
        "test": build_scenarios("test", args.profile, world_bounds=cfg["world_bounds"]),
    }
    manifest = experiment_manifest(args, scenarios)
    root = args.output_dir / args.profile
    if args.dry_run:
        print(json.dumps({"output": portable_path(root), "scenario_counts": manifest["scenario_counts"], "methods": args.methods, "seeds": args.seeds, "method_seeds": manifest["method_seeds"], "vec_env_backend": args.vec_env_backend, "checkpoint_rollouts": args.checkpoint_rollouts}, ensure_ascii=False, indent=2))
        return root
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "manifest.json", manifest)
    for split, values in scenarios.items():
        path = root / f"scenarios_{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for value in values:
                f.write(json.dumps(value.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    for method in args.methods:
        for seed in seeds_for_method(args, method):
            state_path = root / method / f"seed_{seed}" / "state.json"
            if not state_path.exists():
                atomic_write_json(state_path, {
                    "version": VERSION,
                    "method": method,
                    "seed": seed,
                    "status": "prepared",
                    "timesteps_target": args.timesteps,
                    "actual_timesteps": 0,
                    "scenario_hashes": manifest["scenario_hashes"],
                    "signature": experiment_signature(args, method, seed),
                })
    print(f"准备完成：目录={portable_path(root)} 验证场景={len(scenarios['validation'])} 测试场景={len(scenarios['test'])}")
    return root


def read_scenarios(root: Path, split: str, limit: int | None = None) -> list[E4Scenario]:
    path = root / f"scenarios_{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"missing scenario file: {path}; run prepare first")
    values = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                values.append(E4Scenario.from_dict(json.loads(line)))
                if limit is not None and len(values) >= limit:
                    break
    return values


def verify_scenario_manifest(root: Path) -> dict[str, str]:
    """校验封存 JSONL 场景与 manifest 摘要一致，拒绝静默改变难度。"""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}; run prepare first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = dict(manifest.get("scenario_hashes") or {})
    actual = {split: scenario_hash(read_scenarios(root, split)) for split in ("validation", "test")}
    if expected and actual != expected:
        raise ValueError(f"scenario manifest mismatch: expected={expected}, actual={actual}")
    return actual

def state_path(task_dir: Path) -> Path:
    return task_dir / "state.json"


def latest_checkpoint(task_dir: Path) -> Path | None:
    candidates = list((task_dir / "checkpoints").glob("*.zip"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def experiment_signature(args: argparse.Namespace, method: str, seed: int, fixed_sector: int = 5, config_override: dict[str, Any] | None = None) -> str:
    signature_config = base_config(args)
    if config_override:
        signature_config.update(config_override)
    payload = {
        "version": VERSION,
        "method": method,
        "seed": int(seed),
        "profile": args.profile,
        "n_envs": int(args.n_envs),
        "vec_env_backend": args.vec_env_backend,
        "config": signature_config,
        "fixed_sector": int(fixed_sector),
        "checkpoint_rollouts": int(args.checkpoint_rollouts),
        "code_hashes": {
            "runner": sha256_file(Path(__file__).resolve()),
            "experiment": sha256_file(ROOT / "dual_whisker_rl" / "e4_experiments.py"),
            "mobile_env": sha256_file(ROOT / "dual_whisker_rl" / "envs" / "mobile_whisker_env.py"),
            "plume_env": sha256_file(ROOT / "dual_whisker_rl" / "envs" / "whisker_only_env.py"),
            "world_bounds": sha256_file(ROOT / "dual_whisker_rl" / "envs" / "world_bounds.py"),
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def make_model_env(method: str, cfg: dict[str, Any], seed: int, fixed_sector: int = 5):
    from stable_baselines3.common.monitor import Monitor
    env = make_env(method, {**cfg, "seed": seed}, history_length=method_metadata(method)["history_length"], fixed_sector=fixed_sector)
    return Monitor(env)


def train_one(args: argparse.Namespace, root: Path, method: str, seed: int, *, fixed_sector: int = 5, task_dir_override: Path | None = None) -> None:
    scenario_hashes = verify_scenario_manifest(root)
    task_dir = task_dir_override or (root / method / f"seed_{seed}")
    task_dir.mkdir(parents=True, exist_ok=True)
    current_state = json.loads(state_path(task_dir).read_text(encoding="utf-8")) if state_path(task_dir).exists() else {}
    signature = experiment_signature(args, method, seed, fixed_sector=fixed_sector)
    if current_state.get("scenario_hashes") and current_state.get("scenario_hashes") != scenario_hashes:
        if args.resume:
            raise ValueError(f"scenario hash mismatch for {method} seed {seed}; prepare a new experiment directory")
    current_state["scenario_hashes"] = scenario_hashes
    if args.resume and current_state.get("signature") and current_state.get("signature") != signature:
        raise ValueError(f"resume signature mismatch for {method} seed {seed}; prepare a new experiment directory")
    if method in RULE_METHODS:
        atomic_write_json(state_path(task_dir), {**current_state, "status": "ready", "actual_timesteps": 0, "artifact": "rule_controller"})
        return
    if args.resume and current_state.get("status") == "completed" and int(current_state.get("actual_timesteps", 0)) >= int(args.timesteps) and (task_dir / "model.zip").exists():
        return
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.callbacks import CallbackList, EvalCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from dual_whisker_rl.agents import GRUHistoryExtractor

    cfg = base_config(args)
    set_random_seed(seed)
    env_fns = [
        partial(make_model_env, method, cfg, seed + rank, fixed_sector=fixed_sector)
        for rank in range(args.n_envs)
    ]
    if args.vec_env_backend == "subproc":
        # Windows 明确使用 spawn；每个环境在独立进程运行，动态羽流计算不受 GIL 限制。
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    meta = method_metadata(method)
    n_steps = min(512, max(32, args.timesteps // max(args.n_envs, 1)))
    rollout = n_steps * args.n_envs
    batch_size = min(256, rollout)
    while rollout % batch_size != 0:
        batch_size //= 2
        if batch_size < 2:
            batch_size = 2
            break
    observation_dim = int(env.observation_space.shape[0])
    mlp_width = choose_mlp_width(observation_dim) if method == "b5_history_mlp" else 160
    kwargs: dict[str, Any] = {"net_arch": [mlp_width, mlp_width], "share_features_extractor": True}
    if meta["temporal_encoder"] == "gru":
        base_dims = [int(value) for value in env.get_attr("base_observation_dim")]
        if not base_dims or len(set(base_dims)) != 1:
            raise ValueError(f"inconsistent base observation dimensions across workers: {base_dims}")
        kwargs.update({"features_extractor_class": GRUHistoryExtractor, "features_extractor_kwargs": {"history_length": meta["history_length"], "base_observation_dim": base_dims[0]}})
    checkpoint_dir = task_dir / "checkpoints"
    checkpoint_frequency = max(n_steps * args.checkpoint_rollouts, 1)
    callback_checkpoint = CheckpointCallback(save_freq=checkpoint_frequency, save_path=str(checkpoint_dir), name_prefix=f"{method}_seed{seed}")
    tensorboard_dir = task_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    eval_env_fn = partial(make_model_env, method, cfg, seed + 100000, fixed_sector=fixed_sector)
    if args.vec_env_backend == "subproc":
        # EvalCallback 要求训练/验证 VecEnv 类型一致；周期验证只需一个独立 worker。
        eval_env = SubprocVecEnv([eval_env_fn], start_method="spawn")
    else:
        eval_env = DummyVecEnv([eval_env_fn])
    eval_callback = EvalCallback(eval_env, best_model_save_path=str(task_dir / "best_model"), log_path=str(task_dir / "eval"), eval_freq=max(rollout * 25 // max(args.n_envs, 1), 1), n_eval_episodes=5, deterministic=True, render=False)
    eta_callback = make_rollout_eta_callback(method, seed, args.timesteps, rollout)
    callback = CallbackList([callback_checkpoint, eval_callback, eta_callback])
    checkpoint = latest_checkpoint(task_dir) if args.resume else None
    try:
        if checkpoint is not None:
            model = PPO.load(str(checkpoint), env=env, device="auto")
            model.tensorboard_log = str(tensorboard_dir)
            remaining = max(0, args.timesteps - int(model.num_timesteps))
            if remaining > 0:
                model.learn(total_timesteps=remaining, reset_num_timesteps=False, callback=callback, progress_bar=False, tb_log_name=f"e4_{method}_seed{seed}")
            resumed_from = portable_path(checkpoint)
        else:
            model = PPO("MlpPolicy", env, learning_rate=1e-4, n_steps=n_steps, batch_size=batch_size, n_epochs=PPO_EPOCHS, gamma=0.99, gae_lambda=0.95, ent_coef=0.003, target_kl=0.02, policy_kwargs=kwargs, seed=seed, verbose=0, tensorboard_log=str(tensorboard_dir))
            model.learn(total_timesteps=args.timesteps, reset_num_timesteps=True, callback=callback, progress_bar=False, tb_log_name=f"e4_{method}_seed{seed}")
            resumed_from = None
        final_tmp = task_dir / "model.tmp"
        model.save(str(final_tmp))
        saved_model = task_dir / "model.zip"
        final_tmp.replace(saved_model)
        actual = int(model.num_timesteps)
        parameter_count = int(sum(parameter.numel() for parameter in model.policy.parameters()))
        parameter_target = 60000 if method == "b5_history_mlp" else None
        atomic_write_json(task_dir / "metadata.json", {"version": VERSION, "method": method, "seed": seed, "signature": signature, "requested_timesteps": args.timesteps, "actual_timesteps": actual, "history_length": meta["history_length"], "temporal_encoder": meta["temporal_encoder"], "mlp_width": mlp_width, "parameter_count": parameter_count, "parameter_target": parameter_target, "parameter_delta": (parameter_count - parameter_target) if parameter_target is not None else None, "resumed_from": resumed_from, "checkpoint_rollouts": int(args.checkpoint_rollouts), "checkpoint_frequency_env_steps": int(checkpoint_frequency * args.n_envs), "ppo_epochs": PPO_EPOCHS, "eta_window_rollouts": ETA_WINDOW_ROLLOUTS, "eta_unit": "minutes", "vec_env_backend": args.vec_env_backend, "subproc_start_method": "spawn" if args.vec_env_backend == "subproc" else None, "tensorboard_log_dir": portable_path(tensorboard_dir), "config": cfg})
        atomic_write_json(state_path(task_dir), {**current_state, "status": "completed", "timesteps_target": int(args.timesteps), "signature": signature, "actual_timesteps": actual, "artifact": portable_path(saved_model)})
    except KeyboardInterrupt as exc:
        atomic_write_json(state_path(task_dir), {**current_state, "status": "interrupted", "error": repr(exc), "checkpoint": portable_path(latest_checkpoint(task_dir)) if latest_checkpoint(task_dir) else None})
        raise
    except Exception as exc:
        atomic_write_json(state_path(task_dir), {**current_state, "status": "failed", "error": repr(exc), "checkpoint": portable_path(latest_checkpoint(task_dir)) if latest_checkpoint(task_dir) else None})
        raise
    finally:
        env.close()
        eval_env.close()


def load_controller(method: str, task_dir: Path, args: argparse.Namespace, config: dict[str, Any] | None = None):
    if method == "b0_random":
        return RandomController(17)
    if method == "b1_reactive":
        cfg = dict(config or {})
        return ReactiveController(
            hit_threshold=float(cfg.get("b1_hit_threshold", 0.08)),
            trend_threshold=float(cfg.get("b1_trend_threshold", -0.02)),
            contrast_threshold=float(cfg.get("b1_contrast_threshold", 0.03)),
            search_period=int(cfg.get("b1_search_period", 8)),
            scan_period=int(cfg.get("b1_scan_period", 2)),
        )
    from stable_baselines3 import PPO
    from dual_whisker_rl.agents import GRUHistoryExtractor
    model_path = task_dir / "model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"model missing for {method}: {model_path}")
    _ = GRUHistoryExtractor
    return PPO.load(str(model_path), device="auto")


def evaluate_method(args: argparse.Namespace, root: Path, method: str, seed: int, split: str, *, fixed_sector: int = 5, task_dir_override: Path | None = None) -> list[dict[str, Any]]:
    verify_scenario_manifest(root)
    task_dir = task_dir_override or (root / method / f"seed_{seed}")
    task_dir.mkdir(parents=True, exist_ok=True)
    scenarios = read_scenarios(root, split, args.eval_limit)
    cfg = base_config(args)
    if method == "b1_reactive":
        cfg.update(selected_b1_config(root))
    controller = load_controller(method, task_dir, args, cfg)
    signature = experiment_signature(args, method, seed, fixed_sector=fixed_sector, config_override=cfg)
    env = make_env(method, {**cfg, "seed": seed}, fixed_sector=fixed_sector)
    rows: list[dict[str, Any]] = []
    episodes_dir = task_dir / "episodes" / split
    episodes_dir.mkdir(parents=True, exist_ok=True)
    eval_state_path = task_dir / f"evaluation_{split}.json"
    eval_state = {"version": VERSION, "method": method, "seed": seed, "split": split, "signature": signature, "status": "running", "total": len(scenarios), "completed": 0}
    atomic_write_json(eval_state_path, eval_state)
    for scenario in scenarios:
        output = episodes_dir / f"{scenario.scenario_id}.json"
        if args.resume and output.exists():
            try:
                row = json.loads(output.read_text(encoding="utf-8"))
                if row.get("scenario_id") == scenario.scenario_id and row.get("signature") == signature:
                    rows.append(row)
                    atomic_write_json(eval_state_path, {**eval_state, "completed": len(rows)})
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        policy_seed = int(scenario.seed + 7001 + 1000003 * int(seed)) if method == "b0_random" else int(seed)
        if method == "b0_random":
            controller = RandomController(policy_seed)
        if hasattr(controller, "reset"):
            controller.reset()
        row = evaluate_one(controller, env, scenario, max_steps=args.max_steps)
        row["method"] = method
        row["training_seed"] = seed
        row["policy_seed"] = policy_seed
        row["signature"] = signature
        atomic_write_json(output, row)
        rows.append(row)
        atomic_write_json(eval_state_path, {**eval_state, "completed": len(rows)})
    env.close()
    atomic_write_json(task_dir / f"metrics_{split}.json", aggregate(rows))
    atomic_write_json(eval_state_path, {**eval_state, "status": "completed", "completed": len(rows)})
    return rows


def calibrate_command(args: argparse.Namespace, root: Path) -> None:
    """按固定验证集选择 B1 规则参数和 B2 固定镜像扇区。"""
    candidates = list(range(10))
    path = root / "calibration_b2.json"
    b1_path = root / "calibration_b1.json"
    if args.profile == "smoke":
        payload = {
            "version": VERSION,
            "method": "b2_fixed",
            "status": "uncalibrated_smoke",
            "candidate_sectors": candidates,
            "candidate_angles_deg": [float((s + 0.5) * 18.0) for s in candidates],
            "seeds": args.seeds[:2],
            "timesteps_per_candidate": 0,
            "selected_sector": 5,
            "selection_rule": "validation success_rate, then mean_final_distance, then earliest checkpoint",
            "note": "smoke skips candidate training; selected_sector is a placeholder",
        }
        b1_payload = {
            "version": VERSION,
            "method": "b1_reactive",
            "status": "uncalibrated_smoke",
            "selected_config": {"b1_hit_threshold": 0.08, "b1_trend_threshold": -0.02, "b1_contrast_threshold": 0.03, "b1_search_period": 8, "b1_scan_period": 2},
            "note": "smoke skips validation parameter search",
        }
        if not args.dry_run:
            atomic_write_json(path, payload)
            atomic_write_json(b1_path, b1_payload)
        print(json.dumps({"b2": payload, "b1": b1_payload}, ensure_ascii=False, indent=2))
        return
    b1_grid = [
        {"b1_hit_threshold": hit, "b1_trend_threshold": trend, "b1_contrast_threshold": contrast, "b1_search_period": search, "b1_scan_period": scan}
        for hit in (0.06, 0.08, 0.10)
        for trend in (-0.03, -0.02)
        for contrast in (0.02, 0.03)
        for search in (6, 8)
        for scan in (1, 2)
    ]
    if args.dry_run:
        print(json.dumps({"status": "planned", "b1_candidates": len(b1_grid), "candidate_sectors": candidates, "seeds": args.seeds[:2], "timesteps_per_candidate": 102400}, ensure_ascii=False, indent=2))
        return
    calibration_args = argparse.Namespace(**vars(args))
    calibration_args.timesteps = 102400
    calibration_args.eval_limit = None
    validation_scenarios = read_scenarios(root, "validation", None)
    b1_scores: list[dict[str, Any]] = []
    for candidate_index, candidate_config in enumerate(b1_grid):
        candidate_cfg = base_config(args)
        candidate_cfg.update(candidate_config)
        candidate_env = make_env("b1_reactive", candidate_cfg)
        candidate_controller = ReactiveController(
            hit_threshold=float(candidate_config["b1_hit_threshold"]),
            trend_threshold=float(candidate_config["b1_trend_threshold"]),
            contrast_threshold=float(candidate_config["b1_contrast_threshold"]),
            search_period=int(candidate_config["b1_search_period"]),
            scan_period=int(candidate_config["b1_scan_period"]),
        )
        candidate_rows = []
        for validation_scenario in validation_scenarios:
            candidate_controller.reset()
            candidate_rows.append(evaluate_one(candidate_controller, candidate_env, validation_scenario, max_steps=args.max_steps))
        candidate_env.close()
        candidate_summary = aggregate(candidate_rows)
        b1_scores.append({"candidate_index": candidate_index, "config": candidate_config, "success_rate": candidate_summary.get("success_rate", 0.0), "mean_final_distance": candidate_summary.get("mean_final_distance")})
    b1_selected = min(b1_scores, key=lambda item: (-float(item["success_rate"]), float(item["mean_final_distance"]) if item["mean_final_distance"] is not None else float("inf"), int(item["candidate_index"])))
    atomic_write_json(b1_path, {"version": VERSION, "method": "b1_reactive", "status": "completed", "candidate_count": len(b1_grid), "selected_config": b1_selected["config"], "selection_rule": "validation success_rate, then mean_final_distance, then candidate order", "scores": b1_scores})
    scores: dict[str, Any] = {}
    for sector in candidates:
        candidate_values = []
        for seed in args.seeds[:2]:
            task_dir = root / "calibration" / f"sector_{sector}" / f"seed_{seed}"
            train_one(calibration_args, root, "b2_fixed", seed, fixed_sector=sector, task_dir_override=task_dir)
            rows = evaluate_method(calibration_args, root, "b2_fixed", seed, "validation", fixed_sector=sector, task_dir_override=task_dir)
            summary = aggregate(rows)
            checkpoint_steps = []
            for checkpoint_path in (task_dir / "checkpoints").glob("*.zip"):
                try:
                    checkpoint_steps.append(int(checkpoint_path.stem.split("_")[-2]))
                except (IndexError, ValueError):
                    continue
            candidate_values.append({"seed": seed, "success_rate": summary.get("success_rate", 0.0), "mean_final_distance": summary.get("mean_final_distance"), "earliest_checkpoint": min(checkpoint_steps) if checkpoint_steps else None})
        success = float(np.mean([v["success_rate"] for v in candidate_values]))
        distances = [v["mean_final_distance"] for v in candidate_values if v["mean_final_distance"] is not None]
        checkpoint_values = [v["earliest_checkpoint"] for v in candidate_values if v["earliest_checkpoint"] is not None]
        scores[str(sector)] = {"success_rate": success, "mean_final_distance": float(np.mean(distances)) if distances else None, "earliest_checkpoint": min(checkpoint_values) if checkpoint_values else None, "runs": candidate_values}
    selected = min(candidates, key=lambda sector: (-scores[str(sector)]["success_rate"], scores[str(sector)]["mean_final_distance"] if scores[str(sector)]["mean_final_distance"] is not None else float("inf"), scores[str(sector)]["earliest_checkpoint"] if scores[str(sector)]["earliest_checkpoint"] is not None else float("inf"), sector))
    payload = {"version": VERSION, "method": "b2_fixed", "status": "completed", "candidate_sectors": candidates, "candidate_angles_deg": [float((s + 0.5) * 18.0) for s in candidates], "seeds": args.seeds[:2], "timesteps_per_candidate": calibration_args.timesteps, "selected_sector": selected, "selection_rule": "validation success_rate, then mean_final_distance, then earliest checkpoint", "scores": scores}
    atomic_write_json(path, payload)
    print(json.dumps({"status": "completed", "selected_sector": selected, "b1_selected_config": b1_selected["config"], "b2_path": portable_path(path), "b1_path": portable_path(b1_path)}, ensure_ascii=False, indent=2))

def selected_b1_config(root: Path) -> dict[str, Any]:
    path = root / "calibration_b1.json"
    if path.exists():
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            selected = dict(value.get("selected_config") or {})
            return {str(k): selected[k] for k in ("b1_hit_threshold", "b1_trend_threshold", "b1_contrast_threshold", "b1_search_period", "b1_scan_period") if k in selected}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return {}

def selected_fixed_sector(root: Path) -> int:
    path = root / "calibration_b2.json"
    if path.exists():
        try:
            value = int(json.loads(path.read_text(encoding="utf-8")).get("selected_sector", 5))
            return value if 0 <= value <= 9 else 5
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return 5

def train_command(args: argparse.Namespace, root: Path) -> None:
    for method in args.methods:
        for seed in seeds_for_method(args, method):
            print(f"开始训练：方法={method} 随机种子={seed}")
            train_one(args, root, method, seed, fixed_sector=selected_fixed_sector(root) if method == "b2_fixed" else 5)


def evaluate_command(args: argparse.Namespace, root: Path) -> None:
    splits = ("validation", "test") if args.split == "all" else (args.split,)
    for split in splits:
        for method in args.methods:
            for seed in seeds_for_method(args, method):
                print(f"开始评估：数据集={split} 方法={method} 随机种子={seed}")
                evaluate_method(args, root, method, seed, split, fixed_sector=selected_fixed_sector(root) if method == "b2_fixed" else 5)


def summarize(args: argparse.Namespace, root: Path) -> None:
    all_rows: list[dict[str, Any]] = []
    for method in args.methods:
        for seed in seeds_for_method(args, method):
            for split in ("validation", "test"):
                path = root / method / f"seed_{seed}" / "episodes" / split
                if not path.exists():
                    continue
                for item in path.glob("*.json"):
                    try:
                        row = json.loads(item.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    row.update({"method": method, "training_seed": seed})
                    all_rows.append(row)
    if not all_rows:
        print("暂无已完成的回合结果")
        return
    grouped: dict[str, dict[str, Any]] = {}
    for method in args.methods:
        for split in ("validation", "test"):
            rows = [r for r in all_rows if r["method"] == method and r["split"] == split]
            if not rows:
                continue
            summary = aggregate(rows)
            expected_per_seed = len(read_scenarios(root, split, args.eval_limit))
            expected = expected_per_seed * len(seeds_for_method(args, method))
            summary["expected_episodes"] = expected
            summary["missing_episodes"] = max(0, expected - len(rows))
            summary["coverage"] = len(rows) / max(expected, 1)
            summary["by_training_seed"] = {
                str(seed): aggregate([r for r in rows if int(r["training_seed"]) == int(seed)], include_breakdown=False)
                for seed in seeds_for_method(args, method)
                if any(int(r["training_seed"]) == int(seed) for r in rows)
            }
            statuses = []
            for seed in seeds_for_method(args, method):
                state = root / method / f"seed_{seed}" / "state.json"
                if state.exists():
                    try:
                        statuses.append(json.loads(state.read_text(encoding="utf-8")).get("status", "unknown"))
                    except (OSError, json.JSONDecodeError):
                        statuses.append("corrupt_state")
                else:
                    statuses.append("missing_state")
            summary["task_status_counts"] = {status: statuses.count(status) for status in sorted(set(statuses))}
            grouped[f"{method}:{split}"] = summary
    atomic_write_json(root / "summary.json", grouped)
    with (root / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["method", "split", "training_seed", "policy_seed", "scenario_id", "success", "steps", "final_distance", "path_length", "odor_hit_ratio", "blank_body_spin_ratio", "blank_body_motion_ratio", "blank_whisker_motion_ratio", "reacquisition_events", "reacquisition_per_1000_steps", "mean_reacquisition_time_s", "reward"]
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(all_rows)
    # 与主方法按 (split, training_seed, scenario_id) 配对，保留缺失。
    by_key = {(r["split"], r["training_seed"], r["scenario_id"], r["method"]): r for r in all_rows}
    diff_rows = []
    for method in args.methods:
        if method == "main":
            continue
        candidate_keys = {(r["split"], r["training_seed"], r["scenario_id"]) for r in all_rows if r["method"] == method}
        baseline_keys = {(r["split"], r["training_seed"], r["scenario_id"]) for r in all_rows if r["method"] == "main"}
        for split, training_seed, scenario_id in sorted(candidate_keys | baseline_keys):
            baseline = by_key.get((split, training_seed, scenario_id, "main"))
            candidate = by_key.get((split, training_seed, scenario_id, method))
            pair_status = "complete" if baseline is not None and candidate is not None else "missing_main" if baseline is None else "missing_candidate"
            diff_rows.append({
                "method": method,
                "split": split,
                "training_seed": training_seed,
                "scenario_id": scenario_id,
                "pair_status": pair_status,
                "success_diff": int(candidate["success"]) - int(baseline["success"]) if pair_status == "complete" else None,
                "steps_diff": float(candidate["steps"]) - float(baseline["steps"]) if pair_status == "complete" else None,
                "final_distance_diff": float(candidate["final_distance"]) - float(baseline["final_distance"]) if pair_status == "complete" else None,
                "reward_diff": float(candidate.get("reward", 0.0)) - float(baseline.get("reward", 0.0)) if pair_status == "complete" else None,
            })
    with (root / "paired_differences.csv").open("w", encoding="utf-8-sig", newline="") as f:
        fields = ["method", "split", "training_seed", "scenario_id", "pair_status", "success_diff", "steps_diff", "final_distance_diff", "reward_diff"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(diff_rows)
    lines = [f"# E4 summary ({args.profile})", "", "| Method | Split | Episodes / expected | Coverage | Missing | Success rate | Mean steps | Mean final distance |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in grouped.items():
        method, split = key.split(":", 1)
        lines.append(f"| {method} | {split} | {value['episodes']} / {value.get('expected_episodes', value['episodes'])} | {value.get('coverage', 0.0):.3f} | {value.get('missing_episodes', 0)} | {value.get('success_rate', 0.0):.3f} | {value.get('mean_steps', 0.0):.2f} | {value.get('mean_final_distance', 0.0):.4f} |")
    lines.extend(["", "缺失场景保持为缺失，不计入失败回合。smoke 结果仅用于连通性验证。"])
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"汇总已保存：{portable_path(root / 'summary.md')}")


def status(args: argparse.Namespace, root: Path) -> None:
    result = []
    for method in args.methods:
        for seed in seeds_for_method(args, method):
            p = root / method / f"seed_{seed}" / "state.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                evaluation = {}
                for split in ("validation", "test"):
                    eval_path = root / method / f"seed_{seed}" / f"evaluation_{split}.json"
                    if eval_path.exists():
                        try:
                            eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
                            evaluation[split] = "interrupted" if eval_data.get("status") == "running" else eval_data.get("status")
                        except (OSError, json.JSONDecodeError):
                            evaluation[split] = "corrupt_state"
                result.append({"method": method, "seed": seed, "status": data.get("status"), "actual_timesteps": data.get("actual_timesteps", 0), "evaluation": evaluation})
            else:
                checkpoint = latest_checkpoint(root / method / f"seed_{seed}")
                result.append({"method": method, "seed": seed, "status": "interrupted" if checkpoint else "pending", "actual_timesteps": 0, "checkpoint": portable_path(checkpoint) if checkpoint else None})
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "calibrate", "train", "evaluate", "summarize", "run", "status"), nargs="?", default="run")
    parser.add_argument("--profile", choices=("smoke", "formal"), default="foramal")
    parser.add_argument("--methods", default=None, help="逗号分隔的方法；默认全部 M/B0-B6")
    parser.add_argument("--seeds", default=None, help="逗号分隔训练种子；默认 smoke=1、formal=1..5")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--vec-env-backend", choices=("auto", "dummy", "subproc"), default=None, help="训练环境并行后端；默认 smoke=dummy、formal=subproc")
    parser.add_argument("--checkpoint-rollouts", type=int, default=None, help="PPO 每隔多少个 rollout 保存一次 checkpoint；默认 smoke=2、formal=25")
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--split", choices=("validation", "test", "all"), default="all")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="复用已完成场景和最近 checkpoint")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    configure_output_encoding()
    args = build_parser().parse_args()
    resolve_args(args)
    root = args.output_dir / args.profile
    if args.command == "prepare":
        prepare(args); return
    if not (root / "manifest.json").exists():
        prepare(args)
    if args.command == "status":
        status(args, root); return
    if args.command == "calibrate":
        calibrate_command(args, root)
        return
    if args.command == "train":
        train_command(args, root); return
    if args.command == "evaluate":
        evaluate_command(args, root); return
    if args.command == "summarize":
        summarize(args, root); return
    if args.command == "run":
        if args.dry_run:
            print(json.dumps({"profile": args.profile, "methods": args.methods, "seeds": args.seeds, "method_seeds": {method: seeds_for_method(args, method) for method in args.methods}, "timesteps": args.timesteps, "n_envs": args.n_envs, "vec_env_backend": args.vec_env_backend, "checkpoint_rollouts": args.checkpoint_rollouts}, ensure_ascii=False, indent=2)); return
        calibrate_command(args, root)
        train_command(args, root)
        evaluate_command(args, root)
        summarize(args, root)
        return


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
