"""在封存的固定场景库上多进程评估 Mobile PPO 模型。

默认首次运行生成 100 个确定性场景并保存；后续模型复用同一文件。每个 worker
独立加载一次模型和环境，动态领取场景，适合训练完成后使用较多 CPU 核并行复评。
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ProcessPoolExecutor
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import sys
import time
from typing import Any


NUMERIC_THREAD_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
os.environ.update(NUMERIC_THREAD_LIMITS)

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.agents import GRUHistoryExtractor
from dual_whisker_rl.agents import TransformerHistoryExtractor
from dual_whisker_rl.e4_experiments import aggregate
from dual_whisker_rl.e4_experiments import E4Scenario
from dual_whisker_rl.e4_experiments import evaluate_one
from dual_whisker_rl.e4_experiments import ObservationHistory
from dual_whisker_rl.e4_experiments import scenario_hash
from dual_whisker_rl.e4_experiments import sha256_file
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.paths import portable_path
from dual_whisker_rl.paths import project_path
from dual_whisker_rl.paths import resolve_path_args
from train_whisker_only_ppo import load_config


SCENARIO_BANK_VERSION = "mobile-fixed-eval-v1"
_WORKER_MODEL: PPO | None = None
_WORKER_ENV: ObservationHistory | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/models/mobile_whisker_gru_ppo.zip"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mobile_whisker.yaml"))
    parser.add_argument(
        "--scenario-mode",
        choices=("randomized", "fixed"),
        default=None,
        help="覆盖配置中的场景模式；通常与训练保持一致。",
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--scenario-seed", type=int, default=310_000)
    parser.add_argument(
        "--scenario-file",
        type=Path,
        default=None,
        help="默认按有效环境配置、种子和局数生成场景库路径。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="并行评估进程数；14 核服务器建议从 12 开始。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="每个评估进程加载模型的设备；多进程默认使用 cpu。",
    )
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只生成或校验固定场景库，不加载模型。",
    )
    parser.add_argument(
        "--regenerate-scenarios",
        action="store_true",
        help="显式覆盖已有场景库；会改变不同 checkpoint 的评估题目。",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="默认写入 results/evaluation/<模型名>_fixed_<场景数>.json。",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    args.workers = min(args.workers, args.episodes)
    return resolve_path_args(
        args,
        "model_path",
        "config",
        "scenario_file",
        "json_path",
    )


def _config_hash(config: dict[str, Any]) -> str:
    payload = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _heading_quadrant(heading: float) -> int:
    normalized = (float(heading) + 2.0 * math.pi) % (2.0 * math.pi)
    return int(normalized // (math.pi / 2.0)) % 4


def generate_scenarios(
    config: dict[str, Any],
    *,
    count: int,
    base_seed: int,
) -> list[E4Scenario]:
    """从当前移动环境分布采样并封存可重复的场景参数。"""
    env = MobileWhiskerPuffEnv(config)
    scenarios: list[E4Scenario] = []
    try:
        for index in range(count):
            episode_seed = int(base_seed + index)
            env.reset(seed=episode_seed)
            state = env.robot_state
            scenarios.append(
                E4Scenario(
                    scenario_id=f"mobile_fixed_{index:04d}",
                    split="validation",
                    seed=episode_seed,
                    wind_speed=float(env.plume.wind_speed_mean),
                    wind_direction=float(env.plume.wind_direction_base),
                    source_position=(
                        float(env.plume.source_x),
                        float(env.plume.source_y),
                    ),
                    robot_pose=(
                        float(state.x),
                        float(state.y),
                        float(state.heading),
                    ),
                    source_strength=1.0,
                    distance_bin="randomized",
                    crosswind_bin="randomized",
                    heading_quadrant=_heading_quadrant(state.heading),
                    repeat=index,
                )
            )
    finally:
        env.close()
    return scenarios


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def prepare_scenario_bank(
    path: Path,
    config: dict[str, Any],
    *,
    count: int,
    base_seed: int,
    regenerate: bool,
) -> tuple[list[E4Scenario], dict[str, Any]]:
    config_hash = _config_hash(config)
    if path.exists() and not regenerate:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("version") != SCENARIO_BANK_VERSION:
            raise ValueError(f"不支持的场景库版本: {payload.get('version')!r}")
        scenarios = [
            E4Scenario.from_dict(value) for value in payload.get("scenarios", [])
        ]
        if len(scenarios) != count:
            raise ValueError(
                f"场景库包含 {len(scenarios)} 局，但本次要求 {count} 局；"
                "请使用一致的 --episodes，或显式传 --regenerate-scenarios。"
            )
        if payload.get("base_seed") != int(base_seed):
            raise ValueError("场景库的基础随机种子与本次 --scenario-seed 不一致")
        if payload.get("config_hash") != config_hash:
            raise ValueError(
                "场景库的环境配置与本次 --config 不一致；为保证 checkpoint "
                "可比性已拒绝复用。需要新实验时请换一个 --scenario-file。"
            )
        expected_hash = str(payload.get("scenario_hash", ""))
        actual_hash = scenario_hash(scenarios)
        if expected_hash != actual_hash:
            raise ValueError(
                f"场景库摘要不一致: expected={expected_hash}, actual={actual_hash}"
            )
        return scenarios, payload

    scenarios = generate_scenarios(
        config,
        count=count,
        base_seed=base_seed,
    )
    payload = {
        "version": SCENARIO_BANK_VERSION,
        "count": len(scenarios),
        "base_seed": int(base_seed),
        "config_hash": config_hash,
        "scenario_hash": scenario_hash(scenarios),
        "scenarios": [scenario.to_dict() for scenario in scenarios],
    }
    _atomic_write_json(path, payload)
    return scenarios, payload


def _init_worker(
    model_path: str,
    config: dict[str, Any],
    history_length: int,
    device: str,
) -> None:
    global _WORKER_MODEL, _WORKER_ENV
    # 显式引用自定义类，保证旧 checkpoint 反序列化时模块已加载。
    _ = (GRUHistoryExtractor, TransformerHistoryExtractor)
    _WORKER_MODEL = PPO.load(model_path, device=device)
    worker_config = dict(config)
    worker_config["record_trajectory"] = True
    _WORKER_ENV = ObservationHistory(
        MobileWhiskerPuffEnv(worker_config),
        history_length=history_length,
    )


def _evaluate_scenario(scenario_data: dict[str, Any]) -> dict[str, Any]:
    if _WORKER_MODEL is None or _WORKER_ENV is None:
        raise RuntimeError("评估 worker 尚未初始化")
    scenario = E4Scenario.from_dict(scenario_data)
    return evaluate_one(
        _WORKER_MODEL,
        _WORKER_ENV,
        scenario,
        deterministic=True,
    )


def infer_history_length(
    model_path: Path,
    config: dict[str, Any],
    device: str,
) -> tuple[int, int, int, str]:
    _ = (GRUHistoryExtractor, TransformerHistoryExtractor)
    model = PPO.load(str(model_path), device=device)
    probe_env = MobileWhiskerPuffEnv(config)
    try:
        base_dim = int(probe_env.observation_space.shape[0])
    finally:
        probe_env.close()
    stacked_dim = int(model.observation_space.shape[0])
    if stacked_dim % base_dim != 0:
        raise ValueError(
            f"模型观测维度 {stacked_dim} 不是环境单帧维度 {base_dim} 的整数倍"
        )
    extractor_name = type(model.policy.features_extractor).__name__
    return stacked_dim // base_dim, base_dim, stacked_dim, extractor_name


def evaluate_parallel(
    model_path: Path,
    config: dict[str, Any],
    scenarios: list[E4Scenario],
    *,
    history_length: int,
    workers: int,
    device: str,
) -> tuple[list[dict[str, Any]], float]:
    started_at = time.perf_counter()
    rows: list[dict[str, Any]] = []
    progress_interval = max(1, len(scenarios) // 20)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        initializer=_init_worker,
        initargs=(str(model_path), config, history_length, device),
    ) as executor:
        futures = {
            executor.submit(_evaluate_scenario, scenario.to_dict()): scenario
            for scenario in scenarios
        }
        for future in as_completed(futures):
            scenario = futures[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"固定场景评估失败: {scenario.scenario_id}"
                ) from exc
            completed = len(rows)
            if completed % progress_interval == 0 or completed == len(scenarios):
                elapsed = time.perf_counter() - started_at
                remaining_seconds = (
                    elapsed / completed * (len(scenarios) - completed)
                )
                print(
                    f"[固定场景评估] {completed}/{len(scenarios)} "
                    f"已用={elapsed / 60.0:.2f}分钟 "
                    f"预计剩余={remaining_seconds / 60.0:.2f}分钟",
                    flush=True,
                )
    rows.sort(key=lambda row: str(row["scenario_id"]))
    return rows, time.perf_counter() - started_at


def main() -> None:
    multiprocessing.freeze_support()
    args = parse_args()
    config = load_config(args.config)
    if args.scenario_mode is not None:
        config["scenario_mode"] = args.scenario_mode
    if args.domain_randomization:
        config["domain_randomization"] = True
    config_id = _config_hash(config)[:12]
    if args.scenario_file is None:
        args.scenario_file = project_path(
            Path("results/evaluation")
            / f"mobile_fixed_{config_id}_seed{args.scenario_seed}_{args.episodes}.json"
        )
    if args.json_path is None:
        args.json_path = project_path(
            Path("results/evaluation")
            / f"{args.model_path.stem}_fixed_{config_id}_{args.episodes}.json"
        )

    scenarios, bank = prepare_scenario_bank(
        args.scenario_file,
        config,
        count=args.episodes,
        base_seed=args.scenario_seed,
        regenerate=args.regenerate_scenarios,
    )
    print(
        f"fixed_scenarios={len(scenarios)} "
        f"scenario_hash={bank['scenario_hash']} "
        f"scenario_file={portable_path(args.scenario_file)}",
        flush=True,
    )
    if args.prepare_only:
        print("场景库已准备完成；未加载模型。", flush=True)
        return
    if not args.model_path.exists():
        raise FileNotFoundError(f"模型不存在: {args.model_path}")

    history_length, base_dim, stacked_dim, extractor_name = infer_history_length(
        args.model_path,
        config,
        args.device,
    )
    print(
        f"workers={args.workers} device={args.device} "
        f"history_length={history_length} extractor={extractor_name}",
        flush=True,
    )
    rows, elapsed_seconds = evaluate_parallel(
        args.model_path,
        config,
        scenarios,
        history_length=history_length,
        workers=args.workers,
        device=args.device,
    )
    metrics = aggregate(rows)
    result = {
        "version": SCENARIO_BANK_VERSION,
        "model_path": portable_path(args.model_path),
        "environment_config_path": portable_path(args.config),
        "environment_config": config,
        "model_sha256": sha256_file(args.model_path),
        "scenario_file": portable_path(args.scenario_file),
        "scenario_hash": bank["scenario_hash"],
        "episodes": len(scenarios),
        "workers": args.workers,
        "device": args.device,
        "deterministic_policy": True,
        "elapsed_seconds": elapsed_seconds,
        "observation": {
            "base_dim": base_dim,
            "stacked_dim": stacked_dim,
            "history_length": history_length,
            "features_extractor": extractor_name,
        },
        "metrics": metrics,
        "episodes_detail": rows,
    }
    _atomic_write_json(args.json_path, result)
    print(
        f"success_rate={metrics.get('success_rate', 0.0):.4f} "
        f"mean_final_distance={metrics.get('mean_final_distance', 0.0):.4f} "
        f"mean_steps={metrics.get('mean_steps', 0.0):.2f}",
        flush=True,
    )
    print(f"saved_metrics={portable_path(args.json_path)}", flush=True)


if __name__ == "__main__":
    main()
