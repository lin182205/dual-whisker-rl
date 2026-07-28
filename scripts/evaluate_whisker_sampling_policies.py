"""触须采样策略对比评估（阶段四）。

在 `WhiskerOnlyPuffEnv`（固定机器人、仅控制左右触须扇区）上，用**信息获取指标**
而非总回报，对比几类采样策略，回答课题核心问题：

    主动触须（PPO / 周期扫描）是否比固定角度、随机采样获得更多有效气味信息？

为保证仿真评估与硬件评估口径一致，指标定义与动作序列生成**直接复用**硬件脚本
`compare_hardware_sampling_modes.py` 的 `compute_metrics` / `make_action_sequence`。
指标从环境的 hardware 观测槽读取（策略看到什么就用什么算）。

对比策略（--policies 可改）：
    ppo              加载训练好的 PPO 模型（维护与训练一致的历史堆叠）
    periodic         往返三角波周期扫描
    random_neighbor  每步只动 ±1 的随机扫描（贴合舵机、比全随机公平）
    random_any       每步任意扇区随机
    fixed_<k>        左右都固定在扇区 k（多个 k 组成公平的固定角度基线）

所有策略在**同一批 plume seed** 上评估，保证对比公平。

用法示例：
    python scripts\\evaluate_whisker_sampling_policies.py --episodes 20 --steps 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs.whisker_only_env import WhiskerOnlyPuffEnv
from dual_whisker_rl.paths import portable_path
from dual_whisker_rl.paths import resolve_path_args

# 复用硬件脚本的指标与动作序列定义，保证 sim / 硬件评估口径一致。
from compare_hardware_sampling_modes import compute_metrics
from compare_hardware_sampling_modes import make_action_sequence


# hardware 观测向量各槽位（与 WhiskerObservationBuilder.FIELD_NAMES 对应）。
OBS_LEFT_SMOOTH = 2
OBS_RIGHT_SMOOTH = 3
OBS_LEFT_TREND = 4
OBS_RIGHT_TREND = 5
OBS_SMOOTH_DIFF = 6

DEFAULT_POLICIES = (
    "ppo,random_neighbor,periodic,fixed_1,fixed_3,fixed_5,fixed_7,fixed_9"
)
# 表格/柱状图展示的核心信息获取指标。
HEADLINE_METRICS = [
    "odor_hit_rate",
    "mean_max_smooth",
    "mean_abs_smooth_diff",
    "mean_abs_trend_sum",
    "positive_trend_rate",
    "high_information_rate",
    "mean_reacquisition_time_s",
    "sector_entropy_bits",
    "mean_return",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("results/models/whisker_only_ppo.zip"))
    parser.add_argument("--policies", type=str, default=DEFAULT_POLICIES)
    parser.add_argument("--episodes", type=int, default=20, help="评估回合数（= plume seed 数）。")
    parser.add_argument("--steps", type=int, default=300, help="每回合步数。")
    parser.add_argument("--base-seed", type=int, default=10_000)
    parser.add_argument("--init-pose-mode", type=str, default="plume_edge")
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--dwell-steps", type=int, default=2, help="periodic 每扇区停留步数。")
    parser.add_argument("--stochastic", action="store_true", help="PPO 采样动作（默认 deterministic）。")
    # 信息获取阈值（与硬件脚本默认一致）。
    parser.add_argument("--hit-threshold", type=float, default=0.2)
    parser.add_argument("--diff-threshold", type=float, default=0.2)
    parser.add_argument("--trend-threshold", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures/whisker_policy_eval"))
    return resolve_path_args(parser.parse_args(), "model_path", "output_dir")


def resolve_policy(name: str) -> dict:
    """把策略名解析为可执行规格。"""
    if name == "ppo":
        return {"name": name, "type": "ppo"}
    if name == "periodic":
        return {"name": name, "type": "open", "mode": "periodic_scan"}
    if name == "random_neighbor":
        return {"name": name, "type": "open", "mode": "random_scan", "random_mode": "neighbor"}
    if name == "random_any":
        return {"name": name, "type": "open", "mode": "random_scan", "random_mode": "any"}
    if name.startswith("fixed_"):
        return {"name": name, "type": "open", "mode": "fixed_angle", "fixed_sector": int(name.split("_")[1])}
    raise ValueError(f"未知策略: {name}")


def build_env(args: argparse.Namespace) -> WhiskerOnlyPuffEnv:
    return WhiskerOnlyPuffEnv(
        {
            "observation_mode": "hardware",  # 指标依赖 hardware 观测槽
            "init_pose_mode": args.init_pose_mode,
            "domain_randomization": args.domain_randomization,
            "max_steps": args.steps,
        }
    )


def open_loop_sequence(spec: dict, steps: int, dwell_steps: int, seed: int, sector_count: int) -> list[tuple[int, int]]:
    """用硬件脚本的 make_action_sequence 生成开环动作序列（fixed/periodic/random）。"""
    ns = SimpleNamespace(
        mode=spec["mode"],
        steps=steps,
        fixed_sector=spec.get("fixed_sector", 5),
        dwell_steps=dwell_steps,
        random_mode=spec.get("random_mode", "neighbor"),
        seed=seed,  # 每回合不同 seed，使随机策略在回合间变化但可复现
    )
    return make_action_sequence(ns, sector_count)


def rollout_episode(
    env: WhiskerOnlyPuffEnv,
    spec: dict,
    seed: int,
    steps: int,
    dwell_steps: int,
    model: object | None,
    history_length: int,
    deterministic: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """跑一回合，返回逐步观测矩阵、左右扇区、时间轴、总回报。"""
    from collections import deque

    obs, _ = env.reset(seed=seed)
    obs = np.asarray(obs, dtype=np.float32)

    if spec["type"] == "ppo":
        history: deque[np.ndarray] = deque([obs.copy()] * history_length, maxlen=history_length)
        action_seq = None
    else:
        action_seq = open_loop_sequence(spec, steps, dwell_steps, seed, env.whiskers.sector_count)

    obs_rows: list[np.ndarray] = []
    left_sectors: list[int] = []
    right_sectors: list[int] = []
    total_reward = 0.0

    for t in range(steps):
        if spec["type"] == "ppo":
            stacked = np.concatenate(tuple(history)).astype(np.float32)
            action, _ = model.predict(stacked, deterministic=deterministic)
            action = [int(action[0]), int(action[1])]
        else:
            action = list(action_seq[t])

        obs, reward, _, truncated, info = env.step(action)
        obs = np.asarray(obs, dtype=np.float32)
        if spec["type"] == "ppo":
            history.append(obs.copy())
        obs_rows.append(obs)
        left_sectors.append(int(info["left_sector"]))
        right_sectors.append(int(info["right_sector"]))
        total_reward += float(reward)
        if truncated:
            break

    obs_mat = np.asarray(obs_rows, dtype=np.float32)
    sectors = np.asarray(left_sectors + right_sectors, dtype=np.int64)
    dt = float(env.plume.dt)
    t_axis = np.arange(obs_mat.shape[0], dtype=np.float32) * dt
    return obs_mat, sectors, t_axis, total_reward


def episode_metrics(
    obs_mat: np.ndarray,
    sectors: np.ndarray,
    t_axis: np.ndarray,
    total_reward: float,
    metric_args: SimpleNamespace,
    sector_count: int,
) -> dict[str, float]:
    """从逐步观测构造 features，复用硬件脚本的 compute_metrics。"""
    features = {
        "left_norm_smooth": obs_mat[:, OBS_LEFT_SMOOTH],
        "right_norm_smooth": obs_mat[:, OBS_RIGHT_SMOOTH],
        "left_norm_trend": obs_mat[:, OBS_LEFT_TREND],
        "right_norm_trend": obs_mat[:, OBS_RIGHT_TREND],
        "norm_smooth_diff": obs_mat[:, OBS_SMOOTH_DIFF],
    }
    m = compute_metrics(t_axis, sectors, features, metric_args, sector_count)
    m["mean_return"] = float(total_reward)
    return m


def aggregate(per_episode: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """把多回合指标聚合为 mean/std。"""
    keys = per_episode[0].keys()
    out: dict[str, dict[str, float]] = {}
    for k in keys:
        vals = np.array([ep[k] for ep in per_episode], dtype=np.float64)
        out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    return out


def save_summary(path: Path, results: dict[str, dict], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "| policy | " + " | ".join(HEADLINE_METRICS) + " |"
    sep = "|---|" + "|".join(["---:"] * len(HEADLINE_METRICS)) + "|"
    lines = [
        "# 触须采样策略对比（信息获取指标）",
        "",
        f"- init_pose_mode: {args.init_pose_mode}   domain_randomization: {args.domain_randomization}",
        f"- episodes: {args.episodes}   steps: {args.steps}   同一批 seed 公平对比",
        f"- 阈值: hit={args.hit_threshold} diff={args.diff_threshold} trend={args.trend_threshold}",
        "",
        header,
        sep,
    ]
    for policy, agg in results.items():
        cells = [f"{agg[m]['mean']:.3f}" for m in HEADLINE_METRICS]
        lines.append(f"| {policy} | " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_barchart(path: Path, results: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = ["odor_hit_rate", "mean_max_smooth", "mean_abs_smooth_diff", "mean_abs_trend_sum"]
    policies = list(results.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, metric in zip(axes.ravel(), metrics):
        means = [results[p][metric]["mean"] for p in policies]
        stds = [results[p][metric]["std"] for p in policies]
        colors = ["#e45756" if p == "ppo" else "#4c78a8" for p in policies]
        ax.bar(range(len(policies)), means, yerr=stds, color=colors, alpha=0.85, capsize=3)
        ax.set_xticks(range(len(policies)))
        ax.set_xticklabels(policies, rotation=45, ha="right", fontsize=8)
        ax.set_title(metric, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Whisker sampling policies — information-acquisition metrics", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    policy_names = [p.strip() for p in args.policies.split(",") if p.strip()]
    specs = [resolve_policy(n) for n in policy_names]
    seeds = [args.base_seed + i for i in range(args.episodes)]
    metric_args = SimpleNamespace(
        hit_threshold=args.hit_threshold,
        diff_threshold=args.diff_threshold,
        trend_threshold=args.trend_threshold,
    )

    # 若要评估 ppo，加载模型并从堆叠维度反推 history 长度。
    model = None
    history_length = 1
    if any(s["type"] == "ppo" for s in specs):
        if not args.model_path.exists():
            raise FileNotFoundError(f"模型不存在: {args.model_path}（去掉 ppo 或先训练）")
        from stable_baselines3 import PPO

        model = PPO.load(str(args.model_path))
        probe = build_env(args)
        base_dim = int(probe.observation_space.shape[0])
        probe.close()
        stacked_dim = int(model.observation_space.shape[0])
        if stacked_dim % base_dim != 0:
            raise ValueError(f"模型观测维度 {stacked_dim} 不是环境维度 {base_dim} 的整数倍")
        history_length = stacked_dim // base_dim
        print(f"loaded ppo: history_length={history_length} deterministic={not args.stochastic}")

    env = build_env(args)
    results: dict[str, dict] = {}
    for spec in specs:
        per_episode = []
        for seed in seeds:
            obs_mat, sectors, t_axis, total_reward = rollout_episode(
                env, spec, seed, args.steps, args.dwell_steps,
                model, history_length, deterministic=not args.stochastic,
            )
            per_episode.append(
                episode_metrics(obs_mat, sectors, t_axis, total_reward, metric_args, env.whiskers.sector_count)
            )
        results[spec["name"]] = aggregate(per_episode)
        agg = results[spec["name"]]
        print(
            f"{spec['name']:16s} hit={agg['odor_hit_rate']['mean']:.3f} "
            f"max_smooth={agg['mean_max_smooth']['mean']:.3f} "
            f"abs_trend={agg['mean_abs_trend_sum']['mean']:.3f} "
            f"contrast={agg['mean_abs_smooth_diff']['mean']:.3f} "
            f"reacq_s={agg['mean_reacquisition_time_s']['mean']:.2f} "
            f"sec_entropy={agg['sector_entropy_bits']['mean']:.2f}"
        )
    env.close()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "policies": policy_names,
            "episodes": args.episodes,
            "steps": args.steps,
            "init_pose_mode": args.init_pose_mode,
            "domain_randomization": args.domain_randomization,
            "thresholds": {"hit": args.hit_threshold, "diff": args.diff_threshold, "trend": args.trend_threshold},
            "model_path": portable_path(args.model_path),
            "history_length": history_length,
        },
        "results": results,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_summary(args.output_dir / "summary.md", results, args)
    save_barchart(args.output_dir / "plots.png", results)
    print(f"saved_dir={args.output_dir}")


if __name__ == "__main__":
    main()
