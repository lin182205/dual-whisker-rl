"""评估移动机器人 + 双触须气源搜索 PPO。

复用 `dual_whisker_rl.evaluation.evaluate_policy`（源搜索指标、左右触须浓度差、
reacquisition 等）。环境工厂用 `ObservationHistoryWrapper` 包装，堆叠长度从模型
观测维度反推；默认选择首个成功和首个失败 episode 生成 PNG/GIF，并单独输出
这些可视化轨迹的奖励分解 CSV。若评估结果只有一类，则只输出一个 episode。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.agents import TransformerHistoryExtractor
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.evaluation import evaluate_policy
from dual_whisker_rl.paths import resolve_path_args
from train_whisker_only_ppo import ObservationHistoryWrapper
from train_whisker_only_ppo import load_config
from visualize_mobile_whisker_env import capture_rollout
from visualize_mobile_whisker_env import render_animation_rollouts
from visualize_mobile_whisker_env import render_static_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/models/mobile_whisker_transformer_ppo.zip"),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--scenario-mode",
        choices=("randomized", "fixed"),
        default=None,
        help="覆盖配置中的场景初始化模式；默认使用 randomized。",
    )
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument(
        "--json-path",
        type=Path,
        default=None,
        help="默认按模型文件名写入对应日志目录。",
    )
    parser.add_argument(
        "--png-path",
        type=Path,
        default=None,
        help="默认按模型文件名写入评估 PNG。",
    )
    parser.add_argument(
        "--gif-path",
        type=Path,
        default=None,
        help="默认按模型文件名写入评估 GIF。",
    )
    parser.add_argument(
        "--reward-breakdown-path",
        type=Path,
        default=None,
        help="可视化 episode 的奖励分解 CSV；默认写入对应日志目录。",
    )
    parser.add_argument("--visualization-steps", type=int, default=200)
    parser.add_argument("--visualization-resolution", type=int, default=100)
    parser.add_argument("--animation-stride", type=int, default=1)
    parser.add_argument("--animation-fps", type=int, default=6)
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="只计算 JSON 指标，不生成 PNG/GIF。",
    )
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.visualization_steps < 1:
        parser.error("--visualization-steps must be at least 1")
    if args.visualization_resolution < 2:
        parser.error("--visualization-resolution must be at least 2")
    if args.animation_stride < 1:
        parser.error("--animation-stride must be at least 1")
    if args.animation_fps < 1:
        parser.error("--animation-fps must be at least 1")

    if args.json_path is None:
        args.json_path = Path("results/logs") / args.model_path.stem / "eval_metrics.json"
    if args.png_path is None:
        args.png_path = Path("results/figures") / f"{args.model_path.stem}_eval.png"
    if args.gif_path is None:
        args.gif_path = Path("results/figures") / f"{args.model_path.stem}_eval.gif"
    if args.reward_breakdown_path is None:
        args.reward_breakdown_path = (
            args.json_path.parent / "visualized_reward_breakdown.csv"
        )
    return resolve_path_args(
        args,
        "model_path",
        "config",
        "json_path",
        "png_path",
        "gif_path",
        "reward_breakdown_path",
    )


def select_visualization_episode_indices(
    trajectories: list[list[dict]],
    goal_radius: float,
) -> list[tuple[str, int]]:
    """选择首个成功和首个失败 episode；单一结果时只返回一个。"""
    first_success = None
    first_failure = None
    for episode_index, trajectory in enumerate(trajectories):
        succeeded = bool(
            trajectory
            and float(trajectory[-1]["distance_to_source"]) <= goal_radius
        )
        if succeeded and first_success is None:
            first_success = episode_index
        if not succeeded and first_failure is None:
            first_failure = episode_index
        if first_success is not None and first_failure is not None:
            break

    selected: list[tuple[str, int]] = []
    if first_success is not None:
        selected.append(("success", first_success))
    if first_failure is not None:
        selected.append(("failure", first_failure))
    return selected


def write_reward_breakdown(
    trajectories: list[list[dict]],
    selected_episodes: list[tuple[str, int]],
    *,
    base_seed: int,
    path: Path,
) -> Path:
    """写入机器可读 CSV，并生成便于直接查看的定宽文本表。"""
    component_order = list(MobileWhiskerPuffEnv.REWARD_COMPONENT_NAMES)
    rows = []
    for result, episode_index in selected_episodes:
        trajectory = trajectories[episode_index]
        component_sums = {name: 0.0 for name in component_order}
        for step_index, step in enumerate(trajectory):
            components = step.get("reward_components")
            if not isinstance(components, dict):
                raise RuntimeError(
                    "Evaluation trajectory is missing reward_components at "
                    f"episode={episode_index}, step={step_index}"
                )
            for name in component_order:
                component_sums[name] += float(components.get(name, 0.0))

        reward_total = float(sum(float(step["reward"]) for step in trajectory))
        component_total = float(sum(component_sums.values()))
        final_components = trajectory[-1]["reward_components"]
        if result == "success":
            termination = "success"
        elif float(final_components.get("out_of_bounds_penalty", 0.0)) < 0.0:
            termination = "out_of_bounds"
        else:
            termination = "timeout"
        row = {
            "result": result,
            "termination": termination,
            "episode_number": episode_index + 1,
            "episode_index": episode_index,
            "seed": base_seed + episode_index,
            "steps": len(trajectory),
            "final_distance": float(trajectory[-1]["distance_to_source"]),
            **component_sums,
            "component_total": component_total,
            "reward_total": reward_total,
            "reward_mean_per_step": reward_total / max(len(trajectory), 1),
            "reconstruction_error": reward_total - component_total,
        }
        rows.append(row)

    fieldnames = [
        "result",
        "termination",
        "episode_number",
        "episode_index",
        "seed",
        "steps",
        "final_distance",
        *component_order,
        "component_total",
        "reward_total",
        "reward_mean_per_step",
        "reconstruction_error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    aligned_path = path.with_suffix(".txt")
    if aligned_path == path:
        aligned_path = path.with_name(f"{path.stem}_aligned.txt")
    write_aligned_reward_breakdown(rows, component_order, aligned_path)
    return aligned_path


def write_aligned_reward_breakdown(
    rows: list[dict],
    component_order: list[str],
    path: Path,
) -> None:
    """将奖励名称左对齐、累计值右对齐写入定宽文本文件。"""
    summary_fields = (
        "component_total",
        "reward_total",
        "reward_mean_per_step",
        "reconstruction_error",
    )
    label_width = max(
        28,
        *(len(name) for name in component_order),
        *(len(name) for name in summary_fields),
    )
    value_width = 18
    table_width = label_width + 2 + value_width
    separator = f"{'-' * label_width}  {'-' * value_width}"
    lines = ["Visualized reward breakdown", "=" * table_width]

    for row_index, row in enumerate(rows):
        if row_index:
            lines.extend(("", "=" * table_width))
        lines.extend(
            (
                "",
                (
                    f"Episode {int(float(row['episode_number']))} | "
                    f"result={row['result']} | termination={row['termination']} | "
                    f"seed={int(float(row['seed']))} | steps={int(float(row['steps']))}"
                ),
                f"Final distance: {float(row['final_distance']):.6f}",
                "",
                f"{'reward_component':<{label_width}}  {'cumulative_value':>{value_width}}",
                separator,
            )
        )
        for component_name in component_order:
            lines.append(
                f"{component_name:<{label_width}}  "
                f"{float(row[component_name]):>{value_width}.6f}"
            )
        lines.extend(
            (
                separator,
                f"{'component_total':<{label_width}}  "
                f"{float(row['component_total']):>{value_width}.6f}",
                f"{'reward_total':<{label_width}}  "
                f"{float(row['reward_total']):>{value_width}.6f}",
                f"{'reward_mean_per_step':<{label_width}}  "
                f"{float(row['reward_mean_per_step']):>{value_width}.6f}",
                f"{'reconstruction_error':<{label_width}}  "
                f"{float(row['reconstruction_error']):>{value_width}.6e}",
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"模型不存在: {args.model_path}")
    # 显式导入自定义提取器，确保 SB3 反序列化 Transformer 策略时类路径可用。
    _ = TransformerHistoryExtractor
    model = PPO.load(str(args.model_path))

    config = load_config(args.config)
    if args.scenario_mode is not None:
        config["scenario_mode"] = args.scenario_mode
    if args.domain_randomization:
        config["domain_randomization"] = True

    probe_env = MobileWhiskerPuffEnv(config)
    base_dim = int(probe_env.observation_space.shape[0])
    probe_goal_radius = float(probe_env.goal_radius)
    scenario = probe_env.scenario_metadata()
    probe_env.close()
    stacked_dim = int(model.observation_space.shape[0])
    if stacked_dim % base_dim != 0:
        raise ValueError(f"模型观测维度 {stacked_dim} 不是环境维度 {base_dim} 的整数倍")
    history_length = stacked_dim // base_dim
    print(f"history_length={history_length} (base={base_dim}, stacked={stacked_dim})")
    print(
        "features_extractor="
        f"{type(model.policy.features_extractor).__name__}"
    )

    def env_factory():
        return ObservationHistoryWrapper(MobileWhiskerPuffEnv(config), history_length=history_length)

    metrics, trajectories = evaluate_policy(
        model,
        env_factory,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=True,
    )
    metrics["scenario"] = scenario

    selected_episodes = select_visualization_episode_indices(
        trajectories,
        goal_radius=probe_goal_radius,
    )
    if not args.no_visualization:
        metrics["visualized_episodes"] = [
            {
                "result": result,
                "episode_index": episode_index,
                "seed": args.seed + episode_index,
            }
            for result, episode_index in selected_episodes
        ]

    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved_metrics={args.json_path}")

    if not args.no_visualization:
        aligned_reward_breakdown_path = write_reward_breakdown(
            trajectories,
            selected_episodes,
            base_seed=args.seed,
            path=args.reward_breakdown_path,
        )
        print(f"saved_reward_breakdown={args.reward_breakdown_path}")
        print(f"saved_reward_breakdown_table={aligned_reward_breakdown_path}")
        visualization_rollouts = []
        for result, episode_index in selected_episodes:
            episode_seed = args.seed + episode_index
            evaluation_steps = len(trajectories[episode_index])
            visualization_env = MobileWhiskerPuffEnv(config)
            try:
                visualization_data = capture_rollout(
                    visualization_env,
                    model,
                    history_length,
                    max(args.visualization_steps, evaluation_steps),
                    args.visualization_resolution,
                    episode_seed,
                    deterministic=True,
                )
            finally:
                visualization_env.close()

            reproduced_success = visualization_data["termination"] == "reached_goal"
            expected_success = result == "success"
            if reproduced_success != expected_success:
                raise RuntimeError(
                    "Selected evaluation episode could not be reproduced for visualization: "
                    f"episode={episode_index}, seed={episode_seed}, expected={result}, "
                    f"got={visualization_data['termination']}"
                )
            label = (
                f"{result.capitalize()} | episode {episode_index + 1} | "
                f"seed {episode_seed}"
            )
            visualization_rollouts.append((label, visualization_data))
            print(
                f"selected_{result}_episode={episode_index + 1} "
                f"seed={episode_seed} steps={evaluation_steps}"
            )

        title = f"Mobile Whisker PPO Evaluation ({args.model_path.stem})"
        render_static_rollouts(visualization_rollouts, args.png_path, title)
        render_animation_rollouts(
            visualization_rollouts,
            args.gif_path,
            title,
            args.animation_stride,
            fps=args.animation_fps,
        )
        print(f"saved_png={args.png_path}")
        print(f"saved_gif={args.gif_path}")


if __name__ == "__main__":
    main()
