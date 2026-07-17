"""评估移动机器人 + 双触须气源搜索 PPO。

复用 `dual_whisker_rl.evaluation.evaluate_policy`（源搜索指标、左右触须浓度差、
reacquisition 等）。环境工厂用 `ObservationHistoryWrapper` 包装，堆叠长度从模型
观测维度反推；默认额外生成首个评估 seed 的静态 PNG 和动态 GIF。
"""

from __future__ import annotations

import argparse
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
from train_whisker_only_ppo import ObservationHistoryWrapper
from visualize_mobile_whisker_env import capture_rollout
from visualize_mobile_whisker_env import render_animation_rollouts
from visualize_mobile_whisker_env import render_static_rollouts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=ROOT / "results" / "models" / "mobile_whisker_transformer_ppo.zip",
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
        args.json_path = (
            ROOT / "results" / "logs" / args.model_path.stem / "eval_metrics.json"
        )
    if args.png_path is None:
        args.png_path = (
            ROOT / "results" / "figures" / f"{args.model_path.stem}_eval.png"
        )
    if args.gif_path is None:
        args.gif_path = (
            ROOT / "results" / "figures" / f"{args.model_path.stem}_eval.gif"
        )
    return args


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


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"模型不存在: {args.model_path}")
    # 显式导入自定义提取器，确保 SB3 反序列化 Transformer 策略时类路径可用。
    _ = TransformerHistoryExtractor
    model = PPO.load(str(args.model_path))

    config: dict = {}
    if args.domain_randomization:
        config["domain_randomization"] = True

    probe_env = MobileWhiskerPuffEnv(config)
    base_dim = int(probe_env.observation_space.shape[0])
    probe_goal_radius = float(probe_env.goal_radius)
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
