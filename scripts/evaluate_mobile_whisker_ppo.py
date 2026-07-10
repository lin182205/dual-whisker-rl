"""评估移动机器人 + 双触须气源搜索 PPO。

复用 `dual_whisker_rl.evaluation.evaluate_policy`（源搜索指标：success_rate、
mean_final_distance、path_length、odor_hits、reacquisition 等）。环境工厂用
`ObservationHistoryWrapper` 包装，堆叠长度从模型观测维度反推。
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

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.evaluation import evaluate_policy
from train_whisker_only_ppo import ObservationHistoryWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "results" / "models" / "mobile_whisker_ppo.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--domain-randomization", action="store_true")
    parser.add_argument("--json-path", type=Path, default=ROOT / "results" / "logs" / "mobile_whisker_ppo" / "eval_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"模型不存在: {args.model_path}")
    model = PPO.load(str(args.model_path))

    config: dict = {}
    if args.domain_randomization:
        config["domain_randomization"] = True

    base_dim = int(MobileWhiskerPuffEnv(config).observation_space.shape[0])
    stacked_dim = int(model.observation_space.shape[0])
    if stacked_dim % base_dim != 0:
        raise ValueError(f"模型观测维度 {stacked_dim} 不是环境维度 {base_dim} 的整数倍")
    history_length = stacked_dim // base_dim
    print(f"history_length={history_length} (base={base_dim}, stacked={stacked_dim})")

    def env_factory():
        return ObservationHistoryWrapper(MobileWhiskerPuffEnv(config), history_length=history_length)

    metrics = evaluate_policy(
        model,
        env_factory,
        episodes=args.episodes,
        seed=args.seed,
        deterministic=True,
    )

    args.json_path.parent.mkdir(parents=True, exist_ok=True)
    args.json_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"saved_metrics={args.json_path}")


if __name__ == "__main__":
    main()
