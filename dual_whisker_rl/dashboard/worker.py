"""独立训练进程；仅此进程导入 torch 和 PPO。"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys
import time
import traceback

from .files import atomic_json, checkpoint_data, identity, read_json


def apply_source_parameters(args, meta):
    """只恢复训练入口认识的参数；特征布局始终读取冻结的环境配置。"""
    args.history_length = meta.get("history_length", 20)
    args.temporal_encoder = meta.get("temporal_encoder", "gru")
    encoder_fields = {
        "gru": {"hidden_size": "hidden_size", "n_layers": "layers", "dropout": "dropout", "features_dim": "features_dim"},
        "transformer": {"d_model": "d_model", "n_heads": "heads", "n_layers": "layers",
                        "dim_feedforward": "ff_dim", "dropout": "dropout", "features_dim": "features_dim"},
    }
    for kind, fields in encoder_fields.items():
        for field, argument in fields.items():
            if field in (meta.get(kind) or {}):
                setattr(args, kind + "_" + argument, meta[kind][field])
    for field in ("learning_rate", "n_epochs", "ent_coef", "target_kl"):
        if field in (meta.get("ppo") or {}):
            setattr(args, field, meta["ppo"][field])


def run(directory, validate_only=False):
    spec = read_json(directory / "spec.json")
    if not spec:
        raise ValueError("任务说明不存在")
    task = spec["task"]
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts"))
    import train_mobile_whisker_ppo as trainer
    import torch
    if spec.get("require_gpu", True) and not torch.cuda.is_available():
        raise RuntimeError("GPU不可用，任务未转为CPU训练")
    if validate_only:
        if spec.get("resume_checkpoint"):
            model = trainer.PPO.load(spec["resume_checkpoint"], device="cpu")
            config = trainer.load_config(directory / "config.yaml")
            with trainer.MobileWhiskerPuffEnv(config) as env:
                trainer.validate_model_environment(model, env, task.get("source_meta", {}).get("history_length", 20), resume=True)
        print("CHECKPOINT_PREFLIGHT_OK", flush=True)
        return
    original_argv = sys.argv
    try:
        sys.argv = ["train_mobile_whisker_ppo.py"]
        args = trainer.parse_args()
    finally:
        sys.argv = original_argv
    args.config = directory / "config.yaml"
    args.managed_task_dir = directory
    args.resume_from = Path(spec["resume_checkpoint"]) if spec.get("resume_checkpoint") else None
    meta = task.get("source_meta", {})
    apply_source_parameters(args, meta)
    args.seed = task["seed"]
    args.n_envs, args.n_steps, args.batch_size = task["n_envs"], task["n_steps"], task["batch_size"]
    args.vec_env_backend = "subproc" if args.n_envs > 1 else "dummy"
    args.timesteps = task["target_steps"] - task["current_steps"]
    args.checkpoint_freq = max(1, spec.get("checkpoint_freq", task.get("checkpoint_freq", 100000)))
    args.experiment_name = "managed_" + task["id"][:12] + "_" + directory.name
    args.experiment_keyword = "managed_" + task["id"][:12]
    args.run_name = args.experiment_name
    args.model_path = directory / "final_model.zip"
    args.log_dir = directory / "logs"
    args.checkpoint_dir = directory / "checkpoints"
    args.tensorboard_dir = directory / "tensorboard"
    args.tensorboard_run_dir = args.tensorboard_dir / "run"
    args.log_dir.mkdir(parents=True, exist_ok=False)
    args.tensorboard_run_dir.mkdir(parents=True, exist_ok=False)
    args._output_paths_resolved = True
    trainer.train(args)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    directory = args.attempt_dir.resolve()
    if not args.validate_only:
        atomic_json(directory / "status.json", {"state": "starting", "identity": identity(os.getpid()), "at": time.time()})
    try:
        run(directory, args.validate_only)
    except Exception as exc:
        if not args.validate_only:
            previous = read_json(directory / "status.json", {})
            atomic_json(directory / "status.json", {**previous, "state": "failed", "error": str(exc), "at": time.time()})
        traceback.print_exc()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
