"""空间记忆 A/B/C/D 预实验的封存场景、CPU 复评与晋级报告。

本入口不启动训练。每个模型分别执行确定性策略和逐回合固定随机种子的
随机策略；两类结果各自保存，不与训练窗口统计混合。
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any
import zipfile

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

ROOT = Path(__file__).resolve().parents[1]
for _directory in (ROOT, ROOT / "scripts"):
    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))

import numpy as np
import yaml

from dual_whisker_rl.e4_experiments import E4Scenario, evaluate_one, scenario_hash, sha256_file
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.mobile_contract import validate_model_environment
from dual_whisker_rl.paths import portable_path, project_path
from evaluate_mobile_fixed_scenarios import generate_scenarios

PROTOCOL_VERSION = "spatial-exploration-pilot-v1"
SPLITS = {"development": (64, 420_000), "heldout": (128, 520_000)}
CHECKPOINTS = (250_000, 500_000, 1_000_000)
HORIZON = 1200
HISTORY_LENGTH = 20
# 只允许这四个已审查的非物理差异；其它配置（包括未知新键）都进入摘要。
VARIANT_FIELDS = ("spatial_memory_observation", "exploration_bonus",
                  "blank_revisit_penalty", "blank_loop_penalty")
VARIANTS = {
    "A": ("masked", 0.0, 0.0, 0.006),
    "B": ("full", 0.0, 0.0, 0.006),
    "C": ("full", 0.02, 0.0, 0.006),
    "D": ("full", 0.02, 0.006, 0.0),
}
LIMITATION = ("单个训练种子11的探索性筛选，不能据此确认稳定提升，"
              "也不能归因于主动触须或触须加长；四组触须均为22.5厘米。")


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("配置必须为YAML字典")
    return config


def validate_common(config: dict[str, Any]) -> None:
    expected = {"feature_layout": "wind_motion_spatial_memory_v2", "motion_mode": "hybrid",
                "map_layout": "small_office_v1", "max_steps": HORIZON,
                "whisker_length": 0.225, "exploration_budget": 0.5}
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"空间筛选协议要求 {key}={value!r}")


def validate_variant(config: dict[str, Any], variant: str) -> None:
    validate_common(config)
    for key, expected in zip(VARIANT_FIELDS, VARIANTS[variant]):
        if config.get(key) != expected:
            raise ValueError(f"{variant}组配置要求 {key}={expected!r}")


def canonical_config(config: dict[str, Any]) -> dict[str, Any]:
    """用A组非物理设置生成唯一题库；不忽略任何物理、阈值或地图配置。"""
    validate_common(config)
    result = copy.deepcopy(config)
    result.pop("seed", None)
    result.pop("record_trajectory", None)
    result.update(dict(zip(VARIANT_FIELDS, VARIANTS["A"])))
    return result


def physical_summary(config: dict[str, Any]) -> dict[str, Any]:
    canonical = canonical_config(config)
    with MobileWhiskerPuffEnv(canonical) as env:
        return {"canonical_config": canonical, "map_sha256": env.room_navigation.map.digest,
                "dt_s": float(env.dt), "hit_threshold": float(env.hit_threshold),
                "goal_radius_m": float(env.goal_radius), "whisker_length_m": float(env.whiskers.length)}


def bank_path(output_dir: Path, split: str) -> Path:
    return output_dir / "scenes" / f"{split}.json"


def scene_bank(output_dir: Path, config: dict[str, Any], split: str,
               *, create: bool = False) -> tuple[list[E4Scenario], dict[str, Any]]:
    """相同物理摘要复用题库；内容损坏、修改物理或不同协议均拒绝复用。"""
    path = bank_path(output_dir, split)
    count, seed = SPLITS[split]
    physical = physical_summary(config)
    signature = {"protocol_version": PROTOCOL_VERSION, "split": split, "count": count,
                 "base_seed": seed, "physical_sha256": digest(physical), "physical_summary": physical}
    if path.exists():
        payload = read_json(path)
        if any(payload.get(key) != value for key, value in signature.items()):
            raise ValueError(f"封存题库的协议或物理配置不一致，请使用新的输出目录：{path}")
        scenes = [E4Scenario.from_dict(row) for row in payload["scenes"]]
        if (len(scenes) != count or payload.get("scene_sha256") != scenario_hash(scenes)
                or [scene.seed for scene in scenes] != list(range(seed, seed + count))
                or any(scene.split != split for scene in scenes)):
            raise ValueError(f"封存题库内容或场景摘要不一致：{path}")
        return scenes, payload
    if not create:
        raise FileNotFoundError(f"请先运行 prepare 封存题库：{path}")
    scenes = [replace(scene, split=split, scenario_id=f"{split}_{scene.seed}") for scene in
              generate_scenarios(physical["canonical_config"], count=count, base_seed=seed)]
    payload = {**signature, "scene_sha256": scenario_hash(scenes),
               "scenes": [scene.to_dict() for scene in scenes]}
    write_json(path, payload)
    return scenes, payload


def episode_action_seed(scene_seed: int) -> int:
    """相同场景跨模型共用采样种子，独立于模型文件和评估先后顺序。"""
    return int.from_bytes(hashlib.sha256(
        f"{PROTOCOL_VERSION}/action/{scene_seed}".encode()).digest()[:4], "big")


def seed_action_sampling(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def trajectory_metrics(rows: list[dict[str, Any]], *, dt: float) -> dict[str, Any]:
    """首次命中以1200步截尾，未命中及早停失败都保留在均值分母中。"""
    if not rows:
        raise ValueError("评估未记录轨迹，不能将缺失指标补零")
    required = ("odor_hit", "is_success", "collision", "novel_cells", "blank_loop",
                "blank_revisit_active", "reacquisition_event", "reward_components")
    for row in rows:
        if any(key not in row for key in required):
            raise ValueError("轨迹缺少v2筛选指标，不能将缺失指标补零")
        if any(key not in row["reward_components"] for key in ("exploration_bonus", "blank_revisit_penalty")):
            raise ValueError("轨迹缺少空间奖励分量")
    first_hit = next((i for i, row in enumerate(rows, 1) if row["odor_hit"]), None)
    cap = min(first_hit, HORIZON) if first_hit is not None else HORIZON
    collisions = sum(bool(row["collision"]) for row in rows)
    revisit = sum(bool(row["blank_revisit_active"]) for row in rows)
    return {"steps": len(rows), "ever_hit": first_hit is not None, "first_hit_step": first_hit,
            "first_hit_step_capped": cap, "first_hit_time_s_capped": cap * dt,
            "success": bool(rows[-1]["is_success"]), "collision_count": collisions,
            "had_collision": collisions > 0, "collision_rate": collisions / len(rows),
            "unique_visited_cells": int(rows[-1]["novel_cells"]),
            "exploration_bonus_total": float(sum(row["reward_components"]["exploration_bonus"] for row in rows)),
            "blank_revisit_cost_total": float(-sum(row["reward_components"]["blank_revisit_penalty"] for row in rows)),
            "blank_revisit_ratio": revisit / len(rows), "blank_revisit_seconds": revisit * dt,
            "blank_loop_ratio": sum(bool(row["blank_loop"]) for row in rows) / len(rows),
            "reacquisition_events": sum(bool(row["reacquisition_event"]) for row in rows),
            "hit_step_ratio": sum(bool(row["odor_hit"]) for row in rows) / len(rows)}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("不能汇总空评估")
    out = {"episodes": len(rows), "ever_hit_episodes": sum(row["ever_hit"] for row in rows),
           "successes": sum(row["success"] for row in rows)}
    for name, key in (("ever_hit_rate", "ever_hit"), ("success_rate", "success"),
                      ("collision_episode_rate", "had_collision")):
        out[name] = float(np.mean([row[key] for row in rows]))
    for key in ("first_hit_step_capped", "first_hit_time_s_capped", "collision_count", "collision_rate",
                "unique_visited_cells", "exploration_bonus_total", "blank_revisit_cost_total",
                "blank_revisit_ratio", "blank_revisit_seconds", "blank_loop_ratio", "reacquisition_events",
                "hit_step_ratio", "steps"):
        out[f"mean_{key}"] = float(np.mean([row[key] for row in rows]))
    return out


def evaluate_model(output_dir: Path, config: dict[str, Any], variant: str, checkpoint: int,
                   model_path: Path, split: str) -> list[Path]:
    """CPU串行复评，避免多个worker争用全局随机数导致随机策略结果漂移。"""
    from stable_baselines3 import PPO
    import torch
    from dual_whisker_rl.mobile_observation import MobileObservationHistory

    validate_variant(config, variant)
    scenes, bank = scene_bank(output_dir, config, split)
    model_sha = sha256_file(model_path)
    # watch反复扫描时复用已有完整结果，不重新加载神经网络或重复运行回合。
    cached_paths = [output_dir / "evaluations" / f"{variant}_{checkpoint}_{model_sha[:12]}_{split}_{mode}.json"
                    for mode in ("deterministic", "stochastic")]
    if all(path.exists() for path in cached_paths):
        for mode, path in zip(("deterministic", "stochastic"), cached_paths):
            saved = read_json(path)
            expected = {"protocol_version": PROTOCOL_VERSION, "variant": variant,
                        "checkpoint": checkpoint, "split": split, "policy_mode": mode,
                        "model_sha256": model_sha, "scene_sha256": bank["scene_sha256"],
                        "physical_sha256": bank["physical_sha256"], "config_sha256": digest(config),
                        "history_length": HISTORY_LENGTH, "horizon": HORIZON, "device": "cpu", "training_seed": 11}
            if saved.get("signature") != expected:
                raise ValueError(f"已有评估签名不一致：{path}")
            verify_evaluation(saved, bank)
            write_csv(path.with_suffix(".csv"), [{**expected, **row} for row in saved["episodes"]])
        return cached_paths
    model = PPO.load(str(model_path), device="cpu")
    if sha256_file(model_path) != model_sha:
        raise ValueError("加载期间模型文件被改写，请使用已完成保存的检查点")
    if int(model.num_timesteps) != checkpoint:
        raise ValueError(f"模型记录步数{model.num_timesteps}与 --checkpoint {checkpoint} 不符")
    if model.seed != 11:
        raise ValueError(f"本轮筛选要求训练种子11，模型记录为{model.seed!r}")
    torch.set_num_threads(1)
    raw = MobileWhiskerPuffEnv({**config, "record_trajectory": True})
    paths = []
    try:
        history = validate_model_environment(model, raw, HISTORY_LENGTH)
        env = MobileObservationHistory(raw, history)
        for mode in ("deterministic", "stochastic"):
            signature = {"protocol_version": PROTOCOL_VERSION, "variant": variant,
                         "checkpoint": checkpoint, "split": split, "policy_mode": mode,
                         "model_sha256": model_sha, "scene_sha256": bank["scene_sha256"],
                         "physical_sha256": bank["physical_sha256"], "config_sha256": digest(config),
                         "history_length": history, "horizon": HORIZON, "device": "cpu", "training_seed": 11}
            path = output_dir / "evaluations" / f"{variant}_{checkpoint}_{model_sha[:12]}_{split}_{mode}.json"
            if path.exists():
                saved = read_json(path)
                if saved.get("signature") != signature:
                    raise ValueError(f"已有评估签名不一致：{path}")
                verify_evaluation(saved, bank)
                write_csv(path.with_suffix(".csv"), [{**signature, **row} for row in saved["episodes"]])
                paths.append(path)
                continue
            episodes = []
            for index, scene in enumerate(scenes, 1):
                action_seed = episode_action_seed(scene.seed)
                seed_action_sampling(action_seed)
                row = evaluate_one(model, env, scene, deterministic=mode == "deterministic", max_steps=HORIZON)
                row.update(trajectory_metrics(raw.trajectory, dt=float(raw.dt)))
                row.update({"scenario_seed": scene.seed, "action_seed": action_seed})
                episodes.append(row)
                if index == 1 or index % 8 == 0 or index == len(scenes):
                    print(f"{variant} {checkpoint} {split} {mode}: {index}/{len(scenes)}", flush=True)
            payload = {"signature": signature, "model_path": portable_path(model_path),
                       "config": config, "summary": summarize(episodes), "episodes": episodes,
                       "sampling_protocol": "CPU PPO.predict; reset python/numpy/torch RNG per episode; action seed = SHA256(protocol/action/scene_seed)[:4] big-endian",
                       "limitation": LIMITATION}
            write_json(path, payload)
            write_csv(path.with_suffix(".csv"), [{**signature, **row} for row in episodes])
            paths.append(path)
    finally:
        raw.close()
    return paths


def verify_evaluation(payload: dict[str, Any], bank: dict[str, Any]) -> None:
    """报告也重新核对题库配对和汇总，防止手工改文件导致错配。"""
    signature = payload["signature"]
    if (signature.get("protocol_version") != PROTOCOL_VERSION
            or signature.get("scene_sha256") != bank["scene_sha256"]
            or signature.get("physical_sha256") != bank["physical_sha256"]):
        raise ValueError("评估结果的题库或协议摘要不一致")
    if "config" in payload and signature.get("config_sha256") != digest(payload["config"]):
        raise ValueError("评估配置与摘要不一致")
    if "split" in bank and signature.get("split") != bank["split"]:
        raise ValueError("评估题库分组不一致")
    episodes = payload["episodes"]
    if [(row["scenario_id"], row["scenario_seed"]) for row in episodes] != [
            (scene["scenario_id"], scene["seed"]) for scene in bank["scenes"]]:
        raise ValueError("评估回合未完整覆盖封存题库或顺序不同")
    if payload["summary"] != summarize(episodes):
        raise ValueError("评估汇总与逐回合数据不一致")


def select_candidate(results: list[dict[str, Any]]) -> dict[str, Any]:
    """只按100万步开发集确定性结果筛选；其它检查点仅显示趋势。"""
    eligible = {item["signature"]["variant"]: item for item in results
                if item["signature"]["checkpoint"] == 1_000_000
                and item["signature"]["split"] == "development"
                and item["signature"]["policy_mode"] == "deterministic"}
    missing = sorted(set(VARIANTS) - set(eligible))
    if missing:
        return {"status": "pending", "missing_variants": missing, "selected_variant": None}
    if any(item["summary"]["episodes"] != 64 for item in eligible.values()):
        raise ValueError("晋级必须完整评估64个开发场景")
    if len({item["signature"]["scene_sha256"] for item in eligible.values()}) != 1:
        raise ValueError("晋级比较必须使用相同开发题库")
    base = eligible["A"]["summary"]["ever_hit_episodes"]
    gains = {variant: item["summary"]["ever_hit_episodes"] - base
             for variant, item in eligible.items() if variant != "A"}
    promoted = [variant for variant, gain in gains.items() if gain >= 4]
    promoted.sort(key=lambda variant: (-eligible[variant]["summary"]["ever_hit_episodes"],
                                      eligible[variant]["summary"]["mean_first_hit_step_capped"], variant))
    winner = promoted[0] if promoted else None
    return {"status": "candidate_selected" if winner else "no_candidate", "selected_variant": winner,
            "gain_over_A_episodes": gains, "threshold_extra_hit_episodes": 4,
            "selected_model_sha256": eligible[winner]["signature"]["model_sha256"] if winner else None,
            "baseline_model_sha256": eligible["A"]["signature"]["model_sha256"],
            "tie_break": "higher ever_hit_episodes; lower all-episode mean_first_hit_step_capped; variant"}


def build_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    seen = set()
    for item in results:
        sig = item["signature"]
        key = tuple(sig[key] for key in ("variant", "checkpoint", "split", "policy_mode"))
        if key in seen:
            raise ValueError(f"同一组/检查点/策略有多个评估，请分开输出目录以免选择性报告：{key}")
        seen.add(key)
    selection = select_candidate(results)
    comparisons = []
    winner = selection["selected_variant"]
    if winner:
        for mode in ("deterministic", "stochastic"):
            pair = {item["signature"]["variant"]: item for item in results
                    if item["signature"]["split"] == "heldout" and item["signature"]["checkpoint"] == 1_000_000
                    and item["signature"]["policy_mode"] == mode and item["signature"]["variant"] in ("A", winner)}
            if set(pair) == {"A", winner}:
                if pair["A"]["signature"]["scene_sha256"] != pair[winner]["signature"]["scene_sha256"]:
                    raise ValueError("保留集比较必须使用同一题库")
                for variant, sha_key in (("A", "baseline_model_sha256"), (winner, "selected_model_sha256")):
                    if (pair[variant]["signature"]["model_sha256"] != selection[sha_key]
                            or pair[variant]["summary"]["episodes"] != 128):
                        raise ValueError("保留集必须复评开发集选中的同一模型，且完整覆盖128局")
                a = pair["A"]["episodes"]
                b = pair[winner]["episodes"]
                differences = [int(y["ever_hit"]) - int(x["ever_hit"]) for x, y in zip(a, b)]
                comparisons.append({"policy_mode": mode, "episodes": 128,
                                    "candidate_variant": winner, "baseline": pair["A"]["summary"],
                                    "candidate": pair[winner]["summary"],
                                    "gain_ever_hit_episodes": sum(differences),
                                    "new_hit_episodes": differences.count(1), "lost_hit_episodes": differences.count(-1)})
    return {"protocol_version": PROTOCOL_VERSION, "selection": selection,
            "heldout_comparisons": comparisons, "limitation": LIMITATION,
            "results": [{**item["signature"], **item["summary"]} for item in results]}


def report(output_dir: Path) -> dict[str, Any]:
    results = []
    for path in sorted((output_dir / "evaluations").glob("*.json")):
        payload = read_json(path)
        config = payload["config"]
        validate_variant(config, payload["signature"]["variant"])
        _, bank = scene_bank(output_dir, config, payload["signature"]["split"])
        verify_evaluation(payload, bank)
        results.append(payload)
    result = build_report(results)
    write_json(output_dir / "report.json", result)
    write_csv(output_dir / "report.csv", result["results"])
    lines = ["# 空间探索筛选报告", "", LIMITATION, "", "确定性与随机策略分开报告；首次命中为全部回合的1200步截尾均值。", "",
             "|组|检查点|题库|策略|命中局/总局|首次命中截尾步均值|到源率|碰撞步率|独立格数|探索奖励|重复空白成本|旧绕圈率|重捕获次数|",
             "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["results"]:
        lines.append(f"|{row['variant']}|{row['checkpoint']}|{row['split']}|{row['policy_mode']}|"
                     f"{row['ever_hit_episodes']}/{row['episodes']}|{row['mean_first_hit_step_capped']:.1f}|"
                     f"{row['success_rate']:.1%}|{row['mean_collision_rate']:.1%}|{row['mean_unique_visited_cells']:.1f}|"
                     f"{row['mean_exploration_bonus_total']:.3f}|{row['mean_blank_revisit_cost_total']:.3f}|"
                     f"{row['mean_blank_loop_ratio']:.1%}|{row['mean_reacquisition_events']:.2f}|")
    selection = result["selection"]
    lines.extend(["", f"开发集筛选状态：`{selection['status']}`。"])
    if selection["selected_variant"]:
        lines.append(f"按预设规则选择 {selection['selected_variant']} 与A组在保留集复核；保留集不用于重新选组。")
    elif selection["status"] == "no_candidate":
        lines.append("没有组达到比A组多4/64个命中回合的门槛；不自动追加训练。")
    else:
        lines.append("尚缺100万步开发集确定性结果：" + ", ".join(selection["missing_variants"]) + "。")
    for comparison in result["heldout_comparisons"]:
        lines.append(f"保留集 {comparison['policy_mode']}：{comparison['candidate_variant']} 相比A组"
                     f"净增 {comparison['gain_ever_hit_episodes']}/128 个命中回合"
                     f"（新增{comparison['new_hit_episodes']}、丢失{comparison['lost_hit_episodes']}）。")
    if selection["selected_variant"] and len(result["heldout_comparisons"]) < 2:
        lines.append("保留集复核尚未完成两种策略模式，当前结论仅为开发集筛选。")
    lines.extend(["", "模型、题库、物理和配置SHA256保存在report.csv及各评估JSON/CSV；训练窗口成功率不参与上述统计。", ""])
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def read_manifest(path: Path) -> dict[str, dict[str, Any]]:
    """清单列出面板任务根目录或显式模型路径，不修改训练任务与数据库。"""
    value = read_json(path)
    groups = value.get("groups", value.get("variants", {}))
    if value.get("protocol_version") != PROTOCOL_VERSION or set(groups) != set(VARIANTS):
        raise ValueError("watch清单必须使用当前协议并包含A/B/C/D四组")
    result = {}
    for variant, item in groups.items():
        if ("task_root" in item) == ("models" in item):
            raise ValueError(f"{variant}组请指定task_root或models两者之一")
        config_path = project_path(item.get("config", f"configs/mobile_rooms_spatial_{variant.lower()}.yaml"))
        cfg = load_config(config_path)
        validate_variant(cfg, variant)
        result[variant] = {"config": cfg}
        if "task_root" in item:
            result[variant]["task_root"] = project_path(item["task_root"])
        else:
            if set(item["models"]) != {str(step) for step in CHECKPOINTS}:
                raise ValueError(f"{variant}组清单必须显式列出25/50/100万步模型")
            result[variant]["models"] = {int(step): project_path(model) for step, model in item["models"].items()}
    return result


def model_ready(path: Path, checkpoint: int | None = None) -> bool:
    """等写入结束并核对ZIP完整性，避免读取尚在保存中的检查点。"""
    if not path.is_file():
        return False
    before = path.stat()
    if time.time() - before.st_mtime < 2.0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            if not {"data", "policy.pth"}.issubset(archive.namelist()) or archive.testzip() is not None:
                return False
            if checkpoint is not None and json.loads(archive.read("data"))["num_timesteps"] != checkpoint:
                return False
    except (OSError, zipfile.BadZipFile, KeyError, ValueError):
        return False
    after = path.stat()
    return (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)


def discover_model(item: dict[str, Any], checkpoint: int) -> Path | None:
    """检查点优先；仅100万步允许回退到同样实际步数的final_model。"""
    if "models" in item:
        candidate = item["models"][checkpoint]
        return candidate if model_ready(candidate, checkpoint) else None
    root = item["task_root"]
    groups = [list(root.glob(f"attempts/*/checkpoints/*_{checkpoint}_steps.zip"))]
    if checkpoint == 1_000_000:
        groups.append(list(root.glob("attempts/*/final_model.zip")))
    for candidates in groups:
        for candidate in sorted(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True):
            if model_ready(candidate, checkpoint):
                return candidate
    return None


def watch_once(output_dir: Path, manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """串行复评已保存检查点；完成开发集后只复核A与预设规则选中的候选。"""
    pending = []
    ready = {}
    for checkpoint in CHECKPOINTS:
        for variant in VARIANTS:
            item = manifest[variant]
            model = discover_model(item, checkpoint)
            if model is None:
                expected = item["models"][checkpoint] if "models" in item else item["task_root"]
                pending.append({"variant": variant, "checkpoint": checkpoint, "model_or_task_root": portable_path(expected)})
                continue
            ready[variant, checkpoint] = model
            evaluate_model(output_dir, item["config"], variant, checkpoint, model, "development")
    result = report(output_dir)
    winner = result["selection"]["selected_variant"]
    if winner and all((variant, 1_000_000) in ready for variant in ("A", winner)):
        for variant in ("A", winner):
            item = manifest[variant]
            evaluate_model(output_dir, item["config"], variant, 1_000_000,
                           ready[variant, 1_000_000], "heldout")
        result = report(output_dir)
    complete = not pending and (result["selection"]["status"] == "no_candidate"
                                or len(result["heldout_comparisons"]) == 2)
    state = {"protocol_version": PROTOCOL_VERSION, "complete": complete,
             "pending_checkpoints": pending, "selection": result["selection"],
             "updated_at_unix": time.time(), "automatic_training": False}
    write_json(output_dir / "watch_status.json", state)
    return state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "evaluate", "report", "watch"):
        sub = commands.add_parser(name)
        sub.add_argument("--output-dir", type=Path, default=Path("results/spatial_exploration_pilot"))
        if name in ("prepare", "evaluate"):
            sub.add_argument("--config", type=Path, default=None)
        if name == "watch":
            sub.add_argument("--manifest", type=Path, required=True)
            sub.add_argument("--poll-seconds", type=float, default=30.)
            sub.add_argument("--once", action="store_true", help="仅扫描一次已保存模型，不等待训练")
        if name == "evaluate":
            sub.add_argument("--variant", choices=tuple(VARIANTS), required=True)
            sub.add_argument("--checkpoint", type=int, choices=CHECKPOINTS, required=True)
            sub.add_argument("--model", type=Path, required=True)
            sub.add_argument("--split", choices=tuple(SPLITS), default="development")
    args = parser.parse_args(argv)
    args.output_dir = project_path(args.output_dir)
    if args.command in ("prepare", "evaluate"):
        args.config = project_path(args.config or f"configs/mobile_rooms_spatial_{getattr(args, 'variant', 'A').lower()}.yaml")
    if args.command == "watch":
        args.manifest = project_path(args.manifest)
        if not np.isfinite(args.poll_seconds) or args.poll_seconds < 1.:
            parser.error("--poll-seconds 必须为至少1秒的有限值")
    if args.command == "evaluate":
        args.model = project_path(args.model)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "prepare":
        config = load_config(args.config)
        for split in SPLITS:
            scenes, payload = scene_bank(args.output_dir, config, split, create=True)
            print(f"{split}: {len(scenes)} scenes sha256={payload['scene_sha256']}")
    elif args.command == "evaluate":
        for path in evaluate_model(args.output_dir, load_config(args.config), args.variant,
                                   args.checkpoint, args.model, args.split):
            print(portable_path(path))
    elif args.command == "watch":
        manifest = read_manifest(args.manifest)
        for item in manifest.values():
            for split in SPLITS:
                scene_bank(args.output_dir, item["config"], split, create=True)
        while True:
            state = watch_once(args.output_dir, manifest)
            print(json.dumps(state, ensure_ascii=False), flush=True)
            if args.once or state["complete"]:
                break
            time.sleep(args.poll_seconds)
    else:
        result = report(args.output_dir)
        print(json.dumps(result["selection"], ensure_ascii=False))
        print(portable_path(args.output_dir / "report.md"))


if __name__ == "__main__":
    main()
