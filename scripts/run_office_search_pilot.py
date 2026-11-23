"""小型办公场景的触须长度、绕圈奖励及训练步数配对试验。"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
import yaml
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dual_whisker_rl.e4_experiments import E4Scenario, ObservationHistory, reset_scenario, scenario_hash, sha256_file
from dual_whisker_rl.envs.mobile_whisker_env import MobileWhiskerPuffEnv
from dual_whisker_rl.experiment_protocol import paired_interval
from dual_whisker_rl.mobile_contract import environment_contract, validate_model_environment
from dual_whisker_rl.paths import project_path
from evaluate_mobile_fixed_scenarios import generate_scenarios

BASE = ROOT / "configs/mobile_rooms_hybrid_history.yaml"
LENGTHS = (.15, .225, .30)
HISTORY_STEPS = tuple(range(10_000_000, 15_000_001, 1_000_000))
SOURCE_RUN = "office_history_main_10m_2"
PROTOCOL_VERSION = 1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def scene_bank(root, config, split, count, seed):
    path = root / f"{split}_scenes.json"
    if path.exists():
        saved = read_json(path)
        scenes = [E4Scenario.from_dict(row) for row in saved["scenes"]]
        if (saved["base_config"] != config or len(scenes) != count or scenes[0].seed != seed
                or saved["scene_hash"] != scenario_hash(scenes)):
            raise ValueError(f"封存场景与本次配置不一致: {path}")
        return scenes
    scenes = [
        replace(scene, split=split, scenario_id=f"{split}_{scene.seed}")
        for scene in generate_scenarios(config, count=count, base_seed=seed)
    ]
    write_json(path, {"scene_hash": scenario_hash(scenes), "base_config": config,
                      "scenes": [scene.to_dict() for scene in scenes]})
    return scenes


def make_env(config):
    return ObservationHistory(
        MobileWhiskerPuffEnv({**config, "record_trajectory": True}), history_length=20,
    )


def load_policy(path, env):
    model = PPO.load(str(path), device="cpu")
    validate_model_environment(model, env.unwrapped, 20)
    return model


def rollout(env, scene, policy, commands=None):
    obs, info = reset_scenario(env, scene)
    if not np.allclose(
        [env.unwrapped.robot_state.x, env.unwrapped.robot_state.y, env.unwrapped.robot_state.heading],
        scene.robot_pose, rtol=0, atol=1e-10,
    ):
        raise ValueError(f"场景起点不一致: {scene.scenario_id}")
    if not (np.allclose(info["source"], scene.source_position, rtol=0, atol=1e-10)
            and math.isclose(env.unwrapped.plume.wind_direction_base, scene.wind_direction)
            and math.isclose(env.unwrapped.plume.wind_speed_mean, scene.wind_speed)):
        raise ValueError(f"场景气源或风不一致: {scene.scenario_id}")
    initial_path = env.unwrapped.room_navigation.initial_path_distance
    actions = []
    ending = "horizon"
    for index in range(env.unwrapped.max_steps if commands is None else len(commands)):
        action = policy.predict(obs, deterministic=True)[0] if commands is None else commands[index]
        actions.append(np.asarray(action).copy())
        obs, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            ending = "success" if info["is_success"] else "out_of_bounds" if terminated else "timeout"
            break
    rows = env.unwrapped.trajectory
    first_hit = next((i for i, row in enumerate(rows, 1) if row["odor_hit"]), None)
    result = {
        "scenario_id": scene.scenario_id, "scenario_seed": scene.seed,
        "success": bool(info["is_success"]), "termination": ending, "steps": len(rows),
        "start_room": rows[0]["start_room"], "source_room": rows[0]["source_room"],
        "initial_path_m": initial_path,
        "min_path_m": min(initial_path, min(row["path_distance_to_source"] for row in rows)),
        "first_hit_step": first_hit, "ever_hit": first_hit is not None,
        "first_hit_time_s": first_hit * env.unwrapped.dt if first_hit is not None else None,
        "hit_step_ratio": sum(row["odor_hit"] for row in rows) / len(rows),
        "blank_loop_ratio": sum(row["blank_loop"] for row in rows) / len(rows),
        "collision_rate": sum(row["collision"] for row in rows) / len(rows),
        "distance_travelled_m": rows[-1]["path_length"],
        "left_clipped_rate": np.mean([row["left_effective_length"] < env.unwrapped.whiskers.length - 1e-7 for row in rows]),
        "right_clipped_rate": np.mean([row["right_effective_length"] < env.unwrapped.whiskers.length - 1e-7 for row in rows]),
        "mean_effective_tip_length_m": np.mean([
            (row["left_effective_length"] + row["right_effective_length"]) / 2 for row in rows
        ]),
        "reacquisition_events": sum(row["reacquisition_event"] for row in rows),
        "final_distance_m": rows[-1]["distance_to_source"],
        "initial_heading_mode": env.unwrapped.initial_heading_mode,
    }
    return result, rows, actions


def init_worker(model_path, config, replay):
    global policy, environments
    environments = [make_env(config)]
    policy = load_policy(model_path, environments[0])
    if replay:
        environments.extend(make_env({**config, "whisker_length": length}) for length in LENGTHS[1:])


def evaluate_scene(scene):
    row, trace, _ = rollout(environments[0], scene, policy)
    fields = ("x", "y", "heading", "left", "right", "distance_to_source",
              "path_distance_to_source", "left_sector", "right_sector", "collision",
              "blank_loop", "reward")
    return row, {"scene": scene.to_dict(), "trace": [
        {key: step[key] for key in fields} for step in trace
    ]}


def replay_scene(scene):
    baseline, base_trace, actions = rollout(environments[0], scene, policy)
    rows = [{**baseline, "length_m": LENGTHS[0]}]
    for length, env in zip(LENGTHS[1:], environments[1:]):
        row, trace, _ = rollout(env, scene, None, actions)
        if len(trace) != len(base_trace) or not np.allclose(
            [(step["x"], step["y"], step["heading"]) for step in trace],
            [(step["x"], step["y"], step["heading"]) for step in base_trace],
            rtol=0, atol=1e-10,
        ):
            raise ValueError(f"长度对照的底盘轨迹不一致: {scene.scenario_id}")
        rows.append({**row, "length_m": length})
    return rows


def scan_scene(scene):
    active, active_trace, actions = rollout(environments[0], scene, policy)
    count = environments[0].unwrapped.whiskers.sector_count
    sectors = list(range(count)) + list(range(count - 2, 0, -1))
    scan_actions = [
        [*action[:2], sectors[(step // 2) % len(sectors)], sectors[(step // 2) % len(sectors)]]
        for step, action in enumerate(actions)
    ]
    scan, scan_trace, _ = rollout(environments[0], scene, None, scan_actions)
    if len(scan_trace) != len(active_trace) or not np.allclose(
        [(step["x"], step["y"]) for step in scan_trace],
        [(step["x"], step["y"]) for step in active_trace], rtol=0, atol=1e-10,
    ):
        raise ValueError(f"周期扫描的底盘轨迹不一致: {scene.scenario_id}")
    return active, scan


def summarize(rows):
    first_hit_times = [row["first_hit_time_s"] for row in rows if row["ever_hit"]]
    return {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": np.mean([row["success"] for row in rows]),
        "ever_hit_rate": np.mean([row["ever_hit"] for row in rows]),
        "median_first_hit_step_among_hits": (
            float(np.median([row["first_hit_step"] for row in rows if row["ever_hit"]]))
            if any(row["ever_hit"] for row in rows) else None
        ),
        "median_first_hit_time_s_among_hits": float(np.median(first_hit_times)) if first_hit_times else None,
        "median_min_path_m": np.median([row["min_path_m"] for row in rows]),
        "collision_rate": np.mean([row["collision_rate"] for row in rows]),
        "blank_loop_ratio": np.mean([row["blank_loop_ratio"] for row in rows]),
    }


def evaluate(model_path, config, scenes, path, workers):
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = {"model_sha256": sha256_file(model_path), "config": config,
                 "scene_hash": scenario_hash(scenes), "protocol_version": PROTOCOL_VERSION}
    if path.exists():
        result = read_json(path)
        if result["signature"] != signature:
            raise ValueError(f"评估结果的签名不匹配: {path}")
        return result
    print(f"eval {path.name}: {len(scenes)} scenes", flush=True)
    rows, samples = [], {}
    with ProcessPoolExecutor(max_workers=min(workers, len(scenes)),
                             initializer=init_worker, initargs=(model_path, config, False)) as pool:
        for i, (row, trace) in enumerate(pool.map(evaluate_scene, scenes), 1):
            rows.append(row)
            samples.setdefault(row["termination"], trace)
            if i == 1 or i % 8 == 0 or i == len(scenes):
                print(f"eval {path.name}: {i}/{len(scenes)}", flush=True)
    result = {"signature": signature, "summary": summarize(rows), "episodes": rows,
              "by_room": {}}
    for label, same_room in (("same_room", True), ("cross_room", False)):
        group = [row for row in rows if (row["start_room"] == row["source_room"]) == same_room]
        result["by_room"][label] = summarize(group) if group else {"episodes": 0}
    write_json(path.with_name(path.stem + "_trajectories.json"), samples)
    write_json(path, result)
    return result


def probe_tips(config):
    """独立核对固定羽流探针与碰墙几何，探针不进入长度晋级统计。"""
    env = MobileWhiskerPuffEnv(config)
    try:
        env.reset(seed=600_101)
        plume = env.plume
        room_map = env.room_navigation.map
        plume.source_x, plume.source_y = -2.0, -3.5
        probes = {}
        for kind, center, puff_xy in (
            ("plume_edge", (-2.5, -3.0), (-2.25, -3.35)),
            ("open_door", (-.56, -3.5), (-1.1, -3.5)),
            ("closed_wall", (-.56, -2.5), (-1.1, -2.5)),
        ):
            px, py = puff_xy
            plume.puffs = [dict(x=px, y=py, mass=.1, sigma_downwind=.2,
                                sigma_crosswind=.12, cos_direction=1., sin_direction=0.)]
            direction = (-1., 0.) if kind != "plume_edge" else (0., -1.)
            points = [room_map.clip_sample(center, (center[0] + length * direction[0],
                                                     center[1] + length * direction[1]))
                      for length in LENGTHS]
            probes[kind] = [{"length_m": length, "effective_m": math.dist(center, point),
                            "raw": plume.concentration(*point, add_noise=False)}
                           for length, point in zip(LENGTHS, points)]
        return probes
    finally:
        env.close()


def length_screen(root, model, config, scenes, workers):
    path = root / "length_screen.json"
    signature = {"model_sha256": sha256_file(model), "scene_hash": scenario_hash(scenes),
                 "config": config, "protocol_version": PROTOCOL_VERSION}
    if path.exists():
        result = read_json(path)
        if result["signature"] != signature:
            raise ValueError("长度筛选签名不匹配")
        return result["selected_length_m"]
    groups = {length: [] for length in LENGTHS}
    with ProcessPoolExecutor(max_workers=min(workers, len(scenes)),
                             initializer=init_worker, initargs=(model, config, True)) as pool:
        for i, rows in enumerate(pool.map(replay_scene, scenes), 1):
            for row in rows:
                groups[row["length_m"]].append(row)
            if i == 1 or i % 8 == 0 or i == len(scenes):
                print(f"length replay {i}/{len(scenes)}", flush=True)
    summaries = {}
    for length, rows in groups.items():
        summaries[length] = {**summarize(rows), **{
            key: np.mean([row[key] for row in rows])
            for key in ("hit_step_ratio", "left_clipped_rate", "right_clipped_rate", "mean_effective_tip_length_m")
        }}
    baseline = summaries[LENGTHS[0]]["ever_hit_rate"]
    passing = [length for length in LENGTHS[1:] if summaries[length]["ever_hit_rate"] - baseline >= .05]
    selected = passing[0] if passing else LENGTHS[1]
    write_json(path, {"signature": signature, "selected_length_m": selected,
                      "passed_hit_threshold": bool(passing), "summaries": summaries,
                      "probes": probe_tips(config), "episodes": groups,
                      "paired_ever_hit": {
                          length: paired({0: {"episodes": groups[LENGTHS[0]]}},
                                         {0: {"episodes": groups[length]}}, "ever_hit")
                          for length in LENGTHS[1:]
                      }})
    print(f"length selected={selected} passed={bool(passing)}", flush=True)
    return selected


def train(root, name, seed, config, smoke):
    folder = root / "training" / name / str(seed)
    model = folder / "final.zip"
    config_path = folder / "config.yaml"
    steps = 40 if smoke else 1_000_000
    if model.exists():
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"模型目录已绑定其他配置: {folder}")
        with make_env(config) as env:
            if load_policy(model, env).num_timesteps != steps:
                raise ValueError(f"模型步数和预算不一致: {model}")
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    # 中断的训练保留日志，下一次使用新尝试目录并从最后一个完整检查点恢复。
    existing_attempts = sorted(folder.glob("attempt_*"))
    attempt = folder / f"attempt_{len(existing_attempts) + 1:03d}"
    checkpoints = sorted(folder.glob("attempt_*/logs/checkpoints/*_steps.zip"),
                         key=lambda path: int(path.stem.split("_")[-2]), reverse=True)
    resume = []
    remaining = steps
    if existing_attempts:
        if yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError(f"训练目录已绑定其他配置: {folder}")
        with make_env(config) as env:
            for saved in reversed(existing_attempts):
                complete = saved / "training_result.zip"
                if not complete.is_file():
                    continue
                try:
                    previous = load_policy(complete, env)
                except (OSError, EOFError, zipfile.BadZipFile):
                    continue
                if previous.num_timesteps == steps:
                    complete.replace(model)
                    return folder
            for checkpoint in checkpoints:
                try:
                    previous = load_policy(checkpoint, env)
                except (ValueError, OSError, EOFError, zipfile.BadZipFile):
                    continue
                # 终点 CheckpointCallback 在当轮优化之前触发，需由前一个检查点补齐。
                if previous.num_timesteps < steps:
                    remaining -= previous.num_timesteps
                    original_log = next(path / "logs" for path in existing_attempts
                                        if (path / "logs/run_metadata.json").is_file())
                    resume = ["--resume-from", str(checkpoint), "--log-dir", str(attempt / "logs"), "--checkpoint-dir",
                              str(attempt / "logs/checkpoints")]
                    break
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    attempt.mkdir()
    if resume:
        (attempt / "logs").mkdir()
        shutil.copyfile(original_log / "run_metadata.json", attempt / "logs/run_metadata.json")
    command = [sys.executable, "-u", str(ROOT / "scripts/train_mobile_whisker_ppo.py"),
               "--config", str(config_path), "--timesteps", str(remaining), "--seed", str(seed),
               "--n-envs", "2" if smoke else "8", "--vec-env-backend", "subproc",
               "--n-steps", "5" if smoke else "625", "--batch-size", "10" if smoke else "250",
               "--checkpoint-freq", "10" if smoke else "250000",
               "--model-path", str(attempt / "training_result.zip"), *resume]
    if not resume:
        command.extend(["--log-dir", str(attempt / "logs"),
                        "--tensorboard-dir", str(attempt / "tensorboard"),
                        "--experiment-name", f"office_search_{name}_{seed}"])
    print(f"train {name} seed={seed} steps={remaining} log={attempt / 'train.log'}", flush=True)
    with (attempt / "train.log").open("w", encoding="utf-8") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT, check=True)
    with make_env(config) as env:
        if load_policy(attempt / "training_result.zip", env).num_timesteps != steps:
            raise RuntimeError(f"训练步数不一致: {attempt}")
    (attempt / "training_result.zip").replace(model)
    return folder


def checkpoints_for(folder, smoke):
    milestones = (10, 20) if smoke else (250_000, 500_000)
    checkpoints = list(folder.glob("attempt_*/logs/checkpoints/*_steps.zip"))
    return [(step, next(path for path in reversed(sorted(checkpoints))
                        if path.name.endswith(f"_{step}_steps.zip"))) for step in milestones]


def eligible(candidate, baseline):
    return candidate["collision_rate"] - baseline["collision_rate"] <= .01 and (
        candidate["success_rate"] - baseline["success_rate"] >= .03
        or (baseline["median_min_path_m"] - candidate["median_min_path_m"] >= .5
            and candidate["ever_hit_rate"] - baseline["ever_hit_rate"] >= .05)
    )


def paired(before, after, metric):
    def values(records):
        return {(seed, row["scenario_id"]): row[metric]
                for seed, result in records.items() for row in result["episodes"]}
    return paired_interval(values(after), values(before))


def compare_periodic_scan(root, name, config, scenes, workers):
    print(f"scan comparison {name}: active and periodic on the same base path", flush=True)
    active, scan = {}, {}
    for seed in (11, 23, 37):
        folder = root / "training" / name / str(seed)
        model = folder / "final.zip"
        path = folder / "holdout_scan.json"
        signature = {"model_sha256": sha256_file(model), "scene_hash": scenario_hash(scenes),
                     "protocol_version": PROTOCOL_VERSION}
        if path.exists():
            result = read_json(path)
            if result["signature"] != signature:
                raise ValueError(f"周期扫描结果签名不匹配: {path}")
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                     initargs=(model, config, False)) as pool:
                pairs = list(pool.map(scan_scene, scenes))
            result = {"signature": signature, "active": [pair[0] for pair in pairs],
                      "periodic": [pair[1] for pair in pairs]}
            write_json(path, result)
        active[seed] = {"episodes": result["active"]}
        scan[seed] = {"episodes": result["periodic"]}
    write_json(root / f"{name}_active_vs_periodic.json", {
        key: paired(scan, active, key)
        for key in ("ever_hit", "hit_step_ratio", "reacquisition_events")
    })


def training_is_active(session):
    return bool(session) and subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"], capture_output=True, check=False,
    ).returncode == 0


def plot_history(curve, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = sorted(curve)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), layout="constrained")
    axes[0].plot(np.asarray(steps) / 1e6, [100 * curve[s]["success_rate"] for s in steps], "o-")
    axes[0].set_ylabel("Source success (%)")
    axes[1].plot(np.asarray(steps) / 1e6, [curve[s]["median_min_path_m"] for s in steps], "o-")
    axes[1].set_ylabel("Median closest path distance (m)")
    for axis in axes:
        axis.set_xlabel("Training steps (millions)")
        axis.grid(alpha=.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    root = args.output_dir
    base_config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    dev = scene_bank(root, base_config, "dev", 2 if args.smoke else 64, 420_000)
    def old_model(step):
        return args.checkpoint_dir / f"{SOURCE_RUN}_{step}_steps.zip"

    old_summary = {}
    def update_history(workers):
        for step in HISTORY_STEPS:
            if step not in old_summary and old_model(step).is_file():
                try:
                    with zipfile.ZipFile(old_model(step)) as archive:
                        archive.getinfo("data")
                except zipfile.BadZipFile:
                    continue  # 可能正在写入检查点，稍后再检查。
                old_summary[step] = evaluate(old_model(step), base_config, dev,
                                             root / "history" / f"dev_{step}.json", workers)["summary"]
                write_json(root / "history/dev_curve.json", old_summary)
                if not args.smoke:
                    plot_history(old_summary, root / "history/dev_curve.png")

    update_history(args.early_workers)
    length = length_screen(root, old_model(HISTORY_STEPS[0]), base_config, dev, args.early_workers)
    if args.length_only:
        return
    holdout = [] if args.smoke else scene_bank(root, base_config, "holdout", 128, 520_000)

    if training_is_active(args.wait_tmux):
        print(f"waiting for {args.wait_tmux} before using the training GPU", flush=True)
        while training_is_active(args.wait_tmux):
            update_history(args.early_workers)
            time.sleep(60)
    update_history(args.workers)
    write_json(root / "history/status.json", {"available_steps": sorted(old_summary),
               "missing_steps": [step for step in HISTORY_STEPS if step not in old_summary]})
    if not args.smoke and all(step in old_summary for step in (HISTORY_STEPS[0], HISTORY_STEPS[-1])):
        reference, current = [
            evaluate(old_model(step), base_config, holdout,
                     root / "history" / f"holdout_{step}.json", args.workers)
            for step in (HISTORY_STEPS[0], HISTORY_STEPS[-1])
        ]
        write_json(root / "history/holdout_comparison.json", {
            "10M": reference["summary"], "15M": current["summary"],
            **{key: paired({0: reference}, {0: current}, key)
               for key in ("success", "ever_hit", "min_path_m", "blank_loop_ratio")},
        })

    configs = {
        "A": base_config,
        "B": {**base_config, "whisker_length": length},
        "C": {**base_config, "blank_loop_penalty": .006},
        "D": {**base_config, "whisker_length": length, "blank_loop_penalty": .006},
    }
    screening = {}
    for name, config in configs.items():
        folder = train(root, name, 11, config, args.smoke)
        for step, checkpoint in checkpoints_for(folder, args.smoke):
            evaluate(checkpoint, config, dev, folder / f"dev_{step}.json", args.workers)
        screening[name] = evaluate(folder / "final.zip", config, dev, folder / "dev_final.json", args.workers)["summary"]
    promoted = [name for name in ("B", "C", "D") if eligible(screening[name], screening["A"])]
    write_json(root / "screening.json", {"summaries": screening, "promoted": promoted})
    if args.smoke:
        curriculum_config = {**base_config, "office_start_curriculum": True}
        folder = train(root, "A_curriculum", 11, curriculum_config, True)
        evaluate(folder / "final.zip", curriculum_config, dev,
                 folder / "dev_final.json", args.workers)
        return

    if not promoted:
        best = max(configs, key=lambda name: (
            screening[name]["success_rate"], -screening[name]["median_min_path_m"],
            screening[name]["ever_hit_rate"],
        ))
        curriculum_name = f"{best}_curriculum"
        configs[curriculum_name] = {**configs[best], "office_start_curriculum": True}
        folder = train(root, curriculum_name, 11, configs[curriculum_name], False)
        result = evaluate(folder / "final.zip", configs[curriculum_name], dev,
                          folder / "dev_final.json", args.workers)["summary"]
        write_json(root / "curriculum.json", {"reference": best, "curriculum": result,
                                              "passed": eligible(result, screening[best])})
        if not eligible(result, screening[best]):
            return
        reference_name, promoted = best, [curriculum_name]
    else:
        reference_name = "A"

    results = {}
    for name in (reference_name, *promoted):
        results[name] = {}
        for seed in (11, 23, 37):
            folder = train(root, name, seed, configs[name], False)
            results[name][seed] = evaluate(folder / "final.zip", configs[name], holdout,
                                           folder / "holdout_final.json", args.workers)
    write_json(root / "confirmation.json", {
        "per_seed": {name: {seed: result["summary"] for seed, result in records.items()}
                     for name, records in results.items()},
        "paired": {name: {key: paired(results[reference_name], results[name], key)
                          for key in ("success", "ever_hit", "min_path_m", "blank_loop_ratio", "collision_rate")}
                   for name in promoted},
        "reference": reference_name,
    })
    for name in promoted:
        original_arm = name.removesuffix("_curriculum")
        if original_arm in ("B", "D"):
            short_arm = "A" if original_arm == "B" else "C"
            if screening[original_arm]["ever_hit_rate"] - screening[short_arm]["ever_hit_rate"] >= .05:
                compare_periodic_scan(root, name, configs[name], holdout, args.workers)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=project_path)
    parser.add_argument("--checkpoint-dir", required=True, type=project_path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--early-workers", type=int, default=2)
    parser.add_argument("--wait-tmux")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--length-only", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.early_workers) < 1:
        parser.error("worker 数必须为正")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / ".running").open("a+b") as fp:
        fp.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(fp.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            parser.error(f"此输出目录正在运行: {exc}")
        run(args)


if __name__ == "__main__":
    main()
