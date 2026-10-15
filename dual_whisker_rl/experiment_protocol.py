"""红字实验的封存、排程和以实验单元为单位的统计工具。"""
from __future__ import annotations
import csv
import hashlib
import itertools
import json
from pathlib import Path
from uuid import uuid4
import numpy as np

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)

def write_csv(path: Path, rows, fields=None):
    if not fields and not rows:
        raise ValueError("空表必须提供字段")
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def seal(path: Path, payload):
    """封存配置；不允许静默覆盖不同内容的实验。"""
    signed = {**payload, "signature": digest(payload)}
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != signed:
            raise ValueError(f"封存内容不一致，请使用新的输出目录：{path}")
        return old
    write_json(path, signed)
    return signed

def hardware_schedule(kind, *, repeats=30, seed=20261013):
    """只生成排程，不连接硬件；参考浓度与任务真值必须实测。"""
    rng, rows = np.random.default_rng(seed), []
    if kind == "static":
        levels = [0, 25, 50, 100, 200, 400]
        for day in range(1, 4):
            for cycle in range(1, 4):
                for direction in rng.permutation(["ascending", "descending"]):
                    sequence = f"d{day}_c{cycle}_{direction}"
                    for level in levels if direction == "ascending" else levels[::-1]:
                        rows.append(dict(run_id=sequence, date=f"day_{day}", plateau_id=f"{sequence}_{level}",
                                         sequence_id=sequence, direction=direction, target_reference=level,
                                         reference_unit="ppm", reference_value="", duration_s="stable_then_60s"))
    elif kind == "drift":
        for day in range(1, 4):
            for condition, duration, level in [("cold_blank", "until_stable", 0), ("warm_blank", 7200, 0), ("warm_mid", 7200, 100)]:
                rows.append(dict(run_id=f"d{day}_{condition}", date=f"day_{day}", condition=condition,
                                 duration_s=duration, target_reference=level, reference_value="", reference_unit="ppm"))
    elif kind == "real":
        if repeats < 1:
            raise ValueError("repeats必须为正")
        for condition in ["R1_nominal", "R2_low_signal", "R3_wind_change"]:
            for i in range(repeats):
                block = f"{condition}_block_{i+1:03d}"
                for order, method in enumerate(rng.permutation(["main", "b2_fixed", "b3_scan", "b1_reactive"]), 1):
                    rows.append(dict(block_id=block, scenario_id=block, date=f"day_{i*3//repeats+1}",
                        condition=condition, order=order, method=str(method),
                        training_seed=i%5+1 if method!="b1_reactive" else "",
                        start_index=i//5%6, heading_quadrant=i%4, model_sha256="", termination="",
                        success="", elapsed_s="", final_distance="", odor_hit_ratio="",
                        whisker_only_reacquisition_events="", energy_j="", log_path=""))
    else:
        raise ValueError(f"未知排程：{kind}")
    return rows

def holm(values):
    result, running = [0.0]*len(values), 0.0
    for rank, index in enumerate(sorted(range(len(values)), key=values.__getitem__)):
        running = max(running, (len(values)-rank)*values[index])
        result[index] = min(1.0, running)
    return result

def paired_interval(left, right, *, draws=2000, seed=7):
    """交叉重采样训练seed与共享场景；episode不是独立训练重复。"""
    common = set(left)&set(right)
    if not common:
        return dict(status="unpaired", common_episodes=0)
    seeds = sorted({k[0] for k in common})
    scenes = sorted(set.intersection(*[{c for s,c in common if s==item} for item in seeds]))
    if not scenes:
        return dict(status="no_complete_scene_rectangle", common_episodes=len(common))
    matrix = np.asarray([[left[s,c]-right[s,c] for c in scenes] for s in seeds])
    rng = np.random.default_rng(seed)
    samples = [float(matrix[np.ix_(rng.integers(len(seeds), size=len(seeds)),
                                  rng.integers(len(scenes), size=len(scenes)))].mean()) for _ in range(draws)]
    means = matrix.mean(axis=1)
    if len(seeds)<=16:
        null = [abs(float(np.dot(signs,means)/len(seeds))) for signs in itertools.product([-1,1],repeat=len(seeds))]
        p = sum(x>=abs(float(means.mean()))-1e-12 for x in null)/len(null)
    else:
        null = np.abs((rng.choice([-1,1],size=(10000,len(seeds)))*means).mean(axis=1))
        p = float((1+np.count_nonzero(null>=abs(float(means.mean()))-1e-12))/10001)
    return dict(status="paired", difference=float(matrix.mean()),
                ci95=[float(x) for x in np.quantile(samples,[.025,.975])],
                training_seeds=len(seeds), shared_scenarios=len(scenes), common_episodes=len(common),
                excluded_incomplete_episodes=len(common)-matrix.size, seed_sign_permutation_p=p,
                warning="少于5训练seed，仅作探索" if len(seeds)<5 else "种子配对差检验假定可交换；保留所有seed")

def summarize_episodes(rows, *, reference="main", draws=2000):
    metrics = ["odor_hit_ratio","positive_response_trend_ratio","mean_abs_lr_difference","whisker_only_reacquisition_events","reacquisition_events",
               "mean_reacquisition_time_s","success","penalized_time_s","final_distance","path_length"]
    groups = {}
    for row in rows:
        groups.setdefault(str(row["variant"]),[]).append(row)
    summaries, comparisons = {}, []
    for name, items in groups.items():
        values = {}
        for metric in metrics:
            vals = [float(r[metric]) for r in items if r.get(metric) is not None and np.isfinite(float(r[metric]))]
            values[metric] = dict(mean=float(np.mean(vals)) if vals else None,
                median=float(np.median(vals)) if vals else None, available=len(vals), missing=len(items)-len(vals))
        summaries[name] = dict(episodes=len(items),training_seeds=sorted({int(r["training_seed"]) for r in items}),
            termination_counts={v:sum(r.get("termination")==v for r in items)
                                for v in sorted({r.get("termination","unknown") for r in items})},metrics=values)
    for name, items in groups.items():
        if name==reference or reference not in groups:
            continue
        for metric in metrics:
            left = {(int(r["training_seed"]),str(r["scenario_id"])):float(r[metric])
                    for r in items if r.get(metric) is not None}
            right = {(int(r["training_seed"]),str(r["scenario_id"])):float(r[metric])
                     for r in groups[reference] if r.get(metric) is not None}
            comparisons.append(dict(variant=name,reference=reference,metric=metric,
                                    **paired_interval(left,right,draws=draws)))
    for metric in metrics:
        members = [r for r in comparisons if r["metric"]==metric and r["status"]=="paired"]
        for row,p in zip(members,holm([r["seed_sign_permutation_p"] for r in members])):
            row["holm_p_within_metric"] = p
    return dict(groups=summaries,comparisons=comparisons,
                correction_family="每终点内全部预定对比；跨终点主假设需预注册",
                time_definition="成功用到源时间，其余用完整时限；越界不作普通删失")
