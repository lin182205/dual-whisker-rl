"""汇总独立训练学习曲线和冻结模型的故障退化；不重新训练或挑测试模型。"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from dual_whisker_rl.paths import project_path
from dual_whisker_rl.experiment_protocol import write_json,write_csv,paired_interval

def learning(root,names,draws,*,seeds=None):
    curves,seed_rows = [],[]
    rng = np.random.default_rng(73)
    for name in names:
        histories = {}
        for task in (root/name).glob("seed_*"):
            state_path = task/"state.json"
            if not state_path.exists() or json.loads(state_path.read_text(encoding="utf-8"))["status"]!="completed":
                continue
            seed = int(task.name.removeprefix("seed_"))
            if seeds is not None and seed not in seeds:
                continue
            history_path = task/"validation_history.json"
            if not history_path.exists():
                continue
            items = json.loads(history_path.read_text(encoding="utf-8"))
            if not items: continue
            histories[seed] = {r["timesteps"]:r for r in items}
            steps = np.array(sorted(histories[seed]),dtype=float)
            success = np.array([histories[seed][int(s)]["success_rate"] for s in steps])
            auc = float(np.sum(np.diff(steps)*(success[:-1]+success[1:])/2)/(steps[-1]-steps[0])) if len(steps)>1 else None
            crossing = next((int(steps[i]) for i in range(2,len(steps)) if np.all(success[i-2:i+1]>=.8)),None)
            seed_rows.append(dict(variant=name,training_seed=seed,observed_auc=auc,observed_start=int(steps[0]),
                observed_end=int(steps[-1]),reached_80_percent_three_evaluations_at=crossing,
                collapse=bool(len(success)>=2 and np.all(success[-2:]<=success.max()-.2)),
                last_five_success_range=float(np.ptp(success[-5:])),evaluation_count=len(steps)))
        if not histories: continue
        common = sorted(set.intersection(*(set(h) for h in histories.values())))
        for step in common:
            for metric in ("success_rate","mean_penalized_time_s","odor_hit_ratio","whisker_only_reacquisition_events"):
                values = np.asarray([h[step][metric] for h in histories.values()])
                ci = None
                if len(values)>1:
                    samples = values[rng.integers(len(values),size=(draws,len(values)))].mean(axis=1)
                    ci = [float(x) for x in np.quantile(samples,[.025,.975])]
                curves.append(dict(variant=name,timesteps=step,metric=metric,mean=float(values.mean()),
                                   ci95_low=ci[0] if ci else None,ci95_high=ci[1] if ci else None,
                                   training_seeds=len(values)))
    return curves,seed_rows

def robustness(root,names,draws):
    rows = []
    for name in names:
        nominal = {}
        for task in (root/name).glob("seed_*"):
            for path in (task/"robustness"/"nominal").glob("*.json"):
                item = json.loads(path.read_text(encoding="utf-8"))
                nominal[item["training_seed"],item["scenario_id"]] = item
        if not nominal: continue
        faults = sorted({p.name for task in (root/name).glob("seed_*")
                         for p in (task/"robustness").glob("*") if p.is_dir() and p.name!="nominal"})
        for fault in faults:
            treated = {}
            for task in (root/name).glob("seed_*"):
                for path in (task/"robustness"/fault).glob("*.json"):
                    item = json.loads(path.read_text(encoding="utf-8"))
                    treated[item["training_seed"],item["scenario_id"]] = item
            for metric in ("success","odor_hit_ratio","penalized_time_s","whisker_only_reacquisition_events"):
                left = {k:float(v[metric]) for k,v in treated.items() if v.get(metric) is not None}
                right = {k:float(v[metric]) for k,v in nominal.items() if v.get(metric) is not None}
                paired = paired_interval(left,right,draws=draws)
                common = set(left)&set(right)
                baseline = float(np.mean([right[k] for k in common])) if common else None
                value = float(np.mean([left[k] for k in common])) if common else None
                rows.append(dict(variant=name,fault=fault,metric=metric,nominal=baseline,perturbed=value,
                                 ratio=value/baseline if baseline is not None and baseline>1e-8 else None,
                                 ratio_direction="越大越差" if metric=="penalized_time_s" else "越大越好",**paired))
    return rows

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["learning","robustness"])
    p.add_argument("--experiment-dir",required=True,help="实际profile目录，如results/red_experiments/smoke")
    p.add_argument("--variants",default="main,fixed,periodic,single_frame,history_mlp,no_dr")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--bootstrap",type=int,default=2000)
    args = p.parse_args()
    if args.bootstrap<1: p.error("bootstrap必须为正")
    root,out = project_path(args.experiment_dir),project_path(args.output_dir)
    if (root/".run.lock").exists(): raise RuntimeError("实验仍在运行，等待完成后再分析")
    if out.exists() and any(out.iterdir()): raise FileExistsError("输出目录已存在，请使用新分析目录")
    names = args.variants.split(",")
    if args.mode=="learning":
        curves,seeds = learning(root,names,args.bootstrap)
        if not curves: raise ValueError("没有已完成训练的验证曲线")
        write_csv(out/"learning_curves.csv",curves)
        write_csv(out/"training_seed_metrics.csv",seeds)
        write_json(out/"analysis.json",dict(training_seeds=seeds,curves=curves,
                                          note="区间按独立训练seed重采样；单seed不输出算法级CI；AUC仅覆盖实测区间"))
    else:
        rows = robustness(root,names,args.bootstrap)
        if not rows: raise ValueError("需先在同一批场景完成nominal与故障评估")
        write_json(out/"degradation.json",rows)
        flat = [{k:v for k,v in r.items() if k!="ci95"}|dict(ci95_low=r.get("ci95",[None,None])[0],
                                                         ci95_high=r.get("ci95",[None,None])[1]) for r in rows]
        write_csv(out/"degradation.csv",flat)
    print(f"分析已保存到 {out}")

if __name__=="__main__":
    main()
