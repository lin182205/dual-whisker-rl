"""E0静态标定/漂移及E7真实试验的离线分析；不生成虚构硬件结果。"""
from __future__ import annotations
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from dual_whisker_rl.paths import project_path
from dual_whisker_rl.experiment_protocol import write_json,write_csv,hardware_schedule,summarize_episodes
from dual_whisker_rl.hardware.sensor_preprocess import DualGasPreprocessor,SensorPreprocessConfig

def read_inputs(paths):
    rows = []
    for path in paths:
        with project_path(path).open(encoding="utf-8-sig",newline="") as f:
            for row in csv.DictReader(f):
                if not row.get("run_id"):
                    row["run_id"] = Path(path).stem
                rows.append(row)
    if not rows:
        raise ValueError("输入CSV为空")
    return rows

def number(row,key):
    try:
        value = float(row[key])
    except (ValueError,KeyError,TypeError) as exc:
        raise ValueError(f"缺少有效数值{key}，run={row.get('run_id')}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{key}不能为NaN/Inf")
    return value

def static_analysis(rows,*,tail_s=60,degree=2,holdout_date=None):
    """按平台汇总，整天留出；不会随机拆分同平台的ADC点。"""
    grouped = {}
    units = set()
    for row in rows:
        for key in ("plateau_id","date","sequence_id","direction","reference_unit"):
            if not row.get(key):
                raise ValueError(f"静态标定需要{key}")
        units.add(row["reference_unit"])
        grouped.setdefault((row["run_id"],row["plateau_id"]),[]).append(row)
    if len(units)!=1 or not units<= {"ppm","release_level"}:
        raise ValueError("reference_unit统一为ppm或release_level，不能混用")
    platforms = []
    for (_,plateau),items in grouped.items():
        t = np.asarray([number(r,"t_rel_s") for r in items])
        if len(t)<3 or np.any(np.diff(t)<=0):
            raise ValueError(f"{plateau}至少3个严格递增时间点")
        kept = [r for r,include in zip(items,t>=t[-1]-tail_s) if include]
        reference = np.asarray([number(r,"reference_value") for r in kept])
        if np.ptp(reference)>max(.05*abs(float(np.mean(reference))),1e-6):
            raise ValueError(f"{plateau}参考浓度变化>5%，不能作为稳态平台")
        item = {key:items[0][key] for key in ("run_id","date","sequence_id","direction","reference_unit")}
        item.update(plateau_id=plateau,reference_value=float(reference.mean()),samples=len(kept),
                    observed_tail_s=float(t[-1]-t[t>=t[-1]-tail_s][0]))
        for side in ("left","right"):
            adc = np.asarray([number(r,f"{side}_adc") for r in kept])
            item[f"{side}_adc_mean"] = float(adc.mean())
            item[f"{side}_adc_std"] = float(adc.std(ddof=1))
        platforms.append(item)
    dates = sorted({r["date"] for r in platforms})
    holdout = holdout_date or dates[-1]
    training = [r for r in platforms if r["date"]!=holdout]
    testing = [r for r in platforms if r["date"]==holdout]
    if len(dates)<2 or not testing or len({r["reference_value"] for r in training})<3:
        raise ValueError("至少两天数据、训练日3个不同参考水平和非空留出日")
    model = {}
    for side in ("left","right"):
        x = np.asarray([r[f"{side}_adc_mean"] for r in training])
        y = np.asarray([r["reference_value"] for r in training])
        actual_degree = min(degree,len(np.unique(x))-1)
        if actual_degree<1:
            raise ValueError("ADC无响应变化，无法标定")
        coefficients = np.polyfit(x,y,actual_degree)
        derivative = np.polyval(np.polyder(coefficients),np.linspace(x.min(),x.max(),100))
        if np.any(derivative>0) and np.any(derivative<0):
            coefficients = np.polyfit(x,y,1)
        prediction = np.polyval(coefficients,[r[f"{side}_adc_mean"] for r in testing])
        truth = np.asarray([r["reference_value"] for r in testing])
        errors = prediction-truth
        model[side] = dict(coefficients=coefficients.tolist(),adc_valid_range=[float(x.min()),float(x.max())],
                           holdout_mae=float(np.abs(errors).mean()),holdout_rmse=float(np.sqrt(np.mean(errors**2))))
        for row,pred in zip(testing,prediction):
            row[f"{side}_holdout_prediction"] = float(pred)
    hysteresis = []
    for level in sorted({r["reference_value"] for r in platforms}):
        for side in ("left","right"):
            up = [r[f"{side}_adc_mean"] for r in platforms if r["reference_value"]==level and r["direction"]=="ascending"]
            down = [r[f"{side}_adc_mean"] for r in platforms if r["reference_value"]==level and r["direction"]=="descending"]
            hysteresis.append(dict(reference_value=level,side=side,ascending_n=len(up),descending_n=len(down),
                                   adc_difference=float(np.mean(up)-np.mean(down)) if up and down else None))
    return platforms,dict(model=model,reference_unit=next(iter(units)),holdout_date=holdout,training_platforms=len(training),
        holdout_platforms=len(testing),hysteresis=hysteresis,
        scope="仅当前器件；release_level不能解释为ppm；系数只拟合训练日，不外推")

def residual_block_interval(t,y,*,draws=1000,block=20):
    x = np.column_stack([np.ones(len(t)),t-t[0]])
    coef = np.linalg.lstsq(x,y,rcond=None)[0]
    fitted,residual = x@coef,y-x@coef
    rng = np.random.default_rng(19)
    values = []
    block = max(1,min(int(block),len(t)))
    for _ in range(draws):
        starts = rng.integers(len(t),size=int(np.ceil(len(t)/block)))
        index = np.concatenate([(s+np.arange(block))%len(t) for s in starts])[:len(t)]
        values.append(float(np.linalg.lstsq(x,fitted+residual[index],rcond=None)[0][1]*3600))
    return float(coef[1]*3600),[float(v) for v in np.quantile(values,[.025,.975])]

def drift_analysis(rows,*,scale=200,block_s=60,draws=1000):
    runs = {}
    for row in rows:
        runs.setdefault(row["run_id"],[]).append(row)
    reports,features = [],[]
    for run,items in runs.items():
        t = np.asarray([number(r,"t_rel_s") for r in items])
        if len(t)<3 or np.any(np.diff(t)<=0):
            raise ValueError("漂移记录至少3个严格递增时间点")
        dt = float(np.median(np.diff(t)))
        config = SensorPreprocessConfig(dt_s=dt,scale=scale)
        processors = {
            "frozen": DualGasPreprocessor(config),
            "gated": DualGasPreprocessor(config),
            "relaxed": DualGasPreprocessor(replace(config,baseline_update_signal_threshold=1e9,baseline_update_trend_threshold=1e9)),
        }
        for proc in processors.values():
            proc.reset(number(items[0],"left_adc"),number(items[0],"right_adc"))
        traces = {mode:[] for mode in processors}
        for row in items:
            for mode,proc in processors.items():
                feat = proc.update(number(row,"left_adc"),number(row,"right_adc"),allow_baseline_update=mode!="frozen")
                out = dict(run_id=run,t_rel_s=number(row,"t_rel_s"),mode=mode,
                           left_baseline=feat.left.baseline,right_baseline=feat.right.baseline,
                           left_norm_signal=feat.left.norm_signal,right_norm_signal=feat.right.norm_signal,
                           left_norm_smooth=feat.left.norm_smooth,right_norm_smooth=feat.right.norm_smooth,
                           norm_smooth_diff=feat.norm_smooth_diff)
                traces[mode].append(out)
                features.append(out)
        report = dict(run_id=run,samples=len(t),duration_s=float(t[-1]-t[0]),median_dt_s=dt,
                      modes={},unit="ADC/hour；窗口bootstrap而非采样点独立推断")
        for side in ("left","right"):
            y = np.asarray([number(r,f"{side}_adc") for r in items])
            slope,ci = residual_block_interval(t,y,draws=draws,block=max(1,round(block_s/dt)))
            report[side] = dict(drift_adc_per_hour=slope,ci95=ci,drift_percent_scale_per_hour=100*slope/scale,
                                std=float(np.std(y,ddof=1)),cv=float(np.std(y,ddof=1)/abs(y.mean())) if abs(y.mean())>1e-8 else None)
        for mode,data in traces.items():
            report["modes"][mode] = {f"{side}_smooth_mean":float(np.mean([r[f"{side}_norm_smooth"] for r in data]))
                                     for side in ("left","right")}
        report["caution"] = "原始ADC斜率包含浓度/温湿度/电源变化；需参考仪器核实，不能自动归因器件漂移"
        reports.append(report)
    return reports,features

def analyze_trials(rows,horizon):
    converted = []
    for row in rows:
        success_text = row.get("success","").strip().lower()
        if success_text not in ("0","1","true","false"):
            raise ValueError("真实试验必须补全success，不能分析空排程")
        success = success_text in ("1","true")
        elapsed = number(row,"elapsed_s")
        if elapsed<0 or elapsed>horizon or not row.get("termination") or not row.get("scenario_id"):
            raise ValueError("终止原因、scene或elapsed不合法")
        if success != (row["termination"]=="success"):
            raise ValueError("success与termination矛盾")
        seed = int(row.get("training_seed") or 0)
        item = dict(variant=row["method"],training_seed=seed,scenario_id=row["scenario_id"],
                    success=success,termination=row["termination"],penalized_time_s=elapsed if success else horizon,
                    date=row.get("date"),condition=row.get("condition"))
        for key in ("final_distance","odor_hit_ratio","whisker_only_reacquisition_events","energy_j"):
            item[key] = number(row,key) if row.get(key) else None
        converted.append(item)
    report = summarize_episodes(converted)
    report["rule_seed_note"] = "规则法seed=0仅表示无训练；不制造算法重复，与学习组未配对时不输出伪p值"
    return converted,report

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["static","drift","trials"])
    p.add_argument("--input",nargs="+",required=True)
    p.add_argument("--reference-csv", help="外部仪器的平台均值表：plateau_id,reference_value,reference_unit")
    p.add_argument("--output-dir",required=True)
    p.add_argument("--steady-tail-s",type=float,default=60)
    p.add_argument("--degree",type=int,choices=[1,2],default=2)
    p.add_argument("--holdout-date")
    p.add_argument("--scale",type=float,default=200)
    p.add_argument("--block-s",type=float,default=60)
    p.add_argument("--bootstrap",type=int,default=1000)
    p.add_argument("--horizon-s",type=float,default=80)
    args = p.parse_args()
    if min(args.steady_tail_s,args.scale,args.block_s,args.bootstrap,args.horizon_s)<=0:
        p.error("时长、scale和bootstrap必须为正")
    out = project_path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("分析目录已有内容，请使用新目录")
    rows = read_inputs(args.input)
    if args.reference_csv:
        references = read_inputs([args.reference_csv])
        lookup = {}
        for item in references:
            key = item.get("plateau_id")
            if not key or key in lookup:
                raise ValueError("参考表plateau_id必须非空且唯一")
            number(item, "reference_value")
            lookup[key] = item
        for row in rows:
            key = row.get("plateau_id")
            if key not in lookup:
                raise ValueError(f"参考表缺少平台：{key}")
            row["reference_value"] = lookup[key]["reference_value"]
            row["reference_unit"] = lookup[key]["reference_unit"]
    if args.mode=="static":
        table,report = static_analysis(rows,tail_s=args.steady_tail_s,degree=args.degree,holdout_date=args.holdout_date)
        write_csv(out/"platforms.csv",table)
    elif args.mode=="drift":
        report,features = drift_analysis(rows,scale=args.scale,block_s=args.block_s,draws=args.bootstrap)
        write_csv(out/"preprocess_comparison.csv",features)
    else:
        table,report = analyze_trials(rows,args.horizon_s)
        write_csv(out/"trials.csv",table)
    from dual_whisker_rl.e4_experiments import sha256_file
    from dual_whisker_rl.paths import portable_path
    inputs = list(args.input) + ([args.reference_csv] if args.reference_csv else [])
    write_json(out/"inputs.json", {"mode": args.mode, "files": [{"path": portable_path(item), "sha256": sha256_file(project_path(item))} for item in inputs]})
    write_json(out/"analysis.json",report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
