"""E3–E6实验选择、共享训练去重和逐实验论文报告调度。"""
from __future__ import annotations
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from dual_whisker_rl.experiment_protocol import write_csv, write_json
from dual_whisker_rl.paths import portable_path
from scripts.experiment_reports import E5_GROUPS, export_report
ROOT=Path(__file__).resolve().parents[1]
EXPERIMENTS=("e3","e4","e5","e6")


def selected_experiments(args):
    tokens=getattr(args,"experiments",None)
    items=[n.strip().lower() for token in tokens for n in token.split(",")] if tokens else list(EXPERIMENTS)
    if items==["all"]: items=list(EXPERIMENTS)
    if not items or len(set(items))!=len(items) or set(items)-set(EXPERIMENTS):
        raise ValueError("--experiments只接受e3/e4/e5/e6或all，可空格或逗号分隔，不可重复")
    forced={"robustness":"e6","e4":"e4"}.get(args.phase)
    if forced:
        if tokens and items!=[forced]:
            raise ValueError(f"{args.phase}阶段仅对应{forced}，选择多个实验请使用run或evaluate阶段")
        return [forced]
    return [n for n in EXPERIMENTS if n in items]


def variant_plan(args,spec,names):
    custom=bool(args.variants or args.group or args.model_path)
    e5=list(dict.fromkeys(n for group in E5_GROUPS for n in spec["groups"][group]))
    return {"e3":list(names),"e4":list(names) if args.model_path else [],
            "e5":list(names) if custom else e5,"e6":list(names)}


def legacy_command(args,root,profile,stage):
    command=[sys.executable,str(ROOT/"scripts/run_e4_experiments.py"),stage,
        "--profile",args.profile,"--config",str(ROOT/f"configs/e4_{args.profile}.yaml"),
        "--output-dir",str(root/"e4"),"--max-steps",str(profile["max_steps"])]
    for flag,value in [("--timesteps",args.timesteps),("--n-envs",args.n_envs),("--methods",args.e4_methods)]:
        if value is not None: command += [flag,str(value)]
    if args.resume: command.append("--resume")
    return command


def suite_plan(args,spec,profile,names):
    from scripts.run_e4_experiments import METHOD_ORDER, TRAINABLE_METHODS, profile_defaults
    selected=selected_experiments(args)
    variants=variant_plan(args,spec,names)
    native=set(n for e in selected if e!="e4" for n in variants[e]) if not args.model_path else set()
    legacy_methods=args.e4_methods.split(",") if args.e4_methods else list(METHOD_ORDER)
    if len(set(legacy_methods))!=len(legacy_methods) or set(legacy_methods)-set(METHOD_ORDER):
        raise ValueError("--e4-methods包含未知或重复原E4方法")
    defaults=profile_defaults(args.profile)
    legacy_count=(sum(n in TRAINABLE_METHODS for n in legacy_methods)*len(defaults["seeds"])) if "e4" in selected and not args.model_path else 0
    return dict(selected_experiments=selected,native_variants=variants,
        native_training_tasks=len(native)*len(profile["seeds"]),e4_training_tasks=legacy_count,
        training_tasks=len(native)*len(profile["seeds"])+legacy_count,
        e4_methods=legacy_methods if not args.model_path else [],
        e4_timesteps=args.timesteps or defaults["timesteps"],
        evidence_limitations=(["E3需要配套验证日志；E5需要分别重训的对照模型；单模型复评不构成完整方法对比"] if args.model_path else []))


def run_suite(args,root,names,spec,profile,cfg,manifest,*,train_one,evaluate,summarize):
    selected=selected_experiments(args)
    variants=variant_plan(args,spec,names)
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_root=root/"reports"/stamp
    report_root.mkdir(parents=True,exist_ok=False)
    trained=set()
    receipts=[]
    source=manifest.get("external_model")
    if source:
        print(f"[已有模型] {source['path']} 训练seed={source['training_seed']} 已训练步数={source['actual_timesteps']}；跳过训练",flush=True)
    def train(methods):
        for name in methods:
            for seed in profile["seeds"]:
                if (name,seed) not in trained:
                    print(f"[训练] {name} seed={seed}",flush=True)
                    train_one(root,name,spec["variants"][name],seed,profile,cfg,manifest,args)
                    trained.add((name,seed))
    def legacy(stage):
        subprocess.run(legacy_command(args,root,profile,stage),cwd=ROOT,check=True)
    for index,experiment in enumerate(selected,1):
        print(f"[实验 {index}/{len(selected)}] {experiment.upper()} 开始",flush=True)
        methods=variants[experiment]
        failure=None
        legacy_root=root/"e4"/args.profile if experiment=="e4" and not source else None
        try:
            if experiment=="e3":
                if args.phase in ("train","run") and not source: train(methods)
            elif experiment=="e4":
                if source:
                    if args.phase in ("evaluate","run"):
                        evaluate(root,methods,spec,profile,cfg,manifest,args)
                        summarize(root,methods,profile)
                else:
                    stage=args.e4_stage if args.phase=="e4" else args.phase
                    if stage=="train": legacy("calibrate")
                    if stage!="summarize" or (legacy_root/"manifest.json").exists(): legacy(stage)
                    if stage=="evaluate": legacy("summarize")
            elif experiment=="e5":
                if not source:
                    if args.phase in ("train","run"): train(methods)
                    if args.phase in ("evaluate","run"):
                        evaluate(root,methods,spec,profile,cfg,manifest,args)
                        summarize(root,methods,profile)
            elif experiment=="e6":
                if args.phase in ("train","run") and not source: train(methods)
                if args.phase in ("evaluate","run","robustness"):
                    evaluate(root,methods,spec,profile,cfg,manifest,args,robust=True)
                    summarize(root,methods,profile,robust=True)
        except Exception as exc:
            failure=exc
            print(f"[{experiment.upper()} 失败] {exc}",flush=True)
        try:
            receipt=export_report(experiment,root,report_root/experiment,args,methods,spec,profile,manifest,
                                  legacy_root=legacy_root,error=failure)
        except Exception as exc:
            failure=exc
            receipt=export_report(experiment,root,report_root/experiment,args,methods,spec,profile,manifest,error=exc)
        receipts.append(receipt)
        print(f"[{experiment.upper()} 结果] {receipt['status']}；{receipt['report']}",flush=True)
        write_csv(report_root/"experiment_status.csv",receipts)
        write_json(report_root/"run_manifest.json",dict(selected_experiments=selected,phase=args.phase,
            experiment_signature=manifest["signature"],results=receipts,bootstrap_draws=args.bootstrap))
        text=["# 仿真实验结果索引", "", "| 实验 | 状态 | 图片数 | CSV数 | 报告 |", "|---|---|---:|---:|---|"]
        text += [f"| {r['experiment'].upper()} | {r['status']} | {r['figures']} | {r['tables']} | [查看]({r['experiment']}/report.md) |" for r in receipts]
        (report_root/"index.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    print(f"[图表与表格] {portable_path(report_root/'index.md')}",flush=True)
    if any(r["status"]=="failed" for r in receipts):
        raise RuntimeError("部分实验失败，详情见experiment_status.csv和各实验report.md")
    return report_root
