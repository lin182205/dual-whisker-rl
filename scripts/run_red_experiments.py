"""执行E3/E4/E5/E6仿真对比：固定验证训练、消融与冻结模型扰动评估。"""
from __future__ import annotations
import os
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
from functools import partial
import gzip
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from dual_whisker_rl.paths import project_path, portable_path
from scripts.experiment_progress import EvaluationProgress, reporting_compatible_payload, configure_output_encoding
from dual_whisker_rl.e4_experiments import build_scenarios, E4Scenario, metrics_from_trajectory, sha256_file, choose_mlp_width
from dual_whisker_rl.experiment_env import make_experiment_env
from dual_whisker_rl.experiment_protocol import digest, seal, write_json, write_csv, summarize_episodes

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=["plan","prepare","train","evaluate","robustness","summarize","status","run","e4"])
    p.add_argument("--config",default="configs/red_experiments.yaml")
    p.add_argument("--profile",choices=["smoke","formal"],default="smoke")
    p.add_argument("--output-dir",default="results/simulation_experiments")
    p.add_argument("--group",default="core")
    p.add_argument("--variants",help="逗号分隔；覆盖group")
    p.add_argument("--faults",default="nominal,wind24,gain80,servo150,delay2")
    p.add_argument("--timesteps",type=int)
    p.add_argument("--n-envs",type=int)
    p.add_argument("--max-steps",type=int)
    p.add_argument("--resume",action="store_true")
    p.add_argument("--dry-run",action="store_true")
    p.add_argument("--limit",type=int,help="仅用于故障评估；formal默认200共享场景")
    p.add_argument("--e4-stage",choices=["prepare","calibrate","train","evaluate","summarize","run","status"],default="run")
    return p

def resolve(args):
    spec = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    scope = spec.get("scope", "simulation_only")
    if scope != "simulation_only":
        raise ValueError("当前入口仅支持仿真对比，配置scope应为simulation_only")
    profile = dict(spec["profiles"][args.profile])
    for key in ("timesteps","n_envs","max_steps"):
        if getattr(args,key) is not None:
            profile[key] = getattr(args,key)
    for key in ("timesteps","n_envs","max_steps","n_steps","batch_size","eval_every","validation_scenarios","test_scenarios"):
        if int(profile[key])<1:
            raise ValueError(f"{key}必须为正")
    if profile["n_steps"]*profile["n_envs"] % profile["batch_size"]:
        raise ValueError("rollout步数必须能整除batch_size")
    if len(profile["seeds"])!=len(set(profile["seeds"])) or not profile["seeds"]:
        raise ValueError("训练种子不能为空或重复")
    if args.group not in spec["groups"] and not args.variants:
        raise ValueError("未知实验组")
    names = args.variants.split(",") if args.variants else spec["groups"][args.group]
    names = [n.strip() for n in names]
    if not names or len(set(names))!=len(names) or set(names)-set(spec["variants"]):
        raise ValueError(f"未知或重复变体：{names}")
    faults = [n.strip() for n in args.faults.split(",")]
    if set(faults)-set(spec["faults"]) or len(faults)!=len(set(faults)):
        raise ValueError(f"未知或重复故障：{faults}")
    args.faults = ",".join(faults)
    if args.limit is not None and args.limit<1:
        raise ValueError("limit必须为正")
    cfg = yaml.safe_load(project_path(spec["base_config"]).read_text(encoding="utf-8"))
    cfg.update(spec.get("common",{}))
    cfg["max_steps"] = int(profile["max_steps"])
    if cfg.get("plume_model","dynamic")!="dynamic":
        raise ValueError("红字实验只支持dynamic Puff")
    code_paths = sorted((ROOT/"dual_whisker_rl").rglob("*.py")) + [Path(__file__)]
    code = {portable_path(p):sha256_file(p) for p in code_paths}
    payload = dict(version=2,scope=scope,profile=args.profile,settings=profile,base_config=cfg,
                   variants=spec["variants"],faults=spec["faults"],code_sha256=code,
                   selection="validation_success_then_penalized_time",
                   success="radius_and_continuous_hold",time_unit="seconds")
    return spec,profile,cfg,names,faults,payload

def prepare(root,profile,cfg,payload):
    scenarios = {}
    for split,count in [("validation",profile["validation_scenarios"]),("test",profile["test_scenarios"])]:
        candidates = build_scenarios(split=split,profile="formal",world_bounds=cfg.get("world_bounds"))
        if count>len(candidates):
            raise ValueError(f"{split}只有{len(candidates)}场景，不能要求{count}")
        rng = np.random.default_rng(20261013+(1 if split=="validation" else 2))
        indices = sorted(rng.choice(len(candidates),size=count,replace=False).tolist())
        scenarios[split] = [candidates[i].to_dict() for i in indices]
    candidate = {**payload, "scenarios": scenarios}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        compatible = reporting_compatible_payload(candidate, saved)
    else:
        compatible = candidate
    signed = seal(manifest_path, compatible)
    if compatible is not candidate:
        write_json(root / "reporting_upgrade.json", dict(
            reason="evaluation_eta_only", preserved_signature=signed["signature"],
            sealed_runner_sha256=compatible["code_sha256"]["scripts/run_red_experiments.py"],
            current_runner_sha256=payload["code_sha256"]["scripts/run_red_experiments.py"]))
    for split,items in scenarios.items():
        scene_path = root/f"scenarios_{split}.json"
        if scene_path.exists():
            if json.loads(scene_path.read_text(encoding="utf-8")) != items:
                raise ValueError("已封存场景文件遭到修改，拒绝静默修复或复用")
        else:
            write_json(scene_path,items)
    return signed

def scenarios(root,split):
    return [E4Scenario.from_dict(v) for v in json.loads((root/f"scenarios_{split}.json").read_text(encoding="utf-8"))]

def rollout(model,env,scene,*,save_steps=False,probes=None):
    obs,info = env.reset(seed=scene.seed,options={"scenario":scene.to_dict()})
    steps,step_rows,total = 0,[],0.0
    started = time.perf_counter()
    initial_distance = float(info["distance_to_source"])
    while True:
        if probes is not None and len(probes) < 64:
            probes.append(np.asarray(obs).copy())
        before = time.perf_counter()
        action,_ = model.predict(obs,deterministic=True)
        inference_ms = (time.perf_counter()-before)*1000
        input_frame = np.asarray(obs)[-env.base_observation_dim:].tolist()
        obs,reward,terminated,truncated,info = env.step(action)
        total += float(reward)
        steps += 1
        if save_steps:
            step_rows.append({**env.unwrapped.trajectory[-1],"step":steps,"elapsed_s":steps*env.unwrapped.dt,
                              "wind_speed":float(info["wind_speed"]),"wind_direction":float(info["wind_direction"]),"policy_frame":input_frame,"policy_action":np.asarray(action).tolist(),"inference_ms":inference_ms})
        if terminated or truncated:
            break
    base = env.unwrapped
    row = metrics_from_trajectory(base.trajectory,base.goal_radius,(base.world_min,base.world_max))
    success = bool(info["is_success"])
    reading = np.asarray([[r["left"],r["right"]] for r in base.trajectory])
    row.update(success=success,termination="success" if success else "out_of_bounds" if info["out_of_bounds"] else "timeout",
        scenario_id=scene.scenario_id,split=scene.split,scenario_seed=scene.seed,
        condition=f"{scene.distance_bin}/{scene.crosswind_bin}",wind_speed=float(base.plume.wind_speed_mean),
        nominal_wind_speed=scene.wind_speed,
        elapsed_s=steps*base.dt,penalized_time_s=steps*base.dt if success else base.max_steps*base.dt,
        horizon_s=base.max_steps*base.dt,initial_distance=initial_distance,
        path_length=float(sum(r["position_displacement"] for r in base.trajectory)),
        positive_response_trend_ratio=float(np.mean(np.diff(reading,axis=0)/base.dt>.01)) if steps>1 else None,
        mean_abs_lr_difference=float(np.abs(reading[:,0]-reading[:,1]).mean()),
        reward=total,reward_components={key:float(sum(r["reward_components"][key] for r in base.trajectory))
                                      for key in base.REWARD_COMPONENT_NAMES},
        evaluation_wall_s=time.perf_counter()-started)
    return row,step_rows

def vector_factory(cfg,variant,seed):
    from stable_baselines3.common.monitor import Monitor
    return Monitor(make_experiment_env(cfg,variant,seed=seed))

def train_one(root,name,variant,seed,profile,cfg,manifest,args):
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
    from dual_whisker_rl.agents import GRUHistoryExtractor
    directory = root/name/f"seed_{seed}"
    directory.mkdir(parents=True,exist_ok=True)
    state_path = directory/"state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if state:
        if state.get("signature")!=manifest["signature"]:
            raise ValueError(f"{name}旧训练签名不匹配")
        if not args.resume:
            raise FileExistsError(f"实验已存在，使用--resume或新目录：{directory}")
        if state["status"]=="completed":
            if sha256_file(directory/"selected_model.zip")!=state["selected_model_sha256"]:
                raise ValueError("选中模型校验失败")
            return
    write_json(state_path,dict(status="running",signature=manifest["signature"],variant=name,training_seed=seed))
    fns = [partial(vector_factory,cfg,variant,seed+i) for i in range(profile["n_envs"])]
    vec = SubprocVecEnv(fns,start_method="spawn") if profile["backend"]=="subproc" else DummyVecEnv(fns)
    eval_env = make_experiment_env(cfg,variant,seed=seed,evaluating=True)
    validation = scenarios(root,"validation")
    history_path = directory/"validation_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8")) if args.resume and history_path.exists() else []
    best_key = max([(r["success_rate"],-r["mean_penalized_time_s"]) for r in history],default=(-1,float("-inf")))
    class Diagnostics(BaseCallback):
        def __init__(self):
            super().__init__()
            self.last_eval = history[-1]["timesteps"] if history else 0
        def evaluate(self):
            nonlocal best_key
            probes = []
            rows = [rollout(self.model,eval_env,s,probes=probes)[0] for s in validation]
            with torch.no_grad():
                tensor = self.model.policy.obs_to_tensor(np.asarray(probes))[0]
                distributions = self.model.policy.get_distribution(tensor).distribution
                heads = distributions if isinstance(distributions, list) else [distributions]
                head_entropy = [float(head.entropy().mean().cpu()) for head in heads]
            gradient_norm = float(sum(float(p.grad.detach().square().sum().cpu()) for p in self.model.policy.parameters() if p.grad is not None) ** .5)
            item = dict(timesteps=int(self.model.num_timesteps),success_rate=float(np.mean([r["success"] for r in rows])),
                        mean_penalized_time_s=float(np.mean([r["penalized_time_s"] for r in rows])),
                        odor_hit_ratio=float(np.mean([r["odor_hit_ratio"] for r in rows])),
                        whisker_only_reacquisition_events=float(np.mean([r["whisker_only_reacquisition_events"] for r in rows])),
                        reward_components={k:float(np.mean([r["reward_components"][k] for r in rows])) for k in rows[0]["reward_components"]},
                        ppo={k:float(v) for k,v in self.model.logger.name_to_value.items()
                             if k.startswith("train/") and isinstance(v,(int,float,np.number)) and np.isfinite(v)})
            item["action_head_entropy"] = head_entropy
            item["gradient_norm"] = gradient_norm
            history.append(item)
            write_json(history_path,history)
            key = (item["success_rate"],-item["mean_penalized_time_s"])
            if key>best_key:
                self.model.save(directory/"selected_model.zip")
                best_key = key
                write_json(directory/"selection.json",item)
            for key,value in item.items():
                if isinstance(value,(int,float)):
                    self.logger.record(f"red_validation/{key}",value)
            self.last_eval = int(self.model.num_timesteps)
            print(f"[验证] {name} seed={seed} step={self.last_eval} success={item['success_rate']:.3f} hit={item['odor_hit_ratio']:.3f}",flush=True)
        def _on_step(self):
            return True
        def _on_rollout_start(self):
            if self.model.num_timesteps-self.last_eval>=profile["eval_every"]:
                self.evaluate()
            if self.model.num_timesteps > 0:
                self.model.save(directory/"last_model.zip")
        def _on_training_end(self):
            if self.last_eval!=self.model.num_timesteps:
                self.evaluate()
            self.model.save(directory/"last_model.zip")
    try:
        length = int(variant.get("history_length",20))
        width = choose_mlp_width(vec.observation_space.shape[0]) if variant.get("method")=="b5_history_mlp" else 160
        kwargs = dict(net_arch=[width,width],share_features_extractor=True)
        if variant.get("encoder","gru")=="gru":
            kwargs.update(features_extractor_class=GRUHistoryExtractor,
                features_extractor_kwargs=dict(history_length=length,base_observation_dim=int(vec.observation_space.shape[0]//length)))
        checkpoint = directory/"last_model.zip"
        if args.resume and checkpoint.exists():
            model = PPO.load(checkpoint,env=vec,device=profile.get("device", "cpu"))
        else:
            model = PPO("MlpPolicy",vec,learning_rate=1e-4,n_steps=profile["n_steps"],batch_size=profile["batch_size"],
                n_epochs=5,gamma=.99,gae_lambda=.95,ent_coef=.003,target_kl=.02,seed=seed,
                policy_kwargs=kwargs,verbose=0,device=profile.get("device", "cpu"),tensorboard_log=str(directory/"tensorboard"))
        model.learn(total_timesteps=max(0,profile["timesteps"]-model.num_timesteps),reset_num_timesteps=not args.resume,
                    callback=Diagnostics(),tb_log_name=name)
        model.save(directory/"final_model.zip")
        write_json(directory/"metadata.json",dict(variant=variant,training_seed=seed,signature=manifest["signature"],
            actual_timesteps=int(model.num_timesteps),parameter_count=sum(p.numel() for p in model.policy.parameters()),
            history_length=length,base_observation_dim=int(vec.observation_space.shape[0]//length),
            model_selection="固定验证成功率优先，再按失败惩罚时间；并保留final模型"))
        write_json(state_path,dict(status="completed",signature=manifest["signature"],
            actual_timesteps=int(model.num_timesteps),selected_model_sha256=sha256_file(directory/"selected_model.zip")))
    except BaseException as exc:
        write_json(state_path,dict(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
            signature=manifest["signature"],error=repr(exc)))
        raise
    finally:
        vec.close()
        eval_env.close()

def evaluate(root,names,spec,profile,cfg,manifest,args,*,robust=False):
    from stable_baselines3 import PPO
    from dual_whisker_rl.agents import GRUHistoryExtractor
    _ = GRUHistoryExtractor
    entries = scenarios(root,"test")
    if robust:
        limit = args.limit if args.limit is not None else min(200,len(entries))
        if limit>len(entries):
            raise ValueError("故障评估limit超过已封存测试集")
        rng = np.random.default_rng(20261016)
        entries = [entries[i] for i in sorted(rng.choice(len(entries),limit,replace=False))]
    faults = args.faults.split(",") if robust else ["nominal"]
    # 先校验本轮所有缓存，剩余工作量包含全部方法、种子和故障。
    jobs = []
    for name in names:
        variant = spec["variants"][name]
        for seed in profile["seeds"]:
            task = root/name/f"seed_{seed}"
            model_path = task/"selected_model.zip"
            model_sha = sha256_file(model_path)
            state = json.loads((task/"state.json").read_text(encoding="utf-8"))
            if state["status"]!="completed" or state["signature"]!=manifest["signature"] or model_sha!=state["selected_model_sha256"]:
                raise ValueError(f"模型尚未完成或签名不匹配：{task}")
            for label in faults:
                fault = spec["faults"][label]
                folder = task/("robustness" if robust else "episodes")/label
                evaluation_signature = digest(dict(manifest=manifest["signature"],model=model_sha,fault=fault,
                                                     scenes=[s.to_dict() for s in entries]))
                cached = set()
                for scene in entries:
                    output = folder/f"{scene.scenario_id}.json"
                    if not output.exists():
                        continue
                    if not args.resume:
                        raise FileExistsError(f"评估已存在：{output}")
                    old = json.loads(output.read_text(encoding="utf-8"))
                    if old.get("evaluation_signature")!=evaluation_signature or old.get("scenario_id")!=scene.scenario_id:
                        raise ValueError("旧评估签名或场景不匹配，使用新的输出目录")
                    cached.add(scene.scenario_id)
                jobs.append(dict(name=name, seed=seed, label=label, variant=variant,
                                 model_path=model_path, model_sha=model_sha, fault=fault, folder=folder,
                                 signature=evaluation_signature, cached=cached))
    progress = EvaluationProgress(len(entries)*len(jobs), cached=sum(len(j["cached"]) for j in jobs))
    stage = "扰动评估" if robust else "评估"
    progress.report(f"[{stage}] 本轮共{len(jobs)}组", force=True)
    loaded_path, model = None, None
    for index, job in enumerate(jobs, 1):
        if len(job["cached"]) == len(entries):
            continue
        name, seed, label = job["name"], job["seed"], job["label"]
        if loaded_path != job["model_path"]:
            model = PPO.load(job["model_path"],device="cpu")
            loaded_path = job["model_path"]
        env = make_experiment_env(cfg,job["variant"],seed=seed,evaluating=True,fault=job["fault"])
        prefix = f"[{stage}] 组={index}/{len(jobs)} {name} seed={seed} fault={label}"
        progress.report(prefix, force=True)
        try:
            if env.observation_space!=model.observation_space or env.action_space!=model.action_space:
                raise ValueError("模型与故障环境的观测/动作契约不一致")
            for scene in entries:
                if scene.scenario_id in job["cached"]:
                    continue
                started = time.perf_counter()
                row,steps = rollout(model,env,scene,save_steps=profile["save_step_logs"])
                row.update(variant=name,training_seed=seed,fault=label,model_sha256=job["model_sha"],
                           evaluation_signature=job["signature"], evaluation_scene_count=len(entries))
                folder = job["folder"]
                if steps:
                    folder.mkdir(parents=True,exist_ok=True)
                    with gzip.open(folder/f"{scene.scenario_id}.steps.jsonl.gz","wt",encoding="utf-8") as f:
                        for step in steps:
                            f.write(json.dumps(step,ensure_ascii=False,allow_nan=False)+"\n")
                write_json(folder/f"{scene.scenario_id}.json",row)
                progress.finish_episode(time.perf_counter() - started)
                progress.report(prefix)
        finally:
            env.close()

def summarize(root,names,profile,*,robust=False):
    for label in (["nominal"] if not robust else sorted({p.name for name in names for seed in profile["seeds"]
                                                         for p in (root/name/f"seed_{seed}"/"robustness").glob("*") if p.is_dir()})):
        rows = []
        for name in names:
            for seed in profile["seeds"]:
                rows.extend(json.loads(p.read_text(encoding="utf-8")) for p in
                    (root/name/f"seed_{seed}"/("robustness" if robust else "episodes")/label).glob("*.json"))
        if not rows:
            continue
        result = summarize_episodes(rows)
        result["expected_episodes_per_variant"] = len(scenarios(root,"test"))*len(profile["seeds"]) if not robust else None
        result["independent_training_seed_count"] = len(profile["seeds"])
        counts = {name:sum(r["variant"] == name for r in rows) for name in names}
        expected = (rows[0].get("evaluation_scene_count", len(scenarios(root, "test"))) if robust else len(scenarios(root, "test"))) * len(profile["seeds"])
        result["coverage"] = {name:{"actual":count,"expected":expected,"complete":count == expected} for name,count in counts.items()}
        write_json(root/"summaries"/f"{'robustness_' if robust else ''}{label}.json",result)
        flat = [{k:v for k,v in row.items() if k!="reward_components"} for row in rows]
        write_csv(root/"summaries"/f"{'robustness_' if robust else ''}{label}_episodes.csv",flat)

def main():
    configure_output_encoding()
    args = parser().parse_args()
    spec,profile,cfg,names,faults,payload = resolve(args)
    root = project_path(args.output_dir)/args.profile
    if args.phase=="e4":
        command = [sys.executable,str(ROOT/"scripts/run_e4_experiments.py"),args.e4_stage,"--profile",args.profile,
                   "--config",str(ROOT/f"configs/e4_{args.profile}.yaml"),"--output-dir",str(root/"e4")]
        if args.resume: command.append("--resume")
        if args.dry_run:
            print(json.dumps(command,ensure_ascii=False)); return
        subprocess.run(command,cwd=ROOT,check=True); return
    if args.phase=="plan" or args.dry_run:
        print(json.dumps(dict(output=portable_path(root),profile=profile,variants=names,
                              faults=faults,training_tasks=len(names)*len(profile["seeds"]),
                              simulation_only=True,active_experiments=["E3","E4","E5","E6"],deferred_experiments=["E0","E7"],signature=digest(payload)),ensure_ascii=False,indent=2)); return
    manifest = prepare(root,profile,cfg,payload)
    if args.phase=="prepare": return
    if args.phase=="status":
        for name in names:
            for seed in profile["seeds"]:
                path = root/name/f"seed_{seed}"/"state.json"
                print(name,seed,json.loads(path.read_text(encoding="utf-8")) if path.exists() else "pending")
        return
    if args.phase in ("train","evaluate","robustness","run"):
        lock = root / ".run.lock"
        try:
            handle = lock.open("x", encoding="utf-8")
        except FileExistsError as exc:
            raise RuntimeError("同一目录已有运行锁；确认原进程已结束后再处理锁文件") from exc
        with handle:
            json.dump({"pid": os.getpid(), "phase": args.phase}, handle)
    else:
        lock = None
    try:
        execute(args, root, names, spec, profile, cfg, manifest)
    finally:
        if lock is not None:
            lock.unlink(missing_ok=True)

def execute(args, root, names, spec, profile, cfg, manifest):
    if args.phase in ("train","run"):
        for name in names:
            for seed in profile["seeds"]:
                print(f"[训练] {name} seed={seed}",flush=True)
                train_one(root,name,spec["variants"][name],seed,profile,cfg,manifest,args)
    if args.phase in ("evaluate","run","robustness"):
        evaluate(root,names,spec,profile,cfg,manifest,args,robust=args.phase=="robustness")
    if args.phase in ("summarize","run","robustness"):
        summarize(root,names,profile)
        summarize(root,names,profile,robust=True)

if __name__=="__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
