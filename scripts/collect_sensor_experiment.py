"""按实验单元采集MQ-3原始数据；STEP协议只驱动触须，不控制底盘。"""
from __future__ import annotations
import argparse
from collections import deque
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml
from dual_whisker_rl.paths import project_path,portable_path
from dual_whisker_rl.experiment_protocol import write_json
from dual_whisker_rl.e4_experiments import sha256_file
from dual_whisker_rl.hardware.sensor_preprocess import SensorPreprocessConfig
from dual_whisker_rl.observation.whisker_observation_builder import WhiskerObservationBuilder

def parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-id",required=True)
    p.add_argument("--config",default="configs/hardware.yaml")
    p.add_argument("--output-dir",default="results/sensor_experiments")
    p.add_argument("--mode",choices=["fixed","periodic","random","ppo"],default="fixed")
    p.add_argument("--port",help="实采必须显式指定；dry-run不打开串口")
    p.add_argument("--duration-s",type=float,default=120)
    p.add_argument("--period-s",type=float,default=.8)
    p.add_argument("--fixed-sector",type=int,default=5)
    p.add_argument("--scan-hold",type=int,default=2)
    p.add_argument("--seed",type=int,default=1)
    p.add_argument("--date",default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--plateau-id")
    p.add_argument("--sequence-id")
    p.add_argument("--direction",choices=["ascending","descending","none"],default="none")
    p.add_argument("--condition",default="unspecified")
    p.add_argument("--target-reference",type=float,help="设定值；不替代实测reference_value")
    p.add_argument("--reference-unit",choices=["ppm","release_level"],default="release_level")
    p.add_argument("--training-period-s", type=float, help="PPO训练时的实际观测间隔，必须与硬件采样一致")
    p.add_argument("--model",help="只接受12维单帧观测、双扇区动作的固定基座模型")
    p.add_argument("--preprocess-config",help="PPO必须提供已冻结的硬件预处理YAML")
    p.add_argument("--dry-run",action="store_true")
    args = p.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+",args.run_id):
        p.error("run-id只允许英文字母、数字、下划线和连字符")
    if not all(math.isfinite(v) and v>0 for v in (args.duration_s,args.period_s)) or args.scan_hold<1:
        p.error("duration、period和scan-hold必须为正")
    if args.mode=="ppo" and (not args.model or not args.preprocess_config):
        p.error("ppo模式需要model和冻结preprocess-config")
    if args.mode == "ppo" and (args.training_period_s is None or abs(args.training_period_s - args.period_s) > .05 * args.period_s):
        p.error("PPO必须声明与period-s一致的training-period-s；不同采样时钟需要重训")
    if not args.dry_run and not args.port:
        p.error("实采必须指定port")
    return args

def main():
    args = parse()
    cfg = yaml.safe_load(project_path(args.config).read_text(encoding="utf-8"))
    sc = int(cfg["whisker"]["sector_count"])
    if sc!=10 or not 0<=args.fixed_sector<sc:
        raise ValueError("当前STEP固件约定10扇区；fixed-sector应在0..9")
    folder = project_path(args.output_dir)/args.run_id
    metadata = dict(run_id=args.run_id,mode=args.mode,date=args.date,condition=args.condition,
        requested_period_s=args.period_s,duration_s=args.duration_s,seed=args.seed,
        target_reference=args.target_reference,reference_unit=args.reference_unit,
        reference_value_source="外部参考仪器；raw.csv留空，分析副本中按同步测量填写",
        angle_source="command_estimate_not_feedback",mobile_control=False,
        model=portable_path(args.model),preprocess_config=portable_path(args.preprocess_config))
    if args.dry_run:
        print(json.dumps(metadata,ensure_ascii=False,indent=2)); return
    if folder.exists():
        raise FileExistsError("run-id已存在，请为新试验使用新名称；不覆盖或续接采集")
    model,builder,history = None,None,None
    if args.mode=="ppo":
        from stable_baselines3 import PPO
        from dual_whisker_rl.agents import GRUHistoryExtractor
        _ = GRUHistoryExtractor
        model = PPO.load(project_path(args.model),device="cpu")
        if not hasattr(model.action_space,"nvec") or tuple(model.action_space.nvec)!=(10,10):
            raise ValueError("拒绝移动模型：当前串口不能执行底盘动作")
        size = int(np.prod(model.observation_space.shape))
        if size%12 or size<12:
            raise ValueError("模型不符合12维固定基座观测契约")
        settings = yaml.safe_load(project_path(args.preprocess_config).read_text(encoding="utf-8"))
        settings = dict(settings)
        baselines = (settings.pop("left_baseline", None), settings.pop("right_baseline", None))
        if any(v is None or not math.isfinite(float(v)) for v in baselines):
            raise ValueError("PPO预处理配置必须给出实测left_baseline和right_baseline")
        pre = SensorPreprocessConfig(**settings)
        pre.dt_s = args.period_s
        builder = WhiskerObservationBuilder(pre,sc)
        builder._proc.reset(float(baselines[0]), float(baselines[1]))
        history = deque(maxlen=size//12)
        metadata.update(model_sha256=sha256_file(project_path(args.model)),
                        history_length=size//12,preprocess=settings,baselines=list(baselines),training_period_s=args.training_period_s)
    folder.mkdir(parents=True)
    write_json(folder/"metadata.json",metadata)
    write_json(folder/"state.json",dict(status="running",samples=0))
    from dual_whisker_rl.hardware.serial_client import DualWhiskerSerialClient
    fields = ["run_id","date","condition","sequence_id","plateau_id","direction","reference_unit","target_reference",
              "reference_value","step","t_rel_s","timestamp_pc","command_sent_s","response_received_s","serial_rtt_ms",
              "actual_period_s","period_overrun","mode","left_sector","right_sector","left_adc","right_adc",
              "left_angle_estimate_rad","right_angle_estimate_rad","policy_warmup","inference_ms",
              "temperature_c","humidity_percent","supply_voltage"]
    rng,step = np.random.default_rng(args.seed),0
    t0,last_time = time.perf_counter(),None
    status,error = "completed",None
    sectors = (args.fixed_sector,args.fixed_sector)
    try:
        with (folder/"raw.csv").open("x",encoding="utf-8-sig",newline="") as f:
            writer = csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            with DualWhiskerSerialClient(args.port,int(cfg["serial"]["baudrate"]),float(cfg["serial"]["timeout_s"])) as client:
                while time.perf_counter()-t0<args.duration_s:
                    cycle_started = time.perf_counter()
                    inference_ms = 0.0
                    warmup = args.mode=="ppo" and len(history)<history.maxlen
                    if args.mode=="random":
                        sectors = tuple(int(x) for x in rng.integers(0,sc,size=2))
                    elif args.mode=="periodic":
                        index = (step//args.scan_hold)%(2*(sc-1))
                        sector = index if index<sc else 2*(sc-1)-index
                        sectors = (sector,sector)
                    elif args.mode=="ppo" and not warmup:
                        before = time.perf_counter()
                        action,_ = model.predict(np.concatenate(history),deterministic=True)
                        sectors = tuple(int(x) for x in np.asarray(action).reshape(-1))
                        inference_ms = (time.perf_counter()-before)*1000
                    sent = time.perf_counter()
                    sample = client.step(*sectors)
                    received = time.perf_counter()
                    if (sample.left_sector,sample.right_sector)!=sectors:
                        raise ValueError("返回扇区与本次命令不匹配；停止并保留记录")
                    delta = received-last_time if last_time is not None else None
                    angles = ((sample.left_sector+.5)/sc*math.pi,-(sample.right_sector+.5)/sc*math.pi)
                    timing_mismatch = bool(builder and delta is not None and abs(delta - args.training_period_s) > .15 * args.training_period_s)
                    if builder:
                        builder.config.dt_s = delta if delta is not None else args.period_s
                        history.append(builder.build(sample.left_adc,sample.right_adc,*angles,
                                                     sample.left_sector,sample.right_sector))
                    writer.writerow(dict(run_id=args.run_id,date=args.date,condition=args.condition,
                        sequence_id=args.sequence_id or args.run_id,plateau_id=args.plateau_id or args.run_id,
                        direction=args.direction,reference_unit=args.reference_unit,target_reference=args.target_reference,
                        reference_value="",step=step,t_rel_s=received-t0,timestamp_pc=sample.t_pc_s,
                        command_sent_s=sent-t0,response_received_s=received-t0,serial_rtt_ms=(received-sent)*1000,
                        actual_period_s=delta,period_overrun=received-cycle_started>args.period_s,mode=args.mode,
                        left_sector=sample.left_sector,right_sector=sample.right_sector,left_adc=sample.left_adc,
                        right_adc=sample.right_adc,left_angle_estimate_rad=angles[0],right_angle_estimate_rad=angles[1],
                        policy_warmup=warmup,inference_ms=inference_ms,temperature_c="",humidity_percent="",supply_voltage=""))
                    f.flush()
                    last_time = received
                    step += 1
                    if timing_mismatch:
                        raise RuntimeError("实测观测周期偏离训练周期超过15%，停止策略采集并保留原始数据")
                    if step%20==0: print(f"{args.run_id} samples={step} elapsed={received-t0:.1f}s",flush=True)
                    remaining = args.period_s-(time.perf_counter()-cycle_started)
                    if remaining>0:
                        time.sleep(min(remaining,max(0,args.duration_s-(time.perf_counter()-t0))))
    except BaseException as exc:
        status = "interrupted" if isinstance(exc,KeyboardInterrupt) else "failed"
        error = repr(exc)
        raise
    finally:
        write_json(folder/"state.json",dict(status=status,samples=step,error=error,elapsed_s=time.perf_counter()-t0))
    print(f"原始数据：{portable_path(folder/'raw.csv')}")

if __name__=="__main__":
    main()
