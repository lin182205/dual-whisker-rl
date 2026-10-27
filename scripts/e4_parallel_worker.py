"""复制到封存包后执行：仅调用原E4函数，每个进程只处理一个任务。"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(1,str(ROOT/'scripts'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def atomic(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf8')
    os.replace(temp,path)


def result(job,value,artifacts=()):
    data=dict(job_signature=job['signature'],kind=job['kind'],result=value,
              artifacts=[dict(path=str(p),sha256=sha(p)) for p in artifacts])
    data['result_sha256']=digest(data);atomic(job['output'],data)


def arguments(P,job):
    m=json.loads((Path(job['root'])/'manifest.json').read_text(encoding='utf8'))
    if sha(Path(job['root'])/'manifest.json')!=job['manifest_sha256']:raise ValueError('E4封存发生变化')
    for rel,expected in m['code_hashes'].items():
        if sha(ROOT/rel)!=expected:raise ValueError(f'封存科学实现不符：{rel}')
    argv=['train','--profile',m['profile'],'--config',job['config'],'--output-dir',str(Path(job['root']).parent),
          '--timesteps',str(job['target']),'--n-envs',str(m['n_envs']),
          '--vec-env-backend',m['vec_env_backend'],'--checkpoint-rollouts',str(m['checkpoint_rollouts']),
          '--max-steps',str(m['config']['max_steps']),
          '--seeds',','.join(map(str,m['seeds'])),'--resume']
    args=P.build_parser().parse_args(argv);P.resolve_args(args)
    if digest(P.base_config(args))!=digest(m['config']):raise ValueError('E4参数与封存不一致')
    P.verify_scenario_manifest(Path(job['root']))
    return args,m


def inspect(P,args,job,m):
    root=Path(job['root']);signature=P.calibration_signature(args,root)
    legacy=P.legacy_calibration_signatures(args,root);seeds=[int(s) for s in args.seeds[:2]]
    b1=P.load_completed_calibration(root/'calibration_b1.json','b1_reactive',signature,legacy,seeds)
    b2=P.load_completed_calibration(root/'calibration_b2.json','b2_fixed',signature,legacy,seeds)
    signatures={f'{method}:{seed}':P.experiment_signature(args,method,seed,
        fixed_sector=P.selected_fixed_sector(root) if method=='b2_fixed' else 5)
        for method,seeds0 in m['method_seeds'].items() for seed in seeds0 if method not in P.RULE_METHODS}
    return dict(b1=b1,b2=b2,b1_grid=P.b1_calibration_grid(),signature=signature,formal_signatures=signatures,
        signature_version=P.CALIBRATION_SIGNATURE_VERSION,version=P.VERSION,seeds=seeds,
        methods=m['methods'],calibration_steps=102400)


def b1(P,args,job):
    cfg=P.base_config(args);cfg.update(job['candidate_config'])
    env=P.make_env('b1_reactive',cfg)
    c=job['candidate_config']
    controller=P.ReactiveController(hit_threshold=float(c['b1_hit_threshold']),trend_threshold=float(c['b1_trend_threshold']),
        contrast_threshold=float(c['b1_contrast_threshold']),search_period=int(c['b1_search_period']),scan_period=int(c['b1_scan_period']))
    rows=[]
    try:
        for scene in P.read_scenarios(Path(job['root']),'validation',None):
            controller.reset();rows.append(P.evaluate_one(controller,env,scene,max_steps=args.max_steps))
    finally:env.close()
    summary=P.aggregate(rows)
    return dict(candidate_index=job['candidate_index'],config=c,success_rate=summary.get('success_rate',0.),
                mean_final_distance=summary.get('mean_final_distance'))


def training_artifacts(task):
    return [task/'model.zip',task/'metadata.json',task/'state.json']


def train(P,args,job):
    root=Path(job['root']);method=job['method'];seed=int(job['seed'])
    sector=int(job.get('sector',P.selected_fixed_sector(root) if method=='b2_fixed' else 5))
    task=Path(job['task_dir'])
    P.train_one(args,root,method,seed,fixed_sector=sector,task_dir_override=task)
    metadata=json.loads((task/'metadata.json').read_text(encoding='utf8'))
    if metadata['requested_timesteps']!=job['target'] or metadata['actual_timesteps']<job['target']:
        raise ValueError('模型没有完成目标总预算')
    return metadata


def b2(P,args,job):
    train(P,args,job)
    task=Path(job['task_dir']);args.eval_limit=None
    rows=P.evaluate_method(args,Path(job['root']),'b2_fixed',int(job['seed']),'validation',
        fixed_sector=int(job['sector']),task_dir_override=task)
    summary=P.aggregate(rows);checkpoints=[]
    for p in (task/'checkpoints').glob('*.zip'):
        try:checkpoints.append(int(p.stem.split('_')[-2]))
        except (IndexError,ValueError):pass
    artifacts=training_artifacts(task)+list((task/'episodes/validation').glob('*.json'))
    return dict(seed=int(job['seed']),success_rate=summary.get('success_rate',0.),
                mean_final_distance=summary.get('mean_final_distance'),
                earliest_checkpoint=min(checkpoints) if checkpoints else None),artifacts


class ProbeFinished(Exception):pass


def probe(P,args,job):
    from stable_baselines3.common.callbacks import BaseCallback
    shared=Path(job['probe_directory']);actor=job['actor'];values={}
    class Timer(BaseCallback):
        def _on_step(self):return True
        def _on_rollout_start(self):
            if self.model.num_timesteps==0:return
            if not values:
                atomic(shared/f'ready_{actor}.json',dict(pid=os.getpid(),warmup_steps=int(self.model.num_timesteps)))
                timeout=time.monotonic()+180
                while not (shared/'start.json').exists():
                    if time.monotonic()>timeout:raise TimeoutError('预热同步超时')
                    time.sleep(.05)
                start=json.loads((shared/'start.json').read_text(encoding='utf8'))
                values.update(start=start['start'],first=int(self.model.num_timesteps),seconds=start['seconds'])
            elif time.monotonic()>=values['start']+values['seconds']:
                values.update(steps=int(self.model.num_timesteps)-values['first'],end=time.monotonic(),device=str(self.model.device))
                raise ProbeFinished()
    original=P.make_rollout_eta_callback;P.make_rollout_eta_callback=lambda *a,**k:Timer()
    try:
        train(P,args,job)
        raise RuntimeError('诊断未按计时结束，拒绝将短训练作为正式模型')
    except ProbeFinished:
        result(job,dict(scope='performance_probe_only',**values))
    finally:P.make_rollout_eta_callback=original


def main():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    import run_e4_experiments as P
    P.configure_output_encoding()
    job=json.loads(Path(sys.argv[1]).read_text(encoding='utf8'))
    if sha(__file__)!=job['worker_sha256']:raise ValueError('训练包装器内容改变')
    args,m=arguments(P,job)
    if job['kind']=='inspect':result(job,inspect(P,args,job,m))
    elif job['kind']=='b1':result(job,b1(P,args,job))
    elif job['kind']=='b2':
        score,artifacts=b2(P,args,job);result(job,score,artifacts)
    elif job['kind']=='train':
        metadata=train(P,args,job);result(job,metadata,training_artifacts(Path(job['task_dir'])))
    elif job['kind']=='probe':probe(P,args,job)
    else:raise ValueError('未知E4任务')


if __name__=='__main__':main()
