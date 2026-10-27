"""E4校准与训练的上层并行调度；科学实现仍使用原封存入口。"""
from __future__ import annotations
import copy
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import numpy as np
import psutil
from dual_whisker_rl.arrival_protocol import atomic_json,digest,read,sha
from scripts.remaining_formal_support import campaign_lock
from scripts.e4_task_queue import Queue,cached,completed_model,resources,room_for_model,pair_key,process_from_receipt

ROOT=Path(__file__).resolve().parents[1]


def make_job(base,identifier,kind,**values):
    job={**base,**values,'id':identifier,'kind':kind}
    fingerprint={k:v for k,v in job.items() if k not in ('config','cache_root')}
    fingerprint['config_sha256']=sha(job['config'])
    job['signature']=digest(fingerprint)
    job['output']=str(Path(base['cache_root'])/'results'/f'{identifier}_{job["signature"][:16]}.json')
    return job


def read_result(job):
    if not cached(job):raise ValueError(f'任务输出不完整：{job["id"]}')
    return read(job['output'])['result']


def install_worker(bundle):
    source=ROOT/'scripts/e4_parallel_worker.py';target=Path(bundle)/'scripts/e4_parallel_worker.py'
    if target.exists() and sha(target)!=sha(source):raise ValueError('已有训练包装器与新版本不同，请使用新调度封存')
    if not target.exists():target.write_bytes(source.read_bytes())
    return target


def aggregate_b1(info,rows):
    scores=sorted(rows,key=lambda x:x['candidate_index'])
    if [r['candidate_index'] for r in scores]!=list(range(len(info['b1_grid']))):raise ValueError('B1候选覆盖不完整或重复')
    chosen=min(scores,key=lambda x:(-float(x['success_rate']),float(x['mean_final_distance']) if x['mean_final_distance'] is not None else float('inf'),x['candidate_index']))
    return dict(version=info['version'],method='b1_reactive',status='completed',signature_version=info['signature_version'],
        signature=info['signature'],candidate_count=len(scores),selected_config=chosen['config'],
        selection_rule='validation success_rate, then mean_final_distance, then candidate order',scores=scores)


def aggregate_b2(info,rows):
    scores={}
    for sector in range(10):
        values=[rows[sector,seed] for seed in info['seeds']]
        distances=[v['mean_final_distance'] for v in values if v['mean_final_distance'] is not None]
        checkpoints=[v['earliest_checkpoint'] for v in values if v['earliest_checkpoint'] is not None]
        scores[str(sector)]=dict(success_rate=float(np.mean([v['success_rate'] for v in values])),
            mean_final_distance=float(np.mean(distances)) if distances else None,
            earliest_checkpoint=min(checkpoints) if checkpoints else None,runs=values)
    chosen=min(range(10),key=lambda s:(-scores[str(s)]['success_rate'],scores[str(s)]['mean_final_distance'] if scores[str(s)]['mean_final_distance'] is not None else float('inf'),
        scores[str(s)]['earliest_checkpoint'] if scores[str(s)]['earliest_checkpoint'] is not None else float('inf'),s))
    return dict(version=info['version'],method='b2_fixed',status='completed',signature_version=info['signature_version'],
        signature=info['signature'],candidate_sectors=list(range(10)),candidate_angles_deg=[float((s+.5)*18.) for s in range(10)],
        seeds=info['seeds'],timesteps_per_candidate=102400,selected_sector=chosen,
        selection_rule='validation success_rate, then mean_final_distance, then earliest checkpoint',scores=scores)


def select_pairs(measurements):
    by={r['case']:r for r in measurements}
    for name in ('G','M'):
        if not by[name].get('valid') or by[name]['steps_per_second']<=0:raise RuntimeError(f'单模型吞吐诊断失败：{name}')
    g,m=by['G']['steps_per_second'],by['M']['steps_per_second']
    references={'GG':g,'MM':m,'GM':2/(1/g+1/m)}
    allowed={};ratios={}
    for pair,baseline in references.items():
        entry=by[pair];ratio=entry.get('steps_per_second',0.)/baseline if entry.get('valid') else 0.
        allowed[pair]=ratio>=1.1;ratios[pair]=ratio
    return dict(allowed=allowed,speedup=ratios,minimum_speedup=1.1)


def stop_probe(item):
    """只清理本次诊断拥有的进程树，不终止正式模型。"""
    p=process_from_receipt(item['receipt'])
    if p is None:return
    owned=[(c,c.create_time()) for c in p.children(recursive=True)]+[(p,p.create_time())]
    for proc,created in owned:
        try:
            if proc.create_time()==created:proc.terminate()
        except psutil.NoSuchProcess:pass
    _,alive=psutil.wait_procs([p for p,_ in owned],timeout=5)
    for proc in alive:
        try:proc.kill()
        except psutil.NoSuchProcess:pass


def probe_case(base,queue,case,seconds):
    folder=Path(base['cache_root'])/'probes'/case/uuid.uuid4().hex
    folder.mkdir(parents=True);before=resources();peak=copy.deepcopy(before)
    actors=[];started=None;valid=False;reason=None
    try:
        for i,family in enumerate(case):
            if not room_for_model(resources()):raise RuntimeError('诊断初始化资源余量不足')
            method='main' if family=='G' else 'b5_history_mlp'
            job=make_job(base,f'probe_{case}_{folder.name}_{i}','probe',method=method,seed=20261028+i,
                family=family,task_dir=str(folder/f'actor_{i}'),probe_directory=str(folder),actor=i)
            actors.append(queue.start(job))
        deadline=time.monotonic()+180
        while not all((folder/f'ready_{i}.json').exists() for i in range(len(actors))):
            if time.monotonic()>deadline:raise TimeoutError('诊断初始化/预热超时')
            if any(process_from_receipt(a['receipt']) is None for a in actors):raise RuntimeError('诊断进程预热失败')
            snapshot=resources();peak['memory_current']=max(peak['memory_current'],snapshot['memory_current'])
            if snapshot['gpu'] and peak['gpu']:peak['gpu'][0]['free']=min(peak['gpu'][0]['free'],snapshot['gpu'][0]['free'])
            if snapshot['memory_current']>.8*snapshot['memory_limit'] or snapshot['gpu'] and snapshot['gpu'][0]['free']<2**30:
                raise RuntimeError('双模型资源峰值超过保留上限')
            time.sleep(.2)
        started=time.monotonic();atomic_json(folder/'start.json',dict(start=started,seconds=seconds))
        while any(process_from_receipt(a['receipt']) is not None for a in actors):
            if time.monotonic()>started+seconds+300:raise TimeoutError('诊断rollout收尾超时')
            snapshot=resources();peak['memory_current']=max(peak['memory_current'],snapshot['memory_current'])
            if snapshot['gpu'] and peak['gpu']:peak['gpu'][0]['free']=min(peak['gpu'][0]['free'],snapshot['gpu'][0]['free'])
            if snapshot['memory_current']>.8*snapshot['memory_limit'] or snapshot['gpu'] and snapshot['gpu'][0]['free']<2**30:
                raise RuntimeError('诊断采样资源峰值超过保留上限')
            time.sleep(.3)
        values=[read_result(a['job']) for a in actors]
        steps=sum(v['steps'] for v in values);elapsed=max(v['end'] for v in values)-started
        valid=True
        return dict(case=case,valid=True,steps=steps,elapsed_s=elapsed,steps_per_second=steps/elapsed,
                    actors=values,before=before,peak=peak,scope='performance_probe_only')
    except Exception as ex:
        reason=repr(ex)
        return dict(case=case,valid=False,error=reason,before=before,peak=peak,steps_per_second=0.)
    finally:
        for a in actors:
            if process_from_receipt(a['receipt']) is not None:stop_probe(a)
            if a['popen'] is not None:a['popen'].wait(timeout=10)
            a['handle'].close();a['receipt'].update(status='completed' if valid else 'failed',error=reason)
            atomic_json(queue.receipt_path(a['job']),a['receipt'])


def choose_concurrency(args,base,queue):
    hardware=resources()
    key=dict(manifest=base['manifest_sha256'],worker=base['worker_sha256'],manager=sha(__file__),
        cpu_quota=hardware['cpu_quota'],memory_limit=hardware['memory_limit'],
        gpu=[dict(name=g['name'],total=g['total']) for g in hardware['gpu']],python=sys.version,
        libraries={n:importlib.metadata.version(n) for n in ('torch','stable-baselines3','numpy')},
        mode=args.e4_concurrency,seconds=args.e4_probe_seconds)
    path=Path(base['cache_root'])/'concurrency_probe.json'
    if path.exists() and read(path).get('signature')==digest(key):return read(path)['choice']
    if args.e4_concurrency!='auto':
        choice=dict(allowed={p:args.e4_concurrency=='2' for p in ('GG','GM','MM')},reserve={},reason='explicit')
        atomic_json(path,dict(signature=digest(key),inputs=key,choice=choice));return choice
    wait_survivors(queue)
    measurements=[]
    for case in ('G','M','GG','MM','GM'):
        print(f'[E4吞吐短测] {case} 计时预算={args.e4_probe_seconds/5:.1f}秒，初始化与预热另计',flush=True)
        result=probe_case(base,queue,case,args.e4_probe_seconds/5);measurements.append(result)
        print(f'[E4吞吐短测] {case}: {result.get("steps_per_second",0):.1f}步/秒 valid={result["valid"]}',flush=True)
    choice=select_pairs(measurements)
    single=[x for x in measurements if x['case'] in ('G','M')]
    choice['reserve']=dict(uses_gpu=any(str(a.get('device','')).startswith('cuda') for x in single for a in x.get('actors',[])),
        memory=max(512*2**20,int(max(x['peak']['memory_current']-x['before']['memory_current'] for x in single)*1.25)),
        gpu=max(256*2**20,int(max((x['before']['gpu'][0]['free']-x['peak']['gpu'][0]['free'] if x['peak']['gpu'] else 0) for x in single)*1.25)))
    atomic_json(path,dict(signature=digest(key),inputs=key,measurements=measurements,choice=choice))
    return choice


def wait_survivors(queue):
    """硬件改变需要重测前，先等待旧存活任务结束，不能在训练负载下重测。"""
    for path in queue.processes.glob('*.json'):
        r=read(path)
        if r.get('status') not in ('running','monitor_interrupted'):continue
        while process_from_receipt(r) is not None:
            print(f'[E4恢复等待] 已有任务PID={r["pid"]}继续运行，不重复启动',flush=True);time.sleep(queue.interval)


def run_e4_parallel(args,bundle,manifest):
    from scripts.stochastic_campaign import validate_e4_training_manifest
    root=args.e4_training_dir;validate_e4_training_manifest(args,manifest,required=True)
    worker=install_worker(bundle);cache=root/'parallel';cache.mkdir(exist_ok=True)
    base=dict(root=str(root),config=str(Path(bundle)/'configs/e4_formal.yaml'),target=args.e4_timesteps,
              manifest_sha256=sha(root/'manifest.json'),worker_sha256=sha(worker),cache_root=str(cache))
    queue=Queue(cache,worker,bundle,interval=min(5.,args.monitor_interval),stall_seconds=args.stall_seconds)
    with campaign_lock(cache):
        inspect=make_job(base,'inspect','inspect',family='G')
        request=cache/'inspect_request.json';atomic_json(request,inspect)
        with (cache/'inspect.log').open('ab') as log:
            subprocess.run([sys.executable,str(worker),str(request)],cwd=bundle,stdout=log,stderr=subprocess.STDOUT,check=True)
        info=read_result(inspect)
        def formal_jobs(info):
            jobs=[]
            for meta in manifest['methods']:
                method=meta['method']
                if not meta['trainable']:continue
                for seed in manifest['method_seeds'][method]:
                    jobs.append(make_job(base,f'train_{method}_seed_{seed}','train',method=method,seed=seed,
                        expected_signature=info['formal_signatures'][f'{method}:{seed}'],
                        family='G' if meta['temporal_encoder']=='gru' else 'M',task_dir=str(root/method/f'seed_{seed}')))
            return jobs
        formal=formal_jobs(info)
        if info['b1'] is not None and info['b2'] is not None and all(completed_model(j) for j in formal):
            return dict(status='completed',formal_models=len(formal),all_reused=True)
        # 原任务未结束时不可用其占用下的吞吐重新选配置；等待不会丢弃其进度。
        choice=choose_concurrency(args,base,queue)
        if info['b1'] is None:
            jobs=[make_job(base,f'b1_{i:03d}','b1',candidate_index=i,candidate_config=c,family='G') for i,c in enumerate(info['b1_grid'])]
            caps=resources();workers=min(len(jobs),caps['cpu_workers'],int(args.workers) if str(args.workers)!='auto' else caps['cpu_workers'])
            workers=min(workers,max(0,int((.8*caps['memory_limit']-caps['memory_current'])//(512*2**20))))
            if workers<1:raise RuntimeError('内存不足以启动规则参数搜索')
            queue.run(jobs,limit=workers,resource_guard=False)
            info['b1']=aggregate_b1(info,[read_result(j) for j in jobs]);atomic_json(root/'calibration_b1.json',info['b1'])
        if info['b2'] is None:
            jobs=[make_job({**base,'target':102400},f'b2_sector_{sector}_seed_{seed}','b2',method='b2_fixed',seed=seed,
                sector=sector,family='G',task_dir=str(root/'calibration'/f'sector_{sector}'/f'seed_{seed}'))
                for sector in range(10) for seed in info['seeds']]
            queue.run(jobs,limit=2 if choice['allowed']['GG'] else 1,allowed=choice['allowed'],reserve=choice['reserve'])
            info['b2']=aggregate_b2(info,{(j['sector'],j['seed']):read_result(j) for j in jobs})
            atomic_json(root/'calibration_b2.json',info['b2'])
        # 固定扇区在校准完成后由原入口读取，不更改正式训练签名。
        with (cache/'inspect.log').open('ab') as log:
            subprocess.run([sys.executable,str(worker),str(request)],cwd=bundle,stdout=log,stderr=subprocess.STDOUT,check=True)
        info=read_result(inspect);formal=formal_jobs(info)
        formal_result=queue.run(formal,limit=2,allowed=choice['allowed'],reserve=choice['reserve'])
        if not all(completed_model(j) for j in formal):raise RuntimeError('E4仍有未完成模型')
        receipt=dict(status='completed',formal=formal_result,choice=choice,calibration_b1_sha256=sha(root/'calibration_b1.json'),
                     calibration_b2_sha256=sha(root/'calibration_b2.json'),worker_sha256=sha(worker))
        atomic_json(cache/'completed.json',receipt)
        return receipt
