"""E4独立进程队列：全局任务锁由调用者持有，恢复时接管存活进程。"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil
from dual_whisker_rl.arrival_protocol import atomic_json,digest,read,sha,model_header


def cached(job):
    p=Path(job['output'])
    if not p.is_file():return False
    try:
        data=read(p);checksum=data.pop('result_sha256')
        if checksum!=digest(data):return False
        if data.get('job_signature')!=job['signature']:raise ValueError(f'任务来源变更：{p}')
        for item in data.get('artifacts',[]):
            if not Path(item['path']).is_file():return False
            if sha(item['path'])!=item['sha256']:raise ValueError(f'已封存训练或验证产物改变：{item["path"]}')
        return True
    except (OSError,KeyError,json.JSONDecodeError):return False


def completed_model(job):
    """接纳旧串行已完成模型，但不能只相信completed标签。"""
    task=Path(job['task_dir'])
    try:
        state,meta=read(task/'state.json'),read(task/'metadata.json');head=model_header(task/'model.zip')
        return (state['status']=='completed' and state['signature']==meta['signature']
            and state['signature']==job.get('expected_signature',state['signature'])
            and meta['seed']==job['seed'] and head['training_seed']==job['seed']
            and meta['requested_timesteps']==job['target'] and meta['actual_timesteps']>=job['target']
            and head['evaluated_checkpoint_steps']==meta['actual_timesteps']
            and meta['config']==read(Path(job['root'])/'manifest.json')['config'])
    except (OSError,KeyError,ValueError):return False


def process_from_receipt(receipt):
    """PID复用不能作为恢复或终止其他进程的依据。"""
    try:
        p=psutil.Process(receipt['pid'])
        if p.create_time()!=receipt['created'] or p.status()==psutil.STATUS_ZOMBIE:return None
        actual=p.cmdline()
        if actual!=receipt['command']:
            # Linux退出可发生在status与cmdline两次读取之间，此时argv变空。
            if not p.is_running() or p.status()==psutil.STATUS_ZOMBIE:return None
            if not actual:
                # 内核可先清空argv再进入僵尸态；保留占用并在下一轮观察，不释放槽位。
                return p
            raise RuntimeError(f'存活任务命令与记录不一致，拒绝重复启动：{actual!r}')
        return p
    except psutil.NoSuchProcess:return None


def resources():
    from dual_whisker_rl.arrival_runtime import resource_limits
    caps=resource_limits(probe_cuda=False)
    limit=psutil.virtual_memory().total;current=limit-psutil.virtual_memory().available
    for lp,up in [('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                  ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            text=Path(lp).read_text().strip()
            if text!='max':limit=min(limit,int(text));current=int(Path(up).read_text())
        except (OSError,ValueError):pass
    gpu=[]
    try:
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=5).strip().splitlines()
        for line in lines:
            name,total,free=line.rsplit(',',2)
            gpu.append(dict(name=name.strip(),total=int(total)*2**20,free=int(free)*2**20))
    except (OSError,subprocess.SubprocessError,ValueError):pass
    return dict(cpu_quota=caps['cpu_quota'],memory_limit=limit,memory_current=current,
                cpu_workers=caps['cpu_workers_cap'],gpu=gpu)


def room_for_model(snapshot,reserve=None):
    reserve=reserve or {}
    ram=max(512*2**20,int(reserve.get('memory',512*2**20)))
    if snapshot['memory_current']+ram>.8*snapshot['memory_limit']:return False
    if reserve.get('uses_gpu') and not snapshot['gpu']:return False
    if snapshot['gpu'] and snapshot['gpu'][0]['free']<max(256*2**20,int(reserve.get('gpu',256*2**20)))+2**30:return False
    return True


def pair_key(left,right):return ''.join(sorted((left,right)))


class Queue:
    def __init__(self,root,worker,cwd,*,interval=2.,stall_seconds=3600.):
        self.root=Path(root);self.worker=Path(worker);self.cwd=Path(cwd)
        self.interval,self.stall_seconds=interval,stall_seconds
        self.processes=self.root/'processes';self.logs=self.root/'logs'
        self.processes.mkdir(parents=True,exist_ok=True);self.logs.mkdir(parents=True,exist_ok=True)

    def receipt_path(self,job):return self.processes/f'{job["id"]}.json'

    def attach(self,job):
        p=self.receipt_path(job)
        if not p.exists():return None
        receipt=read(p)
        if receipt.get('status') not in ('running','monitor_interrupted'):return None
        process=process_from_receipt(receipt)
        if process is None:return None
        if receipt['job_signature']!=job['signature']:raise RuntimeError('存活任务所属版本不同，保留进度并停止调度')
        return dict(job=job,process=process,popen=None,receipt=receipt,handle=None)

    def start(self,job):
        request=self.root/'requests'/f'{job["id"]}.json';atomic_json(request,job)
        log=self.logs/f'{job["id"]}.log';handle=log.open('ab',buffering=0)
        command=[sys.executable,str(self.worker),str(request)]
        env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        child=subprocess.Popen(command,cwd=self.cwd,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        process=psutil.Process(child.pid)
        receipt=dict(pid=child.pid,created=process.create_time(),command=command,job_signature=job['signature'],
            task_id=job['id'],status='running',log=str(log),started_at=time.time(),log_bytes=log.stat().st_size,
            last_output_change=time.time())
        atomic_json(self.receipt_path(job),receipt)
        return dict(job=job,process=process,popen=child,receipt=receipt,handle=handle)

    def run(self,jobs,*,limit=2,allowed=None,reserve=None,resource_guard=True):
        if limit<1:raise ValueError('任务并行上限必须为正')
        if len({j['id'] for j in jobs})!=len(jobs):raise ValueError('任务ID重复')
        pending=[];active=[];failures=[];reused=0;attached=0
        for job in jobs:
            survivor=self.attach(job)
            if survivor is not None:active.append(survivor);attached+=1
            elif cached(job):reused+=1
            elif job['kind']=='train' and completed_model(job):reused+=1
            else:
                previous=read(self.receipt_path(job)) if self.receipt_path(job).exists() else {}
                if previous.get('status')=='failed':
                    failures.append(dict(id=job['id'],exit_code=previous.get('exit_code'),log=previous.get('log'),
                                         reason='previous_failure_requires_manual_review'))
                else:pending.append(job)
        if len(active)>limit:raise RuntimeError('存活任务超过当前上限，拒绝重复派发')
        baseline=resources() if resource_guard else None
        try:
            while pending or active:
                while pending and len(active)<limit and not failures:
                    index=next((i for i,j in enumerate(pending) if not active or allowed is None
                        or allowed.get(pair_key(active[0]['job']['family'],j['family']),False)),None)
                    if index is None:break
                    if resource_guard:
                        snapshot=resources();reservation=reserve or {}
                        snapshot['memory_current']=max(snapshot['memory_current'],baseline['memory_current']+len(active)*max(512*2**20,reservation.get('memory',512*2**20)))
                        if snapshot['gpu'] and baseline['gpu']:
                            snapshot['gpu'][0]['free']=min(snapshot['gpu'][0]['free'],baseline['gpu'][0]['free']-len(active)*max(256*2**20,reservation.get('gpu',256*2**20)))
                        if not room_for_model(snapshot,reserve):break
                    active.append(self.start(pending.pop(index)))
                for item in list(active):
                    job,p,receipt=item['job'],item['process'],item['receipt']
                    live=process_from_receipt(receipt)
                    if live is None:
                        code=item['popen'].wait(timeout=5) if item['popen'] is not None else None
                        valid=(cached(job) if job['kind']!='train' else completed_model(job))
                        if code not in (0,None):valid=False
                        receipt.update(status='completed' if valid else 'failed',exit_code=code,finished_at=time.time())
                        if not valid:failures.append(dict(id=job['id'],exit_code=code,log=receipt['log']))
                        atomic_json(self.receipt_path(job),receipt)
                        if item['handle'] is not None:item['handle'].close()
                        active.remove(item)
                    else:
                        size=Path(receipt['log']).stat().st_size
                        if size!=receipt['log_bytes']:receipt['last_output_change']=time.time()
                        receipt.update(log_bytes=size,checked_at=time.time(),output_stalled=time.time()-receipt['last_output_change']>=self.stall_seconds)
                        atomic_json(self.receipt_path(job),receipt)
                atomic_json(self.root/'queue_state.json',dict(status='attention_required' if failures else 'running',
                    active=[dict(id=x['job']['id'],pid=x['receipt']['pid']) for x in active],pending=[j['id'] for j in pending],
                    reused=reused,failures=failures))
                if failures and not active:break
                if not active and pending and resource_guard and not failures:
                    print('[E4资源等待] 等待内存/显存余量，不启动额外模型',flush=True)
                if active or pending and not failures:time.sleep(self.interval)
        except BaseException:
            for item in active:
                item['receipt'].update(status='monitor_interrupted',child_may_be_alive=True)
                atomic_json(self.receipt_path(item['job']),item['receipt'])
                if item['handle'] is not None:item['handle'].close()
            raise
        if failures:raise RuntimeError(f'E4任务失败，未重试；健康任务已监控结束：{failures}')
        atomic_json(self.root/'queue_state.json',dict(status='completed',reused=reused,completed=len(jobs)))
        return dict(expected=len(jobs),reused=reused,attached=attached,executed=len(jobs)-reused-attached)
