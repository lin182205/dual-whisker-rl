"""历史2M预算模型与75M的统一随机动作评估；只补缺局，不进行训练。"""
from __future__ import annotations
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from dual_whisker_rl.arrival_protocol import atomic_json,read,sha
from dual_whisker_rl.stochastic_protocol import comparison_spec,native_model,settings,make_jobs,seal
from scripts.e4_task_queue import linux_birth_token,process_from_receipt


def build_spec(args):
    spec=comparison_spec(args.reuse_from,args.model_path,args.sealed_source,settings(args.eval_seed,args.eval_repeats))
    old=read(args.reuse_from/'manifest.json')
    if old['settings']['timesteps']!=2_000_000:raise ValueError('此入口要求历史训练预算恰为2M')
    extra=[]
    for seed in old['settings']['seeds']:
        model=native_model(args.reuse_from,old,'main',seed,spec['sampling'])
        model['variant']='historical_main_2m'
        extra.append(model)
    spec['models'][1:1]=extra
    spec['labels'].update(historical_main_2m='历史主动GRU（2M预算，有DR）',
        fixed='固定触须（2M预算）',periodic='周期扫描（2M预算）',single_frame='单帧MLP（2M预算）',
        history_mlp='历史MLP（2M预算）',no_dr='历史主动GRU（2M预算，无DR）')
    spec['english_labels']=dict(historical_main_2m='Old active 2M, DR',fixed='Fixed 2M',periodic='Periodic 2M',
        single_frame='Single frame 2M',history_mlp='History MLP 2M',no_dr='Old active 2M, no DR')
    spec['selection_note']='原训练验证规则冻结的selected_model；2M为总训练预算，检查点可早于2M，不按新测试成绩重选。'
    return spec


def owned_receipt(p,command=None):
    value=dict(pid=p.pid,created=p.create_time(),command=command or p.cmdline())
    birth=linux_birth_token(p.pid)
    if birth is not None:value['linux_birth_token']=birth
    return value


@contextmanager
def training_window(args):
    """暂停父派发器，等待其既有子任务落盘；退出评估窗口时恢复派发。"""
    import psutil
    pointer=ROOT/'results/active_formal_campaign.json'
    if not pointer.exists():
        yield
        return
    active=read(pointer);deployment=read(Path(active['output_dir'])/'deployment.json')
    expected=dict(pid=active['pid'],created=active['created'],command=deployment['command'])
    try:
        parent=process_from_receipt(expected)
    except psutil.Error:parent=None
    if parent is None:
        if psutil.pid_exists(active['pid']):raise RuntimeError('活动调度身份不确定，拒绝叠加正式评估')
        yield
        return
    if parent.status()==psutil.STATUS_STOPPED:raise RuntimeError('父调度已经暂停，需先核对现有评估窗口')
    receipt=owned_receipt(parent,deployment['command'])
    lease=dict(parent=receipt,previous_output=active['output_dir'],status='pausing',started_at=time.time())
    path=args.output_dir/'training_coordination.json';atomic_json(path,lease)
    suspended=[]
    try:
        parent.suspend()
        children=[]
        for child in parent.children():
            if child.status()==psutil.STATUS_ZOMBIE:continue
            command=child.cmdline()
            if not any(Path(part).name=='e4_parallel_worker.py' for part in command):
                raise RuntimeError('父调度存在其他在途阶段，恢复训练并停止本次评估')
            children.append(owned_receipt(child,command))
        if getattr(args,'pause_inflight_training',False):
            # 先停训练主进程，再停其环境，避免挂起期间产生新子进程；恢复按相反顺序。
            for receipt0 in children:
                child=process_from_receipt(receipt0)
                if child is None:continue
                suspended.append(receipt0);lease['suspended_workers']=suspended;atomic_json(path,lease)
                child.suspend()
                for descendant in child.children(recursive=True):
                    if descendant.status()==psutil.STATUS_ZOMBIE:continue
                    suspended.append(owned_receipt(descendant));atomic_json(path,lease);descendant.suspend()
            lease.update(status='training_suspended_in_memory',children=children);atomic_json(path,lease)
            print(f'[评估窗口] 已挂起{len(suspended)}个训练及环境进程，内存中的进度保留',flush=True)
        else:
            lease.update(status='waiting_for_inflight_training',children=children);atomic_json(path,lease)
            while any(process_from_receipt(r) is not None for r in children):
                print('[等待评估窗口] 保留在途E4任务，等待训练及验证落盘',flush=True)
                time.sleep(min(30.,args.monitor_interval))
        lease.update(status='evaluating',evaluation_started_at=time.time());atomic_json(path,lease)
        yield
    finally:
        errors=[]
        for worker in reversed(suspended):
            try:
                current=process_from_receipt(worker)
                if current is not None:current.resume()
            except psutil.Error as e:errors.append(repr(e))
        current=process_from_receipt(receipt)
        if current is not None:
            current.resume();lease.update(status='training_resumed',resumed_at=time.time())
        else:lease.update(status='parent_no_longer_alive',finished_at=time.time())
        lease['worker_resume_errors']=errors
        atomic_json(path,lease)
        if errors:raise RuntimeError(f'部分训练进程恢复失败：{errors}')


def parser():
    from scripts.run_fast_comparison import parser as original
    p=original();p.description=__doc__
    p.add_argument('--monitor-interval',type=float,default=10.)
    p.add_argument('--pause-inflight-training',action='store_true',help='挂起在途训练进程树并保留内存，评估后原地恢复')
    return p


def main():
    from scripts.stochastic_campaign import validate_args,validate_output
    from scripts.remaining_formal_support import campaign_lock
    from scripts.experiment_progress import configure_output_encoding
    configure_output_encoding();args=parser().parse_args();validate_args(args)
    if args.evaluation_protocol!='stochastic-arrival-v2':raise ValueError('此入口只接受修正后的随机动作协议')
    if args.monitor_interval<=0:raise ValueError('监控间隔必须为正')
    args.output_dir=args.output_dir or ROOT/'results/reevaluation_stochastic_v2/historical_2m_vs_75m'
    args.episode_store=args.episode_store or ROOT/'results/reevaluation_stochastic_v2/episodes'
    for key in ('reuse_from','model_path','sealed_source','output_dir','episode_store'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,(ROOT/value).resolve() if not value.is_absolute() else value.resolve())
    validate_output(args.output_dir,[args.reuse_from,args.model_path.parent,args.episode_store])
    state_path=args.output_dir/'state.json'
    if args.phase=='status':
        print(json.dumps(read(state_path) if state_path.exists() else dict(status='not_started'),ensure_ascii=False,indent=2));return
    spec=build_spec(args);jobs=make_jobs(spec,args.episode_store)
    plan=dict(protocol=spec['protocol'],training_tasks=0,expected_episodes=len(jobs),
        primary_models=1,historical_models=len(spec['models'])-1,test_scenarios=len(spec['scenarios']['test']),
        result_files_present_unverified=sum(Path(j['output']).is_file() for j in jobs),
        missing_result_files=sum(not Path(j['output']).is_file() for j in jobs),selection_note=spec['selection_note'])
    if args.phase=='plan':print(json.dumps(plan,ensure_ascii=False,indent=2));return
    args.output_dir.mkdir(parents=True,exist_ok=True)
    def stop(signum,frame):raise KeyboardInterrupt('收到停止请求，恢复训练派发')
    old_signal=signal.signal(signal.SIGTERM,stop)
    try:
        with campaign_lock(args.output_dir):
            seal(args.output_dir/'manifest.json',spec);atomic_json(args.output_dir/'plan.json',plan)
            atomic_json(args.output_dir/'implementation.json',dict(entry_sha256=sha(__file__),training=False))
            atomic_json(args.output_dir/'figure_contract.json',dict(backend='python',
                question='共同648场景上，冻结75M与历史2M预算模型的成功率及气味信息获取有何差异',
                archetype='quantitative grid',roles=['success: outcome','odor_hit_ratio: information acquisition'],
                model_unit='one 75M model; five actual training seeds per historical method',
                interval='scenario bootstrap conditional on frozen models; baseline seeds averaged within scene',
                exports=['PDF','SVG','PNG'],reuse='existing report template; identical statistics; method labels expanded',
                risks=['unequal training budgets and physics','selected checkpoints can be earlier than 2M']))
            atomic_json(state_path,dict(status='waiting_for_evaluation_window',plan=plan))
            try:
                with training_window(args):
                    from dual_whisker_rl.stochastic_runtime import run_jobs
                    from dual_whisker_rl.arrival_reports import make_report,write_table
                    atomic_json(state_path,dict(status='evaluating',plan=plan))
                    receipt=run_jobs(args,jobs,spec['scenarios']['validation'],args.output_dir) if args.phase=='run' else {}
                    atomic_json(state_path,dict(status='reporting',evaluation=receipt))
                    report=make_report(spec,jobs,args.output_dir,args.bootstrap)
                    latest=read(args.output_dir/'latest_report.json')
                    if latest['status']!='complete':raise RuntimeError('对比仍有缺项，不作为完整报告')
                    write_table(args.output_dir/'models.csv',[dict(variant=m['variant'],seed=m['training_seed'],
                        training_budget_steps=m['training_budget_steps'],evaluated_checkpoint_steps=m['evaluated_checkpoint_steps'],
                        model_sha256=m['sha256'],model_path=m['path']) for m in spec['models']])
                    with report.open('a',encoding='utf8') as f:f.write('\n'+spec['selection_note']+'\n')
                atomic_json(state_path,dict(status='completed',evaluation=receipt,report=str(report)))
            except BaseException as e:
                atomic_json(state_path,dict(status='failed',error=repr(e)));raise
    finally:signal.signal(signal.SIGTERM,old_signal)


if __name__=='__main__':main()
