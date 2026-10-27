"""接管既有训练；以独立签名和共享逐局缓存重评全部正式实验。"""
from __future__ import annotations
import argparse
import copy
from collections import defaultdict
from pathlib import Path
import time
import os
import sys
from dual_whisker_rl.arrival_protocol import atomic_json, cached, digest, model_header, read, sha
from dual_whisker_rl.stochastic_protocol import (PROTOCOL, comparison_spec, make_jobs, native_model,
    nominal_config, scientific_hashes, seal, settings, source_manifest, view_rows)
from dual_whisker_rl.stochastic_runtime import run_jobs

ROOT = Path(__file__).resolve().parents[1]


def path(value):
    p = Path(value)
    return (p if p.is_absolute() else ROOT/p).resolve()


def validate_args(args):
    if str(args.workers) != 'auto' and (not str(args.workers).isdecimal() or int(args.workers)<1):
        raise ValueError('workers必须为auto或正整数')
    if not 0 <= args.calibration_seconds <= 60 or args.bootstrap < 1:
        raise ValueError('预热预算为0至60秒，bootstrap须为正数')
    if args.device not in ('cpu','auto','cuda') and not (args.device.startswith('cuda:') and args.device[5:].isdecimal()):
        raise ValueError('未知推理设备')


def validate_output(output, sources):
    for source in sources:
        source = Path(source).resolve()
        if output == source or source in output.parents or output in source.parents:
            raise ValueError('新评估目录必须与旧结果和模型目录分离')


def episode_store(args):
    return getattr(args,'episode_store',None) or args.output_dir/'episodes'


def e4_training_command(args,bundle,phase):
    """预算通过每次子进程参数显式传递，不依赖封存入口的旧默认值。"""
    if phase not in ('calibrate','train'):raise ValueError('E4训练入口仅允许校准或训练')
    return [sys.executable,Path(bundle)/'scripts/run_e4_experiments.py',phase,'--profile','formal',
        '--config',Path(bundle)/'configs/e4_formal.yaml','--output-dir',args.e4_training_dir.parent,
        '--timesteps',str(args.e4_timesteps),'--resume']


def validate_e4_training_manifest(args,reference,required=False):
    """新预算单独封存，不把旧100万步任务改标成800万步完成。"""
    p=args.e4_training_dir/'manifest.json'
    if not p.exists():
        if required:raise FileNotFoundError('E4校准未创建新预算封存')
        return None
    target=read(p)
    if target.get('timesteps')!=args.e4_timesteps:raise ValueError('E4输出预算不一致，请使用独立训练目录')
    for key in ('profile','methods','method_seeds','seeds','n_envs','vec_env_backend','checkpoint_rollouts',
                'config','scenario_counts','scenario_hashes','code_hashes'):
        if target.get(key)!=reference.get(key):raise ValueError(f'E4新预算改变了原封存条件：{key}')
    return target


def e4_training_completed(task,target):
    state_path,metadata_path=task/'state.json',task/'metadata.json'
    if not state_path.exists() or not metadata_path.exists() or not (task/'model.zip').exists():return False
    state,metadata=read(state_path),read(metadata_path)
    if (state.get('status')!='completed' or metadata.get('requested_timesteps')!=target
            or state.get('actual_timesteps',0)<target):return False
    head=model_header(task/'model.zip')
    return head['evaluated_checkpoint_steps']>=target and head['evaluated_checkpoint_steps']==metadata.get('actual_timesteps')


def fast(args):
    from dual_whisker_rl.arrival_reports import make_report
    from scripts.remaining_formal_support import campaign_lock
    validate_args(args)
    out = path(args.output_dir or 'results/reevaluation_stochastic_v2/main')
    source, model = path(args.reuse_from), path(args.model_path)
    validate_output(out, [source, model.parent])
    spec = comparison_spec(source, model, path(args.sealed_source) if args.sealed_source else None,
                           settings(args.eval_seed, args.eval_repeats))
    store = path(args.episode_store) if args.episode_store else out/'episodes'
    validate_output(store,[source,model.parent])
    jobs = make_jobs(spec, store)
    if args.phase in ('plan','status'):
        return dict(protocol=PROTOCOL, training_tasks=0, expected=len(jobs),
            unique=len({j['signature'] for j in jobs}), completed=sum(cached(j) for j in jobs),
            models=len(spec['models']), goal_radius=.1, success_hold_s=0., legacy_trajectory_reuse=False,
            output=str(out), sampling=spec['sampling'])
    out.mkdir(parents=True, exist_ok=True)
    with campaign_lock(out):
        seal(out/'manifest.json',spec)
        result = {} if args.phase == 'summarize' else run_jobs(args,jobs,spec['scenarios']['validation'],out)
        report = make_report(spec,jobs,out,args.bootstrap)
        result.update(status='completed' if all(cached(j) for j in jobs) else 'partial',report=str(report))
        atomic_json(out/'state.json',result)
        return result


def context(args):
    import yaml
    from scripts.remaining_formal_support import bundle_inputs, verified_manifest, completed_training
    validate_args(args)
    if args.monitor_interval<=0 or args.stall_seconds<=0:raise ValueError('监控时间须为正')
    allowed=('train_mlp','compare','e3','e4','e5','e6','long_faults')
    stages=args.stages.split(',') if isinstance(args.stages,str) else args.stages
    if len(stages)!=len(set(stages)) or set(stages)-set(allowed):raise ValueError('阶段未知或重复')
    args.stages=[s for s in allowed if s in stages]
    if args.e4_timesteps<1:raise ValueError('E4训练预算必须为正整数')
    for key in ('source_dir','e4_dir','primary_model','adopt_training_run','output_dir','sealed_source','e4_training_dir','episode_store'):
        value=getattr(args,key)
        if value is not None:setattr(args,key,path(value))
    suffix=f'{args.e4_timesteps//1_000_000}m' if args.e4_timesteps%1_000_000==0 else str(args.e4_timesteps)
    args.output_dir = args.output_dir or path(f'results/reevaluation_stochastic_v2_e4_{suffix}')
    args.e4_training_dir=args.e4_training_dir or path(f'results/e4_{suffix}/formal')
    if args.e4_training_dir.name!='formal':raise ValueError('E4训练目录末级必须为formal')
    validate_output(args.e4_training_dir,[args.source_dir,args.e4_dir,args.adopt_training_run,args.primary_model.parent])
    validate_output(args.output_dir,[args.source_dir,args.e4_dir,args.adopt_training_run,args.primary_model.parent])
    if args.episode_store:validate_output(args.episode_store,[args.source_dir,args.e4_dir,args.adopt_training_run,args.primary_model.parent])
    old = source_manifest(args.source_dir,args.sealed_source)
    if old['version'] != 2:
        raise ValueError('剩余原训练只支持已核验的第2版封存')
    e4 = read(args.e4_dir/'manifest.json')
    validate_e4_training_manifest(args,e4)
    files = bundle_inputs(old,[args.sealed_source],e4)
    spec = yaml.safe_load((ROOT/'configs/red_experiments.yaml').read_text(encoding='utf8'))
    groups=('history','features','rewards','geometry','sensor','randomization')
    native=list(dict.fromkeys(spec['groups']['core']+[n for g in groups for n in spec['groups'][g]]))
    completed=[(n,s) for n in native for s in old['settings']['seeds']
               if completed_training(args.source_dir/n/f'seed_{s}',old['signature'],s,old['settings']['timesteps'])]
    sampling=settings(args.eval_seed,args.eval_repeats)
    training=read(args.adopt_training_run/'mlp_training.json')
    if training['target_total_steps'] != args.mlp_timesteps:
        raise ValueError('接管MLP训练目标不一致，拒绝重启或增加训练')
    signed=dict(protocol=PROTOCOL,sampling=sampling,source_sha=sha(args.source_dir/'manifest.json'),
        e4_sha=sha(args.e4_dir/'manifest.json'),primary=model_header(args.primary_model),
        e4_training=dict(target_steps_per_seed=args.e4_timesteps,directory=str(args.e4_training_dir)),
        episode_store=str(episode_store(args)),
        adopted_mlp={k:training[k] for k in ('seed','model_path','target_total_steps','training_config')},
        original_training_sources={n:sha(p) for n,p in files.items()},science=scientific_hashes(),
        controller_sha={n:sha(ROOT/n) for n in ('scripts/stochastic_campaign.py','dual_whisker_rl/stochastic_runtime.py',
            'scripts/run_remaining_formal.py','scripts/remaining_formal_support.py','scripts/experiment_reports.py',
            'dual_whisker_rl/arrival_reports.py')},groups=spec['groups'])
    return dict(old=old,e4=e4,files=files,spec=spec,native=native,completed=completed,
                sampling=sampling,signed=signed,training=training)


def preview(args,ctx):
    old=ctx['old'];r=args.eval_repeats
    e5=list(dict.fromkeys(n for g in ('history','features','rewards','geometry','sensor','randomization') for n in ctx['spec']['groups'][g]))
    e4_pending=sum(not e4_training_completed(args.e4_training_dir/n/f'seed_{s}',args.e4_timesteps)
        for n,seeds in ctx['e4']['method_seeds'].items() if n not in ('b0_random','b1_reactive') for s in seeds)
    main_count=(1+5*len(old['settings']['seeds'])+1)*len(old['scenarios']['test'])*r
    e5_count=len(e5)*len(old['settings']['seeds'])*len(old['scenarios']['test'])*r
    e4_count=sum(map(len,ctx['e4']['method_seeds'].values()))*sum(ctx['e4']['scenario_counts'].values())*r
    core=ctx['spec']['groups']['core'];fault_n=min(200,len(old['scenarios']['test']))
    shared_e5=len(set(e5)&{'fixed','periodic','single_frame','history_mlp','no_dr'})*len(old['settings']['seeds'])*len(old['scenarios']['test'])*r
    projected_unique=main_count+e5_count-shared_e5+len(core)*len(old['settings']['seeds'])*4*fault_n*r+2*4*fault_n*r+e4_count
    return dict(protocol=PROTOCOL,sampling=ctx['sampling'],stages=args.stages,output=str(args.output_dir),
        training_policy='continue_sealed_training_with_explicit_e4_budget',
        existing_mlp_target=ctx['training']['target_total_steps'],existing_mlp_status=ctx['training']['status'],
        remaining_native_training=len(ctx['native'])*len(old['settings']['seeds'])-len(ctx['completed']),
        remaining_e4_training=e4_pending,e4_target_steps_each=args.e4_timesteps,
        e4_total_target_steps=sum(len(seeds) for n,seeds in ctx['e4']['method_seeds'].items()
                                 if n not in ('b0_random','b1_reactive'))*args.e4_timesteps,
        e4_training_dir=str(args.e4_training_dir),e4_calibration_budget_unchanged=True,
        comparison_episodes=main_count,e4_report_episodes=e4_count,
        e5_report_episodes=len(e5)*len(old['settings']['seeds'])*len(old['scenarios']['test'])*r,
        e6_report_episodes=len(ctx['spec']['groups']['core'])*len(old['settings']['seeds'])*5*min(200,len(old['scenarios']['test']))*r,
        long_fault_report_episodes=2*5*min(200,len(old['scenarios']['test']))*r,
        projected_unique_episodes=projected_unique,
        projected_unique_note='按不同训练模型权重不同且各阶段标称配置一致估算；各阶段task_plan列出实际去重键与新增局数',
        unique_task_rule='model+effective_environment+scene+fault+sampling+science; shared across reports',
        legacy_trajectory_reuse=False,success_hold_s=0.,goal_radius=.1)


def adopt_training(args,ctx):
    import psutil
    record_path=args.adopt_training_run/'mlp_training.json'
    handoff_path=args.output_dir/'handoff.json'
    handoff=read(handoff_path) if handoff_path.exists() else None
    if handoff and handoff.get('status') != 'adopted':
        while True:
            try:
                trainer=psutil.Process(handoff['trainer_pid'])
                if (trainer.create_time()!=handoff['trainer_created'] or trainer.status()==psutil.STATUS_ZOMBIE):break
                if trainer.cmdline()!=handoff['trainer_command']:raise RuntimeError('训练进程身份变化')
            except psutil.NoSuchProcess:break
            waiting=read(args.output_dir/'state.json') if (args.output_dir/'state.json').exists() else {}
            waiting.update(status='waiting_for_existing_training',trainer_pid=trainer.pid,
                           training_run=str(args.adopt_training_run),updated_at=time.time())
            atomic_json(args.output_dir/'state.json',waiting)
            print(f'[等待既有训练] PID={trainer.pid}；不启动重复训练，60秒后检查',flush=True)
            time.sleep(min(60,args.monitor_interval))
    record=read(record_path)
    if record['status']!='completed':raise RuntimeError('既有MLP训练未成功完成，保留checkpoint并停止后续阶段')
    header=frozen_mlp_header(args)
    if header['sha256']!=record['model_sha256']:raise ValueError('既有MLP最终摘要不一致')
    if handoff and handoff.get('status')!='adopted':
        try:
            parent=psutil.Process(handoff['parent_pid'])
            if parent.create_time()!=handoff['parent_created'] or parent.cmdline()!=handoff['parent_command']:
                raise RuntimeError('旧父调度器身份改变，拒绝退出该进程')
            if parent.status()!=psutil.STATUS_STOPPED:raise RuntimeError('旧调度器未暂停，拒绝并行接管')
            # 训练已结束；仅退出已暂停且身份验证通过的旧父调度器。
            parent.kill()
        except psutil.NoSuchProcess:pass
        atomic_json(handoff_path,dict(handoff,status='adopted',adopted_at=time.time(),final_model=header))
    synchronize_original(args.output_dir)


def frozen_mlp_header(args):
    """首轮检查实际接口；完整缓存恢复只验摘要和元数据，不再次加载PPO。"""
    from scripts.run_remaining_formal import validate_mlp
    source=args.adopt_training_run/'mlp_training.json';record=read(source)
    receipt=args.output_dir/'adopted_training.json'
    metadata=Path(record['log_dir'])/'run_metadata.json' if record.get('log_dir') else None
    metadata_sha=sha(metadata) if metadata and metadata.is_file() else None
    if receipt.exists():
        old=read(receipt)
        if old['source_sha256']!=sha(source) or old.get('metadata_sha256')!=metadata_sha:
            raise ValueError('已接管MLP来源或元数据改变')
        header=model_header(record['model_path'])
        if header!=old['header']:raise ValueError('已接管MLP模型改变')
        return header
    header=validate_mlp(record,args.mlp_timesteps,record['training_config'])
    atomic_json(receipt,dict(source=str(source),source_sha256=sha(source),metadata_sha256=metadata_sha,
                             record=record,header=header))
    return header


def synchronize_original(output):
    """当前训练完成后再同步旧工作目录；差异备份和Git stash均保留。"""
    import subprocess
    receipt=Path(output)/'deployment.json'
    if not receipt.exists():return
    deployment=read(receipt)
    if deployment.get('original_checkout_synchronized'):return
    root=Path(deployment['original_checkout']).resolve()
    expected=deployment['release_commit']
    def git(*argv):
        return subprocess.check_output(['git',*argv],cwd=root,text=True,stderr=subprocess.STDOUT,timeout=60).strip()
    source='origin'
    bundle=deployment.get('verified_git_bundle')
    if bundle:
        source=str(Path(bundle['path']).resolve())
        if bundle['commit']!=expected or sha(source)!=bundle['sha256']:
            raise ValueError('Git同步包摘要或提交不符，拒绝同步原目录')
        git('bundle','verify',source)
        git('fetch',source,'main:refs/remotes/origin/main')
    else:
        git('fetch','origin','main')
    if git('rev-parse','origin/main')!=expected:raise RuntimeError('远端main变化，保留旧目录并停止自动同步')
    changes=git('ls-files','--modified','--others','--exclude-standard').splitlines()
    actual={n:sha(root/n) for n in changes if (root/n).is_file()}
    if actual!=deployment['original_uncommitted_sha256']:raise RuntimeError('原目录未提交改动发生变化，拒绝自动收起')
    if changes:
        git('stash','push','--include-untracked','--message',f'pre-stochastic-v2-{expected[:12]}')
        deployment['preserved_stash']=git('rev-parse','refs/stash')
        atomic_json(receipt,deployment)
    git('pull','--ff-only',source,'main')
    if git('rev-parse','HEAD')!=expected:raise RuntimeError('原目录提交号核对失败')
    deployment.update(original_checkout_synchronized=True,synchronization_transport='verified_git_bundle' if bundle else 'origin')
    atomic_json(receipt,deployment)


def long_model(args,ctx):
    from scripts.run_remaining_formal import single_frame_variant
    r=read(args.adopt_training_run/'mlp_training.json')
    h=frozen_mlp_header(args)
    return dict(variant='single_frame_15m',variant_config=single_frame_variant(),origin='pretrained',**h,
        sampling=ctx['sampling'],training_budget_steps=h['evaluated_checkpoint_steps'],training_success_hold_s=0.,
        training_config_verified=True,training_provenance=dict(record=r,source_sha256=sha(args.adopt_training_run/'mlp_training.json')))


def base_spec(args,ctx,models):
    return dict(protocol=PROTOCOL,sampling=ctx['sampling'],config=nominal_config(ctx['old']['base_config']),
        scenarios=ctx['old']['scenarios'],models=models,scientific_code_sha256=scientific_hashes())


def materialize(jobs,root,*,robust=False,e4=False,split='test'):
    # 报告视图不作为缓存；只读的原始逐局结果留在共享store。
    groups=defaultdict(list)
    rows=view_rows(jobs)
    for j,row in zip(jobs,rows):groups[row['variant'],row['training_seed'],row['scenario_id'],row['fault']].append((j,row))
    for (name,seed,sid,fault),entries in groups.items():
        row=dict(entries[0][1])
        if len(entries)>1:
            for key in row:
                if key not in ('training_seed','scenario_seed','repeat','policy_seed') and isinstance(row[key],(bool,int,float)):
                    row[key]=sum(float(r[key]) for _,r in entries)/len(entries)
        row.update(sampling_repeats=len(entries),source_results=[dict(path=j['output'],sha256=sha(j['output'])) for j,_ in entries])
        folder=root/name/f'seed_{seed}'/('robustness' if robust else 'episodes')/(split if e4 else fault)
        atomic_json(folder/f'{sid}.json',row)


def native_report(args,ctx,experiment,models,*,faults=None):
    from scripts.experiment_reports import export_report
    out=args.output_dir/experiment;store=getattr(args,'shared_episode_store',episode_store(args))
    spec=base_spec(args,ctx,models);scenes=spec['scenarios']['test'][:200] if faults else spec['scenarios']['test']
    all_jobs=[]
    for label,fault in (faults or {'nominal':{}}).items():
        all_jobs+=make_jobs(spec,store,scenes=scenes,fault=fault,label=label)
    seal(out/'evaluation_manifest.json',dict(spec=spec,faults=faults or {},jobs=[j['signature'] for j in all_jobs]))
    receipt=run_jobs(args,all_jobs,spec['scenarios']['validation'],out)
    materialize(all_jobs,out,robust=bool(faults))
    atomic_json(out/'scenarios_test.json',spec['scenarios']['test'])
    manifest=dict(signature=digest(spec),evaluation_protocol=PROTOCOL,sampling=ctx['sampling'])
    if len(models)==1:manifest['external_model']=models[0]
    atomic_json(out/'manifest.json',manifest)
    profile={**ctx['old']['settings'],'seeds':list(dict.fromkeys(m['training_seed'] for m in models))}
    report_args=argparse.Namespace(profile='formal',learning_dir=None,e4_methods=None,model_path=None,
        faults=','.join((faults or {}).keys()),limit=None,bootstrap=args.bootstrap)
    report=export_report(experiment,out,out/'reports',report_args,list(dict.fromkeys(m['variant'] for m in models)),
        ctx['spec'],profile,manifest)
    if report['status']!='complete':raise RuntimeError(f'报告覆盖不全：{report}')
    atomic_json(out/'state.json',dict(receipt,status='completed',report=report))


def e4_evaluate(args,ctx):
    from dual_whisker_rl.e4_experiments import method_metadata,scenario_hash,E4Scenario
    from scripts.experiment_reports import export_report
    out=args.output_dir/'e4';old=ctx['e4'];models=[]
    for name,seeds in old['method_seeds'].items():
        for seed in seeds:
            task=args.e4_dir/name/f'seed_{seed}';meta=method_metadata(name)
            variant=dict(_e4_method=name,method=name,history_length=meta['history_length'],encoder=meta['temporal_encoder'])
            cfg=copy.deepcopy(old['config'])
            calibration={}
            if name=='b2_fixed':
                calibration=read(args.e4_dir/'calibration_b2.json');variant['fixed_sector']=calibration['selected_sector']
            if name=='b1_reactive':
                calibration=read(args.e4_dir/'calibration_b1.json');cfg.update(calibration['selected_config'])
            rule=name in ('b0_random','b1_reactive')
            if rule:
                head=dict(sha256=digest(dict(rule=name,config=cfg,rule_seed=seed)),training_seed=seed,evaluated_checkpoint_steps=0)
            else:
                state=read(task/'state.json');metadata=read(task/'metadata.json');head=model_header(task/'model.zip')
                if (state['status']!='completed' or state['signature']!=metadata['signature']
                    or head['training_seed']!=seed or head['evaluated_checkpoint_steps']<old['timesteps']
                    or metadata.get('requested_timesteps')!=old['timesteps']):
                    raise ValueError('E4训练未完成或来源不符')
            model=dict(variant=name,variant_config=variant,origin='rule' if rule else 'historical',**head,
                sampling=ctx['sampling'],training_budget_steps=head['evaluated_checkpoint_steps'],training_success_hold_s=0.,
                training_config_verified=True,training_provenance=dict(manifest_sha256=sha(args.e4_dir/'manifest.json'),
                    metadata=None if rule else metadata,calibration=calibration))
            models.append((model,cfg))
    all_jobs=[]
    banks={}
    for split in ('validation','test'):
        banks[split]=[__import__('json').loads(line) for line in (args.e4_dir/f'scenarios_{split}.jsonl').read_text(encoding='utf8').splitlines()]
        if scenario_hash([E4Scenario.from_dict(s) for s in banks[split]])!=old['scenario_hashes'][split]:raise ValueError('E4题库摘要不一致')
    split_jobs={}
    for split in ('validation','test'):
        jobs=[]
        for model,cfg in models:
            spec=dict(protocol=PROTOCOL,sampling=ctx['sampling'],config=nominal_config(cfg),scenarios=banks,
                      models=[model],scientific_code_sha256=scientific_hashes())
            jobs+=make_jobs(spec,episode_store(args),scenes=banks[split])
        all_jobs+=jobs
        split_jobs[split]=jobs
    seal(out/'evaluation_manifest.json',dict(source_sha=sha(args.e4_dir/'manifest.json'),sampling=ctx['sampling'],
        science=scientific_hashes(),models=[m for m,_ in models],jobs=[j['signature'] for j in all_jobs]))
    for split,jobs in split_jobs.items():
        run_jobs(args,jobs,banks['validation'],out)
        materialize(jobs,out,e4=True,split=split)
    atomic_json(out/'manifest.json',dict(old,evaluation_protocol=PROTOCOL,sampling=ctx['sampling']))
    rargs=argparse.Namespace(profile='formal',learning_dir=None,e4_methods=None,model_path=None,faults='',limit=None,bootstrap=args.bootstrap)
    report=export_report('e4',out,out/'reports',rargs,[],ctx['spec'],ctx['old']['settings'],
                         dict(signature=digest([j['signature'] for j in all_jobs]),evaluation_protocol=PROTOCOL),legacy_root=out)
    if report['status']!='complete':raise RuntimeError('E4新评估覆盖不全')
    atomic_json(out/'state.json',dict(status='completed',episodes=len(all_jobs),report=report))


def run(args):
    from scripts.remaining_formal_support import campaign_lock,create_bundle,monitored_run,reject_live_children
    from scripts.run_remaining_formal import FROZEN_WORKER,FAULTS,GROUPS,report_args
    ctx=context(args);out=args.output_dir
    if args.phase=='plan':return preview(args,ctx)
    if args.phase=='status':return read(out/'state.json') if (out/'state.json').exists() else {'status':'not_started'}
    out.mkdir(parents=True,exist_ok=True)
    with campaign_lock(out):
        reject_live_children(out)
        seal(out/'campaign.json',ctx['signed'])
        bundle=out/'sealed_v2';create_bundle(bundle,ctx['files'],ctx['old'],ctx['spec']['groups'])
        worker=bundle/'scripts/remaining_worker.py'
        if worker.exists() and worker.read_text(encoding='utf8')!=FROZEN_WORKER:raise ValueError('训练包装器被修改')
        worker.write_text(FROZEN_WORKER,encoding='utf8')
        atomic_json(out/'plan.json',preview(args,ctx))
        stages=args.stages.split(',') if isinstance(args.stages,str) else args.stages
        state=dict(status='running',pid=os.getpid(),stages={},protocol=PROTOCOL)
        def train(command,label):
            return monitored_run(command,bundle,out/'logs'/f'{label}.log',out/'processes'/f'{label}.json',
                interval=args.monitor_interval,stall_seconds=args.stall_seconds)
        try:
            for stage in stages:
                state.update(current_stage=stage,status='running');state['stages'][stage]={'status':'running'}
                atomic_json(out/'state.json',state)
                write_index(out,state)
                if stage=='train_mlp':adopt_training(args,ctx)
                elif stage=='compare':
                    from dual_whisker_rl.arrival_reports import make_report
                    spec=comparison_spec(args.source_dir,args.primary_model,args.sealed_source,ctx['sampling'])
                    spec['models'].append(long_model(args,ctx));spec['labels']={'single_frame_15m':'单帧MLP 15M（单模型、无DR）'}
                    spec['english_labels']={'single_frame_15m':'Single frame 15M, no DR'}
                    jobs=make_jobs(spec,episode_store(args));seal(out/'main/manifest.json',spec)
                    receipt=run_jobs(args,jobs,spec['scenarios']['validation'],out/'main')
                    report=make_report(spec,jobs,out/'main',args.bootstrap)
                    atomic_json(out/'main/state.json',dict(receipt,status='completed',report=str(report)))
                elif stage=='e3':
                    from scripts.experiment_reports import export_report
                    report=export_report('e3',args.source_dir,out/'e3/reports',report_args(args),ctx['spec']['groups']['core'],
                        ctx['spec'],ctx['old']['settings'],ctx['old'])
                    atomic_json(out/'e3/state.json',dict(status=report['status'],report=report,note='原训练验证协议；不改写历史曲线'))
                    with (out/'e3/reports/report.md').open('a',encoding='utf8') as stream:
                        stream.write('\nE3保持原训练内部验证口径；本轮修正后的随机动作协议只用于独立正式重评。\n')
                    if report['status']!='complete':raise RuntimeError('E3训练日志覆盖不全，保留缺项并停止后续阶段')
                elif stage=='e4':
                    validate_e4_training_manifest(args,ctx['e4'])
                    train(e4_training_command(args,bundle,'calibrate'),'e4_calibrate')
                    trained_manifest=validate_e4_training_manifest(args,ctx['e4'],required=True)
                    train(e4_training_command(args,bundle,'train'),'e4_train')
                    sub=copy.copy(args);sub.e4_dir=args.e4_training_dir
                    e4_evaluate(sub,{**ctx,'e4':trained_manifest})
                elif stage=='e5':
                    names=list(dict.fromkeys(n for g in GROUPS for n in ctx['spec']['groups'][g]))
                    train([sys.executable,worker,'train','--profile','formal','--output-dir',args.source_dir.parent,
                           '--variants',','.join(names),'--resume'],'e5_training')
                    models=[native_model(args.source_dir,ctx['old'],n,s,ctx['sampling']) for n in names for s in ctx['old']['settings']['seeds']]
                    native_report(args,ctx,'e5',models)
                elif stage=='e6':
                    models=[native_model(args.source_dir,ctx['old'],n,s,ctx['sampling']) for n in ctx['spec']['groups']['core'] for s in ctx['old']['settings']['seeds']]
                    native_report(args,ctx,'e6',models,faults={k:ctx['old']['faults'][k] for k in FAULTS})
                elif stage=='long_faults':
                    from dual_whisker_rl.stochastic_protocol import primary_model
                    for model in (primary_model(args.primary_model,ctx['sampling']),long_model(args,ctx)):
                        sub=copy.copy(args);sub.output_dir=out/'long_faults'/model['variant']
                        # 共用主任务store，避免nominal前200场景重复运行。
                        sub.shared_episode_store=episode_store(args)
                        native_report(sub,ctx,'e6',[model],faults={k:ctx['old']['faults'][k] for k in FAULTS})
                else:raise ValueError(f'未知阶段：{stage}')
                state['stages'][stage]={'status':'completed'};atomic_json(out/'state.json',state)
                write_index(out,state)
            state.update(status='completed' if len(stages)==7 else 'selected_stages_completed')
        except BaseException as exc:
            if state.get('current_stage'):
                state['stages'][state['current_stage']]={'status':'failed','error':repr(exc)}
            state.update(status='failed',error=repr(exc));atomic_json(out/'state.json',state);write_index(out,state);raise
        atomic_json(out/'state.json',state);write_index(out,state)
        return state


def write_index(out,state):
    text=['# 修正羽流与随机动作正式实验','',f"状态：{state['status']}。",'',
        '新评估：进入0.1米即成功、额外DR关闭、逐局可复现随机采样。',
        '旧10.2%来自7秒气团与确定性动作，完整保留为历史条件结果。',
        '训练保持原封存；训练物理、DR、成功目标与预算差异须随报告解释。','',
        *[f"- {s}：{r['status']}" for s,r in state['stages'].items()],'','## 报告','']
    text += [f'- [{p.parent.parent.relative_to(out).as_posix()}]({p.relative_to(out).as_posix()})' for p in sorted(out.rglob('report.md'))]
    (out/'index.md').write_text('\n'.join(text)+'\n',encoding='utf8')
