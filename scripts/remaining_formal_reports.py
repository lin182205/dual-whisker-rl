"""复用已完成主对比，补充单帧MLP与长训练模型的冻结扰动评估。"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path

from dual_whisker_rl.arrival_protocol import atomic_json, cached, digest, make_jobs, persist, read, sha
from scripts.experiment_progress import EvaluationProgress


def mlp_descriptor(args):
    from scripts.run_remaining_formal import single_frame_variant, validate_mlp
    record = read(args.output_dir / 'mlp_training.json')
    header = validate_mlp(record, args.mlp_timesteps, record['training_config'])
    if record.get('model_sha256') != header['sha256']:
        raise ValueError('单帧MLP来源摘要发生变化')
    return dict(variant='single_frame_15m', variant_config=single_frame_variant(),
        origin='pretrained', training_config_verified=True, **header,
        training_budget_steps=header['evaluated_checkpoint_steps'], training_success_hold_s=0.,
        training_metadata_sha256=sha(Path(record['log_dir']) / 'run_metadata.json'))


def seal(path, value):
    value = {**value, 'signature': digest(value)}
    if path.exists() and read(path) != value:
        raise ValueError(f'评估来源改变，拒绝复用：{path}')
    atomic_json(path, value)


def evaluate_cpu(args, jobs, model, cfg, *, fault=None, label='nominal'):
    from dual_whisker_rl.arrival_runtime import bounded_map, cpu_init, cpu_episode, resource_limits
    pending = [j for j in jobs if not cached(j)]
    if not pending:
        print(f"[复用] {model['variant']} {label} {len(jobs)}局", flush=True)
        return
    limits = resource_limits(probe_cuda=False)
    workers = min(args.workers, limits['cpu_workers_cap'], len(pending))
    if workers < 1:
        raise RuntimeError('内存不足以启动评估进程')
    progress = EvaluationProgress(len(jobs), cached=len(jobs)-len(pending))
    progress.report(f"[补充评估] {model['variant']} {label} workers={workers}", force=True)
    import time
    started = time.perf_counter()
    function, init = (cpu_episode, cpu_init) if fault is None else (_fault_episode, _fault_init)
    initargs = (model, cfg, model['variant_config'])
    if fault is not None:
        initargs += (fault,)
    for _ in bounded_map(function, pending, workers, initializer=init, initargs=initargs):
        now = time.perf_counter()
        progress.finish_episode(now-started); started = now
        progress.report(f"[补充评估] {model['variant']} {label}")
    if not all(cached(j) for j in jobs):
        raise RuntimeError('补充评估仍有缺失或损坏的场景')


def compare_mlp(args, ctx):
    from dual_whisker_rl.arrival_reports import make_report
    output = args.output_dir / 'comparison'
    model = mlp_descriptor(args)
    original = ctx['arrival_spec']
    original_jobs = make_jobs(original, args.arrival_dir)
    # 原目录只读；损坏不能用一个completed标签掩盖，也不覆盖既有75M报告。
    if not all(cached(j) for j in original_jobs):
        raise ValueError('旧75M对比存在缺失/损坏逐局记录，需先恢复原对比')
    spec = copy.deepcopy(original)
    spec['models'].append(model)
    spec['labels'] = {'single_frame_15m': '单帧MLP 15M（单模型、无DR）'}
    spec['english_labels'] = {'single_frame_15m': 'Single frame 15M, no DR'}
    seal(output / 'manifest.json', spec)
    new_jobs = make_jobs({**spec, 'models': [model]}, output)
    evaluate_cpu(args, new_jobs, model, spec['config'])
    jobs = original_jobs + new_jobs
    report = make_report(spec, jobs, output, args.bootstrap)
    if read(output / 'latest_report.json')['status'] != 'complete':
        raise RuntimeError('七方法补充主对比报告不完整')
    atomic_json(output / 'state.json', dict(status='completed', completed=len(jobs), expected=len(jobs),
        reused_episodes=len(original_jobs), new_model_episodes=len(new_jobs), report=str(report)))


_FAULT_POLICY = None
_FAULT_ENV = None


def _fault_init(model, config, variant, fault):
    from dual_whisker_rl.arrival_runtime import load_policy, check_contract
    from dual_whisker_rl.experiment_env import make_experiment_env
    global _FAULT_POLICY, _FAULT_ENV
    if sha(model['path']) != model['sha256']:
        raise ValueError('长训练模型在加载前被修改')
    _FAULT_POLICY = load_policy(model['path'], 'cpu')
    _FAULT_ENV = make_experiment_env(config, variant, seed=model['training_seed'], evaluating=True, fault=fault)
    _FAULT_ENV.unwrapped.record_trajectory = True
    check_contract(_FAULT_POLICY, _FAULT_ENV, variant)


def _fault_episode(job):
    import numpy as np
    import time
    from dual_whisker_rl.arrival_runtime import step_record
    started = time.perf_counter()
    obs, info = _FAULT_ENV.reset(seed=job['scene']['seed'], options={'scenario': job['scene']})
    initial, rows = float(info['distance_to_source']), []
    while True:
        action, _ = _FAULT_POLICY.predict(obs, deterministic=True)
        obs, _, term, trunc, info = _FAULT_ENV.step(action)
        rows.append(step_record(_FAULT_ENV, np.asarray(action), info))
        if term or trunc:
            break
    return persist(job, rows, initial, 'evaluated_cpu', fault=job['fault_label'],
        evaluation_wall_s=time.perf_counter()-started, execution_device='cpu')


def long_faults(args, ctx):
    from scripts.run_remaining_formal import FAULTS, report_args
    from scripts.experiment_reports import export_report
    root = args.output_dir / 'long_faults'
    original = ctx['arrival_spec']
    models = [original['models'][0], mlp_descriptor(args)]
    configuration = dict(protocol='long-frozen-arrival-faults-v1', models=models, config=original['config'],
        scenarios=original['scenarios'], faults={name: ctx['manifest']['faults'][name] for name in FAULTS},
        implementation_sha256=sha(Path(__file__)))
    seal(root / 'manifest.json', configuration)
    atomic_json(root / 'scenarios_test.json', original['scenarios']['test'])
    scenes = original['scenarios']['test'][:200]
    reports, count = [], 0
    for model in models:
        for label in FAULTS:
            fault = ctx['manifest']['faults'][label]
            spec = {**original, 'models': [model], 'scenarios': {**original['scenarios'], 'test': scenes}}
            jobs = make_jobs(spec, root)
            for job in jobs:
                job['fault_label'] = label
                job['signature'] = digest(dict(original=job['signature'], fault=fault, label=label,
                                               implementation=configuration['implementation_sha256']))
                job['output'] = str(root / model['variant'] / f"seed_{model['training_seed']}" /
                                    'robustness' / label / f"{job['scene']['scenario_id']}.json")
            evaluate_cpu(args, jobs, model, original['config'], fault=fault, label=label)
            count += len(jobs)
        profile = {**ctx['manifest']['settings'], 'seeds': [model['training_seed']]}
        out = root / 'reports' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') / model['variant']
        receipt = export_report('e6', root, out, report_args(args), [model['variant']], ctx['spec'],
            profile, dict(signature=digest(configuration), external_model=model))
        if receipt['status'] != 'complete':
            raise RuntimeError('长训练模型扰动覆盖不完整')
        reports.append(receipt)
    atomic_json(root / 'state.json', dict(status='completed', expected=count, completed=count,
        success_hold_s=0., uncertainty_scope='conditional_on_one_model', reports=reports))
