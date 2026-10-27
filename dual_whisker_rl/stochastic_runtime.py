"""新协议执行：共享缓存、逐回合随机流、实际资源选择与规则控制器。"""
from __future__ import annotations
from collections import defaultdict
import copy
import time
from pathlib import Path
from dual_whisker_rl.arrival_protocol import atomic_json, cached, persist, read


def rule_init(model, config, variant):
    from dual_whisker_rl.arrival_runtime import make_env
    global _RULE_ENV, _RULE_CONFIG
    _RULE_ENV, _RULE_CONFIG = make_env(config, variant, 0), config


def rule_episode(job):
    import numpy as np
    from dual_whisker_rl.arrival_runtime import step_record
    from dual_whisker_rl.e4_experiments import RandomController, ReactiveController
    from dual_whisker_rl.stochastic_sampling import policy_seed
    seed = policy_seed(job['sampling'], job['model']['sha256'], job['scene'], job['repeat'])
    name = job['model']['variant']
    controller = RandomController(seed) if name == 'b0_random' else ReactiveController(seed,
        **{k[3:]:v for k,v in _RULE_CONFIG.items() if k.startswith('b1_')})
    if hasattr(controller, 'reset'):
        controller.reset()
    obs, info = _RULE_ENV.reset(seed=job['scene']['seed'], options={'scenario':job['scene']})
    initial, trace = float(info['distance_to_source']), []
    while True:
        action, _ = controller.predict(obs, deterministic=False)
        obs, _, term, trunc, info = _RULE_ENV.step(action)
        trace.append(step_record(_RULE_ENV, np.asarray(action), info))
        if term or trunc:
            break
    return persist(job, trace, initial, 'evaluated_rule', execution_device='cpu')


def run_jobs(args, jobs, validation, output):
    from dual_whisker_rl.arrival_runtime import (resource_limits, bounded_map, cpu_init, cpu_episode,
        choose_execution, gpu_evaluate, cuda_consistency)
    from scripts.experiment_progress import EvaluationProgress
    unique = {j['signature']:j for j in jobs}
    pending = [j for j in unique.values() if not cached(j)]
    atomic_json(Path(output)/'task_plan.json',dict(report_tasks=len(jobs),unique_tasks=len(unique),
        cached=len(unique)-len(pending),new_evaluations=len(pending),
        keys=list(unique),protocol='stochastic-arrival-v2'))
    print(f'[新评估] 协议任务={len(jobs)} 独立任务={len(unique)} 复用={len(unique)-len(pending)} 待评={len(pending)}', flush=True)
    if not pending:
        return dict(expected=len(jobs), unique=len(unique), evaluated=0, reused=len(unique))
    limits = resource_limits()
    groups = defaultdict(list)
    from dual_whisker_rl.arrival_protocol import digest
    for j in pending:
        groups[digest(dict(model=j['model']['sha256'], config=j['config'], variant=j['variant_config']))].append(j)
    progress = EvaluationProgress(len(unique), cached=len(unique)-len(pending))
    choice = None
    choices = []
    started = time.perf_counter()
    def finished(_):
        nonlocal started
        now = time.perf_counter(); progress.finish_episode(now-started); started = now
        progress.report('[随机动作重评]')
    for entries in groups.values():
        sample = entries[0]
        model = copy.deepcopy(sample['model']); model['variant_config'] = sample['variant_config']
        if model['origin'] == 'rule':
            workers = min(limits['cpu_workers_cap'], len(entries),
                          int(args.workers) if str(args.workers) != 'auto' else limits['cpu_workers_cap'])
            if workers < 1:
                raise RuntimeError('内存不足以评估规则控制器')
            for path in bounded_map(rule_episode, entries, workers, initializer=rule_init,
                                    initargs=(model, sample['config'], sample['variant_config'])):
                finished(path)
            continue
        if choice is None:
            choice = choose_execution(limits, model, sample['config'], validation,
                str(args.workers), getattr(args, 'device', 'auto'), getattr(args,'calibration_seconds',32.))
        selected = dict(choice['selected'])
        if selected['backend'] == 'cuda_batched':
            check = cuda_consistency(model, sample['config'], sample['variant_config'], validation, selected['device'])
            if not check['equal']:
                if args.device != 'auto':
                    raise ValueError('新模型CPU/CUDA采样动作不一致')
                selected.update(backend='cpu',device='cpu',workers=limits['cpu_workers_cap'])
        choices.append(dict(model_sha256=model['sha256'], configuration=sample['signature'], selected=selected))
        atomic_json(Path(output)/'execution.json', dict(benchmark=choice, groups=choices, resources=limits))
        workers = min(selected['workers'], len(entries))
        if selected['backend'] == 'cuda_batched':
            gpu_evaluate(entries, model, sample['config'], sample['variant_config'], workers,
                         selected['device'], callback=finished)
        else:
            for path in bounded_map(cpu_episode, entries, workers, initializer=cpu_init,
                                    initargs=(model, sample['config'], sample['variant_config'])):
                finished(path)
    if not all(cached(j) for j in unique.values()):
        raise RuntimeError('新评估仍存在缺局或损坏轨迹')
    return dict(expected=len(jobs), unique=len(unique), evaluated=len(pending), reused=len(unique)-len(pending))
