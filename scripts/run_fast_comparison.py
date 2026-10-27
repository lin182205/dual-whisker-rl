"""复用旧轨迹、只补缺项的75M首次到源主对比；本入口没有训练流程。"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

# 环境进程、转换进程各单线程；GPU协调器集中批量推理。
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from dual_whisker_rl.arrival_protocol import (
    BASELINES, atomic_json, cached, convert_job, describe_sources, digest, make_jobs, read, sha,
)
from scripts.experiment_progress import EvaluationProgress, configure_output_encoding


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('plan', 'run', 'status', 'summarize'), nargs='?', default='plan')
    p.add_argument('--model-path', type=Path, default=Path('results/models/mobile_whisker_gru_ppo_1.zip'))
    p.add_argument('--reuse-from', type=Path, default=Path('results/simulation_experiments/formal'))
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--evaluation-protocol', choices=('legacy-v1','stochastic-arrival-v2'), default='stochastic-arrival-v2')
    p.add_argument('--sealed-source', type=Path, help='与历史训练manifest逐文件匹配的原源码封存')
    p.add_argument('--episode-store', type=Path, help='新协议共享逐局缓存目录')
    p.add_argument('--eval-seed', type=int, default=20261027)
    p.add_argument('--eval-repeats', type=int, default=1)
    p.add_argument('--workers', default='auto', help='auto或正整数；受CPU配额和可用内存约束')
    p.add_argument('--device', default='auto', help='auto/cpu/cuda/cuda:N；auto比较CPU与集中CUDA批量推理')
    p.add_argument('--calibration-seconds', type=float, default=32., help='性能预热采样总预算，0–60秒，不含初始化')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--resume', action='store_true', default=True, help='默认开启，重复命令只补缺项')
    return p


def resolve(args):
    if args.output_dir is None:
        args.output_dir = Path('results/simulation_experiments/arrival_75m')
    for key in ('model_path', 'reuse_from', 'output_dir'):
        value = getattr(args, key)
        setattr(args, key, value.resolve() if value.is_absolute() else (ROOT/value).resolve())
    if args.workers != 'auto' and (not args.workers.isdecimal() or int(args.workers) < 1):
        raise ValueError('workers必须为auto或正整数')
    if args.device not in ('auto', 'cpu', 'cuda') and not (args.device.startswith('cuda:') and args.device[5:].isdecimal()):
        raise ValueError('device必须为auto/cpu/cuda/cuda:N')
    if not 0 <= args.calibration_seconds <= 60 or args.bootstrap < 1:
        raise ValueError('预热预算必须为0–60秒，bootstrap必须为正数')
    spec = describe_sources(args.reuse_from, args.model_path, args.output_dir)
    spec['implementation_sha256'] = {rel: sha(ROOT/rel) for rel in (
        'dual_whisker_rl/arrival_protocol.py', 'dual_whisker_rl/arrival_runtime.py')}
    return spec


def preview(spec, output):
    scenes = len(spec['scenarios']['test'])
    available = missing = existing = 0
    for m in spec['models']:
        for s in spec['scenarios']['test']:
            if (Path(output)/m['variant']/f"seed_{m['training_seed']}"/f"{s['scenario_id']}.json").is_file():
                existing += 1
            if m['origin'] == 'historical':
                base = Path(spec['source'])/m['variant']/f"seed_{m['training_seed']}"/'episodes/nominal'
                if (base/f"{s['scenario_id']}.json").is_file() and (base/f"{s['scenario_id']}.steps.jsonl.gz").is_file():
                    available += 1
                else:
                    missing += 1
    return dict(protocol=spec['protocol'], output=str(output), training_tasks=0,
        test_scenarios=scenes, primary_model=spec['models'][0], baseline_models=len(spec['models'])-1,
        expected_total_episodes=scenes*len(spec['models']), convertible_source_pairs=available,
        missing_source_pairs=missing, primary_evaluation_episodes=scenes,
        existing_result_files_unverified=existing,
        note='计划只检查文件存在性；run/status将验证缓存摘要。无卡模式不启动自动正式评估。')


def seal_output(output, spec):
    payload = {**spec, 'signature': digest(spec)}
    target = Path(output)/'manifest.json'
    if target.exists():
        if read(target) != payload:
            raise ValueError('新输出已属于其他模型/题库/科学实现，请使用新的输出目录')
    else:
        atomic_json(target, payload)


def execute(args, spec):
    output = args.output_dir
    print('[检查] 校验各场景来源摘要与输出缓存；不会重新执行完整缓存', flush=True)
    jobs = make_jobs(spec, output)
    complete, pending = [], []
    for job in jobs:
        (complete if cached(job) else pending).append(job)
    if args.phase == 'status':
        return dict(completed=len(complete), expected=len(jobs), missing=len(pending), training_tasks=0)
    from dual_whisker_rl.arrival_reports import make_report
    if args.phase == 'summarize':
        if not output.is_dir():
            raise FileNotFoundError('输出尚不存在；先run或plan')
        seal_output(output, spec)
        return dict(report=str(make_report(spec, jobs, output, args.bootstrap)))
    resources = None
    if pending:
        from dual_whisker_rl.arrival_runtime import resource_limits
        resources = resource_limits()
        if args.device.startswith('cuda') and not resources['cuda_devices']:
            raise RuntimeError('指定CUDA但当前CUDA不可用；未启动任何转换或评估')
        if args.device == 'auto' and not resources['cuda_devices']:
            raise RuntimeError('当前为无卡模式：请正式开机后再run。plan/status不执行实验；本地验证可显式--device cpu。')
    # 所有写入均在新目录；锁阻止相同目录被并发调度。
    output.mkdir(parents=True, exist_ok=True)
    lock = output/'.run.lock'
    try:
        handle = lock.open('x', encoding='utf-8')
    except FileExistsError as e:
        raise RuntimeError('当前输出已有运行锁；确认原任务结束后再处理，不自动删除锁') from e
    try:
        with handle:
            handle.write(f'{os.getpid()}\n'); handle.flush()
            seal_output(output, spec)
            atomic_json(output/'state.json', dict(status='running', pid=os.getpid(), training_tasks=0))
            historical = [j for j in pending if j['source']]
            evaluation = [j for j in pending if not j['source']]
            if historical:
                from dual_whisker_rl.arrival_runtime import bounded_map
                workers = resources['conversion_workers_cap'] if args.workers == 'auto' else min(int(args.workers), resources['conversion_workers_cap'])
                if workers < 1:
                    raise RuntimeError('当前可用内存不足以启动转换进程')
                workers = min(workers, len(historical))
                progress = EvaluationProgress(len(historical))
                progress.report(f'[旧轨迹转换] workers={workers}', force=True)
                last = time.perf_counter()
                repairs = []
                for result in bounded_map(convert_job, historical, workers):
                    if result['status'] == 'needs_evaluation':
                        evaluation.append(result['job'])
                        repairs.append(dict(output=result['job']['output'], reason=result['reason']))
                    now = time.perf_counter()
                    progress.finish_episode(now-last); last = now
                    progress.report('[旧轨迹转换]')
                atomic_json(output/'repairs.json', repairs)
            if evaluation:
                from dual_whisker_rl.arrival_runtime import choose_execution, bounded_map, cpu_init, cpu_episode, gpu_evaluate
                main_model = spec['models'][0]
                choice = choose_execution(resources, main_model, spec['config'], spec['scenarios']['validation'],
                    args.workers, args.device, args.calibration_seconds)
                atomic_json(output/'execution.json', choice)
                print(f"[执行配置] {choice['selected']}", flush=True)
                groups = defaultdict(list)
                for j in evaluation:
                    groups[j['model']['variant'], j['model']['training_seed']].append(j)
                progress = EvaluationProgress(len(evaluation))
                progress.report('[补评] 仅执行缺失场景', force=True)
                last = time.perf_counter()
                def finished(path):
                    nonlocal last
                    now = time.perf_counter(); progress.finish_episode(now-last); last = now
                    progress.report('[补评]')
                for _, entries in groups.items():
                    m = entries[0]['model']
                    selected = dict(choice['selected'])
                    # 少量历史缺项保持CPU确定性推理，不为每个旧模型再做GPU性能搜索。
                    if m['variant'] != 'main':
                        selected.update(backend='cpu', device='cpu', workers=min(selected['workers'], resources['cpu_workers_cap']))
                    workers = min(selected['workers'], len(entries))
                    if selected['backend'] == 'cuda_batched':
                        gpu_evaluate(entries, m, spec['config'], m['variant_config'], workers, selected['device'], callback=finished)
                    else:
                        for path in bounded_map(cpu_episode, entries, workers, initializer=cpu_init,
                                                initargs=(m, spec['config'], m['variant_config'])):
                            finished(path)
            if not all(cached(j) for j in jobs):
                raise RuntimeError('仍有缺失或损坏场景，不标为完成；重新run可恢复')
            report = make_report(spec, jobs, output, args.bootstrap)
            atomic_json(output/'state.json', dict(status='completed', completed=len(jobs), training_tasks=0,
                expected=len(jobs), report=str(report)))
            return dict(completed=len(jobs), training_tasks=0, report=str(report))
    except BaseException as e:
        atomic_json(output/'state.json', dict(status='interrupted' if isinstance(e, KeyboardInterrupt) else 'failed',
            error=repr(e), training_tasks=0))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main(argv=None):
    configure_output_encoding()
    args = parser().parse_args(argv)
    if args.evaluation_protocol == 'stochastic-arrival-v2':
        from scripts.stochastic_campaign import fast
        result = fast(args)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return result
    spec = resolve(args)
    if args.phase == 'plan':
        result = preview(spec, args.output_dir)
    else:
        result = execute(args, spec)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == '__main__':
    mp.freeze_support()
    main()
