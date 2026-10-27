"""先训练单帧MLP，再严格复用第2版封存完成剩余正式仿真实验。"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / 'scripts'))

from dual_whisker_rl.arrival_protocol import (
    atomic_json, cached, describe_sources, digest, make_jobs, model_header, read, sha,
)
from scripts.remaining_formal_support import (
    bundle_inputs, campaign_lock, completed_training, create_bundle, monitored_run,
    reject_live_children, utc_now, verified_manifest, verify_episode_set, verify_e4,
)
from scripts.experiment_progress import configure_output_encoding

STAGES = ('train_mlp', 'compare', 'e3', 'e4', 'e5', 'e6', 'long_faults')
FAULTS = ('nominal', 'wind24', 'gain80', 'servo150', 'delay2')
GROUPS = ('history', 'features', 'rewards', 'geometry', 'sensor', 'randomization')
FROZEN_WORKER = '''"""仅调度原封存函数，不修改科学实现或汇总旧文件。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_red_experiments as runner
runner.configure_output_encoding()
args = runner.parser().parse_args()
spec, profile, cfg, names, faults, payload = runner.resolve(args)
root = runner.project_path(args.output_dir) / args.profile
manifest = runner.prepare(root, profile, cfg, payload)
lock = root / '.run.lock'
with lock.open('x') as handle:
    handle.write(str(__import__('os').getpid()))
try:
    if args.phase == 'train':
        for name in names:
            for seed in profile['seeds']:
                runner.train_one(root, name, spec['variants'][name], seed, profile, cfg, manifest, args)
    else:
        runner.evaluate(root, names, spec, profile, cfg, manifest, args, robust=args.phase == 'robustness')
finally:
    lock.unlink()
'''


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', nargs='?', choices=('plan', 'run', 'status'), default='plan')
    p.add_argument('--source-dir', type=Path, default=Path('results/simulation_experiments/formal'))
    p.add_argument('--arrival-dir', type=Path, default=Path('results/simulation_experiments/arrival_75m'))
    p.add_argument('--e4-dir', type=Path, default=Path('results/e4/formal'))
    p.add_argument('--e4-timesteps', type=int, default=8_000_000, help='新随机评估流水线的E4每方法每seed目标总步数')
    p.add_argument('--e4-training-dir', type=Path, help='新预算E4训练目录，末级为formal；默认按预算独立建目录')
    p.add_argument('--e4-concurrency', choices=('auto','1','2'), default='auto', help='E4训练并行模型数；auto先短测再选择')
    p.add_argument('--e4-probe-seconds', type=float, default=150., help='E4五种吞吐短测的采样总计时预算，不含初始化和预热')
    p.add_argument('--episode-store', type=Path, help='显式复用同一科学协议的共享逐局缓存')
    p.add_argument('--primary-model', type=Path, default=Path('results/models/mobile_whisker_gru_ppo_1.zip'))
    p.add_argument('--sealed-source', type=Path, help='含原封存源码与E4辅助文件的根目录')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--evaluation-protocol', choices=('legacy-v1','stochastic-arrival-v2'), default='stochastic-arrival-v2')
    p.add_argument('--eval-seed', type=int, default=20261027)
    p.add_argument('--eval-repeats', type=int, default=1)
    p.add_argument('--adopt-training-run', type=Path, default=Path('results/remaining_formal_15m'))
    p.add_argument('--device', default='auto')
    p.add_argument('--calibration-seconds', type=float, default=32.)
    p.add_argument('--mlp-timesteps', type=int, default=15_000_000)
    p.add_argument('--mlp-n-envs', type=int, default=12)
    p.add_argument('--mlp-n-steps', type=int, default=512)
    p.add_argument('--mlp-checkpoint-freq', type=int, default=100_000)
    p.add_argument('--seed', type=int, help='新MLP的种子；省略时首次分配后固定')
    p.add_argument('--workers', default='auto', help='评估进程数或auto；受实际CPU配额和内存限制')
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--monitor-interval', type=float, default=60.)
    p.add_argument('--stall-seconds', type=float, default=3600.)
    p.add_argument('--stages', default=','.join(STAGES), help='用于分阶段恢复；省略时执行全部')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--internal-stage', choices=STAGES, help=argparse.SUPPRESS)
    return p


def resolve(args):
    if args.output_dir is None:
        args.output_dir = Path('results/remaining_formal_15m')
    args.workers = 10 if args.workers == 'auto' else int(args.workers)
    for key in ('source_dir', 'arrival_dir', 'e4_dir', 'primary_model', 'sealed_source', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve() if value.is_absolute() else (ROOT / value).resolve())
    if min(args.mlp_timesteps, args.mlp_n_envs, args.mlp_n_steps, args.mlp_checkpoint_freq,
           args.workers, args.bootstrap, args.monitor_interval, args.stall_seconds) <= 0:
        raise ValueError('预算、并行数和监控间隔必须为正')
    if args.mlp_n_steps * args.mlp_n_envs < 2:
        raise ValueError('PPO采样轮次至少需要2步')
    if args.seed is not None and not 0 <= args.seed < 2**32:
        raise ValueError('seed超出范围')
    selected = args.stages.split(',')
    if len(set(selected)) != len(selected) or set(selected) - set(STAGES):
        raise ValueError('stages包含未知或重复阶段')
    args.stages = [s for s in STAGES if s in selected]
    for source in (args.source_dir, args.arrival_dir, args.e4_dir):
        if args.output_dir == source or args.output_dir in source.parents or source in args.output_dir.parents:
            raise ValueError('流水线输出必须与既有实验目录分离')
    return args


def context(args):
    import yaml
    manifest = verified_manifest(args.source_dir)
    spec = yaml.safe_load((ROOT / 'configs/red_experiments.yaml').read_text(encoding='utf-8'))
    groups = spec['groups']
    native = list(dict.fromkeys(groups['core'] + [n for g in GROUPS for n in groups[g]]))
    if set(native) - set(manifest['variants']):
        raise ValueError('当前实验分组包含旧封存未定义的配置')
    e4 = read(args.e4_dir / 'manifest.json')
    candidates = [p for p in (args.sealed_source,
        ROOT / 'results/server_analysis_20261027/snapshot', ROOT) if p is not None]
    files = bundle_inputs(manifest, candidates, e4)
    completed = [(n, s) for n in native for s in manifest['settings']['seeds']
                 if completed_training(args.source_dir / n / f'seed_{s}', manifest['signature'],
                                       s, manifest['settings']['timesteps'])]
    metadata = args.primary_model.parent.parent / 'logs' / args.primary_model.stem / 'run_metadata.json'
    meta = read(metadata)
    primary = model_header(args.primary_model)
    if (meta.get('base_observation_dim') != 16 or meta.get('temporal_encoder') != 'gru'
            or meta.get('config', {}).get('domain_randomization', False)
            or meta.get('final_num_timesteps') != primary['evaluated_checkpoint_steps']):
        raise ValueError('75M来源元数据与模型不匹配，不能声称对齐训练')
    training_config = copy.deepcopy(meta['config'])
    training_config['domain_randomization'] = False
    # 比较入口原有模型、实现与题库必须仍是同一份；不接受只看状态文件。
    arrival_spec = describe_sources(args.source_dir, args.primary_model, args.arrival_dir)
    arrival_spec['implementation_sha256'] = {rel: sha(ROOT / rel) for rel in (
        'dual_whisker_rl/arrival_protocol.py', 'dual_whisker_rl/arrival_runtime.py')}
    if read(args.arrival_dir / 'manifest.json') != {**arrival_spec, 'signature': digest(arrival_spec)}:
        raise ValueError('已完成75M对比的封存与当前来源不一致')
    signed = dict(version='remaining-formal-v1', source_manifest_sha256=sha(args.source_dir / 'manifest.json'),
        e4_manifest_sha256=sha(args.e4_dir / 'manifest.json'), primary=primary,
        primary_metadata_sha256=sha(metadata), arrival_manifest_sha256=sha(args.arrival_dir / 'manifest.json'),
        frozen_files={n: sha(p) for n, p in files.items()}, groups=groups,
        mlp=dict(target=args.mlp_timesteps, n_envs=args.mlp_n_envs, n_steps=args.mlp_n_steps,
                 checkpoint_freq=args.mlp_checkpoint_freq, config=training_config, seed=args.seed),
        code={rel: sha(ROOT / rel) for rel in ('scripts/run_remaining_formal.py',
            'scripts/remaining_formal_support.py', 'scripts/remaining_formal_reports.py',
            'scripts/train_mobile_whisker_ppo.py', 'scripts/experiment_reports.py',
            'dual_whisker_rl/arrival_reports.py')}, fault_conditions=list(FAULTS))
    return dict(manifest=manifest, spec=spec, groups=groups, native=native, files=files,
                completed=completed, metadata=meta, training_config=training_config,
                arrival_spec=arrival_spec, signed=signed, e4=e4)


def preview(args, ctx):
    old = ctx['manifest']
    pending = len(ctx['native']) * len(old['settings']['seeds']) - len(ctx['completed'])
    e4_pending = 0
    for name, seeds in ctx['e4']['method_seeds'].items():
        if name in ('b0_random', 'b1_reactive'):
            continue
        for seed in seeds:
            task = args.e4_dir / name / f'seed_{seed}'
            state = read(task / 'state.json') if (task / 'state.json').exists() else {}
            if state.get('status') != 'completed' or not (task / 'model.zip').is_file():
                e4_pending += 1
    return dict(stages=args.stages, new_mlp_target_steps=args.mlp_timesteps,
        completed_native_training=len(ctx['completed']), remaining_native_training=pending,
        remaining_native_target_steps=pending * old['settings']['timesteps'],
        remaining_e4_training=e4_pending, e4_target_steps_each=1_000_000,
        e4_calibration_additional=True, formal_seed_count=len(old['settings']['seeds']),
        new_mlp_test_episodes=len(old['scenarios']['test']),
        e6_episodes=len(ctx['groups']['core']) * len(old['settings']['seeds']) * len(FAULTS) * min(200, len(old['scenarios']['test'])),
        long_model_fault_episodes=2 * len(FAULTS) * min(200, len(old['scenarios']['test'])),
        source_signature=old['signature'], output=str(args.output_dir), training_success='first_arrival_0.1m',
        formal_native_success='radius_hold_1s', no_new_hardware=True)


def training_record(args):
    return args.output_dir / 'mlp_training.json'


def mlp_namespace(args, record):
    from scripts import train_mobile_whisker_ppo as trainer
    previous = sys.argv
    sys.argv = ['train_mobile_whisker_ppo.py', '--config', str(args.output_dir / 'mlp_config.yaml'),
        '--timesteps', str(args.mlp_timesteps), '--seed', str(record['seed']),
        '--n-envs', str(args.mlp_n_envs), '--n-steps', str(args.mlp_n_steps),
        '--batch-size', str(min(256, args.mlp_n_envs * args.mlp_n_steps)),
        '--vec-env-backend', 'subproc' if args.mlp_n_envs > 1 else 'dummy',
        '--temporal-encoder', 'mlp', '--history-length', '1',
        '--checkpoint-freq', str(args.mlp_checkpoint_freq), '--experiment-name', 'single_frame_mlp_15m']
    try:
        value = trainer.parse_args()
    finally:
        sys.argv = previous
    if record.get('experiment_name'):
        for key in ('model_path', 'log_dir', 'checkpoint_dir', 'tensorboard_dir', 'tensorboard_run_dir'):
            setattr(value, key, Path(record[key]))
        value.experiment_name = record['experiment_name']
        value.experiment_keyword = 'single_frame_mlp_15m'
        value.run_name = record['experiment_name']
        value._output_paths_resolved = True
    return value


def validate_mlp(record, target, config):
    from stable_baselines3 import PPO
    from stable_baselines3.common.torch_layers import FlattenExtractor
    from dual_whisker_rl.arrival_runtime import make_env, check_contract
    header = model_header(record['model_path'])
    if header['training_seed'] != record['seed'] or header['evaluated_checkpoint_steps'] < target:
        raise ValueError('MLP最终模型的实际seed/总步数不满足预算')
    meta = read(Path(record['log_dir']) / 'run_metadata.json')
    if (meta['config'] != config or meta['history_length'] != 1 or meta['temporal_encoder'] != 'mlp'
            or meta['final_num_timesteps'] != header['evaluated_checkpoint_steps']):
        raise ValueError('MLP最终模型与训练元数据不一致')
    policy = PPO.load(record['model_path'], device='cpu')
    env = make_env({**config, 'success_hold_s': 0.}, single_frame_variant(), record['seed'])
    try:
        check_contract(policy, env, single_frame_variant())
        if not isinstance(policy.policy.features_extractor, FlattenExtractor) or policy.observation_space.shape != (16,):
            raise ValueError('新模型不是16维单帧MLP')
    finally:
        env.close()
    return header


def single_frame_variant():
    return dict(method='b4_single_frame', encoder='mlp', history_length=1,
                config={'domain_randomization': False})


def train_mlp(args, ctx):
    from scripts import train_mobile_whisker_ppo as trainer
    path = training_record(args)
    record = read(path) if path.exists() else dict(seed=args.seed if args.seed is not None else secrets.randbits(31))
    value = mlp_namespace(args, record)
    if not record.get('experiment_name'):
        trainer.resolve_output_paths(value)
        record.update({k: str(getattr(value, k)) for k in (
            'model_path', 'log_dir', 'checkpoint_dir', 'tensorboard_dir', 'tensorboard_run_dir', 'experiment_name')})
        record.update(target_total_steps=args.mlp_timesteps, status='reserved', training_config=ctx['training_config'])
        atomic_json(path, record)
    final = Path(record['model_path'])
    if final.exists():
        header = validate_mlp(record, args.mlp_timesteps, ctx['training_config'])
        atomic_json(path, dict(record, status='completed', model_sha256=header['sha256'],
                              actual_timesteps=header['evaluated_checkpoint_steps']))
        return
    checkpoints = list(Path(record['checkpoint_dir']).glob(record['experiment_name'] + '_*_steps.zip'))
    if checkpoints:
        checkpoint = max(checkpoints, key=lambda p: int(re.search(r'_(\d+)_steps.zip$', p.name).group(1)))
        actual = model_header(checkpoint)['evaluated_checkpoint_steps']
        if actual >= args.mlp_timesteps:
            raise ValueError('checkpoint已达目标但最终模型缺失；拒绝盲目增加训练，请检查训练收尾')
        value.resume_from = checkpoint
        value.timesteps = args.mlp_timesteps - actual
        value._output_paths_resolved = False
    atomic_json(path, dict(record, status='running', started_at=utc_now()))
    trained = trainer.train(value)
    if trained is not None:
        trained.logger.close()
    header = validate_mlp(record, args.mlp_timesteps, ctx['training_config'])
    atomic_json(path, dict(record, status='completed', model_sha256=header['sha256'],
                          actual_timesteps=header['evaluated_checkpoint_steps'], finished_at=utc_now()))


def report_args(args, *, faults=None):
    return argparse.Namespace(profile='formal', learning_dir=None, e4_methods=None, model_path=None,
        faults=','.join(faults or FAULTS), limit=None, bootstrap=args.bootstrap)


def native_report(args, ctx, experiment, names, *, legacy_root=None):
    from scripts.experiment_reports import export_report
    out = args.output_dir / 'reports' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') / experiment
    receipt = export_report(experiment, args.source_dir, out, report_args(args), names,
        ctx['spec'], ctx['manifest']['settings'], ctx['manifest'], legacy_root=legacy_root)
    if receipt['status'] != 'complete':
        raise RuntimeError(f'{experiment}覆盖/报告未完成：{receipt}')
    atomic_json(args.output_dir / f'{experiment}_report.json', receipt)
    return receipt


def frozen_command(args, phase, names):
    return [sys.executable, args.output_dir / 'sealed_v2/scripts/remaining_worker.py', phase,
        '--profile', 'formal', '--output-dir', args.source_dir.parent,
        '--variants', ','.join(names), '--faults', ','.join(FAULTS), '--resume']


def nested_run(args, command, name, cwd):
    return monitored_run(command, cwd, args.output_dir / 'logs' / f'{name}.log',
        args.output_dir / 'processes' / f'{name}.json', interval=args.monitor_interval,
        stall_seconds=args.stall_seconds)


def write_index(args, state):
    lines = ['# 单帧MLP与剩余正式实验', '', f"状态：{state['status']}。", '',
        '新MLP为单帧、无DR、首次进入0.1米；原生E3/E5/E6保持原训练与连续1秒判据，原E4保留独立协议。',
        '不同训练预算与协议的结果分别报告，不把单模型场景区间解释为多seed算法结论。', '',
        '| 阶段 | 状态 |', '|---|---|']
    lines += [f"| {name} | {state['stages'].get(name, {}).get('status', 'pending')} |" for name in STAGES]
    lines += ['', '## 已生成报告', '']
    receipts = [args.output_dir / 'comparison/latest_report.json']
    receipts += [args.output_dir / f'{name}_report.json' for name in ('e3', 'e4', 'e5', 'e6')]
    for path in receipts:
        if path.is_file():
            receipt = read(path)
            report = Path(receipt['report'])
            report = report if report.is_absolute() else ROOT / report
            lines.append(f"- [{path.stem}]({os.path.relpath(report, args.output_dir).replace(os.sep, '/')})")
    long_state = args.output_dir / 'long_faults/state.json'
    if long_state.exists():
        for receipt in read(long_state)['reports']:
            report = Path(receipt['report'])
            report = report if report.is_absolute() else ROOT / report
            lines.append(f"- [长训练模型扰动]({os.path.relpath(report, args.output_dir).replace(os.sep, '/')})")
    (args.output_dir / 'index.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def execute_stage(args, ctx, stage):
    from scripts.remaining_formal_reports import compare_mlp, long_faults
    bundle = args.output_dir / 'sealed_v2'
    if stage == 'train_mlp':
        train_mlp(args, ctx)
    elif stage == 'compare':
        compare_mlp(args, ctx)
    elif stage == 'e3':
        native_report(args, ctx, 'e3', ctx['groups']['core'])
    elif stage == 'e4':
        nested_run(args, [sys.executable, bundle / 'scripts/run_e4_experiments.py', 'run',
            '--profile', 'formal', '--config', bundle / 'configs/e4_formal.yaml',
            '--output-dir', args.e4_dir.parent, '--resume'], 'e4_runner', bundle)
        verify_e4(args.e4_dir, ctx['e4'])
        native_report(args, ctx, 'e4', [], legacy_root=args.e4_dir)
    elif stage == 'e5':
        names = list(dict.fromkeys(n for group in GROUPS for n in ctx['groups'][group]))
        nested_run(args, frozen_command(args, 'train', names), 'e5_train', bundle)
        nested_run(args, frozen_command(args, 'evaluate', names), 'e5_evaluate', bundle)
        verify_episode_set(args.source_dir, names, ctx['manifest']['settings']['seeds'],
                           ctx['manifest']['scenarios']['test'], ctx['manifest'])
        native_report(args, ctx, 'e5', names)
    elif stage == 'e6':
        names = ctx['groups']['core']
        nested_run(args, frozen_command(args, 'robustness', names), 'e6_evaluate', bundle)
        scenes = ctx['manifest']['scenarios']['test'][:200]
        verify_episode_set(args.source_dir, names, ctx['manifest']['settings']['seeds'], scenes,
            ctx['manifest'], faults={name: ctx['manifest']['faults'][name] for name in FAULTS})
        native_report(args, ctx, 'e6', names)
    elif stage == 'long_faults':
        long_faults(args, ctx)


def main():
    import yaml
    configure_output_encoding()
    args = parser().parse_args()
    if args.evaluation_protocol == 'stochastic-arrival-v2':
        from scripts.stochastic_campaign import run
        print(json.dumps(run(args), ensure_ascii=False, indent=2), flush=True)
        return
    args = resolve(args)
    ctx = context(args)
    if args.phase == 'plan':
        print(json.dumps(preview(args, ctx), ensure_ascii=False, indent=2)); return
    state_path = args.output_dir / 'state.json'
    if args.phase == 'status':
        print(json.dumps(read(state_path) if state_path.exists() else {'status': 'not_started'}, ensure_ascii=False, indent=2)); return
    if args.internal_stage:
        execute_stage(args, ctx, args.internal_stage); return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with campaign_lock(args.output_dir):
        reject_live_children(args.output_dir)
        campaign = args.output_dir / 'campaign.json'
        seal = {**ctx['signed'], 'signature': digest(ctx['signed'])}
        if campaign.exists():
            if not args.resume:
                raise FileExistsError('已有流水线，请使用--resume')
            if read(campaign) != seal:
                raise ValueError('流水线源码、模型或预算改变，拒绝混用既有输出')
        else:
            atomic_json(campaign, seal)
        create_bundle(args.output_dir / 'sealed_v2', ctx['files'], ctx['manifest'], ctx['groups'])
        (args.output_dir / 'sealed_v2/scripts/remaining_worker.py').write_text(FROZEN_WORKER, encoding='utf-8')
        (args.output_dir / 'mlp_config.yaml').write_text(yaml.safe_dump(ctx['training_config'], sort_keys=False), encoding='utf-8')
        atomic_json(args.output_dir / 'plan.json', preview(args, ctx))
        state = read(state_path) if state_path.exists() else dict(stages={})
        state.update(status='running', pid=os.getpid(), updated_at=utc_now())
        atomic_json(state_path, state)
        write_index(args, state)
        try:
            for stage in args.stages:
                if state['stages'].get(stage, {}).get('status') == 'completed':
                    # 缓存仍需核对输入/输出，不以一个完成标签代替真实性检查。
                    if stage == 'train_mlp':
                        record = read(training_record(args))
                        header = validate_mlp(record, args.mlp_timesteps, ctx['training_config'])
                        if header['sha256'] != record['model_sha256']:
                            raise ValueError('已完成MLP模型被更改')
                    else:
                        # 后续阶段内部有逐局校验与去重，重新进入不会重复训练。
                        pass
                    if stage == 'train_mlp':
                        print(f'[复用] {stage}', flush=True); continue
                print(f'[阶段] {stage} 开始', flush=True)
                state['current_stage'] = stage
                state['stages'][stage] = dict(status='running', started_at=utc_now())
                atomic_json(state_path, state)
                write_index(args, state)
                command = [sys.executable, Path(__file__).resolve(), 'run']
                # 保留用户参数，内部工作进程也重新校验所有来源。
                argv = sys.argv[1:]
                if argv and argv[0] in ('plan', 'run', 'status'):
                    argv = argv[1:]
                command += argv + ['--internal-stage', stage]
                nested_run(args, command, stage, ROOT)
                state['stages'][stage].update(status='completed', finished_at=utc_now())
                atomic_json(state_path, state)
            state.update(status='completed' if all(state['stages'].get(s, {}).get('status') == 'completed' for s in STAGES)
                         else 'selected_stages_completed', finished_at=utc_now())
            atomic_json(state_path, state)
            write_index(args, state)
            print(json.dumps(state, ensure_ascii=False, indent=2))
        except BaseException as exc:
            stage = state.get('current_stage')
            if stage:
                state['stages'][stage].update(status='failed', error=str(exc))
            state.update(status='interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed', error=str(exc), updated_at=utc_now())
            atomic_json(state_path, state)
            write_index(args, state)
            raise


if __name__ == '__main__':
    main()
