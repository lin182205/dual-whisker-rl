"""剩余正式实验的封存、进程监控与验收；不改变环境或训练协议。"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from dual_whisker_rl.arrival_protocol import atomic_json, digest, read, sha


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verified_manifest(root):
    """既有封存只读；不为新入口放宽旧签名。"""
    root = Path(root)
    value = read(root / 'manifest.json')
    if value.get('signature') != digest({k: v for k, v in value.items() if k != 'signature'}):
        raise ValueError('原生实验manifest签名不正确')
    if value.get('version') != 2 or value.get('scope') != 'simulation_only':
        raise ValueError('此续跑入口只接受已审查的simulation_only第2版封存')
    for split in ('test', 'validation'):
        if read(root / f'scenarios_{split}.json') != value['scenarios'][split]:
            raise ValueError(f'{split}场景与封存不符')
    return value


def find_source(relative, expected, candidates):
    if Path(relative).is_absolute() or '..' in Path(relative).parts:
        raise ValueError('封存源码路径必须位于源码根目录内')
    for root in candidates:
        path = Path(root) / relative
        if path.is_file() and (expected is None or sha(path) == expected):
            return path
    raise FileNotFoundError(f'找不到与封存SHA-256一致的源码：{relative}')


def bundle_inputs(manifest, candidates, e4_manifest):
    """先查齐全部来源，缺失时不创建半个执行包。"""
    files = {name: find_source(name, expected, candidates)
             for name, expected in manifest['code_sha256'].items()}
    for name, expected in e4_manifest['code_hashes'].items():
        path = find_source(name, expected, candidates)
        if name in files and sha(files[name]) != sha(path):
            raise ValueError(f'原生/E4科学代码不相容：{name}')
        files[name] = path
    for name in ('scripts/experiment_progress.py', 'configs/e4_formal.yaml'):
        files[name] = find_source(name, None, candidates)
    return files


def create_bundle(target, files, manifest, groups):
    import yaml
    target = Path(target)
    for name, source in files.items():
        destination = target / name
        if destination.exists():
            if sha(destination) != sha(source):
                raise ValueError(f'执行包源码发生变化：{name}')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    configs = {
        'configs/remaining_base.yaml': manifest['base_config'],
        'configs/red_experiments.yaml': dict(scope='simulation_only',
            base_config='configs/remaining_base.yaml', profiles={'formal': manifest['settings']},
            variants=manifest['variants'], faults=manifest['faults'], groups=groups),
    }
    for name, value in configs.items():
        path = target / name
        if path.exists() and yaml.safe_load(path.read_text(encoding='utf-8')) != value:
            raise ValueError(f'执行包配置发生变化：{name}')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding='utf-8')
    actual = {p.relative_to(target).as_posix(): sha(p)
              for p in (target / 'dual_whisker_rl').rglob('*.py')}
    actual['scripts/run_red_experiments.py'] = sha(target / 'scripts/run_red_experiments.py')
    if actual != manifest['code_sha256']:
        raise ValueError('执行包包含额外/缺失科学源码，不能恢复原签名')
    return target


def completed_training(task, signature, seed, target):
    path = Path(task) / 'state.json'
    if not path.exists():
        return False
    state = read(path)
    if state.get('signature') != signature:
        raise ValueError(f'训练状态签名不匹配：{path}')
    if state.get('status') != 'completed':
        return False
    from dual_whisker_rl.arrival_protocol import model_header
    header = model_header(Path(task) / 'selected_model.zip')
    if (header['sha256'] != state['selected_model_sha256'] or header['training_seed'] != seed
            or state.get('actual_timesteps', 0) < target):
        raise ValueError(f'已完成训练的模型/预算不匹配：{task}')
    final = model_header(Path(task) / 'final_model.zip')
    if final['training_seed'] != seed or final['evaluated_checkpoint_steps'] < target:
        raise ValueError(f'最终checkpoint预算或seed错误：{task}')
    return True


def verify_episode_set(root, names, seeds, scenes, manifest, *, faults=None, e4=False):
    """按唯一模型/场景键检查，压缩轨迹必须完整可读。"""
    faults = faults or {'nominal': {}}
    verified = 0
    for name in names:
        for seed in seeds[name] if isinstance(seeds, dict) else seeds:
            task = Path(root) / name / f'seed_{seed}'
            model_path = task / ('model.zip' if e4 else 'selected_model.zip')
            model_sha = sha(model_path) if model_path.is_file() else None
            for label, fault in faults.items():
                folder = task / ('robustness' if faults != {'nominal': {}} else 'episodes') / label
                if e4:
                    folder = task / 'episodes/test'
                expected_signature = digest(dict(manifest=manifest['signature'], model=model_sha,
                                                  fault=fault, scenes=scenes)) if not e4 else None
                for scene in scenes:
                    path = folder / f"{scene['scenario_id']}.json"
                    row = read(path)
                    if (row.get('scenario_id') != scene['scenario_id']
                            or row.get('scenario_seed') != scene['seed']
                            or row.get('training_seed', row.get('seed')) != seed):
                        raise ValueError(f'逐局模型/场景键不正确：{path}')
                    if not e4:
                        if (row.get('model_sha256') != model_sha
                                or row.get('evaluation_signature') != expected_signature
                                or row.get('evaluation_scene_count') != len(scenes)):
                            raise ValueError(f'逐局来源或场景预算不正确：{path}')
                    trace = path.with_name(path.stem + '.steps.jsonl.gz')
                    if not e4:
                        with gzip.open(trace, 'rt', encoding='utf-8') as stream:
                            count = sum(1 for line in stream if json.loads(line))
                        if count != row['steps']:
                            raise ValueError(f'压缩轨迹不完整：{trace}')
                    verified += 1
    return verified


@contextmanager
def campaign_lock(root):
    """操作系统锁随进程退出释放，记录文件不被当作删除许可。"""
    path = Path(root) / '.campaign.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    stream = path.open('a+b')
    try:
        if os.name == 'nt':
            import msvcrt
            stream.seek(0)
            if path.stat().st_size == 0:
                stream.write(b'0'); stream.flush()
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        stream.close()


def reject_live_children(root):
    """监控器中断后子进程可能仍在工作；PID和命令同时匹配时拒绝重复启动。"""
    import psutil
    for path in (Path(root) / 'processes').glob('*.json'):
        receipt = read(path)
        if receipt.get('status') not in ('running', 'monitor_interrupted'):
            continue
        try:
            process = psutil.Process(receipt['pid'])
            expected = [str(x) for x in receipt['command']]
            if process.is_running() and process.cmdline() == expected:
                raise RuntimeError(f"既有子进程仍在运行，拒绝重复调度：PID={receipt['pid']}，记录={path}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue


def verify_e4(root, manifest):
    """原E4只持久化逐局JSON；不伪称它保存了压缩逐步轨迹。"""
    from dual_whisker_rl.e4_experiments import E4Scenario, scenario_hash
    from dual_whisker_rl.arrival_protocol import model_header
    root = Path(root)
    count = 0
    for split in ('validation', 'test'):
        scenes = [json.loads(line) for line in (root / f'scenarios_{split}.jsonl').read_text(encoding='utf-8').splitlines() if line]
        if (len(scenes) != manifest['scenario_counts'][split]
                or scenario_hash([E4Scenario.from_dict(s) for s in scenes]) != manifest['scenario_hashes'][split]):
            raise ValueError('原E4场景库被修改')
        for name, seeds in manifest['method_seeds'].items():
            for seed in seeds:
                task = root / name / f'seed_{seed}'
                state = read(task / f'evaluation_{split}.json')
                if state['status'] != 'completed' or state['completed'] != len(scenes):
                    raise ValueError(f'原E4评估未完成：{task}/{split}')
                if name not in ('b0_random', 'b1_reactive'):
                    training = read(task / 'state.json')
                    header = model_header(task / 'model.zip')
                    if (training['status'] != 'completed' or header['training_seed'] != seed
                            or header['evaluated_checkpoint_steps'] < training['timesteps_target']):
                        raise ValueError(f'原E4训练未完成：{task}')
                for scene in scenes:
                    row = read(task / 'episodes' / split / f"{scene['scenario_id']}.json")
                    if (row.get('signature') != state['signature'] or row.get('training_seed') != seed
                            or row.get('seed') != scene['seed'] or row.get('scenario_id') != scene['scenario_id']):
                        raise ValueError(f'原E4逐局来源不正确：{task}/{scene["scenario_id"]}')
                    count += 1
    return count


def monitored_run(command, cwd, log_path, receipt_path, *, interval=60., stall_seconds=3600., env=None):
    """只观察已声明的stdout日志；异常不重试，也不自动杀掉存活子进程。"""
    log_path, receipt_path = Path(log_path), Path(receipt_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    with log_path.open('ab', buffering=0) as stream:
        child = subprocess.Popen([str(x) for x in command], cwd=cwd, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        base = dict(command=[str(x) for x in command], cwd=str(cwd), pid=child.pid,
                    started_at=started, log=str(log_path), monitor_interval_s=interval,
                    stall_timeout_s=stall_seconds, hard_timeout=None)
        atomic_json(receipt_path, dict(base, status='running'))
        previous, changed, alerted = log_path.stat().st_size, time.monotonic(), False
        try:
            while child.poll() is None:
                try:
                    child.wait(timeout=interval)
                except subprocess.TimeoutExpired:
                    pass
                size = log_path.stat().st_size
                if size != previous:
                    previous, changed, alerted = size, time.monotonic(), False
                stale = time.monotonic() - changed
                if stale >= stall_seconds and not alerted:
                    print(f'[停滞提示] PID={child.pid} 日志连续{stale/60:.0f}分钟无增长；仅告警，不终止进程', flush=True)
                    alerted = True
                atomic_json(receipt_path, dict(base, status='running', checked_at=utc_now(),
                    log_bytes=size, output_stalled=alerted, last_progress_age_s=stale))
        except BaseException:
            atomic_json(receipt_path, dict(base, status='monitor_interrupted', child_may_be_alive=child.poll() is None))
            raise
        code = child.returncode
        atomic_json(receipt_path, dict(base, status='completed' if code == 0 else 'failed',
                                      exit_code=code, finished_at=utc_now()))
        if code:
            raise subprocess.CalledProcessError(code, command)
    return read(receipt_path)
