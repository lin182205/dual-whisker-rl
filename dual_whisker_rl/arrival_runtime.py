"""首次到源对比的资源探测、多进程环境和集中CUDA推理。"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import math
import multiprocessing as mp
import os
from pathlib import Path
import time

from dual_whisker_rl.arrival_protocol import persist, sha


def resource_limits(probe_cuda=True):
    """取亲和性、cgroup配额和内存的交集，不能把宿主总核数当分配核数。"""
    import psutil
    process = psutil.Process()
    affinity = len(process.cpu_affinity()) if hasattr(process, 'cpu_affinity') else os.cpu_count() or 1
    quota = float(affinity)
    try:
        fields = Path('/sys/fs/cgroup/cpu.max').read_text().split()
        if fields[0] != 'max':
            quota = min(quota, int(fields[0])/int(fields[1]))
    except (OSError, ValueError):
        try:
            q = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read_text())
            period = int(Path('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read_text())
            if q > 0:
                quota = min(quota, q/period)
        except (OSError, ValueError):
            pass
    available = int(psutil.virtual_memory().available)
    for limit_path, usage_path in [('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory.current'),
                                  ('/sys/fs/cgroup/memory/memory.limit_in_bytes', '/sys/fs/cgroup/memory/memory.usage_in_bytes')]:
        try:
            limit = Path(limit_path).read_text().strip()
            if limit != 'max':
                available = min(available, max(0, int(limit)-int(Path(usage_path).read_text())))
        except (OSError, ValueError):
            pass
    cpus = max(1, math.ceil(quota))
    # 子进程首次导入Torch与模拟环境也要占内存；留出协调器和系统空间。
    budget_mb = max(0, available*.8/2**20 - 512)
    cpu_cap = min(cpus, max(0, int(budget_mb//512)))
    env_cap = min(max(1, cpus-1), max(0, int(budget_mb//160)))
    conversion_cap = min(cpus, max(0, int((available*.8/2**20-64)//64)))
    devices = []
    if probe_cuda:
        import torch
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(index)
                devices.append(dict(device=f'cuda:{index}', name=torch.cuda.get_device_name(index),
                                    free_bytes=free, total_bytes=total))
    return dict(cpu_affinity=affinity, cpu_quota=quota, effective_cpus=cpus,
                available_memory_bytes=available, conversion_workers_cap=conversion_cap, cpu_workers_cap=cpu_cap, env_workers_cap=env_cap,
                cuda_devices=devices)


def bounded_map(function, jobs, workers, initializer=None, initargs=()):
    """至多保留两倍worker的在途任务；每个任务只发放一次。"""
    if not jobs:
        return
    context = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=initializer, initargs=initargs) as pool:
        iterator = iter(jobs)
        pending = {}
        for _ in range(min(len(jobs), workers*2)):
            job = next(iterator)
            pending[pool.submit(function, job)] = job
        while pending:
            ready, _ = wait(pending, return_when=FIRST_COMPLETED)
            for f in ready:
                pending.pop(f)
                yield f.result()
                try:
                    job = next(iterator)
                except StopIteration:
                    continue
                pending[pool.submit(function, job)] = job


def make_env(config, variant, seed):
    from dual_whisker_rl.experiment_env import make_experiment_env
    env = make_experiment_env(config, variant, seed=seed, evaluating=True)
    env.unwrapped.record_trajectory = True
    return env


def load_policy(path, device):
    import torch
    from stable_baselines3 import PPO
    from dual_whisker_rl.agents import GRUHistoryExtractor  # 注册自定义提取器
    torch.set_num_threads(1)
    model = PPO.load(path, device=device)
    model.policy.set_training_mode(False)
    return model


def check_contract(model, env, variant=None):
    if model.observation_space != env.observation_space or model.action_space != env.action_space:
        raise ValueError('模型与评估环境的观测/动作空间不一致')
    extractor = model.policy.features_extractor
    if variant is not None:
        actual_gru = type(extractor).__name__ == 'GRUHistoryExtractor'
        if (variant.get('encoder', 'gru') == 'gru') != actual_gru:
            raise ValueError('模型编码器与声明的GRU/MLP不一致')
    if hasattr(extractor, 'history_length') and extractor.history_length != env.history_length:
        raise ValueError('模型历史长度不匹配')
    if hasattr(extractor, 'base_observation_dim') and extractor.base_observation_dim != env.base_observation_dim:
        raise ValueError('模型单帧维度不匹配')


def step_record(env, action, info):
    r = dict(env.unwrapped.trajectory[-1])
    r.update(step=env.unwrapped.step_count, elapsed_s=env.unwrapped.step_count*env.unwrapped.dt,
             policy_action=[int(x) for x in action.reshape(-1)],
             wind_speed=float(info['wind_speed']), wind_direction=float(info['wind_direction']))
    return r


_CPU_MODEL = None
_CPU_ENV = None


def cpu_init(model, config, variant, barrier=None):
    global _CPU_MODEL, _CPU_ENV
    if sha(model['path']) != model['sha256']:
        raise ValueError('加载前模型内容发生变化')
    _CPU_MODEL = load_policy(model['path'], 'cpu')
    _CPU_ENV = make_env(config, variant, model['training_seed'])
    check_contract(_CPU_MODEL, _CPU_ENV, variant)
    if barrier is not None:
        barrier.wait(timeout=180)


def cpu_episode(job):
    import numpy as np
    started = time.perf_counter()
    obs, info = _CPU_ENV.reset(seed=job['scene']['seed'], options={'scenario': job['scene']})
    initial_distance = float(info['distance_to_source'])
    trace = []
    while True:
        action, _ = _CPU_MODEL.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = _CPU_ENV.step(action)
        trace.append(step_record(_CPU_ENV, np.asarray(action), info))
        if terminated or truncated:
            break
    return persist(job, trace, initial_distance, 'evaluated_cpu',
                   evaluation_wall_s=time.perf_counter()-started, execution_device='cpu')


def cpu_probe(scene):
    obs, _ = _CPU_ENV.reset(seed=scene['seed'], options={'scenario': scene})
    steps = 0
    for _ in range(16):
        action, _ = _CPU_MODEL.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = _CPU_ENV.step(action)
        steps += 1
        if terminated or truncated:
            break
    return steps


def ping(_):
    return os.getpid()


def benchmark_cpu(model, cfg, variant, scenes, workers, seconds):
    context = mp.get_context('spawn')
    barrier = context.Barrier(workers)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=cpu_init,
            initargs=(model, cfg, variant, barrier)) as pool:
        # 等全部模型初始化结束后才开始吞吐计时。
        list(pool.map(ping, range(workers)))
        start, steps = time.perf_counter(), 0
        while time.perf_counter()-start < seconds:
            steps += sum(pool.map(cpu_probe, [scenes[i % len(scenes)] for i in range(workers)]))
        elapsed = time.perf_counter()-start
    return dict(backend='cpu', device='cpu', workers=workers, steps=steps,
                elapsed_s=elapsed, steps_per_second=steps/max(elapsed, 1e-9))


def _env_loop(conn, config, variant, slots, seed):
    """GPU worker只推进CPU环境，不导入PPO、不创建CUDA上下文。"""
    envs = []
    try:
        envs = [make_env(config, variant, seed+i) for i in range(slots)]
        conn.send(('ready', None))
        while True:
            command, values = conn.recv()
            if command == 'close':
                return
            result = []
            if command == 'reset':
                for slot, scene in values:
                    obs, info = envs[slot].reset(seed=scene['seed'], options={'scenario': scene})
                    result.append((slot, obs, float(info['distance_to_source'])))
            elif command == 'step':
                import numpy as np
                for slot, action in values:
                    obs, _, term, trunc, info = envs[slot].step(action)
                    result.append((slot, obs, term or trunc, step_record(envs[slot], np.asarray(action), info)))
            else:
                raise ValueError('未知环境worker命令')
            conn.send(('ok', result))
    except BaseException as e:
        try:
            conn.send(('error', repr(e)))
        except (BrokenPipeError, EOFError):
            pass
    finally:
        for env in envs:
            env.close()
        conn.close()


class EnvFarm:
    """一个协调器批量推理；每个环境进程默认放两个slot以摊薄IPC。"""
    def __init__(self, cfg, variant, workers, slots, seed):
        context = mp.get_context('spawn')
        self.connections, self.processes, self.slots = [], [], slots
        try:
            for i in range(workers):
                parent, child = context.Pipe()
                p = context.Process(target=_env_loop, args=(child, cfg, variant, slots, seed+i*slots), daemon=True)
                p.start()
                child.close()
                self.connections.append(parent)
                self.processes.append(p)
            for c in self.connections:
                self.receive(c)
        except BaseException:
            self.close()
            raise

    @staticmethod
    def receive(conn):
        if not conn.poll(180):
            raise RuntimeError('环境worker无响应（180秒），停止该批任务')
        kind, value = conn.recv()
        if kind == 'error':
            raise RuntimeError(f'环境worker失败：{value}')
        return value

    def call(self, command, assignments):
        groups = {}
        for key, value in assignments:
            worker, slot = divmod(key, self.slots)
            groups.setdefault(worker, []).append((slot, value))
        for i, values in groups.items():
            self.connections[i].send((command, values))
        result = []
        for i in groups:
            result.extend((i*self.slots+r[0], *r[1:]) for r in self.receive(self.connections[i]))
        return result

    def close(self):
        for conn, process in zip(self.connections, self.processes):
            try:
                if process.is_alive():
                    conn.send(('close', None))
            except (OSError, EOFError):
                pass
        for conn, process in zip(self.connections, self.processes):
            process.join(timeout=5)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            conn.close()


def cuda_consistency(model, cfg, variant, scenes, device):
    """用验证场景短片段比较单帧CPU、批量CPU与批量CUDA确定性动作。"""
    import numpy as np
    if sha(model['path']) != model['sha256']:
        raise ValueError('加载前模型内容发生变化')
    cpu = load_policy(model['path'], 'cpu')
    env = make_env(cfg, variant, model['training_seed'])
    samples = []
    try:
        check_contract(cpu, env, variant)
        for scene in scenes[:3]:
            obs, _ = env.reset(seed=scene['seed'], options={'scenario': scene})
            for _ in range(8):
                samples.append(obs.copy())
                action, _ = cpu.predict(obs, deterministic=True)
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
        per_frame = np.stack([cpu.predict(obs, deterministic=True)[0] for obs in samples])
        batch_cpu = cpu.predict(np.stack(samples), deterministic=True)[0]
        del cpu
        cuda = load_policy(model['path'], device)
        batch_cuda = cuda.predict(np.stack(samples), deterministic=True)[0]
        equal = bool(np.array_equal(per_frame, batch_cpu) and np.array_equal(per_frame, batch_cuda))
        del cuda
        return dict(equal=equal, observations=len(samples), device=device)
    finally:
        env.close()


def gpu_evaluate(jobs, model, cfg, variant, workers, device, callback=None, probe_seconds=None, probe_scenes=None):
    import numpy as np
    if sha(model['path']) != model['sha256']:
        raise ValueError('加载前模型内容发生变化')
    policy = load_policy(model['path'], device)
    check_env = make_env(cfg, variant, model['training_seed'])
    try:
        check_contract(policy, check_env, variant)
    finally:
        check_env.close()
    slots = 2
    count = min(len(jobs), workers*slots) if probe_seconds is None else workers*slots
    if count == 0:
        return dict(steps=0, elapsed_s=0, steps_per_second=0)
    farm = EnvFarm(cfg, variant, min(workers, math.ceil(count/slots)), slots, model['training_seed'])
    active, observations, initial, traces = {}, {}, {}, {}
    iterator = iter(jobs)
    probe_index = 0

    def get_job():
        nonlocal probe_index
        if probe_seconds is not None:
            scene = probe_scenes[probe_index % len(probe_scenes)]
            probe_index += 1
            return dict(scene=scene)
        return next(iterator, None)

    def reset(assignments):
        result = farm.call('reset', [(slot, job['scene']) for slot, job in assignments])
        for slot, job in assignments:
            active[slot], traces[slot] = job, []
        for slot, obs, distance in result:
            observations[slot], initial[slot] = obs, distance

    steps, start = 0, None
    try:
        reset([(slot, get_job()) for slot in range(count)])
        # 一次预热不计入采样预算；正式评估不推进环境、不丢弃任何动作。
        policy.predict(np.stack([observations[s] for s in sorted(active)]), deterministic=True)
        start = time.perf_counter()
        while active:
            keys = sorted(active)
            actions, _ = policy.predict(np.stack([observations[k] for k in keys]), deterministic=True)
            response = farm.call('step', list(zip(keys, actions)))
            reset_jobs = []
            for slot, obs, done, row in response:
                steps += 1
                traces[slot].append(row)
                observations[slot] = obs
                if done or (probe_seconds is not None and len(traces[slot]) >= 16):
                    job = active.pop(slot)
                    if probe_seconds is None:
                        path = persist(job, traces[slot], initial[slot], 'evaluated_cuda', execution_device=device)
                        if callback:
                            callback(path)
                    new_job = get_job()
                    if new_job is not None:
                        reset_jobs.append((slot, new_job))
                    else:
                        observations.pop(slot, None)
                        initial.pop(slot, None)
                        traces.pop(slot, None)
            if probe_seconds is not None and time.perf_counter()-start >= probe_seconds:
                break
            if reset_jobs:
                reset(reset_jobs)
        elapsed = time.perf_counter()-start
        return dict(backend='cuda_batched', device=device, workers=workers, steps=steps,
                    elapsed_s=elapsed, steps_per_second=steps/max(elapsed, 1e-9))
    finally:
        farm.close()
        del policy


def choose_execution(resources, model, cfg, scenes, workers='auto', device='auto', seconds=32.):
    """只按吞吐选执行配置；验证片段不产生正式测试结果。"""
    if not 0 <= seconds <= 60:
        raise ValueError('性能预热预算必须在0到60秒之间')
    available = resources['cuda_devices']
    if device.startswith('cuda') and not available:
        raise ValueError('指定CUDA，但当前CUDA不可用')
    requested = None if workers == 'auto' else int(workers)
    if requested is not None and requested < 1:
        raise ValueError('workers必须为auto或正整数')
    selected_gpu = None
    if device != 'cpu' and available:
        if device in ('auto', 'cuda'):
            selected_gpu = max(available, key=lambda d: d['free_bytes'])['device']
        elif any(d['device'] == device for d in available):
            selected_gpu = device
        else:
            raise ValueError(f'指定设备不可用：{device}')
    candidates, notes = [], []
    variant = model['variant_config']
    if device in ('cpu', 'auto'):
        cap = resources['cpu_workers_cap']
        if requested and requested > cap:
            if device != 'auto':
                raise ValueError(f'workers={requested}超过当前CPU/内存安全上限{cap}')
            notes.append(dict(backend='cpu', reason='指定并行数超过CPU模型内存预算'))
            cap = 0
        else:
            cap = requested or cap
        for n in (sorted({cap, max(1, cap//2)} if requested is None else {cap}) if cap > 0 else []):
            candidates.append(dict(backend='cpu', device='cpu', workers=n))
    if selected_gpu:
        try:
            if next(d['free_bytes'] for d in available if d['device'] == selected_gpu) < 512*2**20:
                raise RuntimeError('CUDA空闲显存少于512MiB')
            check = cuda_consistency(model, cfg, variant, scenes, selected_gpu)
        except Exception as e:
            if device != 'auto':
                raise
            check = dict(equal=False, error=repr(e), device=selected_gpu)
        notes.append(dict(cuda_consistency=check))
        if not check['equal']:
            if device != 'auto':
                raise ValueError('CPU/CUDA确定性动作不一致，拒绝显式CUDA评估')
        else:
            cap = resources['env_workers_cap']
            if requested and requested > cap:
                if device != 'auto':
                    raise ValueError(f'workers={requested}超过当前环境进程/内存安全上限{cap}')
                notes.append(dict(backend='cuda_batched', reason='指定并行数超过环境进程内存预算'))
                cap = 0
            else:
                cap = requested or cap
            for n in (sorted({cap, max(1, cap//2)} if requested is None else {cap}) if cap > 0 else []):
                candidates.append(dict(backend='cuda_batched', device=selected_gpu, workers=n))
    if not candidates:
        raise ValueError('没有可用执行配置')
    # 显式CPU或CUDA配置允许关闭预热；自动配置至少需要一次真实吞吐测量。
    if seconds == 0:
        if len(candidates) != 1:
            raise ValueError('自动选择需要正的预热预算；关闭预热需显式指定设备和workers')
        return dict(selected=candidates[0], candidates=[], notes=notes, resources=resources)
    measured = []
    budget = seconds/len(candidates)
    for c in candidates:
        print(f"[性能预热] {c} 采样预算={budget:.1f}秒（不含进程初始化）", flush=True)
        try:
            if c['backend'] == 'cpu':
                result = benchmark_cpu(model, cfg, variant, scenes, c['workers'], budget)
            else:
                result = gpu_evaluate([], model, cfg, variant, c['workers'], c['device'],
                                      probe_seconds=budget, probe_scenes=scenes)
            measured.append(result)
            print(f"[性能预热] 吞吐={result['steps_per_second']:.1f}步/秒", flush=True)
        except Exception as e:
            notes.append(dict(candidate=c, error=repr(e)))
            if device != 'auto':
                raise
    if not measured:
        raise RuntimeError(f'所有性能预热配置失败：{notes}')
    winner = max(measured, key=lambda r: r['steps_per_second'])
    return dict(selected={k: winner[k] for k in ('backend', 'device', 'workers')},
                candidates=measured, notes=notes, resources=resources)
