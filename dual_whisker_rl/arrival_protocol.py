"""首次到源协议与旧轨迹转换；本模块及转换进程不加载Torch/PPO。"""
from __future__ import annotations
import copy
import csv
import gzip
import hashlib
import json
import math
import os
import re
from pathlib import Path
import statistics
from uuid import uuid4
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'arrival-comparison-v1'
BASELINES = ('fixed', 'periodic', 'single_frame', 'history_mlp', 'no_dr')
LABELS = {'main': '主动策略75M（单模型）', 'fixed': '固定触须', 'periodic': '周期扫描',
          'single_frame': '单帧MLP', 'history_mlp': '历史MLP', 'no_dr': '主动策略2M（无DR历史对照）'}
KNOWN_RUNNERS = {'4236f4240a76ef99d15ea1e615c643fab3db5af7efb7292b90a27e30807f63c0',
                 'aebc4866f1c800ffd246fac97834fa750bd15fa2e93412fb88043453a2a1985d'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def atomic_trace(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        with gzip.open(temp, 'wt', encoding='utf-8', compresslevel=1) as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def model_header(path):
    with zipfile.ZipFile(path) as z:
        d = json.loads(z.read('data'))
    if not isinstance(d.get('seed'), int) or not isinstance(d.get('num_timesteps'), int):
        raise ValueError(f'模型缺少真实seed或训练步数：{path}')
    return dict(path=str(Path(path).resolve()), sha256=sha(path), training_seed=d['seed'],
                evaluated_checkpoint_steps=d['num_timesteps'])


def describe_sources(source, model, output):
    """只读校验来源；科学代码逐文件匹配，旧运行器只允许已审查版本。"""
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('新输出必须与旧实验目录分离，不能覆盖或嵌套旧结果')
    if output == Path(model).resolve() or output in Path(model).resolve().parents:
        raise ValueError('新输出不得包含源模型')
    old = read(source / 'manifest.json')
    unsigned = {k: v for k, v in old.items() if k != 'signature'}
    if old.get('signature') != digest(unsigned):
        raise ValueError('旧manifest签名错误')
    if old.get('version') not in (2, 3) or old.get('scope') != 'simulation_only':
        raise ValueError('不支持的旧实验协议')
    hashes = old.get('code_sha256', {})
    scientific = {p: h for p, h in hashes.items() if p.startswith('dual_whisker_rl/')}
    required = {'dual_whisker_rl/envs/mobile_whisker_env.py', 'dual_whisker_rl/envs/whisker_only_env.py',
                'dual_whisker_rl/experiment_env.py', 'dual_whisker_rl/e4_experiments.py'}
    if not required <= scientific.keys():
        raise ValueError('旧manifest缺少科学实现摘要')
    for rel, expected in scientific.items():
        p = (ROOT / rel).resolve()
        if ROOT not in p.parents or not p.is_file() or sha(p) != expected:
            raise ValueError(f'科学实现与旧实验不一致，禁止转换：{rel}')
    runner = hashes.get('scripts/run_red_experiments.py')
    if old['version'] == 2 and runner not in KNOWN_RUNNERS:
        raise ValueError('旧运行器不属于已验证的version=2转换来源')
    if old['version'] == 3 and runner != sha(ROOT / 'scripts/run_red_experiments.py'):
        raise ValueError('version=3旧运行器摘要不匹配')
    banks = {}
    for split in ('test', 'validation'):
        banks[split] = read(source / f'scenarios_{split}.json')
        if banks[split] != old['scenarios'][split]:
            raise ValueError(f'{split}题库与封存不一致')
        ids = [s['scenario_id'] for s in banks[split]]
        if any(not isinstance(sid, str) or re.fullmatch(r'[A-Za-z0-9_-]+', sid) is None for sid in ids):
            raise ValueError('场景ID必须是安全的文件名标识')
        seeds = [s['seed'] for s in banks[split]]
        if len(set(ids)) != len(ids) or len(set(seeds)) != len(seeds) or not ids:
            raise ValueError('题库为空或场景/种子重复')
    if {s['seed'] for s in banks['test']} & {s['seed'] for s in banks['validation']}:
        raise ValueError('验证与测试场景种子重叠')
    cfg = copy.deepcopy(old['base_config'])
    if cfg.get('plume_model') != 'dynamic' or float(cfg.get('goal_radius', .1)) != .1:
        raise ValueError('只支持当前动态羽流与0.1米目标半径')
    if float(cfg.get('success_hold_s', 1)) not in (0., 1.):
        raise ValueError('只支持已知的瞬时或保持1秒来源')
    cfg.update(goal_radius=.1, success_hold_s=0., domain_randomization=False)
    models = [dict(variant='main', variant_config={'method': 'main', 'history_length': 20,
                'encoder': 'gru', 'config': {'domain_randomization': False}}, origin='pretrained',
                training_config_verified=False, **model_header(model))]
    models[0]['training_budget_steps'] = models[0]['evaluated_checkpoint_steps']
    metadata_path = Path(model).resolve().parent.parent/'logs'/Path(model).stem/'run_metadata.json'
    models[0]['latest_training_metadata'] = None
    if metadata_path.is_file():
        meta = read(metadata_path)
        models[0]['latest_training_metadata'] = {k: meta.get(k) for k in (
            'seed', 'starting_num_timesteps', 'final_num_timesteps', 'config', 'history_length',
            'temporal_encoder', 'observation_mode', 'reward', 'resume_from')}
        models[0]['training_metadata_sha256'] = sha(metadata_path)
    models[0]['training_success_hold_s'] = 0. if metadata_path.is_file() else None
    for name in BASELINES:
        variant = old['variants'][name]
        if set(variant.get('config', {})) - {'domain_randomization'} or variant.get('feature_mask', 'none') != 'none':
            raise ValueError('主对比转换不接受改变物理或特征的消融配置')
        for seed in old['settings']['seeds']:
            d = source / name / f'seed_{seed}'
            state = read(d / 'state.json')
            header = model_header(d / 'selected_model.zip')
            if (state.get('status') != 'completed' or state.get('signature') != old['signature']
                    or header['sha256'] != state.get('selected_model_sha256') or header['training_seed'] != seed):
                raise ValueError(f'旧模型状态或摘要不匹配：{d}')
            models.append(dict(variant=name, variant_config=variant, origin='historical',
                               training_config_verified=True, training_budget_steps=state['actual_timesteps'],
                               training_success_hold_s=old['base_config'].get('success_hold_s', 1), **header))
    return dict(protocol=PROTOCOL, config=cfg, scenarios=banks, models=models,
                source=str(source), source_manifest_sha256=sha(source / 'manifest.json'),
                source_signature=old['signature'], source_settings=old['settings'],
                scientific_code_sha256=scientific, source_hold_s=old['base_config'].get('success_hold_s', 1))


def make_jobs(spec, output):
    """缓存指纹包含源文件摘要；不因报表代码或并行度改变而重跑。"""
    jobs = []
    for m in spec['models']:
        legacy_signature = digest(dict(manifest=spec['source_signature'], model=m['sha256'], fault={}, scenes=spec['scenarios']['test']))
        for scene in spec['scenarios']['test']:
            folder = Path(output) / m['variant'] / f"seed_{m['training_seed']}"
            source_dir = Path(spec['source']) / m['variant'] / f"seed_{m['training_seed']}" / 'episodes/nominal'
            row_path = source_dir / f"{scene['scenario_id']}.json"
            trace_path = source_dir / f"{scene['scenario_id']}.steps.jsonl.gz"
            origin = None
            if m['origin'] == 'historical':
                origin = dict(row=str(row_path), trace=str(trace_path),
                              row_sha256=sha(row_path) if row_path.is_file() else None,
                              trace_sha256=sha(trace_path) if trace_path.is_file() else None)
            key = dict(protocol=spec['protocol'], config=spec['config'], scene=scene,
                       model=m, source=origin, scientific_code_sha256=spec['scientific_code_sha256'],
                       implementation_sha256=spec.get('implementation_sha256', {}))
            jobs.append(dict(signature=digest(key), model=m, scene=scene, config=spec['config'],
                             source=origin, legacy_evaluation_signature=legacy_signature,
                             source_scene_count=len(spec['scenarios']['test']), output=str(folder / f"{scene['scenario_id']}.json")))
    return jobs


def cached(job):
    p = Path(job['output'])
    if not p.exists():
        return False
    try:
        d = read(p)
        if d.get('job_signature') != job['signature']:
            raise ValueError(f'已有结果来源发生变化，使用新输出目录：{p}')
        checksum = d.pop('result_sha256')
        if checksum != digest(d):
            return False
        trace = p.with_name(p.stem + '.steps.jsonl.gz')
        return trace.is_file() and sha(trace) == d['trace_sha256']
    except (json.JSONDecodeError, KeyError, OSError):
        return False


def compute_metrics(rows, initial_distance, scene, cfg):
    """直接评估与转换共用同一统计定义，路径包含第一步位移。"""
    if not rows:
        raise ValueError('轨迹不能为空')
    n, dt = len(rows), float(cfg.get('dt', .2))
    last = rows[-1]
    success = float(last['distance_to_source']) <= float(cfg['goal_radius'])
    bounds = cfg.get('world_bounds', [-2.5, 2.5])
    oob = not (bounds[0] < last['x'] < bounds[1] and bounds[0] < last['y'] < bounds[1])
    hits = sum(bool(r['odor_hit']) for r in rows)
    blank = [r for r in rows if not r['odor_hit']]
    reacq = [r for r in rows if r['reacquisition_event']]
    component_names = rows[0]['reward_components'].keys()
    components = {k: sum(float(r['reward_components'][k]) for r in rows) for k in component_names}
    trends = [((b[k] - a[k]) / dt > .01) for a, b in zip(rows, rows[1:]) for k in ('left', 'right')]
    return dict(success=success, termination='success' if success else 'out_of_bounds' if oob else 'timeout',
        steps=n, elapsed_s=n*dt, horizon_s=cfg['max_steps']*dt,
        penalized_time_s=n*dt if success else cfg['max_steps']*dt,
        initial_distance=initial_distance, final_distance=float(last['distance_to_source']),
        path_length=sum(float(r['position_displacement']) for r in rows), odor_hits=hits,
        odor_hit_ratio=hits/n, blank_steps=len(blank),
        blank_body_spin_ratio=sum(r['move_action'] in ('spin_left', 'spin_right') for r in blank)/max(len(blank), 1),
        blank_body_motion_ratio=sum(bool(r['body_moved']) for r in blank)/max(len(blank), 1),
        blank_whisker_motion_ratio=sum(bool(r['whisker_moved']) for r in blank)/max(len(blank), 1),
        reacquisition_events=len(reacq), reacquisition_per_1000_steps=len(reacq)/n*1000,
        mean_reacquisition_time_s=statistics.fmean(r['reacquisition_blank_s'] for r in reacq) if reacq else None,
        whisker_only_reacquisition_events=sum(bool(r['whisker_only_reacquisition']) for r in rows),
        body_assisted_reacquisition_events=sum(bool(r['body_assisted_reacquisition']) for r in rows),
        passive_reacquisition_events=sum(bool(r['passive_reacquisition']) for r in rows),
        positive_response_trend_ratio=statistics.fmean(trends) if trends else None,
        mean_abs_lr_difference=statistics.fmean(abs(r['left']-r['right']) for r in rows),
        reward=sum(components.values()), reward_components=components,
        scenario_id=scene['scenario_id'], scenario_seed=scene['seed'], split=scene['split'],
        condition=f"{scene['distance_bin']}/{scene['crosswind_bin']}",
        nominal_wind_speed=scene['wind_speed'], source_strength=scene['source_strength'],
        distance_bin=scene['distance_bin'], crosswind_bin=scene['crosswind_bin'],
        heading_quadrant=scene['heading_quadrant'])


def persist(job, rows, initial_distance, origin, **extra):
    d = compute_metrics(rows, initial_distance, job['scene'], job['config'])
    m = job['model']
    d.update(variant=m['variant'], training_seed=m['training_seed'], model_sha256=m['sha256'],
             evaluated_checkpoint_steps=m['evaluated_checkpoint_steps'], training_budget_steps=m['training_budget_steps'],
             training_config=m['variant_config'], training_config_verified=m['training_config_verified'],
             result_origin=origin, source=job['source'], protocol=PROTOCOL, job_signature=job['signature'], **extra)
    p = Path(job['output'])
    trace_path = p.with_name(p.stem + '.steps.jsonl.gz')
    atomic_trace(trace_path, rows)
    d['trace_sha256'] = sha(trace_path)
    d['result_sha256'] = digest(d)
    atomic_json(p, d)
    return str(p)


def convert_job(job):
    """完整验证旧压缩轨迹，在首次到源处截断；缺项返回待补评任务。"""
    source = job['source']
    if not source or not source['row_sha256'] or not source['trace_sha256']:
        return dict(status='needs_evaluation', job=job, reason='旧逐局记录或轨迹缺失')
    try:
        if sha(source['row']) != source['row_sha256'] or sha(source['trace']) != source['trace_sha256']:
            raise ValueError('读取期间源文件发生变化')
        old = read(source['row'])
        if (old.get('model_sha256') != job['model']['sha256']
                or old.get('scenario_id') != job['scene']['scenario_id']
                or old.get('scenario_seed') != job['scene']['seed']
                or old.get('training_seed') != job['model']['training_seed']
                or old.get('variant') != job['model']['variant']
                or old.get('fault') != 'nominal'
                or old.get('evaluation_scene_count') != job['source_scene_count']
                or old.get('evaluation_signature') != job['legacy_evaluation_signature']):
            raise ValueError('旧回合来源与模型/场景不匹配')
        prefix, reached, count = [], False, 0
        full_components, full_reward, last_row = {}, 0., None
        with gzip.open(source['trace'], 'rt', encoding='utf-8') as f:
            for count, line in enumerate(f, 1):
                r = json.loads(line)
                if (r['step'] != count or not math.isfinite(float(r['distance_to_source']))
                        or abs(float(r['elapsed_s']) - count*job['config'].get('dt', .2)) > 1e-8):
                    raise ValueError('轨迹顺序/距离/时钟错误')
                for key, value in r['reward_components'].items():
                    if not math.isfinite(float(value)):
                        raise ValueError('奖励含非有限值')
                    full_components[key] = full_components.get(key, 0.) + value
                if abs(sum(r['reward_components'].values()) - float(r['reward'])) > 1e-8:
                    raise ValueError('逐步奖励分解不一致')
                full_reward += r['reward']
                last_row = r
                if not reached:
                    prefix.append(r)
                    reached = r['distance_to_source'] <= job['config']['goal_radius']
        if count != old['steps'] or not prefix or count > job['config']['max_steps']:
            raise ValueError('轨迹长度不完整')
        if 'reward' in old and abs(full_reward - old['reward']) > 1e-6:
            raise ValueError('旧汇总与完整轨迹奖励不一致')
        if 'reward_components' in old and any(abs(full_components[k] - old['reward_components'][k]) > 1e-6 for k in full_components):
            raise ValueError('旧奖励分量汇总不一致')
        if 'final_distance' in old and abs(last_row['distance_to_source'] - old['final_distance']) > 1e-8:
            raise ValueError('旧最终距离不一致')
        for r in prefix:
            if any(not math.isfinite(float(r[k])) for k in ('left', 'right', 'x', 'y', 'position_displacement')):
                raise ValueError('轨迹含非有限读数')
        if not reached:
            if old['termination'] not in ('out_of_bounds', 'timeout'):
                raise ValueError('没有到源却被标为成功')
            if old['termination'] == 'timeout' and count != job['config']['max_steps']:
                raise ValueError('未完成的超时轨迹')
        else:
            # 目标首次触发后立即终止，取消该步停滞惩罚，成功奖励只发一次。
            last = prefix[-1]
            last['reward_components']['goal_bonus'] = float(job['config'].get('goal_bonus', 50))
            last['reward_components']['stagnation_penalty'] = 0.
            last['reward'] = sum(last['reward_components'].values())
            last['is_success'] = True
        persist(job, prefix, old['initial_distance'], 'converted_trajectory', source_steps=count)
        return dict(status='converted', output=job['output'])
    except (OSError, EOFError, KeyError, json.JSONDecodeError, ValueError, TypeError) as e:
        return dict(status='needs_evaluation', job=job, reason=str(e))
