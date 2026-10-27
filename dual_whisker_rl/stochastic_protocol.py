"""新评估与旧训练封存分离；旧模型只读，新轨迹按语义任务去重。"""
from __future__ import annotations
import copy
import math
from pathlib import Path
from dual_whisker_rl.arrival_protocol import BASELINES, atomic_json, cached, digest, model_header, read, sha
from dual_whisker_rl.stochastic_sampling import PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
SCIENCE = ('dual_whisker_rl/envs/whisker_only_env.py', 'dual_whisker_rl/envs/mobile_whisker_env.py',
           'dual_whisker_rl/experiment_env.py', 'dual_whisker_rl/e4_experiments.py',
           'dual_whisker_rl/arrival_protocol.py', 'dual_whisker_rl/arrival_runtime.py',
           'dual_whisker_rl/stochastic_sampling.py', 'dual_whisker_rl/stochastic_protocol.py',
           'dual_whisker_rl/stochastic_runtime.py')


def settings(seed=20261027, repeats=1):
    if not 0 <= seed < 2**32 or repeats < 1:
        raise ValueError('eval-seed须为32位非负整数，eval-repeats须为正数')
    return dict(protocol=PROTOCOL, action_mode='stochastic', eval_seed=seed, eval_repeats=repeats,
                sampler='independent_pcg64_inverse_cdf_v1')


def scientific_hashes():
    # 包含环境与观测组件，避免只对入口文件签名。
    paths = set(SCIENCE) | {p.relative_to(ROOT).as_posix() for p in (ROOT/'dual_whisker_rl/envs').glob('*.py')}
    for directory in ('observation','agents'):
        paths |= {p.relative_to(ROOT).as_posix() for p in (ROOT/'dual_whisker_rl'/directory).glob('*.py')}
    paths.add('dual_whisker_rl/hardware/sensor_preprocess.py')
    return {p: sha(ROOT/p) for p in sorted(paths)}


def source_manifest(source, sealed_source):
    from scripts.remaining_formal_support import find_source
    source = Path(source)
    old = read(source/'manifest.json')
    if old.get('signature') != digest({k:v for k,v in old.items() if k != 'signature'}):
        raise ValueError('历史训练manifest签名错误')
    if old.get('scope') != 'simulation_only' or old.get('version') not in (2, 3):
        raise ValueError('未知历史训练来源')
    if not sealed_source:
        raise ValueError('新协议须提供--sealed-source以验证原训练科学源码，不能用修正源码冒充旧实现')
    for rel, expected in old['code_sha256'].items():
        find_source(rel, expected, [Path(sealed_source)])
    for split in ('validation', 'test'):
        bank = read(source/f'scenarios_{split}.json')
        if bank != old['scenarios'][split] or not bank:
            raise ValueError('历史题库被修改或为空')
        ids, seeds = [s['scenario_id'] for s in bank], [s['seed'] for s in bank]
        if len(ids) != len(set(ids)) or len(seeds) != len(set(seeds)):
            raise ValueError('场景标识或种子重复')
    if {s['seed'] for s in old['scenarios']['validation']} & {s['seed'] for s in old['scenarios']['test']}:
        raise ValueError('验证/测试题库重叠')
    return old


def nominal_config(config):
    cfg = copy.deepcopy(config)
    if cfg.get('plume_model', 'dynamic') != 'dynamic':
        raise ValueError('正式修正协议仅支持动态羽流')
    cfg.update(goal_radius=.1, success_hold_s=0., domain_randomization=False)
    return cfg


def native_model(source, manifest, name, seed, sampling):
    from scripts.remaining_formal_support import completed_training
    task = Path(source)/name/f'seed_{seed}'
    if not completed_training(task, manifest['signature'], seed, manifest['settings']['timesteps']):
        raise ValueError(f'原训练未完成：{name}/{seed}')
    head = model_header(task/'selected_model.zip')
    state = read(task/'state.json')
    return dict(variant=name, variant_config=manifest['variants'][name], origin='historical', **head,
        training_budget_steps=state['actual_timesteps'], training_success_hold_s=manifest['base_config'].get('success_hold_s', 1.),
        training_config_verified=True, sampling=sampling,
        training_provenance=dict(manifest_signature=manifest['signature'], model_selection='original_selected_checkpoint',
            base_config=manifest['base_config'], variant=manifest['variants'][name],
            nominal_max_puff_age=float(manifest['base_config'].get('plume_overrides', {}).get('max_puff_age', 7.)),
            implementation='original_sealed_experiment_adapter'))


def primary_model(path, sampling):
    path = Path(path)
    header = model_header(path)
    metadata = path.parent.parent/'logs'/path.stem/'run_metadata.json'
    meta = read(metadata)
    if (meta.get('final_num_timesteps') != header['evaluated_checkpoint_steps'] or meta.get('base_observation_dim') != 16
            or meta.get('history_length') != 20 or meta.get('temporal_encoder') != 'gru'):
        raise ValueError('75M模型与训练元数据不一致')
    return dict(variant='main', variant_config=dict(method='main', encoder='gru', history_length=20,
        config={'domain_randomization':False}), origin='pretrained', **header, sampling=sampling,
        training_budget_steps=header['evaluated_checkpoint_steps'], training_success_hold_s=0.,
        training_config_verified=False, training_metadata_sha256=sha(metadata),
        training_provenance=dict(metadata=meta, metadata_sha256=sha(metadata),
            note='配套元数据描述最后训练阶段；不据此伪造此前各续训阶段条件'))


def comparison_spec(source, path, sealed_source, sampling):
    old = source_manifest(source, sealed_source)
    models = [primary_model(path, sampling)]
    models += [native_model(source, old, n, s, sampling) for n in BASELINES for s in old['settings']['seeds']]
    return dict(protocol=PROTOCOL, sampling=sampling, config=nominal_config(old['base_config']),
        scenarios=old['scenarios'], models=models, source=str(Path(source).resolve()),
        source_signature=old['signature'], source_manifest_sha256=sha(Path(source)/'manifest.json'),
        source_hold_s=old['base_config'].get('success_hold_s',1), source_settings=old['settings'],
        scientific_code_sha256=scientific_hashes(), labels={})


def effective_environment(config, variant):
    from dual_whisker_rl.arrival_runtime import make_env
    cfg = nominal_config(config)
    # 把变体真实配置合并到缓存配置；其训练DR标记不能开启评估DR。
    cfg.update(copy.deepcopy(variant.get('config', {})))
    cfg['plume_overrides'] = {**config.get('plume_overrides', {}), **variant.get('config', {}).get('plume_overrides', {})}
    cfg.update(domain_randomization=False, goal_radius=.1, success_hold_s=0.)
    # 除去只影响训练随机种子的字段；每局reset显式使用场景种子。
    cfg.pop('seed', None)
    env = make_env(cfg, variant, 0)
    try:
        plume = env.unwrapped.plume
        physics = {k:copy.deepcopy(v) for k,v in vars(plume).items()
                   if isinstance(v, (str, bool, int, float, tuple, list, dict, type(None)))
                   and k not in ('puffs', 'obstacles', 'fault')}
        physics = {k:v for k,v in physics.items() if not k.startswith('_') or k == '_dr_nominal'}
        for value in physics.values():
            if isinstance(value,float) and not math.isfinite(value):
                raise ValueError('标称物理参数非有限')
        adapter = {k:v for k,v in variant.items() if k not in ('config','dr_spread')}
        # dr_spread影响训练随机化，关闭评估DR后不应阻止相同冻结任务去重。
        return cfg, dict(parameters=physics, adapter=adapter,
                         observation_shape=list(env.observation_space.shape),
                         action_nvec=env.action_space.nvec.tolist() if hasattr(env.action_space,'nvec') else [int(env.action_space.n)])
    finally:
        env.close()


def make_jobs(spec, store, *, scenes=None, fault=None, label='nominal'):
    jobs = []
    for model in spec['models']:
        variant = copy.deepcopy(model['variant_config'])
        if fault:
            variant['_fault'] = copy.deepcopy(fault)
        cfg, physics = effective_environment(spec['config'], variant)
        for scene in scenes if scenes is not None else spec['scenarios']['test']:
            for repeat in range(spec['sampling']['eval_repeats']):
                # variant显示名、报告名和并行配置不参与任务键。
                key = dict(protocol=PROTOCOL, sampling=spec['sampling'], model=model['sha256'],
                    config=cfg, physics=physics, scene=scene, repeat=repeat, fault=fault or {},
                    scientific_code_sha256=spec['scientific_code_sha256'])
                signature = digest(key)
                jobs.append(dict(signature=signature, model=model, scene=scene, repeat=repeat,
                    sampling=spec['sampling'], config=cfg, effective_physics=physics, variant_config=variant,
                    fault_label=label, source=None, output=str(Path(store)/signature[:2]/f'{signature}.json')))
    return jobs


def seal(path, payload):
    signed = {**payload, 'signature':digest(payload)}
    path = Path(path)
    if path.exists():
        if read(path) != signed:
            raise ValueError(f'输出封存不匹配，必须使用新目录：{path}')
    else:
        atomic_json(path, signed)


def view_rows(jobs):
    rows = []
    for job in jobs:
        if not cached(job):
            raise ValueError(f'场景未完成或缓存损坏：{job["output"]}')
        rows.append({**read(job['output']), 'variant':job['model']['variant'],
                     'method':job['model']['variant'], 'fault':job['fault_label']})
    return rows
