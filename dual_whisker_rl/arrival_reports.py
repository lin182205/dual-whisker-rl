"""冻结模型集合的首次到源主对比报告，不进行算法级训练重复推断。"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path

from dual_whisker_rl.arrival_protocol import BASELINES, LABELS, PROTOCOL, atomic_json, read, sha

METRICS = ('success', 'odor_hit_ratio', 'whisker_only_reacquisition_events',
           'reacquisition_events', 'penalized_time_s', 'final_distance', 'path_length')


def write_table(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys((fields or []) + [k for r in rows for k in r]))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in r.items()})


def make_report(spec, jobs, output, draws=2000):
    import numpy as np
    from dual_whisker_rl.arrival_protocol import cached
    names = tuple(dict.fromkeys(m['variant'] for m in spec['models']))
    labels = dict(LABELS)
    primary_millions = spec['models'][0]['evaluated_checkpoint_steps']/1e6
    labels['main'] = f'主动策略{primary_millions:.2f}M（单模型）'
    labels.update(spec.get('labels', {}))
    directory = Path(output) / 'reports' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    directory.mkdir(parents=True)
    rows, inputs = [], []
    for job in jobs:
        if cached(job):
            rows.append({**read(job['output']), 'variant':job['model']['variant']})
            path = Path(job['output']).resolve()
            try:
                source_path = str(path.relative_to(Path(output).resolve()))
            except ValueError:
                source_path = str(path)
            inputs.append(dict(path=source_path, sha256=sha(path)))
    by_method = defaultdict(list)
    for r in rows:
        by_method[r['variant']].append(r)
    seed_counts = {n: sum(m['variant'] == n for m in spec['models']) for n in names}
    repeats = spec.get('sampling', {}).get('eval_repeats', 1)
    scenes = [s['scenario_id'] for s in spec['scenarios']['test']]
    coverage, summaries, seeds, vectors = [], [], [], {}
    rng = np.random.default_rng(73)
    for name in names:
        items = by_method[name]
        expected = len(scenes)*seed_counts[name]*repeats
        coverage.append(dict(variant=name, label=labels[name], actual=len(items), expected=expected,
                             complete=len(items) == expected, training_models=seed_counts[name]))
        for model in (m for m in spec['models'] if m['variant'] == name):
            subset = [r for r in items if r['training_seed'] == model['training_seed']]
            for metric in METRICS:
                seeds.append(dict(variant=name, training_seed=model['training_seed'], metric=metric,
                    mean=float(np.mean([float(r[metric]) for r in subset])) if subset else None,
                    episodes=len(subset), training_budget_steps=model['training_budget_steps'],
                    evaluated_checkpoint_steps=model['evaluated_checkpoint_steps'],
                    training_config=model['variant_config'], training_success_hold_s=model['training_success_hold_s'],
                    model_sha256=model['sha256']))
        per_scene = defaultdict(list)
        for r in items:
            per_scene[r['scenario_id']].append(r)
        for metric in METRICS:
            vector = {sid: float(np.mean([float(r[metric]) for r in group]))
                      for sid, group in per_scene.items() if len(group) == seed_counts[name]*repeats}
            vectors[name, metric] = vector
            values = np.array(list(vector.values()))
            interval = [None, None]
            if len(values):
                bootstrap = values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1)
                interval = np.quantile(bootstrap, [.025, .975]).tolist()
            summaries.append(dict(variant=name, label=labels[name], metric=metric,
                mean=float(values.mean()) if len(values) else None,
                ci95_low=interval[0], ci95_high=interval[1], complete_scenarios=len(values),
                available_episodes=len(items), expected_episodes=expected, training_models=seed_counts[name],
                uncertainty_scope='conditional_on_frozen_models'))
    differences = []
    for name in names[1:]:
        for metric in METRICS:
            left, right = vectors['main', metric], vectors[name, metric]
            common = sorted(set(left) & set(right))
            values = np.array([left[s]-right[s] for s in common])
            ci = [None, None]
            if len(values):
                ci = np.quantile(values[rng.integers(len(values), size=(draws, len(values)))].mean(axis=1), [.025, .975]).tolist()
            differences.append(dict(reference=name, metric=metric,
                difference=float(values.mean()) if len(values) else None, ci95_low=ci[0], ci95_high=ci[1],
                common_scenarios=len(common), expected_scenarios=len(scenes),
                direction='main_minus_reference', uncertainty_scope='conditional_on_frozen_models',
                training_seed_p_value=None))
    tables = directory / 'tables'
    write_table(tables/'episodes.csv', rows, fields=list(rows[0]) if rows else ['variant', 'scenario_id'])
    write_table(tables/'coverage.csv', coverage)
    write_table(tables/'method_metrics.csv', summaries)
    write_table(tables/'seed_metrics.csv', seeds)
    write_table(tables/'paired_differences.csv', differences)
    for field in ('distance_bin', 'crosswind_bin', 'heading_quadrant', 'source_strength', 'nominal_wind_speed'):
        groups = defaultdict(list)
        for r in rows:
            groups[r['variant'], r[field]].append(r)
        strata = [dict(variant=n, stratum=str(value), episodes=len(group),
                       **{metric: float(np.mean([float(r[metric]) for r in group])) for metric in METRICS})
                  for (n, value), group in sorted(groups.items())]
        write_table(tables/f'strata_{field}.csv', strata, fields=['variant', 'stratum', 'episodes', *METRICS])
    complete = all(r['complete'] for r in coverage)
    status = 'complete' if complete else 'partial'
    figures = []
    if complete:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
            'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False,
            'pdf.fonttype': 42, 'svg.fonttype': 'none'})
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
        fig.subplots_adjust(left=.27, right=.97, bottom=.25, top=.86, wspace=.42)
        default_english = dict(zip(BASELINES, ('Fixed', 'Periodic', 'Single frame', 'History MLP', 'No DR 2M')))
        english = [f'Active {primary_millions:.2f}M (1 model)'] + [
            f"{spec.get('english_labels', {}).get(name, default_english.get(name, name))} ({seed_counts[name]} models)"
            for name in names[1:]]
        for ax, metric, title in zip(axes, ('success', 'odor_hit_ratio'), ('Success (%)', 'Odor hit ratio (%)')):
            values = [next(r for r in summaries if r['variant'] == n and r['metric'] == metric) for n in names]
            for i, r in enumerate(values):
                color = '#217A68' if i == 0 else '#7C8792'
                ax.hlines(i, r['ci95_low']*100, r['ci95_high']*100, color=color, lw=1.2)
                ax.plot(r['mean']*100, i, 'o', color=color, ms=4)
            ax.set_yticks(range(len(names)), english if ax is axes[0] else ['']*len(names))
            ax.set_ylim(len(names)-.4, -.6)
            ax.set_xlabel(title)
            upper = max(max(r['ci95_high']*100 for r in values)*1.15, 1.)
            ax.set_xlim(-.03*upper, upper)
            ax.grid(axis='x', color='#ECEFF1', lw=.6)
        fig.suptitle('Frozen-model comparison | first arrival within 0.1 m', fontsize=10)
        fig.text(.27, .10, '95% scenario-bootstrap intervals, conditional on the available frozen models.', fontsize=7)
        fig.text(.27, .045, 'Training conditions and budgets differ; intervals do not establish algorithm-level superiority.', fontsize=7)
        d = directory/'figures'; d.mkdir()
        fig.savefig(d/'performance.pdf')
        fig.savefig(d/'performance.svg')
        fig.savefig(d/'performance.png', dpi=600)
        plt.close(fig)
        figures.append('figures/performance.png')
    text = ['# 首次到源主对比', '', f'状态：{status}。成功条件：中心进入0.1米半径即结束；评估关闭额外域随机化。', '',
        f"主方法为{spec['models'][0]['evaluated_checkpoint_steps']:,}步单模型；对照是已完成训练的历史模型。训练目标、域随机化和预算存在差异，本报告比较冻结策略在统一测试条件下的表现，不将收益全部归因于算法。", '',
        '| 方法 | 已完成/应完成 | 模型数 | 成功率 | 气味命中率 |', '|---|---:|---:|---:|---:|']
    for c in coverage:
        vals = [next(r for r in summaries if r['variant'] == c['variant'] and r['metric'] == metric)['mean']
                for metric in ('success', 'odor_hit_ratio')]
        display = ['缺失' if value is None else f'{value:.2%}' for value in vals]
        text.append(f"| {c['label']} | {c['actual']}/{c['expected']} | {c['training_models']} | {display[0]} | {display[1]} |")
    text += ['', '所有差值为主方法减对照；每个场景的对照值先对真实训练seed取均值，再按共享场景配对。区间仅描述现有模型集合在场景上的不确定性，不提供训练seed级p值。', '',
             '缺项不填零；完整主图仅在全部模型×场景完成时生成。逐场景、逐seed、覆盖率、模型训练信息和五类分层表保存在tables。']
    text += ['', *[f'![主对比]({f})' for f in figures]]
    if spec.get('sampling'):
        text[3:3] = [f"评估协议：{spec['protocol']}，随机采样动作；eval_seed={spec['sampling']['eval_seed']}，每模型/场景{repeats}次。", '',
            '羽流继承移动环境完整解析参数；当前标称气团寿命约45.83秒（以逐局effective_physics为准）。',
            '旧10.2%为7秒气团、确定性动作的历史结果，不与本轮直接合并。训练物理与预算差异仍须披露。', '']
    (directory/'report.md').write_text('\n'.join(text)+'\n', encoding='utf-8')
    atomic_json(directory/'report_manifest.json', dict(protocol=spec.get('protocol',PROTOCOL), sampling=spec.get('sampling'), status=status, inputs=inputs,
        source_manifest_sha256=spec['source_manifest_sha256'], config=spec['config'], models=spec['models'],
        bootstrap_draws=draws, uncertainty_scope='conditional_on_frozen_models', figures=figures))
    atomic_json(Path(output)/'latest_report.json', dict(status=status, report=str(directory/'report.md')))
    return directory/'report.md'
