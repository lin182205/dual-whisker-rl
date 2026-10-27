"""按真实日志和逐局结果导出E3–E6论文图表；不补造缺失证据。"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from dual_whisker_rl.e4_experiments import sha256_file
from dual_whisker_rl.experiment_protocol import paired_interval, holm, write_csv, write_json
from dual_whisker_rl.paths import portable_path, project_path

METRICS = ("odor_hit_ratio", "success", "whisker_only_reacquisition_events", "penalized_time_s")
LABELS = {"main": "Proposed", "fixed": "Fixed whiskers", "periodic": "Periodic scan",
          "single_frame": "Single frame", "history_mlp": "History MLP", "no_dr": "Without DR",
          "b0_random": "Random rule", "b1_reactive": "Reactive rule", "b2_fixed": "Fixed whiskers",
          "b3_scan": "Periodic scan", "b4_single_frame": "Single frame", "b5_history_mlp": "History MLP",
          "b6_left": "Left sensor", "b6_right": "Right sensor"}
PALETTE = ("#276A9B", "#707070", "#B17C43", "#6D7FA3", "#668B83", "#A57587")
E5_GROUPS = ("history", "features", "rewards", "geometry", "sensor", "randomization")


def label(name):
    return LABELS.get(name, name.replace("_", " "))


def table(out, name, rows, fields):
    write_csv(out / "tables" / f"{name}.csv", rows, fields=list(dict.fromkeys(fields + [k for r in rows for k in r])))


def read_rows(paths, sources):
    rows = []
    for path in sorted(paths):
        sources.add(path)
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def native_rows(root, names, seeds, sources, area="episodes", fault="nominal"):
    return read_rows((p for n in names for s in seeds
        for p in (root / n / f"seed_{s}" / area / fault).glob("*.json")), sources)


def plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "legend.fontsize": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
        "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.7,
        "legend.frameon": False, "savefig.facecolor": "white"})
    return plt


def save_figure(fig, out, name, caption, figures):
    """固定180毫米宽、7pt文字、可编辑向量与600dpi预览；图注单独保留。"""
    directory = out / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    fig.savefig(directory / f"{name}.png", dpi=600, bbox_inches="tight", pad_inches=.12)
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", pad_inches=.12)
    fig.savefig(directory / f"{name}.svg", bbox_inches="tight", pad_inches=.12)
    (directory / f"{name}.md").write_text(caption + "\n", encoding="utf-8")
    figures.append(name)
    plotting().close(fig)


def curve_plot(curves, metric, out, figures):
    rows = [r for r in curves if r["metric"] == metric]
    if not rows:
        return
    plt = plotting()
    fig, ax = plt.subplots(figsize=(180 / 25.4, 3.3))
    fig.subplots_adjust(left=.12, right=.69, bottom=.18, top=.87)
    for i, name in enumerate(dict.fromkeys(r["variant"] for r in rows)):
        items = sorted((r for r in rows if r["variant"] == name), key=lambda r: r["timesteps"])
        x, y = [r["timesteps"] for r in items], [r["mean"] for r in items]
        n = min(r["training_seeds"] for r in items)
        ax.plot(x, y, marker="o", markersize=2.5, linewidth=1.3 if name == "main" else .9,
                color=PALETTE[i % len(PALETTE)], label=f"{label(name)} (n={n})")
        if all(r["ci95_low"] is not None for r in items):
            ax.fill_between(x, [r["ci95_low"] for r in items], [r["ci95_high"] for r in items],
                            alpha=.16, color=PALETTE[i % len(PALETTE)], linewidth=0)
    ax.set(xlabel="Environment steps", ylabel="Success rate" if metric == "success_rate" else "Odor hit ratio",
           title="E3 | Learning across independent training seeds")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    save_figure(fig, out, f"e3_{metric}",
        "Mean across training seeds; band: 95% seed-bootstrap CI when n>=2. "
        "No algorithm-level CI for n=1. Only checkpoint steps shared by all available seeds are plotted; "
        "all original points are retained in the source table. Smoke runs validate the pipeline only.", figures)


def comparison_plot(summary, metric, out, figures):
    items = [r for r in summary if r["metric"] == metric and r["mean"] is not None]
    if not items:
        return
    plt = plotting()
    fig, ax = plt.subplots(figsize=(180 / 25.4, max(2.6, .27 * len(items) + 1.1)))
    fig.subplots_adjust(left=.34, right=.95, bottom=.22, top=.85)
    for i, r in enumerate(items):
        color = PALETTE[0] if r["variant"] == "main" else PALETTE[1]
        ax.plot(r["mean"], i, "o", color=color, markersize=4)
        if r["ci95_low"] is not None:
            ax.hlines(i, r["ci95_low"], r["ci95_high"], color=color, linewidth=1)
    ax.set_yticks(range(len(items)), [f"{label(r['variant'])} ({'rule' if r['unit']=='rule' else 'n='+str(r['training_seeds'])})" for r in items])
    ax.set_ylim(len(items) - .5, -.5)
    ax.set(xlabel="Success rate" if metric == "success" else "Odor hit ratio", title="E4 | Method performance on shared scenarios")
    save_figure(fig, out, f"e4_{metric}", "Points: equally weighted seed means; lines: 95% bootstrap CI across "
        "training seeds when n>=2. Rules have no training seed and no algorithm-level interval. "
        "Coverage is reported in tables. A single supplied model is a reevaluation, not a method comparison.", figures)


def effect_plot(rows, group, out, figures):
    items = [r for r in rows if r["group"] == group and r["metric"] == "odor_hit_ratio" and r.get("difference") is not None]
    if not items:
        return
    plt = plotting()
    fig, ax = plt.subplots(figsize=(180 / 25.4, max(2.6, .28 * len(items) + 1)))
    fig.subplots_adjust(left=.36, right=.95, bottom=.23, top=.84)
    ax.axvline(0, color="#B0B0B0", linewidth=.8, linestyle="--")
    for i, r in enumerate(items):
        ax.plot(r["difference"], i, "o", color=PALETTE[0], markersize=4)
        if r.get("ci95_low") is not None:
            ax.hlines(i, r["ci95_low"], r["ci95_high"], color=PALETTE[0], linewidth=1)
    ax.set_yticks(range(len(items)), [label(r["variant"]) for r in items])
    ax.set_ylim(len(items) - .5, -.5)
    ax.set(xlabel="Odor hit ratio: variant minus reference", title=f"E5 | {group.capitalize()} sensitivity")
    save_figure(fig, out, f"e5_{group}", "Paired differences on shared scenarios and matching training seeds. "
        "Intervals: crossed seed/scenario bootstrap when at least two training seeds are available. "
        "No algorithm-level interval for n=1. Positive values favor the variant. Reference and all "
        "available/missing pairs are recorded in the table. No reward-value comparison across reward designs.", figures)


def fault_plot(rows, metric, out, figures):
    items = [r for r in rows if r["metric"] == metric and r.get("difference") is not None]
    names = list(dict.fromkeys(r["variant"] for r in items))
    faults = list(dict.fromkeys(r["fault"] for r in items))
    if not names or not faults:
        return
    plt = plotting()
    for start in range(0, len(faults), 6):
        subset = faults[start:start + 6]
        matrix = np.full((len(names), len(subset)), np.nan)
        for r in items:
            if r["fault"] in subset:
                matrix[names.index(r["variant"]), subset.index(r["fault"])] = r["difference"]
        finite = matrix[np.isfinite(matrix)]
        scale = max(float(np.max(np.abs(finite))), .01)
        fig, ax = plt.subplots(figsize=(180 / 25.4, max(2.7, .35 * len(names) + 1.7)))
        fig.subplots_adjust(left=.24, right=.83, bottom=.26, top=.83)
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("#EEEEEE")
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, vmin=-scale, vmax=scale, aspect="auto")
        ax.set_yticks(range(len(names)), [label(n) for n in names])
        ax.set_xticks(range(len(subset)), subset, rotation=30, ha="right", rotation_mode="anchor")
        ax.set_title(f"E6 | Change in {'success rate' if metric == 'success' else 'odor hit ratio'}")
        fig.colorbar(im, ax=ax, fraction=.06, pad=.04, label="Fault minus nominal")
        save_figure(fig, out, f"e6_{metric}_{start // 6 + 1}", "Paired point estimates: fault minus nominal "
            "on identical scenarios, with policies frozen. Red indicates a positive change, blue a negative "
            "change; gray denotes missing pairs. Crossed bootstrap intervals, conditional single-model "
            "intervals, seed counts, pair counts and nominal ratios are provided in the source table. "
            "Not every software perturbation is strictly outside the training distribution.", figures)


def coverage_rows(names, seeds, rows, expected, experiment):
    return [dict(experiment=experiment, variant=n, training_seeds=len({r["training_seed"] for r in rows if r["variant"] == n}),
        actual_episodes=sum(r["variant"] == n for r in rows), expected_episodes=expected * len(seeds),
        complete=sum(r["variant"] == n for r in rows) == expected * len(seeds)) for n in names]


def aggregate_rows(rows, draws):
    values, seed_rows = [], []
    # Bootstrap仅对真实训练seed均值重采样，不生成仿造观测。
    rng = np.random.default_rng(73)
    for name in dict.fromkeys(r["variant"] for r in rows):
        group = [r for r in rows if r["variant"] == name]
        rule = name in ("b0_random", "b1_reactive")
        for metric in METRICS:
            means = []
            for seed in sorted({r["training_seed"] for r in group}):
                available = [float(r[metric]) for r in group if r["training_seed"] == seed and r.get(metric) is not None]
                if available:
                    value = float(np.mean(available))
                    means.append(value)
                    seed_rows.append(dict(variant=name, training_seed=None if rule else seed,
                        unit="rule" if rule else "training_seed", metric=metric, mean=value, available=len(available)))
            ci = [None, None]
            if len(means) >= 2 and not rule:
                matrix = np.asarray(means)
                ci = [float(x) for x in np.quantile(matrix[rng.integers(len(matrix), size=(draws, len(matrix)))].mean(axis=1), [.025, .975])]
            values.append(dict(variant=name, metric=metric, mean=float(np.mean(means)) if means else None,
                ci95_low=ci[0], ci95_high=ci[1], training_seeds=0 if rule else len(means),
                unit="rule" if rule else "training_seed", episodes=len(group), missing=sum(r.get(metric) is None for r in group)))
    return values, seed_rows


def paired_row(candidate, reference, metric, draws):
    left = {(int(r["training_seed"]), str(r["scenario_id"])): float(r[metric]) for r in candidate if r.get(metric) is not None}
    right = {(int(r["training_seed"]), str(r["scenario_id"])): float(r[metric]) for r in reference if r.get(metric) is not None}
    result = paired_interval(left, right, draws=draws)
    interval = result.pop("ci95", [None, None])
    result.update(ci95_low=interval[0], ci95_high=interval[1],
                  unmatched_candidate=len(set(left)-set(right)), unmatched_reference=len(set(right)-set(left)))
    return result


def export_report(experiment, root, out, args, names, spec, profile, manifest, *, legacy_root=None, error=None):
    """每次实验至少产出状态CSV和中文说明；缺失值保持缺失。"""
    out.mkdir(parents=True, exist_ok=True)
    sources, figures, tables, notes = set(), [], [], []
    source = manifest.get("external_model")
    corrected = manifest.get('evaluation_protocol') == 'stochastic-arrival-v2'
    if corrected:
        notes += ['修正评估协议：首次进入0.1米、额外DR关闭、逐场景独立随机采样动作。',
                  '训练保持原封存；历史模型可能在7秒气团下训练，测试使用修正后的移动羽流。训练预算与目标差异须一并解释。',
                  '重复采样先按模型/场景取均值，不作为额外独立训练seed。']
    status = "complete"
    if (root / "manifest.json").exists():
        sources.add(root / "manifest.json")
    if error is not None:
        status, notes = "failed", [str(error)]
    elif experiment == "e3":
        from scripts.analyze_experiment_learning import learning
        learning_root = project_path(args.learning_dir) if args.learning_dir else root
        if source and not args.learning_dir:
            status, notes = "missing_training_history", ["已有模型没有配套验证日志，不能从权重重建学习曲线或训练稳定性。"]
        else:
            curves, seed_rows = learning(learning_root, names, args.bootstrap, seeds=profile["seeds"])
            raw = []
            for n in names:
                for s in profile["seeds"]:
                    p = learning_root / n / f"seed_{s}" / "validation_history.json"
                    if p.exists():
                        for r in read_rows([p], sources)[0]:
                            raw.append(dict(variant=n, training_seed=s, **{k:r.get(k) for k in
                                ("timesteps", "success_rate", "odor_hit_ratio", "mean_penalized_time_s", "whisker_only_reacquisition_events")}))
            table(out, "learning_curves", curves, ["variant", "timesteps", "metric", "mean", "ci95_low", "ci95_high", "training_seeds"])
            table(out, "training_seed_metrics", seed_rows, ["variant", "training_seed", "observed_auc", "evaluation_count"])
            table(out, "validation_points", raw, ["variant", "training_seed", "timesteps", "success_rate"])
            counts = [dict(variant=n, actual_seeds=len({r["training_seed"] for r in seed_rows if r["variant"]==n}),
                expected_seeds=len(profile["seeds"])) for n in names]
            table(out, "coverage", counts, ["variant", "actual_seeds", "expected_seeds"])
            status = "complete" if counts and all(r["actual_seeds"] == r["expected_seeds"] for r in counts) else "partial_or_missing_history"
            curve_plot(curves, "success_rate", out, figures)
            curve_plot(curves, "odor_hit_ratio", out, figures)
            notes += ["AUC仅覆盖实际验证点区间；单seed无算法级置信区间；曲线只聚合共同checkpoint。"]
            if source:
                notes.append("外部日志由用户显式提供，尚未自动证明其与该模型同源。")
    elif experiment == "e4":
        if legacy_root is not None:
            legacy_manifest = legacy_root / "manifest.json"
            if legacy_manifest.exists():
                sources.add(legacy_manifest)
                sealed = json.loads(legacy_manifest.read_text(encoding="utf-8"))
                selected = args.e4_methods.split(",") if args.e4_methods else [m["method"] for m in sealed["methods"]]
                paths = [p for n in selected for p in (legacy_root/n).glob("seed_*/episodes/test/*.json")]
                rows = read_rows(paths, sources)
                for r in rows:
                    r["variant"] = r["method"]
                expected = int(sealed["scenario_counts"]["test"])
                coverage = [dict(experiment="e4",variant=n,actual_episodes=sum(r["variant"]==n for r in rows),
                    expected_episodes=expected*len(sealed["method_seeds"][n])) for n in selected]
                notes.append('E4按新统一随机动作协议重评；训练预算和题库仍保留原安排。' if corrected else "原E4：瞬时进入气源半径；其训练预算和场景协议独立，不能与持续1秒判据混合。")
            else:
                rows, coverage = [], []
        else:
            rows = native_rows(root, names, profile["seeds"], sources)
            expected = len(json.loads((root/"scenarios_test.json").read_text(encoding="utf-8")))
            coverage = coverage_rows(names,profile["seeds"],rows,expected,"e4")
            notes.append('已有模型按新首次到源协议复评；单模型结果不构成完整方法对比。' if corrected else "已有模型采用当前持续1秒成功判据复评；单模型结果不构成完整方法对比。")
        summary, per_seed = aggregate_rows(rows, args.bootstrap)
        table(out,"episodes",rows,["variant","training_seed","scenario_id",*METRICS])
        table(out,"method_metrics",summary,["variant","metric","mean","ci95_low","ci95_high","training_seeds","episodes"])
        table(out,"seed_metrics",per_seed,["variant","training_seed","metric","mean","available"])
        table(out,"coverage",coverage,["variant","actual_episodes","expected_episodes"])
        status = "complete" if coverage and all(r["actual_episodes"]==r["expected_episodes"] for r in coverage) else "partial_or_missing_episodes"
        comparison_plot(summary,"odor_hit_ratio",out,figures)
        comparison_plot(summary,"success",out,figures)
    elif experiment == "e5":
        if source:
            status, notes = "missing_retrained_controls", ["一个已有模型不足以形成训练消融；需要各配置分别训练的模型。没有生成推理删特征冒充训练消融的数值。"]
        else:
            rows = native_rows(root,names,profile["seeds"],sources)
            effects = []
            for group in E5_GROUPS:
                members = [n for n in spec["groups"][group] if n in names]
                reference = "nominal_gru" if group == "sensor" else "main"
                for n in members:
                    if n == reference:
                        continue
                    for metric in METRICS:
                        effect = paired_row([r for r in rows if r["variant"]==n], [r for r in rows if r["variant"]==reference], metric,args.bootstrap)
                        if effect.get("training_seeds",0) < 2:
                            effect["ci95_low"],effect["ci95_high"] = None,None
                        effects.append(dict(group=group,variant=n,reference=reference,metric=metric,**effect))
            for group in E5_GROUPS:
                for metric in METRICS:
                    family=[r for r in effects if r["group"]==group and r["metric"]==metric and "seed_sign_permutation_p" in r]
                    for r,p in zip(family,holm([r["seed_sign_permutation_p"] for r in family])):
                        r["holm_p_within_group_metric"] = p
                effect_plot(effects,group,out,figures)
            expected=len(json.loads((root/"scenarios_test.json").read_text(encoding="utf-8")))
            coverage=coverage_rows(names,profile["seeds"],rows,expected,"e5")
            table(out,"episodes",rows,["variant","training_seed","scenario_id",*METRICS])
            table(out,"paired_effects",effects,["group","variant","reference","metric","status","difference","ci95_low","ci95_high"])
            table(out,"coverage",coverage,["variant","actual_episodes","expected_episodes","complete"])
            status="complete" if coverage and all(r["complete"] for r in coverage) and effects and all(r["status"]=="paired" for r in effects) else "partial_or_missing_controls"
            notes += ["差值为配置减参考；sensor组参考nominal_gru，其余组参考main。",
                "Holm只在每组每终点内校正；跨组/跨终点主要检验族仍须预先指定。5seed双侧符号置换最小p=0.0625。"]
    elif experiment == "e6":
        effects, all_rows, coverage = [], [], []
        faults=args.faults.split(",")
        expected=min(args.limit if args.limit is not None else 200, len(json.loads((root/"scenarios_test.json").read_text(encoding="utf-8"))))
        for n in names:
            nominal=native_rows(root,[n],profile["seeds"],sources,"robustness","nominal")
            all_rows.extend(nominal)
            for fault in faults:
                treated=nominal if fault=="nominal" else native_rows(root,[n],profile["seeds"],sources,"robustness",fault)
                if fault!="nominal": all_rows.extend(treated)
                coverage.append(dict(variant=n,fault=fault,actual_episodes=len(treated),expected_episodes=expected*len(profile["seeds"])))
                if fault=="nominal": continue
                for metric in METRICS:
                    effect=paired_row(treated,nominal,metric,args.bootstrap)
                    keys={(r["training_seed"],r["scenario_id"]) for r in treated if r.get(metric) is not None}&{(r["training_seed"],r["scenario_id"]) for r in nominal if r.get(metric) is not None}
                    base=[float(r[metric]) for r in nominal if (r["training_seed"],r["scenario_id"]) in keys and r.get(metric) is not None]
                    changed=[float(r[metric]) for r in treated if (r["training_seed"],r["scenario_id"]) in keys and r.get(metric) is not None]
                    a,b=(float(np.mean(v)) if v else None for v in (base,changed))
                    effects.append(dict(variant=n,fault=fault,metric=metric,nominal=a,perturbed=b,
                        ratio=b/a if a is not None and a>1e-8 and b is not None else None,
                        uncertainty_scope="conditional_on_one_model" if effect.get("training_seeds",0)==1 else "crossed_training_seed_scenario",**effect))
        table(out,"episodes",all_rows,["variant","training_seed","fault","scenario_id",*METRICS])
        table(out,"degradation",effects,["variant","fault","metric","nominal","perturbed","difference","ratio","ci95_low","ci95_high","uncertainty_scope"])
        table(out,"coverage",coverage,["variant","fault","actual_episodes","expected_episodes"])
        status="complete" if coverage and all(r["actual_episodes"]==r["expected_episodes"] for r in coverage) and effects and all(r["status"]=="paired" for r in effects) else "partial_or_missing_faults"
        fault_plot(effects,"odor_hit_ratio",out,figures)
        fault_plot(effects,"success",out,figures)
        notes += ["模型冻结；差值为扰动减nominal；单模型区间只描述该模型的场景差异，不能作为算法级CI。基线为零时比值留空。"]
    notes.append("smoke仅检查连通性，不支持论文效果结论。" if args.profile=="smoke" else "正式结论须核对独立训练重复、覆盖率、协议和事先指定的比较。")
    table(out,"experiment_status",[dict(experiment=experiment,status=status,profile=args.profile,
        model_origin="pretrained" if source else "experiment_training",note="；".join(notes))],
        ["experiment","status","profile","model_origin","note"])
    tables=[p.relative_to(out).as_posix() for p in sorted((out/"tables").glob("*.csv"))]
    text=[f"# {experiment.upper()} 实验结果", "", f"状态：{status}；规模：{args.profile}。", "", *[f"- {n}" for n in notes],
          "", "## 表格", "", *[f"- [{p}]({p})" for p in tables], "", "## 图片", ""]
    text += [f"- [{n}](figures/{n}.png)：PNG预览、PDF/SVG向量版及同名图注。" for n in figures] or ["没有足够的真实数据绘图，缺项已写入状态表。"]
    (out/"report.md").write_text("\n".join(text)+"\n",encoding="utf-8")
    write_json(out/"report_manifest.json",dict(experiment=experiment,status=status,profile=args.profile,
        experiment_signature=manifest["signature"],evaluation_protocol=manifest.get('evaluation_protocol'),external_model=source,backend="python_matplotlib",
        inputs=[dict(path=portable_path(p),sha256=sha256_file(p)) for p in sorted(sources)],
        figures=figures,tables=tables,bootstrap_draws=args.bootstrap,notes=notes,
        figure_contract=dict(question={"e3":"训练学习表现是否稳定？","e4":"方法在共享场景上有何差异？",
            "e5":"独立重训改变单一设计后有何配对差异？","e6":"冻结模型在扰动下退化多少？"}[experiment],
            conclusion="由真实结果判断，不预设主动方法优于基线",archetype="quantitative",width_mm=180,
            formats=["png_600dpi","pdf","svg"],font_size_pt=7)))
    return dict(experiment=experiment,status=status,report=portable_path(out/"report.md"),figures=len(figures),tables=len(tables))
