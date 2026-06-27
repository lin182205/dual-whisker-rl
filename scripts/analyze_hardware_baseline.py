"""硬件 MQ-3 基线分析与传感器预处理验证脚本。

该脚本属于项目短期计划的阶段一（传感器数据质量验证）。它不训练 PPO，
也不比较复杂策略，只回答一个基础问题：

    当前真实 MQ-3 原始 ADC 经过 baseline subtraction、EMA、trend、
    estimated_input 之后，能否把洁净空气漂移压到可接受范围，
    同时保留真实气体刺激响应？

输入：
    results/hardware/live_plot.csv （或 smoke_test.csv / sector_scan.csv）
    至少包含字段：t_rel_s 或 t_pc_s、left_adc、right_adc。

输出：
    results/hardware/baseline_analysis.json   左右传感器统计指标 + 预处理参数
    results/hardware/baseline_preprocess.png  raw/baseline/signal/smooth/trend 曲线
    results/hardware/baseline_summary.md       一句话结论：当前参数是否可用

用法示例：

    python scripts\\analyze_hardware_baseline.py --csv-path results\\hardware\\live_plot.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.hardware.sensor_preprocess import DualGasPreprocessor
from dual_whisker_rl.hardware.sensor_preprocess import SensorPreprocessConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=ROOT / "results" / "hardware" / "live_plot.csv",
    )
    # 输出路径默认 None，main() 中根据 --label 生成，避免洁净段/气体段互相覆盖。
    parser.add_argument("--json-path", type=Path, default=None)
    parser.add_argument("--figure-path", type=Path, default=None, help="分面板图(上一版风格)路径。")
    parser.add_argument("--overlay-path", type=Path, default=None, help="双 Y 轴叠加对比图路径。")
    parser.add_argument("--summary-path", type=Path, default=None)
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="输出文件名后缀，例如 clean / gas。用于区分不同分析段。",
    )
    # 分析窗口：预处理始终在完整连续数据流上运行（保持有状态累积），
    # 这里只框出一个时间区间用于计算分段统计、判定和绘图。
    parser.add_argument("--t-start", type=float, default=None, help="分析窗口起始时间(秒)。")
    parser.add_argument("--t-end", type=float, default=None, help="分析窗口结束时间(秒)。")
    parser.add_argument(
        "--segment-kind",
        choices=["clean", "gas", "mixed"],
        default="mixed",
        help="该段性质，决定判定逻辑：clean 期望特征接近 0，gas 期望有明显响应。",
    )
    parser.add_argument(
        "--window-s",
        type=float,
        default=10.0,
        help="估计基线漂移时，前后窗口的时长（秒）。",
    )
    # 预处理参数：默认沿用 SensorPreprocessConfig，允许命令行覆盖以便调参。
    parser.add_argument("--dt-s", type=float, default=None, help="不指定则从时间戳自动推算。")
    parser.add_argument("--baseline-tau-s", type=float, default=None)
    parser.add_argument("--smooth-tau-s", type=float, default=None)
    parser.add_argument("--response-tau-s", type=float, default=None)
    parser.add_argument("--trend-window", type=int, default=None)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument(
        "--freeze-baseline",
        action="store_true",
        help="冻结基线更新。用于气体刺激段，避免真实响应被 baseline 吃掉。",
    )
    return parser.parse_args()


def load_csv(path: Path) -> dict[str, np.ndarray]:
    """读取硬件 CSV，返回时间和左右 ADC 数组。"""
    import csv

    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if not rows:
        raise ValueError(f"CSV 为空: {path}")

    fields = rows[0].keys()
    if "left_adc" not in fields or "right_adc" not in fields:
        raise ValueError(f"CSV 缺少 left_adc/right_adc 字段: {path}")

    left_adc = np.array([float(r["left_adc"]) for r in rows])
    right_adc = np.array([float(r["right_adc"]) for r in rows])

    # 时间优先用相对时间 t_rel_s；否则用绝对时间 t_pc_s 减去首值；都没有则用步序号。
    if "t_rel_s" in fields:
        t = np.array([float(r["t_rel_s"]) for r in rows])
    elif "t_pc_s" in fields:
        raw_t = np.array([float(r["t_pc_s"]) for r in rows])
        t = raw_t - raw_t[0]
    else:
        t = np.arange(len(rows), dtype=float)

    return {"t": t, "left_adc": left_adc, "right_adc": right_adc}


def estimate_dt(t: np.ndarray) -> float:
    """从时间戳推算采样间隔，取相邻差的中位数，避免个别异常间隔干扰。"""
    if len(t) < 2:
        return 0.8
    diffs = np.diff(t)
    diffs = diffs[diffs > 0.0]
    if diffs.size == 0:
        return 0.8
    return float(np.median(diffs))


def channel_stats(t: np.ndarray, adc: np.ndarray, window_s: float) -> dict[str, float]:
    """计算单通道 raw ADC 的统计量和漂移指标。"""
    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0

    # 线性拟合估计整体漂移斜率（ADC/s）。
    if duration > 0.0:
        slope, _ = np.polyfit(t, adc, 1)
        detrended = adc - np.polyval([slope, np.mean(adc) - slope * np.mean(t)], t)
    else:
        slope = 0.0
        detrended = adc - np.mean(adc)

    # 前窗口与后窗口均值差，作为基线漂移的直观估计。
    if duration > 0.0:
        first_mask = t <= (t[0] + window_s)
        last_mask = t >= (t[-1] - window_s)
        first_mean = float(np.mean(adc[first_mask]))
        last_mean = float(np.mean(adc[last_mask]))
    else:
        first_mean = last_mean = float(np.mean(adc))

    return {
        "mean_adc": float(np.mean(adc)),
        "std_adc": float(np.std(adc)),
        "min_adc": float(np.min(adc)),
        "max_adc": float(np.max(adc)),
        "drift_slope_adc_per_s": float(slope),
        "drift_slope_adc_per_min": float(slope * 60.0),
        "first_window_mean": first_mean,
        "last_window_mean": last_mean,
        "first_last_delta": float(last_mean - first_mean),
        "noise_std_after_detrend": float(np.std(detrended)),
        "duration_s": duration,
    }


def run_preprocess(
    data: dict[str, np.ndarray],
    config: SensorPreprocessConfig,
    freeze_baseline: bool,
) -> dict[str, np.ndarray]:
    """对整段数据逐帧跑预处理，收集左右各项特征序列。"""
    proc = DualGasPreprocessor(config)
    keys = ["baseline", "signal", "smooth", "trend", "estimated_input"]
    norm_keys = ["norm_signal", "norm_smooth", "norm_trend", "norm_estimated_input"]
    out: dict[str, list[float]] = {f"left_{k}": [] for k in keys + norm_keys}
    out.update({f"right_{k}": [] for k in keys + norm_keys})
    out["norm_smooth_diff"] = []
    out["norm_trend_diff"] = []

    for left_adc, right_adc in zip(data["left_adc"], data["right_adc"]):
        feat = proc.update(
            left_adc,
            right_adc,
            allow_baseline_update=not freeze_baseline,
        )
        for k in keys + norm_keys:
            out[f"left_{k}"].append(getattr(feat.left, k))
            out[f"right_{k}"].append(getattr(feat.right, k))
        out["norm_smooth_diff"].append(feat.norm_smooth_diff)
        out["norm_trend_diff"].append(feat.norm_trend_diff)

    return {k: np.array(v) for k, v in out.items()}


def build_config(args: argparse.Namespace, dt_s: float) -> SensorPreprocessConfig:
    """用默认值构造预处理配置，命令行参数覆盖对应字段。"""
    defaults = SensorPreprocessConfig()
    return SensorPreprocessConfig(
        dt_s=args.dt_s if args.dt_s is not None else dt_s,
        baseline_tau_s=args.baseline_tau_s or defaults.baseline_tau_s,
        smooth_tau_s=args.smooth_tau_s or defaults.smooth_tau_s,
        response_tau_s=args.response_tau_s or defaults.response_tau_s,
        trend_window=args.trend_window or defaults.trend_window,
        scale=args.scale or defaults.scale,
    )


def make_verdict(
    left: dict[str, float],
    right: dict[str, float],
    features: dict[str, np.ndarray],
    segment_kind: str,
) -> dict[str, object]:
    """根据分段性质给出预处理可用性判断。

    - clean 段：期望 norm_smooth/norm_trend 接近 0，越小越说明漂移噪声被压住；
    - gas 段：期望 norm_smooth 明显抬升，越大越说明真实响应被保留；
    - mixed 段：不做强判定，只报告特征幅度供人工判断。
    """
    max_norm_smooth = float(
        np.max(np.abs(np.concatenate([features["left_norm_smooth"], features["right_norm_smooth"]])))
    )
    max_norm_trend = float(
        np.max(np.abs(np.concatenate([features["left_norm_trend"], features["right_norm_trend"]])))
    )
    saturated = bool(left["max_adc"] >= 4095.0 or right["max_adc"] >= 4095.0)
    max_drift = max(abs(left["first_last_delta"]), abs(right["first_last_delta"]))

    if segment_kind == "clean":
        smooth_ok = max_norm_smooth <= 0.5
        trend_ok = max_norm_trend <= 0.1
        if smooth_ok and trend_ok:
            status = "PASS"
            message = "洁净空气段特征稳定，漂移和噪声被有效压制，可作为后续对比的基线。"
        elif not smooth_ok:
            status = "TUNE_SCALE_OR_BASELINE"
            message = (
                "洁净段 norm_smooth 波动偏大：scale 可能偏小（放大噪声）或 baseline 跟踪不够快。"
                "建议增大 scale 或减小 baseline_tau_s 后重跑。"
            )
        else:
            status = "TUNE_SMOOTH"
            message = (
                "洁净段 norm_trend 波动偏大：短时趋势仍受噪声影响。"
                "建议增大 smooth_tau_s 或 trend_window 后重跑。"
            )
    elif segment_kind == "gas":
        if saturated:
            status = "SATURATED"
            message = (
                "气体段 ADC 顶到 4095（饱和），该浓度下无法分辨梯度。"
                "建议退远气源或降低浓度，使 ADC 工作在线性区后再采集。"
            )
        elif max_norm_smooth >= 1.0:
            status = "RESPONSE_OK"
            message = (
                f"气体段 norm_smooth 峰值 {max_norm_smooth:.2f}，响应明显被保留。"
                "若已被 clip 上限(默认5)截断，可增大 scale 让响应回到可分辨范围。"
            )
        else:
            status = "RESPONSE_WEAK"
            message = (
                "气体段响应偏弱：可能 scale 过大压低了信号，或气源太远/平滑太狠。"
                "建议减小 scale 或检查气源距离与停留时间。"
            )
    else:  # mixed
        status = "MIXED_REPORT_ONLY"
        message = (
            "混合段未做强判定。建议用 --t-start/--t-end 切出洁净段与气体段分别分析。"
        )

    return {
        "segment_kind": segment_kind,
        "status": status,
        "message": message,
        "saturated": saturated,
        "max_abs_norm_smooth": max_norm_smooth,
        "max_abs_norm_trend": max_norm_trend,
        "max_first_last_drift_adc": max_drift,
    }


def _overlay_channel(ax, t, raw, baseline, signal, smooth, side: str) -> None:
    """在单个坐标系内用双 Y 轴叠加原始 ADC 与预处理后曲线。

    左轴：绝对量 raw ADC 与 baseline（数量级 ~hundreds-thousands）。
    右轴：相对量 signal 与 smooth（扣掉基线后在 0 附近），便于和原始曲线对比形状。
    """
    raw_color = "#4c78a8"
    smooth_color = "#e45756"

    # 左轴：原始 ADC + baseline
    ax.plot(t, raw, color=raw_color, lw=1.0, label="raw ADC")
    ax.plot(t, baseline, color=raw_color, ls="--", lw=1.0, alpha=0.7, label="baseline")
    ax.set_ylabel("raw ADC / baseline", color=raw_color)
    ax.tick_params(axis="y", labelcolor=raw_color)
    ax.set_title(f"{side} sensor: raw vs preprocessed")

    # 右轴：预处理后的相对信号
    ax2 = ax.twinx()
    ax2.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax2.plot(t, signal, color=smooth_color, lw=0.8, alpha=0.5, label="signal (raw-baseline)")
    ax2.plot(t, smooth, color=smooth_color, lw=1.6, label="smooth")
    ax2.set_ylabel("preprocessed (relative ADC)", color=smooth_color)
    ax2.tick_params(axis="y", labelcolor=smooth_color)

    # 合并两个轴的图例。
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")


def save_overlay_figure(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    path: Path,
) -> None:
    """双 Y 轴叠加图：原始 ADC 与预处理后曲线画在同一坐标系，便于对比。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    _overlay_channel(
        axes[0], t,
        data["left_adc"], features["left_baseline"],
        features["left_signal"], features["left_smooth"], "left",
    )
    _overlay_channel(
        axes[1], t,
        data["right_adc"], features["right_baseline"],
        features["right_signal"], features["right_smooth"], "right",
    )
    axes[1].set_xlabel("time (s)")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_panels_figure(
    t: np.ndarray,
    data: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    path: Path,
) -> None:
    """分面板图（上一版风格）：raw/baseline、signal、smooth、trend 各占一行。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True)

    # 1) raw + baseline
    axes[0].plot(t, data["left_adc"], color="#4c78a8", label="left raw", lw=1.0)
    axes[0].plot(t, data["right_adc"], color="#f58518", label="right raw", lw=1.0)
    axes[0].plot(t, features["left_baseline"], color="#4c78a8", ls="--", label="left baseline", lw=1.0)
    axes[0].plot(t, features["right_baseline"], color="#f58518", ls="--", label="right baseline", lw=1.0)
    axes[0].set_ylabel("raw ADC / baseline")
    axes[0].set_title("MQ-3 baseline analysis")
    axes[0].legend(fontsize=8, ncol=2)

    # 2) signal = raw - baseline
    axes[1].axhline(0.0, color="grey", lw=0.6)
    axes[1].plot(t, features["left_signal"], color="#4c78a8", label="left signal", lw=1.0)
    axes[1].plot(t, features["right_signal"], color="#f58518", label="right signal", lw=1.0)
    axes[1].set_ylabel("signal (ADC)")
    axes[1].legend(fontsize=8)

    # 3) smooth + estimated_input
    axes[2].axhline(0.0, color="grey", lw=0.6)
    axes[2].plot(t, features["left_smooth"], color="#4c78a8", label="left smooth", lw=1.2)
    axes[2].plot(t, features["right_smooth"], color="#f58518", label="right smooth", lw=1.2)
    axes[2].plot(t, features["left_estimated_input"], color="#4c78a8", ls=":", lw=0.8, label="left est_input")
    axes[2].plot(t, features["right_estimated_input"], color="#f58518", ls=":", lw=0.8, label="right est_input")
    axes[2].set_ylabel("smooth / est_input")
    axes[2].legend(fontsize=8, ncol=2)

    # 4) trend + 左右 smooth diff
    axes[3].axhline(0.0, color="grey", lw=0.6)
    axes[3].plot(t, features["left_trend"], color="#4c78a8", label="left trend", lw=1.0)
    axes[3].plot(t, features["right_trend"], color="#f58518", label="right trend", lw=1.0)
    axes[3].plot(t, features["norm_smooth_diff"], color="#54a24b", lw=1.0, label="norm smooth diff")
    axes[3].set_ylabel("trend / diff")
    axes[3].set_xlabel("time (s)")
    axes[3].legend(fontsize=8, ncol=3)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary(
    path: Path,
    csv_path: Path,
    config: SensorPreprocessConfig,
    left: dict[str, float],
    right: dict[str, float],
    verdict: dict[str, object],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    window_desc = (
        f"[{args.t_start}, {args.t_end}]"
        if args.t_start is not None or args.t_end is not None
        else "全段"
    )
    lines = [
        "# MQ-3 基线分析摘要",
        "",
        f"- 数据文件: `{csv_path}`",
        f"- 分析标签: {args.label or '(无)'}",
        f"- 段性质: {verdict['segment_kind']}",
        f"- 分析窗口(秒): {window_desc}",
        f"- 采样间隔 dt_s: {config.dt_s:.3f}",
        f"- 预处理参数: baseline_tau_s={config.baseline_tau_s}, "
        f"smooth_tau_s={config.smooth_tau_s}, response_tau_s={config.response_tau_s}, "
        f"trend_window={config.trend_window}, scale={config.scale}",
        "",
        "## 左右 raw ADC 统计",
        "",
        "| 指标 | left | right |",
        "|---|---:|---:|",
        f"| mean | {left['mean_adc']:.1f} | {right['mean_adc']:.1f} |",
        f"| std | {left['std_adc']:.2f} | {right['std_adc']:.2f} |",
        f"| min | {left['min_adc']:.0f} | {right['min_adc']:.0f} |",
        f"| max | {left['max_adc']:.0f} | {right['max_adc']:.0f} |",
        f"| 漂移斜率 (ADC/min) | {left['drift_slope_adc_per_min']:.2f} | "
        f"{right['drift_slope_adc_per_min']:.2f} |",
        f"| 首尾窗口差 (ADC) | {left['first_last_delta']:.2f} | "
        f"{right['first_last_delta']:.2f} |",
        f"| 去趋势后噪声 std | {left['noise_std_after_detrend']:.2f} | "
        f"{right['noise_std_after_detrend']:.2f} |",
        "",
        "## 结论",
        "",
        f"- 状态: **{verdict['status']}**",
        f"- max |norm_smooth| = {verdict['max_abs_norm_smooth']:.3f}",
        f"- max |norm_trend| = {verdict['max_abs_norm_trend']:.3f}",
        f"- 最大首尾漂移 = {verdict['max_first_last_drift_adc']:.2f} ADC",
        "",
        f"> {verdict['message']}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def default_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    """根据 --label 生成默认输出路径，显式指定的路径优先。

    figure_path 为分面板图（上一版风格），overlay_path 为双 Y 轴叠加对比图。
    """
    base = ROOT / "results" / "hardware"
    suffix = f"_{args.label}" if args.label else ""
    json_path = args.json_path or base / f"baseline_analysis{suffix}.json"
    figure_path = args.figure_path or base / f"baseline_preprocess{suffix}.png"
    overlay_path = args.overlay_path or base / f"baseline_overlay{suffix}.png"
    summary_path = args.summary_path or base / f"baseline_summary{suffix}.md"
    return json_path, figure_path, overlay_path, summary_path


def apply_window(arrays: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    """对一组等长数组应用同一个布尔掩码。"""
    return {k: v[mask] for k, v in arrays.items()}


def main() -> None:
    args = parse_args()
    json_path, figure_path, overlay_path, summary_path = default_output_paths(args)

    data = load_csv(args.csv_path)
    dt_s = estimate_dt(data["t"])
    config = build_config(args, dt_s)

    # 预处理在完整连续数据流上运行，保持 baseline/EMA 的有状态累积。
    features_full = run_preprocess(data, config, args.freeze_baseline)

    # 分析窗口只用于框出区间计算分段统计/判定/绘图，不改变预处理输入。
    t = data["t"]
    t_start = args.t_start if args.t_start is not None else float(t[0])
    t_end = args.t_end if args.t_end is not None else float(t[-1])
    mask = (t >= t_start) & (t <= t_end)
    if not mask.any():
        raise ValueError(f"分析窗口 [{t_start}, {t_end}] 内没有样本。")

    t_win = t[mask]
    data_win = apply_window({k: v for k, v in data.items() if k != "t"}, mask)
    features_win = apply_window(features_full, mask)

    left = channel_stats(t_win, data_win["left_adc"], args.window_s)
    right = channel_stats(t_win, data_win["right_adc"], args.window_s)
    verdict = make_verdict(left, right, features_win, args.segment_kind)

    result = {
        "csv_path": str(args.csv_path),
        "label": args.label,
        "segment_kind": args.segment_kind,
        "window": {"t_start": t_start, "t_end": t_end},
        "sample_count_total": int(len(t)),
        "sample_count_window": int(mask.sum()),
        "estimated_dt_s": dt_s,
        "preprocess_config": {
            "dt_s": config.dt_s,
            "baseline_tau_s": config.baseline_tau_s,
            "smooth_tau_s": config.smooth_tau_s,
            "response_tau_s": config.response_tau_s,
            "trend_window": config.trend_window,
            "scale": config.scale,
        },
        "left": left,
        "right": right,
        "verdict": verdict,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    save_panels_figure(t_win, data_win, features_win, figure_path)
    save_overlay_figure(t_win, data_win, features_win, overlay_path)
    save_summary(summary_path, args.csv_path, config, left, right, verdict, args)

    print(
        f"label={args.label} segment={args.segment_kind} "
        f"window=[{t_start:.1f},{t_end:.1f}]s samples={int(mask.sum())}/{len(t)}"
    )
    print(
        f"left drift={left['first_last_delta']:.2f} ADC, "
        f"right drift={right['first_last_delta']:.2f} ADC, "
        f"saturated={verdict['saturated']}"
    )
    print(f"verdict={verdict['status']}: {verdict['message']}")
    print(f"saved_json={json_path}")
    print(f"saved_figure={figure_path}")
    print(f"saved_overlay={overlay_path}")
    print(f"saved_summary={summary_path}")


if __name__ == "__main__":
    main()
