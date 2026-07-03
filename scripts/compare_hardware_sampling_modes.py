"""硬件采样模式对比（阶段二）。

在真实硬件上比较固定传感器、周期扫描、随机扫描三类触须采样方式的
“信息获取能力”，回答整个课题最关键的问题：

    扫描触须是否真的比固定角度传感器获得更多有效气味信息？

该脚本复用阶段一标定的预处理参数（默认 scale=1500），把原始 ADC 转成
norm_smooth / norm_trend / smooth_diff 等相对特征，再据此计算统一的信息
获取指标，便于不同模式横向对比。

支持模式：
    fixed_angle    左右触须固定在某个扇区不动
    periodic_scan  左右触须按往返三角波周期扫描 10 个扇区
    random_scan    随机选扇区（any=每步任意；neighbor=每步只动±1，更贴合舵机）

两种运行方式：
    实采：连接 STM32，按模式下发 STEP 命令并采集（需要硬件）
    回放：--replay-csv 读取已有 CSV，只复算特征/指标/图（离线，无需硬件）

输出（results/hardware/sampling_modes/<run_id>/）：
    raw.csv       step,t_rel_s,mode,left_sector,right_sector,left_adc,right_adc
    features.csv  预处理后的左右特征与差分
    metrics.json  统一信息获取指标 + 运行配置
    summary.md    指标摘要
    plots.png     max_smooth / sector / smooth_diff 时间序列

实采示例：
    python scripts\\compare_hardware_sampling_modes.py --port COM3 --mode periodic_scan --steps 200
回放示例（离线测试指标与绘图）：
    python scripts\\compare_hardware_sampling_modes.py --mode fixed_angle --replay-csv results\\hardware\\live_plot.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.hardware.sensor_preprocess import DualGasPreprocessor
from dual_whisker_rl.hardware.sensor_preprocess import SensorPreprocessConfig


# 一条采样记录：原始数据 + 时间 + 当前扇区。
Record = dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "hardware.yaml")
    parser.add_argument("--port", type=str, default=None)
    parser.add_argument(
        "--mode",
        choices=["fixed_angle", "periodic_scan", "random_scan"],
        default="fixed_angle",
    )
    parser.add_argument("--steps", type=int, default=200, help="下发的 STEP 命令总数。")
    parser.add_argument("--delay-s", type=float, default=0.2, help="每步额外等待(秒)。")
    parser.add_argument("--fixed-sector", type=int, default=5, help="fixed_angle 模式的固定扇区。")
    parser.add_argument("--dwell-steps", type=int, default=2, help="periodic_scan 每个扇区停留步数。")
    parser.add_argument(
        "--random-mode",
        choices=["any", "neighbor"],
        default="neighbor",
        help="random_scan 子模式：any=任意扇区；neighbor=只动±1。",
    )
    parser.add_argument("--seed", type=int, default=0, help="随机扫描的随机种子。")
    # 预处理参数：默认沿用阶段一标定值，scale 在 build_config 里默认 1500。
    parser.add_argument("--scale", type=float, default=1500.0)
    parser.add_argument("--baseline-tau-s", type=float, default=None)
    parser.add_argument("--smooth-tau-s", type=float, default=None)
    parser.add_argument("--trend-window", type=int, default=None)
    parser.add_argument("--dt-s", type=float, default=None, help="不指定则从时间戳自动推算。")
    # 信息获取指标阈值。
    parser.add_argument("--hit-threshold", type=float, default=0.2, help="norm_smooth 命中阈值。")
    parser.add_argument("--diff-threshold", type=float, default=0.2, help="左右不对称阈值。")
    parser.add_argument("--trend-threshold", type=float, default=0.05, help="趋势阈值。")
    # 运行方式与输出。
    parser.add_argument("--replay-csv", type=Path, default=None, help="离线回放：从 CSV 读数据，不连硬件。")
    parser.add_argument("--run-id", type=str, default=None, help="输出子目录名，默认按时间+模式生成。")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "hardware" / "sampling_modes")
    parser.add_argument("--source-note", type=str, default="", help="环境备注，例如气源/风扇/距离。")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_config(args: argparse.Namespace, dt_s: float) -> SensorPreprocessConfig:
    """构造预处理配置，命令行参数覆盖对应字段。"""
    defaults = SensorPreprocessConfig()
    return SensorPreprocessConfig(
        dt_s=args.dt_s if args.dt_s is not None else dt_s,
        baseline_tau_s=args.baseline_tau_s or defaults.baseline_tau_s,
        smooth_tau_s=args.smooth_tau_s or defaults.smooth_tau_s,
        response_tau_s=defaults.response_tau_s,
        trend_window=args.trend_window or defaults.trend_window,
        scale=args.scale,
    )


def make_action_sequence(args: argparse.Namespace, sector_count: int) -> list[tuple[int, int]]:
    """根据模式生成长度为 steps 的 (left_sector, right_sector) 序列。

    左右触须采用对称扇区（与现有 smoke_test/sector_scan 约定一致）。左右传感器
    虽然指向同一扇区编号，但物理基座位置不同，仍是两个不同采样点。
    """
    steps = args.steps
    if args.mode == "fixed_angle":
        s = int(np.clip(args.fixed_sector, 0, sector_count - 1))
        return [(s, s)] * steps

    if args.mode == "periodic_scan":
        # 往返三角波：0->...->9->...->0，避免大角度快速回摆。
        sweep = list(range(sector_count)) + list(range(sector_count - 2, 0, -1))
        expanded: list[int] = []
        for sector in sweep:
            expanded.extend([sector] * args.dwell_steps)
        return [(expanded[i % len(expanded)],) * 2 for i in range(steps)]

    # random_scan
    rng = np.random.default_rng(args.seed)
    seq: list[tuple[int, int]] = []
    current = sector_count // 2
    for _ in range(steps):
        if args.random_mode == "any":
            current = int(rng.integers(0, sector_count))
        else:  # neighbor
            current = int(np.clip(current + rng.integers(-1, 2), 0, sector_count - 1))
        seq.append((current, current))
    return seq


def collect_live(args: argparse.Namespace, serial_cfg: dict, actions: list[tuple[int, int]]) -> list[Record]:
    """连接硬件，按动作序列下发 STEP 并采集返回。"""
    from dual_whisker_rl.hardware.serial_client import DualWhiskerSerialClient

    port = args.port or str(serial_cfg["port"])
    records: list[Record] = []
    t0: float | None = None
    with DualWhiskerSerialClient(
        port=port,
        baudrate=int(serial_cfg["baudrate"]),
        timeout_s=float(serial_cfg["timeout_s"]),
    ) as client:
        for step, (left, right) in enumerate(actions):
            sample = client.step(left, right)
            if t0 is None:
                t0 = sample.t_pc_s
            records.append(
                {
                    "step": step,
                    "t_rel_s": sample.t_pc_s - t0,
                    "left_sector": sample.left_sector,
                    "right_sector": sample.right_sector,
                    "left_adc": float(sample.left_adc),
                    "right_adc": float(sample.right_adc),
                }
            )
            print(
                f"step={step} mode={args.mode} sec={left},{right} "
                f"L={sample.left_adc} R={sample.right_adc}"
            )
            time.sleep(args.delay_s)
    return records


def collect_replay(path: Path) -> list[Record]:
    """离线回放：从已有 CSV 读取，不连硬件。用于测试指标与绘图逻辑。"""
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"回放 CSV 为空: {path}")

    has_rel = "t_rel_s" in rows[0]
    t0 = float(rows[0]["t_pc_s"]) if (not has_rel and "t_pc_s" in rows[0]) else 0.0
    for step, row in enumerate(rows):
        if has_rel:
            t_rel = float(row["t_rel_s"])
        elif "t_pc_s" in row:
            t_rel = float(row["t_pc_s"]) - t0
        else:
            t_rel = float(step)
        records.append(
            {
                "step": step,
                "t_rel_s": t_rel,
                "left_sector": int(row["left_sector"]),
                "right_sector": int(row["right_sector"]),
                "left_adc": float(row["left_adc"]),
                "right_adc": float(row["right_adc"]),
            }
        )
    return records


def estimate_dt(t: np.ndarray) -> float:
    if len(t) < 2:
        return 0.8
    diffs = np.diff(t)
    diffs = diffs[diffs > 0.0]
    return float(np.median(diffs)) if diffs.size else 0.8


def compute_features(records: list[Record], config: SensorPreprocessConfig) -> dict[str, np.ndarray]:
    """在完整连续序列上运行预处理，返回左右特征与差分数组。"""
    proc = DualGasPreprocessor(config)
    keys = ["smooth", "trend", "signal"]
    norm_keys = ["norm_smooth", "norm_trend"]
    out: dict[str, list[float]] = {f"left_{k}": [] for k in keys + norm_keys}
    out.update({f"right_{k}": [] for k in keys + norm_keys})
    out["norm_smooth_diff"] = []
    out["norm_trend_diff"] = []

    for rec in records:
        feat = proc.update(rec["left_adc"], rec["right_adc"])
        for k in keys + norm_keys:
            out[f"left_{k}"].append(getattr(feat.left, k))
            out[f"right_{k}"].append(getattr(feat.right, k))
        out["norm_smooth_diff"].append(feat.norm_smooth_diff)
        out["norm_trend_diff"].append(feat.norm_trend_diff)

    return {k: np.array(v) for k, v in out.items()}


def compute_metrics(
    t: np.ndarray,
    sectors: np.ndarray,
    features: dict[str, np.ndarray],
    args: argparse.Namespace,
    sector_count: int,
) -> dict[str, float]:
    """根据预处理特征计算统一的信息获取指标。"""
    max_smooth = np.maximum(features["left_norm_smooth"], features["right_norm_smooth"])
    abs_smooth_diff = np.abs(features["norm_smooth_diff"])
    abs_trend_sum = np.abs(features["left_norm_trend"]) + np.abs(features["right_norm_trend"])
    max_trend = np.maximum(features["left_norm_trend"], features["right_norm_trend"])

    hits = max_smooth > args.hit_threshold

    # 重新捕获：从“丢失(低于阈值)”状态再次超过阈值的事件与平均耗时。
    reacquire_times: list[float] = []
    in_hit = bool(hits[0])
    loss_time: float | None = None
    for i in range(1, len(hits)):
        if in_hit and not hits[i]:
            in_hit = False
            loss_time = float(t[i])
        elif not in_hit and hits[i]:
            in_hit = True
            if loss_time is not None:
                reacquire_times.append(float(t[i]) - loss_time)

    high_info = (
        (max_smooth > args.hit_threshold)
        | (abs_smooth_diff > args.diff_threshold)
        | (abs_trend_sum > args.trend_threshold)
    )

    # 扇区覆盖与熵（统计左右触须实际访问过的扇区分布）。
    counts = np.bincount(sectors, minlength=sector_count).astype(float)
    probs = counts / counts.sum() if counts.sum() > 0 else counts
    nonzero = probs[probs > 0]
    sector_entropy = float(-np.sum(nonzero * np.log2(nonzero)) + 0.0) if nonzero.size else 0.0
    sector_coverage = float(np.count_nonzero(counts) / sector_count)

    duration = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return {
        "sample_count": int(len(t)),
        "duration_s": duration,
        "mean_left_smooth": float(np.mean(features["left_norm_smooth"])),
        "mean_right_smooth": float(np.mean(features["right_norm_smooth"])),
        "mean_max_smooth": float(np.mean(max_smooth)),
        "odor_hit_rate": float(np.mean(hits)),
        "mean_abs_smooth_diff": float(np.mean(abs_smooth_diff)),
        "mean_abs_trend_sum": float(np.mean(abs_trend_sum)),
        "positive_trend_rate": float(np.mean(max_trend > args.trend_threshold)),
        "high_information_rate": float(np.mean(high_info)),
        "reacquisition_events": int(len(reacquire_times)),
        "mean_reacquisition_time_s": float(np.mean(reacquire_times)) if reacquire_times else 0.0,
        "sector_entropy_bits": sector_entropy,
        "sector_coverage": sector_coverage,
    }


def save_raw_csv(path: Path, records: list[Record], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["step", "t_rel_s", "mode", "left_sector", "right_sector", "left_adc", "right_adc"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow(
                {
                    "step": int(rec["step"]),
                    "t_rel_s": f"{rec['t_rel_s']:.6f}",
                    "mode": mode,
                    "left_sector": int(rec["left_sector"]),
                    "right_sector": int(rec["right_sector"]),
                    "left_adc": int(rec["left_adc"]),
                    "right_adc": int(rec["right_adc"]),
                }
            )


def save_features_csv(path: Path, records: list[Record], features: dict[str, np.ndarray], mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step", "mode", "left_sector", "right_sector",
        "left_smooth", "right_smooth", "left_trend", "right_trend",
        "left_norm_smooth", "right_norm_smooth", "left_norm_trend", "right_norm_trend",
        "norm_smooth_diff", "norm_trend_diff",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for i, rec in enumerate(records):
            writer.writerow([
                int(rec["step"]), mode, int(rec["left_sector"]), int(rec["right_sector"]),
                f"{features['left_smooth'][i]:.3f}", f"{features['right_smooth'][i]:.3f}",
                f"{features['left_trend'][i]:.4f}", f"{features['right_trend'][i]:.4f}",
                f"{features['left_norm_smooth'][i]:.4f}", f"{features['right_norm_smooth'][i]:.4f}",
                f"{features['left_norm_trend'][i]:.4f}", f"{features['right_norm_trend'][i]:.4f}",
                f"{features['norm_smooth_diff'][i]:.4f}", f"{features['norm_trend_diff'][i]:.4f}",
            ])


def save_plots(path: Path, t: np.ndarray, records: list[Record], features: dict[str, np.ndarray], args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_smooth = np.maximum(features["left_norm_smooth"], features["right_norm_smooth"])
    left_sec = np.array([r["left_sector"] for r in records])
    right_sec = np.array([r["right_sector"] for r in records])

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    axes[0].plot(t, max_smooth, color="#e45756", lw=1.2, label="max norm_smooth")
    axes[0].axhline(args.hit_threshold, color="grey", ls="--", lw=0.8, label="hit threshold")
    axes[0].set_ylabel("max norm_smooth")
    axes[0].set_title(f"sampling mode: {args.mode}")
    axes[0].legend(fontsize=8)

    axes[1].step(t, left_sec, where="post", color="#4c78a8", lw=1.0, label="left sector")
    axes[1].step(t, right_sec, where="post", color="#f58518", lw=1.0, ls="--", label="right sector")
    axes[1].set_ylabel("sector")
    axes[1].legend(fontsize=8)

    axes[2].axhline(0.0, color="grey", lw=0.6)
    axes[2].plot(t, features["norm_smooth_diff"], color="#54a24b", lw=1.0, label="norm_smooth_diff")
    axes[2].set_ylabel("L-R smooth diff")
    axes[2].set_xlabel("time (s)")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_summary(path: Path, args: argparse.Namespace, metrics: dict[str, float], config: SensorPreprocessConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# 采样模式对比摘要: {args.mode}",
        "",
        f"- 运行方式: {'回放' if args.replay_csv else '实采'}",
        f"- 环境备注: {args.source_note or '(无)'}",
        f"- 预处理: scale={config.scale}, dt_s={config.dt_s:.3f}, "
        f"smooth_tau_s={config.smooth_tau_s}, trend_window={config.trend_window}",
        f"- 阈值: hit={args.hit_threshold}, diff={args.diff_threshold}, trend={args.trend_threshold}",
        "",
        "## 信息获取指标",
        "",
        "| 指标 | 值 |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        shown = f"{value:.4f}" if isinstance(value, float) else str(value)
        lines.append(f"| {key} | {shown} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    config_dict = load_config(args.config)
    sector_count = int(config_dict["whisker"]["sector_count"])

    if args.replay_csv:
        records = collect_replay(args.replay_csv)
    else:
        actions = make_action_sequence(args, sector_count)
        records = collect_live(args, config_dict["serial"], actions)

    if not records:
        raise ValueError("没有采集到任何样本。")

    t = np.array([r["t_rel_s"] for r in records])
    sectors = np.array([r["left_sector"] for r in records] + [r["right_sector"] for r in records])
    dt_s = estimate_dt(t)
    config = build_config(args, dt_s)
    features = compute_features(records, config)
    metrics = compute_metrics(t, sectors, features, args, sector_count)

    run_id = args.run_id or f"{time.strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    out_dir = args.output_dir / run_id
    save_raw_csv(out_dir / "raw.csv", records, args.mode)
    save_features_csv(out_dir / "features.csv", records, features, args.mode)
    save_plots(out_dir / "plots.png", t, records, features, args)
    save_summary(out_dir / "summary.md", args, metrics, config)

    metrics_payload = {
        "run_id": run_id,
        "mode": args.mode,
        "replay": bool(args.replay_csv),
        "source_note": args.source_note,
        "preprocess_config": {
            "scale": config.scale,
            "dt_s": config.dt_s,
            "smooth_tau_s": config.smooth_tau_s,
            "baseline_tau_s": config.baseline_tau_s,
            "trend_window": config.trend_window,
        },
        "thresholds": {
            "hit": args.hit_threshold,
            "diff": args.diff_threshold,
            "trend": args.trend_threshold,
        },
        "metrics": metrics,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"mode={args.mode} samples={metrics['sample_count']} dt_s={dt_s:.3f}")
    print(
        f"odor_hit_rate={metrics['odor_hit_rate']:.3f} "
        f"mean_max_smooth={metrics['mean_max_smooth']:.3f} "
        f"mean_abs_trend_sum={metrics['mean_abs_trend_sum']:.3f} "
        f"sector_coverage={metrics['sector_coverage']:.2f}"
    )
    print(f"saved_dir={out_dir}")


if __name__ == "__main__":
    main()
