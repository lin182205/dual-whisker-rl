"""羽流间歇性诊断：在固定采样点上量化 whiff/blank 统计。

该脚本不训练、不改环境，只把 `DynamicPuffPlume` 跑一段长时间，在若干
固定采样点记录瞬时浓度时间序列，然后计算并绘制：

- intermittency factor（高于阈值的时间占比）；
- whiff 时长分布（连续高于阈值的时长）；
- blank 时长分布（连续低于阈值的时长）；
- 浓度时间序列与阈值线。

用途：在改进湍流模型/密度之前先量出当前基准，改进之后再用同样的
参数复跑对比，判断"间歇结构是否更接近真实羽流"。

真实间歇羽流的 whiff/blank 时长大致呈指数尾，intermittency factor
在羽流边缘较低、中心较高。可据此判断仿真是否过于平滑（接近常通）或
过于稀疏（几乎全是 blank）。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs.whisker_only_env import DynamicPuffPlume
from dual_whisker_rl.paths import resolve_path_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose whiff/blank intermittency of the dynamic puff plume."
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--config", type=Path, default=None, help="可选 Mobile 配置；未提供时保留旧诊断。")
    parser.add_argument("--source-position", type=float, nargs=2, default=None)
    parser.add_argument("--wind-direction-deg", type=float, default=None)
    parser.add_argument("--steps", type=int, default=3000, help="采样步数。")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument(
        "--wind-mode",
        choices=["fixed", "random"],
        default="fixed",
        help="fixed 给出稳态羽流，便于统计；random 更接近训练分布。",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.08,
        help="判定 whiff 的浓度阈值，默认与环境 hit_threshold 一致。",
    )
    parser.add_argument(
        "--downwind-distance",
        type=float,
        default=0.75,
        help="采样点到气源的下风向距离（米），默认约为场地中心。",
    )
    parser.add_argument(
        "--crosswind-offsets",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3],
        help="相对羽流中心线的横风向采样偏移（米），用于看中心到边缘的间歇结构。",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="",
        help="输出文件名后缀，用于区分 before/after。",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/figures"),
    )
    return resolve_path_args(parser.parse_args(), "out_dir", "config")


def sample_points(
    plume: DynamicPuffPlume, downwind_distance: float, crosswind_offsets: list[float]
) -> list[tuple[float, float]]:
    """在气源下风向固定距离处，沿横风向取一排采样点（中心线 → 边缘）。"""
    wind_x = math.cos(plume.wind_direction_base)
    wind_y = math.sin(plume.wind_direction_base)
    cross_x = -wind_y
    cross_y = wind_x
    base_x = plume.source_x + downwind_distance * wind_x
    base_y = plume.source_y + downwind_distance * wind_y
    points = []
    for c in crosswind_offsets:
        points.append((float(base_x + c * cross_x), float(base_y + c * cross_y)))
    return points


def run_series(
    plume: DynamicPuffPlume, points: list[tuple[float, float]], steps: int
) -> np.ndarray:
    """跑 steps 步，返回形状 (steps, n_points) 的瞬时浓度时间序列。"""
    series = np.zeros((steps, len(points)), dtype=np.float64)
    for t in range(steps):
        plume.advance()
        if hasattr(plume, "_concentrations"):
            series[t] = plume._concentrations(points, add_noise=False)
        else:
            for j, (px, py) in enumerate(points):
                series[t, j] = plume.concentration(px, py, add_noise=False)
    return series


def run_lengths(mask: np.ndarray) -> np.ndarray:
    """返回布尔序列中连续 True 段的长度数组（步数）。"""
    if mask.size == 0:
        return np.array([], dtype=np.int64)
    padded = np.concatenate(([0], mask.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).astype(np.int64)


def summarize_point(signal: np.ndarray, threshold: float, dt: float) -> dict:
    above = signal >= threshold
    whiff_steps = run_lengths(above)
    blank_steps = run_lengths(~above)

    def _stats(lengths: np.ndarray) -> dict:
        if lengths.size == 0:
            return {"count": 0, "mean_s": 0.0, "median_s": 0.0, "max_s": 0.0}
        sec = lengths * dt
        return {
            "count": int(lengths.size),
            "mean_s": float(np.mean(sec)),
            "median_s": float(np.median(sec)),
            "max_s": float(np.max(sec)),
        }

    return {
        "intermittency_factor": float(np.mean(above)),
        "mean_concentration": float(np.mean(signal)),
        "max_concentration": float(np.max(signal)),
        "whiff": _stats(whiff_steps),
        "blank": _stats(blank_steps),
        "_whiff_sec": (whiff_steps * dt).tolist(),
        "_blank_sec": (blank_steps * dt).tolist(),
    }


def make_plot(
    series: np.ndarray,
    points: list[tuple[float, float]],
    offsets: list[float],
    summaries: list[dict],
    threshold: float,
    dt: float,
    out_path: Path,
) -> None:
    n = len(points)
    fig, axes = plt.subplots(n, 2, figsize=(13, 2.6 * n), squeeze=False)
    times = np.arange(series.shape[0]) * dt
    # 时间序列只画前一段，避免过密。
    show = min(series.shape[0], int(120 / dt))
    for j in range(n):
        ax_ts = axes[j][0]
        ax_ts.plot(times[:show], series[:show, j], color="#2d79bd", lw=0.8)
        ax_ts.axhline(threshold, color="#c0392b", ls="--", lw=0.9, label="threshold")
        s = summaries[j]
        ax_ts.set_title(
            f"crosswind offset {offsets[j]:+.2f} m  |  "
            f"intermittency={s['intermittency_factor']:.2f}",
            fontsize=9,
        )
        ax_ts.set_ylabel("concentration")
        if j == n - 1:
            ax_ts.set_xlabel("time (s)")
        ax_ts.legend(fontsize=7, loc="upper right")

        ax_h = axes[j][1]
        whiff = np.array(s["_whiff_sec"])
        blank = np.array(s["_blank_sec"])
        if whiff.size:
            ax_h.hist(whiff, bins=20, alpha=0.6, color="#2d79bd", label="whiff")
        if blank.size:
            ax_h.hist(blank, bins=20, alpha=0.6, color="#e08a3c", label="blank")
        ax_h.set_title(
            f"whiff mean={s['whiff']['mean_s']:.2f}s  blank mean={s['blank']['mean_s']:.2f}s",
            fontsize=9,
        )
        ax_h.set_ylabel("count")
        if j == n - 1:
            ax_h.set_xlabel("duration (s)")
        ax_h.legend(fontsize=7)

    fig.suptitle("Plume intermittency diagnostic", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    env = None
    if args.config:
        import yaml
        from dual_whisker_rl.envs import MobileWhiskerPuffEnv
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        env = MobileWhiskerPuffEnv(config)
        plume = env.plume
        plume.wind_sampling_mode = args.wind_mode
        scenario = {}
        if args.source_position is not None:
            scenario["source_position"] = args.source_position
        if args.wind_direction_deg is not None:
            scenario["wind_direction"] = math.radians(args.wind_direction_deg)
        env.reset(seed=args.seed, options={"scenario": scenario})
        args.dt = plume.dt
    else:
        plume = DynamicPuffPlume(seed=args.seed, dt=args.dt, wind_sampling_mode=args.wind_mode)
        plume.reset(args.seed)

    points = sample_points(plume, args.downwind_distance, list(args.crosswind_offsets))
    if any(not plume.is_position_valid(*p) for p in points) and env is not None:
        raise ValueError("诊断采样点落在墙内或场外，请调整源位置、距离或偏移")
    series = run_series(plume, points, args.steps)

    summaries = [summarize_point(series[:, j], args.threshold, args.dt) for j in range(len(points))]
    offsets = list(args.crosswind_offsets)

    suffix = f"_{args.label}" if args.label else ""
    png_path = args.out_dir / f"plume_intermittency{suffix}.png"
    json_path = args.out_dir / f"plume_intermittency{suffix}.json"

    make_plot(series, points, offsets, summaries, args.threshold, args.dt, png_path)

    # JSON 中去掉内部原始时长列表，保留汇总。
    report = {
        "config": {
            "seed": args.seed,
            "steps": args.steps,
            "dt": args.dt,
            "wind_mode": args.wind_mode,
            "threshold": args.threshold,
            "downwind_distance": args.downwind_distance,
            "crosswind_offsets": offsets,
            "label": args.label,
        },
        "points": [
            {
                "crosswind_offset_m": float(off),
                "position": [float(px), float(py)],
                **{k: v for k, v in s.items() if not k.startswith("_")},
            }
            for off, (px, py), s in zip(offsets, points, summaries)
        ],
    }
    if env is not None:
        from dual_whisker_rl.envs.sensor_model import AsymmetricGasSensor
        sensor_series = np.zeros_like(series)
        for j in range(len(points)):
            sensor = AsymmetricGasSensor(response_tau=env.sensor_response_tau,
                recovery_tau=env.sensor_recovery_tau, dt=env.dt, noise_std=env.sensor_noise_std,
                rng=np.random.default_rng(args.seed+j))
            sensor_series[:, j] = [sensor.update(v) for v in series[:, j]]
        report["environment_config"] = config
        report["map_sha256"] = env.room_navigation.map.digest if env.room_navigation else None
        report["concentration_distribution"] = {
            "quantiles": [0, .1, .5, .9, .99, 1],
            "raw": np.quantile(series, [0, .1, .5, .9, .99, 1]).tolist(),
            "sensor": np.quantile(sensor_series, [0, .1, .5, .9, .99, 1]).tolist(),
            "sensor_hit_ratio": float(np.mean(sensor_series >= env.hit_threshold)),
            "environment_hit_threshold": env.hit_threshold,
        }
        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        times = np.arange(args.steps)*args.dt
        for j, point in enumerate(points):
            axes[0].plot(times, series[:, j], lw=.7, label=f"({point[0]:.2f},{point[1]:.2f})")
            axes[1].plot(times, sensor_series[:, j], lw=.7)
        axes[0].set_ylabel("Raw concentration")
        axes[1].set_ylabel("MQ-3 response")
        axes[1].set_xlabel("Time (s)")
        for ax in axes:
            ax.axhline(env.hit_threshold, color="black", ls="--", lw=.8)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.out_dir/f"plume_sensor_response{suffix}.png", dpi=130)
        plt.close(fig)
        env.close()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"wrote {png_path}")
    print(f"wrote {json_path}")
    for off, s in zip(offsets, summaries):
        print(
            f"  crosswind {off:+.2f}m: intermittency={s['intermittency_factor']:.3f}  "
            f"whiff_mean={s['whiff']['mean_s']:.2f}s (n={s['whiff']['count']})  "
            f"blank_mean={s['blank']['mean_s']:.2f}s (n={s['blank']['count']})  "
            f"mean_c={s['mean_concentration']:.3f}"
        )


if __name__ == "__main__":
    main()
