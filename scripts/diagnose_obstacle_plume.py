"""诊断 D2Q9 LBM 障碍风场与 Puff 羽流耦合。

脚本先运行确定性数值检查，再用指定移动环境配置生成风速、涡量、尾流湍流、
瞬时/时间平均浓度及下风截面图。输出仅用于诊断，不向策略观测添加风场信息。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.envs.lbm_wind_field import SteadyLBMWindField2D
from dual_whisker_rl.envs.whisker_only_env import DynamicPuffPlume
from dual_whisker_rl.paths import resolve_path_args


FAST_LBM_CONFIG = {
    "resolution": 65,
    "padding_cells": 6,
    "max_iterations": 400,
    "averaging_iterations": 100,
    "convergence_tolerance": 1e-4,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mobile_obstacles.yaml"),
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--resolution", type=int, default=160)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/obstacle_plume_lbm_diagnostic.png"),
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("results/figures/obstacle_plume_lbm_diagnostic.json"),
    )
    return resolve_path_args(
        parser.parse_args(),
        "config",
        "output",
        "json_output",
    )


def _assert_no_puffs_inside(plume: DynamicPuffPlume) -> None:
    for puff in plume.puffs:
        for xmin, xmax, ymin, ymax in plume.obstacles:
            assert not (
                xmin <= puff["x"] <= xmax and ymin <= puff["y"] <= ymax
            ), f"puff entered obstacle: {puff}"


def check_no_obstacle_regression() -> None:
    """空障碍时 obstacle_flow 开关不得改变旧随机序列或 puff 状态。"""
    common = {
        "seed": 19,
        "source_position": (-0.3, 0.0),
        "world_half": 0.5,
        "wind_sampling_mode": "fixed",
        "plume_overrides": {"puff_warmup_steps": 20},
    }
    baseline = DynamicPuffPlume(**common)
    disabled = DynamicPuffPlume(
        **common,
        obstacle_flow={"enabled": False},
    )
    baseline.reset(19)
    disabled.reset(19)
    for _ in range(30):
        baseline.advance()
        disabled.advance()
    assert len(baseline.puffs) == len(disabled.puffs)
    fields = (
        "x",
        "y",
        "mass",
        "sigma_downwind",
        "sigma_crosswind",
        "direction",
        "age",
        "vd",
        "vc",
    )
    for left, right in zip(baseline.puffs, disabled.puffs):
        for field in fields:
            assert left[field] == right[field], f"no-obstacle regression in {field}"


def check_lbm_geometry() -> None:
    obstacle = np.asarray([[-0.10, 0.20, -0.90, 0.90]], dtype=np.float32)
    flow = SteadyLBMWindField2D(
        world_min=-2.0,
        world_max=2.0,
        obstacles=obstacle,
        config=FAST_LBM_CONFIG,
    )
    result = flow.prepare(0.0)
    assert result is not None
    assert np.all(np.isfinite(result.velocity_x_normalized))
    assert np.all(np.isfinite(result.velocity_y_normalized))
    assert 0.8 < result.density_min < result.density_max < 1.2
    solid_speed = np.hypot(
        result.velocity_x_normalized[result.solid_mask],
        result.velocity_y_normalized[result.solid_mask],
    )
    np.testing.assert_allclose(solid_speed, 0.0, atol=1e-12)

    _, upstream_y, _ = flow.sample(-0.55, 0.0, wind_speed=1.0)
    wake_x, wake_y, _ = flow.sample(0.55, 0.0, wind_speed=1.0)
    top_x, top_y, _ = flow.sample(0.05, 1.15, wind_speed=1.0)
    bottom_x, bottom_y, _ = flow.sample(0.05, -1.15, wind_speed=1.0)
    assert abs(upstream_y) < 0.08
    assert wake_x < 0.35
    assert top_x > 1.05 and bottom_x > 1.05
    np.testing.assert_allclose(top_x, bottom_x, rtol=0.10, atol=0.08)
    np.testing.assert_allclose(top_y, -bottom_y, rtol=0.20, atol=0.10)
    assert abs(wake_y) < 0.10

    cached = SteadyLBMWindField2D(
        world_min=-2.0,
        world_max=2.0,
        obstacles=obstacle,
        config=FAST_LBM_CONFIG,
    )
    cached_result = cached.prepare(math.radians(3.0))
    assert cached.cache_hit
    assert cached_result is result

    square = np.asarray([[-0.4, 0.4, -0.4, 0.4]], dtype=np.float32)
    horizontal = SteadyLBMWindField2D(
        world_min=-2.0,
        world_max=2.0,
        obstacles=square,
        config=FAST_LBM_CONFIG,
    ).prepare(0.0)
    vertical = SteadyLBMWindField2D(
        world_min=-2.0,
        world_max=2.0,
        obstacles=square,
        config=FAST_LBM_CONFIG,
    ).prepare(math.pi / 2.0)
    assert horizontal is not None and vertical is not None
    expected_vertical_x = -np.rot90(horizontal.velocity_y_normalized)
    expected_vertical_y = np.rot90(horizontal.velocity_x_normalized)
    fluid = ~vertical.solid_mask
    rotation_error_x = np.abs(
        vertical.velocity_x_normalized[fluid] - expected_vertical_x[fluid]
    )
    rotation_error_y = np.abs(
        vertical.velocity_y_normalized[fluid] - expected_vertical_y[fluid]
    )
    assert float(np.mean(rotation_error_x)) < 0.10
    assert float(np.mean(rotation_error_y)) < 0.12
    assert float(np.percentile(rotation_error_x, 90)) < 0.30
    assert float(np.percentile(rotation_error_y, 90)) < 0.30


def check_puff_blocking() -> None:
    plume = DynamicPuffPlume(
        seed=23,
        source_position=(-1.5, 0.0),
        world_half=2.0,
        wind_speed_range=(0.16, 0.16),
        obstacles=[[-0.10, 0.20, -0.90, 0.90]],
        obstacle_flow=FAST_LBM_CONFIG,
        plume_overrides={"puff_warmup_steps": 30, "max_puff_age": 24.0},
    )
    plume.reset(23)
    for _ in range(160):
        plume.advance()
        _assert_no_puffs_inside(plume)

    moved_x, moved_y = plume._move_puff_without_crossing_obstacles(
        -0.20,
        0.0,
        0.80,
        0.0,
    )
    assert moved_x < -0.10 and abs(moved_y) < 1e-8

    # 人工放置一个窄 puff，验证 Gaussian 核不会直接穿过墙体。
    plume.puffs = [
        {
            "x": -0.20,
            "y": 0.0,
            "mass": 0.02,
            "sigma_downwind": 0.20,
            "sigma_crosswind": 0.08,
            "direction": 0.0,
            "age": 1.0,
            "vd": 0.0,
            "vc": 0.0,
        }
    ]
    same_side = plume.concentration(-0.25, 0.0, add_noise=False)
    across_wall = plume.concentration(0.30, 0.0, add_noise=False)
    assert same_side > plume.gas_background
    np.testing.assert_allclose(across_wall, plume.gas_background, atol=1e-8)


def run_deterministic_checks() -> None:
    check_no_obstacle_regression()
    check_lbm_geometry()
    check_puff_blocking()
    print("obstacle_plume_deterministic_checks=PASS")


def _draw_obstacles(ax: plt.Axes, obstacles: np.ndarray) -> None:
    for xmin, xmax, ymin, ymax in obstacles:
        ax.add_patch(
            Rectangle(
                (float(xmin), float(ymin)),
                float(xmax - xmin),
                float(ymax - ymin),
                facecolor="#555555",
                edgecolor="#171717",
                linewidth=1.0,
                zorder=5,
            )
        )


def generate_diagnostic(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
    plume = DynamicPuffPlume(
        seed=args.seed,
        source_position=tuple(config.get("source_position", (-1.8, 0.0))),
        world_half=float(config.get("world_half", 2.0)),
        wind_speed_range=tuple(config.get("wind_speed_range", (0.12, 0.20))),
        wind_sampling_mode=str(config.get("wind_sampling_mode", "random")),
        obstacles=config.get("obstacles"),
        obstacle_flow=config.get("obstacle_flow"),
        plume_overrides={
            **dict(config.get("plume_overrides") or {}),
            "max_puff_age": float(
                dict(config.get("plume_overrides") or {}).get("max_puff_age", 24.0)
            ),
            "puff_warmup_steps": int(
                dict(config.get("plume_overrides") or {}).get("puff_warmup_steps", 120)
            ),
        },
    )
    start = time.perf_counter()
    plume.reset(args.seed)
    reset_seconds = time.perf_counter() - start
    result = plume.obstacle_flow.result
    if result is None:
        raise RuntimeError("diagnostic config must enable obstacle_flow with obstacles")

    xs, ys, instantaneous = plume.grid(args.resolution, add_noise=False)
    average = np.zeros_like(instantaneous, dtype=np.float64)
    sample_count = 0
    for step in range(args.steps):
        plume.advance()
        _assert_no_puffs_inside(plume)
        if (step + 1) % max(1, args.sample_stride) == 0:
            _, _, frame = plume.grid(args.resolution, add_noise=False)
            average += frame
            sample_count += 1
    average /= max(sample_count, 1)

    velocity_x = result.velocity_x_normalized * plume.wind_speed
    velocity_y = result.velocity_y_normalized * plume.wind_speed
    speed = np.hypot(velocity_x, velocity_y)
    grid_x, grid_y = np.meshgrid(result.x, result.y)
    obstacles = plume.obstacles

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    ax = axes[0, 0]
    image = ax.imshow(
        speed,
        extent=[result.x[0], result.x[-1], result.y[0], result.y[-1]],
        origin="lower",
        cmap="viridis",
        interpolation="bicubic",
    )
    stride = max(1, result.x.size // 32)
    ax.streamplot(
        grid_x[::stride, ::stride],
        grid_y[::stride, ::stride],
        velocity_x[::stride, ::stride],
        velocity_y[::stride, ::stride],
        color="white",
        density=1.1,
        linewidth=0.55,
        arrowsize=0.65,
    )
    fig.colorbar(image, ax=ax, label="speed (m/s)")
    ax.set_title("Steady D2Q9 LBM speed and streamlines")

    ax = axes[0, 1]
    vmax_vorticity = max(float(np.percentile(np.abs(result.vorticity_normalized), 99)), 1e-4)
    image = ax.imshow(
        result.vorticity_normalized,
        extent=[result.x[0], result.x[-1], result.y[0], result.y[-1]],
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax_vorticity,
        vmax=vmax_vorticity,
        interpolation="bicubic",
    )
    fig.colorbar(image, ax=ax, label="normalized vorticity")
    ax.set_title("Vorticity / shear layers")

    ax = axes[0, 2]
    image = ax.imshow(
        result.turbulence_factor,
        extent=[result.x[0], result.x[-1], result.y[0], result.y[-1]],
        origin="lower",
        cmap="magma",
        vmin=1.0,
        vmax=3.0,
        interpolation="bicubic",
    )
    fig.colorbar(image, ax=ax, label="puff turbulence multiplier")
    ax.set_title("Wake turbulence multiplier")

    concentration_vmax = max(
        0.05,
        float(np.percentile(np.concatenate([instantaneous.ravel(), average.ravel()]), 99.2)),
    )
    for ax, field, title in (
        (axes[1, 0], instantaneous, "Instantaneous Puff concentration"),
        (axes[1, 1], average, "Time-averaged Puff concentration"),
    ):
        image = ax.imshow(
            field,
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            origin="lower",
            cmap="YlGnBu_r",
            vmin=0.0,
            vmax=concentration_vmax,
            interpolation="bicubic",
        )
        fig.colorbar(image, ax=ax, label="concentration (a.u.)")
        ax.scatter([plume.source_x], [plume.source_y], marker="*", s=120, color="#f2c53d", edgecolor="black")
        ax.set_title(title)

    ax = axes[1, 2]
    wind_x = math.cos(result.direction_rad)
    wind_y = math.sin(result.direction_rad)
    cross_x = -wind_y
    cross_y = wind_x
    offsets = np.linspace(-1.5, 1.5, 121)
    for distance, color in ((0.2, "#c0392b"), (0.7, "#d68910"), (1.2, "#2471a3")):
        px = plume.source_x + distance * wind_x + offsets * cross_x
        py = plume.source_y + distance * wind_y + offsets * cross_y
        values = np.asarray(
            [plume.concentration(float(x), float(y), add_noise=False) for x, y in zip(px, py)]
        )
        ax.plot(offsets, values, color=color, label=f"downwind {distance:.1f}m")
    ax.set_title("Concentration cross-sections")
    ax.set_xlabel("crosswind offset (m)")
    ax.set_ylabel("concentration (a.u.)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)

    for ax in axes.flat[:5]:
        _draw_obstacles(ax, obstacles)
        ax.set_xlim(plume.world_min, plume.world_max)
        ax.set_ylim(plume.world_min, plume.world_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")

    fig.suptitle(
        "Obstacle-aware wind and intermittent Puff plume\n"
        f"requested={math.degrees(plume.wind_direction_base):+.1f}°, "
        f"LBM={math.degrees(result.direction_rad):+.1f}°, "
        f"iterations={result.iterations}, residual={result.residual:.3g}",
        fontsize=13,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)

    report = {
        "reset_seconds": reset_seconds,
        "sample_count": sample_count,
        "puff_count": len(plume.puffs),
        "max_speed_m_s": float(np.max(speed)),
        "mean_concentration": float(np.mean(average)),
        "max_concentration": float(np.max(average)),
        "lbm": plume.obstacle_flow.metadata(),
    }
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output}")
    print(f"wrote {args.json_output}")
    return report


def main() -> None:
    args = parse_args()
    run_deterministic_checks()
    report = generate_diagnostic(args)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
