"""Analyze CLS-to-history attention around mobile whisker events.

Attention weights show where the Transformer attends, but they are not a
causal explanation of the policy. Wind-shift events are privileged simulation
diagnostics: wind direction is not part of the deployable policy observation.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.agents import TransformerHistoryExtractor
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.paths import portable_path
from dual_whisker_rl.paths import resolve_path_args
from dual_whisker_rl.visualization import save_matplotlib_animation
from train_whisker_only_ppo import load_config
from visualize_whisker_only_env import build_plume_colormap
from visualize_whisker_only_env import draw_robot_heading_marker
from visualize_whisker_only_env import draw_whisker_artists


EVENT_ORDER = (
    "odor_detected",
    "rapid_rise",
    "left_right_reversal",
    "odor_lost",
    "wind_shift",
    "whisker_swing_complete",
)
EVENT_LABELS = {
    "odor_detected": "Odor detected",
    "rapid_rise": "Rapid concentration rise",
    "left_right_reversal": "Left/right reversal",
    "odor_lost": "Odor lost",
    "wind_shift": "Wind shift (privileged diagnostic)",
    "whisker_swing_complete": "Whisker swing complete",
}
EVENT_COLORS = {
    "odor_detected": "#2ca02c",
    "rapid_rise": "#bcbd22",
    "left_right_reversal": "#9467bd",
    "odor_lost": "#d62728",
    "wind_shift": "#17becf",
    "whisker_swing_complete": "#ff7f0e",
}
ATTENTION_LIMITATION = (
    "Attention weights indicate model focus, not causal feature importance."
)


@dataclass(frozen=True)
class EventThresholds:
    rise: float
    reversal_deadband: float
    wind_change_rad: float
    swing_tolerance_rad: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("results/models/mobile_whisker_transformer_ppo.zip"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/mobile_whisker.yaml"))
    parser.add_argument("--seed", type=int, default=10_000)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--resolution", type=int, default=80)
    parser.add_argument("--animation-stride", type=int, default=2)
    parser.add_argument("--animation-fps", type=int, default=6)
    parser.add_argument("--event-context", type=int, default=2)
    parser.add_argument("--rise-threshold", type=float, default=0.05)
    parser.add_argument("--reversal-deadband", type=float, default=0.02)
    parser.add_argument("--wind-change-deg", type=float, default=15.0)
    parser.add_argument("--swing-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--domain-randomization", action="store_true")
    args = parser.parse_args()

    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.resolution < 2:
        parser.error("--resolution must be at least 2")
    if args.animation_stride < 1:
        parser.error("--animation-stride must be at least 1")
    if args.animation_fps < 1:
        parser.error("--animation-fps must be at least 1")
    if args.event_context < 0:
        parser.error("--event-context must be non-negative")
    if args.rise_threshold < 0.0:
        parser.error("--rise-threshold must be non-negative")
    if args.reversal_deadband < 0.0:
        parser.error("--reversal-deadband must be non-negative")
    if args.wind_change_deg < 0.0:
        parser.error("--wind-change-deg must be non-negative")
    if args.swing_tolerance_deg < 0.0:
        parser.error("--swing-tolerance-deg must be non-negative")
    if args.output_dir is None:
        args.output_dir = Path("results/attention") / args.model_path.stem
    return resolve_path_args(args, "model_path", "config", "output_dir")


def wrap_angle(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def sector_target(sector: int, sector_count: int, side: str) -> float:
    angle = ((int(sector) + 0.5) / int(sector_count)) * math.pi
    return float(angle if side == "left" else -angle)


def state_diagnostics(env: MobileWhiskerPuffEnv, step: int) -> dict[str, Any]:
    whisker_state = env.whiskers.state(env.robot_state)
    return {
        "step": int(step),
        "x": float(env.robot_state.x),
        "y": float(env.robot_state.y),
        "heading": float(env.robot_state.heading),
        "left": float(env.left_sensor.value),
        "right": float(env.right_sensor.value),
        "left_angle": float(whisker_state.left_angle),
        "right_angle": float(whisker_state.right_angle),
        "left_point": tuple(float(value) for value in whisker_state.left_point),
        "right_point": tuple(float(value) for value in whisker_state.right_point),
        "wind_direction": float(env.plume.wind_direction),
        "distance": float(env._distance_to_source()),
    }


def detect_transition_events(
    previous: dict[str, Any],
    current: dict[str, Any],
    action: np.ndarray,
    *,
    hit_threshold: float,
    thresholds: EventThresholds,
    sector_count: int,
) -> list[dict[str, Any]]:
    """Detect events on a transition and attach them to the new state."""
    previous_peak = max(float(previous["left"]), float(previous["right"]))
    current_peak = max(float(current["left"]), float(current["right"]))
    previous_difference = float(previous["left"] - previous["right"])
    current_difference = float(current["left"] - current["right"])
    rise = current_peak - previous_peak
    wind_change = wrap_angle(
        float(current["wind_direction"] - previous["wind_direction"])
    )

    event_names: list[str] = []
    if previous_peak < hit_threshold <= current_peak:
        event_names.append("odor_detected")
    if rise >= thresholds.rise:
        event_names.append("rapid_rise")
    if (
        previous_difference >= thresholds.reversal_deadband
        and current_difference <= -thresholds.reversal_deadband
    ) or (
        previous_difference <= -thresholds.reversal_deadband
        and current_difference >= thresholds.reversal_deadband
    ):
        event_names.append("left_right_reversal")
    if previous_peak >= hit_threshold > current_peak:
        event_names.append("odor_lost")
    if abs(wind_change) >= thresholds.wind_change_rad:
        event_names.append("wind_shift")

    completed_sides: list[str] = []
    for action_index, angle_key, side in (
        (1, "left_angle", "left"),
        (2, "right_angle", "right"),
    ):
        target = sector_target(int(action[action_index]), sector_count, side)
        previous_error = abs(wrap_angle(float(previous[angle_key]) - target))
        current_error = abs(wrap_angle(float(current[angle_key]) - target))
        if (
            previous_error > thresholds.swing_tolerance_rad
            and current_error <= thresholds.swing_tolerance_rad
        ):
            completed_sides.append(side)
    if completed_sides:
        event_names.append("whisker_swing_complete")

    common = {
        "step": int(current["step"]),
        "left": float(current["left"]),
        "right": float(current["right"]),
        "left_right_difference": current_difference,
        "peak_concentration": current_peak,
        "concentration_rise": rise,
        "wind_direction_deg": math.degrees(float(current["wind_direction"])),
        "wind_change_deg": math.degrees(wind_change),
        "left_angle_deg": math.degrees(float(current["left_angle"])),
        "right_angle_deg": math.degrees(float(current["right_angle"])),
        "action": [int(value) for value in action],
        "swing_sides": completed_sides,
    }
    return [{"event": name, **common} for name in event_names]


def normalized_history_attention(full_attention: np.ndarray) -> np.ndarray:
    """Remove the CLS key and normalize attention over history positions."""
    history_attention = np.asarray(full_attention, dtype=np.float32)[..., 1:]
    denominator = np.sum(history_attention, axis=-1, keepdims=True)
    return history_attention / np.maximum(denominator, 1e-12)


def extract_attention(
    extractor: TransformerHistoryExtractor,
    stacked_observation: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.as_tensor(
        stacked_observation,
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)
    with torch.no_grad():
        _, full_attention = extractor.forward_with_attention(tensor)
    raw = full_attention[0].detach().cpu().numpy().astype(np.float32)
    return raw, normalized_history_attention(raw)


def capture_attention_rollout(
    env: MobileWhiskerPuffEnv,
    model: PPO,
    extractor: TransformerHistoryExtractor,
    *,
    steps: int,
    resolution: int,
    seed: int,
    deterministic: bool,
    thresholds: EventThresholds,
) -> dict[str, Any]:
    base_observation, _ = env.reset(seed=seed)
    history: deque[np.ndarray] = deque(
        [np.asarray(base_observation, dtype=np.float32).copy()] * extractor.history_length,
        maxlen=extractor.history_length,
    )
    xs, ys, initial_grid = env.concentration_grid(resolution=resolution)

    states: list[dict[str, Any]] = []
    stacked_observations: list[np.ndarray] = []
    raw_attentions: list[np.ndarray] = []
    normalized_attentions: list[np.ndarray] = []
    grids: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    events: list[dict[str, Any]] = []

    pending_events: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    state_step = 0
    grid = initial_grid

    while True:
        stacked = np.concatenate(tuple(history)).astype(np.float32, copy=False)
        raw_attention, normalized_attention = extract_attention(
            extractor,
            stacked,
            model.device,
        )
        state = state_diagnostics(env, state_step)
        state["events"] = [row["event"] for row in pending_events]
        states.append(state)
        stacked_observations.append(stacked.copy())
        raw_attentions.append(raw_attention)
        normalized_attentions.append(normalized_attention)
        grids.append(np.asarray(grid, dtype=np.float32).copy())

        mean_attention = normalized_attention.mean(axis=(0, 1))
        cls_self_attention = float(raw_attention[..., 0].mean())
        for event in pending_events:
            event["mean_cls_history_attention"] = [
                float(value) for value in mean_attention
            ]
            event["mean_cls_self_attention"] = cls_self_attention
            events.append(event)

        if terminated or truncated or state_step >= steps:
            actions.append(np.full(3, -1, dtype=np.int64))
            break

        prediction, _ = model.predict(stacked, deterministic=deterministic)
        action = np.asarray(prediction, dtype=np.int64).reshape(-1)[:3]
        actions.append(action.copy())
        previous = state
        observation, _, terminated, truncated, _ = env.step(action)
        history.append(np.asarray(observation, dtype=np.float32))
        state_step += 1
        current = state_diagnostics(env, state_step)
        pending_events = detect_transition_events(
            previous,
            current,
            action,
            hit_threshold=float(env.hit_threshold),
            thresholds=thresholds,
            sector_count=int(env.whiskers.sector_count),
        )
        _, _, grid = env.concentration_grid(resolution=resolution)

    event_codes = np.zeros((len(states), len(EVENT_ORDER)), dtype=np.uint8)
    event_index = {name: index for index, name in enumerate(EVENT_ORDER)}
    for event in events:
        event_codes[int(event["step"]), event_index[event["event"]]] = 1

    return {
        "xs": np.asarray(xs, dtype=np.float32),
        "ys": np.asarray(ys, dtype=np.float32),
        "states": states,
        "stacked_observations": np.asarray(stacked_observations, dtype=np.float32),
        "raw_attention": np.asarray(raw_attentions, dtype=np.float32),
        "normalized_attention": np.asarray(normalized_attentions, dtype=np.float32),
        "grids": np.asarray(grids, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "events": events,
        "event_codes": event_codes,
        "source": (float(env.plume.source_x), float(env.plume.source_y)),
        "hit_threshold": float(env.hit_threshold),
        "termination": (
            "reached_goal"
            if states[-1]["distance"] <= env.goal_radius
            else "out_of_bounds" if terminated else "truncated" if truncated else "step_limit"
        ),
    }


def event_steps(events: list[dict[str, Any]], event_name: str) -> list[int]:
    return sorted(
        {int(event["step"]) for event in events if event["event"] == event_name}
    )


def aligned_event_attention(
    mean_attention: np.ndarray,
    occurrences: list[int],
    context: int,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.arange(-context, context + 1, dtype=np.int64)
    rows: list[np.ndarray] = []
    counts: list[int] = []
    for offset in offsets:
        indices = [step + int(offset) for step in occurrences]
        indices = [index for index in indices if 0 <= index < mean_attention.shape[0]]
        counts.append(len(indices))
        if indices:
            rows.append(np.mean(mean_attention[indices], axis=0))
        else:
            rows.append(np.full(mean_attention.shape[1], np.nan, dtype=np.float32))
    return np.asarray(rows, dtype=np.float32), np.asarray(counts, dtype=np.int64)


def save_timeline_plot(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    states = data["states"]
    steps = np.arange(len(states))
    left = np.asarray([row["left"] for row in states])
    right = np.asarray([row["right"] for row in states])
    wind = np.degrees([row["wind_direction"] for row in states])
    left_angle = np.degrees([row["left_angle"] for row in states])
    right_angle = np.degrees([row["right_angle"] for row in states])
    mean_attention = data["normalized_attention"].mean(axis=(1, 2))
    history_length = mean_attention.shape[1]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(13, 9),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.6]},
    )
    axes[0].plot(steps, left, color="#4c78a8", label="left sensor")
    axes[0].plot(steps, right, color="#f58518", label="right sensor")
    axes[0].axhline(
        data["hit_threshold"], color="#777777", linestyle="--", label="hit threshold"
    )
    axes[0].set_ylabel("concentration")
    axes[0].legend(loc="upper right", ncol=3, fontsize=8)

    axes[1].plot(steps, wind, color="#17becf", label="wind (privileged)")
    axes[1].plot(steps, left_angle, color="#4c78a8", alpha=0.7, label="left angle")
    axes[1].plot(steps, right_angle, color="#f58518", alpha=0.7, label="right angle")
    axes[1].set_ylabel("angle (deg)")
    axes[1].legend(loc="upper right", ncol=3, fontsize=8)

    image = axes[2].imshow(
        mean_attention.T,
        origin="lower",
        aspect="auto",
        extent=[-0.5, len(states) - 0.5, -(history_length - 1) - 0.5, 0.5],
        cmap="magma",
        vmin=0.0,
    )
    axes[2].set_ylabel("history lag")
    axes[2].set_xlabel("state step")
    fig.colorbar(image, ax=axes[2], pad=0.015).set_label("normalized CLS attention")

    event_handles: list[Line2D] = []
    for event_name in EVENT_ORDER:
        occurrences = event_steps(data["events"], event_name)
        if occurrences:
            event_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=EVENT_COLORS[event_name],
                    linewidth=1.5,
                    label=EVENT_LABELS[event_name],
                )
            )
        for index, step in enumerate(occurrences):
            for axis in axes:
                axis.axvline(
                    step,
                    color=EVENT_COLORS[event_name],
                    alpha=0.32,
                    linewidth=0.9,
                    label=None,
                )
    if event_handles:
        fig.legend(
            handles=event_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=3,
            fontsize=7,
        )
    fig.suptitle("CLS attention timeline (layer/head mean)", y=0.995)
    fig.text(0.5, 0.01, ATTENTION_LIMITATION, ha="center", fontsize=8, color="#555555")
    fig.tight_layout(rect=(0.0, 0.035, 0.98, 0.91))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_event_summary_plot(data: dict[str, Any], path: Path, context: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_attention = data["normalized_attention"].mean(axis=(1, 2))
    history_length = mean_attention.shape[1]
    aligned: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    finite_max = 0.0
    for event_name in EVENT_ORDER:
        result = aligned_event_attention(
            mean_attention,
            event_steps(data["events"], event_name),
            context,
        )
        aligned[event_name] = result
        values = result[0][np.isfinite(result[0])]
        if values.size:
            finite_max = max(finite_max, float(np.max(values)))

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    image = None
    for axis, event_name in zip(axes.flat, EVENT_ORDER):
        matrix, counts = aligned[event_name]
        occurrences = event_steps(data["events"], event_name)
        if not occurrences:
            axis.text(0.5, 0.5, "no events", ha="center", va="center", transform=axis.transAxes)
            axis.set_axis_off()
            axis.set_title(EVENT_LABELS[event_name])
            continue
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[-(history_length - 1) - 0.5, 0.5, -context - 0.5, context + 0.5],
            cmap="magma",
            vmin=0.0,
            vmax=finite_max or None,
        )
        axis.set_title(f"{EVENT_LABELS[event_name]}  n={len(occurrences)}")
        axis.set_xlabel("attended history lag")
        axis.set_ylabel("decision offset from event")
        axis.set_yticks(np.arange(-context, context + 1))
        axis.axhline(0.0, color="white", linewidth=0.8, alpha=0.8)
        axis.text(
            0.01,
            0.02,
            "valid n by row: " + ",".join(str(int(value)) for value in counts),
            transform=axis.transAxes,
            fontsize=7,
            color="#222222",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 1.5},
        )
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75).set_label(
            "normalized CLS attention"
        )
    fig.suptitle(
        "Event-aligned CLS attention (layer/head mean)\n"
        + ATTENTION_LIMITATION,
        fontsize=13,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_per_head_plots(data: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    attention = data["normalized_attention"]
    _, n_layers, n_heads, history_length = attention.shape
    lags = np.arange(-(history_length - 1), 1)
    for event_name in EVENT_ORDER:
        occurrences = event_steps(data["events"], event_name)
        fig, axes = plt.subplots(
            n_layers,
            n_heads,
            figsize=(3.2 * n_heads, 2.7 * n_layers),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        if occurrences:
            averaged = np.mean(attention[occurrences], axis=0)
            ymax = max(1.0 / history_length, float(np.max(averaged))) * 1.1
        else:
            averaged = np.zeros((n_layers, n_heads, history_length), dtype=np.float32)
            ymax = 1.0 / history_length
        for layer_index in range(n_layers):
            for head_index in range(n_heads):
                axis = axes[layer_index, head_index]
                if occurrences:
                    axis.plot(lags, averaged[layer_index, head_index], color="#6f2c91")
                    axis.fill_between(
                        lags,
                        0.0,
                        averaged[layer_index, head_index],
                        color="#9467bd",
                        alpha=0.25,
                    )
                else:
                    axis.text(0.5, 0.5, "no events", ha="center", va="center", transform=axis.transAxes)
                axis.set_ylim(0.0, ymax)
                axis.set_title(f"layer {layer_index + 1}, head {head_index + 1}")
                axis.grid(alpha=0.2)
        for axis in axes[-1, :]:
            axis.set_xlabel("history lag")
        for axis in axes[:, 0]:
            axis.set_ylabel("attention")
        fig.suptitle(f"{EVENT_LABELS[event_name]}  n={len(occurrences)}")
        fig.text(0.5, 0.01, ATTENTION_LIMITATION, ha="center", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.95))
        fig.savefig(output_dir / f"attention_heads_{event_name}.png", dpi=170)
        plt.close(fig)


def save_attention_animation(
    data: dict[str, Any],
    path: Path,
    *,
    stride: int,
    fps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    states = data["states"]
    trajectory = np.asarray([[row["x"], row["y"]] for row in states])
    headings = np.asarray([row["heading"] for row in states])
    left_points = np.asarray([row["left_point"] for row in states])
    right_points = np.asarray([row["right_point"] for row in states])
    left = np.asarray([row["left"] for row in states])
    right = np.asarray([row["right"] for row in states])
    grids = data["grids"]
    mean_attention = data["normalized_attention"].mean(axis=(1, 2))
    history_length = mean_attention.shape[1]
    lags = np.arange(-(history_length - 1), 1)
    frame_ids = list(range(0, len(states), max(1, stride)))
    if frame_ids[-1] != len(states) - 1:
        frame_ids.append(len(states) - 1)

    fig = plt.figure(figsize=(13, 6.8))
    grid_spec = fig.add_gridspec(2, 2, width_ratios=(1.25, 1.0), hspace=0.35)
    map_axis = fig.add_subplot(grid_spec[:, 0])
    sensor_axis = fig.add_subplot(grid_spec[0, 1])
    attention_axis = fig.add_subplot(grid_spec[1, 1])

    cmap = build_plume_colormap()
    vmax = max(0.05, float(np.percentile(grids, 99.2)))
    heatmap = map_axis.imshow(
        grids[0],
        extent=[data["xs"].min(), data["xs"].max(), data["ys"].min(), data["ys"].max()],
        origin="lower",
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )
    map_axis.scatter(*data["source"], marker="*", s=180, color="#f2c53d", edgecolor="#222222", zorder=7)
    map_axis.set_xlim(float(data["xs"].min()), float(data["xs"].max()))
    map_axis.set_ylim(float(data["ys"].min()), float(data["ys"].max()))
    map_axis.set_xlabel("x (m)")
    map_axis.set_ylabel("y (m)")
    trace, = map_axis.plot([], [], color="#4a4a4a", linewidth=1.6)
    robot = map_axis.scatter([trajectory[0, 0]], [trajectory[0, 1]], color="#222222", s=70, zorder=8)
    left_line, right_line, left_tip, right_tip = draw_whisker_artists(
        map_axis, trajectory, left_points, right_points, 0
    )
    heading_start, heading_arrow = draw_robot_heading_marker(map_axis, trajectory[0], headings[0])
    info_text = map_axis.text(
        0.02,
        0.98,
        "",
        transform=map_axis.transAxes,
        va="top",
        family="monospace",
        fontsize=8.5,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.75},
    )
    fig.colorbar(heatmap, ax=map_axis, pad=0.02).set_label("gas concentration")

    sensor_left, = sensor_axis.plot([], [], color="#4c78a8", label="left")
    sensor_right, = sensor_axis.plot([], [], color="#f58518", label="right")
    sensor_axis.axhline(data["hit_threshold"], color="#777777", linestyle="--", label="hit")
    sensor_axis.set_ylabel("sensor concentration")
    sensor_axis.set_xlabel("state step")
    sensor_axis.legend(fontsize=8)
    sensor_axis.set_ylim(0.0, max(data["hit_threshold"] * 1.5, float(max(left.max(), right.max())) * 1.1))

    bars = attention_axis.bar(lags, mean_attention[0], color="#8c4aa5")
    attention_axis.set_xlabel("history lag (0 = current observation)")
    attention_axis.set_ylabel("CLS attention")
    attention_axis.set_ylim(0.0, max(1.0 / history_length, float(mean_attention.max())) * 1.15)
    attention_axis.grid(axis="y", alpha=0.2)

    def update(frame_index: int):
        state_index = frame_ids[frame_index]
        state = states[state_index]
        heatmap.set_data(grids[state_index])
        trace.set_data(trajectory[: state_index + 1, 0], trajectory[: state_index + 1, 1])
        robot.set_offsets(trajectory[state_index])
        point = trajectory[state_index]
        heading_end = (
            point[0] + 0.07 * math.cos(headings[state_index]),
            point[1] + 0.07 * math.sin(headings[state_index]),
        )
        heading_start.set_positions(tuple(point), heading_end)
        heading_arrow.set_positions(tuple(point), heading_end)
        left_point = left_points[state_index]
        right_point = right_points[state_index]
        left_line.set_data([point[0], left_point[0]], [point[1], left_point[1]])
        right_line.set_data([point[0], right_point[0]], [point[1], right_point[1]])
        left_tip.set_offsets(left_point)
        right_tip.set_offsets(right_point)

        history_start = max(0, state_index - history_length + 1)
        sensor_steps = np.arange(history_start, state_index + 1)
        sensor_left.set_data(sensor_steps, left[history_start : state_index + 1])
        sensor_right.set_data(sensor_steps, right[history_start : state_index + 1])
        sensor_axis.set_xlim(history_start - 0.5, max(history_start + 1, state_index) + 0.5)
        for bar, value in zip(bars, mean_attention[state_index]):
            bar.set_height(float(value))

        event_text = ", ".join(EVENT_LABELS[name] for name in state["events"]) or "none"
        info_text.set_text(
            f"step {state_index}  dist {state['distance']:.3f} m\n"
            f"L {state['left']:.3f}  R {state['right']:.3f}  "
            f"L-R {state['left'] - state['right']:+.3f}\n"
            f"events: {event_text}"
        )
        attention_axis.set_title(f"CLS → history | events: {event_text}", fontsize=9)
        return (
            heatmap,
            trace,
            robot,
            left_line,
            right_line,
            left_tip,
            right_tip,
            heading_start,
            heading_arrow,
            sensor_left,
            sensor_right,
            info_text,
            *bars,
        )

    fig.suptitle("Mobile Transformer attention rollout")
    fig.text(0.5, 0.005, ATTENTION_LIMITATION, ha="center", fontsize=8, color="#555555")
    rollout_animation = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_ids),
        interval=1000 / fps,
        blit=False,
        repeat=False,
    )
    save_matplotlib_animation(rollout_animation, path, fps=fps, dpi=110)
    plt.close(fig)


def save_data_outputs(
    data: dict[str, Any],
    output_dir: Path,
    *,
    args: argparse.Namespace,
    extractor: TransformerHistoryExtractor,
    thresholds: EventThresholds,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_counts = {
        name: len(event_steps(data["events"], name)) for name in EVENT_ORDER
    }
    payload = {
        "metadata": {
            "model_path": portable_path(args.model_path),
            "config_path": portable_path(args.config),
            "seed": int(args.seed),
            "states": len(data["states"]),
            "history_length": int(extractor.history_length),
            "layers": len(extractor.encoder.layers),
            "heads": int(data["raw_attention"].shape[2]),
            "event_context": int(args.event_context),
            "thresholds": {
                "hit": float(data["hit_threshold"]),
                "rise": thresholds.rise,
                "reversal_deadband": thresholds.reversal_deadband,
                "wind_change_deg": math.degrees(thresholds.wind_change_rad),
                "swing_tolerance_deg": math.degrees(thresholds.swing_tolerance_rad),
            },
            "attention_definition": (
                "CLS query to oldest..current history keys; visualization excludes "
                "the CLS key and renormalizes history weights to sum to one."
            ),
            "wind_event_note": (
                "Wind direction is a privileged simulation diagnostic and is not "
                "part of the deployable policy observation."
            ),
            "interpretation_limit": ATTENTION_LIMITATION,
        },
        "event_counts": event_counts,
        "events": data["events"],
    }
    (output_dir / "events.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    states = data["states"]
    np.savez_compressed(
        output_dir / "attention.npz",
        raw_cls_attention=data["raw_attention"],
        normalized_cls_history_attention=data["normalized_attention"],
        stacked_observations=data["stacked_observations"],
        actions=data["actions"],
        sensor_values=np.asarray([[row["left"], row["right"]] for row in states], dtype=np.float32),
        wind_direction=np.asarray([row["wind_direction"] for row in states], dtype=np.float32),
        whisker_angles=np.asarray([[row["left_angle"], row["right_angle"]] for row in states], dtype=np.float32),
        robot_pose=np.asarray([[row["x"], row["y"], row["heading"]] for row in states], dtype=np.float32),
        event_codes=data["event_codes"],
        event_names=np.asarray(EVENT_ORDER),
    )


def main() -> None:
    args = parse_args()
    if not args.model_path.exists():
        raise FileNotFoundError(f"model does not exist: {args.model_path}")

    config = load_config(args.config)
    if args.domain_randomization:
        config["domain_randomization"] = True
    model = PPO.load(str(args.model_path))
    extractor = model.policy.features_extractor
    if not isinstance(extractor, TransformerHistoryExtractor):
        raise TypeError(
            "Attention analysis requires TransformerHistoryExtractor; "
            f"loaded {type(extractor).__name__}"
        )
    model.policy.set_training_mode(False)
    extractor.eval()

    env = MobileWhiskerPuffEnv(config)
    base_dim = int(env.observation_space.shape[0])
    expected_dim = int(extractor.history_length * base_dim)
    if tuple(model.observation_space.shape) != (expected_dim,):
        env.close()
        raise ValueError(
            f"model observation shape {model.observation_space.shape} does not match "
            f"history {extractor.history_length} * base dim {base_dim}"
        )

    thresholds = EventThresholds(
        rise=float(args.rise_threshold),
        reversal_deadband=float(args.reversal_deadband),
        wind_change_rad=math.radians(float(args.wind_change_deg)),
        swing_tolerance_rad=math.radians(float(args.swing_tolerance_deg)),
    )
    try:
        data = capture_attention_rollout(
            env,
            model,
            extractor,
            steps=args.steps,
            resolution=args.resolution,
            seed=args.seed,
            deterministic=not args.stochastic,
            thresholds=thresholds,
        )
    finally:
        env.close()

    output_dir = args.output_dir
    save_data_outputs(
        data,
        output_dir,
        args=args,
        extractor=extractor,
        thresholds=thresholds,
    )
    save_timeline_plot(data, output_dir / "attention_timeline.png")
    save_event_summary_plot(
        data,
        output_dir / "attention_events.png",
        args.event_context,
    )
    save_per_head_plots(data, output_dir)
    save_attention_animation(
        data,
        output_dir / "attention_rollout.mp4",
        stride=args.animation_stride,
        fps=args.animation_fps,
    )

    counts = {name: len(event_steps(data["events"], name)) for name in EVENT_ORDER}
    print(f"model={args.model_path}")
    print(f"states={len(data['states'])} attention_shape={data['raw_attention'].shape}")
    print("event_counts=" + json.dumps(counts, ensure_ascii=False))
    print(f"saved_output_dir={output_dir}")


if __name__ == "__main__":
    main()
