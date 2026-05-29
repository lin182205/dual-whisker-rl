"""Shared policy evaluation utilities."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def evaluate_policy(
    model: Any,
    env_factory: Callable[[], Any],
    *,
    episodes: int,
    seed: int,
    deterministic: bool = True,
) -> tuple[dict[str, Any], list[list[dict[str, Any]]]]:
    """Evaluate a Stable-Baselines-style model on repeated episodes."""

    returns: list[float] = []
    lengths: list[int] = []
    successes: list[float] = []
    final_distances: list[float] = []
    path_lengths: list[float] = []
    odor_hits: list[float] = []
    reacquisition_times: list[int] = []
    mean_raw_concentrations: list[float] = []
    mean_sensor_concentrations: list[float] = []
    mean_left_right_contrasts: list[float] = []
    plume_contact_ratios: list[float] = []
    odor_loss_durations: list[int] = []
    left_sector_counts: dict[int, int] = {}
    right_sector_counts: dict[int, int] = {}
    sector_pair_counts: dict[str, int] = {}
    trajectories: list[list[dict[str, Any]]] = []

    for idx in range(episodes):
        env = env_factory()
        obs, info = env.reset(seed=seed + idx)
        terminated = False
        truncated = False
        episode_return = 0.0
        last_x = info["robot_state"].x
        last_y = info["robot_state"].y
        path_length = 0.0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(_format_action(action))
            episode_return += float(reward)
            state = info["robot_state"]
            path_length += float(np.hypot(state.x - last_x, state.y - last_y))
            last_x = state.x
            last_y = state.y

        trajectory = list(env.trajectory)
        returns.append(episode_return)
        lengths.append(len(trajectory))
        successes.append(1.0 if info["distance_to_source"] <= env.goal_radius else 0.0)
        final_distances.append(float(info["distance_to_source"]))
        path_lengths.append(path_length)
        hit_count, episode_reacquisitions = _odor_hit_stats(trajectory, env.hit_threshold)
        whisker_stats = _whisker_info_stats(trajectory, env.hit_threshold)
        odor_hits.append(float(hit_count))
        reacquisition_times.extend(episode_reacquisitions)
        mean_raw_concentrations.append(whisker_stats["mean_raw_concentration"])
        mean_sensor_concentrations.append(whisker_stats["mean_sensor_concentration"])
        mean_left_right_contrasts.append(whisker_stats["mean_left_right_contrast"])
        plume_contact_ratios.append(whisker_stats["plume_contact_ratio"])
        odor_loss_durations.extend(whisker_stats["odor_loss_durations"])
        _merge_counts(left_sector_counts, whisker_stats["left_sector_counts"])
        _merge_counts(right_sector_counts, whisker_stats["right_sector_counts"])
        _merge_counts(sector_pair_counts, whisker_stats["sector_pair_counts"])
        trajectories.append(trajectory)
        env.close()

    return (
        {
            "episodes": episodes,
            "success_rate": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "mean_steps": float(np.mean(lengths)),
            "mean_final_distance": float(np.mean(final_distances)),
            "mean_path_length": float(np.mean(path_lengths)),
            "mean_odor_hits": float(np.mean(odor_hits)),
            "mean_reacquisition_time": (
                float(np.mean(reacquisition_times)) if reacquisition_times else None
            ),
            "reacquisition_events": len(reacquisition_times),
            "mean_raw_concentration": float(np.mean(mean_raw_concentrations)),
            "mean_sensor_concentration": float(np.mean(mean_sensor_concentrations)),
            "mean_left_right_contrast": float(np.mean(mean_left_right_contrasts)),
            "mean_plume_contact_ratio": float(np.mean(plume_contact_ratios)),
            "mean_odor_loss_duration": (
                float(np.mean(odor_loss_durations)) if odor_loss_durations else 0.0
            ),
            "odor_loss_events": len(odor_loss_durations),
            "left_sector_counts": _stringify_int_keys(left_sector_counts),
            "right_sector_counts": _stringify_int_keys(right_sector_counts),
            "sector_pair_counts": sector_pair_counts,
        },
        trajectories,
    )


def _odor_hit_stats(
    trajectory: list[dict[str, Any]],
    hit_threshold: float,
) -> tuple[int, list[int]]:
    hits = [
        max(float(row["left"]), float(row["right"])) >= hit_threshold
        for row in trajectory
    ]
    hit_count = int(sum(hits))
    reacquisition_times: list[int] = []
    has_seen_hit = False
    in_loss = False
    lost_steps = 0

    for hit in hits:
        if hit:
            has_seen_hit = True
            if in_loss:
                reacquisition_times.append(lost_steps)
                in_loss = False
                lost_steps = 0
        elif has_seen_hit:
            in_loss = True
            lost_steps += 1

    return hit_count, reacquisition_times


def _format_action(action: Any) -> Any:
    arr = np.asarray(action)
    if arr.size == 1:
        return int(arr.reshape(-1)[0])
    return arr.astype(np.int64)


def _whisker_info_stats(
    trajectory: list[dict[str, Any]],
    hit_threshold: float,
) -> dict[str, Any]:
    if not trajectory:
        return {
            "mean_raw_concentration": 0.0,
            "mean_sensor_concentration": 0.0,
            "mean_left_right_contrast": 0.0,
            "plume_contact_ratio": 0.0,
            "odor_loss_durations": [],
            "left_sector_counts": {},
            "right_sector_counts": {},
            "sector_pair_counts": {},
        }

    raw_means = [
        0.5 * (float(row["raw_left"]) + float(row["raw_right"]))
        for row in trajectory
    ]
    sensor_means = [
        0.5 * (float(row["left"]) + float(row["right"]))
        for row in trajectory
    ]
    contrasts = [
        abs(float(row["left"]) - float(row["right"]))
        for row in trajectory
    ]
    hits = [
        max(float(row["left"]), float(row["right"])) >= hit_threshold
        for row in trajectory
    ]

    left_counts: dict[int, int] = {}
    right_counts: dict[int, int] = {}
    pair_counts: dict[str, int] = {}
    for row in trajectory:
        left_sector = int(row.get("left_sector", -1))
        right_sector = int(row.get("right_sector", -1))
        left_counts[left_sector] = left_counts.get(left_sector, 0) + 1
        right_counts[right_sector] = right_counts.get(right_sector, 0) + 1
        pair_key = f"{left_sector},{right_sector}"
        pair_counts[pair_key] = pair_counts.get(pair_key, 0) + 1

    return {
        "mean_raw_concentration": float(np.mean(raw_means)),
        "mean_sensor_concentration": float(np.mean(sensor_means)),
        "mean_left_right_contrast": float(np.mean(contrasts)),
        "plume_contact_ratio": float(np.mean(hits)),
        "odor_loss_durations": _odor_loss_durations(hits),
        "left_sector_counts": left_counts,
        "right_sector_counts": right_counts,
        "sector_pair_counts": pair_counts,
    }


def _odor_loss_durations(hits: list[bool]) -> list[int]:
    durations: list[int] = []
    current = 0
    for hit in hits:
        if hit:
            if current > 0:
                durations.append(current)
                current = 0
        else:
            current += 1
    if current > 0:
        durations.append(current)
    return durations


def _merge_counts(target: dict[Any, int], source: dict[Any, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def _stringify_int_keys(counts: dict[int, int]) -> dict[str, int]:
    return {str(key): int(value) for key, value in sorted(counts.items())}
