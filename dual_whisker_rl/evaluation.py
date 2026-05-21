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
            obs, reward, terminated, truncated, info = env.step(int(action))
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
        odor_hits.append(float(hit_count))
        reacquisition_times.extend(episode_reacquisitions)
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
