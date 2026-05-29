"""Shared plotting helpers for result figures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ODOR_CMAP = "YlGnBu_r"


def draw_odor_field(
    ax: Any,
    xx: Any,
    yy: Any,
    zz: Any,
    *,
    alpha: float = 0.9,
    add_colorbar: bool = True,
    label: str = "odor concentration",
) -> Any:
    """Draw the odor field with a blue-low/yellow-high color scale."""

    field = ax.pcolormesh(
        xx,
        yy,
        zz,
        cmap=ODOR_CMAP,
        shading="gouraud",
        alpha=alpha,
    )
    if add_colorbar:
        fig = ax.figure
        cbar = fig.colorbar(field, ax=ax, pad=0.035)
        cbar.set_label(label)
    return field


def save_sensor_response(
    trajectory: list[dict[str, Any]],
    path: Path,
) -> None:
    """Save raw and filtered left/right whisker sensor readings."""

    left = np.array([row["left"] for row in trajectory])
    right = np.array([row["right"] for row in trajectory])
    raw_left = np.array([row["raw_left"] for row in trajectory])
    raw_right = np.array([row["raw_right"] for row in trajectory])
    steps = np.arange(len(left))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, raw_left, "--", color="#4c78a8", alpha=0.5, label="left raw")
    ax.plot(steps, raw_right, "--", color="#f58518", alpha=0.5, label="right raw")
    ax.plot(steps, left, color="#4c78a8", label="left sensor")
    ax.plot(steps, right, color="#f58518", label="right sensor")
    ax.set_xlabel("step")
    ax.set_ylabel("concentration")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_whisker_sector_stats(
    trajectory: list[dict[str, Any]],
    path: Path,
    *,
    sector_count: int,
) -> None:
    """Save left/right whisker sector histograms and pair heatmap."""

    left = np.array([int(row["left_sector"]) for row in trajectory], dtype=int)
    right = np.array([int(row["right_sector"]) for row in trajectory], dtype=int)
    bins = np.arange(sector_count + 1) - 0.5
    pair_counts = np.zeros((sector_count, sector_count), dtype=float)
    for left_sector, right_sector in zip(left, right):
        if 0 <= left_sector < sector_count and 0 <= right_sector < sector_count:
            pair_counts[left_sector, right_sector] += 1.0

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))

    axes[0].hist(left, bins=bins, color="#4c78a8", edgecolor="white")
    axes[0].set_title("left whisker")
    axes[0].set_xlabel("sector")
    axes[0].set_ylabel("count")
    axes[0].set_xticks(range(sector_count))

    axes[1].hist(right, bins=bins, color="#f58518", edgecolor="white")
    axes[1].set_title("right whisker")
    axes[1].set_xlabel("sector")
    axes[1].set_ylabel("count")
    axes[1].set_xticks(range(sector_count))

    image = axes[2].imshow(pair_counts, origin="lower", cmap="YlGnBu")
    axes[2].set_title("sector pair")
    axes[2].set_xlabel("right sector")
    axes[2].set_ylabel("left sector")
    axes[2].set_xticks(range(sector_count))
    axes[2].set_yticks(range(sector_count))
    cbar = fig.colorbar(image, ax=axes[2], pad=0.035)
    cbar.set_label("count")

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
