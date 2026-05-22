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
