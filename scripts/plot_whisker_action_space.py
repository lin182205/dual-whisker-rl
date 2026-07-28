"""Plot the discretized left/right whisker action space."""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_whisker_rl.paths import project_path


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config_path = project_path("configs/default.yaml")
    output_path = project_path("results/figures/whisker_action_space.png")
    assert config_path is not None and output_path is not None
    config = load_config(config_path)
    sector_count = int(config.get("whisker_sector_count", 10))
    save_whisker_action_space(sector_count, output_path)
    print(f"saved_figure={output_path}")


def save_whisker_action_space(sector_count: int, path: Path) -> None:
    radius = 1.0
    sector_width = 180.0 / sector_count
    left_colors = plt.cm.Blues(np.linspace(0.35, 0.85, sector_count))
    right_colors = plt.cm.Oranges(np.linspace(0.35, 0.85, sector_count))

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    ax.set_aspect("equal")
    ax.set_xlim(-1.25, 1.45)
    ax.set_ylim(-1.25, 1.25)
    ax.axis("off")

    for idx in range(sector_count):
        theta1 = idx * sector_width
        theta2 = (idx + 1) * sector_width
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                radius,
                theta1,
                theta2,
                width=0.42,
                facecolor=left_colors[idx],
                edgecolor="white",
                linewidth=1.5,
                alpha=0.95,
            )
        )
        center = np.deg2rad(theta1 + 0.5 * sector_width)
        ax.text(
            0.78 * np.cos(center),
            0.78 * np.sin(center),
            str(idx),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

        r_theta1 = -theta2
        r_theta2 = -theta1
        ax.add_patch(
            Wedge(
                (0.0, 0.0),
                radius,
                r_theta1,
                r_theta2,
                width=0.42,
                facecolor=right_colors[idx],
                edgecolor="white",
                linewidth=1.5,
                alpha=0.95,
            )
        )
        r_center = np.deg2rad(-(theta1 + 0.5 * sector_width))
        ax.text(
            0.78 * np.cos(r_center),
            0.78 * np.sin(r_center),
            str(idx),
            ha="center",
            va="center",
            fontsize=10,
            color="white",
            weight="bold",
        )

    ax.add_patch(Circle((0.0, 0.0), 0.13, facecolor="#2b2b2b", edgecolor="white", linewidth=1.5))
    ax.arrow(
        0.0,
        0.0,
        1.14,
        0.0,
        length_includes_head=True,
        head_width=0.055,
        head_length=0.08,
        linewidth=2.0,
        color="#222222",
    )
    ax.plot([-1.08, 1.08], [0.0, 0.0], color="#555555", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.text(1.2, 0.0, "robot heading", ha="left", va="center", fontsize=11, color="#222222")
    ax.text(-0.98, 0.18, "left whisker\n10 sectors", ha="center", va="center", fontsize=12, color="#1f4e79")
    ax.text(-0.98, -0.18, "right whisker\n10 sectors", ha="center", va="center", fontsize=12, color="#9a4f00")
    ax.text(0.0, 1.16, "Whisker Action Space Discretization", ha="center", va="center", fontsize=16, weight="bold")
    ax.text(
        0.0,
        -1.13,
        "Policy action = [move_action, left_sector, right_sector]   |   MultiDiscrete([6, 10, 10])",
        ha="center",
        va="center",
        fontsize=11,
        color="#333333",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
