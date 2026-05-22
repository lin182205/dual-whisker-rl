"""Shared plotting helpers for result figures."""

from __future__ import annotations

from typing import Any


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
