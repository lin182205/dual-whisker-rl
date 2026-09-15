"""仿真场地边界的统一解析工具。"""

from __future__ import annotations

import math
from typing import Any


def resolve_world_bounds(
    world_bounds: Any = None,
    world_half: Any = None,
    *,
    default_half: float = 0.5,
) -> tuple[float, float, float]:
    """解析对称方形场地的边界，返回 ``(world_min, world_max, world_half)``。

    新配置使用 ``world_bounds: [min, max]``；旧配置的 ``world_half`` 仍然支持。
    当前环境的几何、奖励归一化和场景分层都假定场地关于原点对称，因此显式
    边界也要求满足 ``min == -max``，避免只改一侧后产生隐蔽的尺度错误。
    """
    if world_bounds is None:
        try:
            half = float(default_half if world_half is None else world_half)
        except (TypeError, ValueError):
            raise ValueError("world_half must be a finite positive number") from None
        if not math.isfinite(half) or half <= 0.0:
            raise ValueError("world_half must be a finite positive number")
        return -half, half, half

    try:
        values = list(world_bounds)
    except TypeError:
        raise ValueError("world_bounds must contain exactly two numbers") from None
    if len(values) != 2:
        raise ValueError("world_bounds must contain exactly two numbers")
    try:
        world_min, world_max = (float(value) for value in values)
    except (TypeError, ValueError):
        raise ValueError("world_bounds must contain exactly two numbers") from None
    if not math.isfinite(world_min) or not math.isfinite(world_max):
        raise ValueError("world_bounds values must be finite")
    if world_min >= world_max:
        raise ValueError("world_bounds lower bound must be smaller than upper bound")
    if not math.isclose(world_min, -world_max, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("world_bounds must be symmetric around zero, e.g. [-2.5, 2.5]")
    half = 0.5 * (world_max - world_min)
    return world_min, world_max, half


def world_bounds_config(world_min: float, world_max: float) -> list[float]:
    """将运行时边界转成可写入 YAML/JSON 的规范列表。"""
    return [float(world_min), float(world_max)]
