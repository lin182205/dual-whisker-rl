"""纯 Python 二维激光雷达与障碍几何工具。"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def validate_obstacles(
    obstacles: Any,
    *,
    world_min: float,
    world_max: float,
) -> np.ndarray:
    """校验并复制 ``[xmin, xmax, ymin, ymax]`` 矩形障碍数组。"""
    if obstacles is None:
        return np.empty((0, 4), dtype=np.float32)

    array = np.asarray(obstacles, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError(
            "obstacles must have shape (N, 4) with rows "
            "[xmin, xmax, ymin, ymax]"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError("obstacles values must be finite")

    xmin = array[:, 0]
    xmax = array[:, 1]
    ymin = array[:, 2]
    ymax = array[:, 3]
    if np.any(xmin >= xmax) or np.any(ymin >= ymax):
        raise ValueError("each obstacle must have positive width and height")
    if (
        np.any(xmin < world_min)
        or np.any(xmax > world_max)
        or np.any(ymin < world_min)
        or np.any(ymax > world_max)
    ):
        raise ValueError(
            f"obstacles must stay inside world bounds [{world_min}, {world_max}]"
        )
    return array.copy()


def circle_intersects_obstacles(
    x: float,
    y: float,
    radius: float,
    obstacles: np.ndarray,
) -> bool:
    """判断圆形机器人是否与任一轴对齐矩形相交。"""
    radius_sq = float(radius) ** 2
    for xmin, xmax, ymin, ymax in obstacles:
        nearest_x = min(max(float(x), float(xmin)), float(xmax))
        nearest_y = min(max(float(y), float(ymin)), float(ymax))
        if (float(x) - nearest_x) ** 2 + (float(y) - nearest_y) ** 2 <= radius_sq:
            return True
    return False


class SimulatedLidar2D:
    """在二维矩形地图中进行解析式 ray-AABB 求交的前向雷达。"""

    def __init__(
        self,
        *,
        num_beams: int = 12,
        fov_deg: float = 180.0,
        max_range: float = 0.6,
        noise_std: float = 0.0,
    ) -> None:
        if isinstance(num_beams, (bool, np.bool_)) or int(num_beams) != num_beams:
            raise ValueError("lidar_num_beams must be an integer")
        self.num_beams = int(num_beams)
        self.fov_deg = float(fov_deg)
        self.max_range = float(max_range)
        self.noise_std = float(noise_std)

        if self.num_beams < 1:
            raise ValueError("lidar_num_beams must be at least 1")
        if not math.isfinite(self.fov_deg) or not (0.0 < self.fov_deg <= 360.0):
            raise ValueError("lidar_fov_deg must be in (0, 360]")
        if not math.isfinite(self.max_range) or self.max_range <= 0.0:
            raise ValueError("lidar_max_range must be finite and positive")
        if not math.isfinite(self.noise_std) or self.noise_std < 0.0:
            raise ValueError("lidar_noise_std must be finite and non-negative")

        fov_rad = math.radians(self.fov_deg)
        beam_width = fov_rad / self.num_beams
        self.relative_angles = (
            -0.5 * fov_rad
            + (np.arange(self.num_beams, dtype=np.float64) + 0.5) * beam_width
        )

    @staticmethod
    def _ray_aabb_distance(
        origin_x: float,
        origin_y: float,
        direction_x: float,
        direction_y: float,
        bounds: tuple[float, float, float, float],
    ) -> float | None:
        """返回射线首次进入 AABB 的非负距离；无交点时返回 ``None``。"""
        xmin, xmax, ymin, ymax = (float(value) for value in bounds)
        t_near = -math.inf
        t_far = math.inf
        for origin, direction, lower, upper in (
            (origin_x, direction_x, xmin, xmax),
            (origin_y, direction_y, ymin, ymax),
        ):
            if abs(direction) < 1e-12:
                if origin < lower or origin > upper:
                    return None
                continue
            t1 = (lower - origin) / direction
            t2 = (upper - origin) / direction
            t_near = max(t_near, min(t1, t2))
            t_far = min(t_far, max(t1, t2))
            if t_near > t_far:
                return None
        if t_far < 0.0:
            return None
        return max(0.0, t_near)

    @staticmethod
    def _ray_world_exit_distance(
        origin_x: float,
        origin_y: float,
        direction_x: float,
        direction_y: float,
        world_min: float,
        world_max: float,
    ) -> float:
        """返回从世界内部沿射线到正方形边界的距离。"""
        distances: list[float] = []
        if direction_x > 1e-12:
            distances.append((world_max - origin_x) / direction_x)
        elif direction_x < -1e-12:
            distances.append((world_min - origin_x) / direction_x)
        if direction_y > 1e-12:
            distances.append((world_max - origin_y) / direction_y)
        elif direction_y < -1e-12:
            distances.append((world_min - origin_y) / direction_y)
        positive = [distance for distance in distances if distance >= 0.0]
        return min(positive) if positive else 0.0

    def scan(
        self,
        *,
        x: float,
        y: float,
        heading: float,
        obstacles: np.ndarray,
        world_min: float,
        world_max: float,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """返回各束到最近障碍或世界边界的距离（米）。"""
        ranges = np.full(self.num_beams, self.max_range, dtype=np.float64)
        for index, relative_angle in enumerate(self.relative_angles):
            angle = float(heading) + float(relative_angle)
            direction_x = math.cos(angle)
            direction_y = math.sin(angle)
            nearest = self._ray_world_exit_distance(
                float(x),
                float(y),
                direction_x,
                direction_y,
                float(world_min),
                float(world_max),
            )
            for obstacle in obstacles:
                distance = self._ray_aabb_distance(
                    float(x),
                    float(y),
                    direction_x,
                    direction_y,
                    tuple(float(value) for value in obstacle),
                )
                if distance is not None:
                    nearest = min(nearest, distance)
            ranges[index] = min(max(nearest, 0.0), self.max_range)

        if self.noise_std > 0.0:
            if rng is None:
                raise ValueError("rng is required when lidar_noise_std is positive")
            ranges += rng.normal(0.0, self.noise_std, size=self.num_beams)
            np.clip(ranges, 0.0, self.max_range, out=ranges)
        return ranges.astype(np.float32)

    def normalize(self, ranges_m: np.ndarray) -> np.ndarray:
        """把米制距离裁剪并归一化到 ``[0, 1]``。"""
        ranges = np.asarray(ranges_m, dtype=np.float32)
        return np.clip(ranges / self.max_range, 0.0, 1.0).astype(np.float32)

