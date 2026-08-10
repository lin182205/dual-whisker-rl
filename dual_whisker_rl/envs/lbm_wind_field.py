"""二维 D2Q9 LBM 稳态障碍风场。

该模块只负责生成低马赫数、定性的二维平均风场。气味输运仍由
``DynamicPuffPlume`` 完成。矩形障碍通过 bounce-back 形成无滑移固体边界，
求解结果按地图几何和量化风向缓存在进程内，避免 PPO 每个环境步重复求解。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


_C = np.asarray(
    [
        [0, 0],
        [1, 0],
        [0, 1],
        [-1, 0],
        [0, -1],
        [1, 1],
        [-1, 1],
        [-1, -1],
        [1, -1],
    ],
    dtype=np.int8,
)
_W = np.asarray(
    [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4,
    dtype=np.float32,
)
_OPPOSITE = np.asarray([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int8)
_SOUND_SPEED = 1.0 / math.sqrt(3.0)


@dataclass(frozen=True)
class LBMSolveResult:
    """可共享的只读稳态风场及求解诊断。"""

    x: np.ndarray
    y: np.ndarray
    velocity_x_normalized: np.ndarray
    velocity_y_normalized: np.ndarray
    vorticity_normalized: np.ndarray
    turbulence_factor: np.ndarray
    solid_mask: np.ndarray
    direction_rad: float
    iterations: int
    residual: float
    density_min: float
    density_max: float
    time_averaged: bool


class SteadyLBMWindField2D:
    """D2Q9 BGK + bounce-back 的二维稳态障碍风场。"""

    _cache: dict[tuple[Any, ...], LBMSolveResult] = {}

    DEFAULT_CONFIG: dict[str, Any] = {
        "enabled": True,
        "method": "lbm_d2q9",
        "mode": "steady",
        "resolution": 129,
        "direction_bin_deg": 10.0,
        "padding_cells": 12,
        "lattice_speed": 0.04,
        "relaxation_time": 0.60,
        "convergence_tolerance": 1.0e-5,
        "max_iterations": 2500,
        "averaging_iterations": 400,
        "wake_turbulence_gain": 1.5,
        "wall_transmission": 0.0,
        "max_cfl_fraction": 0.5,
    }

    def __init__(
        self,
        *,
        world_min: float,
        world_max: float,
        obstacles: np.ndarray,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is not None and not isinstance(config, dict):
            raise ValueError("obstacle_flow must be a mapping")
        unknown = set(config or {}) - set(self.DEFAULT_CONFIG)
        if unknown:
            raise KeyError(
                "未知 obstacle_flow 配置: " + ", ".join(sorted(unknown))
            )
        values = dict(self.DEFAULT_CONFIG)
        values.update(config or {})

        self.world_min = float(world_min)
        self.world_max = float(world_max)
        self.obstacles = np.asarray(obstacles, dtype=np.float64).reshape(-1, 4).copy()
        self.enabled = self._require_bool("enabled", values["enabled"])
        self.method = str(values["method"])
        self.mode = str(values["mode"])
        self.resolution = self._require_int("resolution", values["resolution"], minimum=33)
        self.direction_bin_deg = self._require_positive(
            "direction_bin_deg", values["direction_bin_deg"]
        )
        self.padding_cells = self._require_int(
            "padding_cells", values["padding_cells"], minimum=2
        )
        self.lattice_speed = self._require_positive(
            "lattice_speed", values["lattice_speed"]
        )
        self.relaxation_time = float(values["relaxation_time"])
        self.convergence_tolerance = self._require_positive(
            "convergence_tolerance", values["convergence_tolerance"]
        )
        self.max_iterations = self._require_int(
            "max_iterations", values["max_iterations"], minimum=50
        )
        self.averaging_iterations = self._require_int(
            "averaging_iterations", values["averaging_iterations"], minimum=1
        )
        self.wake_turbulence_gain = self._require_nonnegative(
            "wake_turbulence_gain", values["wake_turbulence_gain"]
        )
        self.wall_transmission = float(values["wall_transmission"])
        self.max_cfl_fraction = self._require_positive(
            "max_cfl_fraction", values["max_cfl_fraction"]
        )

        if self.method != "lbm_d2q9":
            raise ValueError("obstacle_flow.method must be 'lbm_d2q9'")
        if self.mode != "steady":
            raise ValueError("obstacle_flow.mode must be 'steady'")
        if not 0.5 < self.relaxation_time <= 2.0:
            raise ValueError("relaxation_time must be in (0.5, 2.0]")
        if self.lattice_speed / _SOUND_SPEED >= 0.1:
            raise ValueError(
                "lattice_speed is too large: D2Q9 Mach number must stay below 0.1"
            )
        if self.direction_bin_deg > 180.0:
            raise ValueError("direction_bin_deg must not exceed 180")
        if self.averaging_iterations >= self.max_iterations:
            raise ValueError("averaging_iterations must be smaller than max_iterations")
        if not 0.0 <= self.wall_transmission <= 1.0:
            raise ValueError("wall_transmission must be between 0 and 1")
        if self.max_cfl_fraction > 1.0:
            raise ValueError("max_cfl_fraction must not exceed 1")

        self.cell_size = (self.world_max - self.world_min) / (self.resolution - 1)
        self.lattice_viscosity = (self.relaxation_time - 0.5) / 3.0
        self.mach_number = self.lattice_speed / _SOUND_SPEED
        self.result: LBMSolveResult | None = None
        self.cache_hit = False
        self.requested_direction_rad = 0.0
        self.direction_error_rad = 0.0

    @staticmethod
    def _require_bool(name: str, value: Any) -> bool:
        if not isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{name} must be a boolean")
        return bool(value)

    @staticmethod
    def _require_int(name: str, value: Any, *, minimum: int) -> int:
        if isinstance(value, (bool, np.bool_)) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
        integer = int(value)
        if integer < minimum:
            raise ValueError(f"{name} must be at least {minimum}")
        return integer

    @staticmethod
    def _require_positive(name: str, value: Any) -> float:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        return number

    @staticmethod
    def _require_nonnegative(name: str, value: Any) -> float:
        number = float(value)
        if not math.isfinite(number) or number < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
        return number

    @property
    def active(self) -> bool:
        return bool(self.enabled and self.obstacles.size)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def quantize_direction(self, direction_rad: float) -> float:
        degrees = math.degrees(self._wrap_angle(float(direction_rad)))
        quantized = round(degrees / self.direction_bin_deg) * self.direction_bin_deg
        return self._wrap_angle(math.radians(quantized))

    def _cache_key(self, direction_rad: float) -> tuple[Any, ...]:
        return (
            round(self.world_min, 12),
            round(self.world_max, 12),
            tuple(np.round(self.obstacles.reshape(-1), 9)),
            self.resolution,
            self.padding_cells,
            round(self.lattice_speed, 12),
            round(self.relaxation_time, 12),
            round(self.convergence_tolerance, 14),
            self.max_iterations,
            self.averaging_iterations,
            round(self.wake_turbulence_gain, 12),
            round(direction_rad, 12),
        )

    def prepare(self, direction_rad: float) -> LBMSolveResult | None:
        """绑定当前回合平均风向；必要时求解并缓存稳态场。"""
        self.requested_direction_rad = self._wrap_angle(float(direction_rad))
        if not self.active:
            self.result = None
            self.cache_hit = False
            self.direction_error_rad = 0.0
            return None

        solved_direction = self.quantize_direction(self.requested_direction_rad)
        self.direction_error_rad = self._wrap_angle(
            solved_direction - self.requested_direction_rad
        )
        key = self._cache_key(solved_direction)
        cached = self._cache.get(key)
        if cached is not None:
            self.result = cached
            self.cache_hit = True
            return cached

        result = self._solve(solved_direction)
        self._cache[key] = result
        self.result = result
        self.cache_hit = False
        return result

    @staticmethod
    def _equilibrium(
        rho: np.ndarray,
        velocity_x: np.ndarray,
        velocity_y: np.ndarray,
    ) -> np.ndarray:
        cu = (
            _C[:, 0, None, None] * velocity_x[None, :, :]
            + _C[:, 1, None, None] * velocity_y[None, :, :]
        )
        velocity_sq = velocity_x * velocity_x + velocity_y * velocity_y
        return (
            _W[:, None, None]
            * rho[None, :, :]
            * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * velocity_sq[None, :, :])
        )

    def _grid_and_solid_mask(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, slice, slice]:
        total = self.resolution + 2 * self.padding_cells
        extended_min = self.world_min - self.padding_cells * self.cell_size
        coords = extended_min + np.arange(total, dtype=np.float64) * self.cell_size
        grid_x, grid_y = np.meshgrid(coords, coords)
        solid = np.zeros((total, total), dtype=bool)
        half = 0.5 * self.cell_size
        for xmin, xmax, ymin, ymax in self.obstacles:
            solid |= (
                (grid_x + half >= xmin)
                & (grid_x - half <= xmax)
                & (grid_y + half >= ymin)
                & (grid_y - half <= ymax)
            )
        core = slice(self.padding_cells, self.padding_cells + self.resolution)
        return coords, grid_x, solid, core, core

    @staticmethod
    def _far_field_equilibrium(velocity_x: float, velocity_y: float) -> np.ndarray:
        cu = _C[:, 0] * float(velocity_x) + _C[:, 1] * float(velocity_y)
        velocity_sq = float(velocity_x) ** 2 + float(velocity_y) ** 2
        return (
            _W * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * velocity_sq)
        ).astype(np.float32)

    @staticmethod
    def _apply_far_field(
        distributions: np.ndarray,
        equilibrium: np.ndarray,
    ) -> None:
        distributions[:, 0, :] = equilibrium[:, None]
        distributions[:, -1, :] = equilibrium[:, None]
        distributions[:, :, 0] = equilibrium[:, None]
        distributions[:, :, -1] = equilibrium[:, None]

    @staticmethod
    def _macroscopic(
        distributions: np.ndarray,
        solid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rho = np.sum(distributions, axis=0)
        safe_rho = np.maximum(rho, 1e-12)
        velocity_x = np.sum(
            distributions * _C[:, 0, None, None], axis=0
        ) / safe_rho
        velocity_y = np.sum(
            distributions * _C[:, 1, None, None], axis=0
        ) / safe_rho
        velocity_x[solid] = 0.0
        velocity_y[solid] = 0.0
        return rho, velocity_x, velocity_y

    def _solve(self, direction_rad: float) -> LBMSolveResult:
        _, _, solid, core_y, core_x = self._grid_and_solid_mask()
        total = solid.shape[0]
        target_x = self.lattice_speed * math.cos(direction_rad)
        target_y = self.lattice_speed * math.sin(direction_rad)
        rho = np.ones((total, total), dtype=np.float32)
        velocity_x = np.full((total, total), target_x, dtype=np.float32)
        velocity_y = np.full((total, total), target_y, dtype=np.float32)
        velocity_x[solid] = 0.0
        velocity_y[solid] = 0.0
        distributions = self._equilibrium(rho, velocity_x, velocity_y)
        omega = 1.0 / self.relaxation_time
        previous_velocity: np.ndarray | None = None
        residual = math.inf
        converged = False
        minimum_iterations = min(300, self.max_iterations // 2)
        average_start = self.max_iterations - self.averaging_iterations
        average_x = np.zeros((self.resolution, self.resolution), dtype=np.float64)
        average_y = np.zeros_like(average_x)
        average_count = 0
        far_field_equilibrium = self._far_field_equilibrium(target_x, target_y)

        for iteration in range(1, self.max_iterations + 1):
            rho, velocity_x, velocity_y = self._macroscopic(distributions, solid)
            if (
                not np.all(np.isfinite(rho))
                or not np.all(np.isfinite(velocity_x))
                or not np.all(np.isfinite(velocity_y))
                or float(np.min(rho)) <= 0.05
                or float(np.max(rho)) >= 5.0
            ):
                raise RuntimeError(
                    "D2Q9 LBM 数值失稳: "
                    f"iteration={iteration}, rho=[{float(np.nanmin(rho)):.4g}, "
                    f"{float(np.nanmax(rho)):.4g}], tau={self.relaxation_time}, "
                    f"Mach={self.mach_number:.4f}"
                )

            equilibrium = self._equilibrium(rho, velocity_x, velocity_y)
            post_collision = distributions - omega * (distributions - equilibrium)
            streamed = np.empty_like(distributions)
            for index, (cx, cy) in enumerate(_C):
                streamed[index] = np.roll(
                    post_collision[index],
                    shift=(int(cy), int(cx)),
                    axis=(0, 1),
                )
            bounced = streamed[:, solid].copy()
            streamed[:, solid] = bounced[_OPPOSITE]
            distributions = streamed
            self._apply_far_field(distributions, far_field_equilibrium)

            if iteration >= average_start:
                _, current_x, current_y = self._macroscopic(distributions, solid)
                average_x += current_x[core_y, core_x]
                average_y += current_y[core_y, core_x]
                average_count += 1

            if iteration % 50 == 0:
                _, current_x, current_y = self._macroscopic(distributions, solid)
                core_velocity = np.stack(
                    [current_x[core_y, core_x], current_y[core_y, core_x]], axis=0
                )
                if previous_velocity is not None:
                    numerator = float(np.linalg.norm(core_velocity - previous_velocity))
                    denominator = max(float(np.linalg.norm(previous_velocity)), 1e-12)
                    residual = numerator / denominator
                    if iteration >= minimum_iterations and residual <= self.convergence_tolerance:
                        converged = True
                        break
                previous_velocity = core_velocity.copy()

        iterations = iteration
        rho, final_x, final_y = self._macroscopic(distributions, solid)
        if converged:
            core_x_velocity = final_x[core_y, core_x].copy()
            core_y_velocity = final_y[core_y, core_x].copy()
            time_averaged = False
        else:
            if average_count == 0:
                core_x_velocity = final_x[core_y, core_x].copy()
                core_y_velocity = final_y[core_y, core_x].copy()
            else:
                core_x_velocity = average_x / average_count
                core_y_velocity = average_y / average_count
            time_averaged = True

        core_solid = solid[core_y, core_x].copy()
        core_x_velocity[core_solid] = 0.0
        core_y_velocity[core_solid] = 0.0
        velocity_x_normalized = core_x_velocity / self.lattice_speed
        velocity_y_normalized = core_y_velocity / self.lattice_speed

        dv_dx = np.gradient(velocity_y_normalized, axis=1)
        du_dy = np.gradient(velocity_x_normalized, axis=0)
        vorticity = dv_dx - du_dy
        speed = np.hypot(velocity_x_normalized, velocity_y_normalized)
        deficit = np.clip(1.0 - speed, 0.0, 1.0)
        shear_metric = np.clip(4.0 * np.abs(vorticity), 0.0, 1.0)
        turbulence_factor = np.clip(
            1.0 + self.wake_turbulence_gain * (deficit + shear_metric),
            1.0,
            3.0,
        )
        turbulence_factor[core_solid] = 1.0

        coords = np.linspace(
            self.world_min,
            self.world_max,
            self.resolution,
            dtype=np.float64,
        )
        arrays = (
            coords,
            velocity_x_normalized,
            velocity_y_normalized,
            vorticity,
            turbulence_factor,
            core_solid,
        )
        for array in arrays:
            array.setflags(write=False)

        return LBMSolveResult(
            x=coords,
            y=coords,
            velocity_x_normalized=velocity_x_normalized,
            velocity_y_normalized=velocity_y_normalized,
            vorticity_normalized=vorticity,
            turbulence_factor=turbulence_factor,
            solid_mask=core_solid,
            direction_rad=float(direction_rad),
            iterations=int(iterations),
            residual=float(residual),
            density_min=float(np.min(rho)),
            density_max=float(np.max(rho)),
            time_averaged=bool(time_averaged),
        )

    def sample_many(
        self,
        x: np.ndarray,
        y: np.ndarray,
        *,
        wind_speed: float,
        direction_offset_rad: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """批量返回物理速度和局部尾流湍流倍率。"""
        if self.result is None:
            raise RuntimeError("LBM wind field must be prepared before sampling")
        x_array = np.asarray(x, dtype=np.float64)
        y_array = np.asarray(y, dtype=np.float64)
        gx = np.clip(
            (x_array - self.world_min) / self.cell_size,
            0.0,
            self.resolution - 1.0,
        )
        gy = np.clip(
            (y_array - self.world_min) / self.cell_size,
            0.0,
            self.resolution - 1.0,
        )
        x0 = np.minimum(np.floor(gx).astype(np.intp), self.resolution - 2)
        y0 = np.minimum(np.floor(gy).astype(np.intp), self.resolution - 2)
        tx = gx - x0
        ty = gy - y0
        weight_00 = (1.0 - tx) * (1.0 - ty)
        weight_10 = tx * (1.0 - ty)
        weight_01 = (1.0 - tx) * ty
        weight_11 = tx * ty

        def interpolate(values: np.ndarray) -> np.ndarray:
            return (
                weight_00 * values[y0, x0]
                + weight_10 * values[y0, x0 + 1]
                + weight_01 * values[y0 + 1, x0]
                + weight_11 * values[y0 + 1, x0 + 1]
            )

        base_x = interpolate(self.result.velocity_x_normalized)
        base_y = interpolate(self.result.velocity_y_normalized)
        turbulence = interpolate(self.result.turbulence_factor)
        outside = (
            (x_array < self.world_min)
            | (x_array > self.world_max)
            | (y_array < self.world_min)
            | (y_array > self.world_max)
        )
        if np.any(outside):
            base_x = base_x.copy()
            base_y = base_y.copy()
            turbulence = turbulence.copy()
            base_x[outside] = math.cos(self.result.direction_rad)
            base_y[outside] = math.sin(self.result.direction_rad)
            turbulence[outside] = 1.0

        cosine = math.cos(float(direction_offset_rad))
        sine = math.sin(float(direction_offset_rad))
        cross_x = -math.sin(self.result.direction_rad)
        cross_y = math.cos(self.result.direction_rad)
        velocity_x = float(wind_speed) * (cosine * base_x + sine * cross_x)
        velocity_y = float(wind_speed) * (cosine * base_y + sine * cross_y)
        return velocity_x, velocity_y, np.clip(turbulence, 1.0, 3.0)

    def sample(
        self,
        x: float,
        y: float,
        *,
        wind_speed: float,
        direction_offset_rad: float = 0.0,
    ) -> tuple[float, float, float]:
        """返回物理速度 ``(u, v)`` 和局部尾流湍流倍率。"""
        if self.result is None:
            raise RuntimeError("LBM wind field must be prepared before sampling")
        velocity_x, velocity_y, turbulence = self.sample_many(
            np.asarray([x], dtype=np.float64),
            np.asarray([y], dtype=np.float64),
            wind_speed=wind_speed,
            direction_offset_rad=direction_offset_rad,
        )
        return float(velocity_x[0]), float(velocity_y[0]), float(turbulence[0])

    def metadata(self) -> dict[str, Any]:
        result = self.result
        return {
            "enabled": self.enabled,
            "active": self.active,
            "method": self.method,
            "mode": self.mode,
            "resolution": self.resolution,
            "direction_bin_deg": self.direction_bin_deg,
            "padding_cells": self.padding_cells,
            "cell_size_m": self.cell_size,
            "lattice_speed": self.lattice_speed,
            "relaxation_time": self.relaxation_time,
            "lattice_viscosity": self.lattice_viscosity,
            "mach_number": self.mach_number,
            "convergence_tolerance": self.convergence_tolerance,
            "max_iterations": self.max_iterations,
            "averaging_iterations": self.averaging_iterations,
            "wake_turbulence_gain": self.wake_turbulence_gain,
            "wall_transmission": self.wall_transmission,
            "max_cfl_fraction": self.max_cfl_fraction,
            "cache_hit": self.cache_hit,
            "requested_direction_rad": self.requested_direction_rad,
            "solved_direction_rad": None if result is None else result.direction_rad,
            "direction_error_rad": self.direction_error_rad,
            "iterations": None if result is None else result.iterations,
            "residual": None if result is None else result.residual,
            "density_min": None if result is None else result.density_min,
            "density_max": None if result is None else result.density_max,
            "time_averaged": None if result is None else result.time_averaged,
        }


def segment_aabb_first_hit(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    bounds: np.ndarray | tuple[float, float, float, float],
) -> tuple[float, float, float] | None:
    """返回线段首次进入 AABB 的 ``(t, normal_x, normal_y)``。"""
    xmin, xmax, ymin, ymax = (float(value) for value in bounds)
    dx = float(end_x) - float(start_x)
    dy = float(end_y) - float(start_y)
    t_enter = 0.0
    t_exit = 1.0
    normal_x = 0.0
    normal_y = 0.0
    for origin, delta, lower, upper, lower_normal, upper_normal in (
        (start_x, dx, xmin, xmax, (-1.0, 0.0), (1.0, 0.0)),
        (start_y, dy, ymin, ymax, (0.0, -1.0), (0.0, 1.0)),
    ):
        if abs(delta) < 1e-12:
            if origin < lower or origin > upper:
                return None
            continue
        near = (lower - origin) / delta
        far = (upper - origin) / delta
        near_normal = lower_normal
        if near > far:
            near, far = far, near
            near_normal = upper_normal
        if near > t_enter:
            t_enter = near
            normal_x, normal_y = near_normal
        t_exit = min(t_exit, far)
        if t_enter > t_exit:
            return None
    if t_exit < 0.0 or t_enter > 1.0:
        return None
    return max(0.0, t_enter), normal_x, normal_y


def segment_blocked_by_obstacles(
    start_x: float,
    start_y: float,
    end_x: float,
    end_y: float,
    obstacles: np.ndarray,
) -> bool:
    """判断两点间的直线是否穿过任一矩形障碍。"""
    return any(
        segment_aabb_first_hit(start_x, start_y, end_x, end_y, obstacle)
        is not None
        for obstacle in obstacles
    )


def segment_aabb_first_hit_many(
    start_x: np.ndarray,
    start_y: np.ndarray,
    end_x: np.ndarray,
    end_y: np.ndarray,
    obstacle: np.ndarray | tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """批量返回线段与一个 AABB 的命中掩码、首次参数和外法向。"""
    xmin, xmax, ymin, ymax = (float(value) for value in obstacle)
    start_x = np.asarray(start_x, dtype=np.float64)
    start_y = np.asarray(start_y, dtype=np.float64)
    dx = np.asarray(end_x, dtype=np.float64) - start_x
    dy = np.asarray(end_y, dtype=np.float64) - start_y
    t_enter = np.zeros_like(start_x)
    t_exit = np.ones_like(start_x)
    normal_x = np.zeros_like(start_x)
    normal_y = np.zeros_like(start_x)
    valid = np.ones_like(start_x, dtype=bool)

    for origin, delta, lower, upper, axis in (
        (start_x, dx, xmin, xmax, 0),
        (start_y, dy, ymin, ymax, 1),
    ):
        moving = np.abs(delta) >= 1e-12
        valid &= moving | ((origin >= lower) & (origin <= upper))
        safe_delta = np.where(moving, delta, 1.0)
        t_lower = (lower - origin) / safe_delta
        t_upper = (upper - origin) / safe_delta
        near = np.minimum(t_lower, t_upper)
        far = np.maximum(t_lower, t_upper)
        update = moving & (near > t_enter)
        if axis == 0:
            normal_x[update] = np.where(delta[update] > 0.0, -1.0, 1.0)
            normal_y[update] = 0.0
        else:
            normal_x[update] = 0.0
            normal_y[update] = np.where(delta[update] > 0.0, -1.0, 1.0)
        t_enter = np.where(moving, np.maximum(t_enter, near), t_enter)
        t_exit = np.where(moving, np.minimum(t_exit, far), t_exit)
        valid &= t_enter <= t_exit
    hit = valid & (t_exit >= 0.0) & (t_enter <= 1.0)
    return hit, np.maximum(t_enter, 0.0), normal_x, normal_y


def segment_block_mask(
    start_x: float,
    start_y: float,
    end_x: np.ndarray,
    end_y: np.ndarray,
    obstacle: np.ndarray | tuple[float, float, float, float],
) -> np.ndarray:
    """向量化判断从一个起点到网格终点的线段是否穿过 AABB。"""
    xmin, xmax, ymin, ymax = (float(value) for value in obstacle)
    dx = np.asarray(end_x, dtype=np.float64) - float(start_x)
    dy = np.asarray(end_y, dtype=np.float64) - float(start_y)
    t_enter = np.zeros_like(dx)
    t_exit = np.ones_like(dx)
    valid = np.ones_like(dx, dtype=bool)

    for origin, delta, lower, upper in (
        (float(start_x), dx, xmin, xmax),
        (float(start_y), dy, ymin, ymax),
    ):
        moving = np.abs(delta) >= 1e-12
        if not lower <= origin <= upper:
            valid &= moving
        safe_delta = np.where(moving, delta, 1.0)
        t1 = (lower - origin) / safe_delta
        t2 = (upper - origin) / safe_delta
        near = np.minimum(t1, t2)
        far = np.maximum(t1, t2)
        t_enter = np.where(moving, np.maximum(t_enter, near), t_enter)
        t_exit = np.where(moving, np.minimum(t_exit, far), t_exit)
        valid &= t_enter <= t_exit
    return valid & (t_exit >= 0.0) & (t_enter <= 1.0)
