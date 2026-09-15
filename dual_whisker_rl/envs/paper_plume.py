"""Paper-compatible stochastic puff plume model.

This module reproduces the plume generator used by Singh et al. (2023):

* puffs are emitted from a point source by a Poisson process;
* puff centres are advected by a spatially homogeneous wind field;
* each puff performs an independent cross-wind random walk;
* puff radius grows linearly; and
* concentration inside a puff decays as ``(r0 / r) ** 3``.

The arena geometry and source placement hooks intentionally follow this project so the
same model can be used by both fixed and mobile dual-whisker environments.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from dual_whisker_rl.envs.world_bounds import resolve_world_bounds, world_bounds_config


class PaperPuffPlume:
    """Online version of the particle plume used in the reference paper."""

    VALID_PROFILES = ("constant", "sparse", "sparser", "switch_once", "switch_many")

    def __init__(
        self,
        seed: int = 0,
        source_position: tuple[float, float] | None = None,
        dt: float = 0.2,
        gas_field_mode: str = "paper_puff",
        wind_speed_range: tuple[float, float] = (0.5, 0.5),
        wind_sampling_mode: str = "fixed",
        obstacles: np.ndarray | None = None,
        plume_overrides: dict[str, Any] | None = None,
        world_bounds: tuple[float, float] | list[float] | None = None,
        world_half: float | None = None,
        profile: str = "constant",
    ) -> None:
        if profile not in self.VALID_PROFILES:
            raise ValueError(
                f"paper plume profile must be one of {self.VALID_PROFILES}, got {profile!r}"
            )
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        self.rng = np.random.default_rng(seed)
        self.world_min, self.world_max, self.world_half = resolve_world_bounds(
            world_bounds, world_half, default_half=0.5
        )
        self.source_position_override = source_position
        self.source_clearance = 0.08
        self.source_x = float(source_position[0]) if source_position is not None else -1.2
        self.source_y = float(source_position[1]) if source_position is not None else 0.0
        self.dt = float(dt)
        self.gas_field_mode = str(gas_field_mode)
        self.obstacles = (
            np.asarray(obstacles, dtype=np.float32).copy()
            if obstacles is not None
            else np.empty((0, 4), dtype=np.float32)
        )

        self.profile = str(profile)
        self.paper_sim_dt = 0.01
        self.puff_birth_rate = 1.0
        self.puff_init_radius = 0.01
        self.puff_spread_rate = 0.01
        self.crosswind_velocity_std_scale = 1.0
        # The released generator adds noise to global y. ``wind_perpendicular`` is
        # rotationally symmetric and is preferable for this project's randomized arena.
        self.crosswind_mode = "wind_perpendicular"
        self.puff_warmup_seconds = 6.0
        self.puff_warmup_steps = max(0, int(round(self.puff_warmup_seconds / self.dt)))
        self.puff_bounds_margin = 1.1
        self.max_puffs = 5000
        self.gas_background = 0.0
        self.gas_noise_std = 0.0

        self.wind_speed_range = (
            float(wind_speed_range[0]),
            float(wind_speed_range[1]),
        )
        if self.wind_speed_range[0] <= 0.0 or self.wind_speed_range[1] < self.wind_speed_range[0]:
            raise ValueError("wind_speed_range must be positive and ordered")
        self.wind_speed_mean = 0.5 * sum(self.wind_speed_range)
        self.wind_sampling_mode = str(wind_sampling_mode)
        self.wind_direction_range = (math.radians(-12.0), math.radians(12.0))
        self.wind_direction_base = 0.0
        self.wind_direction = 0.0
        self.wind_speed = self.wind_speed_mean

        self.switch_once_time_s = 6.0
        self.switch_once_angle_rad = math.radians(45.0)
        self.switch_interval_s = 3.0
        self.switch_interval_jitter_s = 0.3
        self.switch_angle_std_rad = math.radians(45.0)
        self.switch_angle_limit_rad = math.radians(60.0)
        self._episode_time = 0.0
        self._next_switch_time = math.inf
        self._switch_once_done = False

        self.far_source_distance = 0.75
        self.far_source_crosswind_range = (-0.18, 0.18)
        self.source_distance_range = (self.far_source_distance, self.far_source_distance)
        self.require_source_in_world = False

        # Columns are x, y and radius. NumPy storage keeps the 100 Hz substeps cheap.
        self.puffs = np.empty((0, 3), dtype=np.float64)
        self.domain_randomization = False

        if self.profile in ("sparse", "sparser"):
            self.puff_birth_rate *= 0.4
        if self.profile == "sparser":
            self.puff_spread_rate *= 0.5

        override_keys = set(plume_overrides or {})
        if plume_overrides:
            for key, value in plume_overrides.items():
                if not hasattr(self, key):
                    raise KeyError(f"unknown paper plume override parameter: {key}")
                setattr(self, key, value)
        if "puff_warmup_seconds" in override_keys and "puff_warmup_steps" not in override_keys:
            self.puff_warmup_steps = max(
                0, int(round(float(self.puff_warmup_seconds) / self.dt))
            )
        self._validate_parameters()
        self._dr_nominal = {
            "puff_birth_rate": float(self.puff_birth_rate),
            "puff_spread_rate": float(self.puff_spread_rate),
            "crosswind_velocity_std_scale": float(self.crosswind_velocity_std_scale),
        }

    def _validate_parameters(self) -> None:
        ratio = self.dt / float(self.paper_sim_dt)
        if self.paper_sim_dt <= 0.0 or not math.isclose(ratio, round(ratio), abs_tol=1e-9):
            raise ValueError("dt must be an integer multiple of paper_sim_dt")
        if self.puff_birth_rate < 0.0:
            raise ValueError("puff_birth_rate must be non-negative")
        if self.puff_init_radius <= 0.0 or self.puff_spread_rate < 0.0:
            raise ValueError("puff radii and spread rate must be non-negative")
        if self.crosswind_mode not in ("global_y", "wind_perpendicular"):
            raise ValueError("crosswind_mode must be 'global_y' or 'wind_perpendicular'")

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def metadata(self) -> dict[str, Any]:
        return {
            "world_min": self.world_min,
            "world_max": self.world_max,
            "world_bounds": world_bounds_config(self.world_min, self.world_max),
            "source_position": (float(self.source_x), float(self.source_y)),
            "obstacles": self.obstacles.copy(),
            "gas_field_mode": "paper_puff",
            "paper_plume_profile": self.profile,
            "wind_direction_base": float(self.wind_direction_base),
            "wind_direction": float(self.wind_direction),
            "wind_speed": float(self.wind_speed),
            "wind_sampling_mode": self.wind_sampling_mode,
            "puff_count": int(len(self.puffs)),
            "paper_sim_dt": float(self.paper_sim_dt),
            "puff_birth_rate": float(self.puff_birth_rate),
            "puff_init_radius": float(self.puff_init_radius),
            "puff_spread_rate": float(self.puff_spread_rate),
        }

    def is_occupied(self, x: float, y: float) -> bool:
        return not self.is_position_valid(x, y)

    def is_position_valid(self, x: float, y: float, margin: float = 0.0) -> bool:
        if not (
            self.world_min + margin < x < self.world_max - margin
            and self.world_min + margin < y < self.world_max - margin
        ):
            return False
        for xmin, xmax, ymin, ymax in self.obstacles:
            if (
                xmin - margin <= x <= xmax + margin
                and ymin - margin <= y <= ymax + margin
            ):
                return False
        return True

    def _sample_wind(self) -> None:
        if self.wind_sampling_mode == "fixed":
            self.wind_direction_base = 0.0
            self.wind_speed = self.wind_speed_mean
        elif self.wind_sampling_mode == "random":
            self.wind_direction_base = float(self.rng.uniform(*self.wind_direction_range))
            self.wind_speed = float(self.rng.uniform(*self.wind_speed_range))
        else:
            raise ValueError("wind_sampling_mode must be 'fixed' or 'random'")
        self.wind_direction = self.wind_direction_base

    def _place_source_upwind(self) -> None:
        if self.source_position_override is not None:
            self.source_x = float(self.source_position_override[0])
            self.source_y = float(self.source_position_override[1])
            return

        wind_x = math.cos(self.wind_direction_base)
        wind_y = math.sin(self.wind_direction_base)
        cross_x, cross_y = -wind_y, wind_x

        def candidate(distance: float, crosswind: float) -> tuple[float, float]:
            return (
                -distance * wind_x + crosswind * cross_x,
                -distance * wind_y + crosswind * cross_y,
            )

        attempts = 256 if self.require_source_in_world else 1
        for _ in range(attempts):
            distance = float(self.rng.uniform(*self.source_distance_range))
            crosswind = float(self.rng.uniform(*self.far_source_crosswind_range))
            x, y = candidate(distance, crosswind)
            if not self.require_source_in_world or self.is_position_valid(
                x, y, margin=self.source_clearance
            ):
                self.source_x, self.source_y = x, y
                return
        raise RuntimeError("could not place paper plume source inside the configured world")

    def _schedule_switches(self) -> None:
        self._episode_time = 0.0
        self._switch_once_done = False
        if self.profile == "switch_many":
            jitter = float(
                self.rng.uniform(-self.switch_interval_jitter_s, self.switch_interval_jitter_s)
            )
            self._next_switch_time = max(self.paper_sim_dt, self.switch_interval_s + jitter)
        else:
            self._next_switch_time = math.inf

    def _update_wind_schedule(self) -> None:
        if (
            self.profile == "switch_once"
            and not self._switch_once_done
            and self._episode_time >= self.switch_once_time_s
        ):
            self.wind_direction = self._normalize_angle(
                self.wind_direction_base + self.switch_once_angle_rad
            )
            self._switch_once_done = True
        elif self.profile == "switch_many" and self._episode_time >= self._next_switch_time:
            offset = float(
                np.clip(
                    self.rng.normal(0.0, self.switch_angle_std_rad),
                    -self.switch_angle_limit_rad,
                    self.switch_angle_limit_rad,
                )
            )
            self.wind_direction = self._normalize_angle(self.wind_direction_base + offset)
            jitter = float(
                self.rng.uniform(-self.switch_interval_jitter_s, self.switch_interval_jitter_s)
            )
            self._next_switch_time += max(self.paper_sim_dt, self.switch_interval_s + jitter)

    def _trim_puffs(self) -> None:
        if not len(self.puffs):
            return
        lower = self.world_min - self.puff_bounds_margin
        upper = self.world_max + self.puff_bounds_margin
        keep = (
            (self.puffs[:, 0] > lower)
            & (self.puffs[:, 0] < upper)
            & (self.puffs[:, 1] > lower)
            & (self.puffs[:, 1] < upper)
        )
        self.puffs = self.puffs[keep]

    def _emit_puffs(self) -> None:
        count = int(self.rng.poisson(self.puff_birth_rate))
        if count <= 0:
            return
        newborn = np.empty((count, 3), dtype=np.float64)
        newborn[:, 0] = self.source_x
        newborn[:, 1] = self.source_y
        newborn[:, 2] = self.puff_init_radius
        self.puffs = np.concatenate((self.puffs, newborn), axis=0)
        if len(self.puffs) > self.max_puffs:
            self.puffs = self.puffs[-self.max_puffs :]

    def _advance_substep(self, *, allow_switches: bool = True) -> None:
        if allow_switches:
            self._update_wind_schedule()
        n_puffs = len(self.puffs)
        if n_puffs:
            cos_w = math.cos(self.wind_direction)
            sin_w = math.sin(self.wind_direction)
            step = self.paper_sim_dt
            self.puffs[:, 0] += self.wind_speed * cos_w * step
            self.puffs[:, 1] += self.wind_speed * sin_w * step
            crosswind = self.rng.normal(
                0.0,
                self.wind_speed * self.crosswind_velocity_std_scale * step,
                size=n_puffs,
            )
            if self.crosswind_mode == "global_y":
                self.puffs[:, 1] += crosswind
            else:
                self.puffs[:, 0] -= sin_w * crosswind
                self.puffs[:, 1] += cos_w * crosswind
            self.puffs[:, 2] += self.puff_spread_rate * step
            self._trim_puffs()
        self._emit_puffs()
        self._episode_time += self.paper_sim_dt

    def advance(self) -> None:
        """Advance one environment step using the paper's 100 Hz simulation."""
        substeps = int(round(self.dt / self.paper_sim_dt))
        for _ in range(substeps):
            self._advance_substep()

    def _puff_strengths(self) -> np.ndarray:
        if not len(self.puffs):
            return np.empty(0, dtype=np.float64)
        return (self.puff_init_radius / self.puffs[:, 2]) ** 3

    def concentration(self, x: float, y: float, add_noise: bool = True) -> float:
        if not len(self.puffs):
            total = self.gas_background
        else:
            inside = (
                (np.abs(float(x) - self.puffs[:, 0]) < self.puffs[:, 2])
                & (np.abs(float(y) - self.puffs[:, 1]) < self.puffs[:, 2])
            )
            total = self.gas_background + float(self._puff_strengths()[inside].sum())
        if add_noise and self.gas_noise_std > 0.0:
            total += float(self.rng.normal(0.0, self.gas_noise_std))
        return max(0.0, float(total))

    def grid(
        self,
        resolution: int = 80,
        add_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if resolution < 2:
            raise ValueError("resolution must be at least 2")
        xs = np.linspace(self.world_min, self.world_max, resolution, dtype=np.float32)
        ys = np.linspace(self.world_min, self.world_max, resolution, dtype=np.float32)
        # A summed-area update reproduces the author's axis-aligned puff support while
        # avoiding an O(puffs * pixels) visualization loop.
        delta = np.zeros((resolution + 1, resolution + 1), dtype=np.float64)
        if len(self.puffs):
            radii = self.puffs[:, 2]
            x0 = np.searchsorted(xs, self.puffs[:, 0] - radii, side="right")
            x1 = np.searchsorted(xs, self.puffs[:, 0] + radii, side="left")
            y0 = np.searchsorted(ys, self.puffs[:, 1] - radii, side="right")
            y1 = np.searchsorted(ys, self.puffs[:, 1] + radii, side="left")
            strengths = self._puff_strengths()
            valid = (x0 < x1) & (y0 < y1)
            np.add.at(delta, (y0[valid], x0[valid]), strengths[valid])
            np.add.at(delta, (y1[valid], x0[valid]), -strengths[valid])
            np.add.at(delta, (y0[valid], x1[valid]), -strengths[valid])
            np.add.at(delta, (y1[valid], x1[valid]), strengths[valid])
        field = delta.cumsum(axis=0).cumsum(axis=1)[:resolution, :resolution]
        field += self.gas_background
        if add_noise and self.gas_noise_std > 0.0:
            field += self.rng.normal(0.0, self.gas_noise_std, size=field.shape)
        return xs, ys, np.maximum(field, 0.0).astype(np.float32)

    def _randomize_params(self) -> None:
        nominal = self._dr_nominal
        self.puff_birth_rate = nominal["puff_birth_rate"] * float(self.rng.uniform(0.7, 1.3))
        self.puff_spread_rate = nominal["puff_spread_rate"] * float(self.rng.uniform(0.7, 1.3))
        self.crosswind_velocity_std_scale = nominal[
            "crosswind_velocity_std_scale"
        ] * float(self.rng.uniform(0.7, 1.3))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.domain_randomization:
            self._randomize_params()
        self._sample_wind()
        self._place_source_upwind()
        # The released generator starts with puff 0 already present at t=0.
        self.puffs = np.array(
            [[self.source_x, self.source_y, self.puff_init_radius]],
            dtype=np.float64,
        )
        self._episode_time = 0.0
        # Establish a mature plume without consuming the episode's wind-switch schedule.
        substeps = int(round(self.dt / self.paper_sim_dt))
        for _ in range(int(self.puff_warmup_steps) * substeps):
            self._advance_substep(allow_switches=False)
        self.wind_direction = self.wind_direction_base
        self._schedule_switches()

    # Compatibility aliases shared with DynamicPuffPlume.
    def get_map_metadata(self) -> dict[str, Any]:
        return self.metadata()

    def get_gas_concentration(self, x: float, y: float, add_noise: bool = True) -> float:
        return self.concentration(x, y, add_noise=add_noise)

    def compute_concentration_grid(
        self,
        resolution: int = 80,
        add_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.grid(resolution=resolution, add_noise=add_noise)

    def advance_gas_field(self) -> None:
        self.advance()
