"""触须独立控制环境：固定机器人，仅训练左右触须主动采样。"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dual_whisker_rl.envs.lidar_model import validate_obstacles
from dual_whisker_rl.envs.lbm_wind_field import segment_aabb_first_hit
from dual_whisker_rl.envs.lbm_wind_field import segment_aabb_first_hit_many
from dual_whisker_rl.envs.lbm_wind_field import segment_block_mask
from dual_whisker_rl.envs.lbm_wind_field import SteadyLBMWindField2D
from dual_whisker_rl.envs.robot_model import RobotState
from dual_whisker_rl.envs.sensor_model import AsymmetricGasSensor
from dual_whisker_rl.envs.sensor_model import FirstOrderGasSensor
from dual_whisker_rl.envs.whisker_model import DualWhiskerSampler
from dual_whisker_rl.hardware.sensor_preprocess import SensorPreprocessConfig
from dual_whisker_rl.observation import WhiskerObservationBuilder


WORLD_MIN = -0.5
WORLD_MAX = 0.5
SOURCE_CLEARANCE = 0.08

DEFAULT_OBSTACLES = np.empty((0, 4), dtype=np.float32)


class DynamicPuffPlume:
    """动态 puff 气味羽流模型。

    该模型保留参考环境中的 puff 释放、随风漂移、扩散和衰减机制，
    但命名和接口尽量贴近本项目已有的 `GaussianPlume`：外部主要使用
    `reset()`、`advance()`、`concentration()` 和 `grid()`。
    """

    def __init__(
        self,
        seed: int = 0,
        source_position: tuple[float, float] | None = None,
        dt: float = 0.2,
        gas_field_mode: str = "puff",
        wind_speed_range: tuple[float, float] = (0.06, 0.11),
        wind_sampling_mode: str = "random",
        obstacles: np.ndarray | None = None,
        obstacle_flow: dict[str, Any] | None = None,
        plume_overrides: dict[str, Any] | None = None,
        world_half: float = WORLD_MAX,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        # 场地半宽可配置（默认 ±0.5 = 1m）。移动机器人环境用更大的场地以显现追踪行为。
        self.world_half = float(world_half)
        self.world_min = -self.world_half
        self.world_max = self.world_half
        self.source_position_override = source_position
        if source_position is None:
            self.source_x = self.world_min - 0.7
            self.source_y = 0.0
        else:
            self.source_x = float(source_position[0])
            self.source_y = float(source_position[1])
        self.source_clearance = float(SOURCE_CLEARANCE)
        self.dt = float(dt)
        self.obstacles = validate_obstacles(
            DEFAULT_OBSTACLES if obstacles is None else obstacles,
            world_min=self.world_min,
            world_max=self.world_max,
        )

        self.gas_peak = 1.0
        self.gas_noise_std = 0.004
        self.gas_background = 0.005
        self.gas_field_mode = str(gas_field_mode)

        self.wind_speed_range = (
            float(wind_speed_range[0]),
            float(wind_speed_range[1]),
        )
        self.wind_sampling_mode = str(wind_sampling_mode)
        self.wind_direction_range = (
            math.radians(-12.0),
            math.radians(12.0),
        )
        self.wind_speed_mean = 0.5 * (
            self.wind_speed_range[0] + self.wind_speed_range[1]
        )
        self.wind_speed_std = 0.006
        self.wind_relaxation = 0.97
        # 平均风向的缓慢蜿蜒（meander）：OU 过程，均值回归到 wind_direction_base。
        # 这是固定点上 whiff/blank 间歇的主要来源之一。std ≈ sigma/sqrt(2*theta)。
        self.wind_dir_meander_theta = 0.30
        self.wind_dir_meander_sigma = 0.32
        self.wind_direction_limit = math.radians(45.0)  # 仅作安全夹紧
        self.wind_dir_offset = 0.0  # OU 状态：当前风向相对 base 的偏移
        self.wind_speed_clip = (
            max(0.005, self.wind_speed_range[0] - 0.03),
            max(self.wind_speed_range[1] + 0.04, 0.12),
        )

        self.source_core_sigma = 0.12
        self.source_core_weight = 0.0
        # 稀疏、细、短寿命的 filament：避免大量 puff 叠加成连续气幕，
        # 使固定点看到一缕一缕经过（whiff/blank），而不是常通。
        # 参数已调稀疏/变窄（release 2→1、sigma 缩小、寿命缩短），使固定点在整段
        # episode 里都呈明显 whiff/blank 间歇（intermittency≈0.3），而非几乎常通。
        # 这样“触须指向哪个扇区”才持续影响信息获取，主动采样才有可学信号。
        self.puff_release_per_step = 1
        self.puff_release_downwind_jitter = 0.02
        self.puff_release_crosswind_jitter = 0.04
        self.puff_init_mass = 0.012
        self.puff_mass_jitter = 0.45
        self.puff_init_sigma_downwind = 0.052
        self.puff_init_sigma_crosswind = 0.024
        self.puff_sigma_jitter = 0.25
        self.diffusion_downwind_rate = 0.008
        self.diffusion_crosswind_rate = 0.003
        self.decay_rate = 0.052
        # 小尺度湍流：per-puff OU 速度扰动（m/s），时间相关 → 相干 filament 蜿蜒。
        self.turbulence_vel_relax = 0.90
        self.turbulence_vel_std_downwind = 0.020
        self.turbulence_vel_std_crosswind = 0.070
        self.max_puff_age = 7.0
        self.min_puff_mass = 0.0010
        self.max_puffs = 360
        self.puff_bounds_margin = 1.1
        self.puff_warmup_steps = 80
        self.far_source_distance = 0.75
        self.far_source_crosswind_range = (-0.18, 0.18)
        self.source_distance_range = (
            self.far_source_distance,
            self.far_source_distance,
        )
        self.require_source_in_world = False

        self.plume_contact_threshold = 0.08
        self.plume_strong_threshold = 0.18
        self.wind_direction_base = 0.0
        self.wind_direction = 0.0
        self.wind_speed = 1.0
        self.puffs: list[dict[str, float]] = []

        # 羽流物理参数覆盖（用于调稀疏/尺寸而无需改代码）。在 _dr_nominal 之前应用，
        # 使域随机化也围绕覆盖后的新值扰动。只允许覆盖已存在的属性。
        if plume_overrides:
            for key, value in plume_overrides.items():
                if not hasattr(self, key):
                    raise KeyError(f"未知 plume 覆盖参数: {key}")
                setattr(self, key, value)
            if (
                "far_source_distance" in plume_overrides
                and "source_distance_range" not in plume_overrides
            ):
                self.source_distance_range = (
                    float(self.far_source_distance),
                    float(self.far_source_distance),
                )

        # 域随机化：每个 episode 在标称值附近扰动羽流物理参数，提高策略鲁棒性。
        self.domain_randomization = False
        self._dr_nominal = {
            "puff_release_per_step": self.puff_release_per_step,
            "puff_init_mass": self.puff_init_mass,
            "diffusion_downwind_rate": self.diffusion_downwind_rate,
            "diffusion_crosswind_rate": self.diffusion_crosswind_rate,
            "decay_rate": self.decay_rate,
            "turbulence_vel_std_crosswind": self.turbulence_vel_std_crosswind,
            "wind_dir_meander_sigma": self.wind_dir_meander_sigma,
            "max_puff_age": self.max_puff_age,
        }
        self.obstacle_flow = SteadyLBMWindField2D(
            world_min=self.world_min,
            world_max=self.world_max,
            obstacles=self.obstacles,
            config=obstacle_flow,
        )

    def metadata(self) -> dict[str, Any]:
        """返回绘图和调试所需的气味场元信息。"""
        return {
            "world_min": self.world_min,
            "world_max": self.world_max,
            "source_position": (float(self.source_x), float(self.source_y)),
            "obstacles": self.obstacles.copy(),
            "gas_field_mode": "puff",
            "wind_direction_base": float(self.wind_direction_base),
            "wind_direction": float(self.wind_direction),
            "wind_speed": float(self.wind_speed),
            "wind_sampling_mode": self.wind_sampling_mode,
            "puff_count": int(len(self.puffs)),
            "obstacle_flow": self.obstacle_flow.metadata(),
        }

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        return math.atan2(math.sin(angle), math.cos(angle))

    def is_occupied(self, x: float, y: float) -> bool:
        if x <= self.world_min or x >= self.world_max or y <= self.world_min or y >= self.world_max:
            return True
        for xmin, xmax, ymin, ymax in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

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
        """每个 episode 开始时随机化风向和风速。"""
        self.wind_dir_offset = 0.0
        if self.wind_sampling_mode == "fixed":
            self.wind_direction_base = math.radians(18.0)
            self.wind_direction = self.wind_direction_base
            self.wind_speed = self.wind_speed_mean
            return
        self.wind_direction_base = float(self.rng.uniform(*self.wind_direction_range))
        self.wind_direction = self.wind_direction_base
        self.wind_speed = float(
            self.rng.uniform(self.wind_speed_range[0], self.wind_speed_range[1])
        )

    def _place_source_upwind(self) -> None:
        """将气源放到场地外的上风侧，使羽流随风进入整个训练区域。"""
        if self.source_position_override is not None:
            self.source_x = float(self.source_position_override[0])
            self.source_y = float(self.source_position_override[1])
            return
        wind_x = math.cos(self.wind_direction_base)
        wind_y = math.sin(self.wind_direction_base)
        cross_x = -wind_y
        cross_y = wind_x

        def candidate(source_distance: float, cross_offset: float) -> tuple[float, float]:
            return (
                -source_distance * wind_x + cross_offset * cross_x,
                -source_distance * wind_y + cross_offset * cross_y,
            )

        attempts = 256 if self.require_source_in_world else 1
        for _ in range(attempts):
            if self.source_distance_range[0] == self.source_distance_range[1]:
                source_distance = float(self.source_distance_range[0])
            else:
                source_distance = float(
                    self.rng.uniform(*self.source_distance_range)
                )
            cross_offset = float(self.rng.uniform(*self.far_source_crosswind_range))
            source_x, source_y = candidate(source_distance, cross_offset)
            if not self.require_source_in_world or self.is_position_valid(
                source_x,
                source_y,
                margin=self.source_clearance,
            ):
                self.source_x = source_x
                self.source_y = source_y
                return

        distance_values = np.linspace(*self.source_distance_range, num=33)
        crosswind_values = sorted(
            np.linspace(*self.far_source_crosswind_range, num=33),
            key=abs,
        )
        for source_distance in distance_values:
            for cross_offset in crosswind_values:
                source_x, source_y = candidate(
                    float(source_distance),
                    float(cross_offset),
                )
                if self.is_position_valid(
                    source_x,
                    source_y,
                    margin=self.source_clearance,
                ):
                    self.source_x = source_x
                    self.source_y = source_y
                    return
        raise RuntimeError("could not place source inside the configured world")

    def _update_wind(self) -> None:
        """每个环境步扰动风场。

        风向蜿蜒（meander）在两种模式下都运行，因为它代表真实气流的大尺度
        摆动，是固定点上 whiff/blank 间歇的主要来源。`fixed` 与 `random` 的
        区别只在于：fixed 用确定的平均风（可复现的诊断基准、不随 episode 变），
        random 每个 episode 随机化平均风向和风速。
        """
        # 风向偏移走 OU 过程：d_off = -theta*off*dt + sigma*sqrt(dt)*N，均值回归到 base。
        theta = self.wind_dir_meander_theta
        sigma = self.wind_dir_meander_sigma
        self.wind_dir_offset += (
            -theta * self.wind_dir_offset * self.dt
            + sigma * math.sqrt(self.dt) * float(self.rng.normal())
        )
        self.wind_dir_offset = float(
            np.clip(self.wind_dir_offset, -self.wind_direction_limit, self.wind_direction_limit)
        )
        self.wind_direction = self._normalize_angle(
            self.wind_direction_base + self.wind_dir_offset
        )
        if self.wind_sampling_mode == "fixed":
            self.wind_speed = self.wind_speed_mean
            return
        speed_noise = float(
            self.rng.normal(0.0, self.wind_speed_std * (1.0 - self.wind_relaxation))
        )
        self.wind_speed = float(
            np.clip(
                self.wind_relaxation * self.wind_speed
                + (1.0 - self.wind_relaxation) * self.wind_speed_mean
                + speed_noise,
                self.wind_speed_clip[0],
                self.wind_speed_clip[1],
            )
        )

    def _source_core_concentration(self, x: Any, y: Any) -> np.ndarray:
        """源点附近的稳定核心浓度，用于避免源点处完全依赖离散 puff。"""
        rel_x = np.asarray(x, dtype=np.float32) - self.source_x
        rel_y = np.asarray(y, dtype=np.float32) - self.source_y
        distance_sq = rel_x**2 + rel_y**2
        return (
            self.gas_peak
            * self.source_core_weight
            * np.exp(-distance_sq / (2.0 * self.source_core_sigma**2))
        )

    def _emit_puffs(self) -> None:
        """从气源位置释放新的气味团。"""
        wind_x = math.cos(self.wind_direction)
        wind_y = math.sin(self.wind_direction)
        cross_x = -wind_y
        cross_y = wind_x
        for _ in range(self.puff_release_per_step):
            downwind_offset = float(
                self.rng.normal(0.0, self.puff_release_downwind_jitter)
            )
            crosswind_offset = float(
                self.rng.normal(0.0, self.puff_release_crosswind_jitter)
            )
            sigma_scale = float(
                self.rng.uniform(
                    1.0 - self.puff_sigma_jitter,
                    1.0 + self.puff_sigma_jitter,
                )
            )
            self.puffs.append(
                {
                    "x": float(
                        self.source_x
                        + downwind_offset * wind_x
                        + crosswind_offset * cross_x
                    ),
                    "y": float(
                        self.source_y
                        + downwind_offset * wind_y
                        + crosswind_offset * cross_y
                    ),
                    "mass": float(
                        self.puff_init_mass
                        * self.rng.uniform(
                            1.0 - self.puff_mass_jitter,
                            1.0 + self.puff_mass_jitter,
                        )
                    ),
                    "sigma_downwind": float(self.puff_init_sigma_downwind * sigma_scale),
                    "sigma_crosswind": float(self.puff_init_sigma_crosswind * sigma_scale),
                    "direction": float(self.wind_direction),
                    "age": 0.0,
                    # per-puff OU 湍流速度扰动（在该 puff 局部 downwind/crosswind 系下，m/s）
                    "vd": 0.0,
                    "vc": 0.0,
                }
            )
        if len(self.puffs) > self.max_puffs:
            self.puffs = self.puffs[-self.max_puffs :]

    def _advance_puffs(self) -> None:
        """推进所有 puff：随风移动、湍流扰动、扩散并逐渐衰减。"""
        if self.obstacle_flow.active:
            self._advance_puffs_with_obstacle_flow()
            return

        wind_x = self.wind_speed * math.cos(self.wind_direction)
        wind_y = self.wind_speed * math.sin(self.wind_direction)
        active_puffs = []
        relax = self.turbulence_vel_relax
        for puff in self.puffs:
            direction = float(puff.get("direction", self.wind_direction))
            downwind_x = math.cos(direction)
            downwind_y = math.sin(direction)
            crosswind_x = -downwind_y
            crosswind_y = downwind_x
            # OU 速度扰动：时间相关（AR(1)），而非每步独立白噪声 →
            # 同一 filament 的扰动在时间上连贯，形成蜿蜒的相干气缕。
            puff["vd"] = relax * float(puff.get("vd", 0.0)) + (
                self.turbulence_vel_std_downwind * float(self.rng.normal())
            )
            puff["vc"] = relax * float(puff.get("vc", 0.0)) + (
                self.turbulence_vel_std_crosswind * float(self.rng.normal())
            )
            puff["age"] += self.dt
            puff["x"] += (
                wind_x
                + puff["vd"] * downwind_x
                + puff["vc"] * crosswind_x
            ) * self.dt
            puff["y"] += (
                wind_y
                + puff["vd"] * downwind_y
                + puff["vc"] * crosswind_y
            ) * self.dt
            puff["sigma_downwind"] += self.diffusion_downwind_rate * self.dt
            puff["sigma_crosswind"] += self.diffusion_crosswind_rate * self.dt
            puff["mass"] *= math.exp(-self.decay_rate * self.dt)
            if puff["age"] > self.max_puff_age or puff["mass"] < self.min_puff_mass:
                continue
            if (
                puff["x"] < self.world_min - self.puff_bounds_margin
                or puff["x"] > self.world_max + self.puff_bounds_margin
                or puff["y"] < self.world_min - self.puff_bounds_margin
                or puff["y"] > self.world_max + self.puff_bounds_margin
            ):
                continue
            active_puffs.append(puff)
        self.puffs = active_puffs[-self.max_puffs :]

    def _local_puff_velocity_many(
        self,
        x: np.ndarray,
        y: np.ndarray,
        velocity_downwind: np.ndarray,
        velocity_crosswind: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """向量化合成一批 puff 的局部速度。"""
        mean_x, mean_y, turbulence = self.obstacle_flow.sample_many(
            x,
            y,
            wind_speed=self.wind_speed,
            direction_offset_rad=self.wind_dir_offset,
        )
        mean_speed = np.hypot(mean_x, mean_y)
        fallback_direction = (
            self.obstacle_flow.result.direction_rad + self.wind_dir_offset
        )
        downwind_x = np.divide(
            mean_x,
            mean_speed,
            out=np.full_like(mean_x, math.cos(fallback_direction)),
            where=mean_speed > 1e-10,
        )
        downwind_y = np.divide(
            mean_y,
            mean_speed,
            out=np.full_like(mean_y, math.sin(fallback_direction)),
            where=mean_speed > 1e-10,
        )
        return (
            mean_x + velocity_downwind * downwind_x - velocity_crosswind * downwind_y,
            mean_y + velocity_downwind * downwind_y + velocity_crosswind * downwind_x,
            turbulence,
            np.arctan2(downwind_y, downwind_x),
        )

    def _move_puff_without_crossing_obstacles(
        self,
        x: float,
        y: float,
        delta_x: float,
        delta_y: float,
    ) -> tuple[float, float]:
        """推进一个子步；发生数值穿墙时截断法向位移并保留切向位移。"""
        current_x = float(x)
        current_y = float(y)
        remaining_x = float(delta_x)
        remaining_y = float(delta_y)
        epsilon = min(1e-5, 0.01 * self.obstacle_flow.cell_size)
        for _ in range(3):
            end_x = current_x + remaining_x
            end_y = current_y + remaining_y
            nearest: tuple[float, float, float] | None = None
            for obstacle in self.obstacles:
                hit = segment_aabb_first_hit(
                    current_x,
                    current_y,
                    end_x,
                    end_y,
                    obstacle,
                )
                if hit is not None and (nearest is None or hit[0] < nearest[0]):
                    nearest = hit
            if nearest is None:
                return end_x, end_y

            hit_t, normal_x, normal_y = nearest
            safe_t = max(0.0, hit_t - epsilon / max(math.hypot(remaining_x, remaining_y), epsilon))
            current_x += safe_t * remaining_x + epsilon * normal_x
            current_y += safe_t * remaining_y + epsilon * normal_y
            untraveled = max(0.0, 1.0 - hit_t)
            remaining_x *= untraveled
            remaining_y *= untraveled
            inward = remaining_x * normal_x + remaining_y * normal_y
            if inward < 0.0:
                remaining_x -= inward * normal_x
                remaining_y -= inward * normal_y
            if math.hypot(remaining_x, remaining_y) <= epsilon:
                return current_x, current_y
        return current_x, current_y

    def _advance_puffs_with_obstacle_flow(self) -> None:
        """按稳态 LBM 局部速度推进 puff，并保证中心不会穿过矩形墙体。"""
        if self.obstacle_flow.result is None:
            raise RuntimeError("obstacle-aware puff advance requires a prepared LBM field")
        if not self.puffs:
            return
        puffs = self.puffs
        count = len(puffs)
        x = np.fromiter((float(puff["x"]) for puff in puffs), dtype=np.float64, count=count)
        y = np.fromiter((float(puff["y"]) for puff in puffs), dtype=np.float64, count=count)
        velocity_downwind = np.fromiter(
            (float(puff.get("vd", 0.0)) for puff in puffs),
            dtype=np.float64,
            count=count,
        )
        velocity_crosswind = np.fromiter(
            (float(puff.get("vc", 0.0)) for puff in puffs),
            dtype=np.float64,
            count=count,
        )
        relax = self.turbulence_vel_relax
        max_substep_distance = (
            self.obstacle_flow.max_cfl_fraction * self.obstacle_flow.cell_size
        )
        _, _, local_turbulence, _ = self._local_puff_velocity_many(
            x,
            y,
            velocity_downwind,
            velocity_crosswind,
        )
        noise_scale = np.sqrt(local_turbulence)
        velocity_downwind = (
            relax * velocity_downwind
            + self.turbulence_vel_std_downwind
            * noise_scale
            * self.rng.normal(size=count)
        )
        velocity_crosswind = (
            relax * velocity_crosswind
            + self.turbulence_vel_std_crosswind
            * noise_scale
            * self.rng.normal(size=count)
        )
        velocity_x, velocity_y, _, direction = self._local_puff_velocity_many(
            x,
            y,
            velocity_downwind,
            velocity_crosswind,
        )
        # CFL 约束针对空间变化的 LBM 平均场；OU 阵风在一个环境步内视为常速度，
        # 可直接由 swept-AABB 处理，不让少数随机速度离群值拖慢所有 puff。
        mean_x, mean_y, _ = self.obstacle_flow.sample_many(
            x,
            y,
            wind_speed=self.wind_speed,
            direction_offset_rad=self.wind_dir_offset,
        )
        substep_counts = np.clip(
            np.ceil(np.hypot(mean_x, mean_y) * self.dt / max_substep_distance),
            1,
            16,
        ).astype(np.int32)
        max_substeps = int(np.max(substep_counts))
        accumulated_turbulence = np.zeros(count, dtype=np.float64)
        for substep_index in range(max_substeps):
            active = substep_counts > substep_index
            active_dt = self.dt / substep_counts[active]
            velocity_x, velocity_y, turbulence, _ = self._local_puff_velocity_many(
                x[active],
                y[active],
                velocity_downwind[active],
                velocity_crosswind[active],
            )
            midpoint_x = x[active] + 0.5 * active_dt * velocity_x
            midpoint_y = y[active] + 0.5 * active_dt * velocity_y
            mid_x, mid_y, mid_turbulence, mid_direction = self._local_puff_velocity_many(
                midpoint_x,
                midpoint_y,
                velocity_downwind[active],
                velocity_crosswind[active],
            )
            delta_x = active_dt * mid_x
            delta_y = active_dt * mid_y
            moved_x, moved_y = self._move_puffs_without_crossing_obstacles(
                x[active],
                y[active],
                delta_x,
                delta_y,
            )
            x[active] = moved_x
            y[active] = moved_y
            accumulated_turbulence[active] += 0.5 * (
                turbulence + mid_turbulence
            )
            direction[active] = mid_direction

        active_puffs = []
        mean_turbulence = accumulated_turbulence / substep_counts
        for index, puff in enumerate(puffs):
            puff["x"] = float(x[index])
            puff["y"] = float(y[index])
            puff["vd"] = float(velocity_downwind[index])
            puff["vc"] = float(velocity_crosswind[index])
            puff["direction"] = float(direction[index])
            puff["age"] += self.dt
            puff["sigma_downwind"] += self.diffusion_downwind_rate * self.dt
            puff["sigma_crosswind"] += (
                self.diffusion_crosswind_rate * mean_turbulence[index] * self.dt
            )
            puff["mass"] *= math.exp(-self.decay_rate * self.dt)
            if puff["age"] > self.max_puff_age or puff["mass"] < self.min_puff_mass:
                continue
            if (
                puff["x"] < self.world_min - self.puff_bounds_margin
                or puff["x"] > self.world_max + self.puff_bounds_margin
                or puff["y"] < self.world_min - self.puff_bounds_margin
                or puff["y"] > self.world_max + self.puff_bounds_margin
            ):
                continue
            active_puffs.append(puff)
        self.puffs = active_puffs[-self.max_puffs :]

    def _move_puffs_without_crossing_obstacles(
        self,
        x: np.ndarray,
        y: np.ndarray,
        delta_x: np.ndarray,
        delta_y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """批量执行 swept AABB 墙面修正，正常情况下完全走 NumPy 路径。"""
        current_x = np.asarray(x, dtype=np.float64).copy()
        current_y = np.asarray(y, dtype=np.float64).copy()
        remaining_x = np.asarray(delta_x, dtype=np.float64).copy()
        remaining_y = np.asarray(delta_y, dtype=np.float64).copy()
        first_end_x = current_x + remaining_x
        first_end_y = current_y + remaining_y
        active = np.zeros(current_x.shape, dtype=bool)
        for xmin, xmax, ymin, ymax in self.obstacles:
            active |= (
                (np.minimum(current_x, first_end_x) <= xmax)
                & (np.maximum(current_x, first_end_x) >= xmin)
                & (np.minimum(current_y, first_end_y) <= ymax)
                & (np.maximum(current_y, first_end_y) >= ymin)
            )
        no_possible_hit = ~active
        current_x[no_possible_hit] = first_end_x[no_possible_hit]
        current_y[no_possible_hit] = first_end_y[no_possible_hit]
        epsilon = min(1e-5, 0.01 * self.obstacle_flow.cell_size)
        for _ in range(3):
            indices = np.flatnonzero(active)
            if indices.size == 0:
                break
            start_x = current_x[indices]
            start_y = current_y[indices]
            end_x = start_x + remaining_x[indices]
            end_y = start_y + remaining_y[indices]
            best_t = np.full(indices.size, np.inf, dtype=np.float64)
            best_normal_x = np.zeros(indices.size, dtype=np.float64)
            best_normal_y = np.zeros(indices.size, dtype=np.float64)
            for obstacle in self.obstacles:
                hit, hit_t, normal_x, normal_y = segment_aabb_first_hit_many(
                    start_x,
                    start_y,
                    end_x,
                    end_y,
                    obstacle,
                )
                nearer = hit & (hit_t < best_t)
                best_t[nearer] = hit_t[nearer]
                best_normal_x[nearer] = normal_x[nearer]
                best_normal_y[nearer] = normal_y[nearer]

            no_hit = ~np.isfinite(best_t)
            if np.any(no_hit):
                no_hit_indices = indices[no_hit]
                current_x[no_hit_indices] = end_x[no_hit]
                current_y[no_hit_indices] = end_y[no_hit]
                active[no_hit_indices] = False

            hit_local = ~no_hit
            if not np.any(hit_local):
                continue
            hit_indices = indices[hit_local]
            hit_t = best_t[hit_local]
            normal_x = best_normal_x[hit_local]
            normal_y = best_normal_y[hit_local]
            distance = np.hypot(
                remaining_x[hit_indices], remaining_y[hit_indices]
            )
            safe_t = np.maximum(
                0.0,
                hit_t - epsilon / np.maximum(distance, epsilon),
            )
            current_x[hit_indices] += (
                safe_t * remaining_x[hit_indices] + epsilon * normal_x
            )
            current_y[hit_indices] += (
                safe_t * remaining_y[hit_indices] + epsilon * normal_y
            )
            untraveled = np.maximum(0.0, 1.0 - hit_t)
            remaining_x[hit_indices] *= untraveled
            remaining_y[hit_indices] *= untraveled
            inward = (
                remaining_x[hit_indices] * normal_x
                + remaining_y[hit_indices] * normal_y
            )
            inward_mask = inward < 0.0
            if np.any(inward_mask):
                inward_indices = hit_indices[inward_mask]
                remaining_x[inward_indices] -= inward[inward_mask] * normal_x[inward_mask]
                remaining_y[inward_indices] -= inward[inward_mask] * normal_y[inward_mask]
            stopped = (
                np.hypot(remaining_x[hit_indices], remaining_y[hit_indices])
                <= epsilon
            )
            active[hit_indices[stopped]] = False
        return current_x, current_y

    def advance(self) -> None:
        """推进一次动态气味场。"""
        self._update_wind()
        self._emit_puffs()
        self._advance_puffs()

    def _puff_concentration(self, x: float, y: float) -> float:
        """计算所有 puff 在指定点叠加得到的瞬时浓度。"""
        if self.obstacle_flow.active and self.puffs:
            count = len(self.puffs)
            puff_x = np.fromiter(
                (float(puff["x"]) for puff in self.puffs),
                dtype=np.float64,
                count=count,
            )
            puff_y = np.fromiter(
                (float(puff["y"]) for puff in self.puffs),
                dtype=np.float64,
                count=count,
            )
            direction = np.fromiter(
                (
                    float(puff.get("direction", self.wind_direction))
                    for puff in self.puffs
                ),
                dtype=np.float64,
                count=count,
            )
            mass = np.fromiter(
                (float(puff["mass"]) for puff in self.puffs),
                dtype=np.float64,
                count=count,
            )
            sigma_downwind = np.fromiter(
                (
                    float(puff.get("sigma_downwind", puff.get("sigma", 0.08)))
                    for puff in self.puffs
                ),
                dtype=np.float64,
                count=count,
            )
            sigma_crosswind = np.fromiter(
                (
                    float(puff.get("sigma_crosswind", puff.get("sigma", 0.04)))
                    for puff in self.puffs
                ),
                dtype=np.float64,
                count=count,
            )
            dx = float(x) - puff_x
            dy = float(y) - puff_y
            cosine = np.cos(direction)
            sine = np.sin(direction)
            downwind = dx * cosine + dy * sine
            crosswind = -dx * sine + dy * cosine
            contributions = (
                mass
                * np.exp(
                    -(
                        downwind * downwind / (2.0 * sigma_downwind**2)
                        + crosswind * crosswind / (2.0 * sigma_crosswind**2)
                    )
                )
                / (2.0 * math.pi * sigma_downwind * sigma_crosswind)
            )
            if self.obstacle_flow.wall_transmission < 1.0:
                blocked = np.zeros(count, dtype=bool)
                end_x = np.full(count, float(x), dtype=np.float64)
                end_y = np.full(count, float(y), dtype=np.float64)
                for obstacle in self.obstacles:
                    hit, _, _, _ = segment_aabb_first_hit_many(
                        puff_x,
                        puff_y,
                        end_x,
                        end_y,
                        obstacle,
                    )
                    blocked |= hit
                contributions[blocked] *= self.obstacle_flow.wall_transmission
            return float(np.sum(contributions))

        total = 0.0
        for puff in self.puffs:
            dx = x - puff["x"]
            dy = y - puff["y"]
            direction = float(puff.get("direction", self.wind_direction))
            downwind = dx * math.cos(direction) + dy * math.sin(direction)
            crosswind = -dx * math.sin(direction) + dy * math.cos(direction)
            sigma_downwind = float(puff.get("sigma_downwind", puff.get("sigma", 0.08)))
            sigma_crosswind = float(puff.get("sigma_crosswind", puff.get("sigma", 0.04)))
            contribution = puff["mass"] * math.exp(
                -(
                    downwind * downwind / (2.0 * sigma_downwind * sigma_downwind)
                    + crosswind * crosswind / (2.0 * sigma_crosswind * sigma_crosswind)
                )
            ) / (2.0 * math.pi * sigma_downwind * sigma_crosswind)
            total += contribution
        return total

    def concentration(self, x: float, y: float, add_noise: bool = True) -> float:
        """返回指定坐标处的气味浓度。"""
        if self.obstacle_flow.active:
            for xmin, xmax, ymin, ymax in self.obstacles:
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    return float(self.gas_background)
        source_core = float(self._source_core_concentration(x, y))
        if self.obstacle_flow.active and self.obstacle_flow.wall_transmission < 1.0:
            if any(
                segment_aabb_first_hit(
                    self.source_x,
                    self.source_y,
                    float(x),
                    float(y),
                    obstacle,
                )
                is not None
                for obstacle in self.obstacles
            ):
                source_core *= self.obstacle_flow.wall_transmission
        concentration = source_core
        concentration += float(self._puff_concentration(x, y))
        concentration += self.gas_background
        if add_noise:
            concentration += float(self.rng.normal(0.0, self.gas_noise_std))
        return max(0.0, concentration)

    def grid(
        self,
        resolution: int = 80,
        add_noise: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回整张二维气味场网格，主要用于绘图和动画。"""
        xs = np.linspace(self.world_min, self.world_max, resolution, dtype=np.float32)
        ys = np.linspace(self.world_min, self.world_max, resolution, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        source_core = self._source_core_concentration(grid_x, grid_y).astype(np.float32)
        if self.obstacle_flow.active and self.obstacle_flow.wall_transmission < 1.0:
            blocked = np.zeros(grid_x.shape, dtype=bool)
            for obstacle in self.obstacles:
                blocked |= segment_block_mask(
                    self.source_x,
                    self.source_y,
                    grid_x,
                    grid_y,
                    obstacle,
                )
            source_core[blocked] *= self.obstacle_flow.wall_transmission
        concentration = source_core
        concentration += self.gas_background
        for puff in self.puffs:
            dx = grid_x - puff["x"]
            dy = grid_y - puff["y"]
            direction = float(puff.get("direction", self.wind_direction))
            downwind = dx * math.cos(direction) + dy * math.sin(direction)
            crosswind = -dx * math.sin(direction) + dy * math.cos(direction)
            sigma_downwind = float(puff.get("sigma_downwind", puff.get("sigma", 0.08)))
            sigma_crosswind = float(puff.get("sigma_crosswind", puff.get("sigma", 0.04)))
            contribution = (
                puff["mass"]
                * np.exp(
                    -(
                        downwind**2 / (2.0 * sigma_downwind * sigma_downwind)
                        + crosswind**2 / (2.0 * sigma_crosswind * sigma_crosswind)
                    )
                )
                / (2.0 * np.pi * sigma_downwind * sigma_crosswind)
            ).astype(np.float32)
            if (
                self.obstacle_flow.active
                and self.obstacle_flow.wall_transmission < 1.0
            ):
                blocked = np.zeros(grid_x.shape, dtype=bool)
                for obstacle in self.obstacles:
                    blocked |= segment_block_mask(
                        float(puff["x"]),
                        float(puff["y"]),
                        grid_x,
                        grid_y,
                        obstacle,
                    )
                if self.obstacle_flow.wall_transmission == 0.0:
                    contribution[blocked] = 0.0
                else:
                    contribution[blocked] *= self.obstacle_flow.wall_transmission
            concentration += contribution
        if add_noise:
            concentration += self.rng.normal(
                0.0, self.gas_noise_std, size=concentration.shape
            ).astype(np.float32)
        if self.obstacle_flow.active:
            for xmin, xmax, ymin, ymax in self.obstacles:
                inside = (
                    (grid_x >= xmin)
                    & (grid_x <= xmax)
                    & (grid_y >= ymin)
                    & (grid_y <= ymax)
                )
                concentration[inside] = self.gas_background
        concentration = np.maximum(concentration, 0.0)
        return xs, ys, concentration.astype(np.float32, copy=False)

    def _sample_source_position(self) -> tuple[float, float]:
        """随机采样场外上风侧气源位置，用于兼容旧调试入口。"""
        x = self.rng.uniform(self.world_min - self.far_source_distance, self.world_min - 0.25)
        y = self.rng.uniform(self.world_min, self.world_max)
        return float(x), float(y)

    def _warmup_puffs(self) -> None:
        """预先推进 puff，使 reset 后的气味场不是空场。"""
        self.puffs = []
        for _ in range(self.puff_warmup_steps):
            self.advance()

    def _randomize_params(self) -> None:
        """在标称值附近随机化羽流物理参数（域随机化）。"""
        n = self._dr_nominal
        r = self.rng
        self.puff_release_per_step = int(r.integers(1, 4))
        self.puff_init_mass = n["puff_init_mass"] * float(r.uniform(0.7, 1.5))
        self.diffusion_downwind_rate = n["diffusion_downwind_rate"] * float(r.uniform(0.7, 1.4))
        self.diffusion_crosswind_rate = n["diffusion_crosswind_rate"] * float(r.uniform(0.7, 1.4))
        self.decay_rate = n["decay_rate"] * float(r.uniform(0.7, 1.4))
        self.turbulence_vel_std_crosswind = n["turbulence_vel_std_crosswind"] * float(
            r.uniform(0.7, 1.4)
        )
        self.wind_dir_meander_sigma = n["wind_dir_meander_sigma"] * float(r.uniform(0.6, 1.4))
        self.max_puff_age = n["max_puff_age"] * float(r.uniform(0.7, 1.3))

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.domain_randomization:
            self._randomize_params()
        self._sample_wind()
        self._place_source_upwind()
        self.obstacle_flow.prepare(self.wind_direction_base)
        self._warmup_puffs()

    # 兼容旧可视化/调试命名，后续新代码优先使用 metadata/concentration/grid/advance。
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


class WhiskerOnlyPuffEnv(gym.Env):
    """固定机器人、只控制左右触须的 Gymnasium 环境。"""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        seed = int(cfg.get("seed", 0))
        self.max_steps = int(cfg.get("max_steps", 300))
        # 奖励阈值/系数已按改造后羽流的浓度量级重标定（传感器读数典型 0.3~1.3）。
        # hit_threshold 仍保持较低，用于观测 hit-rate 特征和初始位姿采样；
        # strong_threshold 才是“强响应” bonus 的门限，按新量级设置。
        self.hit_threshold = float(cfg.get("hit_threshold", 0.2))
        self.strong_threshold = float(cfg.get("strong_threshold", 0.60))
        # 主要奖励来自连续、依赖动作的项：浓度幅值 + 上升趋势 + 左右对比。
        self.odor_hit_reward = float(cfg.get("odor_hit_reward", 0.08))
        self.presence_clip = float(cfg.get("presence_clip", 1.5))
        self.trend_reward_scale = float(cfg.get("trend_reward_scale", 1.0))
        self.trend_clip = float(cfg.get("trend_clip", 0.12))
        self.contrast_reward_scale = float(cfg.get("contrast_reward_scale", 0.15))
        # 近乎常驻的二值 bonus（在新羽流下几乎一直为真）权重调小，避免淹没可学习信号。
        self.tracking_bonus = float(cfg.get("tracking_bonus", 0.05))
        self.strong_bonus = float(cfg.get("strong_bonus", 0.05))
        self.time_penalty = float(cfg.get("time_penalty", 0.04))

        # 场地半宽可配置（默认 ±0.5 = 1m）。移动机器人环境用更大场地以显现追踪。
        self.world_half = float(cfg.get("world_half", WORLD_MAX))
        self.world_min = -self.world_half
        self.world_max = self.world_half

        # 与 `PlumeEnv` 保持一致：环境持有一个 plume 对象负责气味场。
        source_position = cfg.get("source_position")
        self.plume = DynamicPuffPlume(
            seed=seed,
            source_position=tuple(source_position) if source_position is not None else None,
            dt=float(cfg.get("dt", 0.2)),
            gas_field_mode=str(cfg.get("gas_field_mode", "puff")),
            wind_speed_range=tuple(cfg.get("wind_speed_range", (0.06, 0.11))),
            wind_sampling_mode=str(cfg.get("wind_sampling_mode", "random")),
            obstacles=cfg.get("obstacles"),
            obstacle_flow=cfg.get("obstacle_flow"),
            plume_overrides=cfg.get("plume_overrides"),
            world_half=self.world_half,
        )
        self.field = self.plume  # 兼容旧可视化脚本中的 env.field 访问。
        self.whiskers = DualWhiskerSampler(
            length=float(cfg.get("whisker_length", 0.15)),
            sector_count=int(cfg.get("whisker_sector_count", 10)),
            servo_60deg_time_s=float(cfg.get("servo_60deg_time_s", 0.12)),
        )
        self.rng = np.random.default_rng(seed)
        self.dt = float(cfg.get("dt", 0.2))
        self.sensor_model = str(cfg.get("sensor_model", "asymmetric"))
        self.sensor_alpha = float(cfg.get("sensor_alpha", 0.95))
        # 非对称传感器参数（默认贴近 MQ-3：响应快、恢复慢）。
        self.sensor_response_tau = float(cfg.get("sensor_response_tau", 0.6))
        self.sensor_recovery_tau = float(cfg.get("sensor_recovery_tau", 3.0))
        self.sensor_noise_std = float(cfg.get("sensor_noise_std", 0.006))
        self.sensor_baseline_drift_std = float(cfg.get("sensor_baseline_drift_std", 0.0))
        self.sensor_baseline_tau = float(cfg.get("sensor_baseline_tau", 60.0))
        self._sensor_nominal = {
            "response_tau": self.sensor_response_tau,
            "recovery_tau": self.sensor_recovery_tau,
            "noise_std": self.sensor_noise_std,
        }
        self.left_sensor = self._build_sensor()
        self.right_sensor = self._build_sensor()

        # 域随机化：同时随机化羽流物理和传感器特性。默认关闭（便于复现/诊断），
        # 训练时建议在配置中打开 domain_randomization=True 以提高 sim-to-real 鲁棒性。
        self.domain_randomization = bool(cfg.get("domain_randomization", False))
        self.plume.domain_randomization = self.domain_randomization

        # 观测硬件化（阶段三）：默认 "hardware" 只用真机可获得特征，读数走和硬件
        # 一致的预处理；"privileged" 保留旧观测（含真实风向/气源距离）做消融上界。
        self.observation_mode = str(cfg.get("observation_mode", "hardware"))
        if self.observation_mode not in ("hardware", "privileged"):
            raise ValueError(
                f"observation_mode must be 'hardware' or 'privileged', got {self.observation_mode!r}"
            )
        # 仿真预处理参数：sim_sensor_scale 是关键标定值（仿真读数 ~0-1.5，
        # scale=0.35 使强响应归一化到 ~1-3、洁净接近 0，不撞 clip）。硬件用另一套
        # scale（如 1500），但 builder/config 形状共享。
        preprocess_config = SensorPreprocessConfig(
            dt_s=float(cfg.get("preprocess_dt_s", self.dt)),
            baseline_tau_s=float(cfg.get("preprocess_baseline_tau_s", 180.0)),
            smooth_tau_s=float(cfg.get("preprocess_smooth_tau_s", 2.0)),
            response_tau_s=float(cfg.get("preprocess_response_tau_s", 2.0)),
            trend_window=int(cfg.get("preprocess_trend_window", 8)),
            scale=float(cfg.get("sim_sensor_scale", 0.35)),
            clip=float(cfg.get("preprocess_clip", 5.0)),
            baseline_update_signal_threshold=float(
                cfg.get("preprocess_baseline_update_signal_threshold", 0.20)
            ),
            baseline_update_trend_threshold=float(
                cfg.get("preprocess_baseline_update_trend_threshold", 0.05)
            ),
        )
        self.observation_builder = WhiskerObservationBuilder(
            preprocess_config, self.whiskers.sector_count
        )
        self.observation_field_names = self.observation_builder.field_names
        # 记录上一动作的指令扇区（硬件可得），用于观测的扇区/角度特征。
        self.last_left_sector = 0
        self.last_right_sector = 0
        self._last_obs: np.ndarray | None = None

        self.robot_state = RobotState(0.0, 0.0, 0.0)
        self.action_space = spaces.MultiDiscrete(
            [self.whiskers.sector_count, self.whiskers.sector_count]
        )
        # hardware 模式用 builder 提供的有限边界（比 ±inf 更规范）；privileged 旧
        # 向量语义不同（含 cos/sin、归一化距离），仍用 ±inf 兜底。
        obs_dim = self.observation_builder.dim
        if self.observation_mode == "hardware":
            obs_low = self.observation_builder.low
            obs_high = self.observation_builder.high
        else:
            obs_low = np.full(obs_dim, -np.inf, dtype=np.float32)
            obs_high = np.full(obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # 初始位姿模式（实验旋钮）：控制机器人/触须起始落点相对羽流的密度。
        # in_plume  = 浓度密集区（易拿信号，旧默认，会让奖励被“羽流存在”主导）；
        # plume_edge= 羽流边缘/弱覆盖区（whiff/blank 间歇明显，最能考验主动采样）；
        # uniform   = 全场均匀（起始常在羽流外，信号最稀疏）。
        self.init_pose_mode = str(cfg.get("init_pose_mode", "plume_edge"))
        if self.init_pose_mode not in ("in_plume", "plume_edge", "uniform"):
            raise ValueError(
                f"init_pose_mode must be 'in_plume'/'plume_edge'/'uniform', got {self.init_pose_mode!r}"
            )
        # plume_edge 用“活跃格子”浓度分布的分位带定义边缘（默认取 10~50 分位的弱覆盖）。
        self.init_edge_pct_low = float(cfg.get("init_edge_pct_low", 10.0))
        self.init_edge_pct_high = float(cfg.get("init_edge_pct_high", 50.0))

        self.step_count = 0
        self.prev_left = 0.0
        self.prev_right = 0.0
        self.left_hits: list[float] = []
        self.right_hits: list[float] = []
        self.trajectory: list[dict[str, Any]] = []

    def _randomize_sensors(self) -> None:
        """在标称值附近随机化传感器特性，并引入基线漂移（域随机化）。"""
        n = self._sensor_nominal
        r = self.rng
        self.sensor_response_tau = n["response_tau"] * float(r.uniform(0.6, 1.6))
        self.sensor_recovery_tau = n["recovery_tau"] * float(r.uniform(0.6, 1.8))
        self.sensor_noise_std = n["noise_std"] * float(r.uniform(0.5, 2.0))
        # DR 下引入随机基线漂移，模拟真实 MQ-3 每次上电基线不一致。
        self.sensor_baseline_drift_std = float(r.uniform(0.0, 0.010))

    def _build_sensor(self) -> FirstOrderGasSensor | AsymmetricGasSensor:
        """按配置构造单个气体传感器。"""
        if self.sensor_model == "first_order":
            return FirstOrderGasSensor(self.sensor_alpha)
        return AsymmetricGasSensor(
            response_tau=self.sensor_response_tau,
            recovery_tau=self.sensor_recovery_tau,
            dt=self.dt,
            noise_std=self.sensor_noise_std,
            baseline_drift_std=self.sensor_baseline_drift_std,
            baseline_tau=self.sensor_baseline_tau,
            rng=self.rng,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.domain_randomization:
            self._randomize_sensors()
        # 重建传感器：使其使用当前 rng 和（可能已随机化的）参数。
        self.left_sensor = self._build_sensor()
        self.right_sensor = self._build_sensor()
        self.plume.reset(seed)
        self.step_count = 0
        self.prev_left = 0.0
        self.prev_right = 0.0
        self.left_hits.clear()
        self.right_hits.clear()
        self.trajectory.clear()
        self.left_sensor.reset()
        self.right_sensor.reset()
        self.whiskers.reset()
        # 触须 reset 后两侧都回到 0 号扇区中心，指令扇区跟着复位。
        self.last_left_sector = 0
        self.last_right_sector = 0
        # init_baseline=None：用首帧读数锚基线，与硬件上电锚基线一致。
        self.observation_builder.reset(init_baseline=None)

        self.robot_state = self.sample_initial_robot_pose()
        obs = self._build_observation()
        return obs, self._current_info()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        left_sector, right_sector = self._decode_action(action)
        # 记录指令扇区（硬件可得的“上一动作”），供观测的扇区/角度特征使用。
        self.last_left_sector = left_sector
        self.last_right_sector = right_sector
        self.plume.advance()
        whisker_state = self.whiskers.step(
            left_sector,
            right_sector,
            self.robot_state,
            dt=self.plume.dt,
        )

        # raw_* 是触须采样点处的瞬时气味浓度；left/right 是慢响应传感器读数。
        raw_left = self.plume.concentration(*whisker_state.left_point)
        raw_right = self.plume.concentration(*whisker_state.right_point)
        left = self.left_sensor.update(raw_left)
        right = self.right_sensor.update(raw_right)

        hit_left = 1.0 if left >= self.hit_threshold else 0.0
        hit_right = 1.0 if right >= self.hit_threshold else 0.0
        self.left_hits.append(hit_left)
        self.right_hits.append(hit_right)
        self.left_hits = self.left_hits[-20:]
        self.right_hits = self.right_hits[-20:]

        reward = self.get_reward(left, right)
        self.step_count += 1
        self.trajectory.append(
            {
                "x": self.robot_state.x,
                "y": self.robot_state.y,
                "heading": self.robot_state.heading,
                "left": left,
                "right": right,
                "raw_left": raw_left,
                "raw_right": raw_right,
                "left_angle": whisker_state.left_angle,
                "right_angle": whisker_state.right_angle,
                "left_sector": int(left_sector),
                "right_sector": int(right_sector),
                "reward": reward,
            }
        )

        self.prev_left = left
        self.prev_right = right
        obs = self._build_observation()
        info = self._current_info()
        info.update(
            {
                "left_sector": int(left_sector),
                "right_sector": int(right_sector),
                "left": left,
                "right": right,
                "raw_left": raw_left,
                "raw_right": raw_right,
                "wind_direction": self.plume.wind_direction,
                "wind_speed": self.plume.wind_speed,
                "source": (self.plume.source_x, self.plume.source_y),
            }
        )
        terminated = False
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, info

    def concentration_grid(
        self,
        resolution: int = 80,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.plume.grid(resolution=resolution)

    def get_map_metadata(self) -> dict[str, Any]:
        metadata = self.plume.metadata()
        metadata.update(
            {
                "robot_position": (float(self.robot_state.x), float(self.robot_state.y)),
                "robot_heading": float(self.robot_state.heading),
                "whisker_sector_count": int(self.whiskers.sector_count),
            }
        )
        return metadata

    def sample_initial_robot_pose(self) -> RobotState:
        """按 init_pose_mode 选择初始位姿采样策略。"""
        if self.init_pose_mode == "uniform":
            return self.sample_robot_pose_uniform()
        if self.init_pose_mode == "plume_edge":
            return self.sample_robot_pose_plume_edge()
        return self.sample_robot_pose_in_plume()

    def _sample_from_grid_candidates(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
    ) -> RobotState | None:
        """在候选格子里带有效性检查地随机选点；全部失败返回 None。"""
        for _ in range(128):
            idx = int(self.rng.integers(0, rows.size))
            x = float(xs[cols[idx]])
            y = float(ys[rows[idx]])
            if not self.plume.is_position_valid(x, y, margin=0.16):
                continue
            if math.hypot(x - self.plume.source_x, y - self.plume.source_y) < 0.10:
                continue
            theta = self.rng.uniform(-np.pi, np.pi)
            return RobotState(x, y, float(theta))
        return None

    def sample_robot_pose_in_plume(self) -> RobotState:
        """把机器人/触须初始化到当前羽流覆盖的密集区域（浓度高分位）。

        起始就在密集区会让触须几乎每步都读到气味，奖励被“羽流存在”主导、
        主动采样缺乏可学信号；因此这不再是默认，仅作对照。
        """
        xs, ys, concentration = self.plume.grid(resolution=60, add_noise=False)
        threshold = max(
            self.hit_threshold * 0.35,
            float(np.percentile(concentration, 70.0)),
        )
        rows, cols = np.where(concentration >= threshold)
        if rows.size == 0:
            return self.sample_robot_pose_uniform()
        pose = self._sample_from_grid_candidates(xs, ys, rows, cols)
        return pose if pose is not None else self.sample_robot_pose_uniform()

    def sample_robot_pose_plume_edge(self) -> RobotState:
        """把机器人落在羽流边缘 / 弱覆盖区：局部有微弱、间歇气味但非密集核心。

        这样左右触须指向不同扇区时更容易产生 whiff/blank 差异与左右不对称，是
        检验“主动采样是否比固定/随机更有信息”的关键场景。做法：取“活跃格子”
        （浓度高于背景噪声）的浓度分布，选落在 [init_edge_pct_low, init_edge_pct_high]
        分位带内的弱覆盖格子；活跃格子太少时退化为均匀采样。
        """
        xs, ys, concentration = self.plume.grid(resolution=60, add_noise=False)
        active_floor = self.hit_threshold * 0.25
        active_mask = concentration > active_floor
        if not np.any(active_mask):
            return self.sample_robot_pose_uniform()
        active_vals = concentration[active_mask]
        lo = float(np.percentile(active_vals, self.init_edge_pct_low))
        hi = float(np.percentile(active_vals, self.init_edge_pct_high))
        band_mask = active_mask & (concentration >= lo) & (concentration <= hi)
        rows, cols = np.where(band_mask)
        if rows.size == 0:
            return self.sample_robot_pose_uniform()
        pose = self._sample_from_grid_candidates(xs, ys, rows, cols)
        return pose if pose is not None else self.sample_robot_pose_uniform()

    def sample_robot_pose_uniform(self) -> RobotState:
        """兜底采样：如果羽流覆盖区候选失败，则在 1m 场地内均匀采样。"""
        while True:
            x = self.rng.uniform(self.world_min + 0.16, self.world_max - 0.16)
            y = self.rng.uniform(self.world_min + 0.16, self.world_max - 0.16)
            if math.hypot(x - self.plume.source_x, y - self.plume.source_y) < 0.10:
                continue
            theta = self.rng.uniform(-np.pi, np.pi)
            return RobotState(float(x), float(y), float(theta))

    def _build_observation(self) -> np.ndarray:
        """构造并缓存本帧观测。**有状态**：hardware 模式会推进一次预处理器。

        每帧只应在 reset()/step() 各调一次。可视化脚本另外调用的 `_observe()`
        是无副作用的薄壳，只返回这里缓存的 `self._last_obs`，不会重复推进预处理。
        """
        whisker_state = self.whiskers.state(self.robot_state)
        if self.observation_mode == "privileged":
            obs = self._build_privileged_observation(whisker_state)
        else:
            obs = self.observation_builder.build(
                self.left_sensor.value,
                self.right_sensor.value,
                whisker_state.left_angle,
                whisker_state.right_angle,
                self.last_left_sector,
                self.last_right_sector,
            )
        self._last_obs = obs
        return obs

    def _build_privileged_observation(self, whisker_state: Any) -> np.ndarray:
        """旧的 12 维特权观测（含真实风向、气源距离），仅用于消融上界对照。"""
        left = self.left_sensor.value
        right = self.right_sensor.value
        wind_rel = self._wrap_angle(self.plume.wind_direction - self.robot_state.heading)
        source_distance = math.hypot(
            self.robot_state.x - self.plume.source_x,
            self.robot_state.y - self.plume.source_y,
        )
        return np.array(
            [
                left,
                right,
                left - right,
                left - self.prev_left,
                right - self.prev_right,
                float(np.mean(self.left_hits)) if self.left_hits else 0.0,
                float(np.mean(self.right_hits)) if self.right_hits else 0.0,
                whisker_state.left_angle / np.pi,
                whisker_state.right_angle / np.pi,
                math.cos(wind_rel),
                math.sin(wind_rel),
                source_distance / (self.world_max - self.world_min),
            ],
            dtype=np.float32,
        )

    def _current_info(self) -> dict[str, Any]:
        """无副作用地组装 info（不推进预处理器），键与改动前一致。"""
        return {
            "robot_state": self.robot_state,
            "whisker_state": self.whiskers.state(self.robot_state),
            "step": self.step_count,
        }

    def _observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        """无副作用薄壳：返回最近一次构造的观测 + 当前 info。

        保留此方法是为了兼容可视化脚本对 `env._observe()` 的额外调用；它不会
        重复推进有状态的预处理器。若 `_build_observation()` 尚未被调用（异常路径），
        退化为构造一次。
        """
        if self._last_obs is None:
            return self._build_observation(), self._current_info()
        return self._last_obs, self._current_info()

    def get_reward(self, left: float, right: float) -> float:
        gas_concentration = max(left, right)
        gas_trend = max(left - self.prev_left, right - self.prev_right)
        in_plume = gas_concentration >= self.hit_threshold
        strong_plume = gas_concentration >= self.strong_threshold

        # 连续、依赖动作的主项：上升趋势、浓度幅值、左右对比。
        gas_trend_reward = self.trend_reward_scale * float(
            np.clip(gas_trend, -self.trend_clip, self.trend_clip)
        )
        gas_presence_reward = self.odor_hit_reward * float(
            np.clip(gas_concentration, 0.0, self.presence_clip)
        )
        contrast_reward = self.contrast_reward_scale * abs(left - right)
        # 二值 bonus 权重已调小：仅作弱引导，避免在新量级下变成常驻偏置淹没梯度。
        plume_tracking_bonus = self.tracking_bonus if in_plume else 0.0
        strong_plume_bonus = self.strong_bonus if strong_plume else 0.0
        return float(
            gas_trend_reward
            + gas_presence_reward
            + plume_tracking_bonus
            + strong_plume_bonus
            + contrast_reward
            - self.time_penalty
        )

    def _decode_action(self, action: Any) -> tuple[int, int]:
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.size != 2:
            raise ValueError(
                "WhiskerOnlyPuffEnv action must contain [left_sector, right_sector]"
            )
        return int(arr[0]), int(arr[1])

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
