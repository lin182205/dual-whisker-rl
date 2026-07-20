"""触须独立控制环境：固定机器人，仅训练左右触须主动采样。"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

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
        self.obstacles = (
            np.asarray(obstacles, dtype=np.float32).copy()
            if obstacles is not None
            else DEFAULT_OBSTACLES.copy()
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
        self.wind_direction_base = float(
            self.rng.uniform(math.radians(-12.0), math.radians(12.0))
        )
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
        cross_offset = float(self.rng.uniform(*self.far_source_crosswind_range))
        self.source_x = (
            0.0
            - self.far_source_distance * wind_x
            + cross_offset * cross_x
        )
        self.source_y = (
            0.0
            - self.far_source_distance * wind_y
            + cross_offset * cross_y
        )

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

    def advance(self) -> None:
        """推进一次动态气味场。"""
        self._update_wind()
        self._emit_puffs()
        self._advance_puffs()

    def _puff_concentration(self, x: float, y: float) -> float:
        """计算所有 puff 在指定点叠加得到的瞬时浓度。"""
        total = 0.0
        for puff in self.puffs:
            dx = x - puff["x"]
            dy = y - puff["y"]
            direction = float(puff.get("direction", self.wind_direction))
            downwind = dx * math.cos(direction) + dy * math.sin(direction)
            crosswind = -dx * math.sin(direction) + dy * math.cos(direction)
            sigma_downwind = float(puff.get("sigma_downwind", puff.get("sigma", 0.08)))
            sigma_crosswind = float(puff.get("sigma_crosswind", puff.get("sigma", 0.04)))
            total += puff["mass"] * math.exp(
                -(
                    downwind * downwind / (2.0 * sigma_downwind * sigma_downwind)
                    + crosswind * crosswind / (2.0 * sigma_crosswind * sigma_crosswind)
                )
            ) / (2.0 * math.pi * sigma_downwind * sigma_crosswind)
        return total

    def concentration(self, x: float, y: float, add_noise: bool = True) -> float:
        """返回指定坐标处的气味浓度。"""
        concentration = float(self._source_core_concentration(x, y))
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
        concentration = self._source_core_concentration(grid_x, grid_y).astype(
            np.float32
        )
        concentration += self.gas_background
        for puff in self.puffs:
            dx = grid_x - puff["x"]
            dy = grid_y - puff["y"]
            direction = float(puff.get("direction", self.wind_direction))
            downwind = dx * math.cos(direction) + dy * math.sin(direction)
            crosswind = -dx * math.sin(direction) + dy * math.cos(direction)
            sigma_downwind = float(puff.get("sigma_downwind", puff.get("sigma", 0.08)))
            sigma_crosswind = float(puff.get("sigma_crosswind", puff.get("sigma", 0.04)))
            concentration += (
                puff["mass"]
                * np.exp(
                    -(
                        downwind**2 / (2.0 * sigma_downwind * sigma_downwind)
                        + crosswind**2 / (2.0 * sigma_crosswind * sigma_crosswind)
                    )
                )
                / (2.0 * np.pi * sigma_downwind * sigma_crosswind)
            ).astype(np.float32)
        if add_noise:
            concentration += self.rng.normal(
                0.0, self.gas_noise_std, size=concentration.shape
            ).astype(np.float32)
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
