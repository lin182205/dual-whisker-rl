"""触须独立控制环境：固定机器人，仅训练左右触须主动采样。"""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dual_whisker_rl.envs.robot_model import RobotState
from dual_whisker_rl.envs.sensor_model import FirstOrderGasSensor
from dual_whisker_rl.envs.whisker_model import DualWhiskerSampler


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
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.source_position_override = source_position
        if source_position is None:
            self.source_x = WORLD_MIN - 0.7
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
        self.wind_dir_noise_std = 0.01
        self.wind_relaxation = 0.97
        self.wind_direction_limit = math.radians(6.0)
        self.wind_speed_clip = (
            max(0.005, self.wind_speed_range[0] - 0.03),
            max(self.wind_speed_range[1] + 0.04, 0.12),
        )

        self.source_core_sigma = 0.12
        self.source_core_weight = 0.0
        self.puff_release_per_step = 1
        self.puff_release_downwind_jitter = 0.02
        self.puff_release_crosswind_jitter = 0.12
        self.puff_init_mass = 0.18
        self.puff_mass_jitter = 0.45
        self.puff_init_sigma_downwind = 0.10
        self.puff_init_sigma_crosswind = 0.035
        self.puff_sigma_jitter = 0.25
        self.diffusion_downwind_rate = 0.020
        self.diffusion_crosswind_rate = 0.006
        self.decay_rate = 0.018
        self.turbulence_downwind_std = 0.006
        self.turbulence_crosswind_std = 0.018
        self.max_puff_age = 16.0
        self.min_puff_mass = 0.004
        self.max_puffs = 260
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

    def metadata(self) -> dict[str, Any]:
        """返回绘图和调试所需的气味场元信息。"""
        return {
            "world_min": WORLD_MIN,
            "world_max": WORLD_MAX,
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
        if x <= WORLD_MIN or x >= WORLD_MAX or y <= WORLD_MIN or y >= WORLD_MAX:
            return True
        for xmin, xmax, ymin, ymax in self.obstacles:
            if xmin <= x <= xmax and ymin <= y <= ymax:
                return True
        return False

    def is_position_valid(self, x: float, y: float, margin: float = 0.0) -> bool:
        if not (
            WORLD_MIN + margin < x < WORLD_MAX - margin
            and WORLD_MIN + margin < y < WORLD_MAX - margin
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
        """每个环境步小幅扰动风场，模拟真实气流的不稳定性。"""
        if self.wind_sampling_mode == "fixed":
            self.wind_direction = self.wind_direction_base
            self.wind_speed = self.wind_speed_mean
            return
        direction_candidate = self._normalize_angle(
            self.wind_direction + float(self.rng.normal(0.0, self.wind_dir_noise_std))
        )
        direction_offset = self._normalize_angle(
            direction_candidate - self.wind_direction_base
        )
        direction_offset = float(
            np.clip(
                direction_offset,
                -self.wind_direction_limit,
                self.wind_direction_limit,
            )
        )
        self.wind_direction = self._normalize_angle(
            self.wind_direction_base + direction_offset
        )
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
                }
            )
        if len(self.puffs) > self.max_puffs:
            self.puffs = self.puffs[-self.max_puffs :]

    def _advance_puffs(self) -> None:
        """推进所有 puff：随风移动、湍流扰动、扩散并逐渐衰减。"""
        wind_x = self.wind_speed * math.cos(self.wind_direction)
        wind_y = self.wind_speed * math.sin(self.wind_direction)
        active_puffs = []
        for puff in self.puffs:
            direction = float(puff.get("direction", self.wind_direction))
            downwind_x = math.cos(direction)
            downwind_y = math.sin(direction)
            crosswind_x = -downwind_y
            crosswind_y = downwind_x
            downwind_turbulence = float(
                self.rng.normal(0.0, self.turbulence_downwind_std)
            )
            crosswind_turbulence = float(
                self.rng.normal(0.0, self.turbulence_crosswind_std)
            )
            puff["age"] += self.dt
            puff["x"] += (
                wind_x * self.dt
                + downwind_turbulence * downwind_x
                + crosswind_turbulence * crosswind_x
            )
            puff["y"] += (
                wind_y * self.dt
                + downwind_turbulence * downwind_y
                + crosswind_turbulence * crosswind_y
            )
            puff["sigma_downwind"] += self.diffusion_downwind_rate * self.dt
            puff["sigma_crosswind"] += self.diffusion_crosswind_rate * self.dt
            puff["mass"] *= math.exp(-self.decay_rate * self.dt)
            if puff["age"] > self.max_puff_age or puff["mass"] < self.min_puff_mass:
                continue
            if (
                puff["x"] < WORLD_MIN - self.puff_bounds_margin
                or puff["x"] > WORLD_MAX + self.puff_bounds_margin
                or puff["y"] < WORLD_MIN - self.puff_bounds_margin
                or puff["y"] > WORLD_MAX + self.puff_bounds_margin
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
        xs = np.linspace(WORLD_MIN, WORLD_MAX, resolution, dtype=np.float32)
        ys = np.linspace(WORLD_MIN, WORLD_MAX, resolution, dtype=np.float32)
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
        x = self.rng.uniform(WORLD_MIN - self.far_source_distance, WORLD_MIN - 0.25)
        y = self.rng.uniform(WORLD_MIN, WORLD_MAX)
        return float(x), float(y)

    def _warmup_puffs(self) -> None:
        """预先推进 puff，使 reset 后的气味场不是空场。"""
        self.puffs = []
        for _ in range(self.puff_warmup_steps):
            self.advance()

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
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
        self.hit_threshold = float(cfg.get("hit_threshold", 0.08))
        self.odor_hit_reward = float(cfg.get("odor_hit_reward", 0.10))
        self.contrast_reward_scale = float(cfg.get("contrast_reward_scale", 0.04))
        self.trend_reward_scale = float(cfg.get("trend_reward_scale", 4.0))
        self.time_penalty = float(cfg.get("time_penalty", 0.04))

        # 与 `PlumeEnv` 保持一致：环境持有一个 plume 对象负责气味场。
        source_position = cfg.get("source_position")
        self.plume = DynamicPuffPlume(
            seed=seed,
            source_position=tuple(source_position) if source_position is not None else None,
            dt=float(cfg.get("dt", 0.2)),
            gas_field_mode=str(cfg.get("gas_field_mode", "puff")),
            wind_speed_range=tuple(cfg.get("wind_speed_range", (0.06, 0.11))),
            wind_sampling_mode=str(cfg.get("wind_sampling_mode", "random")),
        )
        self.field = self.plume  # 兼容旧可视化脚本中的 env.field 访问。
        self.whiskers = DualWhiskerSampler(
            length=float(cfg.get("whisker_length", 0.15)),
            sector_count=int(cfg.get("whisker_sector_count", 10)),
            servo_60deg_time_s=float(cfg.get("servo_60deg_time_s", 0.12)),
        )
        sensor_alpha = float(cfg.get("sensor_alpha", 0.95))
        self.left_sensor = FirstOrderGasSensor(sensor_alpha)
        self.right_sensor = FirstOrderGasSensor(sensor_alpha)

        self.robot_state = RobotState(0.0, 0.0, 0.0)
        self.action_space = spaces.MultiDiscrete(
            [self.whiskers.sector_count, self.whiskers.sector_count]
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(12,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.prev_left = 0.0
        self.prev_right = 0.0
        self.left_hits: list[float] = []
        self.right_hits: list[float] = []
        self.trajectory: list[dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
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

        self.robot_state = self.sample_robot_pose_in_plume()
        return self._observe()

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        left_sector, right_sector = self._decode_action(action)
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
        obs, info = self._observe()
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

    def sample_robot_pose_in_plume(self) -> RobotState:
        """优先把机器人/触须初始化到当前羽流覆盖区域附近。

        训练触须主动采样时，如果初始位置总在细窄羽流外，策略会看到大量
        接近零的传感器读数。这里先用当前 puff 场生成浓度网格，再从高于
        分位数阈值的覆盖区域中随机选点，使触须更容易获得有效气味信息。
        """
        xs, ys, concentration = self.plume.grid(resolution=60, add_noise=False)
        threshold = max(
            self.hit_threshold * 0.35,
            float(np.percentile(concentration, 70.0)),
        )
        candidate_rows, candidate_cols = np.where(concentration >= threshold)
        if candidate_rows.size == 0:
            return self.sample_robot_pose_uniform()

        for _ in range(128):
            idx = int(self.rng.integers(0, candidate_rows.size))
            x = float(xs[candidate_cols[idx]])
            y = float(ys[candidate_rows[idx]])
            if not self.plume.is_position_valid(x, y, margin=0.16):
                continue
            if math.hypot(x - self.plume.source_x, y - self.plume.source_y) < 0.10:
                continue
            theta = self.rng.uniform(-np.pi, np.pi)
            return RobotState(x, y, float(theta))
        return self.sample_robot_pose_uniform()

    def sample_robot_pose_uniform(self) -> RobotState:
        """兜底采样：如果羽流覆盖区候选失败，则在 1m 场地内均匀采样。"""
        while True:
            x = self.rng.uniform(WORLD_MIN + 0.16, WORLD_MAX - 0.16)
            y = self.rng.uniform(WORLD_MIN + 0.16, WORLD_MAX - 0.16)
            if math.hypot(x - self.plume.source_x, y - self.plume.source_y) < 0.10:
                continue
            theta = self.rng.uniform(-np.pi, np.pi)
            return RobotState(float(x), float(y), float(theta))

    def _observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        whisker_state = self.whiskers.state(self.robot_state)
        left = self.left_sensor.value
        right = self.right_sensor.value
        wind_rel = self._wrap_angle(self.plume.wind_direction - self.robot_state.heading)
        source_distance = math.hypot(
            self.robot_state.x - self.plume.source_x,
            self.robot_state.y - self.plume.source_y,
        )

        obs = np.array(
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
                source_distance / (WORLD_MAX - WORLD_MIN),
            ],
            dtype=np.float32,
        )
        return obs, {
            "robot_state": self.robot_state,
            "whisker_state": whisker_state,
            "step": self.step_count,
        }

    def get_reward(self, left: float, right: float) -> float:
        gas_concentration = max(left, right)
        gas_trend = max(left - self.prev_left, right - self.prev_right)
        in_plume = gas_concentration >= self.hit_threshold
        strong_plume = gas_concentration >= self.plume.plume_strong_threshold

        gas_trend_reward = self.trend_reward_scale * float(
            np.clip(gas_trend, -0.04, 0.04)
        )
        gas_presence_reward = self.odor_hit_reward * float(
            np.clip(gas_concentration, 0.0, 1.0)
        )
        plume_tracking_bonus = 0.12 if in_plume else 0.0
        strong_plume_bonus = 0.10 if strong_plume else 0.0
        contrast_reward = self.contrast_reward_scale * abs(left - right)
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
