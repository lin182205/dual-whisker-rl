"""移动机器人 + 双触须气源搜索环境。

在 `WhiskerOnlyPuffEnv`（固定基座、稀疏动态羽流、硬件可部署观测、舵机限速触须）
之上，把触须装到一个差速驱动的移动机器人上，做气源搜索。默认每个 episode
联合随机化气源方位、风向、下风侧机器人起点和朝向，同时保留旧固定场景用于诊断。

动作空间 `MultiDiscrete([6, 10, 10])` = [移动动作, 左扇区, 右扇区]，与硬件串口协议
兼容（移动交给底盘、扇区对应 STEP 命令）。观测在父类 12 维硬件特征上追加机器人自身
可得的本体感受（朝向 cos/sin、上一步移动动作），不含真实风向/气源方向等特权信息；
奖励可用真实气源距离（仅训练期）。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from gymnasium import spaces

from dual_whisker_rl.envs.robot_model import DifferentialDriveRobot
from dual_whisker_rl.envs.robot_model import RobotState
from dual_whisker_rl.envs.whisker_only_env import WhiskerOnlyPuffEnv


class MobileWhiskerPuffEnv(WhiskerOnlyPuffEnv):
    """移动机器人 + 双触须气源搜索环境。"""

    REWARD_COMPONENT_NAMES = (
        "distance_potential_progress",
        "best_concentration_improvement",
        "time_penalty",
        "stagnation_penalty",
        "goal_bonus",
        "out_of_bounds_penalty",
    )

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        # 注入 mobile 默认：更大场地（2m，±1.0）以显现追踪。
        cfg.setdefault("world_half", 1.0)
        wh = float(cfg["world_half"])
        explicit_source = cfg.get("source_position") is not None
        scenario_mode_explicit = "scenario_mode" in cfg
        scenario_mode = str(
            cfg.get("scenario_mode", "fixed" if explicit_source else "randomized")
        )
        if scenario_mode not in ("randomized", "fixed"):
            raise ValueError(
                "scenario_mode must be 'randomized' or 'fixed', "
                f"got {scenario_mode!r}"
            )
        if scenario_mode == "randomized" and explicit_source:
            detail = "explicitly " if scenario_mode_explicit else ""
            raise ValueError(
                f"{detail}randomized scenario_mode cannot be combined with "
                "source_position; remove source_position or use scenario_mode='fixed'"
            )
        if scenario_mode == "fixed" and not explicit_source:
            cfg["source_position"] = (-(wh - 0.2), 0.0)
        cfg.setdefault("max_steps", 400)
        # 大场地需要羽流能横跨过去，否则下风侧起点闻不到气味：加大风速、延长 puff 寿命。
        cfg.setdefault("wind_speed_range", (0.12, 0.20))
        overrides = dict(cfg.get("plume_overrides") or {})
        overrides.setdefault("max_puff_age", 16.0)
        cfg["plume_overrides"] = overrides
        # 本环境的观测拼接假定父类 12 维硬件观测，禁止 privileged。
        if str(cfg.get("observation_mode", "hardware")) != "hardware":
            raise ValueError("MobileWhiskerPuffEnv 只支持 observation_mode='hardware'")
        cfg["observation_mode"] = "hardware"
        super().__init__(cfg)

        self.scenario_mode = scenario_mode
        self.scenario_source_distance_range = self._validate_range(
            "scenario_source_distance_range",
            cfg.get("scenario_source_distance_range", (0.65 * wh, 0.80 * wh)),
            positive=True,
        )
        self.scenario_source_crosswind_range = self._validate_range(
            "scenario_source_crosswind_range",
            cfg.get("scenario_source_crosswind_range", (-0.15 * wh, 0.15 * wh)),
        )
        self.scenario_robot_downwind_range = self._validate_range(
            "scenario_robot_downwind_range",
            cfg.get("scenario_robot_downwind_range", (0.75 * wh, 1.45 * wh)),
            positive=True,
        )
        self.scenario_robot_crosswind_range = self._validate_range(
            "scenario_robot_crosswind_range",
            cfg.get("scenario_robot_crosswind_range", (-0.35 * wh, 0.35 * wh)),
        )
        self.scenario_uniform_heading_probability = float(
            cfg.get("scenario_uniform_heading_probability", 0.75)
        )
        if not math.isfinite(self.scenario_uniform_heading_probability) or not (
            0.0 <= self.scenario_uniform_heading_probability <= 1.0
        ):
            raise ValueError(
                "scenario_uniform_heading_probability must be between 0 and 1"
            )
        self.scenario_source_facing_jitter = float(
            cfg.get("scenario_source_facing_jitter", 0.4)
        )
        if (
            not math.isfinite(self.scenario_source_facing_jitter)
            or self.scenario_source_facing_jitter < 0.0
        ):
            raise ValueError("scenario_source_facing_jitter must be non-negative")
        self.initial_heading_mode = "source_facing"
        self._scenario_rng = np.random.default_rng(
            np.random.SeedSequence([int(cfg.get("seed", 0)), 0x5343454E])
        )

        if self.scenario_mode == "randomized":
            self.plume.wind_sampling_mode = "random"
            self.plume.wind_direction_range = (-math.pi, math.pi)
            self.plume.source_distance_range = self.scenario_source_distance_range
            self.plume.far_source_crosswind_range = (
                self.scenario_source_crosswind_range
            )
            self.plume.require_source_in_world = True

        # 差速驱动底盘。1m 场 @ dt=0.2：forward=0.03m/步（穿场约 33 步）、
        # turn≈0.30rad/步≈17°/步，机器人比触须舵机慢，贴近现实。
        self.robot = DifferentialDriveRobot(
            forward_speed=float(cfg.get("forward_speed", 0.15)),
            turn_rate=float(cfg.get("turn_rate", 1.5)),
            dt=float(self.plume.dt),
        )
        self.n_move_actions = len(DifferentialDriveRobot.ACTIONS)
        self.stop_action = DifferentialDriveRobot.ACTIONS.index("stop")

        # 动作：[移动, 左扇区, 右扇区]。
        sc = int(self.whiskers.sector_count)
        self.action_space = spaces.MultiDiscrete([self.n_move_actions, sc, sc])

        obsolete_reward_keys = (
            "odor_reach_scale",
            "odor_reach_clip",
            "mobile_odor_hit_reward",
            "mobile_contrast_scale",
        )
        configured_obsolete_keys = [
            key for key in obsolete_reward_keys if key in cfg
        ]
        if configured_obsolete_keys:
            raise ValueError(
                "obsolete mobile reward config keys are not supported: "
                f"{', '.join(configured_obsolete_keys)}; use "
                "best_concentration_reward_scale/best_concentration_clip instead"
            )

        # 防刷分奖励：浓度只奖励 episode 历史最佳值增量；距离项为有符号势差。
        # 默认系数保证朝气源正常前进一步的距离奖励能覆盖单步时间成本，避免
        # “正确移动也持续亏分”；越界惩罚则在下方按整局最坏成本做安全校验。
        self.goal_radius = float(cfg.get("goal_radius", 0.10))
        self.progress_reward_scale = float(cfg.get("progress_reward_scale", 3.0))
        configured_goal_bonus = float(cfg.get("goal_bonus", 50.0))
        if not math.isclose(configured_goal_bonus, 50.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("goal_bonus is fixed at 50.0 for mobile source search")
        self.goal_bonus = 50.0
        self.mobile_time_penalty = float(cfg.get("mobile_time_penalty", 0.03))
        self.oob_penalty = float(cfg.get("oob_penalty", 25.0))
        self.best_concentration_reward_scale = float(
            cfg.get("best_concentration_reward_scale", 3.0)
        )
        self.best_concentration_clip = float(
            cfg.get("best_concentration_clip", 1.0)
        )
        self.concentration_progress_epsilon = float(
            cfg.get("concentration_progress_epsilon", 0.01)
        )
        self.position_progress_epsilon = float(
            cfg.get("position_progress_epsilon", 1e-4)
        )
        self.distance_progress_epsilon = float(
            cfg.get("distance_progress_epsilon", 1e-4)
        )
        stagnation_window_value = float(cfg.get("stagnation_window", 20))
        if not stagnation_window_value.is_integer():
            raise ValueError("stagnation_window must be an integer")
        self.stagnation_window = int(stagnation_window_value)
        self.stagnation_penalty = float(cfg.get("stagnation_penalty", 0.02))

        nonnegative_reward_values = {
            "progress_reward_scale": self.progress_reward_scale,
            "oob_penalty": self.oob_penalty,
            "best_concentration_reward_scale": self.best_concentration_reward_scale,
            "concentration_progress_epsilon": self.concentration_progress_epsilon,
            "position_progress_epsilon": self.position_progress_epsilon,
            "distance_progress_epsilon": self.distance_progress_epsilon,
            "stagnation_penalty": self.stagnation_penalty,
            "mobile_time_penalty": self.mobile_time_penalty,
        }
        for name, value in nonnegative_reward_values.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not math.isfinite(self.best_concentration_clip)
            or self.best_concentration_clip <= 0.0
        ):
            raise ValueError("best_concentration_clip must be finite and positive")
        if self.stagnation_window < 1:
            raise ValueError("stagnation_window must be at least 1")

        world_diameter = 2.0 * math.sqrt(2.0) * self.world_half
        self.positive_auxiliary_reward_upper_bound = (
            self.best_concentration_reward_scale * self.best_concentration_clip
            + self.progress_reward_scale * world_diameter
        )
        if (
            self.positive_auxiliary_reward_upper_bound
            > self.goal_bonus / 4.0 + 1e-12
        ):
            raise ValueError(
                "positive auxiliary reward upper bound must not exceed "
                f"goal_bonus / 4 = {self.goal_bonus / 4.0:.6f}; got "
                f"{self.positive_auxiliary_reward_upper_bound:.6f}"
            )
        self.max_episode_time_cost = self.mobile_time_penalty * self.max_steps
        self.minimum_oob_penalty = (
            self.max_episode_time_cost
            + self.positive_auxiliary_reward_upper_bound
        )
        if self.oob_penalty + 1e-12 < self.minimum_oob_penalty:
            raise ValueError(
                "oob_penalty must cover the maximum episode time cost plus all "
                "positive auxiliary rewards, otherwise early out-of-bounds can be "
                "more profitable than continuing: "
                f"need >= {self.minimum_oob_penalty:.6f}, got {self.oob_penalty:.6f}"
            )

        # 仅用于评估主动采样比例，不参与奖励计算。
        self.whisker_motion_epsilon_rad = float(
            cfg.get("whisker_motion_epsilon_rad", math.radians(0.5))
        )
        if self.whisker_motion_epsilon_rad < 0.0:
            raise ValueError("whisker_motion_epsilon_rad must be non-negative")

        # 供 evaluation.py / 旧可视化按需读取的场地尺寸别名。
        self.width = float(self.world_max - self.world_min)
        self.height = float(self.world_max - self.world_min)

        # 观测 = 父 12 维硬件 + 3 维本体感受（朝向 cos/sin、上一步移动动作）。
        base_low = self.observation_builder.low
        base_high = self.observation_builder.high
        low = np.concatenate([base_low, np.array([-1.0, -1.0, 0.0], dtype=np.float32)])
        high = np.concatenate([base_high, np.array([1.0, 1.0, 1.0], dtype=np.float32)])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.observation_field_names = self.observation_builder.field_names + [
            "heading_cos",
            "heading_sin",
            "last_move_norm",
        ]

        self.last_move = self.stop_action
        self.prev_distance = 0.0
        self.best_concentration = 0.0
        self.stagnation_steps = 0

    # ---- 位姿与观测 ----

    @staticmethod
    def _validate_range(
        name: str,
        values: Any,
        *,
        positive: bool = False,
    ) -> tuple[float, float]:
        try:
            low, high = (float(value) for value in values)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must contain exactly two numbers") from None
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError(f"{name} values must be finite")
        if low > high:
            raise ValueError(f"{name} lower bound must not exceed upper bound")
        if positive and low <= 0.0:
            raise ValueError(f"{name} values must be positive")
        return low, high

    def sample_initial_robot_pose(self) -> RobotState:
        """按场景模式采样机器人起点，并同步到底盘模型。"""
        if self.scenario_mode == "fixed":
            return self._sample_fixed_initial_robot_pose()
        return self._sample_randomized_initial_robot_pose()

    def _sample_fixed_initial_robot_pose(self) -> RobotState:
        """复现旧场景：+x 半场随机起点，大致朝向固定气源。"""
        wh = self.world_max
        for _ in range(128):
            x = float(self.rng.uniform(0.35 * wh, wh - 0.15))
            y = float(self.rng.uniform(-(wh - 0.15), wh - 0.15))
            if not self.plume.is_position_valid(x, y, margin=0.16):
                continue
            if math.hypot(x - self.plume.source_x, y - self.plume.source_y) < 0.15:
                continue
            heading = math.pi + float(self.rng.uniform(-0.4, 0.4))
            self.initial_heading_mode = "source_facing"
            self.robot.reset(x, y, heading)
            return self.robot.state
        # 兜底：场地中偏下风处。
        self.initial_heading_mode = "source_facing"
        self.robot.reset(0.5 * wh, 0.0, math.pi)
        return self.robot.state

    def _sample_randomized_initial_robot_pose(self) -> RobotState:
        """在当前平均风的下风区域采样有效起点和混合分布朝向。"""
        rng = self._scenario_rng
        wind_x = math.cos(self.plume.wind_direction_base)
        wind_y = math.sin(self.plume.wind_direction_base)
        cross_x = -wind_y
        cross_y = wind_x

        for _ in range(256):
            downwind = float(rng.uniform(*self.scenario_robot_downwind_range))
            crosswind = float(rng.uniform(*self.scenario_robot_crosswind_range))
            x = self.plume.source_x + downwind * wind_x + crosswind * cross_x
            y = self.plume.source_y + downwind * wind_y + crosswind * cross_y
            if self._is_valid_randomized_start(x, y):
                return self._reset_robot_with_sampled_heading(x, y)

        downwind_values = np.linspace(*self.scenario_robot_downwind_range, num=33)
        crosswind_values = sorted(
            np.linspace(*self.scenario_robot_crosswind_range, num=33),
            key=abs,
        )
        for downwind in downwind_values:
            for crosswind in crosswind_values:
                x = self.plume.source_x + float(downwind) * wind_x + float(crosswind) * cross_x
                y = self.plume.source_y + float(downwind) * wind_y + float(crosswind) * cross_y
                if self._is_valid_randomized_start(x, y):
                    return self._reset_robot_with_sampled_heading(x, y)

        raise RuntimeError(
            "could not place robot in a valid downwind position for randomized scenario"
        )

    def _is_valid_randomized_start(self, x: float, y: float) -> bool:
        if not self.plume.is_position_valid(x, y, margin=0.16):
            return False
        return math.hypot(x - self.plume.source_x, y - self.plume.source_y) > max(
            0.15,
            self.goal_radius,
        )

    def _reset_robot_with_sampled_heading(self, x: float, y: float) -> RobotState:
        rng = self._scenario_rng
        if rng.random() < self.scenario_uniform_heading_probability:
            heading = float(rng.uniform(-math.pi, math.pi))
            self.initial_heading_mode = "uniform"
        else:
            source_bearing = math.atan2(
                self.plume.source_y - y,
                self.plume.source_x - x,
            )
            heading = source_bearing + float(
                rng.uniform(
                    -self.scenario_source_facing_jitter,
                    self.scenario_source_facing_jitter,
                )
            )
            self.initial_heading_mode = "source_facing"
        self.robot.reset(x, y, heading)
        return self.robot.state

    def scenario_metadata(self) -> dict[str, Any]:
        """返回可序列化的场景初始化分布。"""
        if self.scenario_mode == "fixed":
            return {
                "mode": "fixed",
                "source_position": [
                    float(self.plume.source_x),
                    float(self.plume.source_y),
                ],
                "wind_direction_range_deg": [
                    math.degrees(self.plume.wind_direction_range[0]),
                    math.degrees(self.plume.wind_direction_range[1]),
                ],
                "robot_start_x_range": [0.35 * self.world_max, self.world_max - 0.15],
                "robot_start_y_range": [-(self.world_max - 0.15), self.world_max - 0.15],
                "uniform_heading_probability": 0.0,
                "source_facing_jitter_rad": 0.4,
            }
        return {
            "mode": "randomized",
            "source_position": None,
            "wind_direction_range_deg": [-180.0, 180.0],
            "source_distance_range": list(self.scenario_source_distance_range),
            "source_crosswind_range": list(self.scenario_source_crosswind_range),
            "robot_downwind_range": list(self.scenario_robot_downwind_range),
            "robot_crosswind_range": list(self.scenario_robot_crosswind_range),
            "uniform_heading_probability": self.scenario_uniform_heading_probability,
            "source_facing_jitter_rad": self.scenario_source_facing_jitter,
        }

    def _build_observation(self) -> np.ndarray:
        """父 12 维硬件观测（预处理只推进一次）+ 3 维本体感受 → 15 维。"""
        base = super()._build_observation()
        heading = self.robot_state.heading
        proprio = np.array(
            [
                math.cos(heading),
                math.sin(heading),
                float(self.last_move) / max(1, self.n_move_actions - 1),
            ],
            dtype=np.float32,
        )
        obs = np.concatenate([base, proprio]).astype(np.float32)
        self._last_obs = obs
        return obs

    def _distance_to_source(self) -> float:
        return float(
            math.hypot(
                self.robot_state.x - self.plume.source_x,
                self.robot_state.y - self.plume.source_y,
            )
        )

    # ---- 生命周期 ----

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.last_move = self.stop_action
        self.best_concentration = 0.0
        self.stagnation_steps = 0
        if self.scenario_mode == "randomized":
            if seed is not None:
                self._scenario_rng = np.random.default_rng(
                    np.random.SeedSequence([int(seed), 0x5343454E])
                )
            else:
                self._scenario_rng = np.random.default_rng(
                    int(self.rng.integers(0, 2**63 - 1))
                )
        obs, info = super().reset(seed=seed, options=options)
        self.prev_distance = self._distance_to_source()
        info.update(
            {
                "distance_to_source": self.prev_distance,
                "scenario_mode": self.scenario_mode,
                "initial_heading_mode": self.initial_heading_mode,
                "wind_direction": self.plume.wind_direction,
                "source": (self.plume.source_x, self.plume.source_y),
            }
        )
        return obs, info

    def _decode_action(self, action: Any) -> tuple[int, int, int]:
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.size != 3:
            raise ValueError(
                "MobileWhiskerPuffEnv action must be [move, left_sector, right_sector]"
            )
        return int(arr[0]), int(arr[1]), int(arr[2])

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        move, left_sector, right_sector = self._decode_action(action)
        self.last_move = move
        self.last_left_sector = left_sector
        self.last_right_sector = right_sector

        # 先推进机器人位姿，再采样触须（触须点由 whisker_model 依位姿自动计算）。
        previous_x = self.robot_state.x
        previous_y = self.robot_state.y
        self.robot_state = self.robot.step(move)
        position_displacement = float(
            math.hypot(
                self.robot_state.x - previous_x,
                self.robot_state.y - previous_y,
            )
        )
        self.plume.advance()
        previous_left_angle = self.whiskers.left_angle
        previous_right_angle = self.whiskers.right_angle
        whisker_state = self.whiskers.step(
            left_sector,
            right_sector,
            self.robot_state,
            dt=self.plume.dt,
        )
        left_angle_delta = abs(whisker_state.left_angle - previous_left_angle)
        right_angle_delta = abs(whisker_state.right_angle - previous_right_angle)
        whisker_moved = bool(
            max(left_angle_delta, right_angle_delta)
            > self.whisker_motion_epsilon_rad
        )

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

        distance = self._distance_to_source()
        reached = distance <= self.goal_radius
        oob = not (self.world_min < self.robot_state.x < self.world_max and self.world_min < self.robot_state.y < self.world_max)
        reward_components = self._source_search_reward_components(
            distance,
            reached,
            oob,
            left,
            right,
            position_displacement,
        )
        reward = float(sum(reward_components.values()))

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
                "left_angle_delta": left_angle_delta,
                "right_angle_delta": right_angle_delta,
                "whisker_moved": whisker_moved,
                "left_sector": int(left_sector),
                "right_sector": int(right_sector),
                "move_action": DifferentialDriveRobot.ACTIONS[move],
                "distance_to_source": distance,
                "best_concentration": self.best_concentration,
                "position_displacement": position_displacement,
                "stagnation_steps": self.stagnation_steps,
                "reward": reward,
                "reward_components": reward_components,
            }
        )

        self.prev_distance = distance
        self.prev_left = left
        self.prev_right = right

        obs = self._build_observation()
        info = self._current_info()
        info.update(
            {
                "left_sector": int(left_sector),
                "right_sector": int(right_sector),
                "move_action": DifferentialDriveRobot.ACTIONS[move],
                "left": left,
                "right": right,
                "left_angle_delta": left_angle_delta,
                "right_angle_delta": right_angle_delta,
                "whisker_moved": whisker_moved,
                "raw_left": raw_left,
                "raw_right": raw_right,
                "wind_direction": self.plume.wind_direction,
                "wind_speed": self.plume.wind_speed,
                "source": (self.plume.source_x, self.plume.source_y),
                "scenario_mode": self.scenario_mode,
                "initial_heading_mode": self.initial_heading_mode,
                "distance_to_source": distance,
                "best_concentration": self.best_concentration,
                "position_displacement": position_displacement,
                "stagnation_steps": self.stagnation_steps,
                "out_of_bounds": oob,
                "is_success": reached,
                "reward_components": reward_components,
            }
        )
        terminated = bool(reached or oob)
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, info

    def _source_search_reward(
        self,
        distance: float,
        reached: bool,
        oob: bool,
        left: float,
        right: float,
        position_displacement: float = 0.0,
    ) -> float:
        return float(
            sum(
                self._source_search_reward_components(
                    distance,
                    reached,
                    oob,
                    left,
                    right,
                    position_displacement,
                ).values()
            )
        )

    def _source_search_reward_components(
        self,
        distance: float,
        reached: bool,
        oob: bool,
        left: float,
        right: float,
        position_displacement: float = 0.0,
    ) -> dict[str, float]:
        """计算可望远镜求和、不能靠原地保持高浓度重复刷取的奖励。"""
        clipped_concentration = float(
            np.clip(max(left, right), 0.0, self.best_concentration_clip)
        )
        concentration_improvement = max(
            0.0,
            clipped_concentration - self.best_concentration,
        )
        self.best_concentration = max(
            self.best_concentration,
            clipped_concentration,
        )

        distance_improvement = self.prev_distance - distance
        position_progressed = (
            position_displacement > self.position_progress_epsilon
        )
        concentration_progressed = (
            concentration_improvement > self.concentration_progress_epsilon
        )
        distance_progressed = (
            distance_improvement > self.distance_progress_epsilon
        )
        terminal = reached or oob
        if (
            terminal
            or position_progressed
            or concentration_progressed
            or distance_progressed
        ):
            self.stagnation_steps = 0
        else:
            self.stagnation_steps += 1
        stagnation_active = (
            not terminal and self.stagnation_steps >= self.stagnation_window
        )

        return {
            "distance_potential_progress": float(
                self.progress_reward_scale * distance_improvement
            ),
            "best_concentration_improvement": float(
                self.best_concentration_reward_scale * concentration_improvement
            ),
            "time_penalty": float(-self.mobile_time_penalty),
            "stagnation_penalty": float(
                -self.stagnation_penalty if stagnation_active else 0.0
            ),
            "goal_bonus": float(self.goal_bonus if reached else 0.0),
            "out_of_bounds_penalty": float(-self.oob_penalty if oob else 0.0),
        }
