"""移动机器人 + 双触须气源搜索环境。

在 `WhiskerOnlyPuffEnv`（固定基座、稀疏动态羽流、硬件可部署观测、舵机限速触须）
之上，把触须装到一个差速驱动的移动机器人上，做气源搜索：气源置于 1m 场地上风侧
（场内），机器人从下风侧出发，联合控制"移动 + 左右触须扇区"导航到气源。

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

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        cfg = dict(config or {})
        # 注入 mobile 默认：更大场地（2m，±1.0）以显现追踪；气源置于上风侧场内。
        cfg.setdefault("world_half", 1.0)
        wh = float(cfg["world_half"])
        cfg.setdefault("source_position", (-(wh - 0.2), 0.0))
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

        # 源搜索奖励系数（按 1m 几何 + 传感器 ~0-1.5 量级标定，config 可覆盖）。
        self.goal_radius = float(cfg.get("goal_radius", 0.10))
        # 源搜索奖励必须以“接近并到达气源”为主目标。气味相关项只提供弱塑形；如果按
        # 每步绝对浓度给予较大奖励，策略会在羽流中原地旋转直到超时，而不是继续寻源。
        self.progress_reward_scale = float(cfg.get("progress_reward_scale", 10.0))
        self.goal_bonus = float(cfg.get("goal_bonus", 30.0))
        self.oob_penalty = float(cfg.get("oob_penalty", 20.0))
        # odor_reach 仍让触须指向影响回报，但其累计量级低于 progress + goal。
        self.odor_reach_scale = float(cfg.get("odor_reach_scale", 0.02))
        self.odor_reach_clip = float(cfg.get("odor_reach_clip", 1.0))
        self.mobile_odor_hit_reward = float(cfg.get("mobile_odor_hit_reward", 0.001))
        self.mobile_trend_scale = float(cfg.get("mobile_trend_scale", 0.1))
        self.mobile_trend_clip = float(cfg.get("mobile_trend_clip", 0.1))
        # 恢复强左右差异信号，但默认只在触须实际运动时发放，避免固定在高差异姿态
        # 也能逐步累积分数。absolute 用于复现实验，delta 用于差异增量消融。
        self.mobile_contrast_scale = float(cfg.get("mobile_contrast_scale", 0.6))
        self.mobile_contrast_mode = str(
            cfg.get("mobile_contrast_mode", "motion_gated")
        )
        if self.mobile_contrast_mode not in {"motion_gated", "absolute", "delta"}:
            raise ValueError(
                "mobile_contrast_mode must be one of: motion_gated, absolute, delta"
            )
        self.mobile_contrast_delta_clip = float(
            cfg.get("mobile_contrast_delta_clip", 0.1)
        )
        if self.mobile_contrast_delta_clip <= 0.0:
            raise ValueError("mobile_contrast_delta_clip must be positive")
        self.whisker_motion_epsilon_rad = float(
            cfg.get("whisker_motion_epsilon_rad", math.radians(0.5))
        )
        if self.whisker_motion_epsilon_rad < 0.0:
            raise ValueError("whisker_motion_epsilon_rad must be non-negative")
        self.stationary_sampling_reward_factor = float(
            cfg.get("stationary_sampling_reward_factor", 0.25)
        )
        if not 0.0 <= self.stationary_sampling_reward_factor <= 1.0:
            raise ValueError(
                "stationary_sampling_reward_factor must satisfy 0 <= value <= 1"
            )
        self.mobile_time_penalty = float(cfg.get("mobile_time_penalty", 0.01))

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

    # ---- 位姿与观测 ----

    def sample_initial_robot_pose(self) -> RobotState:
        """在下风侧（+x 半场）随机起点、大致朝上风（朝向气源）。同步底盘位姿。"""
        wh = self.world_max
        for _ in range(128):
            x = float(self.rng.uniform(0.35 * wh, wh - 0.15))
            y = float(self.rng.uniform(-(wh - 0.15), wh - 0.15))
            if not self.plume.is_position_valid(x, y, margin=0.16):
                continue
            if math.hypot(x - self.plume.source_x, y - self.plume.source_y) < 0.15:
                continue
            heading = math.pi + float(self.rng.uniform(-0.4, 0.4))
            self.robot.reset(x, y, heading)
            return self.robot.state
        # 兜底：场地中偏下风处。
        self.robot.reset(0.5 * wh, 0.0, math.pi)
        return self.robot.state

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
        obs, info = super().reset(seed=seed, options=options)
        self.prev_distance = self._distance_to_source()
        info["distance_to_source"] = self.prev_distance
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
        self.robot_state = self.robot.step(move)
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
        reward = self._source_search_reward(
            distance,
            reached,
            oob,
            left,
            right,
            whisker_moved=whisker_moved,
            mobile_translating=(
                DifferentialDriveRobot.ACTIONS[move]
                in {"forward", "turn_left", "turn_right"}
            ),
        )

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
                "reward": reward,
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
                "distance_to_source": distance,
                "out_of_bounds": oob,
                "is_success": reached,
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
        *,
        whisker_moved: bool,
        mobile_translating: bool,
    ) -> float:
        """触须采到的气味为主导稠密项 + 弱 progress 引导 + 终点/惩罚。

        odor_reach / trend / contrast 三项都依赖触须端读数，因而依赖触须指向：
        把触须主动指向气味缕、扩大命中、制造左右不对称都会直接涨分，使主动采样
        对回报有可显现的影响，而非被特权 progress 掩盖。
        """
        r = self.progress_reward_scale * (self.prev_distance - distance)
        # 主导稠密项：触须端采到的气味幅值（依赖触须指向）。
        r += self.odor_reach_scale * float(
            np.clip(max(left, right), 0.0, self.odor_reach_clip)
        )
        if max(left, right) >= self.hit_threshold:
            r += self.mobile_odor_hit_reward
        trend = max(left - self.prev_left, right - self.prev_right)
        r += self.mobile_trend_scale * float(
            np.clip(trend, -self.mobile_trend_clip, self.mobile_trend_clip)
        )
        contrast = abs(left - right)
        if self.mobile_contrast_mode == "motion_gated":
            contrast_signal = contrast if whisker_moved else 0.0
            if not mobile_translating:
                contrast_signal *= self.stationary_sampling_reward_factor
        elif self.mobile_contrast_mode == "absolute":
            contrast_signal = contrast
        else:
            previous_contrast = abs(self.prev_left - self.prev_right)
            contrast_signal = float(
                np.clip(
                    contrast - previous_contrast,
                    -self.mobile_contrast_delta_clip,
                    self.mobile_contrast_delta_clip,
                )
            )
        r += self.mobile_contrast_scale * contrast_signal
        r -= self.mobile_time_penalty
        if reached:
            r += self.goal_bonus
        if oob:
            r -= self.oob_penalty
        return float(r)
