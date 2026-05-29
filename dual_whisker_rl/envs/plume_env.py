"""Gymnasium environment for dual-whisker odor source localization."""

from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from dual_whisker_rl.envs.plume_model import GaussianPlume, PlumeParams
from dual_whisker_rl.envs.robot_model import DifferentialDriveRobot
from dual_whisker_rl.envs.sensor_model import FirstOrderGasSensor
from dual_whisker_rl.envs.whisker_model import DualWhiskerSampler


class PlumeEnv(gym.Env):
    """Minimal 2D RL task for active olfactory sampling."""

    metadata = {"render_modes": []}

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        self.width = float(cfg.get("width", 10.0))
        self.height = float(cfg.get("height", 10.0))
        self.max_steps = int(cfg.get("max_steps", 500))
        self.goal_radius = float(cfg.get("goal_radius", 0.35))
        self.hit_threshold = float(cfg.get("hit_threshold", 0.08))
        self.progress_reward_scale = float(cfg.get("progress_reward_scale", 5.0))
        self.odor_hit_reward = float(cfg.get("odor_hit_reward", 0.02))
        self.gradient_reward_scale = float(cfg.get("gradient_reward_scale", 0.03))
        self.time_penalty = float(cfg.get("time_penalty", 0.03))
        self.stop_penalty = float(cfg.get("stop_penalty", 0.05))
        self.spin_penalty = float(cfg.get("spin_penalty", 0.02))
        self.near_source_rewards = tuple(
            (float(radius), float(reward))
            for radius, reward in cfg.get(
                "near_source_rewards",
                (
                    (2.0, 0.05),
                    (1.0, 0.15),
                    (0.5, 0.35),
                ),
            )
        )

        source = tuple(cfg.get("source", (8.0, 5.0)))
        wind_direction = math.radians(float(cfg.get("wind_direction_deg", 180.0)))
        plume_params = PlumeParams(source=source, wind_direction=wind_direction)
        self.plume = GaussianPlume(plume_params)

        self.robot = DifferentialDriveRobot(
            forward_speed=float(cfg.get("forward_speed", 0.15)),
            turn_rate=math.radians(float(cfg.get("turn_rate_deg", 20.0))),
            dt=float(cfg.get("dt", 1.0)),
        )
        self.whiskers = DualWhiskerSampler(
            length=float(cfg.get("whisker_length", 0.3)),
            sector_count=int(cfg.get("whisker_sector_count", 10)),
        )
        sensor_alpha = float(cfg.get("sensor_alpha", 0.95))
        self.left_sensor = FirstOrderGasSensor(sensor_alpha)
        self.right_sensor = FirstOrderGasSensor(sensor_alpha)

        self.action_space = spaces.MultiDiscrete(
            [
                len(DifferentialDriveRobot.ACTIONS),
                self.whiskers.sector_count,
                self.whiskers.sector_count,
            ]
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32,
        )

        self.rng = np.random.default_rng()
        self.step_count = 0
        self.prev_distance = 0.0
        self.prev_left = 0.0
        self.prev_right = 0.0
        self.prev_action = 0.0
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
        self.prev_action = 0.0
        self.prev_left = 0.0
        self.prev_right = 0.0
        self.left_hits.clear()
        self.right_hits.clear()
        self.trajectory.clear()
        self.left_sensor.reset()
        self.right_sensor.reset()
        self.whiskers.reset()

        start_x = self.rng.uniform(0.5, 2.0)
        start_y = self.rng.uniform(2.0, 8.0)
        heading = self.rng.uniform(-0.3, 0.3)
        state = self.robot.reset(start_x, start_y, heading)
        self.prev_distance = self._distance_to_source(state.x, state.y)

        obs, info = self._observe()
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        move_action, left_sector, right_sector = self._decode_action(action)
        return self._step_with_actions(
            move_action,
            left_sector,
            right_sector,
            self._logged_action(action),
        )

    def _step_with_actions(
        self,
        move_action: int,
        left_sector: int,
        right_sector: int,
        logged_action: float,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        state = self.robot.step(move_action)
        whisker_state = self.whiskers.step(left_sector, right_sector, state)

        raw_left = self.plume.concentration(*whisker_state.left_point, t=self.step_count)
        raw_right = self.plume.concentration(*whisker_state.right_point, t=self.step_count)
        left = self.left_sensor.update(raw_left)
        right = self.right_sensor.update(raw_right)

        hit_left = 1.0 if left >= self.hit_threshold else 0.0
        hit_right = 1.0 if right >= self.hit_threshold else 0.0
        self.left_hits.append(hit_left)
        self.right_hits.append(hit_right)
        self.left_hits = self.left_hits[-20:]
        self.right_hits = self.right_hits[-20:]

        distance = self._distance_to_source(state.x, state.y)
        terminated = distance <= self.goal_radius
        out_of_bounds = not (0.0 <= state.x <= self.width and 0.0 <= state.y <= self.height)
        truncated = self.step_count + 1 >= self.max_steps

        reward = self._reward(
            distance,
            terminated,
            out_of_bounds,
            move_action,
            left,
            right,
        )
        if out_of_bounds:
            terminated = True

        self.step_count += 1
        self.prev_distance = distance
        self.prev_left = left
        self.prev_right = right
        self.prev_action = float(logged_action)

        self.trajectory.append(
            {
                "x": state.x,
                "y": state.y,
                "heading": state.heading,
                "left": left,
                "right": right,
                "raw_left": raw_left,
                "raw_right": raw_right,
                "left_angle": whisker_state.left_angle,
                "right_angle": whisker_state.right_angle,
                "left_sector": int(left_sector),
                "right_sector": int(right_sector),
                "move_action": DifferentialDriveRobot.ACTIONS[move_action],
                "distance_to_source": distance,
                "reward": reward,
            }
        )

        obs, info = self._observe()
        info.update(
            {
                "distance_to_source": distance,
                "out_of_bounds": out_of_bounds,
                "move_action": DifferentialDriveRobot.ACTIONS[move_action],
                "left_sector": int(left_sector),
                "right_sector": int(right_sector),
            }
        )
        return obs, reward, terminated, truncated, info

    def concentration_grid(self, resolution: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.plume.grid(self.width, self.height, resolution, self.step_count)

    def _observe(self) -> tuple[np.ndarray, dict[str, Any]]:
        state = self.robot.state
        whisker_state = self.whiskers.state(state)
        left = self.left_sensor.value
        right = self.right_sensor.value
        wind_rel = self._wrap_angle(self.plume.params.wind_direction - state.heading)

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
                math.cos(state.heading),
                math.sin(state.heading),
                self.prev_action,
            ],
            dtype=np.float32,
        )
        return obs, {
            "robot_state": state,
            "whisker_state": whisker_state,
            "step": self.step_count,
        }

    def _reward(
        self,
        distance: float,
        reached_goal: bool,
        out_of_bounds: bool,
        move_action: int,
        left: float,
        right: float,
    ) -> float:
        reward = self.progress_reward_scale * (self.prev_distance - distance)
        reward += self.odor_hit_reward if max(left, right) >= self.hit_threshold else 0.0
        reward += self.gradient_reward_scale * abs(left - right)
        reward -= self.time_penalty

        move_name = DifferentialDriveRobot.ACTIONS[int(move_action)]
        if move_name == "stop":
            reward -= self.stop_penalty
        elif move_name in {"spin_left", "spin_right"}:
            reward -= self.spin_penalty

        for radius, bonus in self.near_source_rewards:
            if distance <= radius:
                reward += bonus

        if reached_goal:
            reward += 100.0
        if out_of_bounds:
            reward -= 50.0
        return float(reward)

    def _decode_action(self, action: Any) -> tuple[int, int, int]:
        arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if arr.size != 3:
            raise ValueError(
                "PlumeEnv action must contain [move_action, left_sector, right_sector]"
            )
        return int(arr[0]), int(arr[1]), int(arr[2])

    def _logged_action(self, action: Any) -> float:
        move_action, left_sector, right_sector = self._decode_action(action)
        move_scale = max(1, len(DifferentialDriveRobot.ACTIONS) - 1)
        sector_scale = max(1, self.whiskers.sector_count - 1)
        return float(
            (
                move_action / move_scale
                + left_sector / sector_scale
                + right_sector / sector_scale
            )
            / 3.0
        )

    def _distance_to_source(self, x: float, y: float) -> float:
        sx, sy = self.plume.params.source
        return float(math.hypot(x - sx, y - sy))

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
