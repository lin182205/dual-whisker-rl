"""Dual-whisker active sampling model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from dual_whisker_rl.envs.robot_model import RobotState


@dataclass
class WhiskerState:
    left_angle: float
    right_angle: float
    left_point: tuple[float, float]
    right_point: tuple[float, float]


class DualWhiskerSampler:
    """Maps a discrete sensing action to left/right sampling points."""

    ACTIONS = (
        "center_hold",
        "narrow_scan",
        "wide_scan",
        "left_focus",
        "right_focus",
    )

    def __init__(
        self,
        length: float = 0.3,
        max_angle_deg: float = 60.0,
        scan_rate: float = 0.45,
    ) -> None:
        self.length = float(length)
        self.max_angle = math.radians(float(max_angle_deg))
        self.scan_rate = float(scan_rate)
        self.phase = 0.0
        self.left_angle = 0.0
        self.right_angle = 0.0

    def reset(self) -> None:
        self.phase = 0.0
        self.left_angle = 0.0
        self.right_angle = 0.0

    def step(self, action_id: int, robot_state: RobotState) -> WhiskerState:
        action = self.ACTIONS[int(action_id)]
        self.phase += self.scan_rate

        if action == "center_hold":
            left_angle = 0.0
            right_angle = 0.0
        elif action == "narrow_scan":
            amplitude = 0.35 * self.max_angle
            left_angle = amplitude * math.sin(self.phase)
            right_angle = -amplitude * math.sin(self.phase)
        elif action == "wide_scan":
            amplitude = 0.95 * self.max_angle
            left_angle = amplitude * math.sin(self.phase)
            right_angle = -amplitude * math.sin(self.phase)
        elif action == "left_focus":
            left_angle = 0.75 * self.max_angle
            right_angle = 0.35 * self.max_angle
        elif action == "right_focus":
            left_angle = -0.35 * self.max_angle
            right_angle = -0.75 * self.max_angle
        else:
            raise ValueError(f"Unknown whisker action: {action}")

        self.left_angle = float(np.clip(left_angle, -self.max_angle, self.max_angle))
        self.right_angle = float(np.clip(right_angle, -self.max_angle, self.max_angle))
        return self.state(robot_state)

    def state(self, robot_state: RobotState) -> WhiskerState:
        left_point = self._point(robot_state, self.left_angle)
        right_point = self._point(robot_state, self.right_angle)
        return WhiskerState(self.left_angle, self.right_angle, left_point, right_point)

    def _point(self, robot_state: RobotState, relative_angle: float) -> tuple[float, float]:
        angle = robot_state.heading + relative_angle
        return (
            robot_state.x + self.length * math.cos(angle),
            robot_state.y + self.length * math.sin(angle),
        )
