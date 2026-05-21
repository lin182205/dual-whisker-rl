"""Simple unicycle robot model for the first 2D simulation."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class RobotState:
    x: float
    y: float
    heading: float


class DifferentialDriveRobot:
    """Discrete-action mobile robot approximation."""

    ACTIONS = ("forward", "turn_left", "turn_right", "spin_left", "spin_right", "stop")

    def __init__(
        self,
        forward_speed: float = 0.15,
        turn_rate: float = 0.35,
        dt: float = 1.0,
    ) -> None:
        self.forward_speed = float(forward_speed)
        self.turn_rate = float(turn_rate)
        self.dt = float(dt)
        self.state = RobotState(0.0, 0.0, 0.0)

    def reset(self, x: float, y: float, heading: float) -> RobotState:
        self.state = RobotState(float(x), float(y), self._wrap_angle(heading))
        return self.state

    def step(self, action_id: int) -> RobotState:
        action = self.ACTIONS[int(action_id)]
        x, y, heading = self.state.x, self.state.y, self.state.heading

        if action == "forward":
            x += self.forward_speed * self.dt * math.cos(heading)
            y += self.forward_speed * self.dt * math.sin(heading)
        elif action == "turn_left":
            heading += self.turn_rate * self.dt
            x += 0.5 * self.forward_speed * self.dt * math.cos(heading)
            y += 0.5 * self.forward_speed * self.dt * math.sin(heading)
        elif action == "turn_right":
            heading -= self.turn_rate * self.dt
            x += 0.5 * self.forward_speed * self.dt * math.cos(heading)
            y += 0.5 * self.forward_speed * self.dt * math.sin(heading)
        elif action == "spin_left":
            heading += self.turn_rate * self.dt
        elif action == "spin_right":
            heading -= self.turn_rate * self.dt
        elif action == "stop":
            pass
        else:
            raise ValueError(f"Unknown robot action: {action}")

        self.state = RobotState(float(x), float(y), self._wrap_angle(heading))
        return self.state

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)
