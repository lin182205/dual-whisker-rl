"""Dual-whisker active sampling model."""

from __future__ import annotations

from dataclasses import dataclass
import math

from dual_whisker_rl.envs.robot_model import RobotState


@dataclass
class WhiskerState:
    left_angle: float
    right_angle: float
    left_point: tuple[float, float]
    right_point: tuple[float, float]


class DualWhiskerSampler:
    """Maps independent left/right sector actions to sampling points."""

    def __init__(
        self,
        length: float = 0.3,
        sector_count: int = 10,
        servo_60deg_time_s: float | None = 0.12,
    ) -> None:
        self.length = float(length)
        self.sector_count = int(sector_count)
        if self.sector_count <= 0:
            raise ValueError("sector_count must be positive")
        self.servo_60deg_time_s = (
            float(servo_60deg_time_s) if servo_60deg_time_s is not None else None
        )
        if self.servo_60deg_time_s is not None and self.servo_60deg_time_s <= 0.0:
            raise ValueError("servo_60deg_time_s must be positive")
        self.angular_speed_rad_s = (
            math.radians(60.0) / self.servo_60deg_time_s
            if self.servo_60deg_time_s is not None
            else None
        )
        self.left_angle = self._sector_angle(0, side="left")
        self.right_angle = self._sector_angle(0, side="right")

    def reset(self) -> None:
        self.left_angle = self._sector_angle(0, side="left")
        self.right_angle = self._sector_angle(0, side="right")

    def step(
        self,
        left_sector: int,
        right_sector: int,
        robot_state: RobotState,
        dt: float | None = None,
    ) -> WhiskerState:
        left_target = self._sector_angle(left_sector, side="left")
        right_target = self._sector_angle(right_sector, side="right")
        if dt is None or self.angular_speed_rad_s is None:
            self.left_angle = left_target
            self.right_angle = right_target
        else:
            max_delta = self.angular_speed_rad_s * max(float(dt), 0.0)
            self.left_angle = self._move_toward(
                self.left_angle,
                left_target,
                max_delta,
            )
            self.right_angle = self._move_toward(
                self.right_angle,
                right_target,
                max_delta,
            )
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

    def _sector_angle(self, sector: int, *, side: str) -> float:
        sector = int(sector)
        if not 0 <= sector < self.sector_count:
            raise ValueError(
                f"{side}_sector must be in [0, {self.sector_count - 1}], got {sector}"
            )

        center_fraction = (sector + 0.5) / self.sector_count
        angle = center_fraction * math.pi
        if side == "left":
            return float(angle)
        if side == "right":
            return float(-angle)
        raise ValueError(f"Unknown whisker side: {side}")

    @staticmethod
    def _move_toward(current: float, target: float, max_delta: float) -> float:
        """Move one servo angle toward its target without overshooting."""
        delta = target - current
        if abs(delta) <= max_delta:
            return float(target)
        return float(current + math.copysign(max_delta, delta))
