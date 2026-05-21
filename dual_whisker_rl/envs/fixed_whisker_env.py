"""Baseline environment with fixed periodic whisker motion."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from dual_whisker_rl.envs.plume_env import PlumeEnv
from dual_whisker_rl.envs.robot_model import DifferentialDriveRobot


class FixedWhiskerPlumeEnv(PlumeEnv):
    """Baseline: the policy controls movement only, whiskers swing by a fixed rule."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.action_space = spaces.Discrete(len(DifferentialDriveRobot.ACTIONS))

    def step(self, action: int):
        move_action = int(action)
        sector = self.step_count % self.whiskers.sector_count
        return self._step_with_actions(
            move_action=move_action,
            left_sector=sector,
            right_sector=sector,
            logged_action=move_action / max(1, self.action_space.n - 1),
        )
