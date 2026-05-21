"""Baseline environment with fixed periodic whisker motion."""

from __future__ import annotations

from typing import Any

from gymnasium import spaces

from dual_whisker_rl.envs.plume_env import PlumeEnv
from dual_whisker_rl.envs.robot_model import DifferentialDriveRobot
from dual_whisker_rl.envs.whisker_model import DualWhiskerSampler


class FixedWhiskerPlumeEnv(PlumeEnv):
    """Baseline: the policy controls movement only, whiskers swing by a fixed rule."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        cfg = config or {}
        action_name = str(cfg.get("fixed_whisker_action", "wide_scan"))
        if action_name not in DualWhiskerSampler.ACTIONS:
            valid = ", ".join(DualWhiskerSampler.ACTIONS)
            raise ValueError(f"fixed_whisker_action must be one of: {valid}")

        self.fixed_whisker_action = DualWhiskerSampler.ACTIONS.index(action_name)
        self.action_space = spaces.Discrete(len(DifferentialDriveRobot.ACTIONS))

    def step(self, action: int):
        move_action = int(action)
        return self._step_with_actions(
            move_action=move_action,
            sense_action=self.fixed_whisker_action,
            logged_action=move_action,
        )
