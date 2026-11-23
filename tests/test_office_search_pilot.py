"""核对办公寻源预实验的绕圈扣分、起点课程和判定。"""

from pathlib import Path
import unittest

import numpy as np
import yaml

from dual_whisker_rl.e4_experiments import metrics_from_trajectory
from dual_whisker_rl.envs.mobile_whisker_env import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.robot_model import RobotState


ROOT = Path(__file__).resolve().parents[1]
CONFIG = yaml.safe_load((ROOT / "configs/mobile_rooms_hybrid_history.yaml").read_text(encoding="utf-8"))
START = {"source_position": [3., -3.5], "robot_pose": [0., 0., 0.]}


def make_env(**overrides):
    return MobileWhiskerPuffEnv({
        **CONFIG, "plume_overrides": {**CONFIG["plume_overrides"], "puff_warmup_steps": 0},
        **overrides,
    })


class OfficeSearchPilotTests(unittest.TestCase):
    def test_blank_loop_cost_and_reset_at_25_steps(self):
        with make_env(blank_loop_penalty=.006, max_steps=28) as env:
            env.reset(seed=11, options={"scenario": START})
            for step in range(24):
                _, _, _, _, info = env.step([1., 1., 5, 5])
                self.assertEqual(info["reward_components"]["blank_loop_penalty"], 0, step)
            _, _, _, _, info = env.step([1., 1., 5, 5])
            self.assertTrue(info["blank_loop"])
            self.assertEqual(info["reward_components"]["blank_loop_penalty"], -.006)
            _, _, _, _, info = env.step([-1., 0., 0, 9])
            self.assertFalse(info["blank_loop"])
            self.assertEqual(info["reward_components"]["blank_loop_penalty"], 0)
            env.step([1., 1., 5, 5])
            _, _, _, truncated, info = env.step([1., 1., 5, 5])
            self.assertTrue(truncated)
            self.assertTrue(info["blank_loop"])
            self.assertEqual(info["reward_components"]["blank_loop_penalty"], 0)
            env.reset(seed=11, options={"scenario": START})
            _, _, _, _, info = env.step([1., 1., 5, 5])
            self.assertFalse(info["blank_loop"])

    def test_window_with_hit_does_not_penalize(self):
        with make_env(blank_loop_penalty=.006) as env:
            env.reset(seed=11, options={"scenario": START})
            for _ in range(24):
                env.step([1., 1., 5, 5])
            env.left_sensor.update = lambda _: .21
            _, _, _, _, info = env.step([1., 1., 5, 5])
            self.assertTrue(info["odor_hit"])
            self.assertFalse(info["blank_loop"])
            self.assertEqual(info["reward_components"]["blank_loop_penalty"], 0)

    def test_tips_cross_open_door_and_stop_at_wall(self):
        with make_env(whisker_length=.30) as env:
            heading = -env.whiskers.left_angle
            door = env.whiskers.state(RobotState(-1.04, -3.5, heading)).left_point
            wall = env.whiskers.state(RobotState(-1.04, -2.5, heading)).left_point
            self.assertAlmostEqual(door[0], -.74)
            self.assertLess(wall[0], -.85)
            self.assertAlmostEqual(wall[0], -.85, places=5)

    def test_curriculum_use_and_explicit_scenario_override(self):
        with make_env(office_start_curriculum=True) as env:
            for seed in range(30):
                env.reset(seed=seed)
                if env.initial_heading_mode == "curriculum_0":
                    nav = env.room_navigation
                    self.assertEqual(nav.start_room, nav.map.room_id(nav.source))
                    self.assertTrue(1. <= nav.initial_path_distance <= 3.)
                    break
            else:
                self.fail("同房课程未采样到有效回合")
            for stage in (0, 1, 2):
                env.set_office_curriculum_stage(stage)
                env.reset(seed=11, options={"scenario": START})
                np.testing.assert_allclose([env.robot_state.x, env.robot_state.y,
                                            env.robot_state.heading], START["robot_pose"])
                self.assertEqual(env.initial_heading_mode, "scenario")

    def test_visibility_success_flag_has_priority(self):
        last = {"x": 0., "y": 0., "distance_to_source": .08, "is_success": False}
        self.assertFalse(metrics_from_trajectory([last], .10)["success"])
        self.assertTrue(metrics_from_trajectory([{**last, "is_success": True}], .10)["success"])


if __name__ == "__main__":
    unittest.main()
