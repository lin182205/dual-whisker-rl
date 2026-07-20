from __future__ import annotations

from collections import Counter
import math
import unittest

import numpy as np

from dual_whisker_rl.envs import MobileWhiskerPuffEnv


FAST_PLUME = {"plume_overrides": {"puff_warmup_steps": 1}}


class MobileScenarioInitializationTests(unittest.TestCase):
    def test_randomized_reset_is_reproducible(self) -> None:
        env = MobileWhiskerPuffEnv(FAST_PLUME)
        try:
            first_obs, first_info = env.reset(seed=1234)
            second_obs, second_info = env.reset(seed=1234)
        finally:
            env.close()

        np.testing.assert_array_equal(first_obs, second_obs)
        self.assertEqual(first_info["source"], second_info["source"])
        self.assertEqual(first_info["robot_state"], second_info["robot_state"])
        self.assertEqual(
            first_info["initial_heading_mode"],
            second_info["initial_heading_mode"],
        )
        self.assertEqual(first_info["wind_direction"], second_info["wind_direction"])

    def test_randomized_distribution_is_valid_and_covers_all_quadrants(self) -> None:
        env = MobileWhiskerPuffEnv(FAST_PLUME)
        quadrants: Counter[tuple[bool, bool]] = Counter()
        heading_modes: Counter[str] = Counter()
        try:
            for seed in range(256):
                _, info = env.reset(seed=seed)
                source_x, source_y = info["source"]
                robot = info["robot_state"]
                wind = env.plume.wind_direction_base
                wind_x = math.cos(wind)
                wind_y = math.sin(wind)

                quadrants[(source_x >= 0.0, source_y >= 0.0)] += 1
                heading_modes[info["initial_heading_mode"]] += 1
                self.assertTrue(
                    env.plume.is_position_valid(
                        source_x,
                        source_y,
                        margin=env.plume.source_clearance,
                    )
                )
                self.assertTrue(
                    env.plume.is_position_valid(robot.x, robot.y, margin=0.16)
                )

                downwind = (
                    (robot.x - source_x) * wind_x
                    + (robot.y - source_y) * wind_y
                )
                robot_crosswind = (
                    -(robot.x - source_x) * wind_y
                    + (robot.y - source_y) * wind_x
                )
                self.assertGreaterEqual(
                    downwind,
                    env.scenario_robot_downwind_range[0] - 1e-9,
                )
                self.assertLessEqual(
                    downwind,
                    env.scenario_robot_downwind_range[1] + 1e-9,
                )
                self.assertGreaterEqual(
                    robot_crosswind,
                    env.scenario_robot_crosswind_range[0] - 1e-9,
                )
                self.assertLessEqual(
                    robot_crosswind,
                    env.scenario_robot_crosswind_range[1] + 1e-9,
                )
                self.assertGreater(
                    math.hypot(robot.x - source_x, robot.y - source_y),
                    env.goal_radius,
                )
                source_distance = -(source_x * wind_x + source_y * wind_y)
                source_crosswind = -source_x * wind_y + source_y * wind_x
                self.assertGreaterEqual(
                    source_distance,
                    env.scenario_source_distance_range[0] - 1e-9,
                )
                self.assertLessEqual(
                    source_distance,
                    env.scenario_source_distance_range[1] + 1e-9,
                )
                self.assertGreaterEqual(
                    source_crosswind,
                    env.scenario_source_crosswind_range[0] - 1e-9,
                )
                self.assertLessEqual(
                    source_crosswind,
                    env.scenario_source_crosswind_range[1] + 1e-9,
                )
        finally:
            env.close()

        self.assertEqual(len(quadrants), 4)
        self.assertTrue(all(count >= 32 for count in quadrants.values()))
        uniform_ratio = heading_modes["uniform"] / 256.0
        self.assertGreaterEqual(uniform_ratio, 0.65)
        self.assertLessEqual(uniform_ratio, 0.85)

    def test_fixed_mode_preserves_legacy_geometry(self) -> None:
        env = MobileWhiskerPuffEnv({**FAST_PLUME, "scenario_mode": "fixed"})
        try:
            for seed in range(16):
                _, info = env.reset(seed=seed)
                robot = info["robot_state"]
                self.assertEqual(info["source"], (-0.8, 0.0))
                self.assertGreaterEqual(robot.x, 0.35)
                self.assertLessEqual(robot.x, 0.85)
                self.assertGreaterEqual(robot.y, -0.85)
                self.assertLessEqual(robot.y, 0.85)
                heading_error = math.atan2(
                    math.sin(robot.heading - math.pi),
                    math.cos(robot.heading - math.pi),
                )
                self.assertLessEqual(abs(heading_error), 0.4)
                self.assertEqual(info["initial_heading_mode"], "source_facing")
        finally:
            env.close()

    def test_explicit_source_infers_fixed_mode_and_conflicts_with_randomized(self) -> None:
        inferred = MobileWhiskerPuffEnv(
            {**FAST_PLUME, "source_position": (-0.6, 0.2)}
        )
        try:
            self.assertEqual(inferred.scenario_mode, "fixed")
            _, info = inferred.reset(seed=7)
            self.assertEqual(info["source"], (-0.6, 0.2))
        finally:
            inferred.close()

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            MobileWhiskerPuffEnv(
                {
                    **FAST_PLUME,
                    "scenario_mode": "randomized",
                    "source_position": (-0.6, 0.2),
                }
            )

    def test_modes_keep_observation_and_action_spaces_compatible(self) -> None:
        randomized = MobileWhiskerPuffEnv(FAST_PLUME)
        fixed = MobileWhiskerPuffEnv({**FAST_PLUME, "scenario_mode": "fixed"})
        try:
            self.assertEqual(randomized.observation_space, fixed.observation_space)
            self.assertEqual(randomized.action_space, fixed.action_space)
        finally:
            randomized.close()
            fixed.close()

    def test_invalid_randomized_ranges_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "lower bound"):
            MobileWhiskerPuffEnv(
                {"scenario_robot_downwind_range": (1.0, 0.5)}
            )
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            MobileWhiskerPuffEnv(
                {"scenario_uniform_heading_probability": 1.5}
            )


if __name__ == "__main__":
    unittest.main()
