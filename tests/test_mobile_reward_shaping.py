from __future__ import annotations

import math
import unittest

from dual_whisker_rl.envs import MobileWhiskerPuffEnv


FAST_CONFIG = {"plume_overrides": {"puff_warmup_steps": 1}}


class MobileRewardShapingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = MobileWhiskerPuffEnv(FAST_CONFIG)
        self.env.reset(seed=123)

    def tearDown(self) -> None:
        self.env.close()

    def _components(
        self,
        *,
        distance: float = 1.0,
        left: float = 0.0,
        right: float = 0.0,
        displacement: float = 0.01,
        reached: bool = False,
        oob: bool = False,
    ) -> dict[str, float]:
        return self.env._source_search_reward_components(
            distance,
            reached,
            oob,
            left,
            right,
            displacement,
        )

    def test_absolute_concentration_cannot_be_rewarded_repeatedly(self) -> None:
        self.env.prev_distance = 1.0
        first = self._components(left=0.8, right=0.2)
        repeated = self._components(left=0.8, right=0.2)

        self.assertAlmostEqual(first["best_concentration_improvement"], 4.0)
        self.assertEqual(repeated["best_concentration_improvement"], 0.0)

    def test_concentration_improvement_is_bounded_and_ignores_contrast(self) -> None:
        self.env.prev_distance = 1.0
        total = 0.0
        for concentration in (0.2, 0.5, 0.4, 1.2, 0.9):
            components = self._components(left=concentration, right=0.0)
            total += components["best_concentration_improvement"]
        self.assertAlmostEqual(total, 5.0)

        first_env = MobileWhiskerPuffEnv(FAST_CONFIG)
        second_env = MobileWhiskerPuffEnv(FAST_CONFIG)
        try:
            first_env.reset(seed=9)
            second_env.reset(seed=9)
            first_env.prev_distance = second_env.prev_distance = 1.0
            asymmetric = first_env._source_search_reward_components(
                1.0, False, False, 0.8, 0.1, 0.01
            )
            symmetric = second_env._source_search_reward_components(
                1.0, False, False, 0.8, 0.8, 0.01
            )
            self.assertEqual(asymmetric, symmetric)
        finally:
            first_env.close()
            second_env.close()

    def test_signed_distance_potential_telescopes(self) -> None:
        self.env.prev_distance = 1.5
        closer = self._components(distance=1.2)["distance_potential_progress"]
        self.env.prev_distance = 1.2
        farther = self._components(distance=1.4)["distance_potential_progress"]
        self.env.prev_distance = 1.4
        returned = self._components(distance=1.5)["distance_potential_progress"]
        self.env.prev_distance = 1.5
        stationary = self._components(distance=1.5)["distance_potential_progress"]

        self.assertAlmostEqual(closer, 0.3)
        self.assertAlmostEqual(farther, -0.2)
        self.assertAlmostEqual(returned, -0.1)
        self.assertAlmostEqual(closer + farther + returned, 0.0)
        self.assertEqual(stationary, 0.0)

    def test_stagnation_requires_five_consecutive_no_progress_steps(self) -> None:
        self.env.prev_distance = 1.0
        self.env.best_concentration = 0.5
        for expected_streak in range(1, 5):
            components = self._components(
                distance=1.0,
                left=0.5,
                right=0.5,
                displacement=0.0,
            )
            self.assertEqual(self.env.stagnation_steps, expected_streak)
            self.assertEqual(components["stagnation_penalty"], 0.0)

        fifth = self._components(
            distance=1.0,
            left=0.5,
            right=0.5,
            displacement=0.0,
        )
        self.assertEqual(self.env.stagnation_steps, 5)
        self.assertEqual(fifth["stagnation_penalty"], -0.10)

        moved = self._components(
            distance=1.0,
            left=0.5,
            right=0.5,
            displacement=0.01,
        )
        self.assertEqual(self.env.stagnation_steps, 0)
        self.assertEqual(moved["stagnation_penalty"], 0.0)

        self.env.stagnation_steps = 4
        concentration_progress = self._components(
            distance=1.0,
            left=0.52,
            right=0.5,
            displacement=0.0,
        )
        self.assertEqual(self.env.stagnation_steps, 0)
        self.assertEqual(concentration_progress["stagnation_penalty"], 0.0)

        self.env.stagnation_steps = 4
        self.env.prev_distance = 1.0
        distance_progress = self._components(
            distance=0.99,
            left=0.52,
            right=0.5,
            displacement=0.0,
        )
        self.assertEqual(self.env.stagnation_steps, 0)
        self.assertEqual(distance_progress["stagnation_penalty"], 0.0)

    def test_terminal_step_never_receives_stagnation_penalty(self) -> None:
        self.env.prev_distance = 1.0
        self.env.best_concentration = 0.5
        self.env.stagnation_steps = 4
        components = self._components(
            distance=1.0,
            left=0.5,
            right=0.5,
            displacement=0.0,
            reached=True,
        )
        self.assertEqual(self.env.stagnation_steps, 0)
        self.assertEqual(components["stagnation_penalty"], 0.0)
        self.assertEqual(components["goal_bonus"], 50.0)

    def test_reward_component_contract_and_auxiliary_bound(self) -> None:
        self.assertEqual(
            tuple(self._components().keys()),
            MobileWhiskerPuffEnv.REWARD_COMPONENT_NAMES,
        )
        expected_bound = 5.0 + 2.0 * math.sqrt(2.0)
        self.assertAlmostEqual(
            self.env.positive_auxiliary_reward_upper_bound,
            expected_bound,
        )
        self.assertGreaterEqual(
            self.env.goal_bonus,
            5.0 * self.env.positive_auxiliary_reward_upper_bound,
        )

    def test_invalid_and_obsolete_reward_configs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "obsolete mobile reward config"):
            MobileWhiskerPuffEnv({"odor_reach_scale": 0.3})
        with self.assertRaisesRegex(ValueError, "fixed at 50"):
            MobileWhiskerPuffEnv({"goal_bonus": 40.0})
        with self.assertRaisesRegex(ValueError, "upper bound"):
            MobileWhiskerPuffEnv(
                {"best_concentration_reward_scale": 7.2}
            )


if __name__ == "__main__":
    unittest.main()
