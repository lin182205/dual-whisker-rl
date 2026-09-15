import unittest

import numpy as np

from dual_whisker_rl.e4_experiments import (
    METHOD_ORDER,
    ObservationHistory,
    aggregate,
    build_scenarios,
    make_env,
    reset_scenario,
)


class E4ExperimentTests(unittest.TestCase):
    def test_formal_scene_counts_and_hash_reproducibility(self):
        first = build_scenarios("test", "formal")
        second = build_scenarios("test", "formal")
        self.assertEqual(len(first), 648)
        self.assertEqual([s.to_dict() for s in first], [s.to_dict() for s in second])
        self.assertEqual(len(build_scenarios("validation", "formal")), 216)

    def test_all_methods_reset_and_step(self):
        scenario = build_scenarios("test", "smoke")[0]
        for method in METHOD_ORDER:
            env = make_env(method)
            try:
                obs, _ = reset_scenario(env, scenario)
                self.assertEqual(obs.shape, (16 if method == "b4_single_frame" else 320,))
                action = env.action_space.sample()
                _, reward, terminated, truncated, info = env.step(action)
                self.assertTrue(np.isfinite(reward))
                self.assertFalse(terminated and truncated)
                self.assertIn("scenario_mode", info)
            finally:
                env.close()

    def test_single_whisker_masks_inactive_features_and_is_reproducible(self):
        scenario = build_scenarios("test", "smoke")[0]
        for method, zero_indices in (("b6_left", (1, 3, 5, 6, 7, 9, 11)), ("b6_right", (0, 2, 4, 6, 7, 8, 10))):
            env = make_env(method)
            try:
                first, _ = reset_scenario(env, scenario)
                second, _ = reset_scenario(env, scenario)
                np.testing.assert_array_equal(first, second)
                np.testing.assert_array_equal(first[-16:][list(zero_indices)], np.zeros(len(zero_indices), dtype=np.float32))
            finally:
                env.close()

    def test_aggregate_has_one_level_breakdowns(self):
        rows = [{"success": True, "steps": 1, "final_distance": 0.1, "path_length": 0.2, "odor_hit_ratio": 1.0, "reacquisition_events": 0, "whisker_only_reacquisition_events": 0, "blank_body_spin_ratio": 0.0, "blank_whisker_motion_ratio": 0.0, "reward": 1.0, "distance_bin": "near", "wind_speed": 0.12}]
        result = aggregate(rows)
        self.assertEqual(result["episodes"], 1)
        self.assertEqual(result["by_distance_bin"]["near"]["episodes"], 1)
        self.assertNotIn("by_distance_bin", result["by_distance_bin"]["near"])


if __name__ == "__main__":
    unittest.main()
