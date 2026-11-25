"""空间记忆只依赖定位与命中，且新奖励不能通过重复访问刷分。"""

import math
from pathlib import Path
import unittest

import numpy as np
import yaml

from dual_whisker_rl.envs.mobile_whisker_env import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.spatial_memory import EpisodicSpatialMemory


ROOT = Path(__file__).resolve().parents[1]
START = {"source_position": [3., -3.5], "robot_pose": [.49, 0., 0.]}


def config(name="d", **overrides):
    value = yaml.safe_load((ROOT / f"configs/mobile_rooms_spatial_{name}.yaml").read_text(encoding="utf-8"))
    value["plume_overrides"]["puff_warmup_steps"] = 0
    return {**value, **overrides}


def advance(memory, position=(0., 0.), **overrides):
    return memory.update(position, **{
        "dt": .2, "odor_hit": False, "blank_age_s": 5., "terminal": False,
        **overrides,
    })


class SpatialMemoryUnitTests(unittest.TestCase):
    def test_floor_grid_boundaries_and_y_x_flattening(self):
        memory = EpisodicSpatialMemory()
        self.assertEqual(memory.cell_index((-5., -5.)), (0, 0))
        self.assertEqual(memory.cell_index((-.001, -.501)), (8, 9))
        self.assertEqual(memory.cell_index((0., -.5)), (9, 10))
        self.assertEqual(memory.cell_index((5., 5.)), (19, 19))
        self.assertEqual(memory.cell_index((5.2, -5.2)), (0, 19))
        memory.reset((0., -.5))
        obs = memory.observation()
        self.assertEqual(obs.shape, (802,))
        self.assertEqual(obs[9 * 20 + 10], 1.)
        self.assertEqual(np.count_nonzero(obs[:400]), 1)
        self.assertEqual(obs[-1], 1.)

    def test_reset_marks_start_without_reward_and_clears_all_state(self):
        memory = EpisodicSpatialMemory(exploration_bonus=.02)
        advance(memory, (.6, 0.))
        self.assertEqual(memory.novel_cells, 2)
        self.assertAlmostEqual(memory.budget_remaining, .48)
        memory.reset((-1., 1.))
        self.assertEqual(memory.novel_cells, 1)
        self.assertEqual(memory.visited.sum(), 1)
        self.assertEqual(memory.blank_exposure_s.sum(), 0.)
        self.assertEqual(memory.exploration_reward_total, 0.)
        self.assertEqual(memory.blank_revisit_cost_total, 0.)
        self.assertEqual(memory.no_new_age_s, 0.)
        self.assertEqual(memory.budget_remaining, .5)
        self.assertFalse(memory.new_cell)

    def test_first_visit_only_blank_ramp_and_odor_hit(self):
        memory = EpisodicSpatialMemory(exploration_bonus=.02)
        first = advance(memory, (.6, 0.), blank_age_s=1.)
        self.assertAlmostEqual(first["exploration_bonus"], .004)
        self.assertEqual(advance(memory, (.6, 0.))["exploration_bonus"], 0.)
        self.assertEqual(advance(memory, (0., 0.))["exploration_bonus"], 0.)
        hit = advance(memory, (1.1, 0.), odor_hit=True, blank_age_s=0.)
        self.assertEqual(hit["exploration_bonus"], 0.)
        self.assertEqual(memory.blank_exposure_s[memory.current_cell], 0.)
        self.assertEqual(memory.novel_cells, 3)

    def test_novelty_budget_caps_and_cannot_be_refilled(self):
        memory = EpisodicSpatialMemory(exploration_bonus=.02)
        for row in range(5):
            for col in range(20):
                advance(memory, (-4.75 + .5 * col, -4.75 + .5 * row))
        self.assertAlmostEqual(memory.exploration_reward_total, .5)
        self.assertEqual(memory.budget_remaining, 0.)
        self.assertEqual(advance(memory, (4., 4.))["exploration_bonus"], 0.)
        self.assertGreater(memory.novel_cells, 25)

    def test_blank_decay_cap_hit_clear_and_recovery(self):
        memory = EpisodicSpatialMemory()
        for _ in range(40):
            advance(memory)
        self.assertEqual(memory.blank_exposure_s[10, 10], 5.)
        advance(memory, (.6, 0.))
        self.assertAlmostEqual(memory.blank_exposure_s[10, 10], 5. * math.exp(-.2 / 60.))
        self.assertAlmostEqual(memory.blank_exposure_s[10, 11], .2)
        advance(memory, odor_hit=True, blank_age_s=0.)
        self.assertEqual(memory.blank_exposure_s[10, 10], 0.)
        advance(memory)
        self.assertAlmostEqual(memory.blank_exposure_s[10, 10], .2)

    def test_revisit_requires_both_timers_and_new_cell_resets_gate(self):
        memory = EpisodicSpatialMemory(blank_revisit_penalty=.006)
        for _ in range(24):
            reward = advance(memory)
            self.assertEqual(reward["blank_revisit_penalty"], 0.)
        reward = advance(memory)
        expected = -.006 * (memory.blank_exposure_s[10, 10] - 2.) / 3.
        self.assertAlmostEqual(reward["blank_revisit_penalty"], expected)
        self.assertLess(reward["blank_revisit_penalty"], 0.)
        self.assertEqual(advance(memory, blank_age_s=4.9)["blank_revisit_penalty"], 0.)
        self.assertEqual(advance(memory, (.6, 0.))["blank_revisit_penalty"], 0.)
        self.assertEqual(memory.no_new_age_s, 0.)
        self.assertEqual(advance(memory)["blank_revisit_penalty"], 0.)
        self.assertAlmostEqual(memory.no_new_age_s, .2)

    def test_terminal_steps_update_memory_but_give_no_new_reward(self):
        memory = EpisodicSpatialMemory(exploration_bonus=.02, blank_revisit_penalty=.006)
        for _ in range(30):
            advance(memory)
        budget = memory.budget_remaining
        self.assertEqual(advance(memory, terminal=True),
                         {"exploration_bonus": 0., "blank_revisit_penalty": 0.})
        self.assertFalse(memory.blank_revisit_active)
        self.assertEqual(advance(memory, (.6, 0.), terminal=True),
                         {"exploration_bonus": 0., "blank_revisit_penalty": 0.})
        self.assertEqual(memory.novel_cells, 2)
        self.assertEqual(memory.budget_remaining, budget)

    def test_mask_does_not_mutate_memory_and_values_stay_in_box(self):
        memory = EpisodicSpatialMemory(exploration_bonus=.02)
        advance(memory, (.6, 0.))
        self.assertTrue(np.all(memory.observation(masked=True) == 0.))
        full = memory.observation()
        self.assertTrue(np.all((full >= 0.) & (full <= 1.)))
        self.assertEqual(len(memory.field_names()), 802)
        self.assertEqual(memory.visited.sum(), 2)

    def test_invalid_parameters_rejected(self):
        for kwargs in ({"exploration_bonus": -1.}, {"exploration_budget": .51},
                       {"blank_revisit_penalty": float("nan")}):
            with self.assertRaises(ValueError):
                EpisodicSpatialMemory(**kwargs)


class SpatialMemoryEnvironmentTests(unittest.TestCase):
    def test_four_configs_have_only_planned_differences_and_safe_bounds(self):
        configs = [config(name) for name in "abcd"]
        common = {key: value for key, value in configs[0].items() if key not in (
            "spatial_memory_observation", "exploration_bonus", "blank_revisit_penalty", "blank_loop_penalty")}
        for name, cfg in zip("abcd", configs):
            self.assertEqual({key: value for key, value in cfg.items() if key in common}, common)
            with MobileWhiskerPuffEnv(cfg) as env:
                obs, info = env.reset(seed=11, options={"scenario": START})
                self.assertEqual(obs.shape, (859,))
                self.assertEqual(env.core_observation_dim, 57)
                self.assertTrue(env.observation_space.contains(obs))
                self.assertEqual(info["novel_cells"], 1)
                self.assertLessEqual(env.positive_auxiliary_reward_upper_bound, 12.5)
                self.assertLessEqual(env.minimum_oob_penalty, 60.)
                self.assertIn("spatial_memory", env.scenario_metadata())
                if name == "a":
                    self.assertTrue(np.all(obs[57:] == 0.))
                else:
                    self.assertEqual(obs[57:457].sum(), 1.)
                if name == "d":
                    self.assertAlmostEqual(env.positive_auxiliary_reward_upper_bound, 12.5)
                    self.assertAlmostEqual(env.minimum_oob_penalty, 55.7)

    def test_no_rng_consumption_or_old_core_reward_changes(self):
        old_cfg = yaml.safe_load((ROOT / "configs/mobile_rooms_hybrid_history_improved.yaml").read_text(encoding="utf-8"))
        old_cfg["plume_overrides"]["puff_warmup_steps"] = 0
        with MobileWhiskerPuffEnv(old_cfg) as old, MobileWhiskerPuffEnv(config("a")) as a, MobileWhiskerPuffEnv(config("b")) as b:
            obs0, _ = old.reset(seed=71)
            obsa, _ = a.reset(seed=71)
            obsb, _ = b.reset(seed=71)
            np.testing.assert_array_equal(obs0, obsa[:57])
            np.testing.assert_array_equal(obs0, obsb[:57])
            self.assertEqual(obs0.shape, (57,))
            self.assertFalse(any(key in old.effective_config for key in (
                "exploration_bonus", "exploration_budget", "spatial_memory_observation", "blank_revisit_penalty")))
            for step in range(35):
                action = [.4, .5 if step % 2 else -.5, step % 10, (9 - step) % 10]
                old_step, a_step, b_step = old.step(action), a.step(action), b.step(action)
                np.testing.assert_array_equal(old_step[0], a_step[0][:57])
                np.testing.assert_array_equal(old_step[0], b_step[0][:57])
                self.assertEqual(old_step[1:4], a_step[1:4])
                self.assertEqual(old_step[1:4], b_step[1:4])
            self.assertEqual(old.rng.bit_generator.state, a.rng.bit_generator.state)
            self.assertEqual(old.rng.bit_generator.state, b.rng.bit_generator.state)

    def test_post_movement_memory_and_terminal_truncation(self):
        with MobileWhiskerPuffEnv(config("d", max_steps=2)) as env:
            env.reset(seed=11, options={"scenario": START})
            obs, _, _, _, info = env.step([1., 0., 5, 5])
            self.assertTrue(info["new_cell"])
            self.assertEqual(info["novel_cells"], 2)
            self.assertEqual(obs[57 + 10 * 20 + 11], 1.)
            self.assertAlmostEqual(info["reward_components"]["exploration_bonus"], .0008)
            self.assertTrue(env.observation_space.contains(obs))
            _, _, _, truncated, info = env.step([1., 0., 5, 5])
            self.assertTrue(truncated)
            self.assertEqual(info["reward_components"]["exploration_bonus"], 0.)
            self.assertEqual(info["reward_components"]["blank_revisit_penalty"], 0.)
            self.assertNotIn("visited", env.trajectory[-1])
            self.assertIn("novel_cells", env.trajectory[-1])
            reset_obs, reset_info = env.reset(seed=11, options={"scenario": START})
            self.assertEqual(reset_info["novel_cells"], 1)
            self.assertEqual(reset_obs[57:457].sum(), 1.)

    def test_reward_stacking_shape_and_budget_guards(self):
        for overrides in ({"blank_loop_penalty": .006}, {"lidar_beams": 35},
                          {"include_blank_age_observation": False}, {"oob_penalty": 55.6},
                          {"world_bounds": [-4., 4.]}):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                MobileWhiskerPuffEnv(config("d", **overrides))

    def test_success_and_out_of_bounds_steps_get_no_new_rewards(self):
        with MobileWhiskerPuffEnv(config("d")) as env:
            env.reset(seed=11, options={"scenario": START})
            # 只替代终止判定，空间模块仍接收真实位移与传感器命中。
            env._distance_to_source = lambda: 0.0
            env.room_navigation.map.visible = lambda *_: True
            _, _, terminated, _, info = env.step([1., 0., 5, 5])
            self.assertTrue(terminated)
            self.assertTrue(info["new_cell"])
            self.assertEqual(info["reward_components"]["exploration_bonus"], 0.)
            self.assertEqual(info["reward_components"]["blank_revisit_penalty"], 0.)
        with MobileWhiskerPuffEnv(config("d")) as env:
            env.reset(seed=11, options={"scenario": START})
            # 缩小判定边界使下一平移越界；不改变已固定的空间网格。
            env.world_max = .5
            _, _, terminated, _, info = env.step([1., 0., 5, 5])
            self.assertTrue(terminated)
            self.assertTrue(info["out_of_bounds"])
            self.assertEqual(info["reward_components"]["exploration_bonus"], 0.)
            self.assertEqual(info["reward_components"]["blank_revisit_penalty"], 0.)

    def test_d_retains_old_loop_diagnostic_without_stacking_cost(self):
        with MobileWhiskerPuffEnv(config("d")) as env:
            env.reset(seed=11, options={"scenario": {**START, "robot_pose": [.25, .25, 0.]}})
            for _ in range(25):
                _, _, _, _, info = env.step([1., 1., 5, 5])
            self.assertTrue(info["blank_loop"])
            self.assertNotIn("blank_loop_penalty", info["reward_components"])
            self.assertLess(info["reward_components"]["blank_revisit_penalty"], 0.)


if __name__ == "__main__":
    unittest.main()
