"""四房间场景的几何、物理、观测和模型接口回归。"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import yaml

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.room_map import RoomMap
from dual_whisker_rl.envs.room_plume import RoomPuffPlume
from dual_whisker_rl.envs.room_navigation import RoomNavigation
from dual_whisker_rl.mobile_contract import environment_contract, validate_model_environment
from dual_whisker_rl.mobile_contract import navigation_episode_metrics, navigation_summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
# 原四房间验收保留为兼容性回归；六房间默认另有完整几何与接口检查。
CONFIG = yaml.safe_load((ROOT/"configs/mobile_rooms_four.yaml").read_text(encoding="utf-8"))


def make_env(**overrides):
    cfg = copy.deepcopy(CONFIG)
    cfg["plume_overrides"]["puff_warmup_steps"] = 0
    cfg.update(overrides)
    return MobileWhiskerPuffEnv(cfg)


def puff(x, y, sigma=.25):
    return dict(x=x, y=y, mass=.02, sigma_downwind=sigma, sigma_crosswind=sigma,
                cos_direction=1., sin_direction=0., direction=0., age=0., vd=0., vc=0.)


class RoomGeometryTests(unittest.TestCase):
    def setUp(self):
        self.map = RoomMap()

    def test_doors_and_swept_collision(self):
        self.assertFalse(self.map.visible((-4, 1), (4, 1), .15))
        self.assertTrue(self.map.visible((-4, 2.5), (4, 2.5), .15))
        self.assertFalse(self.map.visible((4.8, 4), (5.2, 4), .15))
        self.assertFalse(self.map.visible((-.3, 1.9), (.3, 2.1), .15))

    def test_lidar_distances_and_heading(self):
        ranges = self.map.raycast((-1, 1), 0.)
        self.assertAlmostEqual(float(ranges[0]), .95, places=6)
        self.assertAlmostEqual(float(ranges[27]), .95, places=6)
        np.testing.assert_allclose(self.map.raycast((-1, 1), np.pi/2), np.roll(ranges, -9), atol=1e-6)
        self.assertAlmostEqual(float(self.map.raycast((-1, 2.5), 0.)[0]), 3.)

    def test_reflection_and_large_steps(self):
        reflected = self.map.reflect((-1, 1), (1, 1))
        self.assertLess(reflected[0], -.05)
        self.assertTrue(self.map.valid(reflected))
        np.testing.assert_allclose(self.map.reflect((-1, 2.5), (1, 2.5)), (1, 2.5))
        starts = np.tile([-1., 1.], (100, 1))
        ends = starts+np.random.default_rng(4).uniform(-30, 30, (100, 2))
        self.assertTrue(self.map.valid(self.map.reflect(starts, ends)).all())

    def test_path_uses_door_and_reward_telescopes(self):
        da = self.map.path_distance((-1., 1.), (1., 1.), .15)
        db = self.map.path_distance((-1., 1.2), (1., 1.), .15)
        self.assertGreater(da, 2.)
        self.assertLess(db, da)
        potential = RoomNavigation.potential
        self.assertGreater(potential(db)-potential(da), 0.)
        self.assertAlmostEqual((potential(db)-potential(da))+(potential(da)-potential(db)), 0.)


class RoomEnvironmentTests(unittest.TestCase):
    def test_observation_seed_and_room_coverage(self):
        env = make_env()
        self.addCleanup(env.close)
        obs, info = env.reset(seed=3)
        again, repeated = env.reset(seed=3)
        np.testing.assert_array_equal(obs, again)
        self.assertEqual(info["source"], repeated["source"])
        self.assertEqual(obs.shape, (56,))
        self.assertTrue(env.observation_space.contains(obs))
        state = env.robot_state
        np.testing.assert_allclose(obs[16:18], [state.x/5., state.y/5.])
        relative = env.plume.wind_direction-state.heading
        np.testing.assert_allclose(obs[18:20], [math.cos(relative), math.sin(relative)], atol=1e-7)
        np.testing.assert_allclose(obs[20:], np.array(info["lidar_ranges_m"])/3.)
        rooms = set()
        for seed in range(48):
            _, info = env.reset(seed=seed)
            rooms.add(info["source_room"])
            self.assertTrue(env.room_navigation.map.valid(info["source"], .25))
            self.assertGreaterEqual(info["path_distance_to_source"], 1.)
        self.assertEqual(len(rooms), 4)

    def test_blocked_move_turning_and_penalty(self):
        env = make_env()
        self.addCleanup(env.close)
        env.reset(seed=1)
        env.robot_state = env.robot.reset(-.205, 1., 0.)
        env.prev_distance = env._distance_to_source()
        env._previous_path_distance = env.room_navigation.distance((-.205, 1.))
        _, reward, done, truncated, info = env.step([0, 5, 5])
        self.assertFalse(done or truncated)
        self.assertTrue(info["collision"])
        self.assertFalse(info["body_moved"])
        self.assertAlmostEqual(env.robot_state.x, -.205)
        self.assertEqual(info["reward_components"]["collision_penalty"], -.5)
        self.assertAlmostEqual(sum(info["reward_components"].values()), reward)
        _, _, _, _, info = env.step([1, 5, 5])
        self.assertTrue(info["collision"] and info["body_moved"])
        self.assertAlmostEqual(env.robot_state.heading, .3)
        self.assertEqual(info["collision_count"], 2)
        np.testing.assert_allclose(env.trajectory[-1]["lidar_ranges_m"], info["lidar_ranges_m"])

    def test_door_crossing_and_success(self):
        env = make_env(forward_speed=2.)
        self.addCleanup(env.close)
        env.reset(seed=1)
        env.robot_state = env.robot.reset(-.21, 2.5, 0.)
        env._previous_path_distance = env.room_navigation.distance((-.21, 2.5))
        _, _, _, _, info = env.step([0, 5, 5])
        self.assertFalse(info["collision"])
        self.assertGreater(env.robot_state.x, 0.)
        x, y = info["source"]
        env.robot_state = env.robot.reset(x+.05, y, 0.)
        env._previous_path_distance = env.room_navigation.distance((x+.05, y))
        _, _, done, _, info = env.step([5, 5, 5])
        self.assertTrue(done and info["is_success"])

    def test_whiskers_stop_before_wall(self):
        env = make_env(whisker_length=1.)
        self.addCleanup(env.close)
        env.reset(seed=2)
        env.robot_state = env.robot.reset(-.3, 1., 0.)
        ws = env.whiskers.state(env.robot_state)
        self.assertLess(ws.left_point[0], -.05)
        self.assertTrue(env.room_navigation.map.visible((-.3, 1.), ws.left_point))

    def test_invalid_explicit_scenarios(self):
        env = make_env()
        self.addCleanup(env.close)
        with self.assertRaisesRegex(ValueError, "气源"):
            env.reset(seed=1, options={"scenario": {"source_position": (0, 0)}})
        with self.assertRaisesRegex(ValueError, "位姿"):
            env.reset(seed=1, options={"scenario": {"source_position": (-2, -2), "robot_pose": (0, 1, 0)}})

    def test_legacy_spaces_unchanged(self):
        env = MobileWhiskerPuffEnv({"plume_overrides": {"puff_warmup_steps": 0}})
        self.addCleanup(env.close)
        self.assertEqual(env.reset(seed=2)[0].shape, (16,))
        self.assertIsNone(env.room_navigation)


class RoomGasTests(unittest.TestCase):
    def make_plume(self, room_map=None):
        return RoomPuffPlume(room_map or RoomMap(), source_position=(-1., 1.),
            world_bounds=(-5, 5), plume_overrides={"puff_warmup_steps": 0})

    def test_gaussian_cannot_leak_across_wall(self):
        p = self.make_plume()
        p.puffs = [puff(-.3, 1., 1.)]
        self.assertAlmostEqual(p.concentration(.3, 1., False), p.gas_background)
        self.assertGreater(p.concentration(-.4, 1., False), p.gas_background)
        p.puffs = [puff(-.3, 2.5, 1.)]
        self.assertGreater(p.concentration(.3, 2.5, False), p.gas_background)
        self.assertEqual(p.concentration(0., 1., False), 0.)

    def test_centers_pass_doors_and_reflect(self):
        p = self.make_plume()
        p.dt, p.wind_speed, p.wind_direction = 1., .4, 0.
        p.turbulence_vel_std_crosswind = p.turbulence_vel_std_downwind = 0.
        p.puffs = [puff(-.2, 1.), puff(-.2, 2.5)]
        p._advance_puffs()
        self.assertLess(p.puffs[0]["x"], -.05)
        self.assertGreater(p.puffs[1]["x"], 0.)

    def test_closed_partition(self):
        p = self.make_plume(RoomMap(walls=[(-.05, .05, -5., 5.)]))
        p.puffs = [puff(-.3, 2.5, 3.)]
        self.assertEqual(p.concentration(.3, 2.5, False), p.gas_background)

    def test_point_pair_grid_and_random_stream_consistency(self):
        p = self.make_plume()
        p.puffs = [puff(-.3, 1., .5), puff(.5, 2.5, .3)]
        before = copy.deepcopy(p.rng.bit_generator.state)
        noise_before = copy.deepcopy(p.noise_rng.bit_generator.state)
        xs, ys, grid = p.grid(41)
        for iy, ix in [(24, 18), (24, 22), (30, 20), (20, 20)]:
            point = (float(xs[ix]), float(ys[iy]))
            c = p.concentration(*point, add_noise=False)
            self.assertAlmostEqual(c, float(grid[iy, ix]), places=6)
            self.assertEqual(p.concentration_pair(point, point, False), (c, c))
        self.assertEqual(before, p.rng.bit_generator.state)
        self.assertEqual(noise_before, p.noise_rng.bit_generator.state)


class RoomInterfaceTests(unittest.TestCase):
    def test_contract_save_load_and_semantics(self):
        from stable_baselines3 import PPO
        from train_whisker_only_ppo import ObservationHistoryWrapper
        env = make_env()
        wrapped = ObservationHistoryWrapper(env, history_length=2)
        self.addCleanup(wrapped.close)
        model = PPO("MlpPolicy", wrapped, n_steps=8, batch_size=8, seed=3)
        with self.assertRaisesRegex(ValueError, "契约"):
            validate_model_environment(model, env)
        model.mobile_environment_contract = environment_contract(env, 2)
        self.assertEqual(validate_model_environment(model, env), 2)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(Path(tmp)/"model")
            loaded = PPO.load(Path(tmp)/"model")
            self.assertEqual(validate_model_environment(loaded, env), 2)
            loaded.mobile_environment_contract["field_names"][16] = "wrong_field"
            with self.assertRaisesRegex(ValueError, "契约"):
                validate_model_environment(loaded, env)

    def test_legacy_model_rejected_even_when_dimensions_are_divisible(self):
        from stable_baselines3 import PPO
        from train_whisker_only_ppo import ObservationHistoryWrapper
        legacy = MobileWhiskerPuffEnv({"plume_overrides": {"puff_warmup_steps": 0}})
        wrapped = ObservationHistoryWrapper(legacy, history_length=7)
        self.addCleanup(wrapped.close)
        model = PPO("MlpPolicy", wrapped, n_steps=8, batch_size=8)
        env = make_env()
        self.addCleanup(env.close)
        self.assertEqual(model.observation_space.shape[0], 56*2)
        with self.assertRaisesRegex(ValueError, "契约"):
            validate_model_environment(model, env)

    def test_resume_requires_same_horizon_and_same_physical_config(self):
        from stable_baselines3 import PPO
        from train_whisker_only_ppo import ObservationHistoryWrapper
        env = make_env()
        wrapped = ObservationHistoryWrapper(env, history_length=2)
        self.addCleanup(wrapped.close)
        model = PPO("MlpPolicy", wrapped, n_steps=8, batch_size=8)
        model.mobile_environment_contract = environment_contract(env, 2)
        shorter = make_env(max_steps=1000)
        self.addCleanup(shorter.close)
        self.assertEqual(validate_model_environment(model, shorter), 2)
        with self.assertRaisesRegex(ValueError, "契约"):
            validate_model_environment(model, shorter, resume=True)
        changed_radar = make_env(lidar_range=4.)
        self.addCleanup(changed_radar.close)
        with self.assertRaisesRegex(ValueError, "契约"):
            validate_model_environment(model, changed_radar)

    def test_scenario_bank_and_map_hash(self):
        from evaluate_mobile_fixed_scenarios import prepare_scenario_bank, ROOM_SCENARIO_BANK_VERSION
        cfg = copy.deepcopy(CONFIG)
        cfg["plume_overrides"]["puff_warmup_steps"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bank.json"
            scenarios, bank = prepare_scenario_bank(path, cfg, count=2, base_seed=33, regenerate=False)
            self.assertEqual(bank["version"], ROOM_SCENARIO_BANK_VERSION)
            self.assertEqual(bank["map_sha256"], RoomMap().digest)
            self.assertEqual(len(scenarios), 2)
            bank["map_sha256"] = "wrong"
            path.write_text(json.dumps(bank), encoding="utf-8")
            with self.assertRaises(ValueError):
                prepare_scenario_bank(path, cfg, count=2, base_seed=33, regenerate=False)

    def test_sealed_scenario_replays_sampled_episode(self):
        env = make_env()
        self.addCleanup(env.close)
        env.plume.puff_warmup_steps = 4
        obs, info = env.reset(seed=7)
        scenario = dict(source_position=info["source"],
                        robot_pose=(env.robot_state.x, env.robot_state.y, env.robot_state.heading),
                        wind_direction=env.plume.wind_direction_base,
                        wind_speed=env.plume.wind_speed_mean, source_strength=1.)
        first = [env.step([0, 4, 6])[:4] for _ in range(8)]
        repeated, _ = env.reset(seed=7, options={"scenario": scenario})
        np.testing.assert_array_equal(obs, repeated)
        second = [env.step([0, 4, 6])[:4] for _ in range(8)]
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])

    def test_metrics_include_initial_movement_and_groups(self):
        env = make_env(max_steps=3)
        self.addCleanup(env.close)
        env.reset(seed=1)
        for _ in range(3):
            env.step([0, 5, 5])
        row = navigation_episode_metrics(env.trajectory)
        self.assertAlmostEqual(row["path_length"], .09)
        row["success"] = False
        summary = navigation_summary([row])
        self.assertIn("by_room_relation", summary)
        self.assertIsNone(summary["mean_path_efficiency"])


if __name__ == "__main__":
    unittest.main()
