"""六房间布局、并集导航、雷达隐藏与新旧地图契约的验收。"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.room_layouts import IRREGULAR_SIX_ROOMS, point_in_polygon
from dual_whisker_rl.envs.room_map import RoomMap
from dual_whisker_rl.envs.room_plume import RoomPuffPlume
from dual_whisker_rl.mobile_contract import environment_contract, validate_model_environment, navigation_episode_metrics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"scripts"))
CONFIG = yaml.safe_load((ROOT/"configs/mobile_rooms.yaml").read_text(encoding="utf-8"))
FOUR_CONFIG = yaml.safe_load((ROOT/"configs/mobile_rooms_four.yaml").read_text(encoding="utf-8"))
CENTERS = [(-3., -3.), (-3., .2), (-3., 3.), (2., -3.), (2., -1.), (3., 3.), (0., 0.)]


def make_env(config=CONFIG):
    cfg = copy.deepcopy(config)
    cfg["plume_overrides"]["puff_warmup_steps"] = 0
    return MobileWhiskerPuffEnv(cfg)


def make_puff(x, y):
    return dict(x=x, y=y, mass=.02, sigma_downwind=.4, sigma_crosswind=.4,
                cos_direction=1., sin_direction=0., direction=0., age=0., vd=0., vc=0.)


class IrregularGeometryTests(unittest.TestCase):
    def setUp(self):
        self.map = RoomMap(layout=IRREGULAR_SIX_ROOMS)

    def test_regions_cover_arena_without_overlap(self):
        area = 0.
        for polygon in self.map.regions.values():
            a = np.asarray(polygon)
            b = np.roll(a, -1, axis=0)
            area += abs(np.sum(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]))/2.
        self.assertAlmostEqual(area, 100.)
        # 避开恰落在区域共边的采样坐标，独立检查覆盖与重叠。
        for x in np.linspace(-4.99, 4.99, 61):
            for y in np.linspace(-4.99, 4.99, 61):
                count = sum(point_in_polygon((x, y), p) for p in self.map.regions.values())
                self.assertEqual(count, 1)
        self.assertEqual(self.map.metadata()["room_count"], 6)

    def test_eight_doors_and_adjacent_regions(self):
        self.assertEqual(len(self.map.doors), 8)
        for door in self.map.doors:
            c = np.asarray(door["center"])
            normal = np.array([1., 0.]) if door["axis"] == "vertical" else np.array([0., 1.])
            a, b = c-.4*normal, c+.4*normal
            self.assertTrue(self.map.visible(a, b, .15), door["id"])
            self.assertEqual({self.map.room_id(a), self.map.room_id(b)}, set(door["regions"]))
            tangent = normal[::-1]
            for sign in (-1, 1):
                outside = c+sign*.6*tangent
                self.assertFalse(self.map.visible(outside-.4*normal, outside+.4*normal, .15))

    def test_stable_room_ids_corridor_and_l_shaped_room(self):
        self.assertEqual([self.map.room_id(p) for p in CENTERS],
                         ["R1", "R2", "R3", "R4", "R5", "R6", "corridor"])
        self.assertEqual(self.map.room_id((.8, .9)), "R6")
        self.assertEqual(self.map.room_id((.8, 2.)), "R3")
        self.assertEqual(self.map.room_id((-1., -1.4)), "corridor")
        self.assertIsNone(self.map.room_id((-.6, 1.1)))
        self.assertIsNone(self.map.room_id((6., 0.)))
        self.assertEqual(self.map.room_id((.6, -1.)), "corridor")

    def test_navigation_all_regions_and_t_l_union_corners(self):
        vertices, _ = self.map.navigation_graph(.15)
        for corner in ((.800001, .099999), (.799999, 1.700001)):
            self.assertLess(float(np.linalg.norm(vertices-corner, axis=1).min()), 2e-6)
        for a in CENTERS:
            for b in CENTERS:
                distance = self.map.path_distance(a, b, .15)
                self.assertTrue(np.isfinite(distance))
                self.assertGreaterEqual(distance+1e-8, float(np.linalg.norm(np.asarray(a)-b)))
                self.assertAlmostEqual(distance, self.map.path_distance(b, a, .15), places=6)
        # R5/R6虽在同一象限，隔墙仍要求先绕门。
        self.assertGreater(self.map.path_distance((2., -.2), (2., .8), .15), 1.5)

    def test_swept_collision_and_corner_reflection(self):
        self.assertFalse(self.map.visible((2., -.2), (2., .8), .15))
        self.assertFalse(self.map.visible((.3, .0), (.9, .6), .15))
        self.assertTrue(self.map.visible((-.9, -1.4), (.1, -1.4), .15))
        rng = np.random.default_rng(32)
        starts = np.array([self.map.sample_free(rng, .01) for _ in range(200)])
        ends = starts+rng.uniform(-12, 12, starts.shape)
        self.assertTrue(self.map.valid(self.map.reflect(starts, ends)).all())

    def test_four_room_metadata_and_graph_are_preserved(self):
        old = RoomMap()
        self.assertEqual(old.digest, "a9a9ab9dcb55d7ee4dbfcc85ab976135e0a13436ba7283638ca6d60c297c7ba4")
        self.assertEqual(len(old.navigation_graph(.15)[0]), 16)
        self.assertEqual(old.room_id((-2., 1.)), "north_west")
        self.assertNotEqual(old.digest, self.map.digest)


class IrregularEnvironmentTests(unittest.TestCase):
    def test_default_observation_and_all_regions_sampled(self):
        env = make_env()
        self.addCleanup(env.close)
        self.assertEqual(env.room_navigation.map.layout, IRREGULAR_SIX_ROOMS)
        sources, starts = set(), set()
        for seed in range(256):
            obs, info = env.reset(seed=seed)
            self.assertEqual(obs.shape, (56,))
            self.assertTrue(env.observation_space.contains(obs))
            np.testing.assert_allclose(obs[20:], np.array(info["lidar_ranges_m"])/3.)
            self.assertGreaterEqual(info["path_distance_to_source"], 1.)
            self.assertTrue(env.plume.is_position_valid(*info["source"], margin=.25))
            sources.add(info["source_room"])
            starts.add(info["start_room"])
        expected = {"R1", "R2", "R3", "R4", "R5", "R6", "corridor"}
        self.assertEqual(sources, expected)
        self.assertEqual(starts, expected)

    def test_room_statistics_use_real_regions_not_quadrants(self):
        env = make_env()
        self.addCleanup(env.close)
        env.reset(seed=7, options={"scenario": {"source_position": (-2., 3.), "robot_pose": (-2., .2, 0.)}})
        env.step([5, 5, 5])
        row = navigation_episode_metrics(env.trajectory)
        self.assertEqual(row["source_room"], "R3")
        self.assertEqual(row["start_room"], "R2")
        self.assertEqual(row["room_relation"], "cross_room")

    def test_sealed_scene_replays_with_warmup(self):
        env = make_env()
        self.addCleanup(env.close)
        env.plume.puff_warmup_steps = 4
        obs, info = env.reset(seed=7)
        s = env.robot_state
        scene = dict(source_position=info["source"], robot_pose=(s.x, s.y, s.heading),
                     wind_direction=env.plume.wind_direction_base, wind_speed=env.plume.wind_speed_mean)
        first = [env.step([0, 4, 6])[:4] for _ in range(8)]
        repeated, _ = env.reset(seed=7, options={"scenario": scene})
        np.testing.assert_array_equal(obs, repeated)
        for a in first:
            b = env.step([0, 4, 6])[:4]
            np.testing.assert_array_equal(a[0], b[0])
            self.assertEqual(a[1:], b[1:])

    def test_model_contract_rejects_four_room_weights(self):
        from stable_baselines3 import PPO
        from train_whisker_only_ppo import ObservationHistoryWrapper
        old = make_env(FOUR_CONFIG)
        wrapped = ObservationHistoryWrapper(old, history_length=2)
        self.addCleanup(wrapped.close)
        model = PPO("MlpPolicy", wrapped, n_steps=8, batch_size=8)
        model.mobile_environment_contract = environment_contract(old, 2)
        new = make_env()
        self.addCleanup(new.close)
        self.assertEqual(old.observation_space, new.observation_space)
        with self.assertRaisesRegex(ValueError, "契约"):
            validate_model_environment(model, new)

    def test_scenario_bank_rejects_four_room_map(self):
        from evaluate_mobile_fixed_scenarios import prepare_scenario_bank
        old, new = copy.deepcopy(FOUR_CONFIG), copy.deepcopy(CONFIG)
        for cfg in (old, new):
            cfg["plume_overrides"]["puff_warmup_steps"] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bank.json"
            prepare_scenario_bank(path, old, count=2, base_seed=9, regenerate=False)
            with self.assertRaises(ValueError):
                prepare_scenario_bank(path, new, count=2, base_seed=9, regenerate=False)


class IrregularGasAndPlotTests(unittest.TestCase):
    def test_door_transport_and_solid_wall_occlusion(self):
        m = RoomMap(layout=IRREGULAR_SIX_ROOMS)
        p = RoomPuffPlume(m, source_position=(-2.7, -1.4), world_bounds=(-5, 5),
                          plume_overrides={"puff_warmup_steps": 0})
        p.puffs = [make_puff(-1.6, -1.4), make_puff(-1.6, -2.5)]
        p.dt, p.wind_speed, p.wind_direction = 1., .4, 0.
        p.turbulence_vel_std_downwind = p.turbulence_vel_std_crosswind = 0.
        p._advance_puffs()
        self.assertGreater(p.puffs[0]["x"], -1.4)
        self.assertLess(p.puffs[1]["x"], -1.45)
        p.puffs = [make_puff(-1.6, -2.5)]
        self.assertEqual(p.concentration(-1.2, -2.5, False), p.gas_background)
        p.puffs = [make_puff(-1.6, -1.4)]
        self.assertGreater(p.concentration(-1.2, -1.4, False), p.gas_background)
        xs, ys, field = p.grid(41)
        self.assertAlmostEqual(float(field[15, 14]), p.concentration(float(xs[14]), float(ys[15]), False), places=6)

    def test_capture_preserves_radar_observations_and_random_state(self):
        from visualize_mobile_whisker_env import capture_rollout
        a, b = make_env(), make_env()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        data = capture_rollout(a, None, 1, 8, 12, 7, True)
        b.reset(seed=7)
        rng = np.random.default_rng(130)
        for _ in range(8):
            b.step([int(rng.integers(6)), int(rng.integers(10)), int(rng.integers(10))])
        np.testing.assert_array_equal(a._last_obs, b._last_obs)
        self.assertEqual(a.trajectory, b.trajectory)
        self.assertEqual(a.plume.rng.bit_generator.state, b.plume.rng.bit_generator.state)
        self.assertEqual(a.plume.noise_rng.bit_generator.state, b.plume.noise_rng.bit_generator.state)
        self.assertEqual(len(data["lidar_frames"][0]), 36)

    def test_static_and_animation_layers_have_no_radar_rays(self):
        import visualize_mobile_whisker_env as viz
        from matplotlib.collections import LineCollection
        env = make_env()
        self.addCleanup(env.close)
        data = viz.capture_rollout(env, None, 1, 3, 12, 7, True)
        def check_axes(fig):
            self.assertFalse(any(isinstance(c, LineCollection) for ax in fig.axes
                                 if ax.get_label() != "<colorbar>" for c in ax.collections))
        fig, ax = viz.plt.subplots()
        viz._setup_axes(ax, data, viz.build_plume_colormap(), 3, "six rooms")
        check_axes(fig)
        viz.plt.close(fig)
        def inspect_animation(anim, path, **kwargs):
            for frame in (0, 2):
                anim._func(frame)
            check_axes(anim._fig)
            anim._draw_was_started = True
        with tempfile.TemporaryDirectory() as tmp, patch.object(viz, "save_matplotlib_animation", side_effect=inspect_animation):
            viz.render_animation(data, Path(tmp)/"one.mp4", "six rooms", 1, 6)
            viz.render_animation_rollouts([("A", data), ("B", data)], Path(tmp)/"two.mp4", "six rooms", 1, 6)


if __name__ == "__main__":
    unittest.main()
