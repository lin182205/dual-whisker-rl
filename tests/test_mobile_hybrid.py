"""连续底盘/离散触须的概率、几何、观测、奖励与保存回归。"""

import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.envs.hybrid_action import HybridMotionWhiskerSpace, decode_hybrid_action, arc_pose
from dual_whisker_rl.envs.robot_model import RobotState
from dual_whisker_rl.envs.room_map import RoomMap
from dual_whisker_rl.agents.hybrid_policy import HybridActionDistribution, HybridActorCriticPolicy
from dual_whisker_rl.agents.olfactory_navigation import OlfactoryNavigationExtractor, NAVIGATION_INDICES
from dual_whisker_rl.mobile_contract import environment_contract, validate_model_environment
from dual_whisker_rl.mobile_contract import navigation_episode_metrics, navigation_summary

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from train_whisker_only_ppo import ObservationHistoryWrapper

CONFIG = yaml.safe_load((ROOT/'configs/mobile_rooms_hybrid_six.yaml').read_text(encoding='utf-8'))
torch.set_num_threads(1)


def make_env(**overrides):
    cfg = copy.deepcopy(CONFIG)
    cfg['plume_overrides']['puff_warmup_steps'] = 0
    cfg.update(overrides)
    return MobileWhiskerPuffEnv(cfg)


def set_pose(env, x, y, heading):
    env.robot_state = env.robot.reset(x, y, heading)
    env.prev_distance = env._distance_to_source()
    env._previous_path_distance = env.room_navigation.distance((x, y))


def make_model(env, history=2, encoder='gru'):
    wrapped = ObservationHistoryWrapper(env, history_length=history)
    model = PPO(HybridActorCriticPolicy, wrapped, n_steps=8, batch_size=8, n_epochs=1, seed=3,
                policy_kwargs={'net_arch': [160, 160], 'features_extractor_class': OlfactoryNavigationExtractor,
                               'features_extractor_kwargs': {'history_length': history, 'temporal_encoder': encoder}})
    model.mobile_environment_contract = environment_contract(env, history)
    model.mobile_environment_contract['encoder_config'] = model.policy.features_extractor.configuration
    return model, wrapped


class HybridProbabilityTests(unittest.TestCase):
    def test_action_sampling_and_validation(self):
        a, b = HybridMotionWhiskerSpace(seed=7), HybridMotionWhiskerSpace(seed=7)
        for _ in range(100):
            x = a.sample()
            self.assertTrue(a.contains(x))
            np.testing.assert_array_equal(x, b.sample())
        for invalid in ([0, 0, .5, 3], [0, 0, 10, 3], [0, np.nan, 1, 2], [0, 0, 1]):
            self.assertFalse(a.contains(invalid))
            with self.assertRaises(ValueError):
                decode_hybrid_action(invalid)
        self.assertEqual(decode_hybrid_action([3, -4, 2, 9]), (1., -1., 2, 9))

    def test_joint_log_prob_entropy_and_three_branch_gradients(self):
        parameters = torch.randn(6, 22, requires_grad=True)
        log_std = torch.tensor([-.4, -.2], requires_grad=True)
        d = HybridActionDistribution().proba_distribution(parameters, log_std)
        actions = d.sample().detach()
        expected = (d.motion.log_prob(actions[:, :2]).sum(-1)+d.left.log_prob(actions[:, 2])
                    + d.right.log_prob(actions[:, 3]))
        torch.testing.assert_close(d.log_prob(actions), expected)
        torch.testing.assert_close(d.entropy(), d.motion.entropy().sum(-1)+d.left.entropy()+d.right.entropy())
        (-d.log_prob(actions).mean()).backward()
        for branch in (slice(0, 2), slice(2, 12), slice(12, 22)):
            self.assertGreater(parameters.grad[:, branch].abs().sum().item(), 0.)
        self.assertGreater(log_std.grad.abs().sum().item(), 0.)
        with self.assertRaises(ValueError):
            d.log_prob(actions+torch.tensor([0., 0., .25, 0.]))

    def test_ppo_keeps_raw_motion_samples_and_exact_categories(self):
        env = make_env()
        self.addCleanup(env.close)
        model, wrapped = make_model(env)
        with torch.no_grad():
            model.policy.action_net.bias[:2] = torch.tensor([4., -4.])
            model.policy.log_std[:] = -3.
        model.learn(8)
        actions = model.rollout_buffer.actions.reshape(-1, 4)
        self.assertTrue((actions[:, 0] > 1.).all())
        self.assertTrue((actions[:, 1] < -1.).all())
        np.testing.assert_array_equal(actions[:, 2:], actions[:, 2:].astype(int))
        self.assertLessEqual(env.hybrid_motion.linear, .15)
        self.assertGreaterEqual(env.hybrid_motion.angular, -1.5)
        self.assertTrue(wrapped.action_space.contains(model.predict(wrapped.reset(seed=4)[0])[0]))


class HybridGeometryTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.env.reset(seed=7)
        self.addCleanup(self.env.close)

    def test_straight_arc_zero_limit_and_spin(self):
        p = RobotState(-3., -3., .3)
        straight = arc_pose(p, .15, 0., .2)
        tiny = arc_pose(p, .15, 1e-13, .2)
        np.testing.assert_allclose([straight.x, straight.y], [tiny.x, tiny.y], atol=1e-13)
        arc = arc_pose(p, .15, 1.5, .2)
        self.assertAlmostEqual(arc.x, p.x+.1*(math.sin(.6)-math.sin(.3)))
        self.assertAlmostEqual(arc.y, p.y-.1*(math.cos(.6)-math.cos(.3)))
        spin = arc_pose(p, 0., 1.5, .2)
        self.assertEqual((spin.x, spin.y), (p.x, p.y))
        self.assertAlmostEqual(spin.heading, .6)

    def test_unblocked_arc_length_and_continuous_observation(self):
        set_pose(self.env, -3., -3., .2)
        before = copy.copy(self.env.robot_state)
        expected = arc_pose(before, .09375, .375, .2)
        obs, _, _, _, info = self.env.step([.25, .25, 3, 7])
        self.assertEqual(self.env.robot_state, expected)
        self.assertAlmostEqual(info['path_length'], .01875)
        self.assertGreater(info['path_length'], math.dist((before.x, before.y), (expected.x, expected.y)))
        np.testing.assert_allclose(obs[14:16], [.625, .25])
        self.assertEqual(info['move_action'], 'turn_left')

    def test_collision_cancels_whole_translation_but_retains_turn(self):
        set_pose(self.env, -1.61, -2.6, 0.)
        _, _, terminated, _, info = self.env.step([1., .1, 3, 6])
        self.assertTrue(info['collision'])
        self.assertFalse(terminated)
        self.assertEqual((self.env.robot_state.x, self.env.robot_state.y), (-1.61, -2.6))
        self.assertAlmostEqual(self.env.robot_state.heading, .03)
        self.assertEqual(info['path_length'], 0.)
        self.assertEqual(info['actual_linear_velocity'], 0.)
        self.assertAlmostEqual(info['requested_linear_velocity'], .15)
        self.assertEqual(info['reward_components']['collision_penalty'], -.5)
        self.assertEqual(info['move_action'], 'spin_left')
        _, _, _, _, second = self.env.step([-1., -1., 2, 9])
        self.assertEqual(second['collision_count'], 1)
        self.assertEqual(second['move_action'], 'spin_right')

    def test_large_step_doors_walls_and_corners(self):
        nav = self.env.room_navigation
        for y, blocked in [(-1.4, False), (-2.6, True), (-1.92, True)]:
            before = RobotState(-2., y, 0.)
            result = nav.move_arc(before, .15, 0., 8.)
            self.assertEqual(nav.collision, blocked)
            self.assertEqual(result.x, before.x if blocked else -.8)
        # 外墙角和满圈圆弧：端点/整步弦线安全仍不能漏掉中途碰墙。
        before = RobotState(4.79, 4.79, math.pi/4)
        nav.move_arc(before, .15, .01, 4.)
        self.assertTrue(nav.collision)
        before = RobotState(4.7, 3., -math.pi/2)
        result = nav.move_arc(before, .15, 1., 2*math.pi)
        self.assertTrue(nav.collision)
        self.assertEqual((result.x, result.y), (before.x, before.y))

    def test_arc_sagitta_margin_detects_wall_between_chord_and_arc(self):
        nav = self.env.room_navigation
        radius = nav.radius
        # 很短圆弧向+x弯，弦线距右墙0.1500005，圆弧中途越过安全边界。
        angle, r = .05, .1
        start_x = 5.-radius-0.0000005-r*math.cos(angle/2)
        before = RobotState(start_x+r*math.cos(-angle/2), 3.+r*math.sin(-angle/2), math.pi/2-angle/2)
        final = arc_pose(before, .15, 1.5, angle/1.5)
        self.assertTrue(nav.map.visible((before.x, before.y), (final.x, final.y), radius))
        nav.move_arc(before, .15, 1.5, angle/1.5)
        self.assertTrue(nav.collision)


class HybridObservationRewardTests(unittest.TestCase):
    def setUp(self):
        self.env = make_env()
        self.obs, _ = self.env.reset(seed=7)
        self.addCleanup(self.env.close)

    def test_fields_normalization_wind_and_history(self):
        self.assertEqual(self.obs.shape, (57,))
        self.assertEqual(len(self.env.observation_field_names), 57)
        self.assertEqual(self.env.observation_field_names[14:17], ['actual_linear_velocity_norm', 'actual_angular_velocity_norm', 'blank_age_norm'])
        self.assertTrue(self.env.observation_space.contains(self.obs))
        angle = self.env.plume.wind_direction-self.env.robot_state.heading
        np.testing.assert_allclose(self.obs[19:21], [math.cos(angle), math.sin(angle)], atol=1e-7)
        wrapped = ObservationHistoryWrapper(self.env, 20)
        stacked, _ = wrapped.reset(seed=7)
        self.assertEqual(stacked.shape, (1140,))
        np.testing.assert_array_equal(stacked.reshape(20, 57), np.tile(self.obs, (20, 1)))

    def test_radar_minimum_and_wrap(self):
        nav = self.env.room_navigation
        set_pose(self.env, -3., -3., 0.)
        nav._scan_key = None
        raw = np.full(360, 3., dtype=np.float32)
        raw[359], raw[4], raw[5], raw[354], raw[355] = .7, .8, .9, 1.1, .6
        with patch.object(nav.map, 'raycast', return_value=raw):
            scan = nav.scan(self.env)
        self.assertEqual(float(scan[0]), float(raw[355]))
        self.assertEqual(float(scan[1]), float(raw[5]))
        self.assertEqual(float(scan[35]), float(raw[354]))
        self.assertEqual(len(nav.raw_lidar_ranges), 360)

    def test_known_wall_and_rotated_scan(self):
        nav = self.env.room_navigation
        nav.map = RoomMap(walls=[(-.05, .05, -5., 5.)])
        set_pose(self.env, -1., 1., 0.)
        scan = nav.scan(self.env)
        self.assertAlmostEqual(float(scan[0]), .95, places=6)
        set_pose(self.env, -1., 1., math.pi/2)
        np.testing.assert_allclose(nav.scan(self.env), np.roll(scan, -9), atol=1e-6)

    def test_centered_door_has_no_near_cost(self):
        set_pose(self.env, -1.4, -1.4, 0.)
        components = self.env.hybrid_motion.reward_components(self.env)
        self.assertEqual(components['near_obstacle_penalty'], 0.)
        self.assertGreater(self.env.hybrid_motion.info(self.env)['surface_clearance_m'], .20)

    def test_cost_formulas_and_actual_control_changes(self):
        set_pose(self.env, -1.65, -2.6, 0.)
        _, _, _, _, info = self.env.step([-1., 1., 2, 7])
        expected = -.05*np.clip(1.-info['surface_clearance_m']/.2, 0., 1.)**2
        self.assertAlmostEqual(info['reward_components']['near_obstacle_penalty'], expected)
        self.assertAlmostEqual(info['reward_components']['control_smoothness_penalty'], -.0025)
        _, _, _, _, info = self.env.step([-1., -1., 2, 7])
        self.assertAlmostEqual(info['reward_components']['control_smoothness_penalty'], -.005)

    def test_full_circle_cannot_accumulate_distance_reward(self):
        set_pose(self.env, -3., -3., 0.)
        self.env.robot.turn_rate = 2*math.pi/(40*.2)
        self.env.hybrid_motion.w_max = self.env.robot.turn_rate
        rewards = [self.env.step([1., 1., 4, 6])[4]['reward_components']['distance_potential_progress'] for _ in range(40)]
        self.assertAlmostEqual(sum(rewards), 0., places=10)
        self.assertAlmostEqual(self.env.robot_state.x, -3., places=10)
        self.assertAlmostEqual(self.env.robot_state.y, -3., places=10)

    def test_source_coordinates_do_not_enter_observation(self):
        a, b = make_env(), make_env()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        a.reset(seed=9)
        b.reset(seed=9)
        b.plume.source_x, b.plume.source_y = 3., 3.
        b.room_navigation.source = (3., 3.)
        # 同一传感器状态、位姿和风；真实源只改变奖励/诊断，不应改变特征。
        np.testing.assert_array_equal(a._build_observation(), b._build_observation())
        self.assertFalse(any('source' in name for name in a.observation_field_names))


class HybridInterfaceTests(unittest.TestCase):
    def test_all_encoders_fuse_current_navigation_only(self):
        space = gym.spaces.Box(-1., 1., shape=(171,))
        for encoder in ('gru', 'mlp', 'transformer'):
            net = OlfactoryNavigationExtractor(space, history_length=3, temporal_encoder=encoder)
            net.eval()
            x = torch.randn(2, 3, 57)
            old_nav_changed = x.clone()
            old_nav_changed[:, :-1, list(NAVIGATION_INDICES)] += 10.
            result = net(x.flatten(1))
            self.assertEqual(result.shape, (2, 128))
            torch.testing.assert_close(result, net(old_nav_changed.flatten(1)))
            y = x.clone()
            y[:, -1, 21:] += 2.
            updated = net(y.flatten(1))
            torch.testing.assert_close(result[:, :64], updated[:, :64])
            self.assertGreater((result[:, 64:]-updated[:, 64:]).abs().sum().item(), 0.)
            if encoder == 'transformer':
                features, attention = net.forward_with_attention(x.flatten(1))
                torch.testing.assert_close(features, result)
                self.assertEqual(tuple(attention.shape), (2, 2, 4, 4))

    def test_save_load_resume_and_contract_rejections(self):
        env = make_env()
        self.addCleanup(env.close)
        model, wrapped = make_model(env)
        model.learn(8)
        self.assertEqual(validate_model_environment(model, env), 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'hybrid'
            model.save(path)
            loaded = PPO.load(path, env=wrapped)
            obs = wrapped.reset(seed=7)[0]
            np.testing.assert_array_equal(loaded.predict(obs, deterministic=True)[0], model.predict(obs, deterministic=True)[0])
            loaded.learn(8, reset_num_timesteps=False)
            self.assertEqual(loaded.num_timesteps, 16)
            loaded.mobile_environment_contract['radar']['pooling'] = 'average'
            with self.assertRaisesRegex(ValueError, '契约'):
                validate_model_environment(loaded, env)
        old = make_env(motion_mode='discrete')
        self.addCleanup(old.close)
        with self.assertRaises(ValueError):
            validate_model_environment(model, old)
        changed = make_env(near_obstacle_cost=.1)
        self.addCleanup(changed.close)
        with self.assertRaisesRegex(ValueError, '契约'):
            validate_model_environment(model, changed)

    def test_ordinary_gaussian_cannot_masquerade_as_hybrid_policy(self):
        env = make_env()
        self.addCleanup(env.close)
        wrapped = ObservationHistoryWrapper(env, 2)
        model = PPO('MlpPolicy', wrapped, n_steps=8, batch_size=8)
        model.mobile_environment_contract = environment_contract(env, 2)
        with self.assertRaisesRegex(ValueError, '策略'):
            validate_model_environment(model, env)

    def test_bank_replay_and_old_bank_rejection(self):
        from evaluate_mobile_fixed_scenarios import prepare_scenario_bank, HYBRID_SCENARIO_BANK_VERSION
        cfg = copy.deepcopy(CONFIG)
        cfg['plume_overrides']['puff_warmup_steps'] = 0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'bank.json'
            scenarios, bank = prepare_scenario_bank(path, cfg, count=1, base_seed=7, regenerate=False)
            self.assertEqual(bank['version'], HYBRID_SCENARIO_BANK_VERSION)
            old_cfg = copy.deepcopy(cfg)
            old_cfg.pop('motion_mode')
            with self.assertRaises(ValueError):
                prepare_scenario_bank(path, old_cfg, count=1, base_seed=7, regenerate=False)
            env = make_env()
            self.addCleanup(env.close)
            obs, _ = env.reset(seed=7)
            first = [env.step([.23, -.17, 4, 6])[:4] for _ in range(8)]
            repeated, _ = env.reset(seed=7, options={'scenario': scenarios[0].to_dict()})
            np.testing.assert_array_equal(obs, repeated)
            second = [env.step([.23, -.17, 4, 6])[:4] for _ in range(8)]
            for a, b in zip(first, second):
                np.testing.assert_array_equal(a[0], b[0])
                self.assertEqual(a[1:], b[1:])
            bank['environment_contract']['observation_version'] = 'wrong'
            path.write_text(json.dumps(bank), encoding='utf-8')
            with self.assertRaises(ValueError):
                prepare_scenario_bank(path, cfg, count=1, base_seed=7, regenerate=False)

    def test_plotting_and_evaluation_keep_continuous_actions(self):
        import visualize_mobile_whisker_env as viz
        from dual_whisker_rl.evaluation import _format_action, evaluate_policy
        from evaluate_mobile_whisker_ppo import write_reward_breakdown
        class ConstantPolicy:
            def predict(self, obs, deterministic=True):
                return np.array([.2345, -.1267, 3., 8.], dtype=np.float32), None
        a, b = make_env(max_steps=8), make_env(max_steps=8)
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        data = viz.capture_rollout(a, ConstantPolicy(), 1, 8, 12, 7, True)
        b.reset(seed=7)
        for _ in range(8):
            b.step(_format_action(ConstantPolicy().predict(None)[0]))
        self.assertEqual(a.trajectory, b.trajectory)
        self.assertEqual(a.plume.rng.bit_generator.state, b.plume.rng.bit_generator.state)
        self.assertEqual(a.plume.noise_rng.bit_generator.state, b.plume.noise_rng.bit_generator.state)
        self.assertEqual(data['actions'].shape, (8, 4))
        self.assertAlmostEqual(float(data['actions'][0, 0]), .2345, places=6)
        metrics, trajectories = evaluate_policy(ConstantPolicy(), lambda: make_env(max_steps=8), episodes=1, seed=7)
        self.assertAlmostEqual(metrics['mean_path_length'], a.room_navigation.path_length)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'reward.csv'
            write_reward_breakdown(trajectories, [('failure', 0)], base_seed=7, path=path)
            self.assertIn('control_smoothness_penalty', path.read_text(encoding='utf-8-sig'))

    def test_navigation_metrics_are_shared(self):
        env = make_env(max_steps=3)
        self.addCleanup(env.close)
        env.reset(seed=7)
        for _ in range(3):
            env.step([.2, .4, 4, 6])
        row = navigation_episode_metrics(env.trajectory)
        self.assertAlmostEqual(row['mean_linear_velocity'], .09)
        self.assertEqual(row['motion_state_counts']['turn_left'], 3)
        row['success'] = False
        summary = navigation_summary([row])
        self.assertIn('near_obstacle_step_ratio', summary)
        self.assertIsNone(summary['mean_path_efficiency'])

    def test_random_capture_preserves_radar_and_random_state(self):
        from visualize_mobile_whisker_env import capture_rollout
        a, b = make_env(), make_env()
        self.addCleanup(a.close)
        self.addCleanup(b.close)
        data = capture_rollout(a, None, 1, 8, 12, 7, True)
        b.reset(seed=7)
        b.action_space.seed(130)
        for _ in range(8):
            b.step(b.action_space.sample())
        np.testing.assert_array_equal(a._last_obs, b._last_obs)
        self.assertEqual(a.trajectory, b.trajectory)
        self.assertEqual(a.plume.rng.bit_generator.state, b.plume.rng.bit_generator.state)
        self.assertEqual(a.plume.noise_rng.bit_generator.state, b.plume.noise_rng.bit_generator.state)
        self.assertEqual(data['actions'].shape, (8, 4))


if __name__ == '__main__':
    unittest.main()
