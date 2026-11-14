"""风向与相对运动历史的坐标、消融及模型契约回归。"""

from pathlib import Path
import sys
import tempfile
import unittest

from gymnasium.spaces import Box
import numpy as np
from stable_baselines3 import PPO
import torch
import yaml

from dual_whisker_rl.agents.hybrid_policy import HybridActorCriticPolicy
from dual_whisker_rl.agents.olfactory_navigation import OlfactoryNavigationExtractor
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.mobile_contract import environment_contract, validate_model_environment
from dual_whisker_rl.mobile_features import WIND_MOTION_HISTORY

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.evaluate_mobile_fixed_scenarios import prepare_scenario_bank
from train_whisker_only_ppo import ObservationHistoryWrapper


CONFIG = ROOT / "configs/mobile_rooms_hybrid_history.yaml"


def frames(headings, wind, xy=None):
    headings, wind = np.deg2rad(headings), np.deg2rad(wind)
    result = np.zeros((len(headings), 57), dtype=np.float32)
    result[:, 12:14] = np.column_stack((np.cos(headings), np.sin(headings)))
    result[:, 19:21] = np.column_stack((np.cos(wind - headings), np.sin(wind - headings)))
    if xy is not None:
        result[:, 17:19] = np.asarray(xy) / 5.0
    return torch.from_numpy(result)


def extractor(length, encoder="gru", ablation="none"):
    return OlfactoryNavigationExtractor(
        Box(-np.inf, np.inf, shape=(57 * length,), dtype=np.float32),
        history_length=length, temporal_encoder=encoder,
        feature_layout=WIND_MOTION_HISTORY, history_ablation=ablation,
    )


def branches(net, data):
    temporal, navigation = net._branch_inputs(data.reshape(1, -1))
    return temporal.reshape(net.history_length, 19), navigation[0]


def config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["plume_overrides"]["puff_warmup_steps"] = 0
    return cfg


class WindMotionHistoryTests(unittest.TestCase):
    def test_fixed_wind_during_turn_and_angle_wrap(self):
        data = frames([179, -179, -175], [30, 30, 30])
        history, _ = branches(extractor(3), data)
        torch.testing.assert_close(history[:, 13:15], data[-1:, 19:21].expand(3, 2))
        angle = np.deg2rad([-6, -4, 0])
        expected = torch.tensor(np.column_stack((np.cos(angle), np.sin(angle))), dtype=torch.float32)
        torch.testing.assert_close(history[:, 17:19], expected, atol=1e-7, rtol=1e-5)

    def test_wind_changes_and_displacement_in_latest_body_frame(self):
        data = frames([90, 90], [90, 180], [[0., 0.], [1., 2.]])
        history, nav = branches(extractor(2), data)
        torch.testing.assert_close(history[:, 13:15], torch.tensor([[1., 0.], [0., 1.]]))
        torch.testing.assert_close(history[:, 15:], torch.tensor([[-2., 1., 1., 0.], [0., 0., 1., 0.]]))
        torch.testing.assert_close(nav[:6], data[-1, [12, 13, 14, 15, 17, 18]])

    def test_reset_and_single_frame_have_exact_identity_motion(self):
        env = ObservationHistoryWrapper(MobileWhiskerPuffEnv(config()), history_length=3)
        self.addCleanup(env.close)
        env.reset(seed=3)
        env.step(np.array([.2, .5, 2., 5.], dtype=np.float32))
        obs, _ = env.reset(seed=7)
        data = torch.from_numpy(obs).reshape(3, 57)
        temporal, nav = branches(extractor(3), data)
        single, single_nav = branches(extractor(1), data[-1:])
        self.assertTrue(torch.equal(temporal[:, 15:], torch.tensor([[0., 0., 1., 0.]]).expand(3, 4)))
        self.assertTrue(torch.equal(single[0], temporal[-1]))
        self.assertTrue(torch.equal(nav, single_nav))

    def test_each_ablation_keeps_other_branch_inputs(self):
        data = frames([40, 85, 90], [160, 20, 180], [[0., 0.], [1., 1.], [2., 2.]])
        history, nav = branches(extractor(3), data)
        wind_only, wind_nav = branches(extractor(3, ablation="current_wind_only"), data)
        torch.testing.assert_close(wind_only[:, 13:15], data[-1:, 19:21].expand(3, 2))
        self.assertTrue(torch.equal(wind_only[:, 15:], history[:, 15:]))
        self.assertTrue(torch.equal(wind_nav, nav))
        without_motion, motion_nav = branches(extractor(3, ablation="no_motion_history"), data)
        self.assertTrue(torch.equal(without_motion[:, :15], history[:, :15]))
        self.assertTrue(torch.equal(without_motion[:, 15:], torch.tensor([[0., 0., 1., 0.]]).expand(3, 4)))
        self.assertTrue(torch.equal(motion_nav, nav))
        without_xy, xy_nav = branches(extractor(3, ablation="no_absolute_xy"), data)
        self.assertTrue(torch.equal(without_xy, history))
        self.assertTrue(torch.equal(xy_nav[4:6], torch.zeros(2)))
        self.assertTrue(torch.equal(xy_nav[:4], nav[:4]))
        self.assertTrue(torch.equal(xy_nav[6:], nav[6:]))

    def test_encoders_ignore_older_radar_and_train_both_branches(self):
        data = frames([45, 70, 85], [40, 45, 60], [[.1, .2], [.3, .4], [.5, .6]])
        data[:, :12] = torch.arange(36, dtype=torch.float32).reshape(3, 12) / 36
        changed = data.clone()
        changed[:-1, 21:] = .9
        for encoder in ("gru", "mlp", "transformer"):
            with self.subTest(encoder=encoder):
                net = extractor(3, encoder).eval()
                out = net(data.flatten()[None])
                self.assertEqual(out.shape, (1, 128))
                self.assertTrue(torch.equal(out, net(changed.flatten()[None])))
                out.square().sum().backward()
                for branch in (net.olfactory, net.navigation):
                    gradients = [p.grad for p in branch.parameters() if p.grad is not None]
                    self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
                    self.assertTrue(any(torch.count_nonzero(g) for g in gradients))
                if encoder == "transformer":
                    output, attention = net.forward_with_attention(data.flatten()[None])
                    torch.testing.assert_close(output, out)
                    self.assertTrue(len(attention) > 0)

    def test_raw_frames_remain_identical_and_old_scene_bank_is_rejected(self):
        cfg = config()
        legacy_cfg = {k: v for k, v in cfg.items() if k != "feature_layout"}
        new, old = MobileWhiskerPuffEnv(cfg), MobileWhiskerPuffEnv(legacy_cfg)
        self.addCleanup(new.close)
        self.addCleanup(old.close)
        np.testing.assert_array_equal(new.reset(seed=7)[0], old.reset(seed=7)[0])
        for action in ([.3, .25, 2, 9], [-.8, -.5, 5, 3]):
            left, right = new.step(action), old.step(action)
            np.testing.assert_array_equal(left[0], right[0])
            self.assertEqual(left[1:4], right[1:4])
        with tempfile.TemporaryDirectory() as tmp:
            bank = Path(tmp) / "bank.json"
            prepare_scenario_bank(bank, legacy_cfg, count=1, base_seed=7, regenerate=False)
            with self.assertRaisesRegex(ValueError, "契约"):
                prepare_scenario_bank(bank, cfg, count=1, base_seed=7, regenerate=False)

    def test_model_roundtrip_and_same_shape_contract_rejections(self):
        cfg = config()
        legacy_cfg = {k: v for k, v in cfg.items() if k != "feature_layout"}
        envs = [ObservationHistoryWrapper(MobileWhiskerPuffEnv(c), history_length=2)
                for c in (cfg, legacy_cfg, {**cfg, "history_ablation": "no_absolute_xy"})]
        for env in envs:
            self.addCleanup(env.close)
        new, old, ablated = envs
        self.assertEqual(new.observation_space, old.observation_space)
        with tempfile.TemporaryDirectory() as tmp:
            for env, layout, other in ((new, WIND_MOTION_HISTORY, old), (old, None, new)):
                model = PPO(HybridActorCriticPolicy, env, n_steps=4, batch_size=4, n_epochs=1, device="cpu",
                            policy_kwargs=dict(features_extractor_class=OlfactoryNavigationExtractor,
                                               features_extractor_kwargs=dict(history_length=2, feature_layout=layout)))
                contract = environment_contract(env.unwrapped, 2)
                contract["encoder_config"] = model.policy.features_extractor.configuration
                model.mobile_environment_contract = contract
                model.learn(4)
                model.save(Path(tmp) / "model.zip")
                loaded = PPO.load(Path(tmp) / "model.zip", env=env, device="cpu")
                self.assertEqual(validate_model_environment(loaded, env.unwrapped, 2, resume=True), 2)
                loaded.learn(4, reset_num_timesteps=False)
                self.assertEqual(loaded.num_timesteps, 8)
                for candidate in (other, ablated):
                    with self.assertRaisesRegex(ValueError, "契约"):
                        validate_model_environment(loaded, candidate.unwrapped, 2)


if __name__ == "__main__":
    unittest.main()
