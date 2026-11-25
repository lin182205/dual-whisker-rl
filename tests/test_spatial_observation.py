"""空间记忆观测在训练、复评和预览入口间的协议验证。"""

import copy
from pathlib import Path
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
import yaml
from stable_baselines3 import PPO

from dual_whisker_rl.agents.hybrid_policy import HybridActorCriticPolicy
from dual_whisker_rl.agents.olfactory_navigation import OlfactoryNavigationExtractor
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.e4_experiments import ObservationHistory
from dual_whisker_rl.mobile_contract import environment_contract, validate_model_environment
from dual_whisker_rl.mobile_features import WIND_MOTION_SPATIAL_MEMORY
from dual_whisker_rl.mobile_observation import MobileObservationHistory, ObservationAssembler

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def config(variant="b"):
    cfg = yaml.safe_load((ROOT / f"configs/mobile_rooms_spatial_{variant}.yaml").read_text(encoding="utf-8"))
    cfg["plume_overrides"]["puff_warmup_steps"] = 0
    return cfg


class SpatialObservationTests(unittest.TestCase):
    def test_current_map_only_and_history_reset(self):
        env = MobileWhiskerPuffEnv(config())
        self.addCleanup(env.close)
        history = ObservationAssembler(env, 20)
        frame = np.zeros(859, np.float32)
        frame[:57] = 1
        frame[57:] = 2
        initial = history.reset(frame)
        self.assertEqual(initial.shape, (1942,))
        changed = frame.copy()
        changed[:57] = 3
        changed[57:] = 4
        stacked = history.append(changed)
        np.testing.assert_array_equal(stacked[:19*57], 1)
        np.testing.assert_array_equal(stacked[19*57:20*57], 3)
        np.testing.assert_array_equal(stacked[20*57:], 4)
        frame[:] = 9  # 缓冲与调用者数组隔离。
        np.testing.assert_array_equal(initial[:57], 1)
        reset = history.reset(frame)
        np.testing.assert_array_equal(reset, 9)

    def test_shared_training_and_e4_wrapper_are_identical(self):
        envs = [wrapper(MobileWhiskerPuffEnv(config()), 20) for wrapper in (MobileObservationHistory, ObservationHistory)]
        for env in envs:
            self.addCleanup(env.close)
        observations = [env.reset(seed=31)[0] for env in envs]
        np.testing.assert_array_equal(*observations)
        for _ in range(4):
            observations = [env.step(np.array([.4, .2, 3, 7]))[0] for env in envs]
            np.testing.assert_array_equal(*observations)
        self.assertTrue(envs[0].observation_space.contains(observations[0]))

    def test_spatial_branch_gradients_and_same_architecture(self):
        architectures = []
        for variant in "abcd":
            env = MobileObservationHistory(MobileWhiskerPuffEnv(config(variant)), 20)
            self.addCleanup(env.close)
            obs, _ = env.reset(seed=11)
            torch.manual_seed(11)
            net = OlfactoryNavigationExtractor(env.observation_space, history_length=20,
                                               feature_layout=WIND_MOTION_SPATIAL_MEMORY)
            tensor = torch.tensor(obs[None], requires_grad=True)
            features = net(tensor)
            self.assertEqual(features.shape, (1, 192))
            self.assertTrue(torch.isfinite(features).all())
            features.square().sum().backward()
            self.assertGreater(net.spatial_projection[0].weight.grad.abs().sum().item(), 0)
            if variant != "a":
                self.assertGreater(tensor.grad[:, 1140:1940].abs().sum().item(), 0)
            else:
                np.testing.assert_array_equal(obs[1140:], 0)
            architectures.append({key: tuple(value.shape) for key, value in net.state_dict().items()})
        self.assertTrue(all(item == architectures[0] for item in architectures))

    def test_v2_checkpoint_roundtrip_and_v1_rejection(self):
        raw = MobileWhiskerPuffEnv(config("d"))
        env = MobileObservationHistory(raw, 20)
        self.addCleanup(env.close)
        model = PPO(HybridActorCriticPolicy, env, seed=11, n_steps=8, batch_size=8, n_epochs=1, device="cpu",
                    policy_kwargs={"features_extractor_class": OlfactoryNavigationExtractor,
                                   "features_extractor_kwargs": {"history_length": 20,
                                       "feature_layout": WIND_MOTION_SPATIAL_MEMORY}})
        contract = environment_contract(raw, 20)
        contract["encoder_config"] = model.policy.features_extractor.configuration
        model.mobile_environment_contract = contract
        model.learn(16)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "model.zip"
            model.save(path)
            loaded = PPO.load(path, env=env, device="cpu")
            self.assertEqual(validate_model_environment(loaded, raw, 20, resume=True), 20)
            self.assertEqual(loaded.num_timesteps, 16)
            loaded.learn(8, reset_num_timesteps=False)
            self.assertEqual(loaded.num_timesteps, 24)
            old_cfg = yaml.safe_load((ROOT / "configs/mobile_rooms_hybrid_history_improved.yaml").read_text(encoding="utf-8"))
            old = MobileWhiskerPuffEnv(old_cfg)
            self.addCleanup(old.close)
            with self.assertRaises(ValueError):
                validate_model_environment(loaded, old)
            other = MobileWhiskerPuffEnv(config("c"))
            self.addCleanup(other.close)
            with self.assertRaisesRegex(ValueError, "契约"):
                validate_model_environment(loaded, other)

    def test_v1_checkpoint_still_loads(self):
        cfg = yaml.safe_load((ROOT / "configs/mobile_rooms_hybrid_history_improved.yaml").read_text(encoding="utf-8"))
        cfg["plume_overrides"]["puff_warmup_steps"] = 0
        raw = MobileWhiskerPuffEnv(cfg)
        env = MobileObservationHistory(raw, 2)
        self.addCleanup(env.close)
        model = PPO(HybridActorCriticPolicy, env, seed=11, n_steps=4, batch_size=4, n_epochs=1, device="cpu",
                    policy_kwargs={"features_extractor_class": OlfactoryNavigationExtractor,
                                   "features_extractor_kwargs": {"history_length": 2, "feature_layout": cfg["feature_layout"]}})
        contract = environment_contract(raw, 2)
        contract["encoder_config"] = model.policy.features_extractor.configuration
        model.mobile_environment_contract = contract
        self.assertNotIn("observation_packing", contract)
        self.assertEqual(model.policy.features_extractor.features_dim, 128)
        with tempfile.TemporaryDirectory() as temp:
            model.save(Path(temp) / "old.zip")
            loaded = PPO.load(Path(temp) / "old.zip", device="cpu")
            self.assertEqual(validate_model_environment(loaded, raw, 2), 2)
            new = MobileWhiskerPuffEnv(config())
            self.addCleanup(new.close)
            with self.assertRaises(ValueError):
                validate_model_environment(loaded, new)


if __name__ == "__main__":
    unittest.main()
