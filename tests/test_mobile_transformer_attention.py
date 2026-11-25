"""注意力回放的历史打包、空间尾部与命令行兼容性。"""

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from stable_baselines3 import PPO
import torch
import yaml

from dual_whisker_rl.agents.hybrid_policy import HybridActorCriticPolicy
from dual_whisker_rl.agents.olfactory_navigation import OlfactoryNavigationExtractor
from dual_whisker_rl.envs import MobileWhiskerPuffEnv
from dual_whisker_rl.mobile_contract import environment_contract
from dual_whisker_rl.mobile_observation import MobileObservationHistory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import analyze_mobile_transformer_attention as attention


THRESHOLDS = attention.EventThresholds(.05, .02, .2, .02)


def config(spatial):
    path = ROOT / "configs" / ("mobile_rooms_spatial_b.yaml" if spatial else "mobile_rooms_hybrid_history.yaml")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg["max_steps"] = 4
    cfg["plume_overrides"].update(puff_warmup_steps=0, max_puffs=20)
    return cfg


def untrained_model(cfg):
    """仅初始化并保存网络，不调用learn，不启动训练。"""
    env = MobileObservationHistory(MobileWhiskerPuffEnv(cfg), 3)
    model = PPO(HybridActorCriticPolicy, env, n_steps=4, batch_size=4, n_epochs=1,
                device="cpu", seed=2,
                policy_kwargs={"net_arch": [16], "features_extractor_class": OlfactoryNavigationExtractor,
                    "features_extractor_kwargs": {"history_length": 3, "temporal_encoder": "transformer",
                        "feature_layout": cfg["feature_layout"], "encoder_kwargs": {
                            "d_model": 16, "n_heads": 2, "n_layers": 1,
                            "dim_feedforward": 32, "dropout": 0., "features_dim": 16}}})
    contract = environment_contract(env.unwrapped, 3)
    contract["encoder_config"] = model.policy.features_extractor.configuration
    model.mobile_environment_contract = contract
    model.policy.set_training_mode(False)
    return model, env


class MobileTransformerAttentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_v1_and_v2_attention_match_training_history_without_repeated_map(self):
        for spatial in (False, True):
            with self.subTest(spatial=spatial):
                cfg = config(spatial)
                model, training_env = untrained_model(cfg)
                self.addCleanup(training_env.close)
                raw = MobileWhiskerPuffEnv(cfg)
                self.addCleanup(raw.close)
                data = attention.capture_attention_rollout(raw, model, model.policy.features_extractor,
                    steps=2, resolution=4, seed=7, deterministic=True, thresholds=THRESHOLDS)
                tail_dim = 802 if spatial else 0
                self.assertEqual(data["stacked_observations"].shape, (3, 57 * 3 + tail_dim))
                self.assertEqual(data["core_history_observations"].shape, (3, 3, 57))
                self.assertEqual(data["current_memory_observations"].shape, (3, tail_dim))
                self.assertEqual(data["raw_attention"].shape, (3, 1, 2, 4))
                self.assertEqual(data["normalized_attention"].shape, (3, 1, 2, 3))
                np.testing.assert_allclose(data["normalized_attention"].sum(-1), 1., atol=1e-6)
                replay = MobileObservationHistory(MobileWhiskerPuffEnv(cfg), 3)
                self.addCleanup(replay.close)
                expected = [replay.reset(seed=7)[0]]
                for action in data["actions"][:-1]:
                    expected.append(replay.step(action)[0])
                np.testing.assert_array_equal(data["stacked_observations"], expected)
                if spatial:
                    np.testing.assert_array_equal(data["current_memory_observations"][-1],
                                                  replay.unwrapped.spatial_memory.observation())
                    self.assertFalse(np.array_equal(data["current_memory_observations"][0],
                                                    data["current_memory_observations"][-1]))
                with tempfile.TemporaryDirectory() as tmp:
                    output = Path(tmp)
                    args = Namespace(model_path=output / "model.zip", config=output / "config.yaml",
                                     seed=7, event_context=2)
                    attention.save_data_outputs(data, output, args=args,
                                                extractor=model.policy.features_extractor, thresholds=THRESHOLDS)
                    payload = json.loads((output / "events.json").read_text(encoding="utf-8"))
                    packing = payload["metadata"]["observation_packing"]
                    self.assertEqual(packing["base_observation_dim"], 57)
                    self.assertEqual(packing["raw_observation_dim"], 57 + tail_dim)
                    self.assertEqual(packing["current_memory_dim"], tail_dim)
                    with np.load(output / "attention.npz") as arrays:
                        np.testing.assert_array_equal(arrays["core_history_observations"],
                                                      data["core_history_observations"])
                        np.testing.assert_array_equal(arrays["current_memory_observations"],
                                                      data["current_memory_observations"])

    def test_v2_cli_loads_model_and_exports_attention(self):
        cfg = config(True)
        model, env = untrained_model(cfg)
        self.addCleanup(env.close)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            model_path = directory / "model.zip"
            config_path = directory / "config.yaml"
            output = directory / "attention"
            model.save(model_path)
            config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            arguments = ["analyze_mobile_transformer_attention.py", "--model-path", str(model_path),
                         "--config", str(config_path), "--output-dir", str(output),
                         "--steps", "1", "--resolution", "4"]
            with patch.object(sys, "argv", arguments), \
                    patch.object(attention, "save_timeline_plot"), \
                    patch.object(attention, "save_event_summary_plot"), \
                    patch.object(attention, "save_per_head_plots"), \
                    patch.object(attention, "save_attention_animation"):
                attention.main()
            payload = json.loads((output / "events.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"]["history_length"], 3)
            self.assertEqual(payload["metadata"]["observation_packing"]["stacked_observation_dim"], 973)
            with np.load(output / "attention.npz") as arrays:
                self.assertEqual(arrays["normalized_cls_history_attention"].shape, (2, 1, 2, 3))


if __name__ == "__main__":
    unittest.main()
