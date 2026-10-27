"""已有模型复评的来源、单模型计数、接口和缓存恢复检查。"""
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from stable_baselines3 import PPO
from dual_whisker_rl.agents import GRUHistoryExtractor
from dual_whisker_rl.experiment_env import make_experiment_env
from dual_whisker_rl.e4_experiments import sha256_file
from scripts import run_red_experiments as runner


class PretrainedExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.fixture = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.fixture.name) / "trained model.zip"
        # 实际做短训练生成测试模型，不依赖用户的正式模型或训练产物。
        env = make_experiment_env(dict(world_bounds=[-2.5, 2.5], max_steps=4,
            scenario_mode="randomized", plume_overrides={"puff_warmup_steps": 0}),
            dict(history_length=20), seed=17)
        try:
            model = PPO("MlpPolicy", env, seed=17, device="cpu", n_steps=8,
                batch_size=8, n_epochs=1, policy_kwargs=dict(net_arch=[8],
                    features_extractor_class=GRUHistoryExtractor,
                    features_extractor_kwargs=dict(history_length=20,
                        base_observation_dim=16, hidden_size=8, features_dim=8)))
            model.learn(16)
            model.save(cls.model_path)
        finally:
            env.close()

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def arguments(self, phase="run", *extra):
        return runner.parser().parse_args([phase, "--model-path", str(self.model_path),
                                          "--experiments", "e4", "--bootstrap", "30",
                                          "--max-steps", "4", *extra])

    def resolve(self, args):
        return runner.resolve(args)

    def test_formal_uses_actual_single_seed_and_isolated_output(self):
        args = self.arguments("plan", "--profile", "formal")
        _, profile, _, names, _, payload = self.resolve(args)
        self.assertEqual(names, ["main"])
        self.assertEqual(profile["seeds"], [17])
        self.assertEqual(payload["external_model"]["actual_timesteps"], 16)
        self.assertEqual(payload["selection"], "user_supplied_pretrained")
        self.assertFalse(payload["external_model"]["training_config_verified"])
        self.assertTrue(args.output_dir.endswith("pretrained/trained model"))

    def test_native_defaults_are_unchanged(self):
        args = runner.parser().parse_args(["plan", "--profile", "formal"])
        _, profile, _, names, _, payload = self.resolve(args)
        self.assertEqual(len(names), 6)
        self.assertEqual(profile["seeds"], [1, 2, 3, 4, 5])
        self.assertEqual(args.output_dir, "results/simulation_experiments")
        self.assertNotIn("external_model", payload)

    def test_rejects_training_delegation_multiple_variants_and_seed_relabel(self):
        for phase, extra in [("train", []), ("e4", []),
                ("run", ["--variants", "main,no_dr"]),
                ("run", ["--model-seed", "99"]), ("run", ["--group", "history"])]:
            with self.subTest(phase=phase, extra=extra), self.assertRaises(ValueError):
                self.resolve(self.arguments(phase, *extra))
        with self.assertRaises(ValueError):
            self.resolve(runner.parser().parse_args(["plan", "--model-seed", "17"]))

    def test_missing_seed_requires_explicit_declaration_and_suffix_is_optional(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unknown.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data", json.dumps(dict(seed=None, num_timesteps=123)))
            args = runner.parser().parse_args(["plan", "--model-path", str(path.with_suffix(""))])
            with self.assertRaisesRegex(ValueError, "model-seed"):
                self.resolve(args)
            args.model_seed = 31
            _, profile, _, _, _, payload = self.resolve(args)
            self.assertEqual(profile["seeds"], [31])
            self.assertEqual(payload["external_model"]["seed_origin"], "user_declared")

    def test_run_and_robustness_skip_training_and_resume_without_loading(self):
        original_hash = sha256_file(self.model_path)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            args = self.arguments()
            spec, profile, cfg, names, _, payload = self.resolve(args)
            root = Path(tmp)
            manifest = runner.prepare(root, profile, cfg, payload)
            with patch.object(runner, "train_one", side_effect=AssertionError("不应训练")) as train, \
                    patch.object(PPO, "learn", side_effect=AssertionError("不应训练")):
                runner.execute(args, root, names, spec, profile, cfg, manifest)
            train.assert_not_called()
            episodes = list((root / "main" / "seed_17" / "episodes" / "nominal").glob("*.json"))
            self.assertEqual(len(episodes), 8)
            row = json.loads(episodes[0].read_text(encoding="utf-8"))
            self.assertEqual(row["model_origin"], "pretrained")
            self.assertEqual(row["training_timesteps"], 16)
            self.assertEqual(row["model_sha256"], original_hash)
            summary = json.loads((root / "summaries" / "nominal.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["independent_training_seed_count"], 1)
            self.assertEqual(summary["coverage"]["main"], dict(actual=8, expected=8, complete=True))
            self.assertFalse(list(root.rglob("state.json")))
            self.assertFalse(list(root.rglob("*model.zip")))
            args.resume = True
            with patch.object(PPO, "load", side_effect=AssertionError("全缓存不应加载")):
                runner.execute(args, root, names, spec, profile, cfg, manifest)
            args.phase, args.faults, args.limit = "robustness", "nominal,delay2", 2
            args.experiments = ["e6"]
            runner.execute(args, root, names, spec, profile, cfg, manifest)
            self.assertEqual(len(list((root / "main" / "seed_17" / "robustness").glob("*/*.json"))), 4)
            with patch.object(PPO, "load", side_effect=AssertionError("全缓存不应加载")):
                runner.execute(args, root, names, spec, profile, cfg, manifest)
        self.assertEqual(sha256_file(self.model_path), original_hash)

    def test_incompatible_variant_fails_before_writing_episodes(self):
        for name in ["fixed", "h10", "history_mlp"]:
            with self.subTest(variant=name), tempfile.TemporaryDirectory() as tmp:
                args = self.arguments("evaluate", "--variants", name)
                spec, profile, cfg, names, _, payload = self.resolve(args)
                root = Path(tmp)
                manifest = runner.prepare(root, profile, cfg, payload)
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    runner.evaluate(root, names, spec, profile, cfg, manifest, args)
                self.assertFalse(list(root.rglob("episodes/*/*.json")))

    def test_source_change_rejected_even_when_results_are_cached(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            copied_model = Path(tmp) / "model.zip"
            copied_model.write_bytes(self.model_path.read_bytes())
            args = self.arguments("evaluate")
            args.model_path = copied_model
            spec, profile, cfg, names, _, payload = self.resolve(args)
            root = Path(tmp) / "evaluation"
            manifest = runner.prepare(root, profile, cfg, payload)
            runner.evaluate(root, names, spec, profile, cfg, manifest, args)
            args.resume = True
            copied_model.write_bytes(copied_model.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                runner.evaluate(root, names, spec, profile, cfg, manifest, args)
            *_, new_payload = self.resolve(args)
            with self.assertRaises(ValueError):
                runner.prepare(root, profile, cfg, new_payload)


if __name__ == "__main__":
    unittest.main()
