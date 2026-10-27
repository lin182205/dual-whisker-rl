"""验证评估估时、断点复用和显示升级的封存边界。"""
import copy
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from scripts.experiment_progress import (
    EvaluationProgress, duration_text, reporting_compatible_payload,
    PRE_ETA_RUNNER_SHA256, ETA_RUNNER_SHA256, RUNNER_PATH,
)
from dual_whisker_rl.experiment_protocol import digest, seal
from dual_whisker_rl.e4_experiments import sha256_file

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eta_red_runner", ROOT / RUNNER_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class EvaluationProgressTests(unittest.TestCase):
    def setUp(self):
        self.clock = Mock(return_value=0.0)
        self.printer = Mock()

    def progress(self, total, **kwargs):
        return EvaluationProgress(total, clock=self.clock, printer=self.printer, **kwargs)

    def test_resume_uses_only_new_episode_durations(self):
        progress = self.progress(100, cached=90)
        self.assertIsNone(progress.snapshot()["eta_s"])
        self.clock.return_value = 1800.0  # 缓存读取或加载模型不应伪造速度样本。
        progress.finish_episode(12.0)
        state = progress.snapshot()
        self.assertEqual(state["completed"], 91)
        self.assertEqual(state["elapsed_s"], 1800.0)
        self.assertEqual(state["eta_s"], 108.0)
        progress.finish_episode(8.0)
        self.assertEqual(progress.snapshot()["eta_s"], 80.0)

    def test_initial_first_count_time_and_final_reports(self):
        progress = self.progress(24)
        self.assertTrue(progress.report("评估"))
        self.assertIn("等待首个新场景完成", self.printer.call_args.args[0])
        progress.finish_episode(2.0)
        self.assertTrue(progress.report("评估"))
        self.assertIn("预计剩余=00:00:46", self.printer.call_args.args[0])
        progress.finish_episode(2.0)
        self.assertFalse(progress.report("评估"))
        self.clock.return_value = 11.0
        self.assertTrue(progress.report("评估"))
        for _ in range(20):
            progress.finish_episode(2.0)
        self.assertTrue(progress.report("评估"))
        progress.finish_episode(2.0)
        self.assertFalse(progress.report("评估"))
        progress.finish_episode(2.0)
        self.assertTrue(progress.report("评估"))
        self.assertIn("场景=24/24 (100.0%)", self.printer.call_args.args[0])
        self.assertIn("预计剩余=00:00:00", self.printer.call_args.args[0])
        self.assertFalse(progress.report("评估"))

    def test_all_cached_and_empty_report_zero(self):
        for total in (0, 6):
            progress = self.progress(total, cached=total)
            self.assertEqual(progress.snapshot()["eta_s"], 0.0)
            self.assertTrue(progress.report("评估"))
            self.assertIn("预计剩余=00:00:00", self.printer.call_args.args[0])

    def test_duration_and_invalid_measurements(self):
        self.assertEqual(duration_text(3661.2), "01:01:02")
        progress = self.progress(1)
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                progress.finish_episode(value)
        progress.finish_episode(1)
        with self.assertRaises(ValueError):
            progress.finish_episode(1)


class ReportingCompatibilityTests(unittest.TestCase):
    def payloads(self):
        current = dict(settings=dict(timesteps=64), scenes=["a"], code_sha256={
            RUNNER_PATH: ETA_RUNNER_SHA256, "env.py": "unchanged"})
        previous = copy.deepcopy(current)
        previous["code_sha256"][RUNNER_PATH] = PRE_ETA_RUNNER_SHA256
        saved = {**previous, "signature": digest(previous)}
        return current, previous, saved

    def test_only_known_display_upgrade_preserves_signature(self):
        current, previous, saved = self.payloads()
        compatible = reporting_compatible_payload(current, saved)
        self.assertEqual(compatible, previous)
        self.assertEqual(current["code_sha256"][RUNNER_PATH], ETA_RUNNER_SHA256)

    def test_changed_science_budget_scene_or_runner_rejected(self):
        for change in ("env", "budget", "scene", "future_runner", "tampered_saved"):
            current, _, saved = self.payloads()
            if change == "env": current["code_sha256"]["env.py"] = "changed"
            if change == "budget": current["settings"]["timesteps"] = 65
            if change == "scene": current["scenes"] = ["b"]
            if change == "future_runner": current["code_sha256"][RUNNER_PATH] = "new version"
            if change == "tampered_saved": saved["signature"] = "tampered"
            self.assertIs(reporting_compatible_payload(current, saved), current, change)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "manifest.json"
                path.write_text(json.dumps(saved), encoding="utf-8")
                with self.assertRaises(ValueError):
                    seal(path, reporting_compatible_payload(current, saved))

    def test_new_feature_runner_does_not_reuse_display_upgrade_exception(self):
        args = runner.parser().parse_args(["prepare", "--variants", "main"])
        *_, current = runner.resolve(args)
        self.assertNotIn(sha256_file(ROOT / RUNNER_PATH), [ETA_RUNNER_SHA256, PRE_ETA_RUNNER_SHA256])
        previous = copy.deepcopy(current)
        previous["code_sha256"][RUNNER_PATH] = PRE_ETA_RUNNER_SHA256
        saved = {**previous, "signature": digest(previous)}
        self.assertIs(reporting_compatible_payload(current, saved), current)

    def test_real_prepare_retains_old_signature_and_checks_scenes(self):
        args = runner.parser().parse_args(["prepare", "--variants", "main"])
        _, profile, cfg, _, _, current = runner.resolve(args)
        # 固定模拟当时的显示版本；当前已有模型入口升级不能继承这个兼容例外。
        current["code_sha256"][RUNNER_PATH] = ETA_RUNNER_SHA256
        previous = copy.deepcopy(current)
        previous["code_sha256"][RUNNER_PATH] = PRE_ETA_RUNNER_SHA256
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            saved = runner.prepare(root, profile, cfg, previous)
            manifest_bytes = (root / "manifest.json").read_bytes()
            self.assertEqual(runner.prepare(root, profile, cfg, current), saved)
            self.assertEqual((root / "manifest.json").read_bytes(), manifest_bytes)
            self.assertTrue((root / "reporting_upgrade.json").exists())
            scene_file = root / "scenarios_test.json"
            items = json.loads(scene_file.read_text(encoding="utf-8"))
            items[0]["wind_speed"] += .1
            scene_file.write_text(json.dumps(items), encoding="utf-8")
            with self.assertRaises(ValueError):
                runner.prepare(root, profile, cfg, current)


class ComparisonEvaluationProgressTests(unittest.TestCase):
    def test_global_progress_includes_methods_seeds_faults_and_resume(self):
        # 两方法、两seed、两故障，每组3场景且已完成1场，共24场景复用8场。
        scenes = [SimpleNamespace(scenario_id=f"s{i}", to_dict=lambda i=i: dict(id=i)) for i in range(3)]
        names, seeds, faults = ["main", "fixed"], [11, 22], ["nominal", "delay2"]
        suite = dict(variants={n: dict(method=n) for n in names}, faults={f: {} for f in faults})
        profile = dict(seeds=seeds, save_step_logs=False)
        args = SimpleNamespace(limit=3, faults=",".join(faults), resume=True)
        manifest = dict(signature="manifest")
        model = SimpleNamespace(observation_space="obs", action_space="act")
        env = Mock(observation_space="obs", action_space="act")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in names:
                for seed in seeds:
                    task = root / name / f"seed_{seed}"
                    task.mkdir(parents=True)
                    (task / "selected_model.zip").write_bytes(b"model")
                    model_hash = sha256_file(task / "selected_model.zip")
                    (task / "state.json").write_text(json.dumps(dict(status="completed", signature="manifest", selected_model_sha256=model_hash)), encoding="utf-8")
                    for fault in faults:
                        folder = task / "robustness" / fault
                        folder.mkdir(parents=True)
                        signature = digest(dict(manifest="manifest", model=model_hash, fault={}, scenes=[s.to_dict() for s in scenes]))
                        (folder / "s0.json").write_text(json.dumps(dict(scenario_id="s0", evaluation_signature=signature)), encoding="utf-8")
            output = io.StringIO()
            with patch.object(runner, "scenarios", return_value=scenes), \
                 patch.object(runner, "make_experiment_env", return_value=env), \
                 patch.object(runner, "rollout", side_effect=lambda _, __, scene, **kwargs: (dict(scenario_id=scene.scenario_id), [])) as rollout, \
                 patch("stable_baselines3.PPO.load", return_value=model) as load, redirect_stdout(output):
                runner.evaluate(root, names, suite, profile, {}, manifest, args, robust=True)
                self.assertEqual(rollout.call_count, 16)
                self.assertEqual(load.call_count, 4)
                text = output.getvalue()
                self.assertIn("场景=8/24", text)
                self.assertIn("场景=24/24 (100.0%) 本次新增=16 复用=8", text)
                self.assertIn("预计剩余=00:00:00", text)
                output.seek(0)
                output.truncate()
                rollout.reset_mock()
                load.reset_mock()
                runner.evaluate(root, names, suite, profile, {}, manifest, args, robust=True)
                self.assertEqual(rollout.call_count, 0)
                self.assertEqual(load.call_count, 0)
                self.assertIn("本次新增=0 复用=24", output.getvalue())
                self.assertIn("预计剩余=00:00:00", output.getvalue())


if __name__ == "__main__":
    unittest.main()
