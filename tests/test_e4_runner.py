from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_e4_experiments.py"
SPEC = importlib.util.spec_from_file_location("run_e4_experiments", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class E4RunnerTests(unittest.TestCase):
    def test_default_profile_is_smoke(self):
        args = RUNNER.build_parser().parse_args([])
        self.assertEqual(args.profile, "smoke")

    def test_formal_config_resolves_max_steps_before_calibration(self):
        args = RUNNER.build_parser().parse_args(
            [
                "run",
                "--profile",
                "formal",
                "--config",
                str(ROOT / "configs" / "e4_formal.yaml"),
                "--output-dir",
                "results/e4",
                "--resume",
            ]
        )
        self.assertIsNone(args.max_steps)
        RUNNER.resolve_args(args)
        self.assertEqual(args.max_steps, 400)

    def test_calibration_signature_defensively_reads_config_max_steps(self):
        args = RUNNER.build_parser().parse_args(
            [
                "run",
                "--profile",
                "formal",
                "--config",
                str(ROOT / "configs" / "e4_formal.yaml"),
            ]
        )
        args.methods = list(RUNNER.METHOD_ORDER)
        args.seeds = [1, 2, 3, 4, 5]
        args.n_envs = 8
        args.vec_env_backend = "subproc"
        args.checkpoint_rollouts = 25
        with patch.object(RUNNER, "verify_scenario_manifest", return_value={"validation": "v", "test": "t"}):
            signature = RUNNER.calibration_signature(args, ROOT)
        self.assertEqual(len(signature), 64)

    def test_completed_calibration_requires_matching_signature(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_b1.json"
            path.write_text(
                json.dumps(
                    {
                        "version": RUNNER.VERSION,
                        "method": "b1_reactive",
                        "status": "completed",
                        "signature": "expected",
                        "selected_config": {},
                    }
                ),
                encoding="utf-8",
            )
            payload = RUNNER.load_completed_calibration(path, "b1_reactive", "expected")
            self.assertIsNotNone(payload)
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                RUNNER.load_completed_calibration(path, "b1_reactive", "different")

    def test_known_legacy_calibration_signature_is_migrated_atomically(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_b1.json"
            path.write_text(
                json.dumps(
                    {
                        "version": RUNNER.VERSION,
                        "method": "b1_reactive",
                        "status": "completed",
                        "signature": "legacy",
                        "selected_config": {},
                    }
                ),
                encoding="utf-8",
            )
            payload = RUNNER.load_completed_calibration(
                path,
                "b1_reactive",
                "current",
                {"legacy"},
            )
            self.assertEqual(payload["signature"], "current")
            self.assertEqual(payload["migrated_from_signature"], "legacy")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["signature_version"], RUNNER.CALIBRATION_SIGNATURE_VERSION)

    def test_complete_untracked_b1_calibration_is_structurally_migrated(self):
        grid = RUNNER.b1_calibration_grid()
        scores = [
            {
                "candidate_index": index,
                "config": config,
                "success_rate": 1.0 if index == 0 else 0.0,
                "mean_final_distance": 0.5 + index,
            }
            for index, config in enumerate(grid)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_b1.json"
            path.write_text(
                json.dumps(
                    {
                        "version": RUNNER.VERSION,
                        "method": "b1_reactive",
                        "status": "completed",
                        "signature": "untracked-intermediate-signature",
                        "candidate_count": len(grid),
                        "selected_config": grid[0],
                        "scores": scores,
                    }
                ),
                encoding="utf-8",
            )
            payload = RUNNER.load_completed_calibration(
                path,
                "b1_reactive",
                "current",
                set(),
                [1, 2],
            )
            self.assertEqual(payload["signature"], "current")
            self.assertEqual(payload["migration_validation"], "complete_legacy_artifact_structure")

    def test_incomplete_untracked_b1_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_b1.json"
            path.write_text(
                json.dumps(
                    {
                        "version": RUNNER.VERSION,
                        "method": "b1_reactive",
                        "status": "completed",
                        "signature": "unknown",
                        "candidate_count": 48,
                        "selected_config": {},
                        "scores": [],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "signature mismatch"):
                RUNNER.load_completed_calibration(path, "b1_reactive", "current", set(), [1, 2])

    def test_unstarted_training_state_signature_is_safely_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            state = {
                "status": "prepared",
                "actual_timesteps": 0,
                "signature": "old",
                "scenario_hashes": {"validation": "v", "test": "t"},
            }
            args = RUNNER.argparse.Namespace(resume=True)
            refreshed = RUNNER.reconcile_training_state(
                args,
                task_dir,
                state,
                "current",
                {"validation": "v", "test": "t"},
                "main",
                1,
            )
            self.assertEqual(refreshed["signature"], "current")
            self.assertEqual(refreshed["signature_migrated_from"], "old")
            self.assertEqual(json.loads((task_dir / "state.json").read_text(encoding="utf-8"))["signature"], "current")

    def test_started_training_state_signature_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            task_dir = Path(directory)
            state = {
                "status": "interrupted",
                "actual_timesteps": 4096,
                "signature": "old",
                "scenario_hashes": {"validation": "v", "test": "t"},
            }
            args = RUNNER.argparse.Namespace(resume=True)
            with self.assertRaisesRegex(ValueError, "resume signature mismatch"):
                RUNNER.reconcile_training_state(
                    args,
                    task_dir,
                    state,
                    "current",
                    {"validation": "v", "test": "t"},
                    "main",
                    1,
                )

    def test_corrupt_calibration_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_b2.json"
            path.write_text("{broken", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                RUNNER.load_completed_calibration(path, "b2_fixed", "expected")

    def test_resume_skips_completed_b1_and_b2_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = {"version": RUNNER.VERSION, "status": "completed", "signature": "expected"}
            (root / "calibration_b1.json").write_text(
                json.dumps({**common, "method": "b1_reactive", "selected_config": {}}),
                encoding="utf-8",
            )
            (root / "calibration_b2.json").write_text(
                json.dumps({**common, "method": "b2_fixed", "selected_sector": 5}),
                encoding="utf-8",
            )
            args = RUNNER.argparse.Namespace(profile="formal", dry_run=False, resume=True, seeds=[1, 2])
            with patch.object(RUNNER, "calibration_signature", return_value="expected"):
                with patch.object(RUNNER, "legacy_calibration_signatures", return_value=set()):
                    with patch.object(RUNNER, "read_scenarios", side_effect=AssertionError("不应重新评估")):
                        output = StringIO()
                        with redirect_stdout(output):
                            RUNNER.calibrate_command(args, root)
            self.assertIn("B1 已完成，跳过规则参数搜索", output.getvalue())
            self.assertIn("B2 已完成，跳过固定角度搜索", output.getvalue())


if __name__ == "__main__":
    unittest.main()
