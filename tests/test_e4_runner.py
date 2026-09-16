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
            args = RUNNER.argparse.Namespace(profile="formal", dry_run=False, resume=True)
            with patch.object(RUNNER, "calibration_signature", return_value="expected"):
                with patch.object(RUNNER, "read_scenarios", side_effect=AssertionError("不应重新评估")):
                    output = StringIO()
                    with redirect_stdout(output):
                        RUNNER.calibrate_command(args, root)
            self.assertIn("B1 已完成，跳过规则参数搜索", output.getvalue())
            self.assertIn("B2 已完成，跳过固定角度搜索", output.getvalue())


if __name__ == "__main__":
    unittest.main()
