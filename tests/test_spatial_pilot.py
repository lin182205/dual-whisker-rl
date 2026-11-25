"""核对空间探索筛选的同题库、截尾指标及预设晋级规则。"""
from __future__ import annotations

import copy
import importlib.util
import json
import io
from contextlib import redirect_stdout
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("spatial_pilot", ROOT / "scripts/run_spatial_exploration_pilot.py")
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def config(variant="A"):
    return yaml.safe_load((ROOT / f"configs/mobile_rooms_spatial_{variant.lower()}.yaml").read_text(encoding="utf-8"))


def scenes(count, base_seed):
    return [pilot.E4Scenario(str(index), "validation", base_seed + index, .15, 0.,
                             (3., -3.5), (0., 0., 0.), 1., "randomized", "randomized", 0, index)
            for index in range(count)]


def trajectory(hit=False, *, cells=1, new_bonus=0., revisit_cost=0.):
    return {"odor_hit": hit, "is_success": False, "collision": False, "novel_cells": cells,
            "blank_loop": False, "blank_revisit_active": revisit_cost > 0,
            "reacquisition_event": False,
            "reward_components": {"exploration_bonus": new_bonus, "blank_revisit_penalty": -revisit_cost}}


def result(variant, hits, first_hit=1000., *, checkpoint=1_000_000,
           mode="deterministic", split="development", count=64):
    return {"signature": {"variant": variant, "checkpoint": checkpoint, "policy_mode": mode,
                           "split": split, "model_sha256": variant * 64,
                           "scene_sha256": "shared_scenes"},
            "summary": {"episodes": count, "ever_hit_episodes": hits,
                        "mean_first_hit_step_capped": first_hit}, "episodes": []}


class SpatialPilotTests(unittest.TestCase):
    def test_canonical_ignores_only_reviewed_variant_differences(self):
        reference = pilot.canonical_config(config("A"))
        for variant in "ABCD":
            pilot.validate_variant(config(variant), variant)
            self.assertEqual(reference, pilot.canonical_config(config(variant)))
        changed = config("A")
        changed["plume_overrides"]["puff_init_mass"] *= 2
        self.assertNotEqual(reference, pilot.canonical_config(changed))
        changed = config("A")
        changed["future_unknown_option"] = True
        self.assertNotEqual(reference, pilot.canonical_config(changed))

    def test_seed_and_recording_do_not_change_bank(self):
        altered = {**config(), "seed": 123, "record_trajectory": True}
        self.assertEqual(pilot.canonical_config(config()), pilot.canonical_config(altered))

    def test_wrong_variant_and_protocol_horizon_rejected(self):
        with self.assertRaisesRegex(ValueError, "A组"):
            pilot.validate_variant(config("D"), "A")
        with self.assertRaisesRegex(ValueError, "max_steps"):
            pilot.canonical_config({**config(), "max_steps": 20})

    def test_all_variants_reuse_same_canonical_scenes_and_physics_guard(self):
        def physical(value):
            return {"canonical_config": pilot.canonical_config(value), "map_sha256": "map"}

        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "physical_summary", side_effect=physical), patch.object(
                pilot, "generate_scenarios", side_effect=lambda cfg, count, base_seed: scenes(count, base_seed)) as generate:
            path = Path(directory)
            first, payload = pilot.scene_bank(path, config(), "development", create=True)
            self.assertEqual((len(first), first[0].seed, first[-1].seed), (64, 420000, 420063))
            self.assertEqual(generate.call_args.args[0], pilot.canonical_config(config()))
            for variant in "BCD":
                actual, saved = pilot.scene_bank(path, config(variant), "development")
                self.assertEqual(actual, first)
                self.assertEqual(saved["scene_sha256"], payload["scene_sha256"])
            self.assertEqual(generate.call_count, 1)
            changed = config("D")
            changed["plume_overrides"]["puff_init_mass"] *= 2
            with self.assertRaisesRegex(ValueError, "物理配置"):
                pilot.scene_bank(path, changed, "development")
            heldout, _ = pilot.scene_bank(path, config("B"), "heldout", create=True)
            self.assertEqual((len(heldout), heldout[0].seed, heldout[-1].seed), (128, 520000, 520127))
            self.assertTrue(set(scene.seed for scene in first).isdisjoint(scene.seed for scene in heldout))

    def test_tampered_scene_hash_rejected(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
                pilot, "physical_summary", return_value={"canonical_config": config()}), patch.object(
                pilot, "generate_scenarios", side_effect=lambda cfg, count, base_seed: scenes(count, base_seed)):
            path = Path(directory)
            pilot.scene_bank(path, config(), "development", create=True)
            bankfile = pilot.bank_path(path, "development")
            payload = pilot.read_json(bankfile)
            payload["scenes"][0]["robot_pose"][0] += 1
            pilot.write_json(bankfile, payload)
            with self.assertRaisesRegex(ValueError, "摘要不一致"):
                pilot.scene_bank(path, config(), "development")

    def test_no_hit_episode_stays_in_capped_mean_denominator(self):
        hit = pilot.trajectory_metrics([trajectory(), trajectory(True)], dt=.2)
        nohit = pilot.trajectory_metrics([trajectory()], dt=.2)
        summary = pilot.summarize([hit, nohit])
        self.assertEqual(hit["first_hit_step"], 2)
        self.assertIsNone(nohit["first_hit_step"])
        self.assertEqual(nohit["first_hit_step_capped"], 1200)
        self.assertEqual(summary["mean_first_hit_step_capped"], 601)
        self.assertAlmostEqual(summary["mean_first_hit_time_s_capped"], 120.2)
        self.assertEqual(summary["ever_hit_episodes"], 1)
        self.assertEqual(summary["episodes"], 2)

    def test_metrics_include_reset_cell_and_positive_cost_totals(self):
        rows = [trajectory(cells=1), trajectory(cells=2, new_bonus=.02),
                trajectory(cells=2, revisit_cost=.003)]
        rows[0]["blank_loop"] = True
        rows[1]["collision"] = True
        rows[2]["reacquisition_event"] = True
        metrics = pilot.trajectory_metrics(rows, dt=.2)
        self.assertEqual(metrics["unique_visited_cells"], 2)
        self.assertAlmostEqual(metrics["exploration_bonus_total"], .02)
        self.assertAlmostEqual(metrics["blank_revisit_cost_total"], .003)
        self.assertAlmostEqual(metrics["blank_revisit_seconds"], .2)
        self.assertAlmostEqual(metrics["blank_loop_ratio"], 1/3)
        self.assertEqual(metrics["reacquisition_events"], 1)
        self.assertTrue(metrics["had_collision"])
        with self.assertRaisesRegex(ValueError, "缺少"):
            pilot.trajectory_metrics([{"odor_hit": False}], dt=.2)

    def test_per_scene_seed_reproduces_numpy_and_torch_samples(self):
        seed = pilot.episode_action_seed(420000)
        pilot.seed_action_sampling(seed)
        expected = (np.random.random(), float(torch.rand(1)))
        pilot.seed_action_sampling(pilot.episode_action_seed(420001))
        pilot.seed_action_sampling(seed)
        self.assertEqual(expected, (np.random.random(), float(torch.rand(1))))
        self.assertNotEqual(seed, pilot.episode_action_seed(520000))

    def test_promotion_requires_all_four_and_four_more_hit_episodes(self):
        self.assertEqual(pilot.select_candidate([result("A", 10)])["status"], "pending")
        group = [result("A", 10), result("B", 13), result("C", 12), result("D", 11)]
        self.assertEqual(pilot.select_candidate(group)["status"], "no_candidate")
        group[1] = result("B", 14)
        self.assertEqual(pilot.select_candidate(group)["selected_variant"], "B")
        # 随机策略或较早检查点不参与晋级。
        group.extend([result("D", 64, mode="stochastic"), result("C", 64, checkpoint=500000)])
        self.assertEqual(pilot.select_candidate(group)["selected_variant"], "B")

    def test_promotion_ties_use_all_episode_capped_first_hit_mean(self):
        group = [result("A", 10), result("B", 14, 950), result("C", 14, 900), result("D", 13, 500)]
        chosen = pilot.select_candidate(group)
        self.assertEqual(chosen["selected_variant"], "C")
        self.assertEqual(chosen["gain_over_A_episodes"]["C"], 4)

    def test_promotion_rejects_different_scene_banks(self):
        group = [result("A", 10), result("B", 14), result("C", 12), result("D", 11)]
        group[1]["signature"]["scene_sha256"] = "other_scenes"
        with self.assertRaisesRegex(ValueError, "相同开发题库"):
            pilot.select_candidate(group)

    def test_report_rejects_duplicate_model_selection(self):
        with self.assertRaisesRegex(ValueError, "多个评估"):
            pilot.build_report([result("A", 10), result("A", 12)])

    def test_heldout_must_use_the_exact_screened_model(self):
        group = [result("A", 10), result("B", 14), result("C", 11), result("D", 10)]
        a = result("A", 10, split="heldout", count=128)
        b = result("B", 20, split="heldout", count=128)
        a["episodes"] = [{"ever_hit": index < 10} for index in range(128)]
        b["episodes"] = [{"ever_hit": index < 20} for index in range(128)]
        report = pilot.build_report([*group, a, b])
        self.assertEqual(report["heldout_comparisons"][0]["gain_ever_hit_episodes"], 10)
        b["signature"]["model_sha256"] = "another_model"
        with self.assertRaisesRegex(ValueError, "同一模型"):
            pilot.build_report([*group, a, b])

    def test_saved_evaluation_requires_complete_paired_rows(self):
        rows = [{"scenario_id": "development_420000", "scenario_seed": 420000,
                 **pilot.trajectory_metrics([trajectory()], dt=.2)}]
        payload = {"signature": {"protocol_version": pilot.PROTOCOL_VERSION,
                                  "scene_sha256": "scene", "physical_sha256": "physical"},
                   "episodes": rows, "summary": pilot.summarize(rows)}
        bank = {"scene_sha256": "scene", "physical_sha256": "physical",
                "scenes": [{"scenario_id": "development_420000", "seed": 420000}]}
        pilot.verify_evaluation(payload, bank)
        altered = copy.deepcopy(payload)
        altered["episodes"][0]["scenario_seed"] += 1
        with self.assertRaisesRegex(ValueError, "完整覆盖"):
            pilot.verify_evaluation(altered, bank)

    def test_watch_accepts_only_complete_stable_model_archives(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.zip"
            self.assertFalse(pilot.model_ready(path))
            path.write_text("still saving", encoding="utf-8")
            os.utime(path, (1, 1))
            self.assertFalse(pilot.model_ready(path))
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("data", "{}")
                archive.writestr("policy.pth", "fake weights")
            self.assertFalse(pilot.model_ready(path))
            os.utime(path, (1, 1))
            self.assertTrue(pilot.model_ready(path))

    def test_task_discovery_checks_actual_steps_and_final_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            check = root / "attempts" / "001" / "checkpoints" / "ppo_250000_steps.zip"
            final = root / "attempts" / "001" / "final_model.zip"
            check.parent.mkdir(parents=True)

            def save(path, steps):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("data", json.dumps({"num_timesteps": steps}))
                    archive.writestr("policy.pth", "fake weights")
                os.utime(path, (1, 1))

            save(check, 250000)
            item = {"task_root": root}
            self.assertEqual(pilot.discover_model(item, 250000), check)
            self.assertIsNone(pilot.discover_model(item, 500000))
            save(final, 900000)
            self.assertIsNone(pilot.discover_model(item, 1000000))
            save(final, 1000000)
            self.assertEqual(pilot.discover_model(item, 1000000), final)
            save(check, 200000)
            self.assertIsNone(pilot.discover_model(item, 250000))

    def test_manifest_supports_dashboard_task_roots(self):
        manifest = {"protocol_version": pilot.PROTOCOL_VERSION, "groups": {
            variant: {"task_root": f"results/training_dashboard/tasks/{variant}"} for variant in "ABCD"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            pilot.write_json(path, manifest)
            loaded = pilot.read_manifest(path)
            self.assertEqual(set(loaded), set("ABCD"))
            self.assertEqual(loaded["A"]["task_root"], ROOT / "results/training_dashboard/tasks/A")
            del manifest["groups"]["D"]
            pilot.write_json(path, manifest)
            with self.assertRaisesRegex(ValueError, "四组"):
                pilot.read_manifest(path)

    def test_watch_waits_without_starting_training(self):
        manifest = {variant: {"config": config(variant), "models": {
            step: Path(f"{variant}_{step}.zip") for step in pilot.CHECKPOINTS}} for variant in "ABCD"}
        partial_report = {"selection": {"status": "pending", "selected_variant": None}, "heldout_comparisons": []}
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "model_ready", return_value=False), patch.object(
                pilot, "evaluate_model") as evaluate, patch.object(pilot, "report", return_value=partial_report):
            state = pilot.watch_once(Path(directory), manifest)
            self.assertFalse(state["complete"])
            self.assertEqual(len(state["pending_checkpoints"]), 12)
            self.assertFalse(state["automatic_training"])
            evaluate.assert_not_called()

    def test_watch_entry_prepares_missing_banks_before_polling(self):
        manifest = {variant: {"config": config(variant)} for variant in "ABCD"}
        with patch.object(pilot, "read_manifest", return_value=manifest), patch.object(pilot, "scene_bank") as bank, patch.object(
                pilot, "watch_once", return_value={"complete": True}), redirect_stdout(io.StringIO()):
            pilot.main(["watch", "--manifest", "results/not_read_by_mock.json", "--once"])
        self.assertEqual(bank.call_count, 8)
        self.assertTrue(all(call.kwargs["create"] for call in bank.call_args_list))

    def test_watch_serially_evaluates_checkpoints_then_selected_heldout_pair(self):
        manifest = {variant: {"config": config(variant), "models": {
            step: Path(f"{variant}_{step}.zip") for step in pilot.CHECKPOINTS}} for variant in "ABCD"}
        selection = {"status": "candidate_selected", "selected_variant": "C"}
        reports = [{"selection": selection, "heldout_comparisons": []},
                   {"selection": selection, "heldout_comparisons": [{}, {}]}]
        with tempfile.TemporaryDirectory() as directory, patch.object(pilot, "model_ready", return_value=True), patch.object(
                pilot, "evaluate_model") as evaluate, patch.object(pilot, "report", side_effect=reports):
            state = pilot.watch_once(Path(directory), manifest)
            self.assertTrue(state["complete"])
            self.assertEqual(evaluate.call_count, 14)
            heldout = [(call.args[2], call.args[-1]) for call in evaluate.call_args_list[-2:]]
            self.assertEqual(heldout, [("A", "heldout"), ("C", "heldout")])


if __name__ == "__main__":
    unittest.main()
