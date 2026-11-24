"""验证管理状态、日志口径、接口边界和队列，均不接触正式训练。"""
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest.mock import patch
import uuid
import zipfile

from fastapi.testclient import TestClient
from dual_whisker_rl.dashboard.app import create_app
from dual_whisker_rl.dashboard.collect import LogProgress, thin
from dual_whisker_rl.dashboard.files import alive, atomic_json, identity
from dual_whisker_rl.dashboard.service import Service


def request(**values):
    return {"request_id": uuid.uuid4().hex, **values}


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.service = Service(self.root / "manager", require_gpu=False)

    def tearDown(self):
        self.service.store.close()
        self.tmp.cleanup()

    def create(self, **values):
        return self.service.create(request(target_steps=20000000, **values))["task_id"]

    def test_create_idempotency_and_rounding(self):
        body = request(target_steps=5001)
        first = self.service.create(body)
        self.assertEqual(first["target_steps"], 10000)
        self.assertEqual(self.service.create(body), first)
        self.assertEqual(len(self.service.store.tasks()), 1)
        with self.assertRaises(ValueError):
            self.service.create({**body, "target_steps": 15000})

    def test_queued_pause_resume_stop_and_target(self):
        task_id = self.create()
        self.service.action(task_id, request(action="pause"))
        self.assertEqual(self.service.store.get(task_id)["state"], "paused")
        response = self.service.action(task_id, request(action="set_target", target_steps=25000001))
        self.assertEqual(response["target_steps"], 25005000)
        body = request(action="resume")
        self.assertEqual(self.service.action(task_id, body)["state"], "queued")
        self.assertEqual(self.service.action(task_id, body)["state"], "queued")
        self.service.action(task_id, request(action="stop"))
        self.assertEqual(self.service.store.get(task_id)["state"], "stopped")
        with self.assertRaises(ValueError):
            self.service.action(task_id, request(action="resume"))

    def test_fifo_counts_imported_tasks_and_does_not_duplicate_on_tick(self):
        ids = [self.create(name=f"task-{i}") for i in range(3)]
        calls = []
        def fake_start(task):
            calls.append(task["id"])
            task["state"] = "running"
            self.service.store.save(task)
        with patch.object(self.service, "_start", side_effect=fake_start):
            self.service.tick()
            self.service.tick()
            self.assertEqual(calls, ids[:2])
            task = self.service.store.get(ids[0])
            task["state"] = "paused"
            self.service.store.save(task)
            self.service.tick()
            self.assertEqual(calls, ids)

    def test_log_progress_only_uses_finished_updates_and_handles_truncation(self):
        path = self.root / "train.log"
        path.write_text("| total_timesteps | 15005000 |\n", encoding="utf-8")
        reader = LogProgress()
        self.assertNotIn("current_steps", reader.read(path))
        with path.open("a", encoding="utf-8") as stream:
            stream.write("[训练轮次] 第1/4000轮结束 本轮总耗时=50.00秒 checkpoint=0秒 训练步数=15005000/35000000\n")
        self.assertEqual(reader.read(path)["current_steps"], 15005000)
        self.assertEqual(reader.read(path)["cycle_samples"], 1)
        path.write_text("new\n", encoding="utf-8")
        self.assertNotIn("current_steps", reader.read(path))

    def test_import_own_root_and_read_only_legacy_controls(self):
        cwd = self.root / "old"
        cwd.mkdir()
        config = {"feature_layout": "wind_motion_history_v1"}
        meta = {"config": config, "seed": 11, "n_envs": 8, "ppo": {"n_steps": 625, "batch_size": 250},
                "checkpoint_dir": "results/checkpoints", "tensorboard_run_dir": "results/tb", "model_path": "results/model.zip"}
        atomic_json(cwd / "meta.json", meta)
        (cwd / "train.log").write_text("[训练轮次] 第1/4000轮结束 本轮总耗时=50秒 训练步数=15005000/35000000\n", encoding="utf-8")
        manifest = {"runs": [{"cwd": str(cwd), "metadata_path": str(cwd / "meta.json"), "command": ["no-such-worker"],
                              "session": "legacy", "train_log": str(cwd / "train.log"),
                              "starting_timesteps": 15000000, "target_timesteps": 35000000}]}
        atomic_json(self.root / "manifest.json", manifest)
        self.service.import_manifest(self.root / "manifest.json")
        self.service.import_manifest(self.root / "manifest.json")
        task = self.service.store.tasks()[0]
        self.assertEqual(len(self.service.store.tasks()), 1)
        self.assertEqual(task["current_steps"], 15005000)
        self.assertEqual(task["attempts"][0]["checkpoint_dir"], str(cwd / "results/checkpoints"))
        listing = self.service.tasks()[0]
        self.assertAlmostEqual(listing["session_progress"], 5000 / 20000000)
        self.assertAlmostEqual(listing["total_progress"], 15005000 / 35000000)
        with self.assertRaisesRegex(ValueError, "仅监控"):
            self.service.action(task["id"], request(action="pause"))
        with self.assertRaisesRegex(ValueError, "尚无有效检查点"):
            self.service.takeover_preview(task["id"])

    def test_checkpoints_ignore_partial_files_and_unknown_source(self):
        task_id = self.create()
        task = self.service.store.get(task_id)
        cp = self.root / "checkpoints"
        cp.mkdir()
        (cp / "bad_10_steps.zip").write_bytes(b"partial archive")
        with zipfile.ZipFile(cp / "good_5000_steps.zip", "w") as z:
            z.writestr("data", json.dumps({"num_timesteps": 5000, "n_envs": 4, "n_steps": 1250, "batch_size": 250}))
        task["attempts"] = [{"checkpoint_dir": str(cp)}]
        self.service.store.save(task)
        rows = self.service.checkpoints(task_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["steps"], 5000)
        with self.assertRaises(ValueError):
            self.service._checkpoint(task_id, "../../outside")

    def test_pid_reuse_is_not_alive(self):
        current = identity(os.getpid())
        self.assertTrue(alive(current))
        self.assertFalse(alive({**current, "created": current["created"] - 10}))

    def test_downsampling_preserves_peak_endpoints_and_limit(self):
        points = [[i, 1000 if i == 12345 else 0] for i in range(20000)]
        result = thin(points)
        self.assertLessEqual(len(result), 1000)
        self.assertEqual(result[0], points[0])
        self.assertEqual(result[-1], points[-1])
        self.assertIn(points[12345], result)

    def test_service_restart_keeps_registry(self):
        task_id = self.create()
        self.service.store.close()
        self.service = Service(self.root / "manager", require_gpu=False)
        self.assertEqual(self.service.store.get(task_id)["state"], "queued")

    def test_late_goal_extension_does_not_mark_task_completed(self):
        task_id = self.create()
        task = self.service.store.get(task_id)
        directory = self.root / "finished-attempt"
        directory.mkdir()
        atomic_json(directory / "status.json", {"state": "completed", "current_steps": 10000,
                    "at": 1, "checkpoint": str(directory / "model_10000_steps.zip")})
        task.update(state="running", target_steps=20000)
        task["attempts"] = [{"id": "a", "directory": str(directory), "started": 0,
                              "identity": None, "log": str(directory / "train.log")}]
        self.service._refresh(task)
        self.assertEqual(task["state"], "queued")
        self.assertEqual(task["current_steps"], 10000)

    def test_committed_control_repaired_after_interrupted_write(self):
        task_id = self.create()
        task = self.service.store.get(task_id)
        self.service._control_path(task).unlink()
        with patch.object(self.service, "_start"):
            self.service.tick()
        restored = json.loads(self.service._control_path(task).read_text())
        self.assertEqual(restored["target_steps"], task["target_steps"])
        self.assertEqual(restored["desired_state"], "running")

    def test_pending_takeover_keeps_checkpoint_progress(self):
        task_id = self.create()
        task = self.service.store.get(task_id)
        log = self.root / "old.log"
        log.write_text("[训练轮次] 第1/100轮结束 本轮总耗时=50秒 训练步数=15095000/35000000\n", encoding="utf-8")
        task.update(mode="managed", state="queued", current_steps=15000000)
        task["attempts"] = [{"legacy": True, "log": str(log)}]
        self.service._refresh(task)
        self.assertEqual(task["current_steps"], 15000000)

    def test_goal_dialog_cannot_silently_use_changed_revision(self):
        task_id = self.create()
        before = self.service.store.get(task_id)["revision"]
        self.service.action(task_id, request(action="set_target", target_steps=21000000))
        with self.assertRaisesRegex(ValueError, "已变化"):
            self.service.action(task_id, request(action="set_target", target_steps=22000000, expected_revision=before))
        self.assertEqual(self.service.store.get(task_id)["target_steps"], 21000000)

    def test_api_rejects_cross_site_commands_and_arbitrary_parameters(self):
        with TestClient(create_app(self.service, background=False), base_url="http://localhost") as client:
            self.assertEqual(client.get("/").status_code, 200)
            self.assertEqual(client.post("/api/tasks", json=request(target_steps=20000000)).status_code, 403)
            token = client.get("/api/bootstrap").json()["csrf_token"]
            headers = {"X-CSRF-Token": token}
            body = request(target_steps=20000000)
            self.assertEqual(client.post("/api/tasks", json=body, headers={**headers, "Origin": "https://evil.example"}).status_code, 403)
            self.assertEqual(client.post("/api/tasks", json={**body, "command": "arbitrary-shell"}, headers=headers).status_code, 422)
            response = client.post("/api/tasks", json=body, headers=headers)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(client.post("/api/tasks", json=body, headers=headers).json(), response.json())
            self.assertEqual(len(client.get("/api/tasks").json()), 1)


if __name__ == "__main__":
    unittest.main()
