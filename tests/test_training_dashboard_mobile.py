"""真实 Mobile 双子进程短训，覆盖保存、服务恢复和检查点续训。"""
from pathlib import Path
import tempfile
import time
import unittest
import uuid

import psutil

from dual_whisker_rl.dashboard.files import alive, read_json
from dual_whisker_rl.dashboard.service import Service


class MobileManagedSmoke(unittest.TestCase):
    def test_mobile_pause_service_restart_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            service = Service(temp, require_gpu=False)
            spawned = []
            try:
                task_id = service.create({"request_id": uuid.uuid4().hex, "target_steps": 10000, "seed": 11})["task_id"]
                task = service.store.get(task_id)
                task.update(n_envs=2, n_steps=16, batch_size=16, target_steps=5120)
                task["config"]["max_steps"] = 32
                task["config"]["plume_overrides"].update(puff_warmup_steps=0, max_puffs=20)
                service._write_control(task, "running")
                service.store.save(task)
                service.tick()
                spawned.append(service.store.get(task_id)["attempts"][-1]["identity"])
                self.wait(service, task_id, lambda t: t["current_steps"] >= 32, 90)
                service.action(task_id, {"request_id": uuid.uuid4().hex, "action": "pause"})
                task = self.wait(service, task_id, lambda t: t["state"] == "paused", 30)
                checkpoint = task["resume_checkpoint"]
                paused_steps = task["current_steps"]
                self.assertTrue(Path(checkpoint).is_file())
                metadata = read_json(task["attempts"][-1]["metadata"])
                self.assertFalse(metadata["training_complete"])
                self.assertEqual(metadata["managed_outcome"], "paused")
                # 服务重建后同一任务仍可恢复，不依赖原 Popen 对象。
                service.store.close()
                service = Service(temp, require_gpu=False)
                service.action(task_id, {"request_id": uuid.uuid4().hex, "action": "set_target", "target_steps": paused_steps + 64})
                service.action(task_id, {"request_id": uuid.uuid4().hex, "action": "resume"})
                service.tick()
                spawned.append(service.store.get(task_id)["attempts"][-1]["identity"])
                task = self.wait(service, task_id, lambda t: t["state"] == "completed", 90)
                self.assertEqual(task["current_steps"], paused_steps + 64)
                self.assertEqual(len(task["attempts"]), 2)
                meta2 = read_json(task["attempts"][-1]["metadata"])
                self.assertEqual(meta2["starting_num_timesteps"], paused_steps)
                self.assertTrue(meta2["training_complete"])
                self.assertEqual(meta2["ppo"], metadata["ppo"])
            finally:
                for who in spawned:
                    try:
                        if not alive(who):
                            continue
                        parent = psutil.Process(who["pid"])
                        processes = parent.children(recursive=True) + [parent]
                        for process in processes:
                            process.terminate()
                        psutil.wait_procs(processes, timeout=10)
                    except psutil.Error:
                        pass
                service.store.close()

    def wait(self, service, task_id, condition, timeout):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            service.tick()
            task = service.store.get(task_id)
            if task["state"] == "failed":
                self.fail(service.logs(task_id)["text"])
            if condition(task):
                return task
            time.sleep(.15)
        self.fail("训练控制验收超时\n" + service.logs(task_id)["text"])


if __name__ == "__main__":
    unittest.main()
