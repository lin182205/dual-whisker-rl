"""短训验证控制发生在梯度更新之后，并保留优化器状态。"""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from dual_whisker_rl.dashboard.files import atomic_json, read_json
from dual_whisker_rl.dashboard.training_control import ManagedSession

torch.set_num_threads(1)


class TinyEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (2,), dtype=np.float32)
    action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        self.steps += 1
        obs = np.array([self.steps / 16, action], dtype=np.float32)
        return obs, float(action), False, self.steps >= 8, {}


class CommandAtEight(BaseCallback):
    def __init__(self, path, desired="running", target=24):
        super().__init__()
        self.path, self.desired, self.target = path, desired, target

    def _on_step(self):
        if self.num_timesteps == 8:
            atomic_json(self.path, {"desired_state": self.desired, "target_steps": self.target, "revision": 2})
        return True


def model():
    return PPO("MlpPolicy", DummyVecEnv([TinyEnv]), n_steps=8, batch_size=8,
               n_epochs=1, learning_rate=.0001, seed=17, device="cpu")


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.control = self.root / "control.json"
        atomic_json(self.root / "spec.json", {"control_path": str(self.control)})

    def tearDown(self):
        self.temp.cleanup()

    def session(self, instance, target):
        atomic_json(self.control, {"desired_state": "running", "target_steps": target, "revision": 1})
        args = SimpleNamespace(managed_task_dir=self.root, timesteps=target-instance.num_timesteps,
                               resume_from=None if instance.num_timesteps == 0 else self.root / "resume.zip",
                               log_interval=1, checkpoint_dir=self.root / "checkpoints", experiment_name="test",
                               model_path=self.root / "model.zip")
        return ManagedSession(args, instance, SimpleNamespace())

    def test_uncontrolled_result_matches_standard_ppo(self):
        baseline = model()
        baseline.learn(24)
        expected = {k: v.clone() for k, v in baseline.policy.state_dict().items()}
        baseline.get_env().close()
        controlled = model()
        session = self.session(controlled, 24)
        self.assertEqual(session.learn(controlled, []), "completed")
        self.assertEqual(controlled._n_updates, 3)
        for key, actual in controlled.policy.state_dict().items():
            torch.testing.assert_close(actual, expected[key], rtol=0, atol=0)
        controlled.get_env().close()

    def test_extension_runs_past_original_learn_budget(self):
        instance = model()
        session = self.session(instance, 8)
        self.assertEqual(session.learn(instance, [CommandAtEight(self.control, target=24)]), "completed")
        self.assertEqual(instance.num_timesteps, 24)
        self.assertEqual(instance._n_updates, 3)
        instance.get_env().close()

    def test_reduction_saves_after_update_and_keeps_optimizer(self):
        instance = model()
        session = self.session(instance, 32)
        self.assertEqual(session.learn(instance, [CommandAtEight(self.control, target=8)]), "completed")
        self.assertEqual(instance.num_timesteps, 8)
        self.assertEqual(instance._n_updates, 1)
        restored = PPO.load(session.checkpoint, device="cpu")
        self.assertEqual(restored._n_updates, 1)
        expected = instance.policy.optimizer.state_dict()
        actual = restored.policy.optimizer.state_dict()
        for key, state in expected["state"].items():
            for field, value in state.items():
                torch.testing.assert_close(actual["state"][key][field], value)
        instance.get_env().close()

    def test_pause_resume_and_stop_keep_cumulative_steps(self):
        instance = model()
        session = self.session(instance, 32)
        self.assertEqual(session.learn(instance, [CommandAtEight(self.control, "paused", 32)]), "paused")
        instance.get_env().close()
        session.finish(instance)
        self.assertEqual(read_json(self.root / "status.json")["state"], "paused")
        restored = PPO.load(session.checkpoint, env=DummyVecEnv([TinyEnv]), device="cpu")
        continuation = self.session(restored, 32)
        # 恢复后的首个边界可处理停止请求，不新增未更新的样本。
        atomic_json(self.control, {"desired_state": "stopped", "target_steps": 32, "revision": 3})
        self.assertEqual(continuation.learn(restored, []), "stopped")
        self.assertEqual(restored.num_timesteps, 8)
        restored.get_env().close()


if __name__ == "__main__":
    unittest.main()
