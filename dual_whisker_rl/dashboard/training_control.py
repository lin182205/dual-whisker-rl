"""在完整 PPO 更新之后响应控制，不丢弃正在收集的 rollout。"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time
import uuid

from stable_baselines3.common.callbacks import BaseCallback, CallbackList

from .files import atomic_json, identity, read_json


class BoundaryRequest(Exception):
    def __init__(self, reason):
        self.reason = reason


class BoundaryCallback(BaseCallback):
    def __init__(self, session):
        super().__init__()
        self.session = session

    def _on_rollout_start(self):
        # SB3 在下一轮采样前调用这里；上一轮的 train() 已完整结束。
        self.session.check_boundary(self.model)

    def _on_step(self):
        return True


def atomic_model(model, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.stem + "." + uuid.uuid4().hex + ".zip")
    try:
        model.save(str(tmp))
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


class ManagedSession:
    def __init__(self, args, model, eta):
        self.directory = Path(args.managed_task_dir)
        spec = read_json(self.directory / "spec.json")
        if not spec:
            raise ValueError("缺少管理任务 spec.json")
        self.control_path = Path(spec["control_path"])
        self.args, self.eta = args, eta
        self.initial_steps = int(model.num_timesteps)
        self.rollout_size = model.n_steps * model.n_envs
        self.target = self.initial_steps + args.timesteps
        self.callback = BoundaryCallback(self)
        self.reason = None
        self.checkpoint = None

    def control(self):
        control = read_json(self.control_path)
        if not control or control.get("desired_state") not in {"running", "paused", "stopped"}:
            raise RuntimeError("任务控制文件缺失或损坏，已停止继续采样")
        target = control.get("target_steps")
        if not isinstance(target, int) or target < 1:
            raise RuntimeError("任务目标步数无效")
        self.target = self.initial_steps + math.ceil((target - self.initial_steps) / self.rollout_size) * self.rollout_size
        self.args.timesteps = self.target - self.initial_steps
        self.eta.target_timesteps = self.target
        self.eta.total_rollouts = max(1, math.ceil((self.target - self.initial_steps) / self.rollout_size))
        return control

    def publish(self, model, state, **extra):
        atomic_json(self.directory / "status.json", {"state": state, "current_steps": int(model.num_timesteps),
                    "target_steps": self.target, "at": time.time(), "identity": identity(os.getpid()),
                    "checkpoint": self.checkpoint, **extra})

    def requested_reason(self, model, control):
        if control["desired_state"] in {"paused", "stopped"}:
            return control["desired_state"]
        if model.num_timesteps >= self.target:
            return "completed"
        return None

    def check_boundary(self, model):
        control = self.control()
        self.publish(model, "running", acknowledged_revision=control["revision"])
        reason = self.requested_reason(model, control)
        if reason:
            raise BoundaryRequest(reason)

    def learn(self, model, callbacks):
        # 此回调必须排第一位；受控中断后显式关闭其余回调，估时不会多计一轮。
        callbacks = CallbackList([self.callback, *callbacks])
        first = self.args.resume_from is None
        while True:
            control = self.control()
            reason = self.requested_reason(model, control)
            if reason:
                self.reason = reason
                break
            try:
                model.learn(total_timesteps=self.target - model.num_timesteps,
                            progress_bar=False, callback=callbacks, log_interval=self.args.log_interval,
                            reset_num_timesteps=first)
                first = False
            except BoundaryRequest as request:
                callbacks.on_training_end()
                self.reason = request.reason
                break
            # learn 的局部预算不可热改；延长目标后，在原段结束时保留同一模型、环境
            # 与优化器进入下一段，避免复制 SB3 内部训练循环。
        final = Path(self.args.checkpoint_dir) / f"{self.args.experiment_name}_{model.num_timesteps}_steps.zip"
        atomic_model(model, final)
        atomic_model(model, self.args.model_path)
        self.checkpoint = str(final)
        return self.reason

    def finish(self, model):
        self.publish(model, self.reason)
