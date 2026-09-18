"""Stable-Baselines3 向量环境后端的共享创建工具。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Literal

import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


VecEnvBackend = Literal["auto", "dummy", "subproc"]
ResolvedVecEnvBackend = Literal["dummy", "subproc"]
SUBPROC_START_METHOD = "spawn"


def resolve_vec_env_backend(
    backend: VecEnvBackend,
    n_envs: int,
) -> ResolvedVecEnvBackend:
    """解析实际后端；多环境默认使用真正的独立进程。"""
    if n_envs < 1:
        raise ValueError("n_envs must be at least 1")
    if backend == "auto":
        return "subproc" if n_envs > 1 else "dummy"
    if backend not in ("dummy", "subproc"):
        raise ValueError(f"unsupported vec env backend: {backend}")
    return backend


def make_vec_env(
    env_fns: Sequence[Callable[[], gym.Env]],
    backend: ResolvedVecEnvBackend,
) -> VecEnv:
    """按已解析后端创建 VecEnv；子进程统一使用跨平台的 spawn。"""
    if not env_fns:
        raise ValueError("env_fns must not be empty")
    if backend == "subproc":
        return SubprocVecEnv(list(env_fns), start_method=SUBPROC_START_METHOD)
    if backend == "dummy":
        return DummyVecEnv(list(env_fns))
    raise ValueError(f"unsupported resolved vec env backend: {backend}")
