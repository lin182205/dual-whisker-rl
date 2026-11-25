"""Mobile 历史拼接：旧版堆叠单帧，空间记忆只拼接当前地图。"""

from collections import deque

import gymnasium as gym
import numpy as np

from dual_whisker_rl.mobile_features import WIND_MOTION_SPATIAL_MEMORY


def has_spatial_memory(env):
    """按显式布局判断，禁止用维度整除猜测空间记忆版本。"""
    return getattr(env.unwrapped, "effective_config", {}).get("feature_layout") == WIND_MOTION_SPATIAL_MEMORY


def observation_dimensions(env, history_length):
    """返回核心、当前尾部、实际打包维度；兼容原16/57维环境。"""
    if int(history_length) < 1:
        raise ValueError("history_length 必须至少为1")
    raw = int(env.observation_space.shape[0])
    core, tail = (57, 802) if has_spatial_memory(env) else (raw, 0)
    if raw != core + tail:
        raise ValueError("环境单帧维度与显式空间记忆布局不一致")
    return core, tail, core * int(history_length) + tail


class ObservationAssembler:
    """训练包装和可视化共用的无随机历史缓冲。"""

    def __init__(self, env, history_length):
        self.history_length = int(history_length)
        self.core_dim, self.tail_dim, self.packed_dim = observation_dimensions(env, history_length)
        self.history = deque(maxlen=self.history_length)
        self.current_tail = np.empty(self.tail_dim, dtype=np.float32)

    def reset(self, observation):
        self.history.clear()
        for _ in range(self.history_length):
            self.history.append(np.asarray(observation[:self.core_dim], dtype=np.float32).copy())
        self.current_tail = np.asarray(observation[self.core_dim:], dtype=np.float32).copy()
        return self.as_array()

    def append(self, observation):
        if not self.history:
            return self.reset(observation)
        self.history.append(np.asarray(observation[:self.core_dim], dtype=np.float32).copy())
        self.current_tail = np.asarray(observation[self.core_dim:], dtype=np.float32).copy()
        return self.as_array()

    def as_array(self):
        if len(self.history) != self.history_length:
            raise RuntimeError("请先重置历史缓冲")
        return np.concatenate((*self.history, self.current_tail)).astype(np.float32, copy=False)


class MobileObservationHistory(gym.Wrapper):
    """保持历史公开接口，v2 仅堆叠57维核心，地图不随历史重复。"""

    def __init__(self, env, history_length=20):
        super().__init__(env)
        self.history_length = int(history_length)
        self.assembler = ObservationAssembler(env, history_length)
        self.base_observation_dim = self.assembler.core_dim
        self.history = self.assembler.history
        core = self.base_observation_dim
        low = np.concatenate((np.tile(env.observation_space.low[:core], self.history_length),
                              env.observation_space.low[core:]))
        high = np.concatenate((np.tile(env.observation_space.high[:core], self.history_length),
                               env.observation_space.high[core:]))
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self.assembler.reset(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self.assembler.append(observation), reward, terminated, truncated, info

    def _stack(self):
        return self.assembler.as_array()
