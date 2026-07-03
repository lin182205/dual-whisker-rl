"""Slow-response gas sensor model."""

from __future__ import annotations

import math

import numpy as np


class FirstOrderGasSensor:
    """First-order lag model: y_t = alpha * y_{t-1} + (1 - alpha) * c_t."""

    def __init__(self, alpha: float = 0.95, initial_value: float = 0.0) -> None:
        if not 0.0 <= alpha < 1.0:
            raise ValueError("alpha must be in [0, 1).")
        self.alpha = float(alpha)
        self.value = float(initial_value)

    def reset(self, value: float = 0.0) -> None:
        self.value = float(value)

    def update(self, concentration: float) -> float:
        self.value = self.alpha * self.value + (1.0 - self.alpha) * float(concentration)
        return self.value


class AsymmetricGasSensor:
    """更贴近真实 MQ-3 的气体传感器模型。

    相比对称一阶模型，加入了三个真实特性：

    1. 响应/恢复非对称：浓度上升时用较快的 ``response_tau``，下降时用较慢的
       ``recovery_tau``（MQ-3 吸附快、解吸慢）。时间常数经 ``dt`` 换算为
       一阶系数 ``alpha = exp(-dt / tau)``。
    2. 基线漂移：输出叠加一个缓慢 OU 漂移，模拟 MQ-3 上电后基线缓慢变化。
    3. 测量噪声：输出叠加高斯噪声。

    ``value`` 是智能体实际观测到的读数（含漂移和噪声）；内部干净状态保存在
    ``_state``，便于调试。把漂移/噪声放进仿真传感器，是为了让仿真观测和真实
    MQ-3 的"脏"数据同分布，从而让预处理层和策略在 sim-to-real 时不至于失效。
    """

    def __init__(
        self,
        response_tau: float = 0.6,
        recovery_tau: float = 3.0,
        dt: float = 0.2,
        noise_std: float = 0.0,
        baseline: float = 0.0,
        baseline_drift_std: float = 0.0,
        baseline_tau: float = 60.0,
        rng: np.random.Generator | None = None,
        initial_value: float = 0.0,
    ) -> None:
        self.response_tau = float(response_tau)
        self.recovery_tau = float(recovery_tau)
        self.dt = float(dt)
        self.noise_std = float(noise_std)
        self.baseline0 = float(baseline)
        self.baseline_drift_std = float(baseline_drift_std)
        self.baseline_tau = float(baseline_tau)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._state = float(initial_value)
        self.baseline = float(baseline)
        self.value = float(initial_value) + self.baseline

    def reset(self, value: float = 0.0) -> None:
        self._state = float(value)
        self.baseline = self.baseline0
        self.value = self._state + self.baseline

    def _alpha(self, tau: float) -> float:
        if tau <= 0.0:
            return 0.0
        return math.exp(-self.dt / tau)

    def update(self, concentration: float) -> float:
        target = float(concentration)
        tau = self.response_tau if target >= self._state else self.recovery_tau
        alpha = self._alpha(tau)
        self._state = alpha * self._state + (1.0 - alpha) * target

        if self.baseline_drift_std > 0.0:
            # 基线走 OU：均值回归到 baseline0，相关时间 ~ baseline_tau。
            theta = self.dt / max(self.baseline_tau, 1e-6)
            self.baseline += -theta * (self.baseline - self.baseline0) + (
                self.baseline_drift_std * math.sqrt(self.dt) * float(self.rng.normal())
            )

        reading = self._state + self.baseline
        if self.noise_std > 0.0:
            reading += float(self.rng.normal(0.0, self.noise_std))
        self.value = reading
        return self.value
