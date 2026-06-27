"""MQ-3 气体传感器在线预处理工具。

该模块的目标不是把 MQ-3 标定成精确浓度计，而是把漂移明显、慢响应、
有噪声的原始 ADC 值转换成强化学习策略更容易使用的相对特征。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math


@dataclass
class SensorPreprocessConfig:
    """单个气体传感器预处理参数。

    参数含义：
    - `dt_s`：相邻两次传感器读数的时间间隔。
    - `baseline_tau_s`：基线更新的时间常数，越大表示基线变化越慢。
    - `smooth_tau_s`：EMA 平滑时间常数，用于降低 ADC 噪声。
    - `response_tau_s`：一阶传感器响应时间常数，用于构造响应反推特征。
    - `trend_window`：计算短时间趋势时使用的历史窗口长度。
    - `scale`：归一化尺度，通常取一次气味刺激带来的 ADC 变化量级。
    - `clip`：归一化后裁剪范围，避免异常值破坏 PPO 输入尺度。
    """

    dt_s: float = 0.8
    baseline_tau_s: float = 180.0
    smooth_tau_s: float = 2.0
    response_tau_s: float = 2.0
    trend_window: int = 8
    scale: float = 200.0
    clip: float = 5.0
    baseline_update_signal_threshold: float = 0.20
    baseline_update_trend_threshold: float = 0.05


@dataclass
class SensorFeatures:
    """单个传感器一次更新后的特征。"""

    raw: float
    baseline: float
    signal: float
    smooth: float
    trend: float
    estimated_input: float
    norm_signal: float
    norm_smooth: float
    norm_trend: float
    norm_estimated_input: float
    baseline_updated: bool

    def as_policy_vector(self) -> list[float]:
        """返回适合拼接进 PPO 状态的核心特征。"""
        return [
            self.norm_signal,
            self.norm_smooth,
            self.norm_trend,
            self.norm_estimated_input,
        ]


@dataclass
class DualSensorFeatures:
    """左右两个传感器预处理后的组合特征。"""

    left: SensorFeatures
    right: SensorFeatures
    norm_signal_diff: float
    norm_smooth_diff: float
    norm_trend_diff: float
    norm_estimated_input_diff: float

    def as_policy_vector(self) -> list[float]:
        """返回一个硬件触须 PPO 可用的紧凑观测向量。"""
        return (
            self.left.as_policy_vector()
            + self.right.as_policy_vector()
            + [
                self.norm_signal_diff,
                self.norm_smooth_diff,
                self.norm_trend_diff,
                self.norm_estimated_input_diff,
            ]
        )


class OnlineGasPreprocessor:
    """单个 MQ-3 传感器的在线预处理器。

    使用方式：

    ```python
    proc = OnlineGasPreprocessor()
    features = proc.update(raw_adc)
    ```

    每次调用 `update()` 都会更新内部状态。该类适合在实时串口循环中长期运行。
    """

    def __init__(
        self,
        config: SensorPreprocessConfig | None = None,
        init_baseline: float | None = None,
    ) -> None:
        self.config = config or SensorPreprocessConfig()
        if self.config.trend_window < 2:
            raise ValueError("trend_window must be at least 2")
        if self.config.dt_s <= 0.0:
            raise ValueError("dt_s must be positive")
        if self.config.scale <= 0.0:
            raise ValueError("scale must be positive")

        self.baseline = float(init_baseline) if init_baseline is not None else None
        self.smooth = 0.0
        self.prev_smooth = 0.0
        self.smooth_history: deque[float] = deque(maxlen=self.config.trend_window)

    def reset(self, init_baseline: float | None = None) -> None:
        """重置预处理器状态。

        如果已经在洁净空气中得到一个基线值，可以通过 `init_baseline` 传入。
        如果不传，下一次 `update()` 会用第一帧 raw ADC 初始化基线。
        """
        self.baseline = float(init_baseline) if init_baseline is not None else None
        self.smooth = 0.0
        self.prev_smooth = 0.0
        self.smooth_history.clear()

    def update(
        self,
        raw_adc: float,
        *,
        allow_baseline_update: bool = True,
    ) -> SensorFeatures:
        """输入一帧原始 ADC，返回预处理特征。

        `allow_baseline_update=False` 可用于明确处于气体刺激阶段时冻结基线，
        防止真实气味响应被缓慢更新的 baseline 抵消。
        """
        raw = float(raw_adc)
        if self.baseline is None:
            self.baseline = raw

        signal = raw - self.baseline

        # EMA 平滑：把噪声较强的 ADC 相对信号变成更平稳的传感器特征。
        smooth_alpha = self._alpha(self.config.smooth_tau_s)
        self.prev_smooth = self.smooth
        self.smooth = smooth_alpha * self.smooth + (1.0 - smooth_alpha) * signal

        self.smooth_history.append(self.smooth)
        trend = self._trend_per_step()

        # 一阶响应反推特征：近似估计当前刺激输入。
        # 该特征会放大噪声，因此这里基于平滑后的 smooth 计算。
        response_alpha = self._alpha(self.config.response_tau_s)
        if 1.0 - response_alpha > 1e-9:
            estimated_input = (
                self.smooth - response_alpha * self.prev_smooth
            ) / (1.0 - response_alpha)
        else:
            estimated_input = self.smooth

        baseline_updated = False
        if allow_baseline_update and self._can_update_baseline(trend):
            baseline_alpha = self._alpha(self.config.baseline_tau_s)
            self.baseline = baseline_alpha * self.baseline + (1.0 - baseline_alpha) * raw
            baseline_updated = True

        return SensorFeatures(
            raw=raw,
            baseline=float(self.baseline),
            signal=float(signal),
            smooth=float(self.smooth),
            trend=float(trend),
            estimated_input=float(estimated_input),
            norm_signal=self.normalize(signal),
            norm_smooth=self.normalize(self.smooth),
            norm_trend=self.normalize(trend),
            norm_estimated_input=self.normalize(estimated_input),
            baseline_updated=baseline_updated,
        )

    def normalize(self, value: float) -> float:
        """把 ADC 量级特征压到稳定范围，便于神经网络训练。"""
        normalized = float(value) / self.config.scale
        return float(max(-self.config.clip, min(self.config.clip, normalized)))

    def _trend_per_step(self) -> float:
        """计算最近窗口内的平均每步变化量。"""
        if len(self.smooth_history) < 2:
            return 0.0
        first = self.smooth_history[0]
        last = self.smooth_history[-1]
        return float((last - first) / (len(self.smooth_history) - 1))

    def _can_update_baseline(self, trend: float) -> bool:
        """判断当前是否适合缓慢更新基线。

        只有当平滑信号较小、趋势也较小时才更新 baseline。这样可以降低
        长时间洁净空气漂移的影响，同时避免气体刺激阶段把真实响应吃掉。
        """
        norm_smooth = abs(self.normalize(self.smooth))
        norm_trend = abs(self.normalize(trend))
        return (
            norm_smooth <= self.config.baseline_update_signal_threshold
            and norm_trend <= self.config.baseline_update_trend_threshold
        )

    def _alpha(self, tau_s: float) -> float:
        """把时间常数转换为离散 EMA 系数。"""
        tau_s = max(float(tau_s), 1e-9)
        return float(math.exp(-self.config.dt_s / tau_s))


class DualGasPreprocessor:
    """左右 MQ-3 传感器的成对预处理器。"""

    def __init__(
        self,
        config: SensorPreprocessConfig | None = None,
        left_init_baseline: float | None = None,
        right_init_baseline: float | None = None,
    ) -> None:
        self.config = config or SensorPreprocessConfig()
        self.left = OnlineGasPreprocessor(
            self.config,
            init_baseline=left_init_baseline,
        )
        self.right = OnlineGasPreprocessor(
            self.config,
            init_baseline=right_init_baseline,
        )

    def reset(
        self,
        left_init_baseline: float | None = None,
        right_init_baseline: float | None = None,
    ) -> None:
        """同时重置左右两个传感器预处理器。"""
        self.left.reset(left_init_baseline)
        self.right.reset(right_init_baseline)

    def update(
        self,
        left_adc: float,
        right_adc: float,
        *,
        allow_baseline_update: bool = True,
    ) -> DualSensorFeatures:
        """输入左右原始 ADC，返回左右特征和差分特征。"""
        left_features = self.left.update(
            left_adc,
            allow_baseline_update=allow_baseline_update,
        )
        right_features = self.right.update(
            right_adc,
            allow_baseline_update=allow_baseline_update,
        )
        return DualSensorFeatures(
            left=left_features,
            right=right_features,
            norm_signal_diff=left_features.norm_signal - right_features.norm_signal,
            norm_smooth_diff=left_features.norm_smooth - right_features.norm_smooth,
            norm_trend_diff=left_features.norm_trend - right_features.norm_trend,
            norm_estimated_input_diff=(
                left_features.norm_estimated_input
                - right_features.norm_estimated_input
            ),
        )
