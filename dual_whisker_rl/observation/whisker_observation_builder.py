"""硬件可部署的触须观测构造器。

该模块把左右传感器读数（仿真的慢响应读数，或真机的 MQ-3 ADC）经过和硬件
完全一致的 `DualGasPreprocessor` 预处理，再拼接触须角度/扇区，构造一个
**只含硬件可获得特征**的观测向量。仿真环境和硬件 rollout 脚本都应使用它，
这样仿真训练时的观测和真机部署时构造的观测语义一致。

观测中刻意**不含**仿真特有的特权信息（真实风向、到气源的真实距离、瞬时真实
浓度），避免策略依赖真机上拿不到的量而在部署时失效。
"""

from __future__ import annotations

import math

import numpy as np

from dual_whisker_rl.hardware.sensor_preprocess import DualGasPreprocessor
from dual_whisker_rl.hardware.sensor_preprocess import SensorPreprocessConfig


class WhiskerObservationBuilder:
    """把左右传感器读数 + 触须角度/扇区构造成硬件可得的 12 维观测。

    内部持有一个有状态的 `DualGasPreprocessor`。每个 episode 开始时调用
    `reset()`；每一帧调用一次 `build()`（会推进一次预处理内部状态）。
    """

    # 12 维观测的有序字段名。前 8 维为左右预处理特征与差分，后 4 维为触须几何。
    FIELD_NAMES = [
        "left_norm_signal",
        "right_norm_signal",
        "left_norm_smooth",
        "right_norm_smooth",
        "left_norm_trend",
        "right_norm_trend",
        "norm_smooth_diff",
        "norm_trend_diff",
        "left_angle_norm",
        "right_angle_norm",
        "left_sector_norm",
        "right_sector_norm",
    ]

    def __init__(self, config: SensorPreprocessConfig, sector_count: int) -> None:
        if sector_count <= 1:
            raise ValueError("sector_count must be >= 2")
        self.config = config
        self.sector_count = int(sector_count)
        self._proc = DualGasPreprocessor(config)

    @property
    def dim(self) -> int:
        return len(self.FIELD_NAMES)

    @property
    def field_names(self) -> list[str]:
        return list(self.FIELD_NAMES)

    @property
    def low(self) -> np.ndarray:
        """观测各维下界，供 gym Box 使用。"""
        clip = self.config.clip
        return np.array(
            # 6 个归一化传感器特征 ±clip；2 个差分 ±2*clip
            [-clip] * 6 + [-2.0 * clip] * 2
            # 左角 [0,1]、右角 [-1,0]、左右扇区 [0,1]
            + [0.0, -1.0, 0.0, 0.0],
            dtype=np.float32,
        )

    @property
    def high(self) -> np.ndarray:
        clip = self.config.clip
        return np.array(
            [clip] * 6 + [2.0 * clip] * 2 + [1.0, 0.0, 1.0, 1.0],
            dtype=np.float32,
        )

    def reset(self, init_baseline: float | None = None) -> None:
        """重置内部预处理器。

        `init_baseline=None` 表示用第一帧读数锚定基线，与硬件端上电后在洁净
        空气中锚基线的行为一致，也对将来引入的每 episode 基线偏置随机化鲁棒。
        """
        self._proc.reset(init_baseline, init_baseline)

    def build(
        self,
        left_reading: float,
        right_reading: float,
        left_angle: float,
        right_angle: float,
        left_sector: int,
        right_sector: int,
        *,
        allow_baseline_update: bool = True,
    ) -> np.ndarray:
        """输入一帧读数与触须状态，返回 12 维观测。会推进一次预处理状态。"""
        feats = self._proc.update(
            left_reading,
            right_reading,
            allow_baseline_update=allow_baseline_update,
        )
        denom = self.sector_count - 1
        obs = np.array(
            [
                feats.left.norm_signal,
                feats.right.norm_signal,
                feats.left.norm_smooth,
                feats.right.norm_smooth,
                feats.left.norm_trend,
                feats.right.norm_trend,
                feats.norm_smooth_diff,
                feats.norm_trend_diff,
                left_angle / math.pi,
                right_angle / math.pi,
                float(left_sector) / denom,
                float(right_sector) / denom,
            ],
            dtype=np.float32,
        )
        return obs
