"""GRU feature extractor for fixed-length observation histories."""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GRUHistoryExtractor(BaseFeaturesExtractor):
    """用单层（可多层）GRU 编码扁平化的观测历史，再送入 PPO 策略/价值头。

    与 ``TransformerHistoryExtractor`` 接口保持一致、可直接互换：环境侧观测仍是
    扁平 ``Box``（兼容 SB3 向量环境），内部把它 reshape 回时间维再过 GRU。

    相比 Transformer，GRU 的递归结构天然编码时序顺序、参数更少，更契合
    MQ-3 这类慢响应、强自相关的低维传感信号，在 PPO 小样本下更稳。
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        history_length: int,
        base_observation_dim: int,
        hidden_size: int = 64,
        n_layers: int = 1,
        dropout: float = 0.0,
        features_dim: int = 64,
    ) -> None:
        self._validate_hyperparameters(
            history_length=history_length,
            base_observation_dim=base_observation_dim,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
            features_dim=features_dim,
        )
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("GRUHistoryExtractor requires a Box observation space")
        if len(observation_space.shape) != 1:
            raise ValueError("GRUHistoryExtractor requires flat vector observations")

        expected_dim = int(history_length) * int(base_observation_dim)
        actual_dim = int(observation_space.shape[0])
        if actual_dim != expected_dim:
            raise ValueError(
                "Flattened observation dimension does not match history configuration: "
                f"got {actual_dim}, expected {history_length} * "
                f"{base_observation_dim} = {expected_dim}"
            )

        super().__init__(observation_space, features_dim=int(features_dim))
        self.history_length = int(history_length)
        self.base_observation_dim = int(base_observation_dim)
        self.hidden_size = int(hidden_size)
        self.n_layers = int(n_layers)

        # 单层 GRU（n_layers==1）时 PyTorch 的 dropout 不生效，显式置 0 以消除警告。
        gru_dropout = float(dropout) if self.n_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.base_observation_dim,
            hidden_size=self.hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.output_projection = nn.Sequential(
            nn.Linear(self.hidden_size, int(features_dim)),
            nn.GELU(),
        )

    @staticmethod
    def _validate_hyperparameters(
        *,
        history_length: int,
        base_observation_dim: int,
        hidden_size: int,
        n_layers: int,
        dropout: float,
        features_dim: int,
    ) -> None:
        positive_values = {
            "history_length": history_length,
            "base_observation_dim": base_observation_dim,
            "hidden_size": hidden_size,
            "n_layers": n_layers,
            "features_dim": features_dim,
        }
        for name, value in positive_values.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be at least 1")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sequence = self._reshape_history(observations)
        # h_n: [n_layers, batch, hidden_size]；取最后一层最后时刻隐状态作为整段历史摘要。
        _, h_n = self.gru(sequence)
        return self.output_projection(h_n[-1])

    def _reshape_history(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError(
                "GRUHistoryExtractor expects observations shaped "
                "[batch, history_length * base_observation_dim]"
            )
        batch_size = observations.shape[0]
        return observations.reshape(
            batch_size,
            self.history_length,
            self.base_observation_dim,
        )
