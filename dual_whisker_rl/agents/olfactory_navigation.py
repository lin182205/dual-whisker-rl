"""嗅觉历史与当前帧导航的双分支特征，防止历史雷达拖慢避障反应。"""

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dual_whisker_rl.agents.gru_history import GRUHistoryExtractor
from dual_whisker_rl.agents.transformer_history import TransformerHistoryExtractor


OLFACTORY_INDICES = tuple(range(12))+(16,)
NAVIGATION_INDICES = (12, 13, 14, 15)+tuple(range(17, 57))


class OlfactoryNavigationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, *, history_length, base_observation_dim=57,
                 temporal_encoder="gru", encoder_kwargs=None):
        if base_observation_dim != 57 or history_length < 1 or observation_space.shape != (57*history_length,):
            raise ValueError("双分支编码器需要 mobile_hybrid_nav_v1 的57维历史")
        super().__init__(observation_space, features_dim=128)
        self.base_observation_dim = 57
        self.history_length = int(history_length)
        self.temporal_encoder = temporal_encoder
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.configuration = {"temporal_encoder": temporal_encoder, "encoder_kwargs": self.encoder_kwargs,
                              "olfactory_output_dim": 64, "navigation_layers": [64, 64], "features_dim": 128}
        self.register_buffer("olfactory_indices", torch.tensor(OLFACTORY_INDICES), persistent=False)
        self.register_buffer("navigation_indices", torch.tensor(NAVIGATION_INDICES), persistent=False)
        smell_space = gym.spaces.Box(float('-inf'), float('inf'), shape=(13*self.history_length,))
        if temporal_encoder == "mlp":
            self.olfactory = nn.Sequential(nn.Linear(13*self.history_length, 64), nn.GELU(),
                                           nn.Linear(64, 64), nn.GELU())
            smell_dim = 64
        elif temporal_encoder in ("gru", "transformer"):
            extractor = GRUHistoryExtractor if temporal_encoder == "gru" else TransformerHistoryExtractor
            self.olfactory = extractor(smell_space, history_length=self.history_length,
                                       base_observation_dim=13, **self.encoder_kwargs)
            smell_dim = self.olfactory.features_dim
        else:
            raise ValueError("未知嗅觉历史编码器")
        self.olfactory_projection = nn.Linear(smell_dim, 64) if smell_dim != 64 else nn.Identity()
        self.navigation = nn.Sequential(nn.Linear(44, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())

    def forward(self, observations):
        history = observations.reshape(-1, self.history_length, 57)
        smell = history.index_select(-1, self.olfactory_indices).flatten(1)
        navigation = history[:, -1].index_select(-1, self.navigation_indices)
        return torch.cat((self.olfactory_projection(self.olfactory(smell)), self.navigation(navigation)), dim=-1)

    def forward_with_attention(self, observations):
        """只在评估时读取嗅觉 Transformer 的注意力，导航分支不加入历史 token。"""
        if self.training or self.temporal_encoder != "transformer":
            raise RuntimeError("注意力读取需要评估模式的Transformer嗅觉分支")
        history = observations.reshape(-1, self.history_length, 57)
        smell = history.index_select(-1, self.olfactory_indices).flatten(1)
        navigation = history[:, -1].index_select(-1, self.navigation_indices)
        features, attention = self.olfactory.forward_with_attention(smell)
        return torch.cat((self.olfactory_projection(features), self.navigation(navigation)), dim=-1), attention
