"""气味、风向与运动历史，以及当前帧导航的双分支特征。"""

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from dual_whisker_rl.agents.gru_history import GRUHistoryExtractor
from dual_whisker_rl.agents.transformer_history import TransformerHistoryExtractor
from dual_whisker_rl.mobile_features import (
    CURRENT_NAVIGATION_INDICES, NAVIGATION_INDICES, OLFACTORY_INDICES, feature_layout_contract,
)


class OlfactoryNavigationExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, *, history_length, base_observation_dim=57,
                 temporal_encoder="gru", encoder_kwargs=None, feature_layout=None, history_ablation="none"):
        if base_observation_dim != 57 or history_length < 1 or observation_space.shape != (57*history_length,):
            raise ValueError("双分支编码器需要 mobile_hybrid_nav_v1 的57维历史")
        super().__init__(observation_space, features_dim=128)
        self.base_observation_dim = 57
        self.history_length = int(history_length)
        self.temporal_encoder = temporal_encoder
        self.feature_layout = feature_layout_contract(feature_layout, history_ablation)
        self.history_ablation = history_ablation
        self.encoder_kwargs = dict(encoder_kwargs or {})
        self.configuration = {"temporal_encoder": temporal_encoder, "encoder_kwargs": self.encoder_kwargs,
                              "olfactory_output_dim": 64, "navigation_layers": [64, 64], "features_dim": 128}
        navigation_indices = CURRENT_NAVIGATION_INDICES if self.feature_layout else NAVIGATION_INDICES
        history_width = self.feature_layout["history_width"] if self.feature_layout else len(OLFACTORY_INDICES)
        if self.feature_layout:
            self.configuration["feature_layout"] = self.feature_layout
        self.register_buffer("olfactory_indices", torch.tensor(OLFACTORY_INDICES), persistent=False)
        self.register_buffer("navigation_indices", torch.tensor(navigation_indices), persistent=False)
        history_size = history_width * self.history_length
        smell_space = gym.spaces.Box(float('-inf'), float('inf'), shape=(history_size,))
        if temporal_encoder == "mlp":
            self.olfactory = nn.Sequential(nn.Linear(history_size, 64), nn.GELU(),
                                           nn.Linear(64, 64), nn.GELU())
            smell_dim = 64
        elif temporal_encoder in ("gru", "transformer"):
            extractor = GRUHistoryExtractor if temporal_encoder == "gru" else TransformerHistoryExtractor
            self.olfactory = extractor(smell_space, history_length=self.history_length,
                                       base_observation_dim=history_width, **self.encoder_kwargs)
            smell_dim = self.olfactory.features_dim
        else:
            raise ValueError("未知嗅觉历史编码器")
        self.olfactory_projection = nn.Linear(smell_dim, 64) if smell_dim != 64 else nn.Identity()
        self.navigation = nn.Sequential(nn.Linear(len(navigation_indices), 64), nn.GELU(), nn.Linear(64, 64), nn.GELU())

    def _branch_inputs(self, observations):
        history = observations.reshape(-1, self.history_length, self.base_observation_dim)
        temporal = history.index_select(-1, self.olfactory_indices)
        navigation = history[:, -1].index_select(-1, self.navigation_indices)
        if self.feature_layout is None:
            return temporal.flatten(1), navigation

        # 用余弦和正弦直接计算旋转，跨越 ±π 时无需对角度解包。
        cos_i, sin_i = history[..., 12:14].unbind(-1)
        cos_t, sin_t = cos_i[:, -1:], sin_i[:, -1:]
        cos_delta = cos_i * cos_t + sin_i * sin_t
        sin_delta = sin_i * cos_t - cos_i * sin_t
        cos_delta = torch.where((cos_i == cos_t) & (sin_i == sin_t), 1.0, cos_delta)
        wind = history[..., 19:21]
        if self.history_ablation == "current_wind_only":
            aligned_wind = wind[:, -1:].expand_as(wind)
        else:
            cos_wind, sin_wind = wind.unbind(-1)
            aligned_wind = torch.stack((cos_wind * cos_delta - sin_wind * sin_delta,
                                        sin_wind * cos_delta + cos_wind * sin_delta), dim=-1)

        # 历史位置相对当前位置，投影到当前车身前向和左向。
        position = history[..., 17:19]
        dx, dy = ((position - position[:, -1:]) * 5.0).unbind(-1)
        relative_pose = torch.stack((dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t,
                                     cos_delta, sin_delta), dim=-1)
        if self.history_ablation == "no_motion_history":
            relative_pose = relative_pose.new_tensor([0., 0., 1., 0.]).expand_as(relative_pose)
        if self.history_ablation == "no_absolute_xy":
            # 只屏蔽当前导航的地图坐标。上面的相对位移仍取原始位置。
            navigation = torch.cat((navigation[:, :4], torch.zeros_like(navigation[:, 4:6]),
                                    navigation[:, 6:]), dim=-1)
        return torch.cat((temporal, aligned_wind, relative_pose), dim=-1).flatten(1), navigation

    def forward(self, observations):
        smell, navigation = self._branch_inputs(observations)
        return torch.cat((self.olfactory_projection(self.olfactory(smell)), self.navigation(navigation)), dim=-1)

    def forward_with_attention(self, observations):
        """只在评估时读取嗅觉 Transformer 的注意力，导航分支不加入历史 token。"""
        if self.training or self.temporal_encoder != "transformer":
            raise RuntimeError("注意力读取需要评估模式的Transformer嗅觉分支")
        smell, navigation = self._branch_inputs(observations)
        features, attention = self.olfactory.forward_with_attention(smell)
        return torch.cat((self.olfactory_projection(features), self.navigation(navigation)), dim=-1), attention
