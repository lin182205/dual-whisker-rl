"""Transformer feature extractor for fixed-length observation histories."""

from __future__ import annotations

import gymnasium as gym
import torch
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class TransformerHistoryExtractor(BaseFeaturesExtractor):
    """Encode a flattened observation history before PPO policy/value heads.

    The environment-facing observation remains a flat ``Box`` so it stays
    compatible with Stable-Baselines3 vector environments. Internally, the
    extractor restores the time dimension, embeds each frame, and summarizes
    the complete history with a learnable CLS token.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        *,
        history_length: int,
        base_observation_dim: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        features_dim: int = 64,
    ) -> None:
        self._validate_hyperparameters(
            history_length=history_length,
            base_observation_dim=base_observation_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            features_dim=features_dim,
        )
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("TransformerHistoryExtractor requires a Box observation space")
        if len(observation_space.shape) != 1:
            raise ValueError("TransformerHistoryExtractor requires flat vector observations")

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
        self.d_model = int(d_model)

        self.frame_projection = nn.Sequential(
            nn.Linear(self.base_observation_dim, self.d_model),
            nn.GELU(),
            nn.LayerNorm(self.d_model),
        )
        self.cls_token = nn.Parameter(torch.empty(1, 1, self.d_model))
        self.position_embedding = nn.Parameter(
            torch.empty(1, self.history_length + 1, self.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(dim_feedforward),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=int(n_layers),
            norm=nn.LayerNorm(self.d_model),
        )
        self.output_projection = nn.Linear(self.d_model, int(features_dim))

        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)

    @staticmethod
    def _validate_hyperparameters(
        *,
        history_length: int,
        base_observation_dim: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        dim_feedforward: int,
        dropout: float,
        features_dim: int,
    ) -> None:
        positive_values = {
            "history_length": history_length,
            "base_observation_dim": base_observation_dim,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "dim_feedforward": dim_feedforward,
            "features_dim": features_dim,
        }
        for name, value in positive_values.items():
            if int(value) < 1:
                raise ValueError(f"{name} must be at least 1")
        if int(d_model) % int(n_heads) != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must satisfy 0 <= dropout < 1")

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        sequence = self._embed_history(observations)
        encoded = self.encoder(sequence)
        return self.output_projection(encoded[:, 0])

    def forward_with_attention(
        self,
        observations: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return features and per-layer/head CLS attention for offline analysis.

        The attention tensor is shaped ``[batch, layers, heads, history + 1]``.
        Its last dimension contains the CLS key first, followed by the history
        tokens from oldest to newest. This method deliberately requires eval
        mode so the extra attention pass cannot change dropout randomness.
        """
        if self.training:
            raise RuntimeError(
                "forward_with_attention is for evaluation only; call eval() first"
            )

        sequence = self._embed_history(observations)
        cls_attention: list[torch.Tensor] = []
        for layer in self.encoder.layers:
            attention_input = layer.norm1(sequence) if layer.norm_first else sequence
            _, weights = layer.self_attn(
                attention_input,
                attention_input,
                attention_input,
                need_weights=True,
                average_attn_weights=False,
            )
            if weights is None:
                raise RuntimeError("Transformer self-attention did not return weights")
            cls_attention.append(weights[:, :, 0, :])
            sequence = layer(sequence)

        if self.encoder.norm is not None:
            sequence = self.encoder.norm(sequence)
        features = self.output_projection(sequence[:, 0])
        return features, torch.stack(cls_attention, dim=1)

    def _embed_history(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 2:
            raise ValueError(
                "TransformerHistoryExtractor expects observations shaped "
                "[batch, history_length * base_observation_dim]"
            )

        batch_size = observations.shape[0]
        sequence = observations.reshape(
            batch_size,
            self.history_length,
            self.base_observation_dim,
        )
        sequence = self.frame_projection(sequence)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        sequence = torch.cat((cls_tokens, sequence), dim=1)
        return sequence + self.position_embedding
