"""PPO 联合分布：二维连续底盘 + 两个真正的离散触须分类头。"""

from functools import partial

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from stable_baselines3.common.distributions import Distribution
from stable_baselines3.common.policies import ActorCriticPolicy

from dual_whisker_rl.envs.hybrid_action import HybridMotionWhiskerSpace


class HybridActionDistribution(Distribution):
    def __init__(self, sector_count=10, categorical_branches=2):
        super().__init__()
        self.sector_count = int(sector_count)
        self.categorical_branches = int(categorical_branches)

    def proba_distribution_net(self, latent_dim, log_std_init=0.):
        return nn.Linear(latent_dim, 2+self.categorical_branches*self.sector_count), nn.Parameter(torch.full((2,), float(log_std_init)))

    def proba_distribution(self, parameters, log_std):
        self.motion = Normal(parameters[..., :2], log_std.exp().expand_as(parameters[..., :2]))
        self.categories = [Categorical(logits=parameters[..., 2+i*self.sector_count:2+(i+1)*self.sector_count])
                           for i in range(self.categorical_branches)]
        if self.categories: self.left = self.categories[0]
        if len(self.categories)>1: self.right = self.categories[1]
        self.distribution = [self.motion]+self.categories
        return self

    def log_prob(self, actions):
        sectors = actions[..., 2:]
        if not torch.all(sectors == sectors.round()):
            raise ValueError("分类分支的动作必须是整数，不能对高斯样本取整")
        result = self.motion.log_prob(actions[..., :2]).sum(dim=-1)
        for i, category in enumerate(self.categories): result = result+category.log_prob(sectors[..., i].long())
        return result

    def entropy(self):
        return sum((c.entropy() for c in self.categories), self.motion.entropy().sum(dim=-1))

    def sample(self):
        return torch.cat([self.motion.rsample()]+[c.sample()[..., None] for c in self.categories], dim=-1)

    def mode(self):
        return torch.cat([self.motion.mean]+[c.probs.argmax(-1)[..., None] for c in self.categories], dim=-1)

    def actions_from_params(self, parameters, log_std, deterministic=False):
        return self.proba_distribution(parameters, log_std).get_actions(deterministic)

    def log_prob_from_params(self, parameters, log_std):
        actions = self.actions_from_params(parameters, log_std)
        return actions, self.log_prob(actions)


class HybridActorCriticPolicy(ActorCriticPolicy):
    """沿用 SB3 的 rollout/PPO；Box 仅为存储接口，概率不使用四维高斯。"""

    def _build(self, lr_schedule):
        if not isinstance(self.action_space, HybridMotionWhiskerSpace) or self.use_sde:
            raise ValueError("混合策略需要 HybridMotionWhiskerSpace，且不支持 SDE")
        self._build_mlp_extractor()
        self.action_dist = HybridActionDistribution(self.action_space.sector_count,
                                                   getattr(self.action_space, 'categorical_branches', 2))
        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
            self.mlp_extractor.latent_dim_pi, self.log_std_init)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        if self.ortho_init:
            gains = {self.mlp_extractor: np.sqrt(2), self.action_net: .01, self.value_net: 1.}
            if self.share_features_extractor:
                gains[self.features_extractor] = np.sqrt(2)
            else:
                gains[self.pi_features_extractor] = np.sqrt(2)
                gains[self.vf_features_extractor] = np.sqrt(2)
            for module, gain in gains.items():
                module.apply(partial(self.init_weights, gain=gain))
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(self.action_net(latent_pi), self.log_std)
