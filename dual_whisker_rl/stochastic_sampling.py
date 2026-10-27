"""逐回合独立的PPO概率采样；与并行顺序、Torch全局随机流和推理设备解耦。"""
from __future__ import annotations
import hashlib
import json
import numpy as np

PROTOCOL = 'stochastic-arrival-v2'


def policy_seed(settings, model_sha, scene, repeat=0):
    # 故障不参与种子派生，同模型的nominal/扰动使用共同随机数。
    value = [int(settings['eval_seed']), str(model_sha), scene['scenario_id'],
             int(scene['seed']), int(repeat)]
    raw = json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:16], 'little')


class EpisodeSampler:
    def __init__(self, settings, model_sha, scene, repeat=0):
        self.seed = policy_seed(settings, model_sha, scene, repeat)
        self.rng = np.random.default_rng(self.seed)


def probabilities(model, observations):
    import torch
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(np.asarray(observations))
        distribution = model.policy.get_distribution(tensor).distribution
        heads = distribution if isinstance(distribution, (list, tuple)) else [distribution]
        if any(not hasattr(head, 'probs') for head in heads):
            raise ValueError('随机正式协议仅支持Categorical/MultiCategorical策略')
        return [head.probs.detach().cpu().numpy().astype(np.float64) for head in heads]


def from_uniforms(heads, uniforms):
    """每个动作头一次逆CDF采样，概率沿用SB3策略，不重设温度或加探索噪声。"""
    u = np.asarray(uniforms, dtype=np.float64)
    if u.ndim != 2 or u.shape[1] != len(heads) or np.any((u < 0) | (u >= 1)):
        raise ValueError('采样随机数的形状或范围错误')
    result = []
    for i, head in enumerate(heads):
        p = np.asarray(head, dtype=np.float64)
        if p.ndim != 2 or p.shape[0] != len(u) or not np.all(np.isfinite(p)) or np.any(p < 0) or np.any(p.sum(axis=1) <= 0):
            raise ValueError('策略概率无效')
        cdf = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
        cdf[:, -1] = 1.0
        result.append((u[:, i, None] >= cdf).sum(axis=1))
    return np.stack(result, axis=1).astype(np.int64)


def sampled_actions(model, observations, samplers):
    heads = probabilities(model, observations)
    uniforms = np.stack([s.rng.random(len(heads)) for s in samplers])
    actions = from_uniforms(heads, uniforms)
    if model.action_space.shape == ():
        return actions[:, 0]
    return actions
