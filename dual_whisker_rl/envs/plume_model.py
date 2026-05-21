"""Simplified 2D odor plume model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass
class PlumeParams:
    source: tuple[float, float] = (8.0, 5.0)
    wind_direction: float = math.pi
    c0: float = 1.0
    decay_length: float = 5.0
    sigma0: float = 0.25
    spread_rate: float = 0.18
    noise_std: float = 0.015
    patch_strength: float = 0.35


class GaussianPlume:
    """Downwind Gaussian plume with lightweight intermittent patches."""

    def __init__(self, params: PlumeParams | None = None, seed: int | None = None) -> None:
        self.params = params or PlumeParams()
        self.rng = np.random.default_rng(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

    def concentration(self, x: float, y: float, t: int = 0) -> float:
        p = self.params
        dx = float(x) - p.source[0]
        dy = float(y) - p.source[1]
        wind = np.array([math.cos(p.wind_direction), math.sin(p.wind_direction)])
        cross = np.array([-wind[1], wind[0]])

        downwind = dx * wind[0] + dy * wind[1]
        crosswind = dx * cross[0] + dy * cross[1]
        if downwind < 0.0:
            return self._noise_only(p)

        sigma = p.sigma0 + p.spread_rate * downwind
        mean = p.c0 * math.exp(-downwind / p.decay_length) * math.exp(
            -(crosswind**2) / (2.0 * sigma**2)
        )
        patch = self._intermittent_patch(downwind, crosswind, t, sigma)
        noise = self.rng.normal(0.0, p.noise_std)
        return float(max(0.0, mean + patch + noise))

    def grid(
        self,
        width: float,
        height: float,
        resolution: int = 120,
        t: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs = np.linspace(0.0, width, resolution)
        ys = np.linspace(0.0, height, resolution)
        xx, yy = np.meshgrid(xs, ys)
        zz = np.zeros_like(xx)
        for row in range(resolution):
            for col in range(resolution):
                zz[row, col] = self.concentration(float(xx[row, col]), float(yy[row, col]), t)
        return xx, yy, zz

    def _intermittent_patch(self, downwind: float, crosswind: float, t: int, sigma: float) -> float:
        p = self.params
        center = math.sin(0.18 * t + 1.7 * downwind) * sigma
        envelope = math.exp(-((crosswind - center) ** 2) / (2.0 * (0.45 * sigma) ** 2))
        pulse = max(0.0, math.sin(0.4 * t - 0.9 * downwind))
        return p.patch_strength * envelope * pulse

    def _noise_only(self, p: PlumeParams) -> float:
        return float(max(0.0, self.rng.normal(0.0, p.noise_std)))
