"""Slow-response gas sensor model."""

from __future__ import annotations


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
