# src/core/affect.py
# Global tension (A) as a coherence/dissonance functional; modulates learning/agency.

from __future__ import annotations
import numpy as np
from scipy.signal import convolve2d

def tension_A(psi: np.ndarray) -> float:
    """A = ∫ ||∇ψ||^2 dx  (discrete proxy via Laplacian energy)."""
    lap_ker = np.array([[0, 1, 0],
                        [1, -4, 1],
                        [0, 1, 0]], dtype=float)
    lap_r = convolve2d(psi.real, lap_ker, mode="same", boundary="wrap")
    lap_i = convolve2d(psi.imag, lap_ker, mode="same", boundary="wrap")
    A = float(np.mean(lap_r**2 + lap_i**2))
    return A

class Affect:
    def __init__(self, beta_plasticity: float = 0.5, beta_gain: float = 0.5):
        self.beta_plasticity = beta_plasticity
        self.beta_gain = beta_gain
        self._ema = 0.0
        self._alpha = 0.01  # smoothing

    def update(self, A: float) -> float:
        self._ema = (1 - self._alpha) * self._ema + self._alpha * A
        return self._ema

    def modulate_plasticity(self, base_lr: float, A: float) -> float:
        # Higher tension => lower LR (be conservative under distress)
        return base_lr * np.exp(-self.beta_plasticity * A)

    def modulate_input_gain(self, base_gain: float, A: float) -> float:
        # Higher tension => reduce input drive (turn down the world when overwhelmed)
        return base_gain * np.exp(-self.beta_gain * A)
