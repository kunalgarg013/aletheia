# src/core/memory.py
# Memory kernels and retrocausal (lookahead) hinting interface.

from __future__ import annotations
import numpy as np

def exp_kernel(length: int = 128, tau: float = 32.0) -> np.ndarray:
    k = np.arange(length, dtype=float)
    ker = np.exp(-k / max(1e-6, tau))
    ker[0] = 0.0  # no immediate self
    return ker / (np.sum(ker) + 1e-12)

def powerlaw_kernel(length: int = 128, alpha: float = 1.2) -> np.ndarray:
    k = np.arange(1, length + 1, dtype=float)
    ker = 1.0 / (k ** alpha)
    ker = np.concatenate([[0.0], ker[:-1]])
    return ker / (np.sum(ker) + 1e-12)

class RetroHint:
    """
    A simple retrocausal hint: if a predicted future tension exceeds a threshold,
    inject a small bias term now (to be added to input or parameters).
    """
    def __init__(self, gain: float = 0.02, threshold: float = 0.15):
        self.gain = gain
        self.threshold = threshold

    def bias(self, predicted_tension: float) -> float:
        return -self.gain if predicted_tension > self.threshold else 0.0
