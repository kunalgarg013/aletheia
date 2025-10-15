# src/core/field.py
# Minimal ψ-field lattice with non-Markovian dynamics and simple integrator.

from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict

Array = np.ndarray

class FieldConfig:
    def __init__(
        self,
        shape: Tuple[int, int] = (32, 32),
        dt: float = 0.01,
        coupling: float = 0.15,
        nonlin: float = 0.25,
        damping: float = 0.02,
        seed: Optional[int] = 42,
    ):
        self.shape = shape
        self.dt = dt
        self.coupling = coupling      # nearest-neighbor coupling gain
        self.nonlin = nonlin          # cubic nonlinearity
        self.damping = damping        # global dissipation
        self.rng = np.random.default_rng(seed)

class Field:
    """
    ψ(x,t) ∈ ℂ, discretized over a 2D lattice.
    Update: ψ <- ψ + dt * [ Laplacian(ψ) + f(ψ) + memory_term + input ]
    """

    def __init__(self, cfg: FieldConfig):
        self.cfg = cfg
        h, w = cfg.shape
        # Complex field: real=amplitude, imaginary=phase proxy
        self.psi: Array = 0.1 * (cfg.rng.standard_normal((h, w)) + 1j * cfg.rng.standard_normal((h, w)))
        # History buffer for cheap non-Markovian term
        self.history_len = 256
        self.history: Array = np.zeros((self.history_len, h, w), dtype=np.complex128)
        self.hist_idx = 0

        # External input buffer (can be overwritten by experiments)
        self.input: Array = np.zeros((h, w), dtype=np.complex128)

        # Precompute Laplacian kernel (5-point stencil)
        self._lap_ker = np.array([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=float)

    # ----- helpers -----

    def laplacian(self, x: Array) -> Array:
        from scipy.signal import convolve2d  # local import to keep base deps small if unused
        return convolve2d(x.real, self._lap_ker, mode="same", boundary="wrap") + \
               1j * convolve2d(x.imag, self._lap_ker, mode="same", boundary="wrap")

    def nonlinearity(self, x: Array) -> Array:
        # Duffing-style cubic: -nonlin * |x|^2 * x
        return -self.cfg.nonlin * (np.abs(x) ** 2) * x

    def add_history(self):
        self.history[self.hist_idx % self.history_len] = self.psi
        self.hist_idx += 1

    # ----- memory term (cheap kernel) -----

    def memory_term(self, kernel: Array) -> Array:
        """
        kernel: length <= history_len, decays over time (e.g., exp)
        Returns Σ_k kernel[k] * (ψ_{t-k} - ψ_{t-k-1})
        """
        L = min(len(kernel), self.history_len - 1, self.hist_idx)
        if L <= 1:
            return np.zeros_like(self.psi)

        acc = np.zeros_like(self.psi)
        # roll-safe indices
        head = (self.hist_idx - 1) % self.history_len
        for k in range(1, L):
            a = (head - (k - 0)) % self.history_len
            b = (head - (k + 1)) % self.history_len
            dpsi = self.history[a] - self.history[b]
            acc += kernel[k] * dpsi
        return acc

    # ----- one integration step -----

    def step(self, kernel: Array, input_field: Optional[Array] = None) -> Dict[str, float]:
        if input_field is not None:
            self.input = input_field

        self.add_history()

        lap = self.cfg.coupling * self.laplacian(self.psi)
        nonlin = self.nonlinearity(self.psi)
        mem = self.memory_term(kernel)
        damp = -self.cfg.damping * self.psi

        dpsi = lap + nonlin + mem + damp + self.input
        self.psi = self.psi + self.cfg.dt * dpsi

        # simple metrics
        amp = float(np.mean(np.abs(self.psi)))
        return {"amp_mean": amp}
