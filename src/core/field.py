# src/core/field.py
# Field evolution with memory buffer integration
# Compatible with ablation and retrocausal experiments

import numpy as np
from dataclasses import dataclass

@dataclass
class FieldConfig:
    shape: tuple = (64, 64)
    dt: float = 0.02
    seed: int = 42
    kernel_length: int = 256  # memory depth for non-Markovian term
    init_amp: float = 0.1

class Field:
    def __init__(self, cfg: FieldConfig):
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self.dt = cfg.dt

        # Primary complex field
        self.psi = (
            cfg.init_amp
            * (np.random.randn(*cfg.shape) + 1j * np.random.randn(*cfg.shape))
        )

        # Initialize a circular memory buffer
        self.memory_history = np.zeros(
            (cfg.kernel_length, *cfg.shape), dtype=np.complex128
        )
        self._memory_index = 0
        # preload first frame
        self.memory_history[0] = self.psi.copy()

        # Optionally store diagnostics if needed
        self.meta = {}

    # -------------------------
    #  core evolution step
    # -------------------------
    def step(self, kernel=None):
        """
        Evolve the field by one step.
        If a kernel is provided, include its non-Markovian memory contribution.
        """
        psi = self.psi

        # Basic nonlinear evolution (placeholder – keep your existing dynamics)
        laplace = np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) \
                + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
        nonlinear = np.abs(psi) ** 2 * psi
        psi_next = psi + self.dt * (0.2 * laplace - 0.05 * nonlinear)

        # --- Memory contribution ---
        if kernel is not None and self.memory_history is not None:
            # Tensor contraction: weighted sum over past L states
            mem = np.tensordot(kernel[: self.memory_history.shape[0]], 
                               self.memory_history, axes=([0], [0]))
            psi_next += self.dt * mem

        # Update the circular buffer with the new state
        self.memory_history[self._memory_index] = psi_next.copy()
        self._memory_index = (self._memory_index + 1) % self.memory_history.shape[0]

        # Commit
        self.psi = psi_next

        # Metadata for diagnostics
        self.meta = {
            "mean_amp": float(np.mean(np.abs(self.psi))),
            "phase_coherence": float(np.abs(np.mean(np.exp(1j * np.angle(self.psi))))),
        }
        return self.meta

    # -------------------------
    #  Reset / utilities
    # -------------------------
    def reset(self):
        """Reset field to random initial state."""
        np.random.seed(self.cfg.seed)
        self.psi = (
            self.cfg.init_amp
            * (np.random.randn(*self.cfg.shape) + 1j * np.random.randn(*self.cfg.shape))
        )
        self.memory_history[:] = 0.0
        self.memory_history[0] = self.psi.copy()
        self._memory_index = 0
        self.meta = {}

    def clone(self):
        """Return a deep copy of the field."""
        new = Field(self.cfg)
        new.psi = self.psi.copy()
        new.memory_history = self.memory_history.copy()
        new._memory_index = self._memory_index
        new.meta = dict(self.meta)
        return new