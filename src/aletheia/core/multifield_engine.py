"""
MultiFieldEngine — a modular, agency-gated coupling framework
for synchronizing multiple Field objects in Aletheia / PsiForge ecosystem.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Callable, Optional

# expected imports in your environment
from aletheia.core.field import Field
from aletheia.core.affect import tension_A

def default_tension(psi: np.ndarray) -> float:
    return float(np.mean(np.abs(psi)**2))

def phase_coherence(psi: np.ndarray) -> float:
    phase = np.angle(psi)
    return float(np.abs(np.mean(np.exp(1j * phase))))

@dataclass
class GateParams:
    alpha: float = 5.0   # coherence sharpness
    beta:  float = 5.0   # affect mismatch penalty
    floor: float = 0.05  # minimum coupling
    cap:   float = 1.0   # maximum coupling

class AdaptiveGates:
    def __init__(self, params: GateParams):
        self.p = params

    def link(self, Ai: float, Aj: float, Ci: float, Cj: float) -> float:
        """Compute pairwise gate value between fields i and j."""
        sig = 1.0 / (1.0 + np.exp(-self.p.alpha * ((Ci + Cj) - 1.0)))
        mismatch = np.exp(-self.p.beta * abs(Ai - Aj))
        g = sig * mismatch
        return float(np.clip(g, self.p.floor, self.p.cap))

@dataclass
class Diagnostics:
    A: np.ndarray
    C: np.ndarray
    G: np.ndarray
    step: int

class MultiFieldEngine:
    def __init__(
        self,
        fields: List,                                     # List[Field]
        boundaries: Dict[Tuple[int,int], Tuple[np.ndarray, np.ndarray]],
        lambdas: np.ndarray,
        gates: AdaptiveGates,
        tension_fn: Callable = default_tension,
        agency_hook: Optional[Callable] = None,
        eta: float = 0.4,
        seed: int = 42
    ):
        self.fields = fields
        self.N = len(fields)
        self.boundaries = boundaries
        self.lambdas = lambdas
        self.gates = gates
        self.tension_fn = tension_fn
        self.agency_hook = agency_hook
        self.eta = eta
        self.rng = np.random.default_rng(seed)

        self.A = np.zeros(self.N)
        self.C = np.zeros(self.N)
        self.G = np.ones((self.N, self.N))

    def step(self, t: int, kernel=None) -> Diagnostics:
        # local evolution
        for i, f in enumerate(self.fields):
            f.step(kernel=kernel)
            self.A[i] = self.tension_fn(f.psi)
            self.C[i] = phase_coherence(f.psi)

        # local agency (optional)
        if self.agency_hook:
            for i, f in enumerate(self.fields):
                self.agency_hook(f, A=self.A[i], coherence=self.C[i], t=t)

        # update gates matrix
        for i in range(self.N):
            for j in range(self.N):
                if i == j:
                    self.G[i, j] = 0.0
                    continue
                if self.lambdas[i, j] == 0:
                    self.G[i, j] = 0.0
                    continue
                self.G[i, j] = self.gates.link(self.A[i], self.A[j], self.C[i], self.C[j])

        # boundary couplings
        for (i, j), (idx_i, idx_j) in self.boundaries.items():
            lam = self.lambdas[i, j]
            gij = self.G[i, j]
            if lam == 0 or gij == 0: 
                continue
            psi_i = self.fields[i].psi.ravel()
            psi_j = self.fields[j].psi.ravel()
            Bi, Bj = psi_i[idx_i], psi_j[idx_j]
            diff = (Bi - Bj)
            psi_i[idx_i] = Bi - self.eta * (2.0 * lam * gij) * diff
            psi_j[idx_j] = Bj + self.eta * (2.0 * lam * gij) * diff
            self.fields[i].psi = psi_i.reshape(self.fields[i].psi.shape)
            self.fields[j].psi = psi_j.reshape(self.fields[j].psi.shape)

        return Diagnostics(A=self.A.copy(), C=self.C.copy(), G=self.G.copy(), step=t)

    # convenience methods
    def mean_coherence(self) -> float:
        return float(np.mean(self.C))

    def mean_tension(self) -> float:
        return float(np.mean(self.A))