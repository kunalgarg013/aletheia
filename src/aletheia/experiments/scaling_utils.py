# -*- coding: utf-8 -*-
from __future__ import annotations
import numpy as np
from dataclasses import replace
from typing import Callable, Tuple, Dict, Any
from aletheia.solvers.system_roots import QFCASystemConfig, qfca_solve_system

# ---------- Problem generators ----------

def make_poly_kostlan(d: int, degree: int = 3, density: float = 0.3):
    """
    Build F: C^d→C^d with random polynomial components of given degree.
    Density controls how many cross-terms each equation uses.
    Returns (F, J, z0) batched evaluators.
    """
    rng = np.random.default_rng(42 + d + degree)
    # term structure: for each eq k, sample a set of monomials up to 'degree'
    # To keep code lightweight, we use:
    #   f_k(x) = sum_j A_kj x_j + sum_{i,j} B_kij x_i x_j + ... up to degree
    # with Bernoulli(density) masking.
    masks = []
    coeffs = []
    for k in range(d):
        # linear
        A = (rng.standard_normal(d) + 1j*rng.standard_normal(d))
        A *= (rng.random(d) < min(1.0, density*1.5))
        # quadratic
        B = (rng.standard_normal((d,d)) + 1j*rng.standard_normal((d,d)))
        B *= (rng.random((d,d)) < density)
        # cubic (optional)
        C = None
        if degree >= 3:
            C = (rng.standard_normal((d,d,d)) + 1j*rng.standard_normal((d,d,d)))
            C *= (rng.random((d,d,d)) < density*0.6)
        masks.append((A!=0, B!=0, None if C is None else (C!=0)))
        coeffs.append((A, B, C))

    def F(z):
        # z: (M, d)
        M = z.shape[0]
        out = np.zeros((M, d), dtype=np.complex128)
        for k in range(d):
            A, B, C = coeffs[k]
            # linear
            out[:, k] += z @ A
            # quadratic
            out[:, k] += (z @ B @ z.T).diagonal()
            # cubic
            if C is not None:
                # naive cubic contraction: sum_{i,j,l} C_ijl z_i z_j z_l
                # use einsum batched for clarity (ok for these sizes)
                out[:, k] += np.einsum('ijl,m i,m j,m l->m', C, z, z, z, optimize=True)
        return out

    def J(z):
        M = z.shape[0]
        J = np.zeros((M, d, d), dtype=np.complex128)
        for m in range(M):
            x = z[m]
            for k in range(d):
                A, B, C = coeffs[k]
                # df_k/dx = A + (B + B^T) x + cubic terms
                J[m, k, :] += A
                J[m, k, :] += (B + B.T) @ x
                if C is not None:
                    # derivative of x_i x_j x_l wrt x_p = sum over terms with p
                    # approx: sum_{i,j} (C_{pij} + C_{ipj} + C_{ijp}) x_i x_j
                    Cp = C + C.transpose(1,0,2) + C.transpose(2,1,0)
                    J[m, k, :] += np.einsum('pij,i,j->p', Cp, x, x, optimize=True)
        return J

    # circular complex seeds
    ang = np.linspace(0, 2*np.pi, d, endpoint=False)
    base = 1.0*np.exp(1j*ang)
    z0 = np.stack([np.roll(base, s) for s in range(max(8, d//2))], axis=0).astype(np.complex128)
    return F, J, z0

def make_mixed_trig(d: int, density: float = 0.2):
    """
    Mix of sinusoidal and algebraic constraints; moderately sparse.
    """
    rng = np.random.default_rng(1234 + d)
    W = (rng.standard_normal((d,d)) * (rng.random((d,d)) < density)).astype(np.float64)
    U = (rng.standard_normal((d,d)) * (rng.random((d,d)) < density)).astype(np.float64)

    def F(z):
        x = z
        return np.sin(x) + (x @ W.T) + (x * (x @ U.T))

    def J(z):
        x = z
        M = x.shape[0]
        J = np.zeros((M, d, d), dtype=np.complex128)
        cosx = np.cos(x)
        # diag part from sin and elementwise quadratic
        for m in range(M):
            J[m] = np.diag(cosx[m]) + W + np.diag(2*x[m] @ U.T)
        return J

    M = max(8, d//2)
    z0 = 0.5*np.random.randn(M, d) + 0.2j*np.random.randn(M, d)
    return F, J, z0

def make_sparse_phys(d: int, block: int = 5, density: float = 0.05):
    """
    Block-sparse 'physics-like' local coupling: each equation touches a few neighbors.
    Think nonlinear circuit / lattice model.
    """
    rng = np.random.default_rng(999 + d + block)
    # adjacency
    Jmask = np.zeros((d,d), dtype=bool)
    for i in range(d):
        for j in range(max(0, i-block), min(d, i+block+1)):
            if rng.random() < (density * (block+1)):
                Jmask[i,j] = True

    A = rng.standard_normal(d) + 1j*rng.standard_normal(d)
    B = rng.standard_normal((d,d)) + 1j*rng.standard_normal((d,d))
    B *= Jmask

    def F(z):
        return (A + z @ B.T) + (z**2)  # simple nonlinear local law

    def J(z):
        M = z.shape[0]
        J = np.zeros((M, d, d), dtype=np.complex128)
        for m in range(M):
            J[m] = B + np.diag(2*z[m])
        return J

    M = max(8, d//3)
    z0 = 0.3*np.random.randn(M, d) + 0.1j*np.random.randn(M, d)
    return F, J, z0

# ---------- k-NN repulsion hook (optional) ----------
def patch_knn_repulsion(cfg: QFCASystemConfig, k: int) -> QFCASystemConfig:
    """
    If k>0, we overload gamma behavior by setting a 'kNN' flag in config.
    The solver can check cfg.gamma<0 to signify 'use kNN=k with |gamma|'.
    """
    if k <= 0:
        return cfg
    return replace(cfg, gamma=-abs(cfg.gamma))  # negative gamma ⇒ interpret as kNN mode with k stored elsewhere