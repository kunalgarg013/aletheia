# -*- coding: utf-8 -*-
"""
QFCA System Root Solver (vector case)
-------------------------------------
Solve F(z) = 0 where z ∈ C^d and F: C^d -> C^m (typically m=d)
using non-Markovian QFCA flow with agency gating, multi-walker
repulsion in variable space, and optional safeguarded Newton polish.

Each "walker" is a d-dimensional complex vector. We evolve M walkers
in parallel so they can settle into distinct solution basins.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import Callable, Tuple, Dict, Any, Optional

ArrayC = np.ndarray

@dataclass
class QFCASystemConfig:
    steps: int = 1200
    eta: float = 0.15               # base step size
    lam_mem: float = 0.6            # memory coupling
    L: int = 8                      # memory length
    tau: float = 5.0                # memory decay
    gamma: float = 0.02             # inter-walker repulsion
    retro_every: int = 12           # Newton-like nudge cadence
    retro_mu: float = 0.6           # Newton-like nudge strength
    tol: float = 1e-10              # residual tolerance for success
    keep_history: bool = True
    max_step: float = 3e1           # clip per-iteration update magnitude
    grad_cap: float = 1e6           # cap ||grad||
    eta_decay: float = 0.03         # eta(t) = eta / (1 + eta_decay t)
    rmax_scale: float = 2.0         # clamp |z_k| to radius bound
    cluster_eps: float = 1e-6       # solution clustering threshold
    walkers: int = 6                # number of walkers M
    refine_roots: bool = True       # do safeguarded Newton polish at end

def exp_kernel(L: int, tau: float) -> ArrayC:
    t = np.arange(1, L + 1, dtype=np.float64)
    k = np.exp(-t / tau)
    k /= (k.sum() + 1e-12)
    return k

def _coherence(lastK: ArrayC) -> ArrayC:
    """
    Coherence proxy per walker: inverse recent step variance.
    lastK shape: (H, M, d). Returns (M,) in [0,1].
    """
    H = lastK.shape[0]
    if H < 3:
        return 0.5 * np.ones(lastK.shape[1])
    dz = np.diff(lastK[-min(5, H):], axis=0)           # (h, M, d)
    sig = np.mean(np.linalg.norm(dz, axis=-1), axis=0) # (M,)
    c = 1.0 / (1.0 + sig)
    return np.clip(c, 0.0, 1.0)

def safeguarded_newton_refine(
    z: ArrayC,
    F: Callable[[ArrayC], ArrayC],
    J: Callable[[ArrayC], ArrayC],
    max_iter: int = 40,
    tol: float = 1e-12
) -> ArrayC:
    """
    Post-process each walker independently with a few Newton iterations
    and simple backtracking to drive residuals down.
    z: (M, d), F returns (M, m), J returns (M, m, d)
    """
    M, d = z.shape
    r = z.copy()
    for _ in range(max_iter):
        Fv = F(r)                               # (M, m)
        res = np.linalg.norm(Fv, axis=1)        # (M,)
        if np.max(res) < tol:
            break
        Jv = J(r)                               # (M, m, d)
        for i in range(M):
            Ji = Jv[i]                          # (m, d)
            Fi = Fv[i]                          # (m,)
            # least-squares Newton step: solve Ji * s = Fi
            # (note sign: we subtract step; we solve Ji s = F → s = lstsq(Ji, F))
            try:
                s, *_ = np.linalg.lstsq(Ji, Fi, rcond=None)
            except np.linalg.LinAlgError:
                continue
            # backtracking
            step = s
            new_r = r[i] - step
            new_res = np.linalg.norm(F(new_r[None, :])[0])
            old_res = res[i]
            bt = 0
            while new_res > old_res and bt < 6:
                step *= 0.5
                new_r = r[i] - step
                new_res = np.linalg.norm(F(new_r[None, :])[0])
                bt += 1
            r[i] = new_r
    return r

def qfca_solve_system(
    F: Callable[[ArrayC], ArrayC],
    J: Callable[[ArrayC], ArrayC],
    z0: Optional[ArrayC],
    cfg: QFCASystemConfig
) -> Tuple[ArrayC, Optional[ArrayC], Dict[str, Any]]:
    """
    Solve F(z)=0 using QFCA dynamics.

    Parameters
    ----------
    F : function
        F(z) -> (M, m) residuals for batch of walkers z (M, d).
    J : function
        J(z) -> (M, m, d) Jacobians for batch of walkers.
    z0 : None or (M, d) complex ndarray
        Initial walkers. If None, we sample around a circle in each component.
    cfg : QFCASystemConfig

    Returns
    -------
    roots : (K, d) complex array of fused solutions
    traj  : (T, M, d) complex array of trajectories if keep_history else None
    info  : dict with diagnostics
    """
    M = cfg.walkers if z0 is None else z0.shape[0]
    # infer dimensionality if z0 is None via a dummy eval
    if z0 is None:
        # try d=2 as default; the experiment will pass a proper z0 anyway
        d = 2
        # seed circle radius
        R = 2.5
        theta = np.linspace(0, 2*np.pi, M, endpoint=False)
        # same angle per dim but with small random phase offsets
        ph = np.random.uniform(0, 2*np.pi, size=(M, d))
        mag = R * (0.8 + 0.4*np.random.rand(M, d))
        z = (mag * np.exp(1j * ph)).astype(np.complex128)
        Rmax = cfg.rmax_scale * R
    else:
        z = z0.astype(np.complex128).copy()
        d = z.shape[1]
        R = max(2.0, np.max(np.abs(z)))
        Rmax = cfg.rmax_scale * R

    K = exp_kernel(cfg.L, cfg.tau)
    history = [z.copy()]                      # list of (M, d)
    traj = [z.copy()] if cfg.keep_history else None

    success = False
    iterations = 0

    for t in range(1, cfg.steps + 1):
        iterations = t
        Fv = F(z)                              # (M, m)
        res = np.linalg.norm(Fv, axis=1)       # (M,)
        Jv = J(z)                              # (M, m, d)

        # Agency gate from coherence + residual
        H = np.stack(history, axis=0)          # (H, M, d)
        C = _coherence(H)                      # (M,)
        gate = 1.0 / (1.0 + np.exp(-6.0 * (C - 0.5))) * np.exp(-3.0 * np.clip(res, 0, 1))
        gate = np.clip(gate, 0.1, 1.0)

        # Gradient of ||F||^2 ≈ J^H F (Gauss-Newton style)
        # Construct grad per walker: (d,) complex
        grad = np.zeros_like(z)
        for i in range(M):
            Ji = Jv[i]                          # (m, d)
            Fi = Fv[i][:, None]                 # (m, 1)
            gi = (Ji.conj().T @ Fi).ravel()     # (d,)
            grad[i] = gi

        # cap gradient
        gnorm = np.linalg.norm(grad, axis=1)    # (M,)
        big = gnorm > cfg.grad_cap
        if np.any(big):
            grad[big] = (grad[big].T / gnorm[big]).T * cfg.grad_cap

        # Memory force: sum_k K(k) (z_t - z_{t-k})
        mem = np.zeros_like(z)
        for k, w in enumerate(K, start=1):
            if len(history) > k:
                mem += w * (z - history[-k])

        # Multi-walker soft repulsion in variable space
        rep = np.zeros_like(z)
        for i in range(M):
            diff = z[i] - z                     # (M, d)
            denom = np.sum(np.abs(diff)**2, axis=1) + 1e-9  # (M,)
            # exclude self-term by zero diff[i]
            rep[i] = (diff / denom[:, None]).sum(axis=0)

        # Adaptive step
        eta_eff = cfg.eta / (1.0 + cfg.eta_decay * t)
        dz = (grad + cfg.lam_mem * mem - cfg.gamma * rep)
        # clip dz per walker
        dz_norm = np.linalg.norm(dz, axis=1)
        too_big = dz_norm > cfg.max_step
        if np.any(too_big):
            dz[too_big] = (dz[too_big].T / dz_norm[too_big]).T * cfg.max_step

        # update
        z = z - (eta_eff * gate)[:, None] * dz
        # clamp radius
        z_abs = np.linalg.norm(z, axis=1)
        outside = z_abs > Rmax
        if np.any(outside):
            z[outside] = (z[outside].T / z_abs[outside]).T * Rmax

        # retrocausal Newton nudge every few steps (Gauss-Newton least-squares)
        if cfg.retro_every and (t % cfg.retro_every == 0):
            Fv = F(z); Jv = J(z)
            for i in range(M):
                Ji = Jv[i]                       # (m, d)
                Fi = Fv[i]
                try:
                    s, *_ = np.linalg.lstsq(Ji, Fi, rcond=None)
                except np.linalg.LinAlgError:
                    continue
                z[i] = z[i] - cfg.retro_mu * s

        if cfg.keep_history:
            traj.append(z.copy())
        history.append(z.copy())
        if len(history) > (cfg.L + 3):
            history.pop(0)

        if np.max(res) < cfg.tol:
            success = True
            break

    # ----- cluster solutions across walkers -----
    roots = z.copy()
    taken = np.zeros(M, dtype=bool)
    fused = []
    for i in range(M):
        if taken[i]: continue
        group = [roots[i]]
        taken[i] = True
        for j in range(i+1, M):
            if not taken[j] and np.linalg.norm(roots[j] - roots[i]) < cfg.cluster_eps:
                taken[j] = True
                group.append(roots[j])
        fused.append(np.mean(group, axis=0))
    roots = np.array(fused, dtype=np.complex128)

    # optional polish
    if cfg.refine_roots and roots.size > 0:
        # refine each fused solution as a single-walker batch
        roots = safeguarded_newton_refine(roots, F=lambda x: F(x), J=lambda x: J(x), max_iter=40, tol=max(cfg.tol*1e-2, 1e-12))

    # final diagnostics
    traj_arr = np.asarray(traj) if cfg.keep_history else None
    final_res = np.linalg.norm(F(roots), axis=1) if roots.size else np.array([], dtype=float)

    info: Dict[str, Any] = dict(
        success=bool(success or (roots.size and np.max(final_res) < cfg.tol)),
        iterations=iterations,
        config=asdict(cfg),
        n_solutions=int(roots.shape[0]),
        final_residuals=final_res.tolist() if roots.size else [],
    )
    return roots, traj_arr, info