# -*- coding: utf-8 -*-
"""
QFCA Polynomial Root Solver
---------------------------
Solve cubic, quartic, or general n-degree polynomials via
non-Markovian QFCA flow with agency gating, multi-walker repulsion,
and periodic retrocausal Newton nudges.

This is robust where vanilla Newton is chaotic (multiplicities, clustered roots).
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, asdict
from typing import Iterable, Tuple, Dict, Any


# ---------------------- Utilities ----------------------

def horner_eval(z: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate polynomial p(z) with Horner's method. coeffs = [a_n, ..., a_0]."""
    p = np.zeros_like(z, dtype=np.complex128)
    for a in coeffs:
        p = p * z + a
    return p

def horner_derivative_eval(z: np.ndarray, coeffs: np.ndarray) -> np.ndarray:
    """Evaluate p'(z) using derivative coeffs derived on the fly."""
    n = len(coeffs) - 1
    if n <= 0:
        return np.zeros_like(z, dtype=np.complex128)
    # derivative coeffs: [n*a_n, (n-1)*a_{n-1}, ..., 1*a_1]
    dp = np.zeros_like(z, dtype=np.complex128)
    for k, a in enumerate(coeffs[:-1]):  # skip a_0
        dp = dp * z + a * (n - k)
    return dp

def exp_kernel(L: int = 10, tau: float = 4.0) -> np.ndarray:
    """Positive, normalized exponential memory kernel (1..L)."""
    t = np.arange(1, L + 1, dtype=np.float64)
    k = np.exp(-t / tau)
    k /= (k.sum() + 1e-12)
    return k

def stable_radius(coeffs: np.ndarray) -> float:
    """
    Crude radius for initial seeds (slightly > Cauchy bound).
    Cauchy bound: all roots satisfy |z| <= 1 + max(|a_k/a_n|).
    """
    a = coeffs.astype(np.complex128)
    if a[0] == 0:
        raise ValueError("Leading coefficient a_n must be nonzero.")
    ratios = np.abs(a[1:] / a[0])
    return 1.2 * (1.0 + (0.0 if ratios.size == 0 else ratios.max()))

def refine_roots_newton(roots: np.ndarray, coeffs: np.ndarray, max_iter: int = 50,
                        tol: float = 1e-14, eps: float = 1e-12) -> np.ndarray:
    """
    Post-process roots with a few safeguarded Newton iterations to drive residuals
    to machine precision. Operates independently per root with simple backtracking.
    """
    r = roots.astype(np.complex128).copy()
    for _ in range(max_iter):
        p = horner_eval(r, coeffs)
        dp = horner_derivative_eval(r, coeffs)
        # If derivative is tiny, skip this iteration for that root
        denom = np.where(np.abs(dp) < eps, eps + 0j, dp)
        step = p / denom
        # Backtracking line search: try full step, then halve if residual worsens
        new_r = r - step
        new_p = horner_eval(new_r, coeffs)
        worsen = np.abs(new_p) > np.abs(p)
        bt = 0
        while np.any(worsen) and bt < 6:
            step[worsen] *= 0.5
            new_r[worsen] = r[worsen] - step[worsen]
            new_p[worsen] = horner_eval(new_r[worsen], coeffs)
            worsen = np.abs(new_p[worsen]) > np.abs(p[worsen])
            bt += 1
        r = new_r
        if np.max(np.abs(new_p)) < tol:
            break
    return r

def deflate_monic(coeffs: np.ndarray, root: complex) -> np.ndarray:
    """
    Synthetic division of a monic polynomial by (z - root).
    coeffs must be monic (coeffs[0] == 1).
    Returns the deflated monic coefficient array of length-1.
    """
    a = coeffs.astype(np.complex128)
    if not np.allclose(a[0], 1.0+0j):
        raise ValueError("deflate_monic expects monic polynomial (a[0]==1).")
    n = len(a) - 1
    b = np.zeros(n, dtype=np.complex128)
    b[0] = 1.0 + 0j  # leading stays monic
    acc = 1.0 + 0j
    for k in range(1, n):
        acc = a[k] + acc * root
        b[k] = acc
    return b

def companion_roots_monic(coeffs: np.ndarray) -> np.ndarray:
    """
    Compute all roots of a monic polynomial via the companion matrix.
    coeffs = [1, a_{n-1}, ..., a_0]
    """
    a = coeffs.astype(np.complex128)
    n = len(a) - 1
    if n <= 0:
        return np.array([], dtype=np.complex128)
    C = np.zeros((n, n), dtype=np.complex128)
    C[1:, :-1] = np.eye(n-1, dtype=np.complex128)
    C[0, :] = -a[1:].conj()  # last row in standard form is negative coefficients; using first row as we placed shift
    # Adjust to standard companion form (last row holds -a_0..-a_{n-1})
    C = np.roll(C, shift=1, axis=0)
    vals = np.linalg.eigvals(C)
    return vals


# ---------------------- Config ----------------------

@dataclass
class QFCAPolyConfig:
    steps: int = 600
    eta: float = 0.25                # base step size
    lam_mem: float = 0.6             # memory coupling
    L: int = 8                       # memory length
    tau: float = 3.0                 # memory decay
    gamma: float = 0.02              # multi-walker repulsion
    retro_every: int = 10            # Newton nudge cadence
    retro_mu: float = 0.6            # Newton nudge strength
    tol: float = 1e-12               # residual tolerance for convergence
    seed_radius_scale: float = 1.0   # multiply stable radius for init
    random_jitter: float = 0.3       # multiplicative jitter around circle
    cluster_eps: float = 1e-7        # merging threshold for coincident roots
    keep_history: bool = True        # return full trajectories
    max_step: float = 1e2          # clip per-iteration update magnitude
    grad_cap: float = 1e6          # cap |grad| to avoid overflow
    rmax_scale: float = 2.0        # clamp |z| to rmax_scale * stable_radius
    eta_decay: float = 0.05        # eta(t) = eta / (1 + eta_decay * t)
    fill_missing_via_companion: bool = True  # if unique roots < degree, use companion-matrix to recover missing


# ---------------------- Core Solver ----------------------

def solve_polynomial_qfca(
    coeffs: Iterable[complex],
    seeds: Iterable[complex] | None = None,
    cfg: QFCAPolyConfig = QFCAPolyConfig()
) -> Tuple[np.ndarray, np.ndarray | None, Dict[str, Any]]:
    """
    Solve p(z)=0 with QFCA flow.

    Parameters
    ----------
    coeffs : iterable of complex
        Polynomial coefficients [a_n, ..., a_0], a_n != 0.
    seeds : iterable of complex or None
        Initial guesses. If None, use N=n evenly spaced points on a circle.
    cfg : QFCAPolyConfig
        Dynamics/accuracy parameters.

    Returns
    -------
    roots : (M,) complex np.ndarray
        Fused roots discovered (M<=degree, multiplicities merged by proximity).
    traj : (T,N) complex np.ndarray or None
        Per-iteration positions of all walkers (if keep_history True).
    info : dict
        Diagnostics: residuals, final walkers, config, iterations, success flag.
    """
    a = np.asarray(list(coeffs), dtype=np.complex128)
    n = len(a) - 1
    if n <= 0:
        return np.array([], dtype=np.complex128), None, {"success": True, "iterations": 0}

    # Normalize leading coefficient for numerical stability
    if a[0] == 0:
        raise ValueError("Leading coefficient a_n must be nonzero.")
    a = a / a[0]

    # Additional normalization to keep |p|, |p'| O(1)
    coeff_norm = np.max(np.abs(a))
    if coeff_norm > 0:
        a = a / coeff_norm

    # Seeds
    if seeds is None:
        N = n
        R = cfg.seed_radius_scale * stable_radius(a)
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        jitter = 1.0 - cfg.random_jitter + 2 * cfg.random_jitter * np.random.rand(N)
        z = (R * jitter) * np.exp(1j * theta)
        Rmax = cfg.rmax_scale * R
    else:
        z = np.asarray(list(seeds), dtype=np.complex128)
        N = z.size
        R = stable_radius(a)
        Rmax = cfg.rmax_scale * R

    # Memory kernel & history
    K = exp_kernel(cfg.L, cfg.tau)
    history = [z.copy()]
    traj = [z.copy()] if cfg.keep_history else None

    def coherence(lastK: np.ndarray) -> np.ndarray:
        """
        Coherence proxy per walker: inverse recent step variance.
        Returns values in [0,1].
        """
        if lastK.shape[0] < 3:
            return 0.5 * np.ones(N)
        dz = np.diff(lastK[-min(5, lastK.shape[0]):], axis=0)
        sig = np.mean(np.abs(dz), axis=0)
        c = 1.0 / (1.0 + sig)
        return np.clip(c, 0.0, 1.0)

    iterations = 0
    success = False

    for t in range(1, cfg.steps + 1):
        iterations = t
        p = horner_eval(z, a)
        dp = horner_derivative_eval(z, a)
        resid = np.abs(p)

        # Agency gate per walker
        H = np.array(history)  # (H,N)
        C = coherence(H)
        gate = 1.0 / (1.0 + np.exp(-6.0 * (C - 0.5))) * np.exp(-3.0 * np.clip(resid, 0, 1))
        gate = np.clip(gate, 0.1, 1.0)

        # Base gradient: ∂|p|^2/∂z* = p * conj(p')
        grad = p * np.conjugate(dp)
        # Cap gradient to avoid overflow in extreme cases
        gabs = np.abs(grad)
        big = gabs > cfg.grad_cap
        if np.any(big):
            grad[big] = grad[big] / gabs[big] * cfg.grad_cap

        # Memory force
        mem = 0.0j
        for k, w in enumerate(K, start=1):
            if len(history) > k:
                mem += w * (z - history[-k])

        # Multi-walker soft repulsion (avoid pile-up on one root)
        rep = np.zeros_like(z)
        # vectorized: for each i, sum over j != i
        for i in range(N):
            diff = z[i] - z
            denom = np.abs(diff) ** 2 + 1e-9
            rep[i] = (diff / denom).sum()  # diff[i]=0, harmless

        # QFCA update
        eta_eff = cfg.eta / (1.0 + cfg.eta_decay * t)
        dz = (grad + cfg.lam_mem * mem - cfg.gamma * rep)
        # Clip update to prevent runaway steps
        dz_abs = np.abs(dz)
        too_big = dz_abs > cfg.max_step
        if np.any(too_big):
            dz[too_big] = dz[too_big] / dz_abs[too_big] * cfg.max_step
        z = z - eta_eff * gate * dz

        # Clamp z magnitude to Rmax to prevent runaway
        z_abs = np.abs(z)
        outside = z_abs > Rmax
        if np.any(outside):
            z[outside] = z[outside] / z_abs[outside] * Rmax

        # Retrocausal Newton nudge
        if cfg.retro_every and (t % cfg.retro_every == 0):
            z = z - cfg.retro_mu * (p / (dp + 1e-9))

        if cfg.keep_history:
            traj.append(z.copy())
        history.append(z.copy())
        if len(history) > (cfg.L + 3):
            history.pop(0)

        # Early convergence if all residuals small
        if np.all(resid < cfg.tol):
            success = True
            break

    # Fuse coincident roots (cluster) with adaptive tolerance
    roots = z.copy()
    eps_rel = max(cfg.cluster_eps, 1e-8 * Rmax)
    taken = np.zeros(len(roots), dtype=bool)
    fused_list = []
    for i in range(len(roots)):
        if taken[i]:
            continue
        group = [roots[i]]
        taken[i] = True
        for j in range(i+1, len(roots)):
            if not taken[j] and np.abs(roots[j] - roots[i]) < eps_rel:
                taken[j] = True
                group.append(roots[j])
        fused_list.append(np.mean(group))
    roots = np.array(fused_list, dtype=np.complex128)

    # Optional post-refinement with safeguarded Newton to polish residuals
    roots = refine_roots_newton(roots, a, max_iter=50, tol=max(cfg.tol*1e-2, 1e-14))

    # Final diagnostics after refinement
    final_resid = np.abs(horner_eval(roots, a))
    # If QFCA phase didn't declare success but refined residuals are good, flip success
    if not success and np.all(final_resid < cfg.tol):
        success = True
    # If we have fewer unique roots than the degree, attempt recovery via companion on deflated poly
    if cfg.fill_missing_via_companion:
        m = len(roots)
        if m < n:
            # Deflate the monic normalized polynomial by each known root
            a_def = a.copy()
            # ensure monic
            if not np.allclose(a_def[0], 1.0+0j):
                a_def = a_def / a_def[0]
            for r0 in roots:
                a_def = deflate_monic(a_def, r0)
            # Solve remaining via companion
            rem = companion_roots_monic(a_def)
            # Merge all and re-cluster to avoid near-duplicate returns
            roots = np.concatenate([roots, rem])
            # Re-cluster merged set
            eps_rel2 = max(cfg.cluster_eps, 1e-8 * Rmax)
            taken2 = np.zeros(len(roots), dtype=bool)
            fused2 = []
            for i in range(len(roots)):
                if taken2[i]:
                    continue
                grp = [roots[i]]
                taken2[i] = True
                for j in range(i+1, len(roots)):
                    if not taken2[j] and np.abs(roots[j] - roots[i]) < eps_rel2:
                        taken2[j] = True
                        grp.append(roots[j])
                fused2.append(np.mean(grp))
            roots = np.array(fused2, dtype=np.complex128)
            final_resid = np.abs(horner_eval(roots, a))
            # Success if all residuals small and we recovered degree-many roots
            if len(roots) == n and np.all(final_resid < max(cfg.tol, 1e-10)):
                success = True

    info: Dict[str, Any] = dict(
        success=success,
        iterations=iterations,
        final_walkers=z.copy(),
        final_residuals=np.abs(horner_eval(z, a)),
        roots_residuals=final_resid,
        config=asdict(cfg),
        degree=n,
        n_walkers=N,
    )
    if cfg.keep_history:
        traj_arr = np.asarray(traj)  # (T,N)
    else:
        traj_arr = None
    return roots, traj_arr, info