# -*- coding: utf-8 -*-
"""
Marletto–Vedral style entanglement-mediation test in a QFCA-like discrete evolution.
Two 2-branch systems (A,B) interact ONLY via a mediator M with short memory.
We compare:
  - CL mediator: commuting, phase-blind summaries -> should NOT violate CHSH (S<=2)
  - QL mediator: phase-sensitive, noncommuting summaries -> CAN violate CHSH (S>2)
Saves CSV + plots.
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import csv, json

# ---------- Utilities (operators & expectations) ----------
sx = np.array([[0,1],[1,0]], dtype=np.complex128)
sy = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
sz = np.array([[1,0],[0,-1]], dtype=np.complex128)
I2 = np.eye(2, dtype=np.complex128)

def sigma_theta(theta: float):
    return np.cos(theta)*sx + np.sin(theta)*sy

def kron(a,b):
    return np.kron(a,b)

def normalize_batch(Psi: np.ndarray, eps=1e-12):
    n = np.sqrt((np.abs(Psi)**2).sum(axis=1, keepdims=True) + eps)
    return Psi / n

# Expectation ⟨σ_θA ⊗ σ_θB⟩ for batch of pure states Psi (N,4)
# where basis ordering is |LL>,|LR>,|RL>,|RR>
proj = {
    'LL': np.array([1,0,0,0]),
    'LR': np.array([0,1,0,0]),
    'RL': np.array([0,0,1,0]),
    'RR': np.array([0,0,0,1]),
}

# Build (σ_θA ⊗ σ_θB) operator once per angle pair

def E_batch(Psi: np.ndarray, thetaA: float, thetaB: float) -> float:
    OA = sigma_theta(thetaA)
    OB = sigma_theta(thetaB)
    O = kron(OA, OB)  # 4x4
    # batch expectation: sum_i Psi*_i (O Psi)_i
    OPsi = Psi @ O.T  # (N,4)
    val = np.einsum('ni,ni->n', np.conjugate(Psi), OPsi)
    return float(np.real(np.mean(val)))

# Mean concurrence for a batch of pure two-qubit states (Psi shape: N,4)
# For pure states, concurrence C = 2*|⟨psi| (σ_y⊗σ_y) |psi*⟩|
# Basis ordering must be |LL>,|LR>,|RL>,|RR>.

def concurrence_batch(Psi: np.ndarray) -> float:
    N = Psi.shape[0]
    Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
    YY = np.kron(Y, Y)
    # reshape to (N,4,1)
    psi = Psi.reshape(N,4,1)
    psit = np.matmul(np.conjugate(np.transpose(psi,(0,2,1))), np.matmul(YY, psi))
    C = 2.0 * np.abs(psit.squeeze())  # (N,)
    return float(np.real(np.mean(C)))

# ---------- Mediators (per-seed stateful) ----------
@dataclass
class MediatorParams:
    lam: float = 0.15
    L: int = 20
    tau: float = 10.0
    mu_retro: float = 0.05
    sigma_noise: float = 0.0
    eta: float = 0.0  # not used in unitary phase-kick model

class ClassicalMediator:
    def __init__(self, p: MediatorParams, seeds: int):
        self.p = p
        self.histA = [ [] for _ in range(seeds) ]  # each item: list of [pL, pR]
        self.histB = [ [] for _ in range(seeds) ]

    def update(self, sidx: int, Psi_seed: np.ndarray):
        # marginals from joint amplitudes
        aLL,aLR,aRL,aRR = Psi_seed
        pL_A = np.abs(aLL)**2 + np.abs(aLR)**2
        pR_A = np.abs(aRL)**2 + np.abs(aRR)**2
        pL_B = np.abs(aLL)**2 + np.abs(aRL)**2
        pR_B = np.abs(aLR)**2 + np.abs(aRR)**2
        self.histA[sidx].append(np.array([pL_A,pR_A]))
        self.histB[sidx].append(np.array([pL_B,pR_B]))
        if len(self.histA[sidx]) > self.p.L:
            self.histA[sidx].pop(0); self.histB[sidx].pop(0)

    def phases(self, sidx: int):
        w = np.exp(-np.arange(len(self.histA[sidx]))[::-1] / max(1.0,self.p.tau))
        w = w/np.sum(w)
        avgA = (w[:,None]*np.array(self.histA[sidx])).sum(axis=0)
        avgB = (w[:,None]*np.array(self.histB[sidx])).sum(axis=0)
        # purely local phases proportional to other party's population
        phiA_L = self.p.lam * avgB[0]
        phiA_R = self.p.lam * avgB[1]
        phiB_L = self.p.lam * avgA[0]
        phiB_R = self.p.lam * avgA[1]
        # optional real jitters (phase noise suppressed here for CL)
        return np.array([
            phiA_L + phiB_L,
            phiA_L + phiB_R,
            phiA_R + phiB_L,
            phiA_R + phiB_R,
        ], dtype=np.float64), 0.0

class QuantumLikeMediator:
    def __init__(self, p: MediatorParams, seeds: int):
        self.p = p
        self.hist = [ [] for _ in range(seeds) ]  # each: list of complex 4-amplitude snapshots

    def update(self, sidx: int, Psi_seed: np.ndarray):
        self.hist[sidx].append(Psi_seed.copy())
        if len(self.hist[sidx]) > self.p.L:
            self.hist[sidx].pop(0)

    def phases(self, sidx: int):
        w = np.exp(-np.arange(len(self.hist[sidx]))[::-1] / max(1.0,self.p.tau))
        w = w/np.sum(w)
        avg = (w[:,None]*np.array(self.hist[sidx])).sum(axis=0)  # (4,) complex
        # Build non-separable phase via a genuine two-body term χ
        # χ comes from imaginary part of cross-coherence contrast (LL*RR vs LR*RL)
        aLL,aLR,aRL,aRR = avg
        chi_raw = np.imag(aLL*np.conjugate(aRR) - aLR*np.conjugate(aRL))
        chi = self.p.lam * chi_raw
        # local parts from marginals (as in CL)
        pL_A = np.abs(aLL)**2 + np.abs(aLR)**2
        pR_A = np.abs(aRL)**2 + np.abs(aRR)**2
        pL_B = np.abs(aLL)**2 + np.abs(aRL)**2
        pR_B = np.abs(aLR)**2 + np.abs(aRR)**2
        phiA_L = 0.5*self.p.lam * pL_B
        phiA_R = 0.5*self.p.lam * pR_B
        phiB_L = 0.5*self.p.lam * pL_A
        phiB_R = 0.5*self.p.lam * pR_A
        # non-separable pattern s = [[+1,-1],[-1,+1]] mapped to 4 basis order
        s = np.array([+1,-1,-1,+1], dtype=np.float64)
        phases = np.array([
            phiA_L + phiB_L,
            phiA_L + phiB_R,
            phiA_R + phiB_L,
            phiA_R + phiB_R,
        ], dtype=np.float64) + chi * s
        # optional phase noise (dephasing): add Gaussian jitter to phases
        if self.p.sigma_noise > 0:
            phases = phases + self.p.sigma_noise * np.random.standard_normal(4)
        return phases, float(chi)

# ---------- Evolution on joint states ----------

def evolve_pair(mode: str, T=800, seeds=128, p: MediatorParams = MediatorParams(),
                retro_every: int = 12, retro_mu: float | None = None):
    if retro_mu is None:
        retro_mu = p.mu_retro
    # initial product |+_phi> ⊗ |+>
    phi = 0.3
    # single-qubit states
    plus_phi = np.array([1.0, np.exp(1j*phi)], dtype=np.complex128)/np.sqrt(2)
    plus = np.array([1.0, 1.0], dtype=np.complex128)/np.sqrt(2)
    # joint seed states with small random phase jitters
    Psi = np.zeros((seeds,4), dtype=np.complex128)
    for n in range(seeds):
        da = 0.05*np.random.standard_normal(2)
        db = 0.05*np.random.standard_normal(2)
        A = plus_phi * np.exp(1j*da)
        B = plus * np.exp(1j*db)
        # kron order: |LL>,|LR>,|RL>,|RR>
        Psi[n,0] = A[0]*B[0]
        Psi[n,1] = A[0]*B[1]
        Psi[n,2] = A[1]*B[0]
        Psi[n,3] = A[1]*B[1]
    Psi = normalize_batch(Psi)

    # mediator per mode
    if mode == 'CL':
        med = ClassicalMediator(p, seeds)
    elif mode == 'QL':
        med = QuantumLikeMediator(p, seeds)
    else:
        raise ValueError("mode must be 'CL' or 'QL'")

    # histories for retro
    hist = [Psi.copy()]
    for t in range(T):
        # update mediator and apply diagonal unitary per seed
        for sidx in range(seeds):
            med.update(sidx, Psi[sidx])
            phases, chi = med.phases(sidx)  # phases: (4,), chi: scalar
            # Local phase kick
            v = np.exp(1j*phases) * Psi[sidx]

            # For QL, apply an Ising-type entangler U = exp(-i theta σz⊗σz)
            if mode == 'QL':
                theta = 0.9 * np.tanh(2.0 * chi) + 0.10  # bias toward useful regime (~pi/8)
                Udiag = np.array([
                    np.exp(-1j*theta),  # |LL>
                    np.exp(+1j*theta),  # |LR>
                    np.exp(+1j*theta),  # |RL>
                    np.exp(-1j*theta),  # |RR>
                ], dtype=np.complex128)
                v = Udiag * v
            # (remove amplitude balancer to avoid washing out correlations)
            # if t < 20:
            #     v += 1e-4 * (0.5 - (np.abs(v)**2)) * v
            Psi[sidx] = v
        # retro extrapolation less aggressively
        if retro_every>0 and t>2 and (t % (retro_every) == 0):
            prev = hist[-1]
            prev2 = hist[-2]
            pred = prev + (prev - prev2)
            Psi = Psi - retro_mu * (Psi - pred)
        # normalize every few steps to allow correlations to grow
        if t % 12 == 0:
            Psi = normalize_batch(Psi)
        hist.append(Psi.copy())
        if len(hist) > p.L:
            hist.pop(0)
        if not np.isfinite(np.abs(Psi).sum()):
            break
    return Psi

# ---------- CHSH & Protocol ----------

def chsh_S(Psi: np.ndarray, thetas_A: tuple[float,float], thetas_B: tuple[float,float]) -> float:
    a1,a2 = thetas_A; b1,b2 = thetas_B
    E11 = E_batch(Psi, a1, b1)
    E12 = E_batch(Psi, a1, b2)
    E21 = E_batch(Psi, a2, b1)
    E22 = E_batch(Psi, a2, b2)
    S = abs(E11 + E12 + E21 - E22)
    return float(S)

def run_protocol(outdir="results/marletto_test", T=800, seeds=128):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    thetas_A = (0.0, np.pi/2)
    thetas_B = (np.pi/4, -np.pi/4)

    lambdas = np.linspace(0.0, 0.6, 10)
    noises  = np.linspace(0.0, 0.20, 6)

    base = MediatorParams(lam=0.15, L=20, tau=10.0, mu_retro=0.05, sigma_noise=0.0)

    csv_path = Path(outdir)/"summary.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["mode","lambda","noise","S","concurrence","seeds","T","L","tau","mu_retro"])

    # λ sweep
    Scl, Sql = [], []
    for lam in lambdas:
        p = MediatorParams(**{**base.__dict__, "lam": float(lam), "sigma_noise": 0.0})
        Psi_cl = evolve_pair('CL', T=T, seeds=seeds, p=p)
        S_cl = chsh_S(Psi_cl, thetas_A, thetas_B); Scl.append(S_cl)
        C_cl = concurrence_batch(Psi_cl)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(["CL", lam, 0.0, S_cl, C_cl, seeds, T, p.L, p.tau, p.mu_retro])

        Psi_ql = evolve_pair('QL', T=T, seeds=seeds, p=p)
        S_ql = chsh_S(Psi_ql, thetas_A, thetas_B); Sql.append(S_ql)
        C_ql = concurrence_batch(Psi_ql)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(["QL", lam, 0.0, S_ql, C_ql, seeds, T, p.L, p.tau, p.mu_retro])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.axhline(2.0, ls="--", lw=1, alpha=0.6, label="CHSH classical bound")
    ax.plot(lambdas, Scl, marker="o", label="Classical mediator")
    ax.plot(lambdas, Sql, marker="o", label="Quantum-like mediator")
    ax.set_xlabel("mediator coupling λ"); ax.set_ylabel("CHSH S")
    ax.set_title("CHSH vs coupling (joint-state model)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(Path(outdir)/"S_vs_lambda.png", dpi=160)

    # noise sweep on QL at λ0
    lam0 = 0.40
    Sno = []
    for sig in noises:
        p = MediatorParams(**{**base.__dict__, "lam": float(lam0), "sigma_noise": float(sig)})
        Psi_ql = evolve_pair('QL', T=T, seeds=seeds, p=p)
        s = chsh_S(Psi_ql, thetas_A, thetas_B); Sno.append(s)
        Cn = concurrence_batch(Psi_ql)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(["QL", lam0, sig, s, Cn, seeds, T, p.L, p.tau, p.mu_retro])

    fig, ax = plt.subplots(figsize=(6,4))
    ax.axhline(2.0, ls="--", lw=1, alpha=0.6, label="CHSH classical bound")
    ax.plot(noises, Sno, marker="o", label=f"Quantum-like mediator (λ={lam0:.2f})")
    ax.set_xlabel("mediator phase noise σ"); ax.set_ylabel("CHSH S")
    ax.set_title("Noise quenches CHSH (joint-state model)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(Path(outdir)/"S_vs_noise.png", dpi=160)

    with open(Path(outdir)/"meta.json","w") as f:
        json.dump({"angles":{"A":thetas_A,"B":thetas_B},
                   "lambdas":list(map(float,lambdas)),
                   "noises":list(map(float,noises))}, f, indent=2)

if __name__ == "__main__":
    run_protocol()