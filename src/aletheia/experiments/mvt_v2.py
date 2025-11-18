# -*- coding: utf-8 -*-
"""
Marletto–Vedral test - STRONG ENTANGLEMENT VERSION
Running in background while you're in meetings :)
"""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path
import csv, json

# ---------- Utilities ----------
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

def E_batch(Psi: np.ndarray, thetaA: float, thetaB: float) -> float:
    OA = sigma_theta(thetaA)
    OB = sigma_theta(thetaB)
    O = kron(OA, OB)
    OPsi = Psi @ O.T
    val = np.einsum('ni,ni->n', np.conjugate(Psi), OPsi)
    return float(np.real(np.mean(val)))

def concurrence_batch(Psi: np.ndarray) -> float:
    N = Psi.shape[0]
    Y = np.array([[0,-1j],[1j,0]], dtype=np.complex128)
    YY = np.kron(Y, Y)
    psi = Psi.reshape(N,4,1)
    psit = np.matmul(np.conjugate(np.transpose(psi,(0,2,1))), np.matmul(YY, psi))
    C = 2.0 * np.abs(psit.squeeze())
    return float(np.real(np.mean(C)))

def purity_batch(Psi_batch: np.ndarray) -> float:
    purities = []
    for i in range(Psi_batch.shape[0]):
        psi = Psi_batch[i]
        purity = np.abs(np.vdot(psi, psi))**2
        purities.append(purity)
    return float(np.mean(purities))

def analyze_final_state(Psi_batch: np.ndarray):
    N = Psi_batch.shape[0]
    rho_avg = np.zeros((4,4), dtype=np.complex128)
    for i in range(N):
        psi = Psi_batch[i].reshape(4,1)
        rho_avg += psi @ psi.conj().T
    rho_avg /= N
    
    print("\n=== STATE ANALYSIS ===")
    print("Density matrix diagonal:", np.round(np.diag(rho_avg), 4))
    
    bell_plus = np.array([1,0,0,1])/np.sqrt(2)
    bell_minus = np.array([1,0,0,-1])/np.sqrt(2)
    psi_plus = np.array([0,1,1,0])/np.sqrt(2)
    psi_minus = np.array([0,1,-1,0])/np.sqrt(2)
    
    fid_phi_plus = np.abs(bell_plus @ rho_avg @ bell_plus.conj())
    fid_phi_minus = np.abs(bell_minus @ rho_avg @ bell_minus.conj())
    fid_psi_plus = np.abs(psi_plus @ rho_avg @ psi_plus.conj())
    fid_psi_minus = np.abs(psi_minus @ rho_avg @ psi_minus.conj())
    
    print(f"Fidelity with |Φ⁺⟩: {fid_phi_plus:.4f}")
    print(f"Fidelity with |Φ⁻⟩: {fid_phi_minus:.4f}")
    print(f"Fidelity with |Ψ⁺⟩: {fid_psi_plus:.4f}")
    print(f"Fidelity with |Ψ⁻⟩: {fid_psi_minus:.4f}")
    
    return rho_avg

def find_optimal_angles(Psi_batch: np.ndarray, num_angles: int = 12):
    best_S = 0
    best_angles = None
    
    for a1 in np.linspace(0, np.pi, num_angles):
        for a2 in np.linspace(0, np.pi, num_angles):
            if abs(a1 - a2) < 0.2:
                continue
            for b1 in np.linspace(0, np.pi, num_angles):
                for b2 in np.linspace(0, np.pi, num_angles):
                    if abs(b1 - b2) < 0.2:
                        continue
                    S = chsh_S(Psi_batch, (a1, a2), (b1, b2))
                    if S > best_S:
                        best_S = S
                        best_angles = (a1, a2, b1, b2)
    
    print(f"Optimal S: {best_S:.4f} with angles A:({best_angles[0]:.3f},{best_angles[1]:.3f}) B:({best_angles[2]:.3f},{best_angles[3]:.3f})")
    return best_S, best_angles

# ---------- Mediators ----------
@dataclass
class MediatorParams:
    lam: float = 0.15
    L: int = 20
    tau: float = 10.0
    mu_retro: float = 0.05
    sigma_noise: float = 0.0
    eta: float = 0.0

class ClassicalMediator:
    def __init__(self, p: MediatorParams, seeds: int):
        self.p = p
        self.histA = [ [] for _ in range(seeds) ]
        self.histB = [ [] for _ in range(seeds) ]

    def update(self, sidx: int, Psi_seed: np.ndarray):
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
        phiA_L = self.p.lam * avgB[0]
        phiA_R = self.p.lam * avgB[1]
        phiB_L = self.p.lam * avgA[0]
        phiB_R = self.p.lam * avgA[1]
        return np.array([
            phiA_L + phiB_L,
            phiA_L + phiB_R,
            phiA_R + phiB_L,
            phiA_R + phiB_R,
        ], dtype=np.float64), 0.0

class QuantumLikeMediator:
    def __init__(self, p: MediatorParams, seeds: int):
        self.p = p
        self.hist = [ [] for _ in range(seeds) ]

    def update(self, sidx: int, Psi_seed: np.ndarray):
        self.hist[sidx].append(Psi_seed.copy())
        if len(self.hist[sidx]) > self.p.L:
            self.hist[sidx].pop(0)

    def phases(self, sidx: int):
        w = np.exp(-np.arange(len(self.hist[sidx]))[::-1] / max(1.0,self.p.tau))
        w = w/np.sum(w)
        avg = (w[:,None]*np.array(self.hist[sidx])).sum(axis=0)
        aLL,aLR,aRL,aRR = avg
        chi_raw = np.imag(aLL*np.conjugate(aRR) - aLR*np.conjugate(aRL))
        chi = self.p.lam * chi_raw
        
        pL_A = np.abs(aLL)**2 + np.abs(aLR)**2
        pR_A = np.abs(aRL)**2 + np.abs(aRR)**2
        pL_B = np.abs(aLL)**2 + np.abs(aRL)**2
        pR_B = np.abs(aLR)**2 + np.abs(aRR)**2
        phiA_L = 0.5*self.p.lam * pL_B
        phiA_R = 0.5*self.p.lam * pR_B
        phiB_L = 0.5*self.p.lam * pL_A
        phiB_R = 0.5*self.p.lam * pR_A
        
        s = np.array([+1,-1,-1,+1], dtype=np.float64)
        phases = np.array([
            phiA_L + phiB_L,
            phiA_L + phiB_R,
            phiA_R + phiB_L,
            phiA_R + phiB_R,
        ], dtype=np.float64) + chi * s
        
        if self.p.sigma_noise > 0:
            phases = phases + self.p.sigma_noise * np.random.standard_normal(4)
        return phases, float(chi)

# ---------- Evolution ----------
def evolve_pair(mode: str, T=800, seeds=128, p: MediatorParams = MediatorParams(),
                retro_every: int = 0, normalize_every: int = 50, 
                ent_strength: float = 1.2, initial_entangled: bool = False):
    
    # INITIAL STATE - OPTIONALLY ENTANGLED
    if initial_entangled:
        # Start with partially entangled state
        psi_entangled = np.array([0.7, 0.3j, 0.3j, 0.7], dtype=np.complex128)
        Psi = np.tile(psi_entangled, (seeds, 1))
        Psi = normalize_batch(Psi)
        print("Starting with partially entangled state")
    else:
        # Original product state
        phi = 0.3
        plus_phi = np.array([1.0, np.exp(1j*phi)], dtype=np.complex128)/np.sqrt(2)
        plus = np.array([1.0, 1.0], dtype=np.complex128)/np.sqrt(2)
        Psi = np.zeros((seeds,4), dtype=np.complex128)
        for n in range(seeds):
            da = 0.05*np.random.standard_normal(2)
            db = 0.05*np.random.standard_normal(2)
            A = plus_phi * np.exp(1j*da)
            B = plus * np.exp(1j*db)
            Psi[n,0] = A[0]*B[0]
            Psi[n,1] = A[0]*B[1]
            Psi[n,2] = A[1]*B[0]
            Psi[n,3] = A[1]*B[1]
        Psi = normalize_batch(Psi)

    if mode == 'CL':
        med = ClassicalMediator(p, seeds)
    elif mode == 'QL':
        med = QuantumLikeMediator(p, seeds)
    else:
        raise ValueError("mode must be 'CL' or 'QL'")

    # Track chi values for debugging
    chi_history = []
    
    for t in range(T):
        chi_step = []
        for sidx in range(seeds):
            med.update(sidx, Psi[sidx])
            phases, chi = med.phases(sidx)
            chi_step.append(chi)
            
            v = np.exp(1j*phases) * Psi[sidx]

            if mode == 'QL':
                # STRONG ENTANGLEMENT
                theta = ent_strength * np.tanh(2.0 * chi)
                Udiag = np.array([
                    np.exp(-1j*theta),
                    np.exp(+1j*theta), 
                    np.exp(+1j*theta),
                    np.exp(-1j*theta),
                ], dtype=np.complex128)
                v = Udiag * v

            Psi[sidx] = v
            
        chi_history.append(np.mean(chi_step))
                
        if normalize_every > 0 and (t % normalize_every == 0):
            Psi = normalize_batch(Psi)
                
        if not np.isfinite(np.abs(Psi).sum()):
            print(f"Numerical instability at step {t}")
            break
            
    return Psi, chi_history

def chsh_S(Psi: np.ndarray, thetas_A: tuple[float,float], thetas_B: tuple[float,float]) -> float:
    a1,a2 = thetas_A; b1,b2 = thetas_B
    E11 = E_batch(Psi, a1, b1)
    E12 = E_batch(Psi, a1, b2)
    E21 = E_batch(Psi, a2, b1)
    E22 = E_batch(Psi, a2, b2)
    S = abs(E11 + E12 + E21 - E22)
    return float(S)

def run_strong_tests():
    """Run multiple tests with increasing entanglement strength"""
    Path("results/marletto_test").mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("MARLETTO-VEDRAL - STRONG ENTANGLEMENT TESTS")
    print("="*70)
    
    results = []
    
    # Test different entanglement strengths
    ent_strengths = [0.5, 1.0, 1.5, 2.0, 3.0]
    couplings = [0.4, 0.8, 1.2]
    
    for ent_str in ent_strengths:
        for lam in couplings:
            print(f"\n>>> Testing: λ={lam}, entanglement strength={ent_str}")
            
            p = MediatorParams(lam=lam, L=5, tau=3.0, sigma_noise=0.0, mu_retro=0.0)
            
            Psi_ql, chi_hist = evolve_pair('QL', T=400, seeds=64, p=p, 
                                         ent_strength=ent_str, initial_entangled=False)
            
            purity = purity_batch(Psi_ql)
            concur = concurrence_batch(Psi_ql)
            S_std = chsh_S(Psi_ql, (0.0, np.pi/2), (np.pi/4, -np.pi/4))
            S_opt, angles = find_optimal_angles(Psi_ql, num_angles=8)
            
            avg_chi = np.mean(chi_hist) if chi_hist else 0.0
            
            print(f"   Concurrence: {concur:.4f}, Purity: {purity:.6f}")
            print(f"   CHSH: std={S_std:.4f}, opt={S_opt:.4f}, avg_chi={avg_chi:.6f}")
            
            results.append({
                'lambda': lam, 'ent_strength': ent_str, 
                'concurrence': concur, 'S_optimal': S_opt, 'S_standard': S_std,
                'purity': purity, 'avg_chi': avg_chi
            })
            
            # Save progress after each test
            with open("results/marletto_test/strong_entanglement_results.csv", "w") as f:
                if results:
                    writer = csv.DictWriter(f, fieldnames=results[0].keys())
                    writer.writeheader()
                    writer.writerows(results)
            
            # If we find S > 2, celebrate!
            if S_opt > 2.0:
                print("🎉 🎉 🎉 CHSH VIOLATION DETECTED! 🎉 🎉 🎉")
                analyze_final_state(Psi_ql)
    
    # Plot results
    if results:
        plot_results(results)
    
    return results

def plot_results(results):
    """Plot the results of the parameter sweep"""
    lambdas = sorted(set(r['lambda'] for r in results))
    strengths = sorted(set(r['ent_strength'] for r in results))
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: CHSH vs entanglement strength for different lambdas
    for lam in lambdas:
        subset = [r for r in results if r['lambda'] == lam]
        strengths_sub = [r['ent_strength'] for r in subset]
        S_opt_sub = [r['S_optimal'] for r in subset]
        axes[0,0].plot(strengths_sub, S_opt_sub, 'o-', label=f'λ={lam}')
    axes[0,0].axhline(2.0, color='red', linestyle='--', label='Classical bound')
    axes[0,0].set_xlabel('Entanglement Strength')
    axes[0,0].set_ylabel('CHSH S (optimal)')
    axes[0,0].set_title('CHSH vs Entanglement Strength')
    axes[0,0].legend()
    axes[0,0].grid(alpha=0.3)
    
    # Plot 2: Concurrence vs entanglement strength
    for lam in lambdas:
        subset = [r for r in results if r['lambda'] == lam]
        strengths_sub = [r['ent_strength'] for r in subset]
        concur_sub = [r['concurrence'] for r in subset]
        axes[0,1].plot(strengths_sub, concur_sub, 'o-', label=f'λ={lam}')
    axes[0,1].set_xlabel('Entanglement Strength')
    axes[0,1].set_ylabel('Concurrence')
    axes[0,1].set_title('Concurrence vs Entanglement Strength')
    axes[0,1].legend()
    axes[0,1].grid(alpha=0.3)
    
    # Plot 3: CHSH vs concurrence
    for lam in lambdas:
        subset = [r for r in results if r['lambda'] == lam]
        concur_sub = [r['concurrence'] for r in subset]
        S_opt_sub = [r['S_optimal'] for r in subset]
        axes[1,0].plot(concur_sub, S_opt_sub, 'o-', label=f'λ={lam}')
    axes[1,0].axhline(2.0, color='red', linestyle='--', label='Classical bound')
    axes[1,0].set_xlabel('Concurrence')
    axes[1,0].set_ylabel('CHSH S (optimal)')
    axes[1,0].set_title('CHSH vs Concurrence')
    axes[1,0].legend()
    axes[1,0].grid(alpha=0.3)
    
    # Plot 4: Average chi vs parameters
    for lam in lambdas:
        subset = [r for r in results if r['lambda'] == lam]
        strengths_sub = [r['ent_strength'] for r in subset]
        chi_sub = [r['avg_chi'] for r in subset]
        axes[1,1].plot(strengths_sub, chi_sub, 'o-', label=f'λ={lam}')
    axes[1,1].set_xlabel('Entanglement Strength')
    axes[1,1].set_ylabel('Average Chi')
    axes[1,1].set_title('Mediator Chi vs Parameters')
    axes[1,1].legend()
    axes[1,1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/marletto_test/parameter_sweep_results.png', dpi=120, bbox_inches='tight')
    plt.close()
    
    print("\nPlots saved to results/marletto_test/parameter_sweep_results.png")

if __name__ == "__main__":
    print("Starting strong entanglement parameter sweep...")
    print("This will test multiple entanglement strengths and coupling parameters")
    print("Results will be saved automatically after each test")
    print("Check results/marletto_test/strong_entanglement_results.csv for progress")
    
    results = run_strong_tests()
    
    print("\n" + "="*70)
    print("PARAMETER SWEEP COMPLETE!")
    print("="*70)
    
    # Find best result
    if results:
        best = max(results, key=lambda x: x['S_optimal'])
        print(f"BEST RESULT: λ={best['lambda']}, ent_strength={best['ent_strength']}")
        print(f"  CHSH S = {best['S_optimal']:.4f}, Concurrence = {best['concurrence']:.4f}")
        
        if best['S_optimal'] > 2.0:
            print("🎉 SUCCESS: CHSH VIOLATION ACHIEVED! 🎉")
        else:
            print("💡 Close but no violation yet. Try even stronger parameters.")