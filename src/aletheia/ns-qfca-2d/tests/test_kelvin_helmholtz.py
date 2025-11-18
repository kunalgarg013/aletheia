import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from ns_qfca_solver import NavierStokesQFCA, NavierStokesParams
from initial_conditions import kelvin_helmholtz

def test_kelvin_helmholtz():
    """Test coherence during turbulence onset"""
    print("\n" + "="*60)
    print("TEST: Kelvin-Helmholtz Instability")
    print("="*60 + "\n")
    
    params = NavierStokesParams(
        N=256,
        L=2*np.pi,
        nu=0.001,
        hbar=0.1,
        tau_m=10.0,
        kernel_type='mixed',
        alpha=0.5,
        beta_mix=0.3,
        # --- Agency + stabilization ---
        enable_agency=True,
        A_pause=5e-5,
        A_refuse=2e-4,
        A_reframe=1e-4,
        agency_prediction_factor=0.2,
        coherence_stabilization='adaptive_viscosity',
        C_crit=0.6,
        nu_boost_factor=12.0,
        enable_memory=True,
    )
    
    solver = NavierStokesQFCA(params)
    psi0 = kelvin_helmholtz(solver, delta=0.05, epsilon=0.02)
    
    # Run
    T = 30.0
    dt = 0.005
    psi_final = solver.run(psi0, T, dt, diagnose_every=20)
    
    # Analyze
    t = np.array(solver.history['t'])
    C = np.array(solver.history['coherence'])
    
    # Find transition
    transition_idx = np.where(C < 0.6)[0]
    if len(transition_idx) > 0:
        t_transition = t[transition_idx[0]]
        print(f"\n📉 Coherence transition at t ≈ {t_transition:.2f}")
        print(f"   C_φ: {C[0]:.3f} → {C[transition_idx[0]]:.3f}")
    
    # Diagnostics
    solver.plot_diagnostics('../figures/kelvin_helmholtz_diagnostics.png')
    
    print(f"\n✓ K-H test complete")
    print(f"  Min coherence: {min(C):.4f}")
    print(f"  Final coherence: {C[-1]:.4f}")
    
    return solver

if __name__ == "__main__":
    test_kelvin_helmholtz()
    plt.show()
