import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from ns_qfca_solver import NavierStokesQFCA, NavierStokesParams
from initial_conditions import taylor_green_vortex

def test_taylor_green():
    """Validation test against exact solution"""
    print("\n" + "="*60)
    print("TEST: Taylor-Green Vortex Validation")
    print("="*60 + "\n")
    
    params = NavierStokesParams(
        N=128,
        L=2*np.pi,
        nu=0.01,
        hbar=0.05,
        tau_m=5.0,
        enable_memory=False,  # ← TURN OFF for validation
        kernel_type='exponential',
        # ---- Agency + coherence stabilization ----
        enable_agency=True,
        A_pause=1e-4,
        A_refuse=5e-4,
        A_reframe=2e-4,
        agency_prediction_factor=0.2,
        coherence_stabilization='adaptive_viscosity',
        C_crit=0.5,
        nu_boost_factor=10.0,
    )
    
    solver = NavierStokesQFCA(params)
    psi0 = taylor_green_vortex(solver, amplitude=1.0)
    
    # Run simulation
    T = 10.0
    dt = 0.01
    psi_final = solver.run(psi0, T, dt, diagnose_every=10)
    
    # Compare with exact solution
    t = np.array(solver.history['t'])
    E_qfca = np.array(solver.history['energy'])
    E_exact = E_qfca[0] * np.exp(-4 * params.nu * t)
    
    # Plot
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(t, E_exact, 'k--', linewidth=2, label='Exact')
    plt.plot(t, E_qfca, 'b-', linewidth=2, label='QFCA')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.yscale('log')
    plt.legend()
    plt.title('Energy Decay Validation')
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    error = np.abs(E_qfca - E_exact) / E_exact
    plt.plot(t, error, 'r-', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Relative Error')
    plt.yscale('log')
    plt.title('Validation Error')
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/taylor_green_validation.png', dpi=150)
    print("📊 Saved: figures/taylor_green_validation.png")
    
    # Full diagnostics
    solver.plot_diagnostics('../figures/taylor_green_diagnostics.png')
    
    # Summary
    print(f"\n✓ Validation complete")
    print(f"  Mean relative error: {np.mean(error):.6f}")
    print(f"  Min coherence: {min(solver.history['coherence']):.4f}")
    
    return solver

if __name__ == "__main__":
    test_taylor_green()
    plt.show()
