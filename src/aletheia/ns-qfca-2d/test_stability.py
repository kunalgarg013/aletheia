import sys
sys.path.append('src')

import numpy as np
import matplotlib.pyplot as plt
from ns_qfca_solver import NavierStokesQFCA, NavierStokesParams

def taylor_green_vortex(solver, amplitude=1.0):
    S = -amplitude * np.sin(solver.X) * np.sin(solver.Y) * solver.p.hbar
    rho = np.ones_like(S)
    return np.sqrt(rho) * np.exp(1j * S / solver.p.hbar)

# Test with ADAPTIVE VISCOSITY stabilization
params = NavierStokesParams(
    N=128,
    L=2*np.pi,
    nu=0.01,
    hbar=0.01,
    tau_m=50.0,
    enable_memory=True,
    kernel_type='exponential',
    coherence_stabilization='adaptive_viscosity',  # ← KEY
    C_crit=0.5,
    nu_boost_factor=10.0,
    # ---- Agency gating ----
    enable_agency=True,
    A_pause=1e-4,
    A_refuse=5e-4,
    A_reframe=2e-4,
    agency_prediction_factor=0.2,
)

solver = NavierStokesQFCA(params)
psi0 = taylor_green_vortex(solver, amplitude=1.0)

print("Testing with COHERENCE STABILIZATION...")
psi_final = solver.run(psi0, T=10.0, dt=0.01, diagnose_every=10)

# Validation
t = np.array(solver.history['t'])
E_qfca = np.array(solver.history['energy'])
E_exact = E_qfca[0] * np.exp(-4 * params.nu * t)
C_phi = np.array(solver.history['coherence'])
nu_eff = np.array(solver.history['nu_effective'])

print(f"\n📊 Validation Results:")
print(f"  Min coherence: {np.min(C_phi):.4f} (should stay > 0.5)")
print(f"  Max nu_eff: {np.max(nu_eff):.4f} (shows stabilization kicked in)")
print(f"  Energy ratio: {E_qfca[-1]/E_exact[-1]:.4f}")

# Plot with stabilization diagnostics
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Energy
ax = axes[0, 0]
ax.plot(t, E_exact, 'k--', linewidth=2, label='Exact')
ax.plot(t, E_qfca, 'b-', linewidth=2, label='QFCA Stabilized')
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_yscale('log')
ax.legend()
ax.set_title('Energy Decay with Stabilization')
ax.grid(alpha=0.3)

# Coherence
ax = axes[0, 1]
ax.plot(t, C_phi, 'g-', linewidth=2)
ax.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='$C_\\phi^c$')
ax.set_xlabel('Time')
ax.set_ylabel('Coherence')
ax.set_title('Coherence (Should Stay Above Threshold)')
ax.legend()
ax.grid(alpha=0.3)

# Effective viscosity
ax = axes[1, 0]
ax.plot(t, nu_eff, 'purple', linewidth=2)
ax.axhline(params.nu, color='k', linestyle='--', alpha=0.5, label='Base $\\nu$')
ax.set_xlabel('Time')
ax.set_ylabel('$\\nu_{eff}$')
ax.set_title('Adaptive Viscosity (Stabilization Response)')
ax.legend()
ax.grid(alpha=0.3)

# Phase diagram
ax = axes[1, 1]
scatter = ax.scatter(C_phi, nu_eff, c=t, cmap='viridis', s=20)
ax.set_xlabel('Coherence $C_\\phi$')
ax.set_ylabel('$\\nu_{eff}$')
ax.set_title('Stabilization Phase Diagram')
ax.grid(alpha=0.3)
plt.colorbar(scatter, ax=ax, label='Time')

plt.tight_layout()
plt.savefig('stabilized_validation.png', dpi=150)
plt.show()

solver.plot_diagnostics('stabilized_diagnostics.png')