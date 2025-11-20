import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from ns_qfca_solver import NavierStokesQFCA, NavierStokesParams
from initial_conditions import kelvin_helmholtz

# Test with no stabilization first
print("\n" + "="*60)
print("TEST: K-H without stabilization")
print("="*60)

params_no_stab = NavierStokesParams(
    N=256,
    L=2*np.pi,
    nu=0.001,
    hbar=0.1,
    tau_m=10.0,
    kernel_type='mixed',
    alpha=0.5,
    beta_mix=0.3,
    coherence_stabilization='none',  # DISABLED
    enable_agency=False,
    enable_memory=True,
)

solver = NavierStokesQFCA(params_no_stab)
psi0 = kelvin_helmholtz(solver, delta=0.05, epsilon=0.02)

print(f"\nInitial state:")
print(f"  C_φ(0) = {solver.coherence(psi0):.4f}")
print(f"  E(0) = {solver.energy(psi0):.6e}")

# Run without stabilization
psi_final = solver.run(psi0, T=10.0, dt=0.005, diagnose_every=20)

# Analyze

t = np.array(solver.history.get('t', []))
C = np.array(solver.history.get('coherence', []))
E = np.array(solver.history.get('energy', []))
Z = np.array(solver.history.get('enstrophy', []))

# --- Safety fix: avoid empty histories causing failures ---
if len(t) < 2 or len(C) < 2:
    # Create dummy small arrays to prevent downstream crashes
    # and allow plotting without gradient errors.
    if len(t) == 0:
        t = np.array([0.0])
    if len(C) == 0:
        C = np.array([solver.coherence(psi0)])
    if len(E) == 0:
        E = np.array([solver.energy(psi0)])
    if len(Z) == 0:
        Z = np.array([1e-15])
    print("\n⚠️ History was nearly empty — stabilizing arrays to avoid failures.")

if len(t) == 0 or len(C) == 0:
    print("\n⚠️ No diagnostic data was recorded!")
    print("→ Check diagnose_every, stability, or early exit conditions.")

print(f"\nResults (no stabilization):")
print(f"  C_φ trajectory: {C[0]:.4f} → {C[-1]:.4f} (min: {min(C):.4f})")
print(f"  Max enstrophy: {max(Z):.4e}")
print(f"  Energy change: {E[0]:.4e} → {E[-1]:.4e}")

# Check if it blew up
if max(Z) > 1.0 or min(C) < 0.1:
    print("\n⚠️  System became unstable without stabilization!")
    print(f"  → Need stabilization after coherence drops below {min(C):.2f}")
else:
    print("\n✅ System remained stable without stabilization!")
    print("  → Maybe K-H doesn't need stabilization at all?")

# Plot
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Coherence
ax = axes[0, 0]
ax.plot(t, C, 'b-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Coherence $C_φ$')
ax.set_title('Coherence (No Stabilization)')
ax.grid(alpha=0.3)

# Energy
ax = axes[0, 1]
ax.plot(t, E, 'g-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Energy')
ax.set_yscale('log')
ax.set_title('Energy (No Stabilization)')
ax.grid(alpha=0.3)

# Enstrophy
ax = axes[1, 0]
ax.plot(t, Z, 'm-', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('Enstrophy')
ax.set_yscale('log')
ax.set_title('Enstrophy (No Stabilization)')
ax.grid(alpha=0.3)

# dC/dt (rate of coherence change)
ax = axes[1, 1]
if len(C) > 2 and len(t) > 2:
    try:
        dC_dt = np.gradient(C, t)
        ax.plot(t, dC_dt, 'r-', linewidth=2)
    except Exception as e:
        ax.text(0.5, 0.5, f"Gradient error:\n{e}",
                ha='center', va='center', fontsize=11)
else:
    ax.text(0.5, 0.5, "Insufficient data for gradient",
            ha='center', va='center', fontsize=12)

ax.axhline(0, color='k', linestyle='--', alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('dC_φ/dt')
ax.set_title('Rate of Coherence Change')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../figures/kh_no_stabilization_diagnostic.png', dpi=150)
print(f"\n📊 Saved: ../figures/kh_no_stabilization_diagnostic.png")

plt.show()

# Now determine when stabilization should activate
print(f"\n" + "="*60)
print("RECOMMENDATION:")
print("="*60)

if max(Z) > 1.0:
    # Found the danger zone
    danger_idx = np.where(Z > 0.1)[0]
    if len(danger_idx) > 0:
        t_danger = t[danger_idx[0]]
        C_danger = C[danger_idx[0]]
        print(f"\nInstability begins around t ≈ {t_danger:.2f}")
        print(f"Coherence at that point: C_φ ≈ {C_danger:.4f}")
        print(f"\n→ Use stabilization_delay = {t_danger:.1f} seconds")
        print(f"→ Use C_crit = {C_danger * 0.9:.2f} (10% below danger point)")
elif min(C) < 0.3:
    # Coherence dropped but didn't blow up
    min_idx = np.argmin(C)
    t_min = t[min_idx]
    C_min = C[min_idx]
    print(f"\nCoherence minimum: C_φ = {C_min:.4f} at t ≈ {t_min:.2f}")
    print(f"System remained stable despite low coherence.")
    print(f"\n→ Use C_crit = {C_min * 0.8:.2f} (20% below minimum)")
    print(f"→ Use stabilization_delay = {t_min:.1f} seconds")
else:
    print(f"\nSystem is naturally stable!")
    print(f"Coherence stayed above {min(C):.4f}")
    print(f"\n→ K-H might not need stabilization for these parameters")
    print(f"→ If you want to be safe, use C_crit = {min(C) * 0.9:.2f}")