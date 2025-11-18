import sys
sys.path.append('../src')

import numpy as np
import matplotlib.pyplot as plt
from ns_qfca_solver import NavierStokesQFCA, NavierStokesParams
from initial_conditions import random_vorticity

def test_kernel_comparison():
    """Compare all kernel types"""
    print("\n" + "="*60)
    print("TEST: Memory Kernel Comparison")
    print("="*60 + "\n")
    
    kernel_types = ['exponential', 'powerlaw', 'mittag_leffler', 'mixed']
    results = {}
    
    for kernel_type in kernel_types:
        print(f"\n--- Testing {kernel_type} kernel ---")
        
        params = NavierStokesParams(
            N=128,
            L=2*np.pi,
            nu=0.01,
            hbar=0.1,
            tau_m=5.0,
            kernel_type=kernel_type,
            alpha=0.5,
            beta_mix=0.3
        )
        
        solver = NavierStokesQFCA(params)
        np.random.seed(42)  # Same IC for all
        psi0 = random_vorticity(solver, k_peak=8, energy=1.0)
        
        T = 20.0
        dt = 0.01
        solver.run(psi0, T, dt, diagnose_every=20)
        
        results[kernel_type] = {
            't': np.array(solver.history['t']),
            'coherence': np.array(solver.history['coherence']),
            'min_coherence': min(solver.history['coherence'])
        }
    
    # Comparative plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    for kernel_type, data in results.items():
        ax.plot(data['t'], data['coherence'], linewidth=2, label=kernel_type)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time')
    ax.set_ylabel('Coherence $C_\\phi$')
    ax.set_title('Coherence Evolution by Kernel Type')
    ax.legend()
    ax.grid(alpha=0.3)
    
    ax = axes[1]
    min_coherences = [data['min_coherence'] for data in results.values()]
    ax.bar(kernel_types, min_coherences, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
    ax.axhline(0.5, color='k', linestyle='--', alpha=0.5, label='$C_\\phi^c$')
    ax.set_ylabel('Minimum $C_\\phi$')
    ax.set_title('Coherence Lower Bound by Kernel')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../figures/kernel_comparison.png', dpi=150)
    print("📊 Saved: figures/kernel_comparison.png")
    
    # Summary
    print("\n" + "="*60)
    print("KERNEL COMPARISON SUMMARY")
    print("="*60)
    for kernel_type, data in results.items():
        print(f"{kernel_type:20s} | min C_φ = {data['min_coherence']:.4f}")
    print("="*60)

if __name__ == "__main__":
    test_kernel_comparison()
    plt.show()
