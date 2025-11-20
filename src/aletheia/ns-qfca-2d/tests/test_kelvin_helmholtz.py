# save as: test_kelvin_helmholtz_suite.py

import sys
sys.path.append('../src')

# Toggle Hamiltonian mode for all tests
USE_HAMILTONIAN = False

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime
from ns_qfca_solver import NavierStokesQFCA, NavierStokesParams
from initial_conditions import kelvin_helmholtz

class KHTestSuite:
    """Systematic Kelvin-Helmholtz parameter exploration"""
    
    def __init__(self, output_dir='../results/kh_suite'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
    def run_single_test(self, test_name, params_dict, ic_params, sim_params):
        """Run one K-H test with given parameters"""
        print(f"\n{'='*70}")
        print(f"TEST: {test_name}")
        print(f"{'='*70}")
        
        # Create params
        params = NavierStokesParams(**{**params_dict, 'use_hamiltonian': USE_HAMILTONIAN})
        
        # Print key parameters
        print(f"\nConfiguration:")
        print(f"  C_crit = {params.C_crit}")
        print(f"  nu_boost = {params.nu_boost_factor}")
        print(f"  Initial conditions: delta={ic_params['delta']}, epsilon={ic_params['epsilon']}")
        print(f"  Simulation: T={sim_params['T']}, dt={sim_params['dt']}")
        print(f"  Stabilization: {params.coherence_stabilization}")
        print(f"  Agency: {params.enable_agency}")
        print(f"  Hamiltonian enabled: {params.use_hamiltonian}")
        
        # Setup solver
        solver = NavierStokesQFCA(params)
        psi0 = kelvin_helmholtz(solver, **ic_params)
        
        # Initial diagnostics
        C0 = solver.coherence(psi0)
        E0 = solver.energy(psi0)
        print(f"\nInitial state:")
        print(f"  C_φ(0) = {C0:.4f}")
        print(f"  E(0) = {E0:.6e}")
        
        # Run simulation
        T = sim_params['T']
        dt = sim_params['dt']
        
        try:
            psi_final = solver.run(psi0, T, dt, diagnose_every=20)
            success = True
            error_msg = None
        except Exception as e:
            print(f"\n⚠️  Simulation failed: {e}")
            success = False
            error_msg = str(e)
            psi_final = None
        
        # Extract results
        t = np.array(solver.history['t'])
        C = np.array(solver.history['coherence'])
        E = np.array(solver.history['energy'])
        Z = np.array(solver.history['enstrophy'])
        nu_eff = np.array(solver.history['nu_effective'])
        
        # Compute statistics
        stats = {
            'C_min': float(np.min(C)),
            'C_max': float(np.max(C)),
            'C_mean': float(np.mean(C)),
            'C_final': float(C[-1]),
            'C_initial': float(C[0]),
            'E_final': float(E[-1]),
            'Z_max': float(np.max(Z)),
            'nu_eff_max': float(np.max(nu_eff)),
            'nu_eff_mean': float(np.mean(nu_eff)),
            'success': success,
            'error': error_msg,
            't_final': float(t[-1]) if len(t) > 0 else 0.0,
        }
        
        # Check for pathologies
        C_below_crit = np.sum(C < params.C_crit) / len(C) if len(C) > 0 else 0
        C_std = np.std(C[len(C)//2:]) if len(C) > len(C)//2 else 0
        
        stats['frac_below_crit'] = float(C_below_crit)
        stats['C_std'] = float(C_std)
        stats['is_frozen'] = C_std < 0.01  # Low variance = frozen
        stats['is_turbulent'] = stats['Z_max'] > 1e-2  # High enstrophy = turbulent
        stats['is_evolving'] = np.std(E[len(E)//2:]) / np.mean(E[len(E)//2:]) > 0.1 if len(E) > 0 else False
        
        print(f"\nResults:")
        print(f"  C_φ: {stats['C_initial']:.4f} → {stats['C_final']:.4f} (min: {stats['C_min']:.4f}, std: {stats['C_std']:.4f})")
        print(f"  Max enstrophy: {stats['Z_max']:.4e}")
        print(f"  Max ν_eff: {stats['nu_eff_max']:.4f} (mean: {stats['nu_eff_mean']:.4f})")
        print(f"  Time below C_crit: {stats['frac_below_crit']*100:.1f}%")
        
        # Classification
        if not success:
            classification = "FAILED (blow-up or crash)"
        elif stats['is_frozen'] and not stats['is_evolving']:
            classification = "FROZEN (over-damped)"
        elif stats['is_turbulent'] and stats['C_min'] >= params.C_crit * 0.9:
            classification = "SUCCESS (turbulent + stable)"
        elif stats['is_turbulent'] and stats['C_min'] >= params.C_crit * 0.7:
            classification = "MARGINAL (turbulent but low coherence)"
        elif stats['is_turbulent']:
            classification = "RISKY (turbulent, coherence near limit)"
        elif stats['is_evolving']:
            classification = "LAMINAR (evolving, no turbulence)"
        else:
            classification = "STATIC (no evolution)"
        
        print(f"  Classification: {classification}")
        
        # Save plots
        test_dir = self.output_dir / test_name
        test_dir.mkdir(exist_ok=True)
        
        # Diagnostic plots
        fig_path = test_dir / 'diagnostics.png'
        solver.plot_diagnostics(str(fig_path))
        print(f"\n📊 Saved: {fig_path}")
        
        # Summary plot
        self._plot_summary(solver, test_name, classification, test_dir)
        
        # Save data
        result = {
            'test_name': test_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'C_crit': params.C_crit,
                'nu_boost_factor': params.nu_boost_factor,
                'nu': params.nu,
                'hbar': params.hbar,
                'tau_m': params.tau_m,
                'kernel_type': params.kernel_type,
                'enable_agency': params.enable_agency,
                'coherence_stabilization': params.coherence_stabilization,
                **ic_params,
                **sim_params
            },
            'statistics': stats,
            'classification': classification,
        }
        
        # Save JSON
        json_path = test_dir / 'results.json'
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        self.results.append(result)
        
        return result
    
    def _plot_summary(self, solver, test_name, classification, output_dir):
        """Create 4-panel summary plot"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        t = np.array(solver.history['t'])
        C = np.array(solver.history['coherence'])
        E = np.array(solver.history['energy'])
        Z = np.array(solver.history['enstrophy'])
        nu_eff = np.array(solver.history['nu_effective'])
        
        # Coherence
        ax = axes[0, 0]
        ax.plot(t, C, 'b-', linewidth=2)
        ax.axhline(solver.p.C_crit, color='r', linestyle='--', alpha=0.5, 
                   label=f'$C_φ^c$ = {solver.p.C_crit}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Coherence $C_φ$')
        ax.set_title('Phase Coherence Evolution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Energy
        ax = axes[0, 1]
        ax.plot(t, E, 'g-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy $E$')
        ax.set_yscale('log')
        ax.set_title('Kinetic Energy')
        ax.grid(alpha=0.3)
        
        # Enstrophy
        ax = axes[1, 0]
        ax.plot(t, Z, 'm-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Enstrophy $Z$')
        ax.set_yscale('log')
        ax.set_title('Vorticity Variance')
        ax.grid(alpha=0.3)
        
        # Adaptive viscosity
        ax = axes[1, 1]
        ax.plot(t, nu_eff, 'purple', linewidth=2)
        ax.axhline(solver.p.nu, color='k', linestyle='--', alpha=0.5, 
                   label=f'Base ν = {solver.p.nu}')
        ax.set_xlabel('Time')
        ax.set_ylabel('$ν_{eff}$')
        ax.set_title('Effective Viscosity')
        ax.legend()
        ax.grid(alpha=0.3)
        
        fig.suptitle(f'{test_name}\nClassification: {classification}', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        summary_path = output_dir / 'summary.png'
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved: {summary_path}")
    
    def run_test_matrix(self):
        """Run systematic parameter sweep"""
        
        print("\n" + "="*70)
        print("KELVIN-HELMHOLTZ TEST SUITE")
        print("="*70)
        
        # Base parameters
        base_params = {
            'N': 256,
            'L': 2*np.pi,
            'nu': 0.001,
            'hbar': 0.1,
            'tau_m': 10.0,
            'kernel_type': 'mixed',
            'alpha': 0.5,
            'beta_mix': 0.3,
            'enable_memory': True,
        }
        
        # Test configurations
        tests = [
            # Test 1: Current (over-damped)
            {
                'name': '01_baseline_overdamped',
                'params': {
                    **base_params,
                    'C_crit': 0.6,
                    'nu_boost_factor': 12.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': False,
                },
                'ic': {'delta': 0.05, 'epsilon': 0.02},
                'sim': {'T': 15.0, 'dt': 0.005}
            },
            
            # Test 2: Lower threshold
            {
                'name': '02_lower_threshold',
                'params': {
                    **base_params,
                    'C_crit': 0.35,
                    'nu_boost_factor': 5.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': False,
                },
                'ic': {'delta': 0.05, 'epsilon': 0.02},
                'sim': {'T': 15.0, 'dt': 0.005}
            },
            
            # Test 3: Smoother initial conditions
            {
                'name': '03_smoother_ic',
                'params': {
                    **base_params,
                    'C_crit': 0.5,
                    'nu_boost_factor': 5.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': False,
                },
                'ic': {'delta': 0.08, 'epsilon': 0.015},
                'sim': {'T': 15.0, 'dt': 0.005}
            },
            
            # Test 4: Gentle stabilization
            {
                'name': '04_gentle_stabilization',
                'params': {
                    **base_params,
                    'C_crit': 0.4,
                    'nu_boost_factor': 3.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': False,
                },
                'ic': {'delta': 0.05, 'epsilon': 0.02},
                'sim': {'T': 15.0, 'dt': 0.005}
            },
            
            # Test 5: Agency + adaptive viscosity
            {
                'name': '05_agency_enabled',
                'params': {
                    **base_params,
                    'C_crit': 0.35,
                    'nu_boost_factor': 5.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': True,
                    'A_pause': 1e-4,
                    'A_refuse': 5e-4,
                    'A_reframe': 2e-4,
                    'agency_prediction_factor': 0.2,
                },
                'ic': {'delta': 0.05, 'epsilon': 0.02},
                'sim': {'T': 15.0, 'dt': 0.005}
            },
            
            # Test 6: Very permissive (risk blow-up)
            {
                'name': '06_permissive',
                'params': {
                    **base_params,
                    'C_crit': 0.25,
                    'nu_boost_factor': 2.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': False,
                },
                'ic': {'delta': 0.05, 'epsilon': 0.02},
                'sim': {'T': 15.0, 'dt': 0.005}
            },
            
            # Test 7: Strong perturbation
            {
                'name': '07_strong_perturbation',
                'params': {
                    **base_params,
                    'C_crit': 0.4,
                    'nu_boost_factor': 5.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': False,
                },
                'ic': {'delta': 0.03, 'epsilon': 0.04},
                'sim': {'T': 15.0, 'dt': 0.005}
            },
            
            # Test 8: Best guess (Goldilocks)
            {
                'name': '08_goldilocks',
                'params': {
                    **base_params,
                    'C_crit': 0.35,
                    'nu_boost_factor': 5.0,
                    'coherence_stabilization': 'adaptive_viscosity',
                    'enable_agency': True,
                    'A_pause': 1e-4,
                    'A_refuse': 5e-4,
                    'A_reframe': 2e-4,
                    'agency_prediction_factor': 0.2,
                },
                'ic': {'delta': 0.08, 'epsilon': 0.015},
                'sim': {'T': 20.0, 'dt': 0.005}
            },
        ]
        
        # Run all tests
        for test in tests:
            try:
                result = self.run_single_test(
                    test['name'],
                    test['params'],
                    test['ic'],
                    test['sim']
                )
            except Exception as e:
                print(f"\n❌ Test {test['name']} crashed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Generate comparison report
        self.generate_comparison_report()
    
    def generate_comparison_report(self):
        """Create summary comparison of all tests"""
        
        if not self.results:
            print("\nNo results to compare")
            return
        
        print("\n" + "="*70)
        print("COMPARISON REPORT")
        print("="*70)
        
        # Create comparison table
        print(f"\n{'Test':<30} {'C_min':>8} {'C_std':>8} {'Z_max':>10} {'ν_max':>8} {'Class':<30}")
        print("-"*110)
        
        for r in self.results:
            s = r['statistics']
            print(f"{r['test_name']:<30} "
                  f"{s['C_min']:>8.4f} "
                  f"{s['C_std']:>8.5f} "
                  f"{s['Z_max']:>10.3e} "
                  f"{s['nu_eff_max']:>8.4f} "
                  f"{r['classification']:<30}")
        
        # Create comparison plots
        self._plot_comparison()
        
        # Save summary JSON
        summary_path = self.output_dir / 'suite_summary.json'
        with open(summary_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'n_tests': len(self.results),
                'results': self.results
            }, f, indent=2)
        
        print(f"\n📊 Summary saved: {summary_path}")
    
    def _plot_comparison(self):
        """Plot comparison across all tests"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Extract data
        names = [r['test_name'] for r in self.results]
        C_mins = [r['statistics']['C_min'] for r in self.results]
        C_stds = [r['statistics']['C_std'] for r in self.results]
        Z_maxs = [r['statistics']['Z_max'] for r in self.results]
        nu_maxs = [r['statistics']['nu_eff_max'] for r in self.results]
        classes = [r['classification'] for r in self.results]
        
        # Color by classification
        colors = []
        for c in classes:
            if 'SUCCESS' in c:
                colors.append('green')
            elif 'FROZEN' in c or 'STATIC' in c:
                colors.append('red')
            elif 'FAILED' in c:
                colors.append('darkred')
            elif 'MARGINAL' in c:
                colors.append('orange')
            elif 'RISKY' in c:
                colors.append('darkorange')
            else:
                colors.append('blue')
        
        # Coherence comparison
        ax = axes[0, 0]
        x = np.arange(len(names))
        ax.bar(x, C_mins, alpha=0.7, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in names], 
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Min Coherence $C_φ$')
        ax.set_title('Coherence Comparison')
        ax.grid(alpha=0.3, axis='y')
        
        # Coherence variance
        ax = axes[0, 1]
        ax.bar(x, C_stds, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in names], 
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Coherence Std Dev')
        ax.set_title('Coherence Variability (Low = Frozen)')
        ax.grid(alpha=0.3, axis='y')
        
        # Enstrophy comparison
        ax = axes[1, 0]
        ax.bar(x, Z_maxs, color=colors, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('_', '\n') for n in names], 
                           rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Max Enstrophy')
        ax.set_yscale('log')
        ax.set_title('Enstrophy Comparison (High = Turbulent)')
        ax.grid(alpha=0.3, axis='y')
        
        # Phase space
        ax = axes[1, 1]
        for i, (c_min, z_max, name, color, cls) in enumerate(
            zip(C_mins, Z_maxs, names, colors, classes)):
            if 'SUCCESS' in cls:
                marker = 'o'
                size = 300
            elif 'FROZEN' in cls or 'STATIC' in cls:
                marker = 'x'
                size = 200
            else:
                marker = 's'
                size = 200
            
            ax.scatter(c_min, z_max, c=color, s=size, marker=marker, 
                      alpha=0.7, edgecolors='black', linewidth=2)
            ax.annotate(f"{i+1}", (c_min, z_max), 
                       fontsize=9, ha='center', va='center', 
                       fontweight='bold', color='white')
        
        ax.set_xlabel('Min Coherence $C_φ$')
        ax.set_ylabel('Max Enstrophy')
        ax.set_yscale('log')
        ax.set_title('Phase Space\n(○=Success, ×=Frozen, □=Other)')
        ax.grid(alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', alpha=0.7, label='Success'),
            Patch(facecolor='orange', alpha=0.7, label='Marginal'),
            Patch(facecolor='darkorange', alpha=0.7, label='Risky'),
            Patch(facecolor='red', alpha=0.7, label='Frozen'),
            Patch(facecolor='darkred', alpha=0.7, label='Failed'),
            Patch(facecolor='blue', alpha=0.7, label='Laminar'),
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=9)
        
        plt.tight_layout()
        
        comparison_path = self.output_dir / 'comparison.png'
        plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Comparison plot: {comparison_path}")
        plt.close()


def main():
    """Run complete test suite"""
    suite = KHTestSuite(output_dir='../results/kh_suite')
    suite.run_test_matrix()
    
    print("\n" + "="*70)
    print("✅ TEST SUITE COMPLETE")
    print("="*70)
    print(f"\nResults saved in: {suite.output_dir}")
    print("\nNext steps:")
    print("1. Review comparison.png for overview")
    print("2. Look at individual test summaries")
    print("3. Identify 'SUCCESS' configurations")
    print("4. Run longer simulations with winning parameters")


if __name__ == "__main__":
    main()