import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

from aletheia.core.multifield_engine import MultiFieldEngine, AdaptiveGates, GateParams
from aletheia.core.field import Field, FieldConfig
from aletheia.core.memory import exp_kernel


@dataclass
class TestResults:
    """Container for test run results"""
    name: str
    coherence_history: np.ndarray
    tension_history: np.ndarray
    gate_history: List[np.ndarray]
    field_coherence_history: np.ndarray  # per-field coherence over time
    metadata: Dict
    
    def save(self, path: str):
        """Save results to disk"""
        np.savez(
            path,
            name=self.name,
            coherence_history=self.coherence_history,
            tension_history=self.tension_history,
            gate_history=np.array(self.gate_history),
            field_coherence_history=self.field_coherence_history,
            metadata=json.dumps(self.metadata)
        )
    
    @classmethod
    def load(cls, path: str):
        """Load results from disk"""
        data = np.load(path, allow_pickle=True)
        return cls(
            name=str(data['name']),
            coherence_history=data['coherence_history'],
            tension_history=data['tension_history'],
            gate_history=list(data['gate_history']),
            field_coherence_history=data['field_coherence_history'],
            metadata=json.loads(str(data['metadata']))
        )


class MultiFieldTestSuite:
    """Comprehensive testing framework for MultiFieldEngine"""
    
    def __init__(self, output_dir: str = "multifield_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: Dict[str, TestResults] = {}
    
    # ============================================================
    # TEST 1: TOPOLOGY ANALYSIS
    # ============================================================
    
    def test_topology_evolution(
        self, 
        N: int = 5, 
        steps: int = 500,
        grid_size: int = 16,
        snapshot_interval: int = 50
    ) -> TestResults:
        """
        Track how coupling topology evolves over time.
        Shows formation of coherence clusters.
        """
        print(f"\n{'='*60}")
        print("TEST 1: TOPOLOGY EVOLUTION")
        print(f"{'='*60}")
        
        # Setup fields
        cfgs = [FieldConfig(shape=(grid_size, grid_size), dt=0.05, seed=100+i) for i in range(N)]
        fields = [Field(cfg) for cfg in cfgs]
        
        # Random initial conditions with varying "difficulty"
        for i, f in enumerate(fields):
            # Some fields start more coherent than others
            noise_level = 0.5 + 0.5 * (i / N)
            f.psi = (np.random.randn(*f.cfg.shape) + 1j * np.random.randn(*f.cfg.shape)) * noise_level
        
        # Full connectivity - let gates decide who couples
        shared_idx = np.arange(min(50, grid_size**2))
        boundaries = {
            (i,j): (shared_idx, shared_idx) 
            for i in range(N) for j in range(i+1, N)
        }
        lambdas = np.ones((N,N)) - np.eye(N)
        gates = AdaptiveGates(GateParams(alpha=5, beta=5, floor=0.01))
        kernel = exp_kernel(length=grid_size**2, tau=20.0)
        
        engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.3)
        
        # Recording
        coherence_hist = []
        tension_hist = []
        gate_hist = []
        field_coh_hist = []
        
        snapshots = []
        snapshot_times = []
        
        for t in range(steps):
            diag = engine.step(t, kernel=kernel)
            coherence_hist.append(engine.mean_coherence())
            tension_hist.append(engine.mean_tension())
            gate_hist.append(diag.G.copy())
            field_coh_hist.append(diag.C.copy())
            
            if t % snapshot_interval == 0:
                snapshots.append(diag.G.copy())
                snapshot_times.append(t)
                print(f"t={t:3d} | ⟨C⟩={engine.mean_coherence():.3f} | "
                      f"⟨A⟩={engine.mean_tension():.3f} | "
                      f"Gate range: [{diag.G[diag.G>0].min():.2f}, {diag.G.max():.2f}]")
        
        results = TestResults(
            name="topology_evolution",
            coherence_history=np.array(coherence_hist),
            tension_history=np.array(tension_hist),
            gate_history=gate_hist,
            field_coherence_history=np.array(field_coh_hist),
            metadata={
                "N": N,
                "steps": steps,
                "grid_size": grid_size,
                "snapshots": snapshot_times
            }
        )
        
        self._plot_topology_evolution(results, snapshots, snapshot_times)
        self.results["topology"] = results
        return results
    
    def _plot_topology_evolution(self, results: TestResults, snapshots: List, times: List):
        """Visualize topology evolution"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, len(snapshots), figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: Gate matrices at different times
        for i, (G, t) in enumerate(zip(snapshots, times)):
            ax = fig.add_subplot(gs[0, i])
            im = ax.imshow(G, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f't={t}')
            ax.set_xlabel('Field j')
            ax.set_ylabel('Field i')
            if i == len(snapshots) - 1:
                plt.colorbar(im, ax=ax, label='Gate Gᵢⱼ')
        
        # Row 2: Coherence evolution
        ax = fig.add_subplot(gs[1, :])
        ax.plot(results.coherence_history, 'b-', lw=2, label='Mean coherence')
        ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Target')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence')
        ax.grid(alpha=0.3)
        ax.legend()
        ax.set_title('Global Coherence Evolution')
        
        # Row 3: Per-field coherence trajectories
        ax = fig.add_subplot(gs[2, :])
        N = results.field_coherence_history.shape[1]
        for i in range(N):
            ax.plot(results.field_coherence_history[:, i], label=f'Field {i}', alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Coherence')
        ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5))
        ax.grid(alpha=0.3)
        ax.set_title('Individual Field Coherence')
        
        plt.savefig(self.output_dir / "test1_topology_evolution.png", dpi=150, bbox_inches='tight')
        print(f"→ Saved: test1_topology_evolution.png")
    
    # ============================================================
    # TEST 2: PERTURBATION RESPONSE
    # ============================================================
    
    def test_perturbation_response(
        self,
        N: int = 4,
        stabilization_steps: int = 300,
        perturbation_strength: float = 2.0,
        recovery_steps: int = 200,
        grid_size: int = 12
    ) -> TestResults:
        """
        Stabilize system, perturb one field, measure recovery.
        Tests fault isolation and self-healing.
        """
        print(f"\n{'='*60}")
        print("TEST 2: PERTURBATION RESPONSE")
        print(f"{'='*60}")
        
        # Setup
        cfgs = [FieldConfig(shape=(grid_size, grid_size), dt=0.05, seed=200+i) for i in range(N)]
        fields = [Field(cfg) for cfg in cfgs]
        
        for f in fields:
            f.psi = np.random.randn(*f.cfg.shape) + 1j * np.random.randn(*f.cfg.shape)
        
        shared_idx = np.arange(min(30, grid_size**2))
        boundaries = {(i,j): (shared_idx, shared_idx) for i in range(N) for j in range(i+1,N)}
        lambdas = np.ones((N,N)) - np.eye(N)
        gates = AdaptiveGates(GateParams(alpha=6, beta=6, floor=0.01))
        kernel = exp_kernel(length=grid_size**2, tau=15.0)
        
        engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.35)
        
        # Phase 1: Stabilization
        print("\nPhase 1: Stabilization")
        coherence_hist = []
        tension_hist = []
        gate_hist = []
        field_coh_hist = []
        
        for t in range(stabilization_steps):
            diag = engine.step(t, kernel=kernel)
            coherence_hist.append(engine.mean_coherence())
            tension_hist.append(engine.mean_tension())
            gate_hist.append(diag.G.copy())
            field_coh_hist.append(diag.C.copy())
            
            if t % 100 == 0:
                print(f"  t={t} | ⟨C⟩={engine.mean_coherence():.3f}")
        
        baseline_coherence = engine.mean_coherence()
        print(f"  Baseline coherence: {baseline_coherence:.3f}")
        
        # Phase 2: Perturbation
        print(f"\nPhase 2: Perturbing field 0 (strength={perturbation_strength})")
        perturb_idx = 0
        noise = perturbation_strength * (np.random.randn(*fields[perturb_idx].psi.shape) + 
                                         1j * np.random.randn(*fields[perturb_idx].psi.shape))
        fields[perturb_idx].psi += noise
        
        perturbation_time = stabilization_steps
        
        # Phase 3: Recovery
        print("\nPhase 3: Recovery")
        for t in range(stabilization_steps, stabilization_steps + recovery_steps):
            diag = engine.step(t, kernel=kernel)
            coherence_hist.append(engine.mean_coherence())
            tension_hist.append(engine.mean_tension())
            gate_hist.append(diag.G.copy())
            field_coh_hist.append(diag.C.copy())
            
            if (t - stabilization_steps) % 50 == 0:
                print(f"  t={t-stabilization_steps} | ⟨C⟩={engine.mean_coherence():.3f} | "
                      f"C₀={diag.C[perturb_idx]:.3f}")
        
        results = TestResults(
            name="perturbation_response",
            coherence_history=np.array(coherence_hist),
            tension_history=np.array(tension_hist),
            gate_history=gate_hist,
            field_coherence_history=np.array(field_coh_hist),
            metadata={
                "N": N,
                "stabilization_steps": stabilization_steps,
                "perturbation_time": perturbation_time,
                "perturbation_field": perturb_idx,
                "perturbation_strength": perturbation_strength,
                "baseline_coherence": baseline_coherence
            }
        )
        
        self._plot_perturbation_response(results)
        self.results["perturbation"] = results
        return results
    
    def _plot_perturbation_response(self, results: TestResults):
        """Visualize perturbation and recovery"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        perturb_t = results.metadata["perturbation_time"]
        perturb_field = results.metadata["perturbation_field"]
        
        # Global coherence
        ax = axes[0, 0]
        ax.plot(results.coherence_history, 'b-', lw=2)
        ax.axvline(perturb_t, color='r', linestyle='--', label='Perturbation')
        ax.axhline(results.metadata["baseline_coherence"], color='g', 
                   linestyle=':', alpha=0.5, label='Baseline')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Global Coherence Response')
        
        # Per-field coherence
        ax = axes[0, 1]
        N = results.field_coherence_history.shape[1]
        for i in range(N):
            style = '-' if i == perturb_field else '--'
            lw = 2.5 if i == perturb_field else 1.5
            ax.plot(results.field_coherence_history[:, i], style, lw=lw, 
                   label=f'Field {i}' + (' (perturbed)' if i == perturb_field else ''))
        ax.axvline(perturb_t, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Coherence')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Per-Field Coherence')
        
        # Gate isolation: connections TO perturbed field
        ax = axes[1, 0]
        gate_array = np.array(results.gate_history)
        for i in range(N):
            if i != perturb_field:
                # Gate from field i to perturbed field
                gate_to_perturbed = gate_array[:, i, perturb_field]
                ax.plot(gate_to_perturbed, label=f'G({i}→{perturb_field})', alpha=0.7)
        ax.axvline(perturb_t, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Gate Strength')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Gate Isolation Response')
        
        # Recovery time analysis
        ax = axes[1, 1]
        recovery_phase = results.coherence_history[perturb_t:]
        recovery_delta = recovery_phase - results.metadata["baseline_coherence"]
        ax.plot(recovery_delta, 'purple', lw=2)
        ax.axhline(0, color='g', linestyle='--', alpha=0.5)
        ax.set_xlabel('Steps after perturbation')
        ax.set_ylabel('Coherence deficit')
        ax.grid(alpha=0.3)
        ax.set_title('Recovery Dynamics')
        
        # Find recovery time (95% of baseline)
        threshold = 0.95 * results.metadata["baseline_coherence"]
        recovery_idx = np.where(recovery_phase >= threshold)[0]
        if len(recovery_idx) > 0:
            recovery_time = recovery_idx[0]
            ax.axvline(recovery_time, color='orange', linestyle=':', 
                      label=f'95% recovery: t={recovery_time}')
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test2_perturbation_response.png", dpi=150)
        print(f"→ Saved: test2_perturbation_response.png")
    
    # ============================================================
    # TEST 3: SCALING BEHAVIOR
    # ============================================================
    
    def test_scaling(
        self,
        N_values: List[int] = [2, 3, 5, 8, 12],
        steps: int = 300,
        grid_size: int = 10
    ) -> TestResults:
        """
        Test how system behavior scales with number of fields.
        Does coherence converge faster/slower? What's the overhead?
        """
        print(f"\n{'='*60}")
        print("TEST 3: SCALING BEHAVIOR")
        print(f"{'='*60}")
        
        results_by_N = {}
        
        for N in N_values:
            print(f"\n--- N = {N} fields ---")
            
            cfgs = [FieldConfig(shape=(grid_size, grid_size), dt=0.05, seed=300+i) for i in range(N)]
            fields = [Field(cfg) for cfg in cfgs]
            
            for f in fields:
                f.psi = np.random.randn(*f.cfg.shape) + 1j * np.random.randn(*f.cfg.shape)
            
            shared_idx = np.arange(min(20, grid_size**2))
            boundaries = {(i,j): (shared_idx, shared_idx) for i in range(N) for j in range(i+1,N)}
            lambdas = np.ones((N,N)) - np.eye(N)
            gates = AdaptiveGates(GateParams(alpha=5, beta=5, floor=0.02))
            kernel = exp_kernel(length=grid_size**2, tau=15.0)
            
            engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.3)
            
            coherence_hist = []
            
            for t in range(steps):
                diag = engine.step(t, kernel=kernel)
                coherence_hist.append(engine.mean_coherence())
            
            final_coherence = coherence_hist[-1]
            # Measure convergence time (time to reach 80% of final value)
            threshold = 0.8 * final_coherence
            conv_idx = np.where(np.array(coherence_hist) >= threshold)[0]
            convergence_time = conv_idx[0] if len(conv_idx) > 0 else steps
            
            results_by_N[N] = {
                "coherence": coherence_hist,
                "final": final_coherence,
                "convergence_time": convergence_time
            }
            
            print(f"  Final ⟨C⟩: {final_coherence:.3f}")
            print(f"  Convergence time: {convergence_time} steps")
        
        # Package results
        max_coh = np.array([results_by_N[N]["coherence"] for N in N_values])
        
        results = TestResults(
            name="scaling",
            coherence_history=max_coh,
            tension_history=np.array([]),  # Not tracking tension here
            gate_history=[],
            field_coherence_history=np.array([]),
            metadata={
                "N_values": N_values,
                "steps": steps,
                "results_by_N": {N: {"final": results_by_N[N]["final"], 
                                     "convergence_time": int(results_by_N[N]["convergence_time"])}
                                for N in N_values}
            }
        )
        
        self._plot_scaling(results, results_by_N)
        self.results["scaling"] = results
        return results
    
    def _plot_scaling(self, results: TestResults, results_by_N: Dict):
        """Visualize scaling behavior"""
        N_values = results.metadata["N_values"]
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Evolution for each N
        ax = axes[0]
        for i, N in enumerate(N_values):
            ax.plot(results_by_N[N]["coherence"], label=f'N={N}', lw=2, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Coherence Evolution vs. System Size')
        
        # Final coherence vs N
        ax = axes[1]
        final_values = [results_by_N[N]["final"] for N in N_values]
        ax.plot(N_values, final_values, 'o-', markersize=8, lw=2, color='blue')
        ax.set_xlabel('Number of Fields (N)')
        ax.set_ylabel('Final Coherence')
        ax.grid(alpha=0.3)
        ax.set_title('Final Coherence vs. N')
        
        # Convergence time vs N
        ax = axes[2]
        conv_times = [results_by_N[N]["convergence_time"] for N in N_values]
        ax.plot(N_values, conv_times, 's-', markersize=8, lw=2, color='red')
        ax.set_xlabel('Number of Fields (N)')
        ax.set_ylabel('Convergence Time (steps)')
        ax.grid(alpha=0.3)
        ax.set_title('Convergence Time vs. N')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test3_scaling.png", dpi=150)
        print(f"→ Saved: test3_scaling.png")
    
    # ============================================================
    # TEST 4: GATE PARAMETER SENSITIVITY
    # ============================================================
    
    def test_gate_sensitivity(
        self,
        alpha_values: List[float] = [1.0, 3.0, 5.0, 8.0],
        beta_values: List[float] = [1.0, 3.0, 5.0, 8.0],
        N: int = 4,
        steps: int = 300,
        grid_size: int = 12
    ) -> TestResults:
        """
        Sweep gate parameters (alpha, beta) to understand sensitivity.
        Alpha controls coherence sharpness, beta controls affect mismatch penalty.
        """
        print(f"\n{'='*60}")
        print("TEST 4: GATE PARAMETER SENSITIVITY")
        print(f"{'='*60}")
        
        results_grid = {}
        
        for alpha in alpha_values:
            for beta in beta_values:
                key = (alpha, beta)
                print(f"\n--- α={alpha}, β={beta} ---")
                
                cfgs = [FieldConfig(shape=(grid_size, grid_size), dt=0.05, seed=400+hash(key)%1000) 
                       for i in range(N)]
                fields = [Field(cfg) for cfg in cfgs]
                
                for f in fields:
                    f.psi = np.random.randn(*f.cfg.shape) + 1j * np.random.randn(*f.cfg.shape)
                
                shared_idx = np.arange(min(30, grid_size**2))
                boundaries = {(i,j): (shared_idx, shared_idx) for i in range(N) for j in range(i+1,N)}
                lambdas = np.ones((N,N)) - np.eye(N)
                gates = AdaptiveGates(GateParams(alpha=alpha, beta=beta, floor=0.01))
                kernel = exp_kernel(length=grid_size**2, tau=15.0)
                
                engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.3)
                
                coherence_hist = []
                
                for t in range(steps):
                    diag = engine.step(t, kernel=kernel)
                    coherence_hist.append(engine.mean_coherence())
                
                results_grid[key] = {
                    "coherence": coherence_hist,
                    "final": coherence_hist[-1]
                }
                
                print(f"  Final ⟨C⟩: {coherence_hist[-1]:.3f}")
        
        # Package
        results = TestResults(
            name="gate_sensitivity",
            coherence_history=np.array([results_grid[k]["coherence"] for k in results_grid]),
            tension_history=np.array([]),
            gate_history=[],
            field_coherence_history=np.array([]),
            metadata={
                "alpha_values": alpha_values,
                "beta_values": beta_values,
                "N": N,
                "steps": steps,
                "results_grid": {str(k): {"final": results_grid[k]["final"]} 
                                for k in results_grid}
            }
        )
        
        self._plot_gate_sensitivity(results, results_grid, alpha_values, beta_values)
        self.results["gate_sensitivity"] = results
        return results
    
    def _plot_gate_sensitivity(self, results: TestResults, results_grid: Dict, 
                               alpha_values: List, beta_values: List):
        """Visualize gate parameter effects"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap of final coherence
        ax = axes[0]
        final_matrix = np.zeros((len(beta_values), len(alpha_values)))
        for i, beta in enumerate(beta_values):
            for j, alpha in enumerate(alpha_values):
                final_matrix[i, j] = results_grid[(alpha, beta)]["final"]
        
        im = ax.imshow(final_matrix, cmap='viridis', aspect='auto', origin='lower')
        ax.set_xticks(range(len(alpha_values)))
        ax.set_yticks(range(len(beta_values)))
        ax.set_xticklabels([f'{a:.1f}' for a in alpha_values])
        ax.set_yticklabels([f'{b:.1f}' for b in beta_values])
        ax.set_xlabel('α (coherence sharpness)')
        ax.set_ylabel('β (affect penalty)')
        ax.set_title('Final Coherence Map')
        plt.colorbar(im, ax=ax, label='Final ⟨C⟩')
        
        # Evolution curves for different alpha (fixed beta)
        ax = axes[1]
        fixed_beta = beta_values[len(beta_values)//2]  # Middle value
        for alpha in alpha_values:
            coh = results_grid[(alpha, fixed_beta)]["coherence"]
            ax.plot(coh, label=f'α={alpha:.1f}', lw=2, alpha=0.8)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f'Evolution vs. α (β={fixed_beta:.1f})')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test4_gate_sensitivity.png", dpi=150)
        print(f"→ Saved: test4_gate_sensitivity.png")
    
    # ============================================================
    # TEST 5: INITIAL CONDITION SENSITIVITY
    # ============================================================
    
    def test_initial_conditions(
        self,
        n_trials: int = 10,
        N: int = 4,
        steps: int = 300,
        grid_size: int = 12
    ) -> TestResults:
        """
        Run multiple trials with different random seeds.
        Tests robustness: does system always converge? How variable are outcomes?
        """
        print(f"\n{'='*60}")
        print("TEST 5: INITIAL CONDITION SENSITIVITY")
        print(f"{'='*60}")
        
        all_coherence = []
        final_coherences = []
        
        for trial in range(n_trials):
            print(f"\nTrial {trial+1}/{n_trials}")
            
            cfgs = [FieldConfig(shape=(grid_size, grid_size), dt=0.05, seed=500+trial*100+i) 
                   for i in range(N)]
            fields = [Field(cfg) for cfg in cfgs]
            
            for f in fields:
                f.psi = np.random.randn(*f.cfg.shape) + 1j * np.random.randn(*f.cfg.shape)
            
            shared_idx = np.arange(min(30, grid_size**2))
            boundaries = {(i,j): (shared_idx, shared_idx) for i in range(N) for j in range(i+1,N)}
            lambdas = np.ones((N,N)) - np.eye(N)
            gates = AdaptiveGates(GateParams(alpha=5, beta=5, floor=0.01))
            kernel = exp_kernel(length=grid_size**2, tau=15.0)
            
            engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.3)
            
            coherence_hist = []
            
            for t in range(steps):
                diag = engine.step(t, kernel=kernel)
                coherence_hist.append(engine.mean_coherence())
            
            all_coherence.append(coherence_hist)
            final_coherences.append(coherence_hist[-1])
            print(f"  Final ⟨C⟩: {coherence_hist[-1]:.3f}")
        
        all_coherence = np.array(all_coherence)
        
        results = TestResults(
            name="initial_conditions",
            coherence_history=all_coherence,
            tension_history=np.array([]),
            gate_history=[],
            field_coherence_history=np.array([]),
            metadata={
                "n_trials": n_trials,
                "N": N,
                "steps": steps,
                "final_coherences": final_coherences,
                "mean_final": float(np.mean(final_coherences)),
                "std_final": float(np.std(final_coherences))
            }
        )
        
        self._plot_initial_conditions(results)
        self.results["initial_conditions"] = results
        return results
    
    def _plot_initial_conditions(self, results: TestResults):
        """Visualize robustness across initial conditions"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # All trajectories
        ax = axes[0]
        n_trials = results.metadata["n_trials"]
        for i in range(n_trials):
            ax.plot(results.coherence_history[i], alpha=0.5, lw=1.5, color='blue')
        
        # Mean trajectory
        mean_coh = np.mean(results.coherence_history, axis=0)
        std_coh = np.std(results.coherence_history
                         , axis=0)
        ax.plot(mean_coh, 'r-', lw=3, label='Mean')
        ax.fill_between(range(len(mean_coh)), mean_coh - std_coh, mean_coh + std_coh,
                        color='red', alpha=0.2, label='±1σ')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title(f'Convergence Across {n_trials} Random Initializations')
        
        # Distribution of final coherences
        ax = axes[1]
        final_vals = results.metadata["final_coherences"]
        ax.hist(final_vals, bins=15, color='purple', alpha=0.7, edgecolor='black')
        ax.axvline(results.metadata["mean_final"], color='red', linestyle='--', 
                   lw=2, label=f'Mean: {results.metadata["mean_final"]:.3f}')
        ax.axvline(results.metadata["mean_final"] - results.metadata["std_final"], 
                   color='orange', linestyle=':', alpha=0.7)
        ax.axvline(results.metadata["mean_final"] + results.metadata["std_final"], 
                   color='orange', linestyle=':', alpha=0.7, label=f'±1σ: {results.metadata["std_final"]:.3f}')
        ax.set_xlabel('Final Coherence')
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Distribution of Final Coherences')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "test5_initial_conditions.png", dpi=150)
        print(f"→ Saved: test5_initial_conditions.png")
    
    # ============================================================
    # TEST 6: ASYMMETRIC COUPLING
    # ============================================================
    
    def test_asymmetric_coupling(
        self,
        N: int = 5,
        steps: int = 400,
        grid_size: int = 12
    ) -> TestResults:
        """
        Create heterogeneous coupling structure (not fully connected).
        Tests: Can system form coherence clusters? Does topology matter?
        """
        print(f"\n{'='*60}")
        print("TEST 6: ASYMMETRIC COUPLING TOPOLOGY")
        print(f"{'='*60}")
        
        cfgs = [FieldConfig(shape=(grid_size, grid_size), dt=0.05, seed=600+i) for i in range(N)]
        fields = [Field(cfg) for cfg in cfgs]
        
        for f in fields:
            f.psi = np.random.randn(*f.cfg.shape) + 1j * np.random.randn(*f.cfg.shape)
        
        # Create ring topology: 0-1-2-3-4-0
        shared_idx = np.arange(min(30, grid_size**2))
        boundaries = {}
        lambdas = np.zeros((N, N))
        
        # Ring connections
        for i in range(N):
            j = (i + 1) % N
            boundaries[(i, j)] = (shared_idx, shared_idx)
            lambdas[i, j] = 1.0
            lambdas[j, i] = 1.0
        
        # Add one long-range connection (creates shortcut)
        boundaries[(0, 3)] = (shared_idx, shared_idx)
        lambdas[0, 3] = 0.5
        lambdas[3, 0] = 0.5
        
        print(f"Topology: Ring with shortcut (0-3)")
        print(f"Connections: {list(boundaries.keys())}")
        
        gates = AdaptiveGates(GateParams(alpha=5, beta=5, floor=0.01))
        kernel = exp_kernel(length=grid_size**2, tau=15.0)
        
        engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.3)
        
        coherence_hist = []
        tension_hist = []
        gate_hist = []
        field_coh_hist = []
        
        for t in range(steps):
            diag = engine.step(t, kernel=kernel)
            coherence_hist.append(engine.mean_coherence())
            tension_hist.append(engine.mean_tension())
            gate_hist.append(diag.G.copy())
            field_coh_hist.append(diag.C.copy())
            
            if t % 100 == 0:
                print(f"t={t} | ⟨C⟩={engine.mean_coherence():.3f}")
        
        results = TestResults(
            name="asymmetric_coupling",
            coherence_history=np.array(coherence_hist),
            tension_history=np.array(tension_hist),
            gate_history=gate_hist,
            field_coherence_history=np.array(field_coh_hist),
            metadata={
                "N": N,
                "steps": steps,
                "topology": "ring_with_shortcut",
                "lambda_matrix": lambdas.tolist()
            }
        )
        
        self._plot_asymmetric_coupling(results)
        self.results["asymmetric"] = results
        return results
    
    def _plot_asymmetric_coupling(self, results: TestResults):
        """Visualize behavior with asymmetric topology"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Lambda matrix (structural connectivity)
        ax = fig.add_subplot(gs[0, 0])
        lam_matrix = np.array(results.metadata["lambda_matrix"])
        im = ax.imshow(lam_matrix, cmap='Blues', vmin=0, vmax=1)
        ax.set_title('Structural Coupling (λ)')
        ax.set_xlabel('Field j')
        ax.set_ylabel('Field i')
        plt.colorbar(im, ax=ax)
        
        # Final gate matrix (effective connectivity)
        ax = fig.add_subplot(gs[0, 1])
        final_gates = results.gate_history[-1]
        im = ax.imshow(final_gates, cmap='hot', vmin=0, vmax=1)
        ax.set_title('Final Gates (G)')
        ax.set_xlabel('Field j')
        ax.set_ylabel('Field i')
        plt.colorbar(im, ax=ax)
        
        # Difference (gates - lambdas shows adaptive modulation)
        ax = fig.add_subplot(gs[0, 2])
        # Only compare where lambda > 0
        diff = np.where(lam_matrix > 0, final_gates - lam_matrix, 0)
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_title('Adaptive Modulation (G - λ)')
        ax.set_xlabel('Field j')
        ax.set_ylabel('Field i')
        plt.colorbar(im, ax=ax)
        
        # Global coherence evolution
        ax = fig.add_subplot(gs[1, :])
        ax.plot(results.coherence_history, 'b-', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence')
        ax.grid(alpha=0.3)
        ax.set_title('Global Coherence Evolution (Ring Topology)')
        
        # Per-field coherence
        ax = fig.add_subplot(gs[2, :])
        N = results.field_coherence_history.shape[1]
        for i in range(N):
            ax.plot(results.field_coherence_history[:, i], label=f'Field {i}', lw=2, alpha=0.7)
        ax.set_xlabel('Step')
        ax.set_ylabel('Coherence')
        ax.legend(loc='right', bbox_to_anchor=(1.12, 0.5))
        ax.grid(alpha=0.3)
        ax.set_title('Individual Field Coherence')
        
        plt.savefig(self.output_dir / "test6_asymmetric_coupling.png", dpi=150, bbox_inches='tight')
        print(f"→ Saved: test6_asymmetric_coupling.png")
    
    # ============================================================
    # TEST 7: COHERENCE CLUSTERING
    # ============================================================
    
    def test_coherence_clustering(
        self,
        N: int = 6,
        steps: int = 500,
        grid_size: int = 12
    ) -> TestResults:
        """
        Initialize fields with different 'types' (different initial coherence levels).
        Test: Do similar fields cluster together? Does hierarchy emerge?
        """
        print(f"\n{'='*60}")
        print("TEST 7: COHERENCE CLUSTERING")
        print(f"{'='*60}")
        
        cfgs = [FieldConfig(shape=(grid_size, grid_size), dt=0.05, seed=700+i) for i in range(N)]
        fields = [Field(cfg) for cfg in cfgs]
        
        # Create three groups with different initial properties
        n_per_group = N // 3
        groups = []
        
        print("\nInitializing field groups:")
        for i, f in enumerate(fields):
            group = i // n_per_group
            groups.append(group)
            
            if group == 0:  # High coherence group
                phase = 0.0
                amplitude = 1.0
                noise = 0.2
                print(f"  Field {i}: Group A (high coherence)")
            elif group == 1:  # Medium coherence group
                phase = np.pi / 2
                amplitude = 0.8
                noise = 0.5
                print(f"  Field {i}: Group B (medium coherence)")
            else:  # Low coherence group
                phase = np.pi
                amplitude = 0.6
                noise = 0.8
                print(f"  Field {i}: Group C (low coherence)")
            
            base = amplitude * np.exp(1j * phase) * np.ones(f.cfg.shape)
            perturbation = noise * (np.random.randn(*f.cfg.shape) + 
                                   1j * np.random.randn(*f.cfg.shape))
            f.psi = base + perturbation
        
        # Full connectivity - let gates decide clustering
        shared_idx = np.arange(min(30, grid_size**2))
        boundaries = {(i,j): (shared_idx, shared_idx) for i in range(N) for j in range(i+1,N)}
        lambdas = np.ones((N,N)) - np.eye(N)
        gates = AdaptiveGates(GateParams(alpha=6, beta=6, floor=0.01))
        kernel = exp_kernel(length=grid_size**2, tau=15.0)
        
        engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.3)
        
        coherence_hist = []
        tension_hist = []
        gate_hist = []
        field_coh_hist = []
        
        for t in range(steps):
            diag = engine.step(t, kernel=kernel)
            coherence_hist.append(engine.mean_coherence())
            tension_hist.append(engine.mean_tension())
            gate_hist.append(diag.G.copy())
            field_coh_hist.append(diag.C.copy())
            
            if t % 100 == 0:
                print(f"t={t} | ⟨C⟩={engine.mean_coherence():.3f}")
        
        results = TestResults(
            name="coherence_clustering",
            coherence_history=np.array(coherence_hist),
            tension_history=np.array(tension_hist),
            gate_history=gate_hist,
            field_coherence_history=np.array(field_coh_hist),
            metadata={
                "N": N,
                "steps": steps,
                "groups": groups,
                "n_per_group": n_per_group
            }
        )
        
        self._plot_coherence_clustering(results)
        self.results["clustering"] = results
        return results
    
    def _plot_coherence_clustering(self, results: TestResults):
        """Visualize clustering behavior"""
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        groups = results.metadata["groups"]
        N = len(groups)
        
        # Gate matrix at t=0, t=mid, t=final
        times = [0, len(results.gate_history)//2, -1]
        time_labels = ['Initial', 'Mid', 'Final']
        
        for idx, (t, label) in enumerate(zip(times, time_labels)):
            ax = fig.add_subplot(gs[0, idx])
            G = results.gate_history[t]
            im = ax.imshow(G, cmap='hot', vmin=0, vmax=1)
            ax.set_title(f'Gates: {label}')
            ax.set_xlabel('Field j')
            ax.set_ylabel('Field i')
            
            # Add group boundaries
            for i in range(1, 3):
                line_pos = i * results.metadata["n_per_group"] - 0.5
                ax.axhline(line_pos, color='cyan', linestyle='--', lw=1, alpha=0.5)
                ax.axvline(line_pos, color='cyan', linestyle='--', lw=1, alpha=0.5)
            
            if idx == 2:
                plt.colorbar(im, ax=ax, label='Gate Gᵢⱼ')
        
        # Per-field coherence colored by group
        ax = fig.add_subplot(gs[1, :2])
        colors = ['red', 'blue', 'green']
        for i in range(N):
            group = groups[i]
            ax.plot(results.field_coherence_history[:, i], 
                   color=colors[group], lw=2, alpha=0.7,
                   label=f'Field {i} (Group {chr(65+group)})')
        ax.set_xlabel('Step')
        ax.set_ylabel('Coherence')
        ax.legend(loc='right', bbox_to_anchor=(1.15, 0.5), fontsize=8)
        ax.grid(alpha=0.3)
        ax.set_title('Per-Field Coherence (colored by initial group)')
        
        # Clustering metric: within-group vs between-group coupling
        ax = fig.add_subplot(gs[1, 2])
        
        within_group_coupling = []
        between_group_coupling = []
        
        for G in results.gate_history:
            within = []
            between = []
            for i in range(N):
                for j in range(i+1, N):
                    if groups[i] == groups[j]:
                        within.append(G[i,j])
                    else:
                        between.append(G[i,j])
            within_group_coupling.append(np.mean(within) if within else 0)
            between_group_coupling.append(np.mean(between) if between else 0)
        
        ax.plot(within_group_coupling, 'g-', lw=2, label='Within-group ⟨G⟩')
        ax.plot(between_group_coupling, 'r-', lw=2, label='Between-group ⟨G⟩')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Gate Strength')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Clustering: Within vs Between Groups')
        
        plt.savefig(self.output_dir / "test7_coherence_clustering.png", dpi=150, bbox_inches='tight')
        print(f"→ Saved: test7_coherence_clustering.png")
    
    # ============================================================
    # MASTER RUN FUNCTION
    # ============================================================
    
    def run_all(self):
        """Run complete test suite"""
        print("\n" + "="*60)
        print("MULTI-FIELD ENGINE: COMPREHENSIVE TEST SUITE")
        print("="*60)
        
        try:
            self.test_topology_evolution()
            self.test_perturbation_response()
            self.test_scaling()
            self.test_gate_sensitivity()
            self.test_initial_conditions()
            self.test_asymmetric_coupling()
            self.test_coherence_clustering()
            
            print("\n" + "="*60)
            print("ALL TESTS COMPLETE")
            print("="*60)
            self._generate_summary()
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
    
    def _generate_summary(self):
        """Generate summary report"""
        print("\n" + "-"*60)
        print("SUMMARY")
        print("-"*60)
        
        for name, result in self.results.items():
            print(f"\n{name.upper()}:")
            
            if name == "topology":
                final_coh = result.coherence_history[-1]
                print(f"  Final coherence: {final_coh:.3f}")
                print(f"  Convergence achieved: {'Yes' if final_coh > 0.3 else 'No'}")
            
            elif name == "perturbation":
                baseline = result.metadata["baseline_coherence"]
                final_coh = result.coherence_history[-1]
                recovery = (final_coh / baseline) * 100
                print(f"  Baseline: {baseline:.3f}")
                print(f"  Final: {final_coh:.3f}")
                print(f"  Recovery: {recovery:.1f}%")
            
            elif name == "scaling":
                results_by_n = result.metadata["results_by_N"]
                print(f"  Tested N = {list(results_by_n.keys())}")
                for n, data in results_by_n.items():
                    print(f"    N={n}: final={data['final']:.3f}, conv_time={data['convergence_time']}")
            
            elif name == "gate_sensitivity":
                print(f"  Parameter sweep: α={result.metadata['alpha_values']}, β={result.metadata['beta_values']}")
                finals = [v['final'] for v in result.metadata['results_grid'].values()]
                print(f"  Coherence range: [{min(finals):.3f}, {max(finals):.3f}]")
            
            elif name == "initial_conditions":
                mean = result.metadata["mean_final"]
                std = result.metadata["std_final"]
                cv = (std / mean) * 100 if mean > 0 else 0
                print(f"  Mean final coherence: {mean:.3f} ± {std:.3f}")
                print(f"  Coefficient of variation: {cv:.1f}%")
                print(f"  Robustness: {'High' if cv < 15 else 'Medium' if cv < 30 else 'Low'}")
            
            elif name == "asymmetric":
                final_coh = result.coherence_history[-1]
                print(f"  Topology: {result.metadata['topology']}")
                print(f"  Final coherence: {final_coh:.3f}")
            
            elif name == "clustering":
                print(f"  Number of groups: 3")
                final_coh = result.coherence_history[-1]
                print(f"  Final global coherence: {final_coh:.3f}")
        
        print(f"\n📁 All results saved to: {self.output_dir}/")
        print("-"*60)


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    suite = MultiFieldTestSuite(output_dir="multifield_comprehensive_tests")
    suite.run_all()