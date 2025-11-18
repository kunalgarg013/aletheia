"""
Multi-Field TSP with Your Agency-Gated Coupling Engine
========================================================

Uses your actual aletheia multi-field infrastructure:
- field.py: Complex field evolution with memory
- affect.py: Tension metric
- agency.py: Agency thresholds and decisions
- memory.py: Exponential kernels
- multifield_engine.py: Adaptive gate coupling

Each field explores TSP space independently.
Agency-gated coupling shares structure between high-confidence fields.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import os

# Your actual imports (adjust paths as needed)
from aletheia.core.field import Field, FieldConfig
from aletheia.core.affect import Affect, tension_A
from aletheia.core.agency import Agency, AgencyThresholds
from aletheia.core.memory import exp_kernel
from aletheia.core.multifield_engine import MultiFieldEngine, AdaptiveGates, GateParams


# =====================================================
# TSP-SPECIFIC METRICS
# =====================================================

def tsp_tension(psi: np.ndarray) -> float:
    """
    Tension metric for TSP edge fields with size normalization.
    
    High tension = field violating tour constraints
    Low tension = field has valid tour structure
    
    CRITICAL: Normalizes by sqrt(n) so tension values are comparable
    across different problem sizes. Without this, small problems saturate
    tension → gates can't differentiate → no coupling.
    
    Measures constraint violation + gradient energy
    """
    n = psi.shape[0]  # Number of cities
    
    # Constraint violation (rows/cols should sum to ~1)
    row_sums = np.abs(psi).sum(axis=1)
    col_sums = np.abs(psi).sum(axis=0)
    constraint_violation = (
        np.mean((row_sums - 1.0)**2) + 
        np.mean((col_sums - 1.0)**2)
    )
    
    # Normalize by problem size (critical for cross-scale comparison!)
    # Larger problems naturally have more constraint violation
    constraint_violation = constraint_violation / np.sqrt(n)
    
    # Gradient energy (spatial smoothness)
    try:
        from scipy.signal import convolve2d
        lap_ker = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]], dtype=float)
        lap_r = convolve2d(psi.real, lap_ker, mode="same", boundary="wrap")
        lap_i = convolve2d(psi.imag, lap_ker, mode="same", boundary="wrap")
        gradient_energy = float(np.mean(lap_r**2 + lap_i**2))
        
        # Also normalize gradient energy
        gradient_energy = gradient_energy / np.sqrt(n)
    except:
        # Fallback if scipy not available
        gradient_energy = 0.0
    
    # Combined (weighted toward constraint violation for TSP)
    # These weights emphasize tour validity over smoothness
    tension = 0.7 * constraint_violation + 0.3 * gradient_energy
    
    # Ensure never exactly zero (would break gate computation)
    return float(max(tension, 1e-6))


# =====================================================
# TSP-SPECIFIC FIELD WRAPPER
# =====================================================

class TSPField(Field):
    """
    Extend your Field class to include TSP-specific encoding.
    
    Field shape: (N_cities, N_cities) - edge probabilities
    psi[i,j] = complex amplitude of edge i→j
    """
    
    def __init__(self, n_cities: int, distance_matrix: np.ndarray, 
                 dt: float = 0.02, alpha: float = 0.15):
        """
        Args:
            n_cities: number of cities
            distance_matrix: (N, N) distances
            dt: time step
            alpha: cost bias strength
        """
        # Initialize base field
        cfg = FieldConfig(
            shape=(n_cities, n_cities),
            dt=dt,
            seed=np.random.randint(0, 10000),
            kernel_length=128,
            init_amp=1.0  # INCREASED from 0.1
        )
        super().__init__(cfg)
        
        self.n_cities = n_cities
        self.distance_matrix = distance_matrix
        self.alpha = alpha
        
        # CRITICAL FIX: Initialize with cost-biased structure
        # Start with preference for short edges (not pure random)
        cost_bias = np.exp(-0.5 * alpha * distance_matrix)
        self.psi = self.psi * cost_bias
        
        # Zero diagonal (no self-loops)
        np.fill_diagonal(self.psi, 0)
        
        # Soft normalize (tour structure)
        row_sum = np.abs(self.psi).sum(axis=1, keepdims=True)
        self.psi = self.psi / (row_sum + 1e-9)
        col_sum = np.abs(self.psi).sum(axis=0, keepdims=True)
        self.psi = self.psi / (col_sum + 1e-9)
        
        # Global normalize
        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-9)
    
    def step(self, kernel=None):
        """
        Override step to include TSP-specific dynamics:
        1. Cost biasing toward short edges
        2. Soft normalization (tour constraints)
        3. Memory-coherent evolution
        """
        # Store current state
        psi_old = self.psi.copy()
        
        # --- TSP-specific cost biasing ---
        cost_bias = np.exp(-self.alpha * self.distance_matrix)
        self.psi = self.psi * cost_bias
        
        # --- Soft normalization (tour constraints) ---
        # Each row should sum to ~1 (one outgoing edge per city)
        row_sum = np.abs(self.psi).sum(axis=1, keepdims=True)
        self.psi = self.psi / (row_sum + 1e-9)
        
        # Each column should sum to ~1 (one incoming edge per city)
        col_sum = np.abs(self.psi).sum(axis=0, keepdims=True)
        self.psi = self.psi / (col_sum + 1e-9)
        
        # Global normalization
        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-9)
        
        # --- Memory contribution (if kernel provided) ---
        if kernel is not None:
            # Update circular memory
            self.memory_history = np.roll(self.memory_history, 1, axis=0)
            self.memory_history[0] = psi_old.copy()
            
            # Compute memory term
            try:
                T = min(len(kernel), self.memory_history.shape[0])
                kernel_flat = np.ravel(kernel[:T])
                mem = np.tensordot(kernel_flat, self.memory_history[:T], axes=(0, 0))
                
                # Blend current state with memory
                memory_strength = 0.3
                self.psi = (1 - memory_strength) * self.psi + memory_strength * mem
                
                # Renormalize after memory
                self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-9)
            except:
                pass
        
        # --- Coherence push (late-stage sharpening) ---
        # Encourage binary-like solutions by amplifying strong edges
        amplitudes = np.abs(self.psi)
        threshold = np.percentile(amplitudes, 70)  # Top 30% of edges
        mask = amplitudes > threshold
        self.psi = np.where(mask, self.psi * 1.1, self.psi * 0.9)
        
        # Final normalization
        self.psi = self.psi / (np.linalg.norm(self.psi) + 1e-9)
        np.fill_diagonal(self.psi, 0)
        
        # Update diagnostics
        self.meta = {
            "mean_amp": float(np.mean(np.abs(self.psi))),
            "phase_coherence": float(np.abs(np.mean(np.exp(1j * np.angle(self.psi))))),
        }
        
        return self.meta
    
    def decode_tour(self, start=0):
        """
        Decode field into tour via greedy following.
        
        Returns:
            tour: list of city indices
        """
        tour = [start]
        visited = set([start])
        current = start
        
        for _ in range(self.n_cities - 1):
            # Get edge amplitudes from current city
            probs = np.abs(self.psi[current])
            
            # Mask visited cities
            for v in visited:
                probs[v] = 0
            
            # Pick highest amplitude unvisited city
            if probs.max() > 0:
                next_city = int(np.argmax(probs))
            else:
                # Fallback: pick any unvisited
                remaining = list(set(range(self.n_cities)) - visited)
                next_city = remaining[0] if remaining else start
            
            tour.append(next_city)
            visited.add(next_city)
            current = next_city
        
        return tour


# =====================================================
# BOUNDARY SETUP FOR TSP FIELDS
# =====================================================

def create_tsp_boundaries(K: int, n_cities: int):
    """
    Create boundary connections between TSP fields.
    
    For TSP, boundaries are edge-to-edge correspondences.
    We'll connect fields via shared edge subsets.
    
    Returns:
        boundaries: dict of (i,j) -> (idx_i, idx_j)
        lambdas: (K, K) coupling strengths
    """
    boundaries = {}
    lambdas = np.zeros((K, K))
    
    # Connect each field to its neighbors (ring topology)
    for i in range(K):
        for j in range(i+1, K):
            # Each pair shares a random subset of edges (20% of total)
            n_total = n_cities * n_cities
            n_shared = int(0.2 * n_total)
            
            # Random shared edge indices
            shared_idx = np.random.choice(n_total, n_shared, replace=False)
            
            boundaries[(i, j)] = (shared_idx, shared_idx)
            lambdas[i, j] = 0.5  # Moderate coupling
            lambdas[j, i] = 0.5
    
    return boundaries, lambdas


# =====================================================
# MULTI-FIELD TSP SOLVER
# =====================================================

def multi_field_tsp_solve(distance_matrix, K=8, steps=500, 
                          alpha=0.15, tau=32.0,
                          gate_params=None, verbose=True):
    """
    Multi-field TSP solver using your agency-gated coupling engine.
    
    Args:
        distance_matrix: (N, N) city distances
        K: number of fields
        steps: evolution steps
        alpha: cost bias strength
        tau: memory kernel timescale
        gate_params: GateParams for adaptive gates
        verbose: print progress
    
    Returns:
        best_tour: decoded tour from highest-coherence field
        best_cost: tour cost
        history: evolution diagnostics
    """
    n_cities = distance_matrix.shape[0]
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Multi-Field TSP Solver")
        print(f"Cities: {n_cities}, Fields: {K}, Steps: {steps}")
        print(f"{'='*70}\n")
    
    # --- Initialize K TSP fields ---
    fields = []
    for k in range(K):
        field = TSPField(n_cities, distance_matrix, dt=0.02, alpha=alpha)
        fields.append(field)
    
    # --- Setup boundaries and coupling ---
    boundaries, lambdas = create_tsp_boundaries(K, n_cities)
    
    if gate_params is None:
        gate_params = GateParams(
            alpha=2.0,   # Lower (easier to open gates)
            beta=1.0,    # Lower (less penalty for dissimilarity)
            floor=0.15,  # Higher (minimum coupling even when different)
            cap=0.85     # Slightly lower cap
        )
    
    gates = AdaptiveGates(gate_params)
    
    # --- Initialize agency ---
    agency = Agency(AgencyThresholds(pause_A=0.12, refuse_A=0.18, reframe_A=0.22))
    
    def agency_hook(field, A, coherence, t):
        """Apply agency decisions to each field"""
        agency.act(field, A=A, coherence=coherence, t=t)
    
    # --- Create multi-field engine ---
    engine = MultiFieldEngine(
        fields=fields,
        boundaries=boundaries,
        lambdas=lambdas,
        gates=gates,
        tension_fn=tsp_tension,  # Use TSP-specific tension
        agency_hook=agency_hook,
        eta=0.4,
        seed=42
    )
    
    # --- Memory kernel ---
    kernel = exp_kernel(length=128, tau=tau)
    
    # --- Evolution history ---
    history = {
        'coherences': [],
        'tensions': [],
        'mean_gates': [],
        'best_costs': []
    }
    
    # --- Main evolution loop ---
    if verbose:
        print("Evolving fields...\n")
    
    for t in range(steps):
        # Step all fields with coupling
        diag = engine.step(t, kernel=kernel)
        
        # Track diagnostics
        history['coherences'].append(diag.C.copy())
        history['tensions'].append(diag.A.copy())
        history['mean_gates'].append(float(np.mean(diag.G[diag.G > 0])))
        
        # Decode tours and track best
        costs = []
        for k in range(K):
            tour = fields[k].decode_tour()
            cost = compute_tour_cost(tour, distance_matrix)
            costs.append(cost)
        
        best_cost = min(costs)
        history['best_costs'].append(best_cost)
        
        # Progress report
        if verbose and (t % 100 == 0 or t == steps - 1):
            mean_coh = np.mean(diag.C)
            mean_tension = np.mean(diag.A)
            mean_gate = history['mean_gates'][-1]
            
            print(f"Step {t:4d}: "
                  f"Coherence={mean_coh:.3f}, "
                  f"Tension={mean_tension:.4f}, "
                  f"Gates={mean_gate:.3f}, "
                  f"Best={best_cost:.1f}")
    
    # --- Final selection: highest coherence field ---
    final_coherences = diag.C
    best_idx = np.argmax(final_coherences)
    
    best_field = fields[best_idx]
    best_tour = best_field.decode_tour()
    best_cost = compute_tour_cost(best_tour, distance_matrix)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FINAL RESULT:")
        print(f"  Selected field {best_idx} (coherence: {final_coherences[best_idx]:.3f})")
        print(f"  Tour cost: {best_cost:.2f}")
        print(f"{'='*70}\n")
    
    return best_tour, best_cost, history


# =====================================================
# UTILITIES
# =====================================================

def compute_tour_cost(tour, distance_matrix):
    """Compute total tour distance"""
    cost = 0
    for i in range(len(tour)):
        cost += distance_matrix[tour[i], tour[(i + 1) % len(tour)]]
    return cost


def nearest_neighbor(distance_matrix, start=0):
    """Baseline greedy algorithm"""
    n = distance_matrix.shape[0]
    tour = [start]
    unvisited = set(range(n)) - {start}
    
    while unvisited:
        last = tour[-1]
        nearest = min(unvisited, key=lambda c: distance_matrix[last, c])
        tour.append(nearest)
        unvisited.remove(nearest)
    
    return tour


# =====================================================
# VISUALIZATION
# =====================================================

def plot_evolution(history, save_path='tsp_evolution.png'):
    """Plot evolution statistics"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Coherence evolution
    ax = axes[0, 0]
    coherences = np.array(history['coherences'])
    for k in range(coherences.shape[1]):
        ax.plot(coherences[:, k], alpha=0.6, linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Field Coherence')
    ax.set_title('Coherence Evolution per Field')
    ax.grid(True, alpha=0.3)
    
    # Tension evolution
    ax = axes[0, 1]
    tensions = np.array(history['tensions'])
    for k in range(tensions.shape[1]):
        ax.plot(tensions[:, k], alpha=0.6, linewidth=1)
    ax.set_xlabel('Step')
    ax.set_ylabel('Tension (A)')
    ax.set_title('Tension Evolution per Field')
    ax.grid(True, alpha=0.3)
    
    # Mean gate strength
    ax = axes[1, 0]
    ax.plot(history['mean_gates'], linewidth=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Gate Strength')
    ax.set_title('Coupling Dynamics')
    ax.grid(True, alpha=0.3)
    
    # Best cost evolution
    ax = axes[1, 1]
    ax.plot(history['best_costs'], linewidth=2, color='green')
    ax.set_xlabel('Step')
    ax.set_ylabel('Best Tour Cost')
    ax.set_title('Solution Quality Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved evolution plot: {save_path}")


def plot_tour(tour, positions, title, save_path):
    """Plot TSP tour"""
    plt.figure(figsize=(10, 10))
    
    # Draw tour edges
    for i in range(len(tour)):
        p1 = positions[tour[i]]
        p2 = positions[tour[(i + 1) % len(tour)]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', alpha=0.6, linewidth=2)
    
    # Draw cities
    plt.scatter(positions[:, 0], positions[:, 1], c='red', s=100, zorder=5)
    
    # Mark start
    plt.scatter([positions[tour[0], 0]], [positions[tour[0], 1]], 
                c='green', s=300, marker='*', zorder=10, label='Start')
    
    plt.title(title, fontsize=14, weight='bold')
    plt.axis('equal')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved tour plot: {save_path}")


# =====================================================
# EXAMPLE USAGE
# =====================================================

def run_experiment(n_cities=100, K=8, steps=500, output_dir='multi_field_tsp'):
    """
    Complete experiment: multi-field vs single-field vs baseline
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*70}")
    print(f"Multi-Field TSP Experiment")
    print(f"{'='*70}\n")
    
    # Generate problem
    np.random.seed(42)
    positions = np.random.rand(n_cities, 2) * 100
    distance_matrix = np.linalg.norm(
        positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
    )
    
    # Baseline
    print("Computing baseline (Nearest Neighbor)...")
    nn_tour = nearest_neighbor(distance_matrix)
    nn_cost = compute_tour_cost(nn_tour, distance_matrix)
    print(f"NN cost: {nn_cost:.2f}\n")
    
    # Multi-field solve
    print("Running multi-field solver...")
    start = time.time()
    tour, cost, history = multi_field_tsp_solve(
        distance_matrix,
        K=K,
        steps=steps,
        alpha=0.15,
        tau=32.0,
        verbose=True
    )
    elapsed = time.time() - start
    
    ratio = cost / nn_cost
    print(f"\nMulti-field result:")
    print(f"  Cost: {cost:.2f}")
    print(f"  vs NN: {ratio:.3f}× ({'better' if ratio < 1 else 'worse'})")
    print(f"  Time: {elapsed:.2f}s")
    
    # Plots
    plot_evolution(history, 
                   save_path=os.path.join(output_dir, f'evolution_{timestamp}.png'))
    
    plot_tour(tour, positions,
              f'Multi-Field TSP ({K} fields, {n_cities} cities)\nCost: {cost:.1f} ({ratio:.2f}× NN)',
              save_path=os.path.join(output_dir, f'tour_{timestamp}.png'))
    
    return {
        'nn_cost': nn_cost,
        'multi_cost': cost,
        'ratio': ratio,
        'time': elapsed,
        'history': history
    }


if __name__ == "__main__":
    """
    USAGE EXAMPLES:
    
    # Small test (quick)
    results = run_experiment(n_cities=50, K=5, steps=300)
    
    # Medium scale (your 2k result)
    results = run_experiment(n_cities=2000, K=10, steps=500)
    
    # Large scale (your 10k result)
    results = run_experiment(n_cities=10000, K=15, steps=1000)
    """
    
    # Start with moderate size
    results = run_experiment(n_cities=100, K=8, steps=500)
    
    print("\n" + "="*70)
    print("NEXT STEPS:")
    print("="*70)
    print("\n1. Scale up:")
    print("   results = run_experiment(n_cities=2000, K=10, steps=500)")
    print("\n2. Tune coupling:")
    print("   - Adjust gate_params (alpha, beta)")
    print("   - Try different K (number of fields)")
    print("   - Vary tau (memory timescale)")
    print("\n3. Ablation study:")
    print("   - Run with lambdas=0 (no coupling) vs full coupling")
    print("   - Compare agency_hook vs no agency")
    print("   - Test different boundary topologies")