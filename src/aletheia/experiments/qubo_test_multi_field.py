"""
QUBO Multi-Field Solver for TSP/CVRP
Tests whether agency-gated field decomposition can solve real combinatorial problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import networkx as nx
from pathlib import Path

from aletheia.core.multifield_engine import MultiFieldEngine, AdaptiveGates, GateParams
from aletheia.core.field import Field, FieldConfig
from aletheia.core.memory import exp_kernel


# ============================================================
# PROBLEM GENERATORS
# ============================================================

@dataclass
class TSPInstance:
    """Traveling Salesman Problem instance"""
    n_cities: int
    coords: np.ndarray  # (n_cities, 2)
    distance_matrix: np.ndarray  # (n_cities, n_cities)
    
    @classmethod
    def random_euclidean(cls, n_cities: int, seed: int = 42):
        """Generate random Euclidean TSP instance"""
        rng = np.random.default_rng(seed)
        coords = rng.uniform(0, 100, size=(n_cities, 2))
        
        # Compute distance matrix
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        return cls(n_cities=n_cities, coords=coords, distance_matrix=dist_matrix)
    
    @classmethod
    def clustered(cls, n_clusters: int, cities_per_cluster: int, 
                  cluster_spread: float = 10.0, cluster_separation: float = 50.0, 
                  seed: int = 42):
        """Generate clustered TSP instance (good for decomposition)"""
        rng = np.random.default_rng(seed)
        n_cities = n_clusters * cities_per_cluster
        coords = np.zeros((n_cities, 2))
        
        # Place cluster centers on a circle
        angles = np.linspace(0, 2*np.pi, n_clusters, endpoint=False)
        cluster_centers = cluster_separation * np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Add cities around each cluster
        for cluster_id in range(n_clusters):
            start_idx = cluster_id * cities_per_cluster
            end_idx = start_idx + cities_per_cluster
            
            # Cities scattered around cluster center
            offsets = rng.normal(0, cluster_spread, size=(cities_per_cluster, 2))
            coords[start_idx:end_idx] = cluster_centers[cluster_id] + offsets
        
        # Compute distance matrix
        dist_matrix = np.zeros((n_cities, n_cities))
        for i in range(n_cities):
            for j in range(n_cities):
                if i != j:
                    dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        return cls(n_cities=n_cities, coords=coords, distance_matrix=dist_matrix)
    
    def tour_length(self, tour: np.ndarray) -> float:
        """Calculate total tour length"""
        length = 0.0
        for i in range(len(tour)):
            length += self.distance_matrix[tour[i], tour[(i+1) % len(tour)]]
        return length
    
    def nearest_neighbor_heuristic(self) -> Tuple[np.ndarray, float]:
        """Simple nearest neighbor heuristic for baseline"""
        unvisited = set(range(self.n_cities))
        tour = [0]  # Start at city 0
        unvisited.remove(0)
        
        while unvisited:
            current = tour[-1]
            nearest = min(unvisited, key=lambda city: self.distance_matrix[current, city])
            tour.append(nearest)
            unvisited.remove(nearest)
        
        tour = np.array(tour)
        return tour, self.tour_length(tour)


@dataclass
class CVRPInstance:
    """Capacitated Vehicle Routing Problem instance"""
    n_customers: int
    depot_idx: int  # Usually 0
    coords: np.ndarray  # (n_customers+1, 2) including depot
    demands: np.ndarray  # (n_customers,)
    vehicle_capacity: float
    distance_matrix: np.ndarray
    
    @classmethod
    def from_tsp(cls, tsp: TSPInstance, n_vehicles: int, seed: int = 42):
        """Convert TSP to CVRP by adding depot and demands"""
        rng = np.random.default_rng(seed)
        
        # Depot at center
        depot_coord = np.mean(tsp.coords, axis=0, keepdims=True)
        coords = np.vstack([depot_coord, tsp.coords])
        
        # Random demands
        total_demand = 100.0 * n_vehicles
        demands = rng.uniform(5, 20, size=tsp.n_cities)
        demands = demands * (total_demand / demands.sum())  # Normalize
        
        vehicle_capacity = total_demand / n_vehicles * 1.2  # 20% slack
        
        # Distance matrix with depot
        n_total = tsp.n_cities + 1
        dist_matrix = np.zeros((n_total, n_total))
        for i in range(n_total):
            for j in range(n_total):
                if i != j:
                    dist_matrix[i, j] = np.linalg.norm(coords[i] - coords[j])
        
        return cls(
            n_customers=tsp.n_cities,
            depot_idx=0,
            coords=coords,
            demands=demands,
            vehicle_capacity=vehicle_capacity,
            distance_matrix=dist_matrix
        )


# ============================================================
# TSP TO QUBO ENCODING
# ============================================================

class TSPToQUBO:
    """
    Encode TSP as QUBO using position-based representation.
    Binary variable x_{i,t} = 1 if city i is visited at time t.
    """
    
    def __init__(self, tsp: TSPInstance, penalty_scale: float = 2.0):
        self.tsp = tsp
        self.n = tsp.n_cities
        self.penalty_scale = penalty_scale
        
        # QUBO is n x n matrix (city × time)
        self.qubo_size = self.n * self.n
        self.Q = self._build_qubo()
    
    def _build_qubo(self) -> np.ndarray:
        """Construct QUBO matrix"""
        n = self.n
        Q = np.zeros((self.qubo_size, self.qubo_size))
        
        # Penalty for visiting city i at multiple times
        penalty = self.penalty_scale * np.max(self.tsp.distance_matrix)
        
        for i in range(n):
            for t1 in range(n):
                idx1 = i * n + t1
                
                # Diagonal: negative reward for visiting (encourages coverage)
                Q[idx1, idx1] = -1.0
                
                # Penalty for same city at different times
                for t2 in range(t1+1, n):
                    idx2 = i * n + t2
                    Q[idx1, idx2] += penalty
                    Q[idx2, idx1] += penalty
        
        # Penalty for multiple cities at same time
        for t in range(n):
            for i1 in range(n):
                idx1 = i1 * n + t
                for i2 in range(i1+1, n):
                    idx2 = i2 * n + t
                    Q[idx1, idx2] += penalty
                    Q[idx2, idx1] += penalty
        
        # Distance costs: if city i at time t and city j at time t+1
        for t in range(n):
            t_next = (t + 1) % n
            for i in range(n):
                for j in range(n):
                    if i != j:
                        idx_i = i * n + t
                        idx_j = j * n + t_next
                        cost = self.tsp.distance_matrix[i, j]
                        Q[idx_i, idx_j] += cost / 2
                        Q[idx_j, idx_i] += cost / 2
        
        return Q
    
    def decode_solution(self, spin_config: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Decode binary/spin configuration to TSP tour.
        Returns: (tour, quality, is_valid)
        """
        n = self.n
        
        # Convert to binary if spin
        if np.min(spin_config) < 0:
            binary = (spin_config + 1) / 2
        else:
            binary = spin_config
        
        # Reshape to (city, time)
        assignment = binary.reshape(n, n)
        
        # Extract tour by finding which city is visited at each time
        tour = []
        valid = True
        
        for t in range(n):
            cities_at_t = np.where(assignment[:, t] > 0.5)[0]
            if len(cities_at_t) == 1:
                tour.append(cities_at_t[0])
            elif len(cities_at_t) == 0:
                # No city at this time - invalid
                tour.append(-1)
                valid = False
            else:
                # Multiple cities - pick highest probability
                tour.append(cities_at_t[np.argmax(assignment[cities_at_t, t])])
                valid = False
        
        tour = np.array(tour)
        
        # Check if all cities visited exactly once
        if len(set(tour)) != n or -1 in tour:
            valid = False
        
        # Calculate quality (QUBO objective)
        if valid:
            quality = self.tsp.tour_length(tour)
        else:
            # For invalid tours, use QUBO value
            quality = binary @ self.Q @ binary
        
        return tour, quality, valid
    
    def partition_for_multifield(self, n_fields: int) -> Dict:
        """
        Partition QUBO variables into sub-problems for multi-field solver.
        
        Strategy: Spatial clustering of cities
        Each field handles a cluster of cities across all time steps.
        """
        n = self.n
        
        # Cluster cities spatially using k-means
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_fields, random_state=42)
        city_clusters = kmeans.fit_predict(self.tsp.coords)
        
        # Build partition info
        partition = {
            'n_fields': n_fields,
            'city_clusters': city_clusters,
            'field_cities': [np.where(city_clusters == k)[0] for k in range(n_fields)],
            'boundaries': {}  # Will store shared variables
        }
        
        # Identify boundary variables (edges between clusters)
        # A variable x_{i,t} is on boundary if city i in cluster k
        # and there exists edge (i,j) with j in different cluster
        
        for k in range(n_fields):
            partition[f'field_{k}_vars'] = []
            cities_in_k = partition['field_cities'][k]
            
            # All (city, time) pairs for cities in this cluster
            for i in cities_in_k:
                for t in range(n):
                    var_idx = i * n + t
                    partition[f'field_{k}_vars'].append(var_idx)
        
        # Boundary variables: time slots where different clusters must coordinate
        # For each time t, if multiple clusters have cities, those time steps are shared
        for t in range(n):
            clusters_at_t = set()
            for k in range(n_fields):
                if len(partition['field_cities'][k]) > 0:
                    clusters_at_t.add(k)
            
            # If multiple clusters active at time t, mark as boundary
            if len(clusters_at_t) > 1:
                for k in clusters_at_t:
                    for k2 in clusters_at_t:
                        if k < k2:
                            if (k, k2) not in partition['boundaries']:
                                partition['boundaries'][(k, k2)] = []
                            
                            # Shared variables: cities in different clusters at same time
                            vars_k = [i * n + t for i in partition['field_cities'][k]]
                            vars_k2 = [i * n + t for i in partition['field_cities'][k2]]
                            partition['boundaries'][(k, k2)].append((vars_k, vars_k2))
        
        return partition


# ============================================================
# MULTI-FIELD TSP SOLVER
# ============================================================

class MultiFieldTSPSolver:
    """
    Solve TSP using multi-field QFCA with agency-gated coupling.
    """
    
    def __init__(
        self,
        tsp_instance: TSPInstance,
        n_fields: int,
        gate_params: GateParams,
        field_size: int = 16,  # Spatial resolution for field representation
        output_dir: str = "tsp_results"
    ):
        self.tsp = tsp_instance
        self.n_fields = n_fields
        self.qubo_encoder = TSPToQUBO(tsp_instance)
        self.partition = self.qubo_encoder.partition_for_multifield(n_fields)
        self.field_size = field_size
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Build multi-field engine
        self.fields, self.boundaries, self.lambdas = self._build_fields()
        self.engine = MultiFieldEngine(
            fields=self.fields,
            boundaries=self.boundaries,
            lambdas=self.lambdas,
            gates=AdaptiveGates(gate_params),
            eta=0.4
        )
        
        # History tracking
        self.coherence_history = []
        self.solution_quality_history = []
        self.best_solution = None
        self.best_quality = float('inf')
        self.validity_history = []
    
    def _build_fields(self) -> Tuple[List[Field], Dict, np.ndarray]:
        """
        Build field representations for each partition.
        Each field's psi encodes spin configuration for its variables.
        """
        fields = []
        n = self.n_fields
        
        for k in range(n):
            cfg = FieldConfig(
                shape=(self.field_size, self.field_size),
                dt=0.05,
                seed=1000 + k
            )
            field = Field(cfg)
            
            # Initialize with problem-aware state
            # Encode QUBO submatrix into field potential
            var_indices = self.partition[f'field_{k}_vars']
            sub_Q = self.qubo_encoder.Q[np.ix_(var_indices, var_indices)]
            
            # Initialize psi with structure hint from QUBO
            # Use eigenvector of sub_Q as guide
            if len(var_indices) > 0:
                eigvals, eigvecs = np.linalg.eigh(sub_Q)
                # Use lowest eigenvector (ground state approximation)
                ground_hint = eigvecs[:, 0]
                
                # Map to field
                psi_flat = np.zeros(self.field_size * self.field_size, dtype=complex)
                for i, val in enumerate(ground_hint[:len(psi_flat)]):
                    psi_flat[i] = val + 1j * np.random.randn() * 0.1
                
                field.psi = psi_flat.reshape(field.cfg.shape)
            else:
                field.psi = np.random.randn(*field.cfg.shape) + \
                           1j * np.random.randn(*field.cfg.shape)
            
            # Inject QUBO energy landscape as local potential
            field.potential = self._qubo_to_potential(sub_Q, field.cfg.shape)
            
            fields.append(field)
        
        # Build boundary coupling
        boundaries = {}
        lambdas = np.zeros((n, n))
        
        for (k1, k2), boundary_vars in self.partition['boundaries'].items():
            # Shared indices in flattened field
            # For simplicity: share first N positions where N = overlap size
            overlap_size = min(50, self.field_size * self.field_size // 4)
            shared_idx = np.arange(overlap_size)
            
            boundaries[(k1, k2)] = (shared_idx, shared_idx)
            lambdas[k1, k2] = 1.0
            lambdas[k2, k1] = 1.0
        
        return fields, boundaries, lambdas
    
    def _qubo_to_potential(self, Q_sub: np.ndarray, field_shape: Tuple) -> np.ndarray:
        """
        Convert QUBO submatrix to field potential V(x,y).
        This creates an energy landscape that guides field evolution.
        """
        size = field_shape[0] * field_shape[1]
        potential = np.zeros(size)
        
        # Map diagonal elements of Q to potential
        diag = np.diag(Q_sub)
        potential[:len(diag)] = diag
        
        return potential.reshape(field_shape)
    
    def _extract_solution_from_fields(self) -> Tuple[np.ndarray, float, bool]:
        """
        Extract TSP solution from current field states.
        """
        # Collect spin configuration from all fields
        full_config = np.zeros(self.qubo_encoder.qubo_size)
        
        for k in range(self.n_fields):
            var_indices = self.partition[f'field_{k}_vars']
            psi_flat = self.fields[k].psi.ravel()
            
            # Convert field amplitude to spins
            spins = np.sign(np.real(psi_flat[:len(var_indices)]))
            full_config[var_indices] = spins
        
        return self.qubo_encoder.decode_solution(full_config)
    
    def solve(
        self,
        max_steps: int = 500,
        check_interval: int = 50,
        kernel: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run multi-field optimization.
        """
        if kernel is None:
            kernel = exp_kernel(length=self.field_size**2, tau=20.0)
        
        print(f"\n{'='*60}")
        print(f"MULTI-FIELD TSP SOLVER")
        print(f"{'='*60}")
        print(f"Cities: {self.tsp.n_cities}")
        print(f"Fields: {self.n_fields}")
        print(f"QUBO size: {self.qubo_encoder.qubo_size}")
        print(f"Partition: {[len(self.partition['field_cities'][k]) for k in range(self.n_fields)]} cities/field")
        print(f"{'='*60}\n")
        
        # Baseline: Nearest neighbor heuristic
        nn_tour, nn_quality = self.tsp.nearest_neighbor_heuristic()
        print(f"Baseline (Nearest Neighbor): {nn_quality:.2f}")
        print(f"Starting optimization...\n")
        
        solution_check_steps = []
        
        for t in range(max_steps):
            # Step multi-field engine
            diag = self.engine.step(t, kernel=kernel)
            
            # Track coherence
            self.coherence_history.append(self.engine.mean_coherence())
            
            # Check solution quality periodically
            if t % check_interval == 0:
                tour, quality, valid = self._extract_solution_from_fields()
                self.solution_quality_history.append(quality)
                self.validity_history.append(valid)
                solution_check_steps.append(t)
                
                if valid and quality < self.best_quality:
                    self.best_quality = quality
                    self.best_solution = tour
                    improvement = (nn_quality - quality) / nn_quality * 100
                    print(f"t={t:3d} | C={self.engine.mean_coherence():.3f} | "
                          f"Quality={quality:.2f} | Valid={valid} | "
                          f"Best={self.best_quality:.2f} ({improvement:+.1f}% vs NN)")
                else:
                    print(f"t={t:3d} | C={self.engine.mean_coherence():.3f} | "
                          f"Quality={quality:.2f} | Valid={valid}")
        
        # Final solution
        final_tour, final_quality, final_valid = self._extract_solution_from_fields()
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Baseline (NN): {nn_quality:.2f}")
        print(f"Best found: {self.best_quality:.2f} ({(nn_quality-self.best_quality)/nn_quality*100:+.1f}%)")
        print(f"Final: {final_quality:.2f} (Valid: {final_valid})")
        
        return {
            'best_tour': self.best_solution,
            'best_quality': self.best_quality,
            'baseline_quality': nn_quality,
            'coherence_history': np.array(self.coherence_history),
            'solution_quality_history': self.solution_quality_history,
            'solution_check_steps': solution_check_steps,
            'validity_history': self.validity_history
        }

    def visualize_results(self, results: Dict):
        """
        Create comprehensive visualization of results.
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Row 1: TSP instance and solutions
        # Original problem
        ax = fig.add_subplot(gs[0, 0])
        ax.scatter(self.tsp.coords[:, 0], self.tsp.coords[:, 1], 
                  c='blue', s=100, alpha=0.6, edgecolors='black')
        for i, (x, y) in enumerate(self.tsp.coords):
            ax.text(x, y, str(i), ha='center', va='center', fontsize=8)
        
        # Show cluster assignment
        for k in range(self.n_fields):
            cities_in_k = self.partition['field_cities'][k]
            if len(cities_in_k) > 0:
                cluster_coords = self.tsp.coords[cities_in_k]
                hull = self._convex_hull(cluster_coords)
                ax.fill(hull[:, 0], hull[:, 1], alpha=0.2, label=f'Field {k}')
        
        ax.set_title('TSP Instance & Partitioning')
        ax.legend()
        ax.axis('equal')
        
        # Baseline solution
        ax = fig.add_subplot(gs[0, 1])
        nn_tour, nn_quality = self.tsp.nearest_neighbor_heuristic()
        self._plot_tour(ax, nn_tour, f'Baseline (NN): {nn_quality:.2f}')
        
        # Best solution found
        ax = fig.add_subplot(gs[0, 2])
        if self.best_solution is not None:
            self._plot_tour(ax, self.best_solution, 
                          f'Best Found: {self.best_quality:.2f}')
        else:
            ax.text(0.5, 0.5, 'No valid solution found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Best Solution')
    
        # Row 2: Optimization dynamics
        # Coherence evolution
        ax = fig.add_subplot(gs[1, :2])
        ax.plot(results['coherence_history'], 'b-', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence')
        ax.grid(alpha=0.3)
        ax.set_title('Field Coherence Evolution')
        
        # Solution quality evolution
        ax = fig.add_subplot(gs[1, 2])
        quality_steps = results.get('solution_check_steps', 
                                    np.arange(len(results['solution_quality_history'])) * 50)
        ax.plot(quality_steps, results['solution_quality_history'], 'go-', lw=2, markersize=6)
        ax.axhline(results['baseline_quality'], color='r', linestyle='--', 
                  label='Baseline', lw=2)
        if self.best_quality < float('inf'):
            ax.axhline(self.best_quality, color='orange', linestyle=':', 
                      label='Best', lw=2)
        ax.set_xlabel('Step')
        ax.set_ylabel('Tour Length')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Solution Quality Over Time')
        
        # Row 3: Field analysis
        # Final gate matrix
        ax = fig.add_subplot(gs[2, 0])
        final_gates = self.engine.G
        im = ax.imshow(final_gates, cmap='hot', vmin=0, vmax=1)
        ax.set_title('Final Gate Matrix')
        ax.set_xlabel('Field j')
        ax.set_ylabel('Field i')
        plt.colorbar(im, ax=ax, label='Gate strength')
        
        # Coherence-Quality correlation
        ax = fig.add_subplot(gs[2, 1])
        # Plot coherence at solution check steps
        coherence_at_checks = [results['coherence_history'][s] for s in results['solution_check_steps']]
        ax.scatter(coherence_at_checks, results['solution_quality_history'], 
                  c=results['solution_check_steps'], cmap='viridis', s=100, alpha=0.7)
        ax.set_xlabel('Coherence')
        ax.set_ylabel('Tour Length')
        ax.set_title('Coherence vs Quality')
        ax.grid(alpha=0.3)
        
        # Validity over time
        ax = fig.add_subplot(gs[2, 2])
        validity_int = [1 if v else 0 for v in results['validity_history']]
        ax.plot(quality_steps, validity_int, 'ro-', lw=2, markersize=6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Valid Solution')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.grid(alpha=0.3)
        ax.set_title('Solution Validity')
        
        plt.savefig(self.output_dir / f"tsp_n{self.tsp.n_cities}_f{self.n_fields}.png", 
                   dpi=150, bbox_inches='tight')
        print(f"\n→ Saved visualization to {self.output_dir}/")
    
    def _plot_tour(self, ax, tour, title):
        """Plot TSP tour"""
        if tour is None or -1 in tour:
            ax.text(0.5, 0.5, 'Invalid tour', ha='center', va='center',
                   transform=ax.transAxes)
            ax.set_title(title)
            return
        
        coords = self.tsp.coords
        ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, 
                  alpha=0.6, edgecolors='black', zorder=3)
        
        # Draw tour
        for i in range(len(tour)):
            start = tour[i]
            end = tour[(i+1) % len(tour)]
            ax.plot([coords[start, 0], coords[end, 0]],
                   [coords[start, 1], coords[end, 1]],
                   'g-', lw=2, alpha=0.7, zorder=1)
        
        ax.set_title(title)
        ax.axis('equal')
    
    def _convex_hull(self, points):
        """Compute convex hull for visualization"""
        from scipy.spatial import ConvexHull
        if len(points) < 3:
            return points
        hull = ConvexHull(points)
        return points[hull.vertices]


# ============================================================
# EXPERIMENTAL SUITE
# ============================================================

def experiment_1_small_tsp():
    """
    Experiment 1: Small random TSP (10 cities, 3 fields)
    Sanity check - can system find reasonable solutions?
    """
    print("\n" + "="*60)
    print("EXPERIMENT 1: Small Random TSP")
    print("="*60)
    
    tsp = TSPInstance.random_euclidean(n_cities=10, seed=42)
    solver = MultiFieldTSPSolver(
        tsp_instance=tsp,
        n_fields=3,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.05),
        field_size=12,
        output_dir="tsp_experiments/exp1_small"
    )
    
    results = solver.solve(max_steps=500, check_interval=25)
    solver.visualize_results(results)
    
    return results


def experiment_2_clustered_tsp():
    """
    Experiment 2: Clustered TSP (16 cities, 4 clusters, 4 fields)
    Natural decomposition - each field handles one cluster.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 2: Clustered TSP")
    print("="*60)
    
    tsp = TSPInstance.clustered(n_clusters=4, cities_per_cluster=4, seed=43)
    solver = MultiFieldTSPSolver(
        tsp_instance=tsp,
        n_fields=4,
        gate_params=GateParams(alpha=6.0, beta=6.0, floor=0.02),
        field_size=16,
        output_dir="tsp_experiments/exp2_clustered"
    )
    
    results = solver.solve(max_steps=600, check_interval=30)
    solver.visualize_results(results)
    
    return results


def experiment_3_scaling():
    """
    Experiment 3: Scaling test
    Fix problem size (12 cities), vary number of fields (2, 3, 4, 6).
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: Scaling Analysis")
    print("="*60)
    
    tsp = TSPInstance.random_euclidean(n_cities=12, seed=44)
    n_fields_list = [2, 3, 4, 6]
    
    results_by_n = {}
    
    for n_fields in n_fields_list:
        print(f"\n--- Testing with {n_fields} fields ---")
        solver = MultiFieldTSPSolver(
            tsp_instance=tsp,
            n_fields=n_fields,
            gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.05),
            field_size=12,
            output_dir=f"tsp_experiments/exp3_scaling/n{n_fields}"
        )
        
        results = solver.solve(max_steps=400, check_interval=50)
        results_by_n[n_fields] = results
        solver.visualize_results(results)
        # Comparative visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Solution quality comparison
    ax = axes[0]
    baseline = results_by_n[n_fields_list[0]]['baseline_quality']
    for n_fields in n_fields_list:
        best_q = results_by_n[n_fields]['best_quality']
        ax.bar(n_fields, best_q, alpha=0.7, label=f'N={n_fields}')
    ax.axhline(baseline, color='r', linestyle='--', lw=2, label='Baseline')
    ax.set_xlabel('Number of Fields')
    ax.set_ylabel('Best Tour Length')
    ax.set_title('Solution Quality vs. Field Count')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Coherence evolution comparison
    ax = axes[1]
    for n_fields in n_fields_list:
        coh = results_by_n[n_fields]['coherence_history']
        ax.plot(coh, label=f'N={n_fields}', lw=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Coherence')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Coherence Evolution')
    
    # Improvement over baseline
    ax = axes[2]
    improvements = []
    for n_fields in n_fields_list:
        best_q = results_by_n[n_fields]['best_quality']
        improvement = (baseline - best_q) / baseline * 100
        improvements.append(improvement)
    ax.plot(n_fields_list, improvements, 'go-', markersize=10, lw=2)
    ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax.set_xlabel('Number of Fields')
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.grid(alpha=0.3)
    ax.set_title('Multi-Field Performance Gain')
    
    plt.tight_layout()
    plt.savefig("tsp_experiments/exp3_scaling/scaling_comparison.png", dpi=150)
    print("\n→ Saved scaling comparison")
    
    return results_by_n


def experiment_4_cvrp():
    """
    Experiment 4: Vehicle Routing Problem
    Test on more complex problem with capacity constraints.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 4: CVRP (Vehicle Routing)")
    print("="*60)
    
    # Create base TSP
    base_tsp = TSPInstance.clustered(n_clusters=3, cities_per_cluster=4, seed=45)
    cvrp = CVRPInstance.from_tsp(base_tsp, n_vehicles=3, seed=45)
    
    print(f"CVRP Instance:")
    print(f"  Customers: {cvrp.n_customers}")
    print(f"  Vehicles: 3")
    print(f"  Vehicle capacity: {cvrp.vehicle_capacity:.2f}")
    print(f"  Total demand: {cvrp.demands.sum():.2f}")
    
    # For now, solve as TSP ignoring capacity (full CVRP encoding is more complex)
    # This is a simplification to test the framework
    tsp_from_cvrp = TSPInstance(
        n_cities=cvrp.n_customers,
        coords=cvrp.coords[1:],  # Exclude depot for TSP
        distance_matrix=cvrp.distance_matrix[1:, 1:]
    )
    
    solver = MultiFieldTSPSolver(
        tsp_instance=tsp_from_cvrp,
        n_fields=3,  # One field per vehicle
        gate_params=GateParams(alpha=6.0, beta=6.0, floor=0.03),
        field_size=14,
        output_dir="tsp_experiments/exp4_cvrp"
    )
    
    results = solver.solve(max_steps=500, check_interval=25)
    solver.visualize_results(results)
    
    # Additional CVRP-specific visualization
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Plot depot
    depot = cvrp.coords[0]
    ax.scatter(depot[0], depot[1], c='red', s=300, marker='s', 
              label='Depot', zorder=5, edgecolors='black', linewidths=2)
    
    # Plot customers colored by demand
    customer_coords = cvrp.coords[1:]
    scatter = ax.scatter(customer_coords[:, 0], customer_coords[:, 1],
                        c=cvrp.demands, s=200, cmap='viridis', 
                        alpha=0.7, edgecolors='black', linewidths=1.5,
                        zorder=3)
    plt.colorbar(scatter, ax=ax, label='Demand')
    
    # Show cluster assignment
    for k in range(solver.n_fields):
        cities_in_k = solver.partition['field_cities'][k]
        if len(cities_in_k) > 0:
            cluster_coords = tsp_from_cvrp.coords[cities_in_k]
            hull = solver._convex_hull(cluster_coords)
            ax.fill(hull[:, 0], hull[:, 1], alpha=0.15, label=f'Route {k+1}')
    
    ax.set_title('CVRP Instance with Multi-Field Partitioning')
    ax.legend()
    ax.axis('equal')
    plt.tight_layout()
    plt.savefig("tsp_experiments/exp4_cvrp/cvrp_instance.png", dpi=150)
    
    return results


def experiment_5_single_vs_multi_field():
    """
    Experiment 5: Direct comparison - single field vs multi-field
    Same problem, different decompositions.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 5: Single vs Multi-Field Comparison")
    print("="*60)
    
    tsp = TSPInstance.random_euclidean(n_cities=15, seed=46)
    
    results = {}
    
    # Single field (baseline)
    print("\n--- Single Field (N=1) ---")
    solver_single = MultiFieldTSPSolver(
        tsp_instance=tsp,
        n_fields=1,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.05),
        field_size=16,
        output_dir="tsp_experiments/exp5_comparison/single"
    )
    results['single'] = solver_single.solve(max_steps=500, check_interval=25)
    solver_single.visualize_results(results['single'])
    
    # Multi-field (N=3)
    print("\n--- Multi-Field (N=3) ---")
    solver_multi3 = MultiFieldTSPSolver(
        tsp_instance=tsp,
        n_fields=3,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.05),
        field_size=16,
        output_dir="tsp_experiments/exp5_comparison/multi3"
    )
    results['multi3'] = solver_multi3.solve(max_steps=500, check_interval=25)
    solver_multi3.visualize_results(results['multi3'])
    
    # Multi-field (N=5)
    print("\n--- Multi-Field (N=5) ---")
    solver_multi5 = MultiFieldTSPSolver(
        tsp_instance=tsp,
        n_fields=5,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.05),
        field_size=16,
        output_dir="tsp_experiments/exp5_comparison/multi5"
    )
    results['multi5'] = solver_multi5.solve(max_steps=500, check_interval=25)
    solver_multi5.visualize_results(results['multi5'])
    
    # Comparative analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Coherence evolution
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['coherence_history'], label=name.upper(), lw=2, alpha=0.8)
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Coherence')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Coherence Evolution Comparison')
    
    # Solution quality over time
    ax = axes[0, 1]
    baseline = results['single']['baseline_quality']
    for name, res in results.items():
        steps = np.arange(0, len(res['coherence_history']), 25)
        ax.plot(steps, res['solution_quality_history'], 'o-', 
               label=name.upper(), lw=2, markersize=5)
    ax.axhline(baseline, color='r', linestyle='--', lw=2, label='Baseline')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tour Length')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Solution Quality Over Time')
    
    # Final comparison
    ax = axes[1, 0]
    names = list(results.keys())
    best_qualities = [results[name]['best_quality'] for name in names]
    colors = ['blue', 'green', 'orange']
    ax.bar(names, best_qualities, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(baseline, color='r', linestyle='--', lw=2, label='Baseline')
    ax.set_ylabel('Best Tour Length')
    ax.set_title('Final Solution Quality')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Improvement percentage
    ax = axes[1, 1]
    improvements = [(baseline - results[name]['best_quality']) / baseline * 100 
                   for name in names]
    ax.bar(names, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.axhline(0, color='r', linestyle='--', lw=1, alpha=0.5)
    ax.set_ylabel('Improvement over Baseline (%)')
    ax.set_title('Relative Performance')
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig("tsp_experiments/exp5_comparison/comparison_summary.png", dpi=150)
    print("\n→ Saved comparison summary")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("COMPARISON SUMMARY")
    print("="*60)
    print(f"{'Configuration':<15} {'Best Quality':<15} {'Improvement':<15} {'Valid Solutions'}")
    print("-"*60)
    print(f"{'Baseline':<15} {baseline:<15.2f} {'-':<15} {'-'}")
    for name in names:
        best = results[name]['best_quality']
        imp = (baseline - best) / baseline * 100
        n_valid = sum(results[name]['validity_history'])
        total = len(results[name]['validity_history'])
        print(f"{name.upper():<15} {best:<15.2f} {imp:<15.1f}% {n_valid}/{total}")
    
    return results


# =================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    # Run all experiments
    
    print("\n" + "="*70)
    print(" "*15 + "MULTI-FIELD QFCA TSP/CVRP EXPERIMENTS")
    print("="*70)
    
    # Experiment 1: Basic sanity check
    exp1_results = experiment_1_small_tsp()
    
    # Experiment 2: Clustered problem (natural decomposition)
    exp2_results = experiment_2_clustered_tsp()
    
    # Experiment 3: Scaling analysis
    exp3_results = experiment_3_scaling()
    
    # Experiment 4: CVRP test
    exp4_results = experiment_4_cvrp()
    
    # Experiment 5: Direct comparison
    exp5_results = experiment_5_single_vs_multi_field()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*70)
    print("\nResults saved to tsp_experiments/")
    print("\nKey Findings:")
    print("-" * 70)
    
    # Summary of exp5
    if exp5_results:
        baseline = exp5_results['single']['baseline_quality']
        single_best = exp5_results['single']['best_quality']
        multi3_best = exp5_results['multi3']['best_quality']
        multi5_best = exp5_results['multi5']['best_quality']
        
        print(f"Single Field: {(baseline-single_best)/baseline*100:+.1f}% vs baseline")
        print(f"Multi-Field (N=3): {(baseline-multi3_best)/baseline*100:+.1f}% vs baseline")
        print(f"Multi-Field (N=5): {(baseline-multi5_best)/baseline*100:+.1f}% vs baseline")
        
        if multi3_best < single_best:
            print(f"\n✓ Multi-field (N=3) BEATS single field by {(single_best-multi3_best)/single_best*100:.1f}%")
        else:
            print(f"\n✗ Multi-field (N=3) does NOT beat single field")
            
        if multi5_best < single_best:
            print(f"✓ Multi-field (N=5) BEATS single field by {(single_best-multi5_best)/single_best*100:.1f}%")
        else:
            print(f"✗ Multi-field (N=5) does NOT beat single field")
    
    print("\n" + "="*70)