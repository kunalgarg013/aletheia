"""
Multi-Field TSP Solver with Edge-Based QUBO Encoding
The CORRECT way to do this.
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
# EDGE-BASED TSP QUBO ENCODING
# ============================================================

class TSPToQUBO_EdgeBased:
    """
    Encode TSP using edge variables.
    Binary variable e_{ij} = 1 if edge (i,j) is in tour.
    
    This is the RIGHT encoding for QFCA.
    """
    
    def __init__(self, tsp, penalty_scale: float = 3.0):
        self.tsp = tsp
        self.n = tsp.n_cities
        self.penalty_scale = penalty_scale
        
        # Build edge list
        self.edge_list = []
        self.edge_to_idx = {}
        self.idx_to_edge = {}
        
        edge_idx = 0
        for i in range(self.n):
            for j in range(i+1, self.n):
                self.edge_list.append((i, j))
                self.edge_to_idx[(i, j)] = edge_idx
                self.edge_to_idx[(j, i)] = edge_idx  # Undirected
                self.idx_to_edge[edge_idx] = (i, j)
                edge_idx += 1
        
        self.n_edges = len(self.edge_list)
        self.qubo_size = self.n_edges
        
        print(f"Edge-based encoding: {self.n} cities → {self.n_edges} edges")
        
        self.Q = self._build_qubo()
    
    def _build_qubo(self) -> np.ndarray:
        """
        QUBO for TSP with edge variables.
        
        Minimize: Σ d_ij * e_ij  (tour length)
        Subject to: each city has degree 2
        """
        Q = np.zeros((self.n_edges, self.n_edges))
        penalty = self.penalty_scale * np.max(self.tsp.distance_matrix)
        
        print(f"Building QUBO with penalty scale: {penalty:.2f}")
        
        # 1. Objective: edge distances (linear terms)
        for edge_idx, (i, j) in enumerate(self.edge_list):
            dist = self.tsp.distance_matrix[i, j]
            Q[edge_idx, edge_idx] += dist
        
        # 2. Constraint: Each city must have degree exactly 2
        # For city c: (Σ_{e touching c} e - 2)² 
        # Expand: Σ e_i² + Σ_{i≠j} e_i*e_j - 4Σ e_i + 4
        # In QUBO form: Σ_{i<j} 2*e_i*e_j + Σ (1-4)*e_i + const
        
        for city in range(self.n):
            edges_touching = self._get_edges_touching(city)
            
            # Quadratic terms: e_i * e_j for all pairs
            for idx1, e1 in enumerate(edges_touching):
                # Linear coefficient: -3 (comes from expanding (x-2)²)
                Q[e1, e1] += penalty * (-3)
                
                # Quadratic interactions
                for e2 in edges_touching[idx1+1:]:
                    Q[e1, e2] += penalty * 2
                    Q[e2, e1] += penalty * 2
        
        return Q
    
    def _get_edges_touching(self, city: int) -> List[int]:
        """Get all edge indices that touch a given city"""
        touching = []
        for edge_idx, (i, j) in enumerate(self.edge_list):
            if i == city or j == city:
                touching.append(edge_idx)
        return touching
    
    def decode_solution(self, field_config: np.ndarray) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Decode field configuration to TSP tour.
        field_config: either spins {-1,+1} or binary {0,1}
        """
        # Convert to binary
        if np.min(field_config) < 0:
            binary = (field_config + 1) / 2
        else:
            binary = field_config
        
        # Threshold to get selected edges
        threshold = 0.5
        selected_edges = []
        edge_weights = []
        
        for edge_idx in range(min(len(binary), self.n_edges)):
            if binary[edge_idx] > threshold:
                selected_edges.append(self.edge_list[edge_idx])
                edge_weights.append(binary[edge_idx])
        
        # Try to construct tour
        tour, valid = self._edges_to_tour(selected_edges)
        
        if valid and tour is not None:
            quality = self.tsp.tour_length(tour)
        else:
            # Invalid solution - compute QUBO objective
            padded = np.zeros(self.n_edges)
            padded[:len(binary)] = binary[:self.n_edges]
            quality = padded @ self.Q @ padded
        
        return tour, quality, valid
    
    def _edges_to_tour(self, edges: List[Tuple[int, int]]) -> Tuple[Optional[np.ndarray], bool]:
        """
        Construct Hamiltonian tour from edge list.
        Uses graph traversal.
        """
        if len(edges) != self.n:
            # Wrong number of edges
            return None, False
        
        # Build adjacency list
        adj = {i: [] for i in range(self.n)}
        for i, j in edges:
            adj[i].append(j)
            adj[j].append(i)
        
        # Check degree constraint
        for city, neighbors in adj.items():
            if len(neighbors) != 2:
                return None, False
        
        # Follow path starting from city 0
        tour = [0]
        visited = {0}
        prev = -1
        current = 0
        
        for step in range(self.n - 1):
            neighbors = adj[current]
            
            # Pick unvisited neighbor (not where we came from)
            next_city = None
            for neighbor in neighbors:
                if neighbor != prev:
                    next_city = neighbor
                    break
            
            if next_city is None or next_city in visited:
                # Dead end or cycle
                return None, False
            
            tour.append(next_city)
            visited.add(next_city)
            prev = current
            current = next_city
        
        # Check if we can close the loop
        if 0 not in adj[current]:
            return None, False
        
        return np.array(tour), True
    
    def partition_for_multifield(self, n_fields: int) -> Dict:
        """
        Partition edges by spatial clustering of their midpoints.
        """
        # Compute edge midpoints
        edge_midpoints = np.zeros((self.n_edges, 2))
        for idx, (i, j) in enumerate(self.edge_list):
            edge_midpoints[idx] = (self.tsp.coords[i] + self.tsp.coords[j]) / 2
        
        # Cluster edges
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_fields, random_state=42)
        edge_clusters = kmeans.fit_predict(edge_midpoints)
        
        partition = {
            'n_fields': n_fields,
            'edge_clusters': edge_clusters,
            'field_edges': [np.where(edge_clusters == k)[0].tolist() for k in range(n_fields)],
            'edge_midpoints': edge_midpoints,
            'boundaries': {}
        }
        
        print(f"\nPartition: {[len(partition['field_edges'][k]) for k in range(n_fields)]} edges/field")
        
        # Identify boundary edges - edges that connect cities in different cluster regions
        # Or edges whose midpoint is near cluster boundaries
        for k1 in range(n_fields):
            for k2 in range(k1+1, n_fields):
                boundary_edges_k1 = []
                boundary_edges_k2 = []
                
                # Get edges near the boundary between clusters k1 and k2
                edges_k1 = partition['field_edges'][k1]
                edges_k2 = partition['field_edges'][k2]
                
                # Simple heuristic: edges are on boundary if their endpoints 
                # are close to edges in the other cluster
                for e1_idx in edges_k1:
                    i1, j1 = self.edge_list[e1_idx]
                    for e2_idx in edges_k2:
                        i2, j2 = self.edge_list[e2_idx]
                        # Check if edges share a vertex or are geometrically close
                        if (i1 == i2 or i1 == j2 or j1 == i2 or j1 == j2):
                            if e1_idx not in boundary_edges_k1:
                                boundary_edges_k1.append(e1_idx)
                            if e2_idx not in boundary_edges_k2:
                                boundary_edges_k2.append(e2_idx)
                
                # Take subset for boundary coupling (too many is expensive)
                max_boundary = 20
                if len(boundary_edges_k1) > max_boundary:
                    boundary_edges_k1 = boundary_edges_k1[:max_boundary]
                if len(boundary_edges_k2) > max_boundary:
                    boundary_edges_k2 = boundary_edges_k2[:max_boundary]
                
                if boundary_edges_k1 and boundary_edges_k2:
                    partition['boundaries'][(k1, k2)] = (boundary_edges_k1, boundary_edges_k2)
        
        print(f"Boundary edges: {[(k, len(v[0])) for k, v in partition['boundaries'].items()]}")
        
        return partition


# ============================================================
# ENHANCED FIELD WITH QUBO GRADIENT
# ============================================================

class QUBOField(Field):
    """
    Field that is driven by QUBO energy gradient.
    Combines diffusion dynamics with gradient descent on QUBO.
    """
    
    def __init__(self, cfg: FieldConfig, qubo_matrix: np.ndarray, var_indices: List[int]):
        super().__init__(cfg)
        self.qubo_matrix = qubo_matrix
        self.var_indices = var_indices
        self.qubo_alpha = 0.15  # Strength of QUBO gradient
    
    def step_with_qubo_gradient(self, kernel=None):
        """
        Enhanced step that includes QUBO gradient descent.
        """
        # Standard field evolution
        self.step(kernel=kernel)
        
        # Extract current spin configuration from field
        psi_flat = self.psi.ravel()
        n_vars = len(self.var_indices)
        spins = np.tanh(np.real(psi_flat[:n_vars]))  # Smooth approximation of sign
        
        # Compute QUBO gradient: ∂E/∂s_i = 2 * Σ_j Q_ij s_j
        qubo_gradient = 2.0 * (self.qubo_matrix @ spins)
        
        # Inject gradient into field (gradient descent on QUBO energy)
        gradient_field = np.zeros_like(psi_flat)
        gradient_field[:n_vars] = -qubo_gradient  # Negative for minimization
        
        # Apply gradient update
        self.psi -= self.qubo_alpha * gradient_field.reshape(self.psi.shape)
        
        # Renormalize to prevent explosion
        max_amp = np.max(np.abs(self.psi))
        if max_amp > 10.0:
            self.psi = self.psi / max_amp * 10.0


# ============================================================
# MULTI-FIELD TSP SOLVER (EDGE-BASED)
# ============================================================

class MultiFieldTSPSolver_EdgeBased:
    """
    Multi-field solver using edge-based QUBO encoding.
    Fields represent edge selection probabilities.
    """
    
    def __init__(
        self,
        tsp_instance,
        n_fields: int,
        gate_params: GateParams,
        field_size: int = 16,
        qubo_driven: bool = True,
        output_dir: str = "tsp_edge_results"
    ):
        self.tsp = tsp_instance
        self.n_fields = n_fields
        self.qubo_encoder = TSPToQUBO_EdgeBased(tsp_instance, penalty_scale=3.0)
        self.partition = self.qubo_encoder.partition_for_multifield(n_fields)
        self.field_size = field_size
        self.qubo_driven = qubo_driven
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Build fields
        self.fields, self.boundaries, self.lambdas = self._build_fields()
        
        # Build multi-field engine
        self.engine = MultiFieldEngine(
            fields=self.fields,
            boundaries=self.boundaries,
            lambdas=self.lambdas,
            gates=AdaptiveGates(gate_params),
            eta=0.35
        )
        
        # Tracking
        self.coherence_history = []
        self.solution_quality_history = []
        self.best_solution = None
        self.best_quality = float('inf')
        self.validity_history = []
        self.qubo_energy_history = []
    
    def _build_fields(self) -> Tuple[List, Dict, np.ndarray]:
        """
        Build fields for edge-based encoding.
        Each field handles a subset of edges.
        """
        fields = []
        
        for k in range(self.n_fields):
            cfg = FieldConfig(
                shape=(self.field_size, self.field_size),
                dt=0.04,
                seed=2000 + k
            )
            
            # Get edges for this field
            edge_indices = self.partition['field_edges'][k]
            
            if self.qubo_driven and len(edge_indices) > 0:
                # Extract QUBO submatrix
                sub_Q = self.qubo_encoder.Q[np.ix_(edge_indices, edge_indices)]
                field = QUBOField(cfg, sub_Q, edge_indices)
            else:
                field = Field(cfg)
            
            # Initialize field based on edge costs (prefer short edges)
            psi_init = np.random.randn(*cfg.shape) + 1j * np.random.randn(*cfg.shape)
            psi_flat = psi_init.ravel()
            
            # Inject edge cost information
            for local_idx, edge_idx in enumerate(edge_indices[:len(psi_flat)]):
                i, j = self.qubo_encoder.edge_list[edge_idx]
                dist = self.tsp.distance_matrix[i, j]
                max_dist = np.max(self.tsp.distance_matrix)
                
                # Short edges → positive amplitude (favor selection)
                # Long edges → negative amplitude (disfavor)
                amplitude = (1.0 - dist / max_dist) * 2.0 - 1.0
                psi_flat[local_idx] = amplitude + 0.3j * np.random.randn()
            
            field.psi = psi_flat.reshape(cfg.shape)
            fields.append(field)
        
        # Build boundaries
        boundaries = {}
        lambdas = np.zeros((self.n_fields, self.n_fields))
        
        for (k1, k2), (edges_k1, edges_k2) in self.partition['boundaries'].items():
            # Map edge indices to field positions
            # Use first N positions where N = min(overlap_size, field_capacity)
            max_boundary = min(len(edges_k1), len(edges_k2), self.field_size * self.field_size // 4)
            
            shared_k1 = list(range(max_boundary))
            shared_k2 = list(range(max_boundary))
            
            boundaries[(k1, k2)] = (np.array(shared_k1), np.array(shared_k2))
            lambdas[k1, k2] = 1.5  # Stronger coupling for edge consistency
            lambdas[k2, k1] = 1.5
        
        return fields, boundaries, lambdas
    
    def _extract_solution_from_fields(self) -> Tuple[Optional[np.ndarray], float, bool]:
        """
        Extract TSP solution from current field states.
        """
        # Collect edge probabilities from all fields
        edge_config = np.zeros(self.qubo_encoder.n_edges)
        
        for k in range(self.n_fields):
            edge_indices = self.partition['field_edges'][k]
            psi_flat = self.fields[k].psi.ravel()
            
            # Extract edge probabilities from field amplitudes
            for local_idx, edge_idx in enumerate(edge_indices):
                if local_idx < len(psi_flat):
                    # Use real part of field as edge probability
                    edge_config[edge_idx] = np.real(psi_flat[local_idx])
        
        # Decode to tour
        return self.qubo_encoder.decode_solution(edge_config)
    
    def solve(
        self,
        max_steps: int = 800,
        check_interval: int = 40,
        kernel: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Run optimization.
        """
        if kernel is None:
            kernel = exp_kernel(length=self.field_size**2, tau=25.0)
        
        print(f"\n{'='*70}")
        print(f"MULTI-FIELD EDGE-BASED TSP SOLVER")
        print(f"{'='*70}")
        print(f"Cities: {self.tsp.n_cities}")
        print(f"Edges: {self.qubo_encoder.n_edges}")
        print(f"Fields: {self.n_fields}")
        print(f"QUBO-driven: {self.qubo_driven}")
        print(f"{'='*70}\n")
        
        # Baseline
        nn_tour, nn_quality = self.tsp.nearest_neighbor_heuristic()
        print(f"Baseline (Nearest Neighbor): {nn_quality:.2f}\n")
        
        solution_check_steps = []
        
        for t in range(max_steps):
            # Step fields with QUBO gradient if enabled
            if self.qubo_driven:
                for field in self.fields:
                    if isinstance(field, QUBOField):
                        field.step_with_qubo_gradient(kernel=kernel)
                    else:
                        field.step(kernel=kernel)
                
                # Update engine state without re-stepping
                for i, f in enumerate(self.fields):
                    self.engine.A[i] = self.engine.tension_fn(f.psi)
                    from aletheia.core.multifield_engine import phase_coherence
                    self.engine.C[i] = phase_coherence(f.psi)
                
                # Update gates
                for i in range(self.n_fields):
                    for j in range(self.n_fields):
                        if i == j:
                            self.engine.G[i, j] = 0.0
                            continue
                        if self.engine.lambdas[i, j] == 0:
                            self.engine.G[i, j] = 0.0
                            continue
                        self.engine.G[i, j] = self.engine.gates.link(
                            self.engine.A[i], self.engine.A[j],
                            self.engine.C[i], self.engine.C[j]
                        )
                
                # Apply boundary coupling
                for (i, j), (idx_i, idx_j) in self.engine.boundaries.items():
                    lam = self.engine.lambdas[i, j]
                    gij = self.engine.G[i, j]
                    if lam > 0 and gij > 0:
                        psi_i = self.fields[i].psi.ravel()
                        psi_j = self.fields[j].psi.ravel()
                        Bi, Bj = psi_i[idx_i], psi_j[idx_j]
                        diff = (Bi - Bj)
                        psi_i[idx_i] = Bi - self.engine.eta * (2.0 * lam * gij) * diff
                        psi_j[idx_j] = Bj + self.engine.eta * (2.0 * lam * gij) * diff
                        self.fields[i].psi = psi_i.reshape(self.fields[i].psi.shape)
                        self.fields[j].psi = psi_j.reshape(self.fields[j].psi.shape)
            else:
                # Standard engine step
                diag = self.engine.step(t, kernel=kernel)
            
            # Track coherence
            self.coherence_history.append(self.engine.mean_coherence())
            
            # Check solution periodically
            if t % check_interval == 0:
                tour, quality, valid = self._extract_solution_from_fields()
                self.solution_quality_history.append(quality)
                self.validity_history.append(valid)
                solution_check_steps.append(t)
                
                # Track QUBO energy
                edge_config = np.zeros(self.qubo_encoder.n_edges)
                for k in range(self.n_fields):
                    edge_indices = self.partition['field_edges'][k]
                    psi_flat = self.fields[k].psi.ravel()
                    for local_idx, edge_idx in enumerate(edge_indices):
                        if local_idx < len(psi_flat):
                            edge_config[edge_idx] = np.real(psi_flat[local_idx])
                
                qubo_energy = edge_config @ self.qubo_encoder.Q @ edge_config
                self.qubo_energy_history.append(qubo_energy)
                
                if valid and quality < self.best_quality:
                    self.best_quality = quality
                    self.best_solution = tour
                    improvement = (nn_quality - quality) / nn_quality * 100
                    print(f"t={t:4d} | C={self.engine.mean_coherence():.3f} | "
                          f"Q={quality:.2f} | QUBO_E={qubo_energy:.1f} | Valid={valid} | "
                          f"★ BEST={self.best_quality:.2f} ({improvement:+.1f}% vs NN)")
                else:
                    status = "✓" if valid else "✗"
                    print(f"t={t:4d} | C={self.engine.mean_coherence():.3f} | "
                          f"Q={quality:.2f} | QUBO_E={qubo_energy:.1f} | {status}")
        
        # Final
        final_tour, final_quality, final_valid = self._extract_solution_from_fields()
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"Baseline (NN):     {nn_quality:.2f}")
        if self.best_quality < float('inf'):
            improvement = (nn_quality - self.best_quality) / nn_quality * 100
            print(f"Best found:        {self.best_quality:.2f} ({improvement:+.2f}%)")
        else:
            print(f"Best found:        No valid solution")
        print(f"Final:             {final_quality:.2f} (Valid: {final_valid})")
        print(f"Valid solutions:   {sum(self.validity_history)}/{len(self.validity_history)}")
        
        return {
            'best_tour': self.best_solution,
            'best_quality': self.best_quality,
            'baseline_quality': nn_quality,
            'coherence_history': np.array(self.coherence_history),
            'solution_quality_history': self.solution_quality_history,
            'solution_check_steps': solution_check_steps,
            'validity_history': self.validity_history,
            'qubo_energy_history': self.qubo_energy_history
        }
    
    def visualize_results(self, results: Dict):
        """Comprehensive visualization"""
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Row 1: Problem and solutions
        ax = fig.add_subplot(gs[0, 0])
        self._plot_tsp_with_partition(ax)
        
        ax = fig.add_subplot(gs[0, 1])
        nn_tour, nn_quality = self.tsp.nearest_neighbor_heuristic()
        self._plot_tour(ax, nn_tour, f'Baseline: {nn_quality:.2f}')
        
        ax = fig.add_subplot(gs[0, 2])
        if self.best_solution is not None:
            self._plot_tour(ax, self.best_solution, f'Best: {self.best_quality:.2f}')
        else:
            ax.text(0.5, 0.5, 'No valid solution', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Best Solution')
        
        # Row 2: Dynamics
        ax = fig.add_subplot(gs[1, :2])
        ax.plot(results['coherence_history'], 'b-', lw=2, label='Coherence')
        ax.set_xlabel('Step')
        ax.set_ylabel('Mean Coherence', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax.grid(alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(results['solution_check_steps'], results['qubo_energy_history'], 
                'r-', lw=2, label='QUBO Energy', alpha=0.7)
        ax2.set_ylabel('QUBO Energy', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_title('Coherence & QUBO Energy Evolution')
        
        ax = fig.add_subplot(gs[1, 2])
        steps = results['solution_check_steps']
        ax.plot(steps, results['solution_quality_history'], 'go-', lw=2, markersize=5)
        ax.axhline(results['baseline_quality'], color='r', linestyle='--', lw=2, label='Baseline')
        if self.best_quality < float('inf'):
            ax.axhline(self.best_quality, color='orange', linestyle=':', lw=2, label='Best')
        ax.set_xlabel('Step')
        ax.set_ylabel('Tour Length')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_title('Solution Quality')
        
        # Row 3: Field analysis
        ax = fig.add_subplot(gs[2, 0])
        im = ax.imshow(self.engine.G, cmap='hot', vmin=0, vmax=1)
        ax.set_title('Final Gate Matrix')
        ax.set_xlabel('Field j')
        ax.set_ylabel('Field i')
        plt.colorbar(im, ax=ax, label='Gate')
        
        ax = fig.add_subplot(gs[2, 1])
        # Edge selection heatmap
        edge_config = np.zeros(self.qubo_encoder.n_edges)
        for k in range(self.n_fields):
            edge_indices = self.partition['field_edges'][k]
            psi_flat = self.fields[k].psi.ravel()
            for local_idx, edge_idx in enumerate(edge_indices):
                if local_idx < len(psi_flat):
                    edge_config[edge_idx] = np.real(psi_flat[local_idx])
        
        # Plot edges colored by selection probability
        for edge_idx, (i, j) in enumerate(self.qubo_encoder.edge_list):
            prob = (edge_config[edge_idx] + 1) / 2  # Normalize to [0,1]
            coords = self.tsp.coords
            ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                   'b-', lw=0.5, alpha=prob)
        ax.scatter(self.tsp.coords[:,0], self.tsp.coords[:,1], c='red', s=50, zorder=5)
        ax.set_title('Edge Selection Probabilities')
        ax.axis('equal')
        
        ax = fig.add_subplot(gs[2, 2])
        validity_int = [1 if v else 0 for v in results['validity_history']]
        ax.plot(steps, validity_int, 'ro-', lw=2, markersize=6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Valid')
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['No', 'Yes'])
        ax.grid(alpha=0.3)
        ax.set_title('Solution Validity')
        
        # Row 4: Statistics
        ax = fig.add_subplot(gs[3, :])
        stats_text = f"""
STATISTICS:
-----------
Problem: {self.tsp.n_cities} cities, {self.qubo_encoder.n_edges} edges
Decomposition: {self.n_fields} fields
QUBO-driven: {self.qubo_driven}

Baseline (NN): {results['baseline_quality']:.2f}
Best found: {self.best_quality:.2f} ({(results['baseline_quality']-self.best_quality)/results['baseline_quality']*100:+.2f}%)
Valid solutions: {sum(results['validity_history'])}/{len(results['validity_history'])} ({sum(results['validity_history'])/len(results['validity_history'])*100:.1f}%)

Final coherence: {results['coherence_history'][-1]:.3f}
Final QUBO energy: {results['qubo_energy_history'][-1]:.2f}
        """
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, fontsize=11, 
               verticalalignment='center', family='monospace')
        ax.axis('off')
        
        plt.savefig(self.output_dir / f"tsp_edge_n{self.tsp.n_cities}_f{self.n_fields}.png",
                   dpi=150, bbox_inches='tight')
        print(f"\n→ Saved: {self.output_dir}/tsp_edge_n{self.tsp.n_cities}_f{self.n_fields}.png")
    
    def _plot_tsp_with_partition(self, ax):
        """Plot TSP instance with edge partitioning"""
        coords = self.tsp.coords
        
        # Plot all cities
        ax.scatter(coords[:, 0], coords[:, 1], c='black', s=100, zorder=5, 
                  edgecolors='white', linewidths=2)
        
        # Label cities
        for i, (x, y) in enumerate(coords):
            ax.text(x, y, str(i), ha='center', va='center', 
                   fontsize=8, color='white', weight='bold')
        
        # Color edges by field assignment
        colors = plt.cm.tab10(np.linspace(0, 1, self.n_fields))
        
        for k in range(self.n_fields):
            edge_indices = self.partition['field_edges'][k]
            for edge_idx in edge_indices:
                i, j = self.qubo_encoder.edge_list[edge_idx]
                ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                       color=colors[k], alpha=0.3, lw=2, label=f'Field {k}' if edge_idx == edge_indices[0] else '')
        
        # Highlight boundary edges
        for (k1, k2), (edges_k1, edges_k2) in self.partition['boundaries'].items():
            for edge_idx in edges_k1[:5]:  # Show first few boundary edges
                i, j = self.qubo_encoder.edge_list[edge_idx]
                ax.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]],
                       'r--', alpha=0.5, lw=1.5)
        
        ax.set_title('TSP Instance & Edge Partitioning')
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=8)
        ax.axis('equal')
    
    def _plot_tour(self, ax, tour, title):
        """Plot a tour"""
        if tour is None or -1 in tour:
            ax.text(0.5, 0.5, 'Invalid tour', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title(title)
            return
        
        coords = self.tsp.coords
        ax.scatter(coords[:, 0], coords[:, 1], c='blue', s=100, 
                  alpha=0.7, edgecolors='black', zorder=3, linewidths=1.5)
        
        # Draw tour edges
        for i in range(len(tour)):
            start = tour[i]
            end = tour[(i+1) % len(tour)]
            ax.plot([coords[start, 0], coords[end, 0]],
                   [coords[start, 1], coords[end, 1]],
                   'g-', lw=2.5, alpha=0.8, zorder=1)
        
        # Label start city
        start = tour[0]
        ax.scatter(coords[start, 0], coords[start, 1], 
                  c='red', s=200, marker='*', zorder=4, edgecolors='black', linewidths=1)
        
        ax.set_title(title)
        ax.axis('equal')


# ============================================================
# EXPERIMENT SUITE
# ============================================================

def experiment_edge_small():
    """Small TSP with edge encoding"""
    print("\n" + "="*70)
    print("EXPERIMENT: Edge-Based Small TSP")
    print("="*70)
    
    from aletheia.experiments.qubo_test_multi_field import TSPInstance
    
    tsp = TSPInstance.random_euclidean(n_cities=10, seed=100)
    
    solver = MultiFieldTSPSolver_EdgeBased(
        tsp_instance=tsp,
        n_fields=3,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.03),
        field_size=14,
        qubo_driven=True,
        output_dir="tsp_edge_experiments/exp1_small"
    )
    
    results = solver.solve(max_steps=600, check_interval=30)
    solver.visualize_results(results)
    
    return results


def experiment_edge_clustered():
    """Clustered TSP - perfect for decomposition"""
    print("\n" + "="*70)
    print("EXPERIMENT: Edge-Based Clustered TSP")
    print("="*70)
    
    from aletheia.experiments.qubo_test_multi_field import TSPInstance
    
    tsp = TSPInstance.clustered(n_clusters=4, cities_per_cluster=3, 
                                cluster_spread=8.0, cluster_separation=40.0, seed=101)
    
    solver = MultiFieldTSPSolver_EdgeBased(
        tsp_instance=tsp,
        n_fields=4,
        gate_params=GateParams(alpha=6.0, beta=6.0, floor=0.02),
        field_size=16,
        qubo_driven=True,
        output_dir="tsp_edge_experiments/exp2_clustered"
    )
    
    results = solver.solve(max_steps=800, check_interval=40)
    solver.visualize_results(results)
    
    return results


def experiment_edge_vs_position():
    """Direct comparison: edge vs position encoding"""
    print("\n" + "="*70)
    print("EXPERIMENT: Edge vs Position Encoding Comparison")
    print("="*70)
    
    from aletheia.experiments.qubo_test_multi_field import TSPInstance
    
    tsp = TSPInstance.random_euclidean(n_cities=12, seed=102)
    
    results = {}
    
    # Edge-based (single field)
    print("\n--- Edge-Based Encoding (N=1) ---")
    solver_edge = MultiFieldTSPSolver_EdgeBased(
        tsp_instance=tsp,
        n_fields=1,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.05),
        field_size=16,
        qubo_driven=True,
        output_dir="tsp_edge_experiments/exp3_comparison/edge_single"
    )
    results['edge_single'] = solver_edge.solve(max_steps=600, check_interval=30)
    solver_edge.visualize_results(results['edge_single'])
    
    # Edge-based (multi-field)
    print("\n--- Edge-Based Encoding (N=3) ---")
    solver_edge_multi = MultiFieldTSPSolver_EdgeBased(
        tsp_instance=tsp,
        n_fields=3,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.05),
        field_size=16,
        qubo_driven=True,
        output_dir="tsp_edge_experiments/exp3_comparison/edge_multi3"
    )
    results['edge_multi3'] = solver_edge_multi.solve(max_steps=600, check_interval=30)
    solver_edge_multi.visualize_results(results['edge_multi3'])
    
    # Comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Coherence
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['coherence_history'], label=name.upper(), lw=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Coherence')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Coherence Evolution')
    
    # Solution quality
    ax = axes[0, 1]
    baseline = results['edge_single']['baseline_quality']
    for name, res in results.items():
        steps = res['solution_check_steps']
        ax.plot(steps, res['solution_quality_history'], 'o-', label=name.upper(), lw=2, markersize=4)
    ax.axhline(baseline, color='r', linestyle='--', lw=2, label='Baseline')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tour Length')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Solution Quality')
    
    # QUBO energy
    ax = axes[1, 0]
    for name, res in results.items():
        steps = res['solution_check_steps']
        ax.plot(steps, res['qubo_energy_history'], 'o-', label=name.upper(), lw=2, markersize=4)
    ax.set_xlabel('Step')
    ax.set_ylabel('QUBO Energy')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('QUBO Energy Minimization')
    
    # Summary
    ax = axes[1, 1]
    names = list(results.keys())
    best_qualities = [results[name]['best_quality'] for name in names]
    valid_counts = [sum(results[name]['validity_history']) for name in names]
    total_checks = len(results[names[0]]['validity_history'])
    
    x = np.arange(len(names))
    width = 0.35
    
    ax2 = ax.twinx()
    bars1 = ax.bar(x - width/2, best_qualities, width, label='Best Quality', color='blue', alpha=0.7)
    bars2 = ax2.bar(x + width/2, valid_counts, width, label='Valid Solutions', color='green', alpha=0.7)
    
    ax.axhline(baseline, color='r', linestyle='--', lw=2, alpha=0.5)
    ax.set_ylabel('Tour Length', color='blue')
    ax2.set_ylabel('Valid Solutions', color='green')
    ax.set_xlabel('Configuration')
    ax.set_xticks(x)
    ax.set_xticklabels([n.upper() for n in names], rotation=15)
    ax.set_title('Final Performance')
    ax.grid(alpha=0.3, axis='y')
    
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig("tsp_edge_experiments/exp3_comparison/comparison_summary.png", dpi=150)
    
    # Print summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"{'Configuration':<20} {'Best':<12} {'Improvement':<15} {'Valid %'}")
    print("-"*70)
    print(f"{'Baseline (NN)':<20} {baseline:<12.2f} {'-':<15} {'-'}")
    for name in names:
        best = results[name]['best_quality']
        imp = (baseline - best) / baseline * 100
        valid_pct = sum(results[name]['validity_history']) / len(results[name]['validity_history']) * 100
        print(f"{name.upper():<20} {best:<12.2f} {imp:<15.1f}% {valid_pct:.1f}%")
    
    return results


def experiment_edge_qubo_ablation():
    """Test QUBO-driven vs standard field dynamics"""
    print("\n" + "="*70)
    print("EXPERIMENT: QUBO-Driven Ablation Study")
    print("="*70)
    
    from aletheia.experiments.qubo_test_multi_field import TSPInstance
    
    tsp = TSPInstance.random_euclidean(n_cities=10, seed=103)
    
    results = {}
    
    # With QUBO gradient
    print("\n--- WITH QUBO Gradient ---")
    solver_qubo = MultiFieldTSPSolver_EdgeBased(
        tsp_instance=tsp,
        n_fields=3,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.03),
        field_size=14,
        qubo_driven=True,
        output_dir="tsp_edge_experiments/exp4_ablation/with_qubo"
    )
    results['with_qubo'] = solver_qubo.solve(max_steps=600, check_interval=30)
    solver_qubo.visualize_results(results['with_qubo'])
    
    # Without QUBO gradient
    print("\n--- WITHOUT QUBO Gradient ---")
    solver_no_qubo = MultiFieldTSPSolver_EdgeBased(
        tsp_instance=tsp,
        n_fields=3,
        gate_params=GateParams(alpha=5.0, beta=5.0, floor=0.03),
        field_size=14,
        qubo_driven=False,
        output_dir="tsp_edge_experiments/exp4_ablation/without_qubo"
    )
    results['without_qubo'] = solver_no_qubo.solve(max_steps=600, check_interval=30)
    solver_no_qubo.visualize_results(results['without_qubo'])
    
    # Comparison
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    for name, res in results.items():
        ax.plot(res['coherence_history'], label=name.replace('_', ' ').upper(), lw=2)
    ax.set_xlabel('Step')
    ax.set_ylabel('Coherence')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Coherence Evolution')
    
    ax = axes[0, 1]
    baseline = results['with_qubo']['baseline_quality']
    for name, res in results.items():
        steps = res['solution_check_steps']
        ax.plot(steps, res['solution_quality_history'], 'o-', 
               label=name.replace('_', ' ').upper(), lw=2, markersize=5)
    ax.axhline(baseline, color='r', linestyle='--', lw=2, label='Baseline')
    ax.set_xlabel('Step')
    ax.set_ylabel('Tour Length')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('Solution Quality')
    
    ax = axes[1, 0]
    for name, res in results.items():
        steps = res['solution_check_steps']
        ax.plot(steps, res['qubo_energy_history'], 'o-', 
               label=name.replace('_', ' ').upper(), lw=2, markersize=5)
    ax.set_xlabel('Step')
    ax.set_ylabel('QUBO Energy')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_title('QUBO Energy (Lower = Better)')
    
    ax = axes[1, 1]
    comparison_data = {
        'Metric': ['Best Quality', 'Valid %', 'Final Coherence', 'Final QUBO Energy'],
        'WITH QUBO': [
            results['with_qubo']['best_quality'],
            sum(results['with_qubo']['validity_history']) / len(results['with_qubo']['validity_history']) * 100,
            results['with_qubo']['coherence_history'][-1],
            results['with_qubo']['qubo_energy_history'][-1]
        ],
        'WITHOUT QUBO': [
            results['without_qubo']['best_quality'],
            sum(results['without_qubo']['validity_history']) / len(results['without_qubo']['validity_history']) * 100,
            results['without_qubo']['coherence_history'][-1],
            results['without_qubo']['qubo_energy_history'][-1]
        ]
    }
    
    table_text = f"""
ABLATION STUDY RESULTS
{'='*50}

Baseline (NN): {baseline:.2f}

WITH QUBO Gradient:
  Best quality: {results['with_qubo']['best_quality']:.2f}
  Improvement:  {(baseline - results['with_qubo']['best_quality'])/baseline*100:+.1f}%
  Valid:        {sum(results['with_qubo']['validity_history'])}/{len(results['with_qubo']['validity_history'])}

WITHOUT QUBO Gradient:
  Best quality: {results['without_qubo']['best_quality']:.2f}
  Improvement:  {(baseline - results['without_qubo']['best_quality'])/baseline*100:+.1f}%
  Valid:        {sum(results['without_qubo']['validity_history'])}/{len(results['without_qubo']['validity_history'])}

CONCLUSION:
QUBO gradient is {"ESSENTIAL" if results['with_qubo']['best_quality'] < results['without_qubo']['best_quality'] else "NOT helping"}
    """
    
    ax.text(0.1, 0.5, table_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='center', family='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig("tsp_edge_experiments/exp4_ablation/ablation_summary.png", dpi=150)
    
    print(table_text)
    
    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(" "*10 + "EDGE-BASED MULTI-FIELD TSP EXPERIMENTS")
    print("="*70)
    
    # Run experiments
    exp1 = experiment_edge_small()
    exp2 = experiment_edge_clustered()
    exp3 = experiment_edge_vs_position()
    exp4 = experiment_edge_qubo_ablation()
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*70)
    print("\nResults saved to tsp_edge_experiments/")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    
    all_results = {
        'Small (N=3)': exp1,
        'Clustered (N=4)': exp2,
        'Edge Single': exp3['edge_single'],
        'Edge Multi': exp3['edge_multi3'],
        'With QUBO': exp4['with_qubo'],
        'Without QUBO': exp4['without_qubo']
    }
    
    print(f"\n{'Experiment':<20} {'Best':<10} {'Baseline':<10} {'Improv %':<12} {'Valid %'}")
    print("-"*70)
    
    for name, res in all_results.items():
        best = res['best_quality']
        baseline = res['baseline_quality']
        imp = (baseline - best) / baseline * 100
        valid_pct = sum(res['validity_history']) / len(res['validity_history']) * 100
        
        status = "✓" if best < baseline else "✗"
        print(f"{name:<20} {best:<10.2f} {baseline:<10.2f} {imp:+<12.1f} {valid_pct:<8.1f}% {status}")
    
    print("\n" + "="*70)