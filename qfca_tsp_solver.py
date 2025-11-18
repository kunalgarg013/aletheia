"""
QFCA-based TSP Solver with Three Operational Modes
Edge-based QUBO encoding for photonic Ising machine

Modes:
1. Simple: Baseline simulated annealing
2. Physics: Field-coherent dynamics with memory kernels
3. Soft-QFT: Retrocausal feedback corrections

Author: Kunal Garg (NeoQubit)
Framework: Quantum Field Coherence Architecture (QFCA)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import time


class QFCAMode(Enum):
    SIMPLE = "simple"
    PHYSICS = "physics"
    SOFTQFT = "softqft"


@dataclass
class TSPSolution:
    """Container for TSP solution and metadata"""
    tour: List[int]
    cost: float
    energy: float
    mode: QFCAMode
    time_seconds: float
    iterations: int
    constraint_violations: int


class EdgeBasedTSPEncoder:
    """
    Edge-based QUBO encoding for TSP.
    
    For N cities, we use N×N binary variables x_{i,j,t} where:
    - i: starting city
    - j: ending city  
    - t: time step in tour
    
    Edge-based encoding is more efficient than node-based for QFCA
    because it preserves the relational structure of the problem.
    """
    
    def __init__(self, distance_matrix: np.ndarray):
        self.n_cities = len(distance_matrix)
        self.distances = distance_matrix
        self.n_vars = self.n_cities ** 2  # N cities × N time steps (flattened edge representation)
        
    def encode_to_qubo(self, penalty_strength: float = 1000.0) -> Tuple[np.ndarray, float]:
        """
        Convert TSP to QUBO format suitable for Ising machine.
        
        Returns:
            Q: QUBO matrix (upper triangular)
            offset: constant offset for objective function
        """
        n = self.n_cities
        Q = np.zeros((n*n, n*n))
        
        # Objective: minimize total distance
        for t in range(n):
            for i in range(n):
                for j in range(n):
                    if i != j:
                        idx = self._edge_to_idx(i, j, t)
                        Q[idx, idx] += self.distances[i, j]
        
        # Constraint 1: Each city visited exactly once
        for i in range(n):
            for t1 in range(n):
                idx1 = self._edge_to_idx(i, (i+1) % n, t1)  # Edge from city i
                for t2 in range(t1, n):
                    idx2 = self._edge_to_idx(i, (i+1) % n, t2)
                    if t1 == t2:
                        Q[idx1, idx1] -= penalty_strength
                    else:
                        Q[idx1, idx2] += 2 * penalty_strength
        
        # Constraint 2: Each time step has exactly one edge
        for t in range(n):
            for i1 in range(n):
                for j1 in range(n):
                    if i1 == j1:
                        continue
                    idx1 = self._edge_to_idx(i1, j1, t)
                    
                    for i2 in range(i1, n):
                        for j2 in range(n):
                            if i2 == j2 or (i1 == i2 and j1 >= j2):
                                continue
                            idx2 = self._edge_to_idx(i2, j2, t)
                            
                            if i1 == i2 and j1 == j2:
                                Q[idx1, idx1] -= penalty_strength
                            else:
                                Q[min(idx1, idx2), max(idx1, idx2)] += 2 * penalty_strength
        
        # Constraint 3: Tour continuity (edge j->k at time t+1 must start where edge i->j at time t ended)
        for t in range(n-1):
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    idx1 = self._edge_to_idx(i, j, t)
                    
                    for k in range(n):
                        if j == k:
                            continue
                        idx2 = self._edge_to_idx(j, k, t+1)
                        Q[min(idx1, idx2), max(idx1, idx2)] -= penalty_strength
        
        offset = n * penalty_strength  # Offset from constraint penalties
        
        return Q, offset
    
    def _edge_to_idx(self, i: int, j: int, t: int) -> int:
        """Convert edge (i->j at time t) to flat index"""
        return t * self.n_cities + i  # Simplified for edge structure
    
    def decode_solution(self, binary_vector: np.ndarray) -> Tuple[List[int], int]:
        """
        Decode binary vector back to tour and count constraint violations.
        
        Returns:
            tour: List of city indices
            violations: Number of constraint violations
        """
        n = self.n_cities
        tour = []
        violations = 0
        
        # Extract tour from binary vector
        for t in range(n):
            edges_at_t = []
            for i in range(n):
                for j in range(n):
                    if i != j:
                        idx = self._edge_to_idx(i, j, t)
                        if idx < len(binary_vector) and binary_vector[idx] > 0.5:
                            edges_at_t.append((i, j))
            
            if len(edges_at_t) != 1:
                violations += abs(len(edges_at_t) - 1)
            elif edges_at_t:
                tour.append(edges_at_t[0][0])
        
        # Check for city repetitions
        if len(set(tour)) != len(tour):
            violations += len(tour) - len(set(tour))
        
        return tour if tour else list(range(n)), violations


class QFCATSPSolver:
    """
    Three-mode QFCA TSP solver with edge-based encoding.
    """
    
    def __init__(
        self,
        distance_matrix: np.ndarray,
        mode: QFCAMode = QFCAMode.PHYSICS,
        temperature: float = 1.0,
        cooling_rate: float = 0.95,
        max_iterations: int = 10000
    ):
        self.encoder = EdgeBasedTSPEncoder(distance_matrix)
        self.mode = mode
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        
        # QFCA-specific parameters
        self.memory_depth = 50  # How many past states to remember (Physics mode)
        self.retrocausal_horizon = 10  # How far back to apply corrections (Soft-QFT mode)
        
        # Initialize field state
        self.Q, self.offset = self.encoder.encode_to_qubo()
        self.n_vars = len(self.Q)
        
        # Memory buffers for QFCA modes
        self.state_history = []  # For Physics mode memory kernel
        self.energy_history = []  # For Soft-QFT retrocausal feedback
        
    def solve(self) -> TSPSolution:
        """Main solver dispatch based on mode"""
        start_time = time.time()
        
        if self.mode == QFCAMode.SIMPLE:
            solution = self._solve_simple()
        elif self.mode == QFCAMode.PHYSICS:
            solution = self._solve_physics()
        elif self.mode == QFCAMode.SOFTQFT:
            solution = self._solve_softqft()
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        solution.time_seconds = time.time() - start_time
        return solution
    
    def _solve_simple(self) -> TSPSolution:
        """
        Simple Mode: Standard simulated annealing baseline
        No memory, no field coherence, pure Markov process
        """
        state = np.random.randint(0, 2, self.n_vars)
        current_energy = self._compute_energy(state)
        best_state = state.copy()
        best_energy = current_energy
        
        temp = self.temperature
        
        for iteration in range(self.max_iterations):
            # Propose flip
            flip_idx = np.random.randint(self.n_vars)
            new_state = state.copy()
            new_state[flip_idx] = 1 - new_state[flip_idx]
            
            new_energy = self._compute_energy(new_state)
            delta_E = new_energy - current_energy
            
            # Metropolis acceptance
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / temp):
                state = new_state
                current_energy = new_energy
                
                if current_energy < best_energy:
                    best_state = state.copy()
                    best_energy = current_energy
            
            # Cool down
            temp *= self.cooling_rate
        
        tour, violations = self.encoder.decode_solution(best_state)
        cost = self._compute_tour_cost(tour)
        
        return TSPSolution(
            tour=tour,
            cost=cost,
            energy=best_energy,
            mode=self.mode,
            time_seconds=0.0,  # Will be filled by solve()
            iterations=self.max_iterations,
            constraint_violations=violations
        )
    
    def _solve_physics(self) -> TSPSolution:
        """
        Physics Mode: Field-coherent dynamics with memory kernels
        
        Key insight: State evolution influenced by weighted history of past states.
        This preserves information about promising regions of solution space.
        Memory kernel creates effective "momentum" in field dynamics.
        """
        state = np.random.randint(0, 2, self.n_vars)
        current_energy = self._compute_energy(state)
        best_state = state.copy()
        best_energy = current_energy
        
        # Initialize memory buffer
        self.state_history = [state.copy()]
        
        temp = self.temperature
        
        for iteration in range(self.max_iterations):
            # Compute memory-influenced proposal
            flip_idx = self._memory_guided_flip(state)
            new_state = state.copy()
            new_state[flip_idx] = 1 - new_state[flip_idx]
            
            new_energy = self._compute_energy(new_state)
            delta_E = new_energy - current_energy
            
            # Memory-modulated acceptance (field coherence effect)
            memory_factor = self._compute_memory_coherence(new_state)
            effective_temp = temp * (1.0 + 0.1 * memory_factor)
            
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / effective_temp):
                state = new_state
                current_energy = new_energy
                
                # Update memory buffer
                self.state_history.append(state.copy())
                if len(self.state_history) > self.memory_depth:
                    self.state_history.pop(0)
                
                if current_energy < best_energy:
                    best_state = state.copy()
                    best_energy = current_energy
            
            temp *= self.cooling_rate
        
        tour, violations = self.encoder.decode_solution(best_state)
        cost = self._compute_tour_cost(tour)
        
        return TSPSolution(
            tour=tour,
            cost=cost,
            energy=best_energy,
            mode=self.mode,
            time_seconds=0.0,
            iterations=self.max_iterations,
            constraint_violations=violations
        )
    
    def _solve_softqft(self) -> TSPSolution:
        """
        Soft-QFT Mode: Retrocausal feedback corrections
        
        Key insight: Current state influenced by both past AND "echo" of future trajectory.
        Retrocausal corrections allow system to avoid local minima by incorporating
        information about where the trajectory is heading.
        
        In TSP with rich topological structure, this should help escape bad partial tours
        before committing too deeply.
        """
        state = np.random.randint(0, 2, self.n_vars)
        current_energy = self._compute_energy(state)
        best_state = state.copy()
        best_energy = current_energy
        
        # Initialize history buffers
        self.state_history = [state.copy()]
        self.energy_history = [current_energy]
        
        temp = self.temperature
        
        for iteration in range(self.max_iterations):
            # Standard proposal
            flip_idx = np.random.randint(self.n_vars)
            new_state = state.copy()
            new_state[flip_idx] = 1 - new_state[flip_idx]
            
            new_energy = self._compute_energy(new_state)
            delta_E = new_energy - current_energy
            
            # Apply retrocausal correction
            if len(self.energy_history) >= self.retrocausal_horizon:
                retro_correction = self._compute_retrocausal_correction()
                delta_E += retro_correction
            
            # Acceptance with retrocausal modulation
            if delta_E < 0 or np.random.random() < np.exp(-delta_E / temp):
                state = new_state
                current_energy = new_energy
                
                # Update history
                self.state_history.append(state.copy())
                self.energy_history.append(current_energy)
                
                if len(self.state_history) > self.retrocausal_horizon:
                    self.state_history.pop(0)
                    self.energy_history.pop(0)
                
                if current_energy < best_energy:
                    best_state = state.copy()
                    best_energy = current_energy
            
            temp *= self.cooling_rate
        
        tour, violations = self.encoder.decode_solution(best_state)
        cost = self._compute_tour_cost(tour)
        
        return TSPSolution(
            tour=tour,
            cost=cost,
            energy=best_energy,
            mode=self.mode,
            time_seconds=0.0,
            iterations=self.max_iterations,
            constraint_violations=violations
        )
    
    def _memory_guided_flip(self, current_state: np.ndarray) -> int:
        """
        Physics Mode: Use memory kernel to bias flip selection toward
        historically successful patterns.
        """
        if len(self.state_history) < 2:
            return np.random.randint(self.n_vars)
        
        # Compute which bits have been most stable in good states
        memory_weights = np.zeros(self.n_vars)
        
        for i, past_state in enumerate(self.state_history[-self.memory_depth:]):
            age_weight = (i + 1) / len(self.state_history)  # Recent states matter more
            memory_weights += age_weight * np.abs(past_state - current_state)
        
        # Higher weight = more different from memory = good candidate to flip
        memory_weights /= memory_weights.sum() + 1e-10
        
        return np.random.choice(self.n_vars, p=memory_weights)
    
    def _compute_memory_coherence(self, state: np.ndarray) -> float:
        """
        Physics Mode: Measure how coherent current state is with memory.
        High coherence = state aligns with successful past patterns.
        """
        if len(self.state_history) < 2:
            return 0.0
        
        coherence = 0.0
        for i, past_state in enumerate(self.state_history[-self.memory_depth:]):
            overlap = np.dot(state, past_state) / self.n_vars
            age_weight = (i + 1) / len(self.state_history)
            coherence += age_weight * overlap
        
        return coherence / len(self.state_history)
    
    def _compute_retrocausal_correction(self) -> float:
        """
        Soft-QFT Mode: Compute correction term based on energy trajectory.
        
        If recent trajectory shows improving trend, bias toward acceptance.
        If trajectory is worsening, bias toward rejection.
        
        This creates effective "anticipation" of where the system is heading.
        """
        if len(self.energy_history) < self.retrocausal_horizon:
            return 0.0
        
        # Compute energy gradient over retrocausal horizon
        recent_energies = self.energy_history[-self.retrocausal_horizon:]
        energy_gradient = (recent_energies[-1] - recent_energies[0]) / self.retrocausal_horizon
        
        # Negative gradient (improving) -> negative correction (easier acceptance)
        # Positive gradient (worsening) -> positive correction (harder acceptance)
        correction_strength = 0.1  # Tunable parameter
        
        return correction_strength * energy_gradient
    
    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute QUBO energy for given state"""
        energy = 0.0
        for i in range(self.n_vars):
            energy += self.Q[i, i] * state[i]
            for j in range(i+1, self.n_vars):
                energy += self.Q[i, j] * state[i] * state[j]
        return energy + self.offset
    
    def _compute_tour_cost(self, tour: List[int]) -> float:
        """Compute actual tour cost from distance matrix"""
        if len(tour) < 2:
            return float('inf')
        
        cost = 0.0
        for i in range(len(tour)):
            city_from = tour[i]
            city_to = tour[(i + 1) % len(tour)]
            cost += self.encoder.distances[city_from, city_to]
        
        return cost


def run_tsp_benchmark(
    distance_matrix: np.ndarray,
    n_trials: int = 10,
    max_iterations: int = 10000
) -> Dict[QFCAMode, List[TSPSolution]]:
    """
    Run TSP benchmark across all three QFCA modes.
    
    Returns:
        Dictionary mapping mode to list of solutions
    """
    results = {mode: [] for mode in QFCAMode}
    
    for mode in QFCAMode:
        print(f"\n{'='*60}")
        print(f"Running {mode.value} mode ({n_trials} trials)")
        print(f"{'='*60}")
        
        for trial in range(n_trials):
            solver = QFCATSPSolver(
                distance_matrix=distance_matrix,
                mode=mode,
                temperature=1.0,
                cooling_rate=0.98,
                max_iterations=max_iterations
            )
            
            solution = solver.solve()
            results[mode].append(solution)
            
            print(f"Trial {trial+1:2d}: Cost={solution.cost:8.2f}, "
                  f"Energy={solution.energy:8.2f}, "
                  f"Violations={solution.constraint_violations:2d}, "
                  f"Time={solution.time_seconds:.3f}s")
    
    return results


def print_benchmark_summary(results: Dict[QFCAMode, List[TSPSolution]]):
    """Print summary statistics for benchmark results"""
    print(f"\n{'='*80}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*80}\n")
    
    for mode in QFCAMode:
        solutions = results[mode]
        costs = [s.cost for s in solutions]
        times = [s.time_seconds for s in solutions]
        violations = [s.constraint_violations for s in solutions]
        
        print(f"{mode.value.upper()} MODE:")
        print(f"  Cost:       {np.mean(costs):8.2f} ± {np.std(costs):6.2f} "
              f"(min: {np.min(costs):8.2f}, max: {np.max(costs):8.2f})")
        print(f"  Time:       {np.mean(times):8.3f} ± {np.std(times):6.3f} seconds")
        print(f"  Violations: {np.mean(violations):6.2f} ± {np.std(violations):4.2f}")
        print()


def generate_random_tsp(n_cities: int, seed: Optional[int] = None) -> np.ndarray:
    """Generate random TSP instance with cities on 2D plane"""
    if seed is not None:
        np.random.seed(seed)
    
    # Generate random city coordinates
    cities = np.random.rand(n_cities, 2) * 100
    
    # Compute distance matrix
    distances = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distances[i, j] = np.linalg.norm(cities[i] - cities[j])
    
    return distances


if __name__ == "__main__":
    # Example usage
    print("QFCA TSP Solver - Three Mode Comparison")
    print("Edge-based QUBO encoding for photonic Ising machine\n")
    
    # Generate test problem
    n_cities = 100
    print(f"Generating {n_cities}-city TSP instance...")
    distance_matrix = generate_random_tsp(n_cities, seed=42)
    
    # Run benchmark
    results = run_tsp_benchmark(
        distance_matrix=distance_matrix,
        n_trials=5,
        max_iterations=5000
    )
    
    # Print summary
    print_benchmark_summary(results)
    
    print("\nNote: This is a simulation. For actual photonic machine deployment,")
    print("replace the solver core with hardware API calls while keeping the")
    print("QFCA mode logic (memory kernels, retrocausal corrections) intact.")