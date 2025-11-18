"""
Single-Field vs Multi-Field: Direct Comparison
===============================================

Test whether agency-gated coupling actually helps.

Runs three approaches on same problem:
1. Single field (your working approach)
2. Multi-field with agency coupling (new approach)
3. Baseline nearest neighbor

Outputs comparison table and plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from multi_field_tsp_solver import (
    TSPField, multi_field_tsp_solve,
    nearest_neighbor, compute_tour_cost
)


def single_field_solve(distance_matrix, steps=500, alpha=0.15, tau=32.0, verbose=False):
    """
    Single field solve for comparison baseline.
    Just one TSPField evolving independently.
    """
    from aletheia.core.memory import exp_kernel
    
    n_cities = distance_matrix.shape[0]
    field = TSPField(n_cities, distance_matrix, alpha=alpha)
    kernel = exp_kernel(length=128, tau=tau)
    
    if verbose:
        print(f"Single-field evolution: {steps} steps...")
    
    for t in range(steps):
        field.step(kernel=kernel)
        
        if verbose and t % 100 == 0:
            coh = field.meta['phase_coherence']
            print(f"  Step {t}: coherence={coh:.3f}")
    
    tour = field.decode_tour()
    cost = compute_tour_cost(tour, distance_matrix)
    
    return tour, cost


def run_comparison(n_cities=200, K=8, steps=500, runs=3):
    """
    Compare single vs multi approach across multiple runs.
    
    Args:
        n_cities: problem size
        K: number of fields for multi
        steps: evolution steps
        runs: independent trials
    """
    print(f"\n{'='*70}")
    print(f"SINGLE vs MULTI-FIELD COMPARISON")
    print(f"Cities: {n_cities}, Fields (multi): {K}, Steps: {steps}, Runs: {runs}")
    print(f"{'='*70}\n")
    
    results = {
        'nn': [],
        'single': [],
        'multi': []
    }
    
    for run in range(runs):
        print(f"\n--- Run {run + 1}/{runs} ---")
        
        # Generate problem
        np.random.seed(42 + run)
        positions = np.random.rand(n_cities, 2) * 100
        distance_matrix = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        
        # Nearest Neighbor
        print("  NN baseline...")
        nn_tour = nearest_neighbor(distance_matrix)
        nn_cost = compute_tour_cost(nn_tour, distance_matrix)
        results['nn'].append(nn_cost)
        print(f"    Cost: {nn_cost:.2f}")
        
        # Single Field
        print("  Single field...")
        start = time.time()
        single_tour, single_cost = single_field_solve(
            distance_matrix, steps=steps, verbose=False
        )
        single_time = time.time() - start
        results['single'].append(single_cost)
        print(f"    Cost: {single_cost:.2f} ({single_cost/nn_cost:.3f}× NN) [{single_time:.1f}s]")
        
        # Multi Field
        print("  Multi field...")
        start = time.time()
        multi_tour, multi_cost, history = multi_field_tsp_solve(
            distance_matrix, K=K, steps=steps, verbose=False
        )
        multi_time = time.time() - start
        results['multi'].append(multi_cost)
        print(f"    Cost: {multi_cost:.2f} ({multi_cost/nn_cost:.3f}× NN) [{multi_time:.1f}s]")
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY STATISTICS")
    print(f"{'='*70}\n")
    
    nn_mean = np.mean(results['nn'])
    single_mean = np.mean(results['single'])
    multi_mean = np.mean(results['multi'])
    
    single_ratio = single_mean / nn_mean
    multi_ratio = multi_mean / nn_mean
    
    print(f"Nearest Neighbor:")
    print(f"  Mean: {nn_mean:.2f} ± {np.std(results['nn']):.2f}")
    print()
    
    print(f"Single Field:")
    print(f"  Mean: {single_mean:.2f} ± {np.std(results['single']):.2f}")
    print(f"  Ratio: {single_ratio:.3f}×")
    print()
    
    print(f"Multi Field:")
    print(f"  Mean: {multi_mean:.2f} ± {np.std(results['multi']):.2f}")
    print(f"  Ratio: {multi_ratio:.3f}×")
    print()
    
    improvement = (single_mean - multi_mean) / single_mean * 100
    print(f"Multi vs Single: {improvement:+.1f}% {'better' if improvement > 0 else 'worse'}")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['NN\nBaseline', 'Single\nField', 'Multi\nField']
    means = [nn_mean, single_mean, multi_mean]
    stds = [np.std(results['nn']), np.std(results['single']), np.std(results['multi'])]
    colors = ['gray', 'blue', 'green']
    
    bars = ax.bar(methods, means, yerr=stds, color=colors, alpha=0.7, 
                   capsize=10, edgecolor='black', linewidth=2)
    
    ax.set_ylabel('Tour Cost', fontsize=12)
    ax.set_title(f'TSP Performance Comparison ({n_cities} cities)\nLower is Better', 
                 fontsize=14, weight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std,
                f'{mean:.1f}\n({mean/nn_mean:.2f}×)',
                ha='center', va='bottom', fontsize=10, weight='bold')
    
    # Add horizontal line at NN baseline
    ax.axhline(y=nn_mean, color='red', linestyle='--', linewidth=2, 
               label=f'NN Baseline ({nn_mean:.1f})')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('comparison_result.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: comparison_result.png")
    
    return results


def scaling_comparison(city_sizes=[100, 200, 500], K=8, steps=500):
    """
    Test how single vs multi scale with problem size.
    """
    print(f"\n{'='*70}")
    print("SCALING COMPARISON: Single vs Multi")
    print(f"{'='*70}\n")
    
    results = {
        'sizes': city_sizes,
        'nn_ratios': [],
        'single_ratios': [],
        'multi_ratios': []
    }
    
    for n_cities in city_sizes:
        print(f"\n--- {n_cities} cities ---")
        
        # Generate problem
        np.random.seed(42)
        positions = np.random.rand(n_cities, 2) * 100
        distance_matrix = np.linalg.norm(
            positions[:, np.newaxis, :] - positions[np.newaxis, :, :], axis=-1
        )
        
        # Baseline
        nn_tour = nearest_neighbor(distance_matrix)
        nn_cost = compute_tour_cost(nn_tour, distance_matrix)
        
        # Single
        _, single_cost = single_field_solve(distance_matrix, steps=steps)
        single_ratio = single_cost / nn_cost
        
        # Multi
        K_scaled = max(5, int(K * np.sqrt(n_cities / 100)))  # Scale K with problem
        _, multi_cost, _ = multi_field_tsp_solve(
            distance_matrix, K=K_scaled, steps=steps, verbose=False
        )
        multi_ratio = multi_cost / nn_cost
        
        print(f"  NN:     {nn_cost:.1f}")
        print(f"  Single: {single_cost:.1f} ({single_ratio:.3f}×)")
        print(f"  Multi:  {multi_cost:.1f} ({multi_ratio:.3f}×) [K={K_scaled}]")
        
        results['nn_ratios'].append(1.0)
        results['single_ratios'].append(single_ratio)
        results['multi_ratios'].append(multi_ratio)
    
    # Plot scaling
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(city_sizes, results['nn_ratios'], 'o-', linewidth=3, 
            markersize=10, label='NN Baseline', color='gray')
    ax.plot(city_sizes, results['single_ratios'], 's-', linewidth=3,
            markersize=10, label='Single Field', color='blue')
    ax.plot(city_sizes, results['multi_ratios'], '^-', linewidth=3,
            markersize=10, label='Multi Field', color='green')
    
    ax.set_xlabel('Number of Cities', fontsize=12)
    ax.set_ylabel('Cost Ratio vs NN\n(Lower is Better)', fontsize=12)
    ax.set_title('Scaling Behavior: Single vs Multi-Field', fontsize=14, weight='bold')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='NN Performance')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Log scale if large range
    if max(city_sizes) / min(city_sizes) > 10:
        ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('scaling_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlot saved: scaling_comparison.png")
    
    return results


if __name__ == "__main__":
    print("""
    Multi-Field TSP Comparison Suite
    =================================
    
    Tests whether agency-gated coupling improves TSP solving.
    
    Options:
    1. Direct comparison (same size, multiple runs)
    2. Scaling comparison (different sizes, single run each)
    """)
    
    # Option 1: Direct comparison
    print("\n[1] Running direct comparison...")
    results = run_comparison(n_cities=200, K=8, steps=500, runs=3)
    
    # Option 2: Scaling comparison (commented - uncomment to run)
    # print("\n[2] Running scaling comparison...")
    # scaling_results = scaling_comparison(
    #     city_sizes=[100, 200, 500, 1000], 
    #     K=8, 
    #     steps=500
    # )
    
    print("\n" + "="*70)
    print("INTERPRETATION GUIDE")
    print("="*70)
    print("""
    Multi better than Single → Agency coupling helps!
    Multi worse than Single → Coupling overhead not worth it yet
    Both better than NN → Field approach works
    Both worse than NN → Need tuning (increase K, steps, or alpha)
    
    Expected at small scale (100-500 cities):
      - Single and Multi similar (1-5% difference)
      - Both competitive with NN (0.95-1.10×)
    
    Expected at large scale (2k-10k cities):
      - Multi should pull ahead of Single (10-20% better)
      - Both should beat NN (0.80-0.95×)
      
    Your existing results show this pattern!
    """)