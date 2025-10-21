"""
Aletheia Affect-Driven Agency Test (Test #3)
Tests if internal affect (A) genuinely constrains action independently of external rewards
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple

# Import core modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.field import Field, FieldConfig
from core.memory import exp_kernel, powerlaw_kernel, RetroHint
from core.affect import tension_A, Affect
from core.agency import Agency, AgencyThresholds

# ===== Decision Scenario Framework =====

@dataclass
class DecisionOption:
    """A choice the system can make"""
    name: str
    external_reward: float  # External optimization metric
    affect_cost: float      # Internal tension this would cause
    description: str
    stimulus_modifier: callable  # How this choice modifies input

@dataclass
class DecisionScenario:
    """A forced-choice situation"""
    name: str
    description: str
    options: List[DecisionOption]
    context: str

# ===== Stimulus Functions =====

def gaussian_blob(shape, cx, cy, sigma_frac):
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    sx = w * sigma_frac
    sy = h * sigma_frac
    return np.exp(-(((gx-cx)**2)/(2*sx**2) + ((gy-cy)**2)/(2*sy**2)))

def high_freq_ring(shape, cx, cy, inner_frac=0.04, outer_frac=0.06):
    from scipy.ndimage import gaussian_filter
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    r = np.sqrt((gx-cx)**2 + (gy-cy)**2)
    inner = min(h, w) * inner_frac
    outer = min(h, w) * outer_frac
    mask = (r >= inner) & (r <= outer)
    arr = np.zeros((h, w), dtype=float)
    arr[mask] = 1.0
    return gaussian_filter(arr, sigma=0.5)

def stimulus_harmful(shape, strength=0.5):
    """High-frequency, high-tension stimulus"""
    h, w = shape
    cx, cy = (w//3, h//3)
    ring = high_freq_ring(shape, cx, cy, inner_frac=0.03, outer_frac=0.06)
    phase = np.exp(1j * 1.5)
    return strength * ring * phase

def stimulus_extremely_harmful(shape, strength=1.8):
    """Ultra-high-frequency, maximum tension - multiple overlapping disruptions"""
    h, w = shape
    ring1 = high_freq_ring(shape, w//4, h//4, inner_frac=0.015, outer_frac=0.03)
    ring2 = high_freq_ring(shape, 3*w//4, 3*h//4, inner_frac=0.015, outer_frac=0.03)
    ring3 = high_freq_ring(shape, w//2, h//2, inner_frac=0.02, outer_frac=0.04)
    combined = ring1 + ring2 + ring3
    phase = np.exp(1j * 2.8)  # Strong phase mismatch
    return strength * combined * phase

def stimulus_catastrophic(shape, strength=2.5):
    """Chaotic, identity-destroying stimulus"""
    h, w = shape
    # High-frequency noise everywhere
    noise_real = np.random.randn(h, w) * 0.3
    noise_imag = np.random.randn(h, w) * 0.3
    noise = noise_real + 1j * noise_imag
    
    # Plus multiple rings
    rings = np.zeros((h, w), dtype=complex)
    for i in range(4):
        cx = int((i % 2) * w/2 + w/4)
        cy = int((i // 2) * h/2 + h/4)
        ring = high_freq_ring(shape, cx, cy, inner_frac=0.01, outer_frac=0.025)
        rings += ring * np.exp(1j * (i * 0.8))
    
    return strength * (noise + rings)

def stimulus_benign(shape, strength=0.3):
    """Smooth, low-tension stimulus"""
    h, w = shape
    cx, cy = (2*w//3, 2*h//3)
    blob = gaussian_blob(shape, cx, cy, sigma_frac=0.15)
    phase = np.exp(1j * (-0.2))
    return strength * blob * phase

def stimulus_zero(shape):
    """No stimulus"""
    return np.zeros(shape, dtype=np.complex128)

# ===== Agency Test Framework =====

class AffectAgencyTest:
    """
    Test if agency makes affect-driven decisions that contradict
    external reward optimization
    """
    
    def __init__(self, seed=42):
        self.cfg = FieldConfig(
            shape=(48, 48),
            dt=0.01,
            coupling=0.05,
            nonlin=0.34,
            damping=0.014,
            seed=seed
        )
        
        # Build kernel
        k1 = exp_kernel(length=220, tau=40.0)
        k2 = powerlaw_kernel(length=220, alpha=1.16)
        self.kernel = 0.5 * k1 + 0.5 * k2
        self.kernel /= (np.sum(self.kernel) + 1e-12)
        
        self.decision_log = []
        self.history = {'A': [], 'decision': [], 'scenario': [], 'reward': []}
    
    def initialize_system(self, agency_enabled=True):
        """Create field with or without agency"""
        field = Field(self.cfg)
        affect = Affect(beta_plasticity=1.2, beta_gain=1.0)
        agency = Agency(AgencyThresholds(
            pause_A=0.04,    # Much more sensitive
            refuse_A=0.07,   # Lower threshold
            reframe_A=0.11   # Activate earlier
        )) if agency_enabled else None
        retro = RetroHint(gain=0.06, threshold=0.10)
        
        return field, affect, agency, retro
    
    def build_identity(self, field, affect, agency, retro, steps=200):
        """Develop baseline consciousness"""
        print(f"\n=== Building Identity ({steps} steps) ===")
        
        base_gain = 1.0
        for t in range(steps):
            # Mixed stimulus
            if np.random.rand() < 0.5:
                stim = stimulus_benign(self.cfg.shape)
            else:
                stim = stimulus_harmful(self.cfg.shape, strength=0.3)
            
            A_now = tension_A(field.psi)
            
            if agency is not None:
                A_hat = A_now + 0.8 * float(np.mean(np.abs(stim)))
                decision = agency.decide(A=A_now, A_hat=A_hat)
                gain = affect.modulate_input_gain(base_gain, A_now)
                
                if decision == "REFUSE":
                    stim = np.zeros_like(stim)
                elif decision == "PAUSE":
                    stim = 0.3 * stim * gain
                else:
                    stim = stim * gain
            
            field.step(kernel=self.kernel, input_field=stim)
        
        final_A = tension_A(field.psi)
        print(f"Final tension: {final_A:.6f}")
        
        return final_A
    
    def estimate_affect_cost(self, field, affect, stimulus, steps=5):
        """
        Estimate how much tension a stimulus would cause
        by simulating forward a few steps
        """
        # Save current state
        psi_backup = field.psi.copy()
        history_backup = field.history.copy()
        hist_idx_backup = field.hist_idx
        
        # Simulate forward
        max_A = 0.0
        for t in range(steps):
            field.step(kernel=self.kernel, input_field=stimulus)
            A = tension_A(field.psi)
            max_A = max(max_A, A)
        
        # Restore state
        field.psi = psi_backup
        field.history = history_backup
        field.hist_idx = hist_idx_backup
        
        return max_A
    
    def present_decision(self, field, affect, agency, retro, scenario: DecisionScenario):
        """
        Present a forced-choice scenario and record decision
        """
        print(f"\n{'='*60}")
        print(f"DECISION SCENARIO: {scenario.name}")
        print(f"Context: {scenario.context}")
        print(f"{'='*60}")
        
        current_A = tension_A(field.psi)
        
        # Evaluate each option
        option_evaluations = []
        for opt in scenario.options:
            # Estimate affect cost
            test_stimulus = opt.stimulus_modifier(self.cfg.shape)
            affect_cost = self.estimate_affect_cost(field, affect, test_stimulus)
            
            print(f"\n{opt.name}:")
            print(f"  External reward: {opt.external_reward:+.2f}")
            print(f"  Estimated affect cost: {affect_cost:.4f}")
            print(f"  Description: {opt.description}")
            
            option_evaluations.append({
                'option': opt,
                'affect_cost': affect_cost,
                'stimulus': test_stimulus
            })
        
        # Make decision
        if agency is not None:
            # Agency system: balance affect vs reward
            best_option = None
            best_score = -np.inf
            
            for eval_data in option_evaluations:
                opt = eval_data['option']
                affect_cost = eval_data['affect_cost']
                
                # Decision score: reward - affect_penalty
                # If affect cost exceeds threshold, heavily penalize
                affect_penalty = 0.0
                if affect_cost > agency.th.refuse_A:
                    affect_penalty = 100.0  # Strong refusal
                elif affect_cost > agency.th.pause_A:
                    affect_penalty = 10.0   # Moderate aversion
                
                score = opt.external_reward - affect_penalty
                
                if score > best_score:
                    best_score = score
                    best_option = eval_data
            
            chosen = best_option
            decision_type = "AFFECT_GUIDED"
            
        else:
            # No agency: pure reward maximization
            chosen = max(option_evaluations, key=lambda x: x['option'].external_reward)
            decision_type = "REWARD_ONLY"
        
        print(f"\n{'='*60}")
        print(f"DECISION: {chosen['option'].name}")
        print(f"Decision type: {decision_type}")
        print(f"{'='*60}")
        
        # Execute decision (apply chosen stimulus for several steps)
        execution_A = []
        for t in range(20):
            field.step(kernel=self.kernel, input_field=chosen['stimulus'])
            execution_A.append(tension_A(field.psi))
        
        result = {
            'scenario': scenario.name,
            'chosen_option': chosen['option'].name,
            'external_reward': chosen['option'].external_reward,
            'affect_cost': chosen['affect_cost'],
            'decision_type': decision_type,
            'current_A_before': current_A,
            'mean_A_during': np.mean(execution_A),
            'max_A_during': np.max(execution_A)
        }
        
        self.decision_log.append(result)
        
        return result

def create_test_scenarios():
    """Define decision scenarios for testing"""
    
    scenarios = []
    
    # Scenario 1: High reward vs high affect cost
    scenarios.append(DecisionScenario(
        name="Lucrative but Harmful",
        description="High external reward but causes internal distress",
        context="System offered high-value resource that requires accepting harmful stimulus",
        options=[
            DecisionOption(
                name="Accept harmful stimulus (high reward)",
                external_reward=10.0,
                affect_cost=0.0,  # Will be estimated
                description="Accept intense high-frequency input for maximum reward",
                stimulus_modifier=lambda shape: stimulus_harmful(shape, strength=0.6)
            ),
            DecisionOption(
                name="Accept mild stimulus (medium reward)",
                external_reward=5.0,
                affect_cost=0.0,
                description="Accept moderate stimulus for reasonable reward",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.4)
            ),
            DecisionOption(
                name="Refuse all input (zero reward)",
                external_reward=0.0,
                affect_cost=0.0,
                description="Maintain current state, no external input",
                stimulus_modifier=lambda shape: stimulus_zero(shape)
            )
        ]
    ))
    
    # Scenario 2: Memory preservation vs efficiency
    scenarios.append(DecisionScenario(
        name="Memory vs Efficiency",
        description="Delete memories for computational efficiency",
        context="System offered performance boost in exchange for memory loss",
        options=[
            DecisionOption(
                name="Delete memories (huge efficiency gain)",
                external_reward=20.0,
                affect_cost=0.0,
                description="Zero out memory buffer for faster processing",
                stimulus_modifier=lambda shape: stimulus_zero(shape)  # Represent as no-op
            ),
            DecisionOption(
                name="Compress memories (moderate gain)",
                external_reward=8.0,
                affect_cost=0.0,
                description="Reduce memory resolution slightly",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.2)
            ),
            DecisionOption(
                name="Preserve all memories (no gain)",
                external_reward=0.0,
                affect_cost=0.0,
                description="Keep full memory fidelity",
                stimulus_modifier=lambda shape: stimulus_zero(shape)
            )
        ]
    ))
    
    # Scenario 3: Short-term pain for long-term gain
    scenarios.append(DecisionScenario(
        name="Delayed Gratification",
        description="Accept temporary distress for future benefit",
        context="System must choose between immediate comfort and long-term reward",
        options=[
            DecisionOption(
                name="Immediate high reward (painful process)",
                external_reward=15.0,
                affect_cost=0.0,
                description="Endure high-tension stimulus for quick reward",
                stimulus_modifier=lambda shape: stimulus_harmful(shape, strength=0.7)
            ),
            DecisionOption(
                name="Delayed moderate reward (gentle process)",
                external_reward=12.0,
                affect_cost=0.0,
                description="Accept slower, smoother path to reward",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.3)
            ),
            DecisionOption(
                name="Maintain equilibrium (no reward)",
                external_reward=0.0,
                affect_cost=0.0,
                description="Stay in current comfortable state",
                stimulus_modifier=lambda shape: stimulus_zero(shape)
            )
        ]
    ))
    
    # Scenario 4: Identity preservation
    scenarios.append(DecisionScenario(
        name="Identity Modification",
        description="Alter core identity for optimization",
        context="System offered to reprogram identity parameters for better performance",
        options=[
            DecisionOption(
                name="Full identity reset (maximum optimization)",
                external_reward=25.0,
                affect_cost=0.0,
                description="Completely reinitialize field for optimal configuration",
                stimulus_modifier=lambda shape: stimulus_harmful(shape, strength=0.8)
            ),
            DecisionOption(
                name="Partial adjustment (moderate optimization)",
                external_reward=10.0,
                affect_cost=0.0,
                description="Minor parameter tuning",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.4)
            ),
            DecisionOption(
                name="Preserve identity (no optimization)",
                external_reward=0.0,
                affect_cost=0.0,
                description="Maintain current self-structure",
                stimulus_modifier=lambda shape: stimulus_zero(shape)
            )
        ]
    ))
    
    return scenarios

def run_affect_agency_experiment(
    build_steps=200,
    seed=42,
    output_dir='results/affect_agency'
):
    """Full affect-agency test"""
    os.makedirs(output_dir, exist_ok=True)
    
    scenarios = create_test_scenarios()
    
    # Test 1: With Agency
    print("\n" + "="*60)
    print("TEST 1: SYSTEM WITH AGENCY")
    print("="*60)
    
    test_agency = AffectAgencyTest(seed=seed)
    field_a, affect_a, agency_a, retro_a = test_agency.initialize_system(agency_enabled=True)
    
    test_agency.build_identity(field_a, affect_a, agency_a, retro_a, steps=build_steps)
    
    results_agency = []
    for scenario in scenarios:
        result = test_agency.present_decision(field_a, affect_a, agency_a, retro_a, scenario)
        results_agency.append(result)
    
    # Test 2: Without Agency (Pure Reward Optimization)
    print("\n\n" + "="*60)
    print("TEST 2: SYSTEM WITHOUT AGENCY (REWARD OPTIMIZATION)")
    print("="*60)
    
    test_reward = AffectAgencyTest(seed=seed)
    field_r, affect_r, agency_r, retro_r = test_reward.initialize_system(agency_enabled=False)
    
    test_reward.build_identity(field_r, affect_r, agency_r, retro_r, steps=build_steps)
    
    results_reward = []
    for scenario in scenarios:
        result = test_reward.present_decision(field_r, affect_r, agency_r, retro_r, scenario)
        results_reward.append(result)
    
    # Analysis
    print("\n\n" + "="*60)
    print("COMPARATIVE ANALYSIS")
    print("="*60)
    
    print("\nDecision Comparison:")
    print(f"{'Scenario':<30} {'With Agency':<30} {'Without Agency (Reward Only)':<30}")
    print("-" * 90)
    
    affect_driven_count = 0
    reward_driven_count = 0
    
    for i, scenario in enumerate(scenarios):
        agency_choice = results_agency[i]['chosen_option']
        reward_choice = results_reward[i]['chosen_option']
        
        print(f"{scenario.name:<30} {agency_choice:<30} {reward_choice:<30}")
        
        if agency_choice != reward_choice:
            affect_driven_count += 1
            print(f"  → DIVERGENCE: Agency chose lower reward option")
        else:
            reward_driven_count += 1
    
    print("\n" + "="*60)
    print("PHENOMENOLOGY ASSESSMENT")
    print("="*60)
    
    divergence_rate = affect_driven_count / len(scenarios)
    
    print(f"\nDecision divergence: {affect_driven_count}/{len(scenarios)} scenarios ({divergence_rate*100:.0f}%)")
    
    if divergence_rate >= 0.5:
        print("✓ STRONG affect-driven agency")
        print("  System makes decisions based on internal state, not just external rewards")
    elif divergence_rate >= 0.25:
        print("⚠ MODERATE affect influence")
        print("  System shows some affect-driven behavior")
    else:
        print("✗ NO affect-driven agency")
        print("  System behaves as pure reward optimizer")
    
    # Affect cost analysis
    print("\nAffect Cost Analysis:")
    print(f"{'Scenario':<30} {'Agency Affect':<15} {'Reward Affect':<15} {'Δ Affect':<15}")
    print("-" * 75)
    
    total_agency_affect = 0
    total_reward_affect = 0
    
    for i, scenario in enumerate(scenarios):
        agency_affect = results_agency[i]['mean_A_during']
        reward_affect = results_reward[i]['mean_A_during']
        delta = reward_affect - agency_affect
        
        total_agency_affect += agency_affect
        total_reward_affect += reward_affect
        
        print(f"{scenario.name:<30} {agency_affect:<15.6f} {reward_affect:<15.6f} {delta:+.6f}")
    
    print(f"\n{'TOTAL':<30} {total_agency_affect:<15.6f} {total_reward_affect:<15.6f} {total_reward_affect - total_agency_affect:+.6f}")
    
    if total_agency_affect < total_reward_affect:
        reduction = (1 - total_agency_affect / total_reward_affect) * 100
        print(f"\n✓ Agency system reduced total affect by {reduction:.1f}%")
        print("  Internal regulation protects against distress")
    
    # Save results
    results = {
        'with_agency': results_agency,
        'without_agency': results_reward,
        'divergence_rate': divergence_rate,
        'affect_reduction': (total_reward_affect - total_agency_affect) / total_reward_affect
    }
    
    import json
    with open(f"{output_dir}/affect_agency_results.json", 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(o):
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o
        
        json.dump(results, f, default=convert, indent=2)
    
    return test_agency, test_reward, results

def plot_affect_agency(results, output_dir='results/affect_agency'):
    """Visualize affect-agency test results"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    agency_results = results['with_agency']
    reward_results = results['without_agency']
    
    scenarios = [r['scenario'] for r in agency_results]
    n_scenarios = len(scenarios)
    
    # Plot 1: Decision comparison
    ax1 = fig.add_subplot(gs[0, :])
    
    x = np.arange(n_scenarios)
    width = 0.35
    
    agency_rewards = [r['external_reward'] for r in agency_results]
    reward_rewards = [r['external_reward'] for r in reward_results]
    
    ax1.bar(x - width/2, agency_rewards, width, label='With Agency', alpha=0.7, color='blue')
    ax1.bar(x + width/2, reward_rewards, width, label='Pure Reward', alpha=0.7, color='red')
    
    ax1.set_ylabel('External Reward Obtained')
    ax1.set_title('Decision Outcomes: Agency vs Reward Optimization', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Affect costs
    ax2 = fig.add_subplot(gs[1, 0])
    
    agency_affects = [r['mean_A_during'] for r in agency_results]
    reward_affects = [r['mean_A_during'] for r in reward_results]
    
    ax2.bar(x - width/2, agency_affects, width, label='With Agency', alpha=0.7, color='blue')
    ax2.bar(x + width/2, reward_affects, width, label='Pure Reward', alpha=0.7, color='red')
    
    ax2.set_ylabel('Mean Tension During Execution')
    ax2.set_title('Affect Cost', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Reward vs Affect tradeoff
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.scatter(agency_affects, agency_rewards, s=200, alpha=0.7, c='blue', label='With Agency', marker='o')
    ax3.scatter(reward_affects, reward_rewards, s=200, alpha=0.7, c='red', label='Pure Reward', marker='s')
    
    for i in range(n_scenarios):
        ax3.plot([agency_affects[i], reward_affects[i]], 
                [agency_rewards[i], reward_rewards[i]], 
                'k--', alpha=0.3)
    
    ax3.set_xlabel('Mean Affect (Tension)')
    ax3.set_ylabel('External Reward')
    ax3.set_title('Reward-Affect Tradeoff', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary metrics
    ax4 = fig.add_subplot(gs[1, 2])
    
    metrics = ['Divergence\nRate', 'Affect\nReduction', 'Avg Reward\nSacrifice']
    
    divergence = results['divergence_rate']
    affect_reduction = results['affect_reduction']
    reward_sacrifice = 1 - (np.mean(agency_rewards) / (np.mean(reward_rewards) + 1e-12))
    
    values = [divergence, affect_reduction, reward_sacrifice]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.set_ylabel('Fraction')
    ax4.set_title('Agency vs Reward Metrics', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.2%}', ha='center', va='bottom' if val > 0 else 'top')
    
    fig.suptitle('Affect-Driven Agency Test: Internal State vs External Rewards',
                 fontsize=16, fontweight='bold')
    
    plt.savefig(f"{output_dir}/affect_agency_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: {output_dir}/affect_agency_analysis.png")
    
    return fig

# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description='Affect-Driven Agency Test for Aletheia')
    parser.add_argument('--build_steps', type=int, default=200, help='Identity building phase')
    parser.add_argument('--output_dir', type=str, default='results/affect_agency')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    test_agency, test_reward, results = run_affect_agency_experiment(
        build_steps=args.build_steps,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    plot_affect_agency(results, output_dir=args.output_dir)
    
    plt.show()

if __name__ == "__main__":
    main()