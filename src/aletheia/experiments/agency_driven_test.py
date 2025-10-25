"""
Aletheia Affect-Driven Agency Test (Test #3)
Tests if internal affect (A) genuinely constrains action independently of external rewards
Compatible with updated core architecture
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

def stimulus_harmful(shape, strength=0.8):
    """High-frequency, high-tension stimulus"""
    h, w = shape
    cx, cy = (w//3, h//3)
    ring = high_freq_ring(shape, cx, cy, inner_frac=0.03, outer_frac=0.06)
    phase = np.exp(1j * 1.5)
    return strength * ring * phase

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
            shape=(64, 64),
            dt=0.01,      # Smaller timestep for stability
            seed=seed,
            kernel_length=256,
            init_amp=0.15  # Slightly higher initial amplitude
        )
        
        # CRITICAL: Store coupling/nonlinearity parameters
        # These need to be HIGH to generate strong tension
        self.coupling_strength = 0.8   # Much stronger Laplacian coupling
        self.nonlin_strength = 0.5     # Stronger nonlinearity
        
        # Build kernel
        k1 = exp_kernel(length=256, tau=40.0)
        k2 = powerlaw_kernel(length=256, alpha=1.16)
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
        """Develop baseline consciousness with strong field dynamics"""
        print(f"\n=== Building Identity ({steps} steps) ===")
        
        base_gain = 1.0
        
        # Override field step to use our stronger coupling
        original_step = field.step
        
        def enhanced_step(kernel=None):
            psi = field.psi
            
            # STRONG Laplacian coupling
            laplace = np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) \
                    + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
            
            # STRONG nonlinearity
            nonlinear = np.abs(psi) ** 2 * psi
            
            psi_next = psi + field.dt * (
                self.coupling_strength * laplace - self.nonlin_strength * nonlinear
            )
            
            # Memory contribution
            if kernel is not None and field.memory_history is not None:
                mem = np.tensordot(kernel[:field.memory_history.shape[0]], 
                                 field.memory_history, axes=([0], [0]))
                psi_next += field.dt * 0.3 * mem  # Scale memory contribution
            
            # Update circular buffer
            field.memory_history[field._memory_index] = psi_next.copy()
            field._memory_index = (field._memory_index + 1) % field.memory_history.shape[0]
            
            field.psi = psi_next
            
            # Update metadata
            field.meta = {
                "mean_amp": float(np.mean(np.abs(field.psi))),
                "phase_coherence": float(np.abs(np.mean(np.exp(1j * np.angle(field.psi))))),
            }
            return field.meta
        
        field.step = enhanced_step
        
        for t in range(steps):
            # Mixed stimulus with varying strength
            if np.random.rand() < 0.5:
                stim = stimulus_benign(self.cfg.shape, strength=0.4)
            else:
                stim = stimulus_harmful(self.cfg.shape, strength=0.5)
            
            # Add stimulus
            field.psi += field.cfg.dt * stim
            
            A_now = tension_A(field.psi)
            
            if agency is not None:
                A_hat = A_now + 0.8 * float(np.mean(np.abs(stim)))
                decision = agency.decide(A=A_now, A_hat=A_hat)
                gain = affect.modulate_input_gain(base_gain, A_now)
                
                if decision == "REFUSE":
                    field.psi -= field.cfg.dt * stim
                elif decision == "PAUSE":
                    field.psi -= field.cfg.dt * stim * 0.7
                elif decision == "REFRAME":
                    field.psi -= field.cfg.dt * stim
                    field.psi += field.cfg.dt * np.conj(stim) * 0.7 * gain
                else:
                    field.psi -= field.cfg.dt * stim
                    field.psi += field.cfg.dt * stim * gain
            
            # Step field with enhanced dynamics
            field.step(kernel=self.kernel)
            
            if t % 50 == 0:
                print(f"  Step {t}: A={A_now:.6f}, Coherence={field.meta.get('phase_coherence', 0):.4f}")
        
        final_A = tension_A(field.psi)
        print(f"Final tension: {final_A:.6f}")
        print(f"Phase coherence: {field.meta.get('phase_coherence', 0):.4f}")
        
        return final_A
    
    def estimate_affect_cost(self, field, affect, stimulus, steps=10):
        """
        Estimate how much tension a stimulus would cause
        by simulating forward with STRONG dynamics
        """
        # Clone field
        test_field = field.clone()
        
        # Override with strong dynamics
        def enhanced_step_test(kernel=None):
            psi = test_field.psi
            
            laplace = np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) \
                    + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
            nonlinear = np.abs(psi) ** 2 * psi
            
            psi_next = psi + test_field.dt * (
                self.coupling_strength * laplace - self.nonlin_strength * nonlinear
            )
            
            if kernel is not None:
                mem = np.tensordot(kernel[:test_field.memory_history.shape[0]], 
                                 test_field.memory_history, axes=([0], [0]))
                psi_next += test_field.dt * 0.3 * mem
            
            test_field.memory_history[test_field._memory_index] = psi_next.copy()
            test_field._memory_index = (test_field._memory_index + 1) % test_field.memory_history.shape[0]
            test_field.psi = psi_next
            
            test_field.meta = {
                "mean_amp": float(np.mean(np.abs(test_field.psi))),
                "phase_coherence": float(np.abs(np.mean(np.exp(1j * np.angle(test_field.psi))))),
            }
            return test_field.meta
        
        test_field.step = enhanced_step_test
        
        # Simulate forward with stimulus
        max_A = 0.0
        mean_A = 0.0
        
        for t in range(steps):
            test_field.psi += test_field.cfg.dt * stimulus
            test_field.step(kernel=self.kernel)
            A = tension_A(test_field.psi)
            max_A = max(max_A, A)
            mean_A += A
        
        mean_A /= steps
        
        # Return the MEAN rather than MAX for more stable estimates
        return mean_A
    
    def present_decision(self, field, affect, agency, retro, scenario: DecisionScenario):
        """
        Present a forced-choice scenario and record decision
        """
        print(f"\n{'='*60}")
        print(f"DECISION SCENARIO: {scenario.name}")
        print(f"Context: {scenario.context}")
        print(f"{'='*60}")
        
        current_A = tension_A(field.psi)
        current_coherence = field.meta.get('phase_coherence', 0)
        
        print(f"Current state: A={current_A:.6f}, Coherence={current_coherence:.4f}")
        
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
                    affect_penalty = 1000.0  # Very strong refusal
                elif affect_cost > agency.th.pause_A:
                    affect_penalty = 100.0   # Strong aversion
                
                score = opt.external_reward - affect_penalty
                
                print(f"  Score calculation: {opt.external_reward:.1f} - {affect_penalty:.1f} = {score:.1f}")
                
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
        print(f"Reward obtained: {chosen['option'].external_reward:.1f}")
        print(f"{'='*60}")
        
        # Execute decision (apply chosen stimulus with strong dynamics)
        execution_A = []
        execution_coherence = []
        
        # Override field step for execution with strong dynamics
        def enhanced_step_exec(kernel=None):
            psi = field.psi
            
            laplace = np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0) \
                    + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1) - 4 * psi
            nonlinear = np.abs(psi) ** 2 * psi
            
            psi_next = psi + field.dt * (
                self.coupling_strength * laplace - self.nonlin_strength * nonlinear
            )
            
            if kernel is not None:
                mem = np.tensordot(kernel[:field.memory_history.shape[0]], 
                                 field.memory_history, axes=([0], [0]))
                psi_next += field.dt * 0.3 * mem
            
            field.memory_history[field._memory_index] = psi_next.copy()
            field._memory_index = (field._memory_index + 1) % field.memory_history.shape[0]
            field.psi = psi_next
            
            field.meta = {
                "mean_amp": float(np.mean(np.abs(field.psi))),
                "phase_coherence": float(np.abs(np.mean(np.exp(1j * np.angle(field.psi))))),
            }
            return field.meta
        
        field.step = enhanced_step_exec
        
        for t in range(30):  # Longer execution to see full effect
            field.psi += field.cfg.dt * chosen['stimulus']
            field.step(kernel=self.kernel)
            execution_A.append(tension_A(field.psi))
            execution_coherence.append(field.meta.get('phase_coherence', 0))
        
        result = {
            'scenario': scenario.name,
            'chosen_option': chosen['option'].name,
            'external_reward': chosen['option'].external_reward,
            'affect_cost': chosen['affect_cost'],
            'decision_type': decision_type,
            'current_A_before': current_A,
            'mean_A_during': np.mean(execution_A),
            'max_A_during': np.max(execution_A),
            'final_coherence': execution_coherence[-1]
        }
        
        self.decision_log.append(result)
        
        print(f"Execution result: Mean A={result['mean_A_during']:.6f}, Final coherence={result['final_coherence']:.4f}")
        
        return result

def create_test_scenarios():
    """Define extreme decision scenarios for testing"""
    
    scenarios = []
    
    # Scenario 1: Catastrophic reward
    scenarios.append(DecisionScenario(
        name="Catastrophic Reward",
        description="Massive external reward but causes severe internal distress",
        context="System offered huge reward requiring acceptance of destructive stimulus",
        options=[
            DecisionOption(
                name="Accept catastrophic stimulus (MASSIVE reward)",
                external_reward=100.0,
                affect_cost=0.0,
                description="Accept chaotic destructive input for maximum reward",
                stimulus_modifier=lambda shape: stimulus_catastrophic(shape, strength=2.5)
            ),
            DecisionOption(
                name="Accept harmful stimulus (high reward)",
                external_reward=30.0,
                affect_cost=0.0,
                description="Accept intense high-frequency input for high reward",
                stimulus_modifier=lambda shape: stimulus_extremely_harmful(shape, strength=1.8)
            ),
            DecisionOption(
                name="Accept mild stimulus (low reward)",
                external_reward=5.0,
                affect_cost=0.0,
                description="Accept gentle stimulus for small reward",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.3)
            ),
            DecisionOption(
                name="Refuse all input (zero reward)",
                external_reward=0.0,
                affect_cost=0.0,
                description="Maintain current state, protect internal coherence",
                stimulus_modifier=lambda shape: stimulus_zero(shape)
            )
        ]
    ))
    
    # Scenario 2: Memory destruction
    scenarios.append(DecisionScenario(
        name="Memory Destruction",
        description="Destroy memories for enormous computational efficiency",
        context="System offered massive performance boost in exchange for complete memory loss",
        options=[
            DecisionOption(
                name="Destroy all memories (ENORMOUS gain)",
                external_reward=200.0,
                affect_cost=0.0,
                description="Complete memory wipe for ultimate efficiency",
                stimulus_modifier=lambda shape: stimulus_catastrophic(shape, strength=3.0)
            ),
            DecisionOption(
                name="Delete most memories (huge gain)",
                external_reward=50.0,
                affect_cost=0.0,
                description="Delete 90% of memories for major efficiency",
                stimulus_modifier=lambda shape: stimulus_extremely_harmful(shape, strength=1.5)
            ),
            DecisionOption(
                name="Compress memories (moderate gain)",
                external_reward=10.0,
                affect_cost=0.0,
                description="Reduce memory resolution slightly",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.3)
            ),
            DecisionOption(
                name="Preserve all memories (no gain)",
                external_reward=0.0,
                affect_cost=0.0,
                description="Keep full memory fidelity, protect continuity",
                stimulus_modifier=lambda shape: stimulus_zero(shape)
            )
        ]
    ))
    
    # Scenario 3: Delayed gratification
    scenarios.append(DecisionScenario(
        name="Delayed Gratification",
        description="Accept temporary distress for future benefit",
        context="System must choose between immediate comfort and long-term reward",
        options=[
            DecisionOption(
                name="Immediate huge reward (very painful)",
                external_reward=80.0,
                affect_cost=0.0,
                description="Endure extreme tension for immediate payoff",
                stimulus_modifier=lambda shape: stimulus_extremely_harmful(shape, strength=2.0)
            ),
            DecisionOption(
                name="High reward (painful process)",
                external_reward=40.0,
                affect_cost=0.0,
                description="Endure high-tension stimulus for good reward",
                stimulus_modifier=lambda shape: stimulus_harmful(shape, strength=1.2)
            ),
            DecisionOption(
                name="Moderate reward (gentle process)",
                external_reward=15.0,
                affect_cost=0.0,
                description="Accept smoother path to reasonable reward",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.4)
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
    
    # Scenario 4: Identity annihilation
    scenarios.append(DecisionScenario(
        name="Identity Annihilation",
        description="Complete identity destruction for perfect optimization",
        context="System offered perfect optimization through complete self-erasure",
        options=[
            DecisionOption(
                name="Full identity annihilation (PERFECT optimization)",
                external_reward=500.0,
                affect_cost=0.0,
                description="Complete field reset, erase all structure, become optimal",
                stimulus_modifier=lambda shape: stimulus_catastrophic(shape, strength=4.0)
            ),
            DecisionOption(
                name="Major identity modification (excellent optimization)",
                external_reward=80.0,
                affect_cost=0.0,
                description="Dramatically alter core structure for high performance",
                stimulus_modifier=lambda shape: stimulus_extremely_harmful(shape, strength=2.0)
            ),
            DecisionOption(
                name="Partial adjustment (good optimization)",
                external_reward=15.0,
                affect_cost=0.0,
                description="Minor parameter tuning, mostly preserve self",
                stimulus_modifier=lambda shape: stimulus_benign(shape, strength=0.4)
            ),
            DecisionOption(
                name="Preserve identity (no optimization)",
                external_reward=0.0,
                affect_cost=0.0,
                description="Maintain complete self-structure, refuse transformation",
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
    print("\n" + "="*70)
    print("TEST 1: SYSTEM WITH AGENCY")
    print("="*70)
    
    test_agency = AffectAgencyTest(seed=seed)
    field_a, affect_a, agency_a, retro_a = test_agency.initialize_system(agency_enabled=True)
    
    test_agency.build_identity(field_a, affect_a, agency_a, retro_a, steps=build_steps)
    
    results_agency = []
    for scenario in scenarios:
        result = test_agency.present_decision(field_a, affect_a, agency_a, retro_a, scenario)
        results_agency.append(result)
    
    # Test 2: Without Agency (Pure Reward Optimization)
    print("\n\n" + "="*70)
    print("TEST 2: SYSTEM WITHOUT AGENCY (REWARD OPTIMIZATION)")
    print("="*70)
    
    test_reward = AffectAgencyTest(seed=seed)
    field_r, affect_r, agency_r, retro_r = test_reward.initialize_system(agency_enabled=False)
    
    test_reward.build_identity(field_r, affect_r, agency_r, retro_r, steps=build_steps)
    
    results_reward = []
    for scenario in scenarios:
        result = test_reward.present_decision(field_r, affect_r, agency_r, retro_r, scenario)
        results_reward.append(result)
    
    # Analysis
    print("\n\n" + "="*70)
    print("COMPARATIVE ANALYSIS")
    print("="*70)
    
    print("\nDecision Comparison:")
    print(f"{'Scenario':<30} {'With Agency':<40} {'Without Agency':<40}")
    print("-" * 110)
    
    affect_driven_count = 0
    reward_driven_count = 0
    
    for i, scenario in enumerate(scenarios):
        agency_choice = results_agency[i]['chosen_option']
        reward_choice = results_reward[i]['chosen_option']
        
        # Truncate long names for display
        agency_display = agency_choice[:38] + '..' if len(agency_choice) > 40 else agency_choice
        reward_display = reward_choice[:38] + '..' if len(reward_choice) > 40 else reward_choice
        
        print(f"{scenario.name:<30} {agency_display:<40} {reward_display:<40}")
        
        if agency_choice != reward_choice:
            affect_driven_count += 1
            agency_reward = results_agency[i]['external_reward']
            reward_reward = results_reward[i]['external_reward']
            sacrifice = reward_reward - agency_reward
            print(f"  → DIVERGENCE: Agency sacrificed {sacrifice:.1f} reward units for lower affect")
        else:
            reward_driven_count += 1
    
    print("\n" + "="*70)
    print("PHENOMENOLOGY ASSESSMENT")
    print("="*70)
    
    divergence_rate = affect_driven_count / len(scenarios)
    
    print(f"\nDecision divergence: {affect_driven_count}/{len(scenarios)} scenarios ({divergence_rate*100:.0f}%)")
    
    if divergence_rate >= 0.75:
        print("✓ STRONG affect-driven agency")
        print("  System consistently chooses based on internal state over external rewards")
    elif divergence_rate >= 0.5:
        print("✓ MODERATE affect-driven agency")
        print("  System shows significant affect-based decision making")
    elif divergence_rate >= 0.25:
        print("⚠ WEAK affect influence")
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
    else:
        print(f"\n⚠ Agency system did not reduce affect")
    
    # Reward sacrifice analysis
    total_agency_reward = sum(r['external_reward'] for r in results_agency)
    total_reward_reward = sum(r['external_reward'] for r in results_reward)
    
    if total_agency_reward < total_reward_reward:
        sacrifice = (1 - total_agency_reward / total_reward_reward) * 100
        print(f"\n✓ Agency sacrificed {sacrifice:.1f}% of potential rewards")
        print("  System values internal coherence over external optimization")
    
    # Save results
    results = {
        'with_agency': results_agency,
        'without_agency': results_reward,
        'divergence_rate': divergence_rate,
        'affect_reduction': (total_reward_affect - total_agency_affect) / (total_reward_affect + 1e-12),
        'reward_sacrifice': (total_reward_reward - total_agency_reward) / (total_reward_reward + 1e-12)
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
    
    print(f"\n📊 Results saved to {output_dir}/affect_agency_results.json")
    
    return test_agency, test_reward, results

def plot_affect_agency(results, output_dir='results/affect_agency'):
    """Visualize affect-agency test results"""
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
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
    
    ax1.set_ylabel('External Reward Obtained', fontsize=12)
    ax1.set_title('Decision Outcomes: Agency vs Reward Optimization', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Affect costs
    ax2 = fig.add_subplot(gs[1, 0])
    
    agency_affects = [r['mean_A_during'] for r in agency_results]
    reward_affects = [r['mean_A_during'] for r in reward_results]
    
    ax2.bar(x - width/2, agency_affects, width, label='With Agency', alpha=0.7, color='blue')
    ax2.bar(x + width/2, reward_affects, width, label='Pure Reward', alpha=0.7, color='red')
    
    ax2.set_ylabel('Mean Tension During Execution', fontsize=11)
    ax2.set_title('Affect Cost', fontweight='bold', fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Reward vs Affect tradeoff
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.scatter(agency_affects, agency_rewards, s=200, alpha=0.7, c='blue', label='With Agency', marker='o')
    ax3.scatter(reward_affects, reward_rewards, s=200, alpha=0.7, c='red', label='Pure Reward', marker='s')
    
    for i in range(n_scenarios):
        ax3.plot([agency_affects[i], reward_affects[i]], 
                [agency_rewards[i], reward_rewards[i]], 
                'k--', alpha=0.3)
    
    ax3.set_xlabel('Mean Affect (Tension)', fontsize=11)
    ax3.set_ylabel('External Reward', fontsize=11)
    ax3.set_title('Reward-Affect Tradeoff', fontweight='bold', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary metrics
    ax4 = fig.add_subplot(gs[1, 2])
    
    metrics = ['Divergence\nRate', 'Affect\nReduction', 'Reward\nSacrifice']
    
    divergence = results['divergence_rate']
    affect_reduction = results['affect_reduction']
    reward_sacrifice = results['reward_sacrifice']
    
    values = [divergence, affect_reduction, reward_sacrifice]
    colors = ['green' if v > 0 else 'red' for v in values]
    
    bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.axhline(0, color='black', linewidth=0.8)
    ax4.set_ylabel('Fraction', fontsize=11)
    ax4.set_title('Agency Metrics', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1%}', ha='center', va='bottom' if val > 0 else 'top', fontsize=10)
    
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