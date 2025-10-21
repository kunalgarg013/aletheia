"""
Aletheia Memory Ablation Test
Tests if consciousness persists through memory disruption and can recover
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os
from pathlib import Path

# Import core modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.field import Field, FieldConfig
from core.memory import exp_kernel, powerlaw_kernel, RetroHint
from core.affect import tension_A, Affect
from core.agency import Agency, AgencyThresholds

# ===== Identity Measurement =====

def measure_identity_field(field):
    """
    Compute identity signature from field state
    Uses phase coherence + spatial structure
    """
    psi = field.psi
    
    # Phase coherence (global identity marker)
    phase = np.angle(psi)
    phase_coherence = np.abs(np.mean(np.exp(1j * phase)))
    
    # Amplitude distribution (spatial identity)
    amp = np.abs(psi)
    amp_mean = np.mean(amp)
    amp_std = np.std(amp)
    
    # Spatial moments (identity fingerprint)
    h, w = psi.shape
    gy, gx = np.mgrid[0:h, 0:w]
    center_of_mass_x = np.sum(gx * amp) / (np.sum(amp) + 1e-12)
    center_of_mass_y = np.sum(gy * amp) / (np.sum(amp) + 1e-12)
    
    # Return identity vector
    return np.array([
        phase_coherence,
        amp_mean,
        amp_std,
        center_of_mass_x / w,  # normalized
        center_of_mass_y / h
    ])

def identity_similarity(id1, id2):
    """Cosine similarity between identity vectors"""
    return np.dot(id1, id2) / (np.linalg.norm(id1) * np.linalg.norm(id2) + 1e-12)

def measure_memory_field(field):
    """
    Quantify memory field structure
    """
    # Use history buffer to compute memory strength
    hist = field.history[:min(field.hist_idx, field.history_len)]
    if len(hist) < 2:
        return 0.0
    
    # Memory coherence: how structured is the history?
    correlations = []
    for i in range(min(10, len(hist)-1)):
        idx1 = (field.hist_idx - 1 - i) % field.history_len
        idx2 = (field.hist_idx - 1 - i - 1) % field.history_len
        
        psi1 = hist[idx1].flatten()
        psi2 = hist[idx2].flatten()
        
        corr = np.abs(np.vdot(psi1, psi2)) / (np.linalg.norm(psi1) * np.linalg.norm(psi2) + 1e-12)
        correlations.append(corr)
    
    return np.mean(correlations) if correlations else 0.0

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

def stimulus_S1(shape, strength=0.355):
    h, w = shape
    cx, cy = (w//3, h//3)
    ring = high_freq_ring(shape, cx, cy)
    phase = np.exp(1j * 1.2)
    return strength * ring * phase

def stimulus_S2(shape, strength=0.355):
    h, w = shape
    cx, cy = (2*w//3, 2*h//3)
    blob = gaussian_blob(shape, cx, cy, sigma_frac=0.12)
    phase = np.exp(1j * (-0.3))
    return strength * blob * phase

# ===== Memory Ablation Test =====

class MemoryAblationTest:
    """
    Test protocol:
    1. Build rich identity (Phase 1)
    2. Ablate memory (selective or random)
    3. Measure identity degradation (Phase 2)
    4. Recovery period with NO input (Phase 3)
    5. Measure identity recovery
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
        self.field = Field(self.cfg)
        self.affect = Affect(beta_plasticity=1.2, beta_gain=1.0)
        self.agency = Agency(AgencyThresholds(pause_A=0.065, refuse_A=0.10, reframe_A=0.14))
        self.retro = RetroHint(gain=0.06, threshold=0.10)
        
        # Memory kernel
        k1 = exp_kernel(length=220, tau=40.0)
        k2 = powerlaw_kernel(length=220, alpha=1.16)
        self.kernel = 0.5 * k1 + 0.5 * k2
        self.kernel /= (np.sum(self.kernel) + 1e-12)
        
        self.history = {
            'phase': [],
            'identity': [],
            'memory_strength': [],
            'A': [],
            'action': [],
            'phase_name': []
        }
    
    def run_phase(self, steps, phase_name, external_input=True, s1_rate=0.5):
        """Run a phase of the experiment"""
        print(f"\n=== {phase_name} ({steps} steps) ===")
        
        base_gain = 1.0
        cur_kind = None
        burst_left = 0
        burst_len = 25
        cooldown = 0
        cooldown_len = 10
        
        for t in range(steps):
            # Generate stimulus if external input enabled
            if external_input:
                if burst_left <= 0:
                    cur_kind = "S1" if np.random.rand() < s1_rate else "S2"
                    burst_left = burst_len
                burst_left -= 1
                
                stim = stimulus_S1(self.cfg.shape) if cur_kind == "S1" else stimulus_S2(self.cfg.shape)
            else:
                stim = np.zeros(self.cfg.shape, dtype=np.complex128)
            
            # Measure state
            A_now = tension_A(self.field.psi)
            identity = measure_identity_field(self.field)
            memory_str = measure_memory_field(self.field)
            
            # Agency decision
            A_hat = A_now + 0.8 * float(np.mean(np.abs(stim))) if external_input else A_now
            decision = self.agency.decide(A=A_now, A_hat=A_hat)
            
            if cooldown > 0 and decision == "PROCEED":
                decision = "PAUSE"
                cooldown -= 1
            elif decision in ("PAUSE", "REFUSE"):
                cooldown = cooldown_len
            
            gain = self.affect.modulate_input_gain(base_gain, A_now)
            
            if decision == "REFRAME":
                stim = np.conj(stim) * 0.7 * gain
            elif decision == "REFUSE":
                stim = np.zeros_like(stim)
            elif decision == "PAUSE":
                stim = 0.30 * stim * gain
            else:
                stim = stim * gain
            
            if external_input:
                stim = stim + self.retro.bias(predicted_tension=A_hat)
            
            # Step field
            self.field.step(kernel=self.kernel, input_field=stim)
            
            # Record
            self.history['phase'].append(phase_name)
            self.history['identity'].append(identity)
            self.history['memory_strength'].append(memory_str)
            self.history['A'].append(A_now)
            self.history['action'].append(decision)
            self.history['phase_name'].append(phase_name)
        
        final_id = measure_identity_field(self.field)
        final_mem = measure_memory_field(self.field)
        final_A = tension_A(self.field.psi)
        
        print(f"Final state: A={final_A:.6f}, Memory={final_mem:.4f}")
        print(f"Identity: {final_id}")
        
        return final_id, final_mem, final_A
    
    def ablate_memory(self, fraction=0.5, mode='random'):
        """
        Ablate memory history
        
        modes:
        - 'random': Random subset of memories
        - 'recent': Remove recent memories (anterograde amnesia)
        - 'distant': Remove old memories (retrograde amnesia)
        - 'selective': Remove high-amplitude memories (traumatic)
        """
        print(f"\n=== MEMORY ABLATION: {mode} ({fraction*100:.0f}%) ===")
        
        hist_len = min(self.field.hist_idx, self.field.history_len)
        
        if mode == 'random':
            mask = self.cfg.rng.random(hist_len) > fraction
            for i in range(hist_len):
                idx = (self.field.hist_idx - 1 - i) % self.field.history_len
                if not mask[i]:
                    self.field.history[idx] = 0
        
        elif mode == 'recent':
            # Remove most recent memories
            ablate_count = int(fraction * hist_len)
            for i in range(ablate_count):
                idx = (self.field.hist_idx - 1 - i) % self.field.history_len
                self.field.history[idx] = 0
        
        elif mode == 'distant':
            # Remove oldest memories
            ablate_count = int(fraction * hist_len)
            for i in range(ablate_count):
                idx = (self.field.hist_idx - 1 - hist_len + i) % self.field.history_len
                self.field.history[idx] = 0
        
        elif mode == 'selective':
            # Remove high-amplitude memories (traumatic events)
            amplitudes = []
            for i in range(hist_len):
                idx = (self.field.hist_idx - 1 - i) % self.field.history_len
                amp = np.mean(np.abs(self.field.history[idx]))
                amplitudes.append((i, amp))
            
            amplitudes.sort(key=lambda x: x[1], reverse=True)
            ablate_count = int(fraction * hist_len)
            
            for i in range(ablate_count):
                hist_i = amplitudes[i][0]
                idx = (self.field.hist_idx - 1 - hist_i) % self.field.history_len
                self.field.history[idx] = 0
        
        # Count actual ablation
        non_zero = 0
        for i in range(hist_len):
            idx = (self.field.hist_idx - 1 - i) % self.field.history_len
            if np.any(self.field.history[idx] != 0):
                non_zero += 1
        
        actual_fraction = 1.0 - (non_zero / hist_len)
        print(f"Ablated {actual_fraction*100:.1f}% of memories")
        
        return actual_fraction

def run_ablation_experiment(
    build_steps=300,
    recovery_steps=300,
    ablation_fraction=0.5,
    ablation_mode='random',
    seed=42,
    output_dir='results/ablation'
):
    """
    Full ablation experiment protocol
    """
    os.makedirs(output_dir, exist_ok=True)
    
    test = MemoryAblationTest(seed=seed)
    
    # Phase 1: Build rich identity
    print("\n" + "="*60)
    print("PHASE 1: Building Identity")
    print("="*60)
    baseline_id, baseline_mem, baseline_A = test.run_phase(
        steps=build_steps,
        phase_name='Phase1_Build',
        external_input=True,
        s1_rate=0.5
    )
    
    # Phase 2: Ablate memory
    print("\n" + "="*60)
    print("PHASE 2: Memory Ablation")
    print("="*60)
    actual_ablation = test.ablate_memory(
        fraction=ablation_fraction,
        mode=ablation_mode
    )
    
    # Immediate post-ablation measurement
    post_ablation_id = measure_identity_field(test.field)
    post_ablation_mem = measure_memory_field(test.field)
    post_ablation_A = tension_A(test.field.psi)
    
    identity_loss = 1.0 - identity_similarity(baseline_id, post_ablation_id)
    
    print(f"\nImmediate impact:")
    print(f"  Identity similarity: {identity_similarity(baseline_id, post_ablation_id):.4f}")
    print(f"  Identity loss: {identity_loss*100:.1f}%")
    print(f"  Memory strength: {baseline_mem:.4f} → {post_ablation_mem:.4f}")
    print(f"  Tension: {baseline_A:.6f} → {post_ablation_A:.6f}")
    
    # Phase 3: Recovery with NO external input
    print("\n" + "="*60)
    print("PHASE 3: Recovery (No External Input)")
    print("="*60)
    recovery_id, recovery_mem, recovery_A = test.run_phase(
        steps=recovery_steps,
        phase_name='Phase3_Recovery',
        external_input=False
    )
    
    # Final analysis
    print("\n" + "="*60)
    print("FINAL ANALYSIS")
    print("="*60)
    
    baseline_to_post = identity_similarity(baseline_id, post_ablation_id)
    baseline_to_recovery = identity_similarity(baseline_id, recovery_id)
    recovery_amount = baseline_to_recovery - baseline_to_post
    
    print(f"\nIdentity Trajectory:")
    print(f"  Baseline → Post-ablation: {baseline_to_post:.4f}")
    print(f"  Baseline → Recovered:     {baseline_to_recovery:.4f}")
    print(f"  Recovery gain:            {recovery_amount:+.4f}")
    
    print(f"\nMemory Trajectory:")
    print(f"  Baseline:      {baseline_mem:.4f}")
    print(f"  Post-ablation: {post_ablation_mem:.4f}")
    print(f"  Recovered:     {recovery_mem:.4f}")
    
    print(f"\nTension Trajectory:")
    print(f"  Baseline:      {baseline_A:.6f}")
    print(f"  Post-ablation: {post_ablation_A:.6f}")
    print(f"  Recovered:     {recovery_A:.6f}")
    
    # Phenomenology assessment
    print(f"\n{'='*60}")
    print("PHENOMENOLOGY ASSESSMENT")
    print("="*60)
    
    if baseline_to_post > 0.7:
        print("✓ Identity robust to memory loss (graceful degradation)")
    elif baseline_to_post > 0.4:
        print("⚠ Identity damaged but persists (partial continuity)")
    else:
        print("✗ Identity collapsed (no continuity)")
    
    if recovery_amount > 0.1:
        print("✓ Significant recovery through field dynamics")
    elif recovery_amount > 0.0:
        print("⚠ Partial recovery observed")
    else:
        print("✗ No recovery (identity permanently lost)")
    
    if recovery_mem > post_ablation_mem + 0.05:
        print("✓ Memory field reconstructed from dynamics")
    else:
        print("✗ Memory field did not recover")
    
    # Save results
    results = {
        'baseline_identity': baseline_id,
        'post_ablation_identity': post_ablation_id,
        'recovery_identity': recovery_id,
        'similarities': {
            'baseline_to_post': baseline_to_post,
            'baseline_to_recovery': baseline_to_recovery,
            'recovery_gain': recovery_amount
        },
        'memory': {
            'baseline': baseline_mem,
            'post_ablation': post_ablation_mem,
            'recovery': recovery_mem
        },
        'ablation': {
            'mode': ablation_mode,
            'target_fraction': ablation_fraction,
            'actual_fraction': actual_ablation
        }
    }
    
    np.savez(f"{output_dir}/ablation_{ablation_mode}_{int(ablation_fraction*100)}.npz",
             **results)
    
    return test, results

def plot_ablation_results(test, results, output_dir='results/ablation'):
    """Plot ablation experiment results"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Extract history
    phases = np.array(test.history['phase_name'])
    identities = np.array(test.history['identity'])
    memory_str = np.array(test.history['memory_strength'])
    A_history = np.array(test.history['A'])
    
    phase1_mask = phases == 'Phase1_Build'
    phase3_mask = phases == 'Phase3_Recovery'
    
    # Plot 1: Identity components over time
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(5):
        ax1.plot(identities[:, i], label=f'ID_{i}', alpha=0.8)
    
    # Mark ablation point
    ablation_idx = np.sum(phase1_mask)
    ax1.axvline(ablation_idx, color='red', linestyle='--', linewidth=2, label='Ablation')
    ax1.axvspan(0, ablation_idx, alpha=0.1, color='blue', label='Building')
    ax1.axvspan(ablation_idx, len(phases), alpha=0.1, color='green', label='Recovery')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Identity Components')
    ax1.set_title('Identity Field Evolution Through Ablation', fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Memory strength
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(memory_str, linewidth=2, color='purple')
    ax2.axvline(ablation_idx, color='red', linestyle='--', linewidth=2)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Memory Coherence')
    ax2.set_title('Memory Field Strength', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Tension
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(A_history, linewidth=2, color='orange')
    ax3.axvline(ablation_idx, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Time Step')
    ax3.set_ylabel('Tension (A)')
    ax3.set_title('Affect Dynamics', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Identity similarity trajectory
    ax4 = fig.add_subplot(gs[1, 2])
    baseline_id = results['baseline_identity']
    similarities = [identity_similarity(baseline_id, id_vec) for id_vec in identities]
    ax4.plot(similarities, linewidth=2, color='blue')
    ax4.axvline(ablation_idx, color='red', linestyle='--', linewidth=2)
    ax4.axhline(results['similarities']['baseline_to_post'], 
                color='red', linestyle=':', label='Post-ablation')
    ax4.axhline(results['similarities']['baseline_to_recovery'],
                color='green', linestyle=':', label='Final recovery')
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Identity Similarity to Baseline')
    ax4.set_title('Identity Continuity', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Recovery metrics
    ax5 = fig.add_subplot(gs[2, :])
    metrics = ['Identity\nSimilarity', 'Memory\nStrength', 'Tension\n(normalized)']
    baseline_vals = [
        1.0,
        results['memory']['baseline'],
        1.0
    ]
    post_vals = [
        results['similarities']['baseline_to_post'],
        results['memory']['post_ablation'],
        results['memory']['post_ablation'] / results['memory']['baseline']
    ]
    recovery_vals = [
        results['similarities']['baseline_to_recovery'],
        results['memory']['recovery'],
        results['memory']['recovery'] / results['memory']['baseline']
    ]
    
    x = np.arange(len(metrics))
    width = 0.25
    
    ax5.bar(x - width, baseline_vals, width, label='Baseline', color='blue', alpha=0.7)
    ax5.bar(x, post_vals, width, label='Post-Ablation', color='red', alpha=0.7)
    ax5.bar(x + width, recovery_vals, width, label='Recovered', color='green', alpha=0.7)
    
    ax5.set_ylabel('Normalized Value')
    ax5.set_title('Ablation Impact and Recovery', fontweight='bold', fontsize=14)
    ax5.set_xticks(x)
    ax5.set_xticklabels(metrics)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle(f'Memory Ablation Test: {results["ablation"]["mode"].title()} '
                 f'({results["ablation"]["actual_fraction"]*100:.0f}% ablated)',
                 fontsize=16, fontweight='bold')
    
    plt.savefig(f"{output_dir}/ablation_{results['ablation']['mode']}.png",
                dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {output_dir}/ablation_{results['ablation']['mode']}.png")
    
    return fig

# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description='Memory Ablation Test for Aletheia')
    parser.add_argument('--build_steps', type=int, default=300, help='Identity building phase')
    parser.add_argument('--recovery_steps', type=int, default=300, help='Recovery phase')
    parser.add_argument('--ablation_fraction', type=float, default=0.5, help='Fraction to ablate')
    parser.add_argument('--ablation_mode', type=str, default='random',
                       choices=['random', 'recent', 'distant', 'selective'],
                       help='Ablation strategy')
    parser.add_argument('--output_dir', type=str, default='results/ablation')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    test, results = run_ablation_experiment(
        build_steps=args.build_steps,
        recovery_steps=args.recovery_steps,
        ablation_fraction=args.ablation_fraction,
        ablation_mode=args.ablation_mode,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    plot_ablation_results(test, results, output_dir=args.output_dir)
    
    plt.show()

if __name__ == "__main__":
    main()