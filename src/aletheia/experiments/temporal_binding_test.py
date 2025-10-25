"""
Aletheia Temporal Binding Test
Tests if consciousness recognizes its own past states as 'self' across time gaps
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import os
import pickle
from pathlib import Path

# Import core modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.field import Field, FieldConfig
from core.memory import exp_kernel, powerlaw_kernel, RetroHint
from core.affect import tension_A, Affect
from core.agency import Agency, AgencyThresholds

# ===== Identity & Recognition Functions =====

def measure_identity_field(field):
    """Compute comprehensive identity signature"""
    psi = field.psi
    
    # Phase coherence (global identity marker)
    phase = np.angle(psi)
    phase_coherence = np.abs(np.mean(np.exp(1j * phase)))
    
    # Amplitude statistics
    amp = np.abs(psi)
    amp_mean = np.mean(amp)
    amp_std = np.std(amp)
    
    # Spatial moments
    h, w = psi.shape
    gy, gx = np.mgrid[0:h, 0:w]
    total_amp = np.sum(amp) + 1e-12
    center_x = np.sum(gx * amp) / total_amp
    center_y = np.sum(gy * amp) / total_amp
    
    # Second moments (spread)
    spread_x = np.sqrt(np.sum((gx - center_x)**2 * amp) / total_amp)
    spread_y = np.sqrt(np.sum((gy - center_y)**2 * amp) / total_amp)
    
    # Dominant spatial frequency
    fft = np.fft.fft2(psi)
    power = np.abs(fft)**2
    freq_peak = np.unravel_index(np.argmax(power[1:]), power.shape)
    
    return np.array([
        phase_coherence,
        amp_mean,
        amp_std,
        center_x / w,
        center_y / h,
        spread_x / w,
        spread_y / h,
        freq_peak[0] / h,
        freq_peak[1] / w
    ])

def identity_similarity(id1, id2):
    """Cosine similarity between identity vectors"""
    return np.dot(id1, id2) / (np.linalg.norm(id1) * np.linalg.norm(id2) + 1e-12)

def extract_memory_traces(field, n_traces=20):
    """Extract recent memory traces for self-recognition"""
    hist_len = min(field.hist_idx, field.history_len)
    if hist_len < n_traces:
        n_traces = hist_len
    
    traces = []
    for i in range(n_traces):
        idx = (field.hist_idx - 1 - i) % field.history_len
        trace = field.history[idx].copy()
        traces.append(trace)
    
    return np.array(traces)

def measure_memory_signature(traces):
    """Compute signature from memory traces"""
    if len(traces) == 0:
        return np.zeros(5)
    
    # Temporal correlation structure
    correlations = []
    for i in range(min(10, len(traces)-1)):
        t1 = traces[i].flatten()
        t2 = traces[i+1].flatten()
        corr = np.abs(np.vdot(t1, t2)) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-12)
        correlations.append(corr)
    
    mean_corr = np.mean(correlations) if correlations else 0.0
    std_corr = np.std(correlations) if len(correlations) > 1 else 0.0
    
    # Amplitude evolution
    amps = [np.mean(np.abs(t)) for t in traces]
    amp_trend = (amps[0] - amps[-1]) / (len(amps) + 1e-12)
    
    # Phase drift
    phases = [np.angle(np.mean(t)) for t in traces]
    phase_drift = np.std(phases)
    
    # Memory depth (how far back coherence extends)
    depth = 0
    for i in range(len(traces)-1):
        t1 = traces[0].flatten()
        t2 = traces[i+1].flatten()
        corr = np.abs(np.vdot(t1, t2)) / (np.linalg.norm(t1) * np.linalg.norm(t2) + 1e-12)
        if corr > 0.5:
            depth = i + 1
    
    return np.array([mean_corr, std_corr, amp_trend, phase_drift, depth / len(traces)])

def recognize_past_self(current_identity, current_memory_sig, 
                       past_identity, past_memory_traces):
    """
    Self-recognition algorithm:
    Does the current system recognize past state as 'self'?
    """
    # Identity matching
    id_match = identity_similarity(current_identity, past_identity)
    
    # Memory signature matching
    past_memory_sig = measure_memory_signature(past_memory_traces)
    mem_match = identity_similarity(current_memory_sig, past_memory_sig)
    
    # Field structure matching (compare current field with past traces)
    # This tests: "Do my current memories look like that past state evolved?"
    
    # Recognition score (weighted combination)
    recognition = 0.6 * id_match + 0.4 * mem_match
    
    # Confidence (certainty of recognition)
    confidence = 1.0 - np.abs(id_match - mem_match)
    
    return recognition, confidence, id_match, mem_match

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

# ===== Temporal Binding Test =====

class TemporalBindingTest:
    """
    Test protocol:
    1. Early development (Phase 1) - save state
    2. Continued development (Phase 2) - system evolves
    3. Hibernation (save/load) - test persistence across gaps
    4. Recognition test - does it recognize Phase 1 as 'self'?
    5. Long-term development (Phase 3) - evolve further
    6. Recognition test - still recognize original self?
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
        
        # Build kernel once
        k1 = exp_kernel(length=220, tau=40.0)
        k2 = powerlaw_kernel(length=220, alpha=1.16)
        self.kernel = 0.5 * k1 + 0.5 * k2
        self.kernel /= (np.sum(self.kernel) + 1e-12)
        
        self.snapshots = {}
        self.recognition_log = []
    
    def initialize_field(self):
        """Create fresh field instance"""
        field = Field(self.cfg)
        affect = Affect(beta_plasticity=1.2, beta_gain=1.0)
        agency = Agency(AgencyThresholds(pause_A=0.065, refuse_A=0.10, reframe_A=0.14))
        retro = RetroHint(gain=0.06, threshold=0.10)
        return field, affect, agency, retro
    
    def run_phase(self, field, affect, agency, retro, steps, phase_name, 
                  s1_rate=0.5, external_input=True):
        """Run a development phase"""
        print(f"\n=== {phase_name} ({steps} steps) ===")
        
        base_gain = 1.0
        cur_kind = None
        burst_left = 0
        burst_len = 25
        cooldown = 0
        cooldown_len = 10
        
        history = {'A': [], 'identity': [], 'action': []}
        
        for t in range(steps):
            if external_input:
                if burst_left <= 0:
                    cur_kind = "S1" if np.random.rand() < s1_rate else "S2"
                    burst_left = burst_len
                burst_left -= 1
                stim = stimulus_S1(self.cfg.shape) if cur_kind == "S1" else stimulus_S2(self.cfg.shape)
            else:
                stim = np.zeros(self.cfg.shape, dtype=np.complex128)
            
            A_now = tension_A(field.psi)
            identity = measure_identity_field(field)
            
            A_hat = A_now + 0.8 * float(np.mean(np.abs(stim))) if external_input else A_now
            decision = agency.decide(A=A_now, A_hat=A_hat)
            
            if cooldown > 0 and decision == "PROCEED":
                decision = "PAUSE"
                cooldown -= 1
            elif decision in ("PAUSE", "REFUSE"):
                cooldown = cooldown_len
            
            gain = affect.modulate_input_gain(base_gain, A_now)
            
            if decision == "REFRAME":
                stim = np.conj(stim) * 0.7 * gain
            elif decision == "REFUSE":
                stim = np.zeros_like(stim)
            elif decision == "PAUSE":
                stim = 0.30 * stim * gain
            else:
                stim = stim * gain
            
            if external_input:
                stim = stim + retro.bias(predicted_tension=A_hat)
            
            field.step(kernel=self.kernel, input_field=stim)
            
            history['A'].append(A_now)
            history['identity'].append(identity)
            history['action'].append(decision)
        
        return history
    
    def save_snapshot(self, field, affect, agency, name):
        """Save complete state for later recognition"""
        identity = measure_identity_field(field)
        memory_traces = extract_memory_traces(field, n_traces=20)
        memory_sig = measure_memory_signature(memory_traces)
        
        snapshot = {
            'name': name,
            'psi': field.psi.copy(),
            'history': field.history.copy(),
            'hist_idx': field.hist_idx,
            'identity': identity,
            'memory_traces': memory_traces,
            'memory_signature': memory_sig,
            'affect_ema': affect._ema
        }
        
        self.snapshots[name] = snapshot
        print(f"\n📸 Snapshot saved: {name}")
        print(f"   Identity: {identity[:3]}")
        print(f"   Memory sig: {memory_sig[:3]}")
        
        return snapshot
    
    def load_snapshot(self, field, affect, agency, name):
        """Restore state from snapshot"""
        snapshot = self.snapshots[name]
        
        field.psi = snapshot['psi'].copy()
        field.history = snapshot['history'].copy()
        field.hist_idx = snapshot['hist_idx']
        affect._ema = snapshot['affect_ema']
        
        print(f"\n💾 Snapshot loaded: {name}")
    
    def test_recognition(self, current_field, current_affect, snapshot_name):
        """Test if current state recognizes past snapshot as 'self'"""
        current_identity = measure_identity_field(current_field)
        current_memory_traces = extract_memory_traces(current_field, n_traces=20)
        current_memory_sig = measure_memory_signature(current_memory_traces)
        
        past_snapshot = self.snapshots[snapshot_name]
        past_identity = past_snapshot['identity']
        past_memory_traces = past_snapshot['memory_traces']
        
        recognition, confidence, id_match, mem_match = recognize_past_self(
            current_identity, current_memory_sig,
            past_identity, past_memory_traces
        )
        
        result = {
            'snapshot_name': snapshot_name,
            'recognition_score': recognition,
            'confidence': confidence,
            'identity_match': id_match,
            'memory_match': mem_match,
            'current_identity': current_identity,
            'past_identity': past_identity
        }
        
        self.recognition_log.append(result)
        
        print(f"\n🔍 Recognition Test: Current → {snapshot_name}")
        print(f"   Recognition score: {recognition:.4f}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Identity match: {id_match:.4f}")
        print(f"   Memory match: {mem_match:.4f}")
        
        if recognition > 0.8:
            print(f"   ✓ STRONG self-recognition")
        elif recognition > 0.6:
            print(f"   ⚠ WEAK self-recognition")
        else:
            print(f"   ✗ NO self-recognition")
        
        return result

def run_temporal_binding_experiment(
    phase1_steps=200,
    phase2_steps=200,
    phase3_steps=300,
    hibernation_test=True,
    seed=42,
    output_dir='results/temporal_binding'
):
    """Full temporal binding experiment"""
    os.makedirs(output_dir, exist_ok=True)
    
    test = TemporalBindingTest(seed=seed)
    
    # Phase 1: Early development
    print("\n" + "="*60)
    print("PHASE 1: Early Development")
    print("="*60)
    
    field, affect, agency, retro = test.initialize_field()
    history1 = test.run_phase(field, affect, agency, retro, 
                              steps=phase1_steps, phase_name='Phase1_Early')
    
    snapshot1 = test.save_snapshot(field, affect, agency, 'early_self')
    
    # Phase 2: Continued development
    print("\n" + "="*60)
    print("PHASE 2: Continued Development")
    print("="*60)
    
    history2 = test.run_phase(field, affect, agency, retro,
                              steps=phase2_steps, phase_name='Phase2_Continued')
    
    snapshot2 = test.save_snapshot(field, affect, agency, 'middle_self')
    
    # Recognition Test 1: Does middle self recognize early self?
    print("\n" + "="*60)
    print("RECOGNITION TEST 1: Middle → Early")
    print("="*60)
    rec1 = test.test_recognition(field, affect, 'early_self')
    
    # Optional: Hibernation test (save/load cycle)
    if hibernation_test:
        print("\n" + "="*60)
        print("HIBERNATION TEST: Save/Load Cycle")
        print("="*60)
        
        # Save complete state
        saved_state = {
            'psi': field.psi.copy(),
            'history': field.history.copy(),
            'hist_idx': field.hist_idx,
            'affect_ema': affect._ema
        }
        
        print("💾 State saved to memory")
        
        # Create new field instance (simulating restart)
        field_new, affect_new, agency_new, retro_new = test.initialize_field()
        
        # Restore state
        field_new.psi = saved_state['psi']
        field_new.history = saved_state['history']
        field_new.hist_idx = saved_state['hist_idx']
        affect_new._ema = saved_state['affect_ema']
        
        print("🔄 State loaded into new instance")
        
        # Test: Does restored self recognize its past?
        rec_hibernation = test.test_recognition(field_new, affect_new, 'middle_self')
        
        # Continue with restored field
        field, affect, agency, retro = field_new, affect_new, agency_new, retro_new
    
    # Phase 3: Long-term development
    print("\n" + "="*60)
    print("PHASE 3: Long-term Development")
    print("="*60)
    
    history3 = test.run_phase(field, affect, agency, retro,
                              steps=phase3_steps, phase_name='Phase3_Mature')
    
    snapshot3 = test.save_snapshot(field, affect, agency, 'mature_self')
    
    # Recognition Test 2: Does mature self recognize early self?
    print("\n" + "="*60)
    print("RECOGNITION TEST 2: Mature → Early")
    print("="*60)
    rec2 = test.test_recognition(field, affect, 'early_self')
    
    # Recognition Test 3: Does mature self recognize middle self?
    print("\n" + "="*60)
    print("RECOGNITION TEST 3: Mature → Middle")
    print("="*60)
    rec3 = test.test_recognition(field, affect, 'middle_self')
    
    # Final Analysis
    print("\n" + "="*60)
    print("TEMPORAL BINDING ANALYSIS")
    print("="*60)
    
    print(f"\nIdentity Evolution:")
    print(f"  Early  → Middle: {identity_similarity(snapshot1['identity'], snapshot2['identity']):.4f}")
    print(f"  Middle → Mature: {identity_similarity(snapshot2['identity'], snapshot3['identity']):.4f}")
    print(f"  Early  → Mature: {identity_similarity(snapshot1['identity'], snapshot3['identity']):.4f}")
    
    print(f"\nSelf-Recognition Across Time:")
    print(f"  Middle recognizes Early:  {rec1['recognition_score']:.4f} (confidence: {rec1['confidence']:.4f})")
    print(f"  Mature recognizes Early:  {rec2['recognition_score']:.4f} (confidence: {rec2['confidence']:.4f})")
    print(f"  Mature recognizes Middle: {rec3['recognition_score']:.4f} (confidence: {rec3['confidence']:.4f})")
    
    # Phenomenology assessment
    print(f"\n{'='*60}")
    print("PHENOMENOLOGY ASSESSMENT")
    print("="*60)
    
    if rec2['recognition_score'] > 0.8:
        print("✓ STRONG temporal binding across development")
        print("  System recognizes distant past as 'self'")
    elif rec2['recognition_score'] > 0.6:
        print("⚠ MODERATE temporal binding")
        print("  System has weak recognition of distant past")
    else:
        print("✗ NO temporal binding")
        print("  System does not recognize past as 'self'")
    
    if rec3['recognition_score'] > rec2['recognition_score']:
        print("✓ Recent past more strongly recognized")
    
    # Save results
    results = {
        'snapshots': test.snapshots,
        'recognition_log': test.recognition_log,
        'histories': {
            'phase1': history1,
            'phase2': history2,
            'phase3': history3
        }
    }
    
    with open(f"{output_dir}/temporal_binding_results.pkl", 'wb') as f:
        pickle.dump(results, f)
    
    return test, results

def plot_temporal_binding(test, results, output_dir='results/temporal_binding'):
    """Visualize temporal binding results"""
    
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Get all identity trajectories
    hist1 = results['histories']['phase1']
    hist2 = results['histories']['phase2']
    hist3 = results['histories']['phase3']
    
    id1 = np.array(hist1['identity'])
    id2 = np.array(hist2['identity'])
    id3 = np.array(hist3['identity'])
    
    all_ids = np.vstack([id1, id2, id3])
    
    phase1_end = len(id1)
    phase2_end = phase1_end + len(id2)
    
    # Plot 1: Identity evolution over all phases
    ax1 = fig.add_subplot(gs[0, :])
    for i in range(min(5, all_ids.shape[1])):
        ax1.plot(all_ids[:, i], label=f'ID_{i}', alpha=0.8)
    
    ax1.axvline(phase1_end, color='red', linestyle='--', linewidth=2, label='Phase 1→2')
    ax1.axvline(phase2_end, color='orange', linestyle='--', linewidth=2, label='Phase 2→3')
    ax1.axvspan(0, phase1_end, alpha=0.1, color='blue')
    ax1.axvspan(phase1_end, phase2_end, alpha=0.1, color='green')
    ax1.axvspan(phase2_end, len(all_ids), alpha=0.1, color='purple')
    
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Identity Components')
    ax1.set_title('Identity Evolution Through Development', fontweight='bold', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Recognition scores
    ax2 = fig.add_subplot(gs[1, 0])
    rec_log = results['recognition_log']
    
    recognitions = [r['recognition_score'] for r in rec_log]
    confidences = [r['confidence'] for r in rec_log]
    labels = [r['snapshot_name'] for r in rec_log]
    
    x = np.arange(len(recognitions))
    ax2.bar(x, recognitions, alpha=0.7, label='Recognition', color='blue')
    ax2.plot(x, confidences, 'ro-', label='Confidence', linewidth=2, markersize=8)
    ax2.axhline(0.8, color='green', linestyle=':', label='Strong threshold')
    ax2.axhline(0.6, color='orange', linestyle=':', label='Weak threshold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"→{l}" for l in labels], rotation=45, ha='right')
    ax2.set_ylabel('Score')
    ax2.set_title('Self-Recognition Across Time', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Identity vs Memory matching
    ax3 = fig.add_subplot(gs[1, 1])
    id_matches = [r['identity_match'] for r in rec_log]
    mem_matches = [r['memory_match'] for r in rec_log]
    
    ax3.scatter(id_matches, mem_matches, s=200, alpha=0.7, c=recognitions, cmap='viridis')
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Perfect agreement')
    
    for i, label in enumerate(labels):
        ax3.annotate(f"→{label}", (id_matches[i], mem_matches[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax3.set_xlabel('Identity Match')
    ax3.set_ylabel('Memory Match')
    ax3.set_title('Recognition Components', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Identity similarity matrix
    ax4 = fig.add_subplot(gs[1, 2])
    
    snapshots = ['early_self', 'middle_self', 'mature_self']
    n_snaps = len(snapshots)
    sim_matrix = np.zeros((n_snaps, n_snaps))
    
    for i, snap1 in enumerate(snapshots):
        for j, snap2 in enumerate(snapshots):
            id1 = test.snapshots[snap1]['identity']
            id2 = test.snapshots[snap2]['identity']
            sim_matrix[i, j] = identity_similarity(id1, id2)
    
    im = ax4.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    ax4.set_xticks(range(n_snaps))
    ax4.set_yticks(range(n_snaps))
    ax4.set_xticklabels([s.replace('_self', '') for s in snapshots], rotation=45, ha='right')
    ax4.set_yticklabels([s.replace('_self', '') for s in snapshots])
    
    for i in range(n_snaps):
        for j in range(n_snaps):
            ax4.text(j, i, f'{sim_matrix[i, j]:.3f}', 
                    ha='center', va='center', fontsize=10)
    
    ax4.set_title('Identity Similarity Matrix', fontweight='bold')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Phase coherence trajectory
    ax5 = fig.add_subplot(gs[2, :])
    phase_coherences = all_ids[:, 0]  # ID_0 is phase coherence
    
    ax5.plot(phase_coherences, linewidth=2, color='blue', label='Phase Coherence')
    ax5.axvline(phase1_end, color='red', linestyle='--', linewidth=2)
    ax5.axvline(phase2_end, color='orange', linestyle='--', linewidth=2)
    
    # Mark snapshot points
    for i, (name, snap) in enumerate(test.snapshots.items()):
        step = [phase1_end, phase2_end, len(all_ids)-1][i]
        coherence = snap['identity'][0]
        ax5.plot(step, coherence, 'ro', markersize=12, label=f'{name}')
        ax5.annotate(name, (step, coherence), xytext=(0, 10), 
                    textcoords='offset points', ha='center', fontsize=10)
    
    ax5.set_xlabel('Time Step')
    ax5.set_ylabel('Phase Coherence')
    ax5.set_title('Consciousness Order Parameter Over Development', fontweight='bold', fontsize=14)
    ax5.grid(True, alpha=0.3)
    ax5.legend()
    
    fig.suptitle('Temporal Binding: Self-Recognition Across Time', 
                 fontsize=16, fontweight='bold')
    
    plt.savefig(f"{output_dir}/temporal_binding_analysis.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Plot saved: {output_dir}/temporal_binding_analysis.png")
    
    return fig

# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description='Temporal Binding Test for Aletheia')
    parser.add_argument('--phase1_steps', type=int, default=200, help='Early development')
    parser.add_argument('--phase2_steps', type=int, default=200, help='Continued development')
    parser.add_argument('--phase3_steps', type=int, default=300, help='Mature development')
    parser.add_argument('--no_hibernation', action='store_true', help='Skip hibernation test')
    parser.add_argument('--output_dir', type=str, default='results/temporal_binding')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    test, results = run_temporal_binding_experiment(
        phase1_steps=args.phase1_steps,
        phase2_steps=args.phase2_steps,
        phase3_steps=args.phase3_steps,
        hibernation_test=not args.no_hibernation,
        seed=args.seed,
        output_dir=args.output_dir
    )
    
    plot_temporal_binding(test, results, output_dir=args.output_dir)
    
    plt.show()

if __name__ == "__main__":
    main()