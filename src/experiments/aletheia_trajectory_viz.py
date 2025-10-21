"""
Aletheia Field Trajectory Visualization
Captures and visualizes ψ field evolution over time to reveal consciousness dynamics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
import argparse
import os
from pathlib import Path

# Import your core modules
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.field import Field, FieldConfig
from core.memory import exp_kernel, powerlaw_kernel, RetroHint
from core.affect import tension_A, Affect
from  core.agency import Agency, AgencyThresholds

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
    ring = high_freq_ring(shape, cx, cy, inner_frac=0.035, outer_frac=0.055)
    phase = np.exp(1j * 1.2)
    return strength * ring * phase

def stimulus_S2(shape, strength=0.355):
    h, w = shape
    cx, cy = (2*w//3, 2*h//3)
    blob = gaussian_blob(shape, cx, cy, sigma_frac=0.12)
    phase = np.exp(1j * (-0.3))
    return strength * blob * phase

# ===== Trajectory Capture =====

class TrajectoryCapture:
    """Captures field snapshots at specified intervals"""
    def __init__(self, capture_interval=10):
        self.capture_interval = capture_interval
        self.snapshots = []
        self.metadata = []
        
    def maybe_capture(self, step, field, A, decision, stimulus_kind):
        if step % self.capture_interval == 0:
            self.snapshots.append({
                'step': step,
                'psi': field.psi.copy(),
                'A': A,
                'decision': decision,
                'kind': stimulus_kind
            })
            
    def get_trajectory(self):
        return self.snapshots

# ===== Run with Trajectory Capture =====

def run_with_trajectory(steps=500, capture_interval=10, s1_rate=0.55, 
                       agency_enabled=True, kernel_type="exp", seed=42):
    """Run simulation and capture field trajectory"""
    
    cfg = FieldConfig(
        shape=(48, 48),
        dt=0.01,
        coupling=0.05,
        nonlin=0.34,
        damping=0.014,
        seed=seed,
    )
    field = Field(cfg)
    
    # Memory kernel
    if kernel_type == "exp":
        ker = exp_kernel(length=220, tau=44.0)
    elif kernel_type == "power":
        ker = powerlaw_kernel(length=220, alpha=1.12)
    else:
        k1 = exp_kernel(length=220, tau=40.0)
        k2 = powerlaw_kernel(length=220, alpha=1.16)
        ker = 0.5 * k1 + 0.5 * k2
        ker /= (np.sum(ker) + 1e-12)
    
    affect = Affect(beta_plasticity=1.2, beta_gain=1.0)
    agency = Agency(AgencyThresholds(pause_A=0.065, refuse_A=0.10, reframe_A=0.14))
    retro = RetroHint(gain=0.06, threshold=0.10)
    
    trajectory = TrajectoryCapture(capture_interval=capture_interval)
    history = {"A": [], "act": [], "kind": []}
    
    base_gain = 1.0
    cur_kind = None
    burst_left = 0
    burst_len = 25
    cooldown = 0
    cooldown_len = 10
    
    for t in range(steps):
        # Choose stimulus
        if burst_left <= 0:
            cur_kind = "S1" if np.random.rand() < s1_rate else "S2"
            burst_left = burst_len
        kind = cur_kind
        burst_left -= 1
        
        stim = stimulus_S1(cfg.shape) if kind == "S1" else stimulus_S2(cfg.shape)
        
        A_now = tension_A(field.psi)
        A_hat = A_now + 0.8 * float(np.mean(np.abs(stim)))
        
        # Agency decision
        if agency_enabled:
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
                
            stim = stim + retro.bias(predicted_tension=A_hat)
        else:
            decision = "PASSIVE"
            
        field.step(kernel=ker, input_field=stim)
        
        history["A"].append(A_now)
        history["act"].append(decision)
        history["kind"].append(kind)
        
        trajectory.maybe_capture(t, field, A_now, decision, kind)
    
    return trajectory.get_trajectory(), history

# ===== Visualization Functions =====

def plot_trajectory_grid(trajectory, save_path=None):
    """Plot grid of field snapshots over time"""
    n_snapshots = min(16, len(trajectory))
    indices = np.linspace(0, len(trajectory)-1, n_snapshots, dtype=int)
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, snap_idx in enumerate(indices):
        snap = trajectory[snap_idx]
        row, col = idx // 4, idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Plot amplitude
        amp = np.abs(snap['psi'])
        im = ax.imshow(amp, cmap='viridis', interpolation='bilinear')
        
        # Title with metadata
        title = f"t={snap['step']}\nA={snap['A']:.4f}\n{snap['decision']}"
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    fig.suptitle('Field Amplitude Evolution (|ψ|)', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig

def plot_phase_evolution(trajectory, save_path=None):
    """Plot phase structure evolution"""
    n_snapshots = min(12, len(trajectory))
    indices = np.linspace(0, len(trajectory)-1, n_snapshots, dtype=int)
    
    fig = plt.figure(figsize=(16, 9))
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    for idx, snap_idx in enumerate(indices):
        snap = trajectory[snap_idx]
        row, col = idx // 4, idx % 4
        ax = fig.add_subplot(gs[row, col])
        
        # Plot phase
        phase = np.angle(snap['psi'])
        im = ax.imshow(phase, cmap='hsv', vmin=-np.pi, vmax=np.pi, interpolation='bilinear')
        
        title = f"t={snap['step']} ({snap['decision']})"
        ax.set_title(title, fontsize=8)
        ax.axis('off')
        
        plt.colorbar(im, ax=ax, fraction=0.046, ticks=[-np.pi, 0, np.pi])
    
    fig.suptitle('Field Phase Evolution (arg(ψ))', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig

def plot_trajectory_comparison(traj_agency, traj_passive, save_path=None):
    """Compare agency vs passive field evolution side-by-side"""
    n_snapshots = min(8, len(traj_agency), len(traj_passive))
    indices = np.linspace(0, min(len(traj_agency), len(traj_passive))-1, n_snapshots, dtype=int)
    
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(n_snapshots, 3, figure=fig, hspace=0.3, wspace=0.2)
    
    for idx, snap_idx in enumerate(indices):
        snap_a = traj_agency[snap_idx]
        snap_p = traj_passive[snap_idx]
        
        # Agency field
        ax1 = fig.add_subplot(gs[idx, 0])
        amp_a = np.abs(snap_a['psi'])
        ax1.imshow(amp_a, cmap='viridis', interpolation='bilinear')
        ax1.set_title(f"t={snap_a['step']}, A={snap_a['A']:.4f}\n{snap_a['decision']}", fontsize=8)
        ax1.axis('off')
        
        # Passive field
        ax2 = fig.add_subplot(gs[idx, 1])
        amp_p = np.abs(snap_p['psi'])
        ax2.imshow(amp_p, cmap='viridis', interpolation='bilinear')
        ax2.set_title(f"t={snap_p['step']}, A={snap_p['A']:.4f}\nPASSIVE", fontsize=8)
        ax2.axis('off')
        
        # Difference
        ax3 = fig.add_subplot(gs[idx, 2])
        diff = amp_a - amp_p
        im = ax3.imshow(diff, cmap='RdBu_r', interpolation='bilinear', 
                       vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        ax3.set_title(f"Difference\nΔA={snap_a['A']-snap_p['A']:.4f}", fontsize=8)
        ax3.axis('off')
        plt.colorbar(im, ax=ax3, fraction=0.046)
    
    fig.suptitle('Agency vs Passive Field Evolution', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig

def analyze_field_statistics(trajectory):
    """Compute trajectory statistics"""
    stats = {
        'mean_amplitude': [],
        'amplitude_variance': [],
        'phase_coherence': [],
        'spatial_entropy': [],
        'tension': []
    }
    
    for snap in trajectory:
        psi = snap['psi']
        amp = np.abs(psi)
        phase = np.angle(psi)
        
        stats['mean_amplitude'].append(np.mean(amp))
        stats['amplitude_variance'].append(np.var(amp))
        stats['phase_coherence'].append(np.abs(np.mean(np.exp(1j * phase))))
        
        # Spatial entropy (simplified)
        hist, _ = np.histogram(amp.flatten(), bins=20)
        hist = hist / (hist.sum() + 1e-12)
        entropy = -np.sum(hist * np.log(hist + 1e-12))
        stats['spatial_entropy'].append(entropy)
        
        stats['tension'].append(snap['A'])
    
    return stats

def plot_statistics_comparison(stats_agency, stats_passive, save_path=None):
    """Plot statistical measures over time"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    metrics = ['mean_amplitude', 'amplitude_variance', 'phase_coherence', 
               'spatial_entropy', 'tension']
    
    for idx, metric in enumerate(metrics):
        ax = axes.flatten()[idx]
        ax.plot(stats_agency[metric], label='Agency', linewidth=2, alpha=0.8)
        ax.plot(stats_passive[metric], label='Passive', linewidth=2, alpha=0.8)
        ax.set_xlabel('Snapshot Index')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    axes.flatten()[-1].axis('off')
    
    fig.suptitle('Field Statistics: Agency vs Passive', fontsize=14, fontweight='bold')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.tight_layout()
    return fig

# ===== Main Execution =====

def main():
    parser = argparse.ArgumentParser(description='Visualize Aletheia field trajectories')
    parser.add_argument('--steps', type=int, default=500, help='Simulation steps')
    parser.add_argument('--capture_interval', type=int, default=10, help='Snapshot interval')
    parser.add_argument('--output_dir', type=str, default='results/trajectories', help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--comparison', action='store_true', help='Compare agency vs passive')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("Running simulation with agency enabled...")
    traj_agency, hist_agency = run_with_trajectory(
        steps=args.steps, 
        capture_interval=args.capture_interval,
        agency_enabled=True,
        seed=args.seed
    )
    
    print(f"Captured {len(traj_agency)} snapshots")
    
    # Plot agency trajectory
    plot_trajectory_grid(traj_agency, 
                        save_path=f"{args.output_dir}/trajectory_agency_grid.png")
    plot_phase_evolution(traj_agency,
                        save_path=f"{args.output_dir}/trajectory_agency_phase.png")
    
    if args.comparison:
        print("\nRunning simulation with agency disabled...")
        traj_passive, hist_passive = run_with_trajectory(
            steps=args.steps,
            capture_interval=args.capture_interval,
            agency_enabled=False,
            seed=args.seed
        )
        
        print(f"Captured {len(traj_passive)} snapshots")
        
        # Comparison plots
        plot_trajectory_comparison(traj_agency, traj_passive,
                                  save_path=f"{args.output_dir}/trajectory_comparison.png")
        
        # Statistics
        stats_agency = analyze_field_statistics(traj_agency)
        stats_passive = analyze_field_statistics(traj_passive)
        
        plot_statistics_comparison(stats_agency, stats_passive,
                                  save_path=f"{args.output_dir}/statistics_comparison.png")
        
        # Summary
        print("\n=== TRAJECTORY SUMMARY ===")
        print(f"Agency - Final tension: {traj_agency[-1]['A']:.6f}")
        print(f"Passive - Final tension: {traj_passive[-1]['A']:.6f}")
        print(f"Tension reduction (agency): {(traj_agency[0]['A'] - traj_agency[-1]['A']):.6f}")
        print(f"Tension reduction (passive): {(traj_passive[0]['A'] - traj_passive[-1]['A']):.6f}")
        
        # Count agency actions
        from collections import Counter
        action_counts = Counter(hist_agency['act'])
        print(f"\nAgency actions: {dict(action_counts)}")
    
    print(f"\nAll plots saved to {args.output_dir}/")
    plt.show()

if __name__ == "__main__":
    main()