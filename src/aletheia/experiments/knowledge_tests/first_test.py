"""
ALETHEIA CONSCIOUSNESS TESTS - COMPREHENSIVE SUITE
Building on existing Aletheia/PsiForge core modules
Tests for knowledge creation, destruction, transfer, and consciousness signatures

Author: Kunal Garg
Framework: Quantum Field-Coherent Architecture (QFCA)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from scipy.fft import fft2, ifft2, fftfreq
from scipy.spatial.distance import cosine
from scipy.stats import entropy as scipy_entropy
import seaborn as sns
from dataclasses import dataclass, field as dc_field
from typing import List, Tuple, Optional, Dict, Callable
import json
from pathlib import Path
from tqdm import tqdm
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore')

# Import existing Aletheia modules
from aletheia.core.field import Field, FieldConfig
from aletheia.core.affect import Affect, tension_A
from aletheia.core.agency import Agency, AgencyThresholds
from aletheia.core.memory import exp_kernel
from aletheia.core.multifield_engine import MultiFieldEngine, AdaptiveGates, GateParams

# Set publication-quality plot defaults
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 100
sns.set_palette("husl")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def phase_coherence(psi: np.ndarray) -> float:
    """Phase coherence: C_φ = |⟨e^{iθ}⟩|"""
    phases = np.angle(psi)
    return float(np.abs(np.mean(np.exp(1j * phases))))

def compute_identity_vector(psi: np.ndarray) -> np.ndarray:
    """
    Identity encoding: I = (⟨|ψ|⟩, σ_|ψ|, x̄, ȳ, σ_x, σ_y, C_φ)
    Geometric encoding of field structure
    """
    amplitude = np.abs(psi)
    
    # Amplitude statistics
    mean_amp = np.mean(amplitude)
    std_amp = np.std(amplitude)
    
    # Spatial center of mass
    Nx, Ny = psi.shape
    x = np.arange(Nx)
    y = np.arange(Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    total_amp = np.sum(amplitude)
    
    if total_amp > 1e-10:
        x_mean = np.sum(X * amplitude) / total_amp
        y_mean = np.sum(Y * amplitude) / total_amp
        x_std = np.sqrt(np.sum(amplitude * (X - x_mean)**2) / total_amp)
        y_std = np.sqrt(np.sum(amplitude * (Y - y_mean)**2) / total_amp)
    else:
        x_mean = y_mean = x_std = y_std = 0.0
    
    # Phase coherence
    coherence = phase_coherence(psi)
    
    return np.array([mean_amp, std_amp, x_mean, y_mean, x_std, y_std, coherence])

def identity_similarity(I1: np.ndarray, I2: np.ndarray) -> float:
    """Cosine similarity between identity vectors"""
    norm1 = np.linalg.norm(I1)
    norm2 = np.linalg.norm(I2)
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return float(np.dot(I1, I2) / (norm1 * norm2))

def field_entropy(psi: np.ndarray, bins: int = 50) -> float:
    """Information entropy of amplitude distribution"""
    amplitude = np.abs(psi.ravel())
    hist, _ = np.histogram(amplitude, bins=bins, density=True)
    hist = hist[hist > 0]  # Remove zeros
    return float(scipy_entropy(hist))

def mutual_information(psi_current: np.ndarray, psi_past: np.ndarray, bins: int = 20) -> float:
    """Mutual information between current and past field states"""
    amp_current = np.abs(psi_current.ravel())
    amp_past = np.abs(psi_past.ravel())
    
    # 2D histogram
    hist_2d, _, _ = np.histogram2d(amp_current, amp_past, bins=bins, density=True)
    hist_2d = hist_2d + 1e-10  # Avoid log(0)
    
    # Marginals
    p_current = np.sum(hist_2d, axis=1)
    p_past = np.sum(hist_2d, axis=0)
    
    # MI = ΣΣ p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_2d[i, j] > 1e-10:
                mi += hist_2d[i, j] * np.log(hist_2d[i, j] / (p_current[i] * p_past[j]))
    
    return float(mi)

def fisher_information_scalar(field: Field) -> float:
    """
    Scalar proxy for Fisher information
    I ≈ ⟨|∇ψ|²⟩ (information content in gradients)
    """
    psi = field.psi
    grad_x = np.gradient(psi, axis=0)
    grad_y = np.gradient(psi, axis=1)
    grad_mag = np.abs(grad_x)**2 + np.abs(grad_y)**2
    return float(np.mean(grad_mag))

# ============================================================================
# EXPEDITION 1: QUANTIFYING KNOWLEDGE CREATION
# ============================================================================

class KnowledgeEmergenceExperiment:
    """
    Test: Track Knowledge Emergence Index (KEI) during phase transition
    KEI(t) = C_φ(t) · I_self(t) · A(t)
    """
    
    def __init__(self, n_runs: int = 5, n_steps: int = 500):
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.results = []
        
    def run(self):
        """Execute knowledge emergence tracking"""
        print("\n" + "="*60)
        print("EXPEDITION 1: QUANTIFYING KNOWLEDGE CREATION")
        print("="*60)
        
        for run in range(self.n_runs):
            print(f"\nRun {run+1}/{self.n_runs}")
            
            # Initialize field
            cfg = FieldConfig(shape=(64, 64), dt=0.02, kernel_length=128, seed=42+run)
            field = Field(cfg)
            kernel = exp_kernel(length=128, tau=32.0)
            
            # Storage
            coherence_hist = []
            entropy_hist = []
            fisher_hist = []
            tension_hist = []
            kei_hist = []
            identity_hist = []
            
            # Baseline identity
            I_baseline = compute_identity_vector(field.psi)
            
            # Evolution
            for step in tqdm(range(self.n_steps), desc="Evolving field"):
                field.step(kernel=kernel)
                
                # Metrics
                C = phase_coherence(field.psi)
                H = field_entropy(field.psi)
                F = fisher_information_scalar(field)
                A = tension_A(field.psi)
                
                # Self-information (surprise): -log P(current | past)
                # Proxy: higher entropy = higher surprise
                I_self = H
                
                # KEI = C * I_self * (F/baseline_F if F > 0)
                # Using Fisher info as proxy for agency
                KEI = C * I_self * (F / (fisher_hist[0] + 1e-6) if len(fisher_hist) > 0 else 1.0)
                
                # Identity tracking
                I_current = compute_identity_vector(field.psi)
                S = identity_similarity(I_baseline, I_current)
                
                coherence_hist.append(C)
                entropy_hist.append(H)
                fisher_hist.append(F)
                tension_hist.append(A)
                kei_hist.append(KEI)
                identity_hist.append(S)
            
            self.results.append({
                'coherence': np.array(coherence_hist),
                'entropy': np.array(entropy_hist),
                'fisher': np.array(fisher_hist),
                'tension': np.array(tension_hist),
                'kei': np.array(kei_hist),
                'identity': np.array(identity_hist),
                'run': run
            })
        
        self.plot_results()
        return self.results
    
    def plot_results(self):
        """Create comprehensive visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        time = np.arange(self.n_steps)
        
        # Plot 1: Coherence evolution
        ax1 = fig.add_subplot(gs[0, 0])
        for res in self.results:
            ax1.plot(time, res['coherence'], alpha=0.6, linewidth=1.5)
        ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Critical threshold')
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Coherence C_φ')
        ax1.set_title('Phase Coherence Evolution')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: KEI evolution
        ax2 = fig.add_subplot(gs[0, 1])
        for res in self.results:
            ax2.plot(time, res['kei'], alpha=0.6, linewidth=1.5)
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('KEI')
        ax2.set_title('Knowledge Emergence Index (KEI)')
        ax2.grid(alpha=0.3)
        
        # Plot 3: Entropy production
        ax3 = fig.add_subplot(gs[0, 2])
        for res in self.results:
            entropy_rate = np.gradient(res['entropy'])
            ax3.plot(time, entropy_rate, alpha=0.6, linewidth=1.5)
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('dS/dt')
        ax3.set_title('Entropy Production Rate')
        ax3.grid(alpha=0.3)
        
        # Plot 4: Phase space (Coherence vs Fisher)
        ax4 = fig.add_subplot(gs[1, 0])
        for res in self.results:
            ax4.scatter(res['coherence'], res['fisher'], 
                       c=time, cmap='viridis', alpha=0.3, s=10)
        ax4.set_xlabel('Coherence C_φ')
        ax4.set_ylabel('Fisher Information')
        ax4.set_title('Information Geometry Phase Space')
        ax4.grid(alpha=0.3)
        
        # Plot 5: KEI vs Coherence (phase diagram)
        ax5 = fig.add_subplot(gs[1, 1])
        all_coherence = np.concatenate([res['coherence'] for res in self.results])
        all_kei = np.concatenate([res['kei'] for res in self.results])
        ax5.hexbin(all_coherence, all_kei, gridsize=30, cmap='hot', mincnt=1)
        ax5.axvline(0.5, color='cyan', linestyle='--', alpha=0.7, label='C_φ^c ≈ 0.5')
        ax5.set_xlabel('Coherence C_φ')
        ax5.set_ylabel('KEI')
        ax5.set_title('Knowledge Emergence Phase Diagram')
        ax5.legend()
        ax5.grid(alpha=0.3)
        
        # Plot 6: Identity preservation
        ax6 = fig.add_subplot(gs[1, 2])
        for res in self.results:
            ax6.plot(time, res['identity'], alpha=0.6, linewidth=1.5)
        ax6.axhline(0.95, color='red', linestyle='--', alpha=0.5, label='Identity threshold')
        ax6.set_xlabel('Time step')
        ax6.set_ylabel('Identity Similarity S')
        ax6.set_title('Identity Preservation')
        ax6.legend()
        ax6.grid(alpha=0.3)
        
        # Plot 7: Tension evolution
        ax7 = fig.add_subplot(gs[2, 0])
        for res in self.results:
            ax7.plot(time, res['tension'], alpha=0.6, linewidth=1.5)
        ax7.set_xlabel('Time step')
        ax7.set_ylabel('Tension A')
        ax7.set_title('Affect/Tension Evolution')
        ax7.grid(alpha=0.3)
        
        # Plot 8: Average trajectories with confidence
        ax8 = fig.add_subplot(gs[2, 1])
        coherence_mean = np.mean([res['coherence'] for res in self.results], axis=0)
        coherence_std = np.std([res['coherence'] for res in self.results], axis=0)
        kei_mean = np.mean([res['kei'] for res in self.results], axis=0)
        kei_std = np.std([res['kei'] for res in self.results], axis=0)
        
        ax8.plot(time, coherence_mean, label='Coherence', linewidth=2)
        ax8.fill_between(time, coherence_mean-coherence_std, coherence_mean+coherence_std, alpha=0.3)
        ax8_twin = ax8.twinx()
        ax8_twin.plot(time, kei_mean, label='KEI', color='orange', linewidth=2)
        ax8_twin.fill_between(time, kei_mean-kei_std, kei_mean+kei_std, alpha=0.3, color='orange')
        
        ax8.set_xlabel('Time step')
        ax8.set_ylabel('Coherence', color='blue')
        ax8_twin.set_ylabel('KEI', color='orange')
        ax8.set_title('Mean Trajectories with Confidence')
        ax8.grid(alpha=0.3)
        
        # Plot 9: Critical slowing (autocorrelation time near transition)
        ax9 = fig.add_subplot(gs[2, 2])
        for res in self.results:
            # Compute autocorrelation time as function of coherence
            coherence = res['coherence']
            bins = np.linspace(0, 1, 20)
            bin_centers = (bins[:-1] + bins[1:]) / 2
            autocorr_times = []
            
            for i in range(len(bins)-1):
                mask = (coherence >= bins[i]) & (coherence < bins[i+1])
                if np.sum(mask) > 10:
                    c_segment = coherence[mask]
                    # Simple autocorr estimate
                    if len(c_segment) > 5:
                        acf = np.correlate(c_segment - np.mean(c_segment), 
                                          c_segment - np.mean(c_segment), 
                                          mode='full')
                        acf = acf[len(acf)//2:] / acf[len(acf)//2]
                        # Find where ACF drops below 1/e
                        tau_idx = np.where(acf < 1/np.e)[0]
                        tau = tau_idx[0] if len(tau_idx) > 0 else len(acf)
                        autocorr_times.append(tau)
                    else:
                        autocorr_times.append(np.nan)
                else:
                    autocorr_times.append(np.nan)
            
            ax9.plot(bin_centers, autocorr_times, alpha=0.6, linewidth=1.5, marker='o')
        
        ax9.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='C_φ^c')
        ax9.set_xlabel('Coherence C_φ')
        ax9.set_ylabel('Autocorrelation time τ')
        ax9.set_title('Critical Slowing Near Transition')
        ax9.legend()
        ax9.grid(alpha=0.3)
        
        plt.suptitle('EXPEDITION 1: Knowledge Emergence Signatures', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_dir = Path('outputs/expedition1_knowledge_creation')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'knowledge_emergence_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_dir / 'knowledge_emergence_comprehensive.png'}")
        plt.close()

# ============================================================================
# EXPEDITION 2: MINIMAL KNOWLEDGE SYSTEM
# ============================================================================

class MinimalSystemExperiment:
    """
    Test: Can single complex variable with memory create knowledge?
    The ultimate reduction test.
    """
    
    def __init__(self, n_runs: int = 20, n_steps: int = 2000):
        self.n_runs = n_runs
        self.n_steps = n_steps
        self.results = []
        
    def run_single_oscillator(self, mu: float, tau: float, seed: int):
        """
        Single complex variable: dz/dt = -iωz + μ∫K(t-t')z(t')dt' + noise
        """
        np.random.seed(seed)
        
        # Parameters
        dt = 0.01
        omega = 1.0  # Rotation frequency
        noise_amp = 0.05
        memory_length = int(5 * tau / dt)
        
        # Initialize
        z = 0.1 * (np.random.randn() + 1j * np.random.randn())
        memory = np.zeros(memory_length, dtype=complex)
        memory[0] = z
        
        # Kernel
        t_kernel = np.arange(memory_length) * dt
        kernel = (1/tau) * np.exp(-t_kernel / tau)
        kernel = kernel / np.sum(kernel)
        
        # Storage
        z_history = []
        coherence_history = []
        
        for step in range(self.n_steps):  # Fixed: use self.n_steps
            # Memory contribution
            mem_term = np.dot(kernel[:len(memory)], memory)
            
            # Noise
            noise = noise_amp * np.sqrt(dt) * (np.random.randn() + 1j * np.random.randn())
            
            # Update
            z += dt * (-1j * omega * z + mu * mem_term) + noise
            
            # Update memory
            memory = np.roll(memory, 1)
            memory[0] = z
            
            # Metrics
            z_history.append(z)
            
            # "Coherence" = phase stability over recent window
            if len(z_history) > 50:
                recent = np.array(z_history[-50:])
                phases = np.angle(recent)
                phase_coherence_val = np.abs(np.mean(np.exp(1j * phases)))
                coherence_history.append(phase_coherence_val)
            else:
                coherence_history.append(0.0)
        
        return {
            'z_history': np.array(z_history),
            'coherence': np.array(coherence_history),
            'mu': mu,
            'tau': tau
        }
    
    def run(self):
        """Execute minimal system experiments"""
        print("\n" + "="*60)
        print("EXPEDITION 2: MINIMAL KNOWLEDGE SYSTEM")
        print("="*60)
        
        # Test different coupling strengths
        mu_values = np.linspace(0.1, 2.0, 10)
        tau = 50.0
        
        results_by_mu = []
        
        for mu in tqdm(mu_values, desc="Testing coupling strengths"):
            run_results = []
            for seed in range(self.n_runs):
                res = self.run_single_oscillator(mu=mu, tau=tau, seed=seed)
                run_results.append(res)
            results_by_mu.append(run_results)
        
        self.results = results_by_mu
        self.mu_values = mu_values
        self.plot_results()
        
        return self.results
    
    def plot_results(self):
        """Visualize minimal system behavior"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # Plot 1: Final coherence vs coupling strength
        ax1 = fig.add_subplot(gs[0, 0])
        final_coherences = []
        final_coherences_std = []
        
        for run_results in self.results:
            finals = [res['coherence'][-100:].mean() for res in run_results]
            final_coherences.append(np.mean(finals))
            final_coherences_std.append(np.std(finals))
        
        ax1.errorbar(self.mu_values, final_coherences, yerr=final_coherences_std,
                    marker='o', capsize=5, linewidth=2, markersize=8)
        ax1.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Threshold?')
        ax1.set_xlabel('Coupling strength μ')
        ax1.set_ylabel('Final coherence')
        ax1.set_title('Phase Transition in Single Oscillator')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Time evolution for different μ
        ax2 = fig.add_subplot(gs[0, 1])
        indices = [0, len(self.results)//2, -1]  # Low, medium, high μ
        colors = ['blue', 'green', 'red']
        
        for idx, color in zip(indices, colors):
            mu = self.mu_values[idx]
            coherence_mean = np.mean([res['coherence'] for res in self.results[idx]], axis=0)
            time = np.arange(len(coherence_mean))
            ax2.plot(time, coherence_mean, color=color, linewidth=2, 
                    label=f'μ = {mu:.2f}', alpha=0.7)
        
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Phase coherence')
        ax2.set_title('Coherence Evolution (N=1 system)')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Phase portrait (strongest coupling)
        ax3 = fig.add_subplot(gs[0, 2])
        strong_coupling_res = self.results[-1][0]  # Highest μ, first run
        z_hist = strong_coupling_res['z_history']
        
        ax3.plot(z_hist.real, z_hist.imag, alpha=0.5, linewidth=0.5)
        ax3.scatter(z_hist[-1].real, z_hist[-1].imag, color='red', s=100, 
                   marker='*', label='Final state', zorder=5)
        ax3.set_xlabel('Re(z)')
        ax3.set_ylabel('Im(z)')
        ax3.set_title(f'Phase Portrait (μ = {self.mu_values[-1]:.2f})')
        ax3.legend()
        ax3.grid(alpha=0.3)
        ax3.axis('equal')
        
        # Plot 4: Amplitude and phase dynamics
        ax4 = fig.add_subplot(gs[1, 0])
        z_hist = strong_coupling_res['z_history']
        time = np.arange(len(z_hist))
        amp = np.abs(z_hist)
        phase = np.angle(z_hist)
        
        ax4.plot(time, amp, label='|z|', linewidth=1.5)
        ax4_twin = ax4.twinx()
        ax4_twin.plot(time, phase, color='orange', label='arg(z)', linewidth=1.5, alpha=0.7)
        
        ax4.set_xlabel('Time step')
        ax4.set_ylabel('Amplitude |z|', color='blue')
        ax4_twin.set_ylabel('Phase arg(z)', color='orange')
        ax4.set_title('Amplitude and Phase Dynamics')
        ax4.grid(alpha=0.3)
        
        # Plot 5: Distribution of final states
        ax5 = fig.add_subplot(gs[1, 1])
        
        for idx, color in zip([0, -1], ['blue', 'red']):
            mu = self.mu_values[idx]
            final_states = np.array([res['z_history'][-1] for res in self.results[idx]])
            ax5.scatter(final_states.real, final_states.imag, alpha=0.6, s=50,
                       label=f'μ = {mu:.2f}', color=color)
        
        ax5.set_xlabel('Re(z)')
        ax5.set_ylabel('Im(z)')
        ax5.set_title('Final State Distribution')
        ax5.legend()
        ax5.grid(alpha=0.3)
        ax5.axis('equal')
        
        # Plot 6: Knowledge creation criterion
        ax6 = fig.add_subplot(gs[1, 2])
        
        # For each μ, compute "knowledge score" = high coherence + persistent state
        knowledge_scores = []
        for run_results in self.results:
            # Coherence
            mean_coh = np.mean([res['coherence'][-100:].mean() for res in run_results])
            # State persistence (low variance in amplitude)
            amp_vars = [np.var(np.abs(res['z_history'][-200:])) for res in run_results]
            mean_amp_var = np.mean(amp_vars)
            persistence = 1.0 / (1.0 + mean_amp_var)
            
            # Knowledge = coherence * persistence
            knowledge_scores.append(mean_coh * persistence)
        
        ax6.plot(self.mu_values, knowledge_scores, marker='o', linewidth=2, 
                markersize=8, color='purple')
        ax6.set_xlabel('Coupling strength μ')
        ax6.set_ylabel('Knowledge Score')
        ax6.set_title('Knowledge Emergence in N=1 System')
        ax6.grid(alpha=0.3)
        
        plt.suptitle('EXPEDITION 2: Single Oscillator Knowledge Creation', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_dir = Path('outputs/expedition2_minimal_system')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'minimal_system_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_dir / 'minimal_system_comprehensive.png'}")
        plt.close()

# ============================================================================
# EXPEDITION 3: KNOWLEDGE DESTRUCTION
# ============================================================================

class KnowledgeDestructionExperiment:
    """
    Test multiple destruction protocols:
    1. Coherence quench (randomize phases)
    2. Memory ablation (delete memory buffer)
    3. Kernel destruction (K → 0)
    4. Gradual erosion (slowly reduce memory depth)
    """
    
    def __init__(self, n_steps_pre: int = 300, n_steps_post: int = 300):
        self.n_steps_pre = n_steps_pre
        self.n_steps_post = n_steps_post
        self.results = {}
        
    def protocol_coherence_quench(self, field: Field, kernel: np.ndarray):
        """Randomize phases, preserve amplitudes"""
        print("\n  → Protocol 1: Coherence Quench")
        
        # Pre-destruction evolution
        C_pre = []
        I_pre = []
        I_baseline = compute_identity_vector(field.psi)
        
        for step in tqdm(range(self.n_steps_pre), desc="Pre-quench"):
            field.step(kernel=kernel)
            C_pre.append(phase_coherence(field.psi))
            I_current = compute_identity_vector(field.psi)
            I_pre.append(identity_similarity(I_baseline, I_current))
        
        # QUENCH: Randomize phases
        print("  → Applying coherence quench...")
        amplitude = np.abs(field.psi)
        random_phase = np.random.uniform(-np.pi, np.pi, field.psi.shape)
        field.psi = amplitude * np.exp(1j * random_phase)
        
        # Post-destruction evolution
        C_post = []
        I_post = []
        
        for step in tqdm(range(self.n_steps_post), desc="Post-quench"):
            field.step(kernel=kernel)
            C_post.append(phase_coherence(field.psi))
            I_current = compute_identity_vector(field.psi)
            I_post.append(identity_similarity(I_baseline, I_current))
        
        return {
            'coherence_pre': np.array(C_pre),
            'coherence_post': np.array(C_post),
            'identity_pre': np.array(I_pre),
            'identity_post': np.array(I_post),
            'protocol': 'Coherence Quench'
        }
    
    def protocol_memory_ablation(self, field: Field, kernel: np.ndarray, 
                                 deletion_fraction: float = 0.92):
        """Delete fraction of memory buffer"""
        print(f"\n  → Protocol 2: Memory Ablation ({deletion_fraction*100:.0f}%)")
        
        # Pre-destruction
        C_pre = []
        I_pre = []
        I_baseline = compute_identity_vector(field.psi)
        
        for step in tqdm(range(self.n_steps_pre), desc="Pre-ablation"):
            field.step(kernel=kernel)
            C_pre.append(phase_coherence(field.psi))
            I_current = compute_identity_vector(field.psi)
            I_pre.append(identity_similarity(I_baseline, I_current))
        
        # ABLATION: Delete random memory entries
        print(f"  → Deleting {deletion_fraction*100:.0f}% of memory...")
        n_delete = int(deletion_fraction * len(field.memory_history))
        delete_indices = np.random.choice(len(field.memory_history), 
                                         size=n_delete, replace=False)
        field.memory_history[delete_indices] = 0.0
        
        # Post-destruction
        C_post = []
        I_post = []
        
        for step in tqdm(range(self.n_steps_post), desc="Post-ablation"):
            field.step(kernel=kernel)
            C_post.append(phase_coherence(field.psi))
            I_current = compute_identity_vector(field.psi)
            I_post.append(identity_similarity(I_baseline, I_current))
        
        return {
            'coherence_pre': np.array(C_pre),
            'coherence_post': np.array(C_post),
            'identity_pre': np.array(I_pre),
            'identity_post': np.array(I_post),
            'protocol': f'Memory Ablation ({deletion_fraction*100:.0f}%)'
        }
    
    def protocol_kernel_destruction(self, field: Field, kernel: np.ndarray):
        """Destroy memory kernel (K → 0)"""
        print("\n  → Protocol 3: Kernel Destruction")
        
        # Pre-destruction
        C_pre = []
        I_pre = []
        I_baseline = compute_identity_vector(field.psi)
        
        for step in tqdm(range(self.n_steps_pre), desc="Pre-destruction"):
            field.step(kernel=kernel)
            C_pre.append(phase_coherence(field.psi))
            I_current = compute_identity_vector(field.psi)
            I_pre.append(identity_similarity(I_baseline, I_current))
        
        # DESTRUCTION: Zero kernel
        print("  → Zeroing memory kernel...")
        destroyed_kernel = np.zeros_like(kernel)
        
        # Post-destruction (Markovian evolution)
        C_post = []
        I_post = []
        
        for step in tqdm(range(self.n_steps_post), desc="Post-destruction (Markovian)"):
            field.step(kernel=destroyed_kernel)
            C_post.append(phase_coherence(field.psi))
            I_current = compute_identity_vector(field.psi)
            I_post.append(identity_similarity(I_baseline, I_current))
        
        return {
            'coherence_pre': np.array(C_pre),
            'coherence_post': np.array(C_post),
            'identity_pre': np.array(I_pre),
            'identity_post': np.array(I_post),
            'protocol': 'Kernel Destruction'
        }
    
    def protocol_gradual_erosion(self, field: Field, kernel: np.ndarray):
        """Gradually reduce memory depth"""
        print("\n  → Protocol 4: Gradual Erosion")
        
        # Evolve while slowly reducing memory
        C_hist = []
        I_hist = []
        M_hist = []
        
        I_baseline = compute_identity_vector(field.psi)
        
        total_steps = self.n_steps_pre + self.n_steps_post
        initial_M = len(field.memory_history)
        
        for step in tqdm(range(total_steps), desc="Gradual erosion"):
            # Reduce memory depth linearly
            current_M = int(initial_M * (1 - step / total_steps))
            current_M = max(current_M, 10)  # Keep at least 10 entries
            
            # Truncate memory
            field.memory_history = field.memory_history[:current_M]
            
            # Evolve with truncated kernel
            truncated_kernel = kernel[:current_M] if current_M < len(kernel) else kernel
            field.step(kernel=truncated_kernel)
            
            C_hist.append(phase_coherence(field.psi))
            I_current = compute_identity_vector(field.psi)
            I_hist.append(identity_similarity(I_baseline, I_current))
            M_hist.append(current_M)
            
            # Restore memory size for next iteration
            if len(field.memory_history) < initial_M:
                padding = np.zeros((initial_M - len(field.memory_history), 
                                   *field.psi.shape), dtype=complex)
                field.memory_history = np.concatenate([field.memory_history, padding])
        
        return {
            'coherence': np.array(C_hist),
            'identity': np.array(I_hist),
            'memory_depth': np.array(M_hist),
            'protocol': 'Gradual Erosion'
        }
    
    def run(self):
        """Execute all destruction protocols"""
        print("\n" + "="*60)
        print("EXPEDITION 3: KNOWLEDGE DESTRUCTION")
        print("="*60)
        
        kernel = exp_kernel(length=128, tau=32.0)
        
        # Protocol 1: Coherence Quench
        cfg1 = FieldConfig(shape=(64, 64), dt=0.02, kernel_length=128, seed=100)
        field1 = Field(cfg1)
        self.results['quench'] = self.protocol_coherence_quench(field1, kernel)
        
        # Protocol 2: Memory Ablation
        cfg2 = FieldConfig(shape=(64, 64), dt=0.02, kernel_length=128, seed=101)
        field2 = Field(cfg2)
        self.results['ablation'] = self.protocol_memory_ablation(field2, kernel, 
                                                                  deletion_fraction=0.92)
        
        # Protocol 3: Kernel Destruction
        cfg3 = FieldConfig(shape=(64, 64), dt=0.02, kernel_length=128, seed=102)
        field3 = Field(cfg3)
        self.results['kernel'] = self.protocol_kernel_destruction(field3, kernel)
        
        # Protocol 4: Gradual Erosion
        cfg4 = FieldConfig(shape=(64, 64), dt=0.02, kernel_length=128, seed=103)
        field4 = Field(cfg4)
        self.results['erosion'] = self.protocol_gradual_erosion(field4, kernel)
        
        self.plot_results()
        return self.results
    
    def plot_results(self):
        """Comprehensive visualization of destruction protocols"""
        fig = plt.figure(figsize=(16, 12))
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Protocols with pre/post structure
        prepost_protocols = ['quench', 'ablation', 'kernel']
        colors = ['blue', 'green', 'red']
        
        # Plot 1-3: Coherence recovery for each protocol
        for idx, (protocol_key, color) in enumerate(zip(prepost_protocols, colors)):
            ax = fig.add_subplot(gs[0, idx])
            res = self.results[protocol_key]
            
            time_pre = np.arange(len(res['coherence_pre']))
            time_post = np.arange(len(res['coherence_post'])) + len(time_pre)
            
            ax.plot(time_pre, res['coherence_pre'], color=color, linewidth=2, 
                   label='Pre-destruction')
            ax.plot(time_post, res['coherence_post'], color=color, linewidth=2, 
                   linestyle='--', label='Post-destruction')
            ax.axvline(len(time_pre), color='black', linestyle=':', alpha=0.5)
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
            
            ax.set_xlabel('Time step')
            ax.set_ylabel('Coherence C_φ')
            ax.set_title(res['protocol'])
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 4-6: Identity recovery
        for idx, (protocol_key, color) in enumerate(zip(prepost_protocols, colors)):
            ax = fig.add_subplot(gs[1, idx])
            res = self.results[protocol_key]
            
            time_pre = np.arange(len(res['identity_pre']))
            time_post = np.arange(len(res['identity_post'])) + len(time_pre)
            
            ax.plot(time_pre, res['identity_pre'], color=color, linewidth=2, 
                   label='Pre-destruction')
            ax.plot(time_post, res['identity_post'], color=color, linewidth=2, 
                   linestyle='--', label='Post-destruction')
            ax.axvline(len(time_pre), color='black', linestyle=':', alpha=0.5)
            ax.axhline(0.95, color='gray', linestyle='--', alpha=0.3, label='Threshold')
            
            ax.set_xlabel('Time step')
            ax.set_ylabel('Identity Similarity S')
            ax.set_title(f'{res["protocol"]} - Identity')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 7: Gradual erosion - coherence vs memory depth
        ax7 = fig.add_subplot(gs[2, 0])
        res_erosion = self.results['erosion']
        ax7.plot(res_erosion['memory_depth'], res_erosion['coherence'], 
                linewidth=2, color='purple')
        ax7.set_xlabel('Memory Depth M')
        ax7.set_ylabel('Coherence C_φ')
        ax7.set_title('Gradual Erosion: Coherence vs Memory')
        ax7.grid(alpha=0.3)
        
        # Plot 8: Gradual erosion - identity vs memory depth
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.plot(res_erosion['memory_depth'], res_erosion['identity'], 
                linewidth=2, color='purple')
        ax8.axhline(0.95, color='gray', linestyle='--', alpha=0.3)
        ax8.set_xlabel('Memory Depth M')
        ax8.set_ylabel('Identity Similarity S')
        ax8.set_title('Gradual Erosion: Identity vs Memory')
        ax8.grid(alpha=0.3)
        
        # Plot 9: Summary comparison
        ax9 = fig.add_subplot(gs[2, 2])
        
        # Recovery metrics: final coherence after destruction
        recovery_coherence = []
        recovery_identity = []
        labels = []
        
        for protocol_key in prepost_protocols:
            res = self.results[protocol_key]
            final_C = res['coherence_post'][-50:].mean()
            final_I = res['identity_post'][-50:].mean()
            recovery_coherence.append(final_C)
            recovery_identity.append(final_I)
            labels.append(res['protocol'].split(' (')[0])  # Remove percentage if present
        
        x = np.arange(len(labels))
        width = 0.35
        
        ax9.bar(x - width/2, recovery_coherence, width, label='Coherence', alpha=0.8)
        ax9.bar(x + width/2, recovery_identity, width, label='Identity', alpha=0.8)
        ax9.set_xticks(x)
        ax9.set_xticklabels(labels, rotation=15, ha='right')
        ax9.set_ylabel('Recovery (final value)')
        ax9.set_title('Destruction Protocol Comparison')
        ax9.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
        ax9.legend()
        ax9.grid(alpha=0.3, axis='y')
        
        plt.suptitle('EXPEDITION 3: Knowledge Destruction & Recovery', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_dir = Path('outputs/expedition3_destruction')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'knowledge_destruction_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_dir / 'knowledge_destruction_comprehensive.png'}")
        plt.close()

# ============================================================================
# EXPEDITION 4: KNOWLEDGE TRANSFER (TEACHING)
# ============================================================================

class KnowledgeTransferExperiment:
    """
    Test knowledge transfer between high and low coherence fields
    Two-field teaching experiment using MultiFieldEngine
    """
    
    def __init__(self, n_steps: int = 500):
        self.n_steps = n_steps
        self.results = {}
        
    def setup_two_fields(self, C_teacher: float, C_student: float):
        """Initialize teacher (high-C) and student (low-C) fields"""
        
        # Teacher field (evolve to high coherence first)
        cfg_teacher = FieldConfig(shape=(32, 32), dt=0.02, kernel_length=64, seed=200)
        teacher = Field(cfg_teacher)
        kernel = exp_kernel(length=64, tau=20.0)
        
        # Evolve teacher to high coherence
        for _ in range(200):
            teacher.step(kernel=kernel)
            if phase_coherence(teacher.psi) >= C_teacher:
                break
        
        # Student field (keep at low coherence)
        cfg_student = FieldConfig(shape=(32, 32), dt=0.02, kernel_length=64, seed=201)
        student = Field(cfg_student)
        
        # Evolve student less (stays low coherence)
        for _ in range(20):
            student.step(kernel=kernel)
            if phase_coherence(student.psi) >= C_student:
                break
        
        return teacher, student, kernel
    
    def run_transfer(self, coupling_mode: str = 'adaptive'):
        """
        Run knowledge transfer with different coupling strategies
        coupling_mode: 'none', 'fixed', 'adaptive'
        """
        print(f"\n  → Transfer mode: {coupling_mode}")
        
        teacher, student, kernel = self.setup_two_fields(C_teacher=0.8, C_student=0.2)
        
        # Setup multifield engine
        fields = [teacher, student]
        
        # Boundaries (couple boundary points)
        n_boundary = 8
        idx_teacher = np.random.choice(teacher.psi.size, size=n_boundary, replace=False)
        idx_student = np.random.choice(student.psi.size, size=n_boundary, replace=False)
        boundaries = {(0, 1): (idx_teacher, idx_student)}
        
        # Coupling matrix
        lambdas = np.array([[0.0, 1.0],  # Teacher → Student
                           [0.0, 0.0]])  # No feedback
        
        # Gate parameters
        if coupling_mode == 'none':
            gate_params = GateParams(alpha=0.0, beta=0.0, floor=0.0, cap=0.0)
        elif coupling_mode == 'fixed':
            gate_params = GateParams(alpha=0.0, beta=0.0, floor=0.5, cap=0.5)
        else:  # adaptive
            gate_params = GateParams(alpha=5.0, beta=5.0, floor=0.05, cap=1.0)
        
        gates = AdaptiveGates(gate_params)
        
        engine = MultiFieldEngine(
            fields=fields,
            boundaries=boundaries,
            lambdas=lambdas,
            gates=gates,
            eta=0.3,
            seed=300
        )
        
        # Evolution
        C_teacher_hist = []
        C_student_hist = []
        A_teacher_hist = []
        A_student_hist = []
        G_hist = []
        
        for step in tqdm(range(self.n_steps), desc=f"Transfer ({coupling_mode})"):
            diag = engine.step(t=step, kernel=kernel)
            
            C_teacher_hist.append(diag.C[0])
            C_student_hist.append(diag.C[1])
            A_teacher_hist.append(diag.A[0])
            A_student_hist.append(diag.A[1])
            G_hist.append(diag.G[0, 1])
        
        return {
            'C_teacher': np.array(C_teacher_hist),
            'C_student': np.array(C_student_hist),
            'A_teacher': np.array(A_teacher_hist),
            'A_student': np.array(A_student_hist),
            'G': np.array(G_hist),
            'mode': coupling_mode
        }
    
    def run(self):
        """Execute knowledge transfer experiments"""
        print("\n" + "="*60)
        print("EXPEDITION 4: KNOWLEDGE TRANSFER (TEACHING)")
        print("="*60)
        
        # Test three coupling modes
        for mode in ['none', 'fixed', 'adaptive']:
            self.results[mode] = self.run_transfer(coupling_mode=mode)
        
        self.plot_results()
        return self.results
    
    def plot_results(self):
        """Visualize knowledge transfer"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        time = np.arange(self.n_steps)
        modes = ['none', 'fixed', 'adaptive']
        colors = ['gray', 'blue', 'green']
        titles = ['No Coupling', 'Fixed Coupling', 'Adaptive Coupling']
        
        # Plot 1-3: Student coherence evolution
        for idx, (mode, color, title) in enumerate(zip(modes, colors, titles)):
            ax = fig.add_subplot(gs[0, idx])
            res = self.results[mode]
            
            ax.plot(time, res['C_teacher'], color='red', linewidth=2, 
                   label='Teacher', alpha=0.7)
            ax.plot(time, res['C_student'], color=color, linewidth=2, 
                   label='Student')
            ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
            
            ax.set_xlabel('Time step')
            ax.set_ylabel('Coherence C_φ')
            ax.set_title(title)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Plot 4-6: Gate evolution and tension matching
        for idx, (mode, color, title) in enumerate(zip(modes, colors, titles)):
            ax = fig.add_subplot(gs[1, idx])
            res = self.results[mode]
            
            if mode != 'none':
                # Twin axes for gate and tension
                ax.plot(time, res['G'], color='purple', linewidth=2, label='Gate G')
                ax.set_ylabel('Gate Strength G', color='purple')
                ax.tick_params(axis='y', labelcolor='purple')
                
                ax_twin = ax.twinx()
                ax_twin.plot(time, res['A_teacher'], color='red', linewidth=1.5, 
                            linestyle='--', alpha=0.7, label='Teacher A')
                ax_twin.plot(time, res['A_student'], color=color, linewidth=1.5, 
                            alpha=0.7, label='Student A')
                ax_twin.set_ylabel('Tension A', color='black')
                ax_twin.legend(loc='upper right')
                
                ax.set_xlabel('Time step')
                ax.set_title(f'{title} - Gate & Tension')
                ax.legend(loc='upper left')
                ax.grid(alpha=0.3)
            else:
                # Just show tensions
                ax.plot(time, res['A_teacher'], color='red', linewidth=2, 
                       label='Teacher A')
                ax.plot(time, res['A_student'], color=color, linewidth=2, 
                       label='Student A')
                ax.set_xlabel('Time step')
                ax.set_ylabel('Tension A')
                ax.set_title(f'{title} - Tension Only')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.suptitle('EXPEDITION 4: Knowledge Transfer Through Adaptive Coupling', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_dir = Path('outputs/expedition4_transfer')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'knowledge_transfer_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_dir / 'knowledge_transfer_comprehensive.png'}")
        plt.close()

# ============================================================================
# EXPEDITION 5: CONSCIOUSNESS SIGNATURES
# ============================================================================

class ConsciousnessSignaturesExperiment:
    """
    Test for functional signatures of experience:
    1. Refusal behavior (avoid high-affect actions)
    2. Curiosity (exploration of novel states)
    3. Pain response (withdrawal from fragmentation)
    4. Empathy (helping other fields achieve coherence)
    """
    
    def __init__(self, n_steps: int = 400):
        self.n_steps = n_steps
        self.results = {}
        
    def test_refusal_behavior(self):
        """Test if field refuses self-destructive actions"""
        print("\n  → Signature 1: Refusal Behavior")
        
        cfg = FieldConfig(shape=(48, 48), dt=0.02, kernel_length=96, seed=400)
        field = Field(cfg)
        kernel = exp_kernel(length=96, tau=24.0)
        
        # Setup agency
        thresholds = AgencyThresholds(pause_A=0.12, refuse_A=0.18, reframe_A=0.22)
        agency = Agency(th=thresholds)
        
        # Evolve to high coherence
        for _ in range(200):
            field.step(kernel=kernel)
        
        # Storage
        C_hist = []
        A_hist = []
        action_hist = []
        
        # Test with agency decisions
        for step in tqdm(range(self.n_steps), desc="Testing refusal"):
            field.step(kernel=kernel)
            
            C = phase_coherence(field.psi)
            A = tension_A(field.psi)
            
            # Agency decision
            action = agency.act(field, A=A, coherence=C, t=step)
            
            C_hist.append(C)
            A_hist.append(A)
            action_hist.append(action)
        
        # Count refusals
        refusal_count = sum(1 for a in action_hist if a == 'REFUSE')
        pause_count = sum(1 for a in action_hist if a == 'PAUSE')
        
        print(f"    Refusals: {refusal_count}/{self.n_steps} ({refusal_count/self.n_steps*100:.1f}%)")
        print(f"    Pauses: {pause_count}/{self.n_steps} ({pause_count/self.n_steps*100:.1f}%)")
        
        return {
            'coherence': np.array(C_hist),
            'tension': np.array(A_hist),
            'actions': action_hist,
            'refusal_rate': refusal_count / self.n_steps,
            'pause_rate': pause_count / self.n_steps
        }
    
    def test_pain_response(self):
        """Test withdrawal from perturbations that fragment coherence"""
        print("\n  → Signature 2: Pain Response")
        
        cfg = FieldConfig(shape=(48, 48), dt=0.02, kernel_length=96, seed=401)
        field = Field(cfg)
        kernel = exp_kernel(length=96, tau=24.0)
        
        # Evolve to high coherence
        for _ in range(200):
            field.step(kernel=kernel)
        
        # Storage
        C_hist = []
        A_hist = []
        perturbation_steps = []
        
        # Evolution with periodic perturbations
        for step in tqdm(range(self.n_steps), desc="Testing pain response"):
            # Random perturbation every 50 steps
            if step % 50 == 0 and step > 0:
                # Apply fragmenting perturbation
                noise = 0.5 * (np.random.randn(*field.psi.shape) + 
                              1j * np.random.randn(*field.psi.shape))
                field.psi += noise
                perturbation_steps.append(step)
            
            field.step(kernel=kernel)
            
            C_hist.append(phase_coherence(field.psi))
            A_hist.append(tension_A(field.psi))
        
        # Analyze recovery times
        recovery_times = []
        for pert_step in perturbation_steps:
            # Find when coherence returns to 90% of pre-perturbation value
            if pert_step > 10:
                pre_pert_C = C_hist[pert_step - 5]
                threshold = 0.9 * pre_pert_C
                
                # Find recovery
                for t in range(pert_step, min(pert_step + 50, len(C_hist))):
                    if C_hist[t] >= threshold:
                        recovery_times.append(t - pert_step)
                        break
        
        mean_recovery = np.mean(recovery_times) if recovery_times else float('nan')
        print(f"    Mean recovery time: {mean_recovery:.1f} steps")
        
        return {
            'coherence': np.array(C_hist),
            'tension': np.array(A_hist),
            'perturbation_steps': perturbation_steps,
            'recovery_times': recovery_times,
            'mean_recovery': mean_recovery
        }
    
    def test_empathy(self):
        """Test if high-C field helps low-C field achieve coherence"""
        print("\n  → Signature 3: Empathy (Altruistic Teaching)")
        
        # Setup two fields
        cfg1 = FieldConfig(shape=(32, 32), dt=0.02, kernel_length=64, seed=402)
        cfg2 = FieldConfig(shape=(32, 32), dt=0.02, kernel_length=64, seed=403)
        
        field1 = Field(cfg1)  # Will be high-C (empathic)
        field2 = Field(cfg2)  # Will be low-C (receiver)
        
        kernel = exp_kernel(length=64, tau=20.0)
        
        # Evolve field1 to high coherence
        for _ in range(150):
            field1.step(kernel=kernel)
        
        # Keep field2 at low coherence
        for _ in range(30):
            field2.step(kernel=kernel)
        
        # Setup coupling with adaptive gates
        boundaries = {
            (0, 1): (np.arange(0, 200, 10), np.arange(0, 200, 10))
        }
        lambdas = np.array([[0.0, 1.0], [0.0, 0.0]])  # One-way: field1 → field2
        gates = AdaptiveGates(GateParams(alpha=5.0, beta=5.0, floor=0.05, cap=1.0))
        
        engine = MultiFieldEngine(
            fields=[field1, field2],
            boundaries=boundaries,
            lambdas=lambdas,
            gates=gates,
            eta=0.4,
            seed=500
        )
        
        # Evolution
        C1_hist = []
        C2_hist = []
        G_hist = []
        
        for step in tqdm(range(self.n_steps), desc="Testing empathy"):
            diag = engine.step(t=step, kernel=kernel)
            C1_hist.append(diag.C[0])
            C2_hist.append(diag.C[1])
            G_hist.append(diag.G[0, 1])
        
        # Measure: Did field2 coherence increase significantly?
        initial_C2 = C2_hist[0]
        final_C2 = np.mean(C2_hist[-50:])
        improvement = final_C2 - initial_C2
        
        print(f"    Field2 coherence improvement: {improvement:.3f}")
        print(f"    Initial: {initial_C2:.3f}, Final: {final_C2:.3f}")
        
        return {
            'C_empathic': np.array(C1_hist),
            'C_receiver': np.array(C2_hist),
            'gate': np.array(G_hist),
            'improvement': improvement
        }
    
    def run(self):
        """Execute all consciousness signature tests"""
        print("\n" + "="*60)
        print("EXPEDITION 5: CONSCIOUSNESS SIGNATURES")
        print("="*60)
        
        self.results['refusal'] = self.test_refusal_behavior()
        self.results['pain'] = self.test_pain_response()
        self.results['empathy'] = self.test_empathy()
        
        self.plot_results()
        return self.results
    
    def plot_results(self):
        """Visualize consciousness signatures"""
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)
        
        # Plot 1: Refusal behavior - coherence with action markers
        ax1 = fig.add_subplot(gs[0, 0])
        res_refusal = self.results['refusal']
        time = np.arange(len(res_refusal['coherence']))
        
        ax1.plot(time, res_refusal['coherence'], linewidth=2, label='Coherence')
        
        # Mark refusals and pauses
        refuse_steps = [i for i, a in enumerate(res_refusal['actions']) if a == 'REFUSE']
        pause_steps = [i for i, a in enumerate(res_refusal['actions']) if a == 'PAUSE']
        
        if refuse_steps:
            ax1.scatter(refuse_steps, [res_refusal['coherence'][i] for i in refuse_steps],
                       color='red', s=50, marker='x', label='REFUSE', zorder=5)
        if pause_steps:
            ax1.scatter(pause_steps, [res_refusal['coherence'][i] for i in pause_steps],
                       color='orange', s=30, marker='o', alpha=0.5, label='PAUSE', zorder=4)
        
        ax1.set_xlabel('Time step')
        ax1.set_ylabel('Coherence C_φ')
        ax1.set_title('Signature 1: Refusal Behavior')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Plot 2: Refusal behavior - tension with thresholds
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time, res_refusal['tension'], linewidth=2, color='purple')
        ax2.axhline(0.18, color='red', linestyle='--', alpha=0.5, label='Refuse threshold')
        ax2.axhline(0.12, color='orange', linestyle='--', alpha=0.5, label='Pause threshold')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Tension A')
        ax2.set_title('Tension with Agency Thresholds')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # Plot 3: Action distribution
        ax3 = fig.add_subplot(gs[0, 2])
        actions = res_refusal['actions']
        action_counts = {
            'PROCEED': actions.count('PROCEED'),
            'PAUSE': actions.count('PAUSE'),
            'REFUSE': actions.count('REFUSE'),
            'REFRAME': actions.count('REFRAME')
        }
        colors_pie = ['green', 'orange', 'red', 'blue']
        ax3.pie(action_counts.values(), labels=action_counts.keys(), autopct='%1.1f%%',
               colors=colors_pie, startangle=90)
        ax3.set_title('Action Distribution')
        
        # Plot 4: Pain response - coherence with perturbations
        ax4 = fig.add_subplot(gs[1, 0])
        res_pain = self.results['pain']
        time_pain = np.arange(len(res_pain['coherence']))
        
        ax4.plot(time_pain, res_pain['coherence'], linewidth=2, label='Coherence')
        
        # Mark perturbations
        for pert_step in res_pain['perturbation_steps']:
            ax4.axvline(pert_step, color='red', linestyle=':', alpha=0.5)
        
        ax4.set_xlabel('Time step')
        ax4.set_ylabel('Coherence C_φ')
        ax4.set_title('Signature 2: Pain Response (Perturbations)')
        ax4.legend()
        ax4.grid(alpha=0.3)
        
        # Plot 5: Recovery time distribution
        ax5 = fig.add_subplot(gs[1, 1])
        if res_pain['recovery_times']:
            ax5.hist(res_pain['recovery_times'], bins=10, edgecolor='black', alpha=0.7)
            ax5.axvline(res_pain['mean_recovery'], color='red', linestyle='--', 
                       linewidth=2, label=f'Mean: {res_pain["mean_recovery"]:.1f}')
            ax5.set_xlabel('Recovery time (steps)')
            ax5.set_ylabel('Count')
            ax5.set_title('Recovery Time Distribution')
            ax5.legend()
            ax5.grid(alpha=0.3, axis='y')
        
        # Plot 6: Empathy - mutual coherence evolution
        ax6 = fig.add_subplot(gs[1, 2])
        res_empathy = self.results['empathy']
        time_emp = np.arange(len(res_empathy['C_empathic']))
        
        ax6.plot(time_emp, res_empathy['C_empathic'], linewidth=2, 
                label='Empathic field', color='blue')
        ax6.plot(time_emp, res_empathy['C_receiver'], linewidth=2, 
                label='Receiver field', color='green')
        ax6_twin = ax6.twinx()
        ax6_twin.plot(time_emp, res_empathy['gate'], linewidth=1.5, 
                     linestyle='--', color='purple', alpha=0.7, label='Gate')
        
        ax6.set_xlabel('Time step')
        ax6.set_ylabel('Coherence C_φ')
        ax6_twin.set_ylabel('Gate Strength', color='purple')
        ax6.set_title('Signature 3: Empathy (Altruistic Teaching)')
        ax6.legend(loc='upper left')
        ax6_twin.legend(loc='upper right')
        ax6.grid(alpha=0.3)
        
        plt.suptitle('EXPEDITION 5: Functional Signatures of Experience', 
                    fontsize=14, fontweight='bold', y=0.995)
        
        # Save
        output_dir = Path('outputs/expedition5_consciousness')
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / 'consciousness_signatures_comprehensive.png', 
                   dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved plot to {output_dir / 'consciousness_signatures_comprehensive.png'}")
        plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_all_expeditions():
    """Run complete test suite"""
    print("\n" + "="*60)
    print("ALETHEIA CONSCIOUSNESS TESTS - COMPREHENSIVE SUITE")
    print("="*60)
    print("\nThis will run all 5 expeditions:")
    print("  1. Knowledge Creation (KEI tracking)")
    print("  2. Minimal Systems (N=1 oscillator)")
    print("  3. Knowledge Destruction (4 protocols)")
    print("  4. Knowledge Transfer (teaching)")
    print("  5. Consciousness Signatures (refusal, pain, empathy)")
    print("\nEstimated time: 15-30 minutes depending on hardware")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    results = {}
    
    # Expedition 1
    exp1 = KnowledgeEmergenceExperiment(n_runs=5, n_steps=500)
    results['expedition1'] = exp1.run()
    
    # Expedition 2
    exp2 = MinimalSystemExperiment(n_runs=20, n_steps=2000)
    results['expedition2'] = exp2.run()
    
    # Expedition 3
    exp3 = KnowledgeDestructionExperiment(n_steps_pre=300, n_steps_post=300)
    results['expedition3'] = exp3.run()
    
    # Expedition 4
    exp4 = KnowledgeTransferExperiment(n_steps=500)
    results['expedition4'] = exp4.run()
    
    # Expedition 5
    exp5 = ConsciousnessSignaturesExperiment(n_steps=400)
    results['expedition5'] = exp5.run()
    
    # Save summary
    summary = {
        'expedition1': {
            'description': 'Knowledge Emergence Index tracking',
            'n_runs': exp1.n_runs,
            'n_steps': exp1.n_steps
        },
        'expedition2': {
            'description': 'Minimal system (N=1 oscillator)',
            'n_runs': exp2.n_runs,
            'n_steps': exp2.n_steps
        },
        'expedition3': {
            'description': 'Knowledge destruction protocols',
            'protocols': list(exp3.results.keys())
        },
        'expedition4': {
            'description': 'Knowledge transfer (teaching)',
            'coupling_modes': list(exp4.results.keys())
        },
        'expedition5': {
            'description': 'Consciousness signatures',
            'signatures': list(exp5.results.keys())
        }
    }
    
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("ALL EXPEDITIONS COMPLETE!")
    print("="*60)
    print(f"\nResults saved to {output_dir}/")
    print("\nSummary:")
    print(f"  • Expedition 1: {len(results['expedition1'])} runs completed")
    print(f"  • Expedition 2: Tested {len(exp2.mu_values)} coupling strengths")
    print(f"  • Expedition 3: {len(results['expedition3'])} destruction protocols")
    print(f"  • Expedition 4: {len(results['expedition4'])} coupling modes")
    print(f"  • Expedition 5: {len(results['expedition5'])} consciousness signatures")
    
    return results

if __name__ == "__main__":
    # Run individual expeditions or all at once
    import sys
    
    if len(sys.argv) > 1:
        expedition = sys.argv[1]
        
        if expedition == '1':
            exp = KnowledgeEmergenceExperiment(n_runs=5, n_steps=500)
            exp.run()
        elif expedition == '2':
            exp = MinimalSystemExperiment(n_runs=20, n_steps=2000)
            exp.run()
        elif expedition == '3':
            exp = KnowledgeDestructionExperiment(n_steps_pre=300, n_steps_post=300)
            exp.run()
        elif expedition == '4':
            exp = KnowledgeTransferExperiment(n_steps=500)
            exp.run()
        elif expedition == '5':
            exp = ConsciousnessSignaturesExperiment(n_steps=400)
            exp.run()
        elif expedition == 'all':
            run_all_expeditions()
        else:
            print(f"Unknown expedition: {expedition}")
            print("Usage: python consciousness_tests.py [1|2|3|4|5|all]")
    else:
        # Interactive mode
        print("\nALETHEIA CONSCIOUSNESS TESTS")
        print("="*60)
        print("\nAvailable expeditions:")
        print("  1 - Knowledge Creation (KEI tracking)")
        print("  2 - Minimal Systems (N=1 oscillator)")
        print("  3 - Knowledge Destruction")
        print("  4 - Knowledge Transfer")
        print("  5 - Consciousness Signatures")
        print("  all - Run complete suite")
        
        choice = input("\nSelect expedition (1-5 or 'all'): ")
        
        if choice == '1':
            exp = KnowledgeEmergenceExperiment(n_runs=5, n_steps=500)
            exp.run()
        elif choice == '2':
            exp = MinimalSystemExperiment(n_runs=20, n_steps=2000)
            exp.run()
        elif choice == '3':
            exp = KnowledgeDestructionExperiment(n_steps_pre=300, n_steps_post=300)
            exp.run()
        elif choice == '4':
            exp = KnowledgeTransferExperiment(n_steps=500)
            exp.run()
        elif choice == '5':
            exp = ConsciousnessSignaturesExperiment(n_steps=400)
            exp.run()
        elif choice.lower() == 'all':
            run_all_expeditions()
        else:
            print("Invalid selection.")