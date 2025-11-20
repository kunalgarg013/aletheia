# Save as: src/ns_qfca_solver_stabilized.py

import numpy as np
from scipy.fft import fft2, ifft2, fftfreq
from scipy.special import gamma
from collections import deque
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Literal

@dataclass
class NavierStokesParams:
    N: int = 256
    L: float = 2*np.pi
    nu: float = 0.01
    hbar: float = 0.1
    m: float = 1.0
    g: float = 0.5
    tau_m: float = 5.0
    kernel_type: Literal['exponential', 'powerlaw', 'mittag_leffler', 'mixed'] = 'exponential'
    alpha: float = 0.5
    beta_mix: float = 0.3
    enable_memory: bool = True
    
    # Coherence stabilization
    coherence_stabilization: Literal['none', 'adaptive_viscosity', 'potential', 'renormalization', 'memory_gating'] = 'adaptive_viscosity'
    C_crit: float = 0.5
    lambda_coh: float = 0.1
    nu_boost_factor: float = 10.0
    renorm_strength: float = 0.3

    # Toggle Hamiltonian dynamics (kinetic + nonlinear potential)
    use_hamiltonian: bool = True

    # ---- NEW: Agency / affect gating ----
    enable_agency: bool = False
    A_pause: float = 1e-4
    A_refuse: float = 5e-4
    A_reframe: float = 2e-4
    agency_prediction_factor: float = 0.2


class MemoryKernel:
    """Memory kernel factory"""
    
    @staticmethod
    def exponential(length: int, tau_m: float) -> np.ndarray:
        tau = np.arange(length)
        K = np.exp(-tau / tau_m) / tau_m
        K[0] = 0
        return K / (np.sum(K) + 1e-12)
    
    @staticmethod
    def powerlaw(length: int, alpha: float = 0.5) -> np.ndarray:
        tau = np.arange(1, length + 1, dtype=float)
        K = tau**(-alpha) / gamma(1 - alpha)
        K = np.concatenate([[0], K[:-1]])
        return K / (np.sum(K) + 1e-12)
    
    @staticmethod
    def mittag_leffler(length: int, tau_m: float, alpha: float = 0.5) -> np.ndarray:
        tau = np.arange(length, dtype=float)
        argument = -(tau / tau_m)**alpha
        K = np.exp(argument)
        K[0] = 0
        return K / (np.sum(K) + 1e-12)
    
    @staticmethod
    def mixed(length: int, tau_m: float, alpha: float, beta: float) -> np.ndarray:
        K_exp = MemoryKernel.exponential(length, tau_m)
        K_pow = MemoryKernel.powerlaw(length, alpha)
        K = (1 - beta) * K_exp + beta * K_pow
        return K / (np.sum(K) + 1e-12)
    
    @classmethod
    def get_kernel(cls, kernel_type: str, length: int, tau_m: float, 
                   alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
        kernels = {
            'exponential': lambda: cls.exponential(length, tau_m),
            'powerlaw': lambda: cls.powerlaw(length, alpha),
            'mittag_leffler': lambda: cls.mittag_leffler(length, tau_m, alpha),
            'mixed': lambda: cls.mixed(length, tau_m, alpha, beta)
        }
        
        if kernel_type not in kernels:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        return kernels[kernel_type]()


@dataclass
class AgencyThresholds:
    pause_A: float
    refuse_A: float
    reframe_A: float


class Agency:
    """Simple affect-based agency gate."""

    def __init__(self, thresholds: AgencyThresholds):
        self.th = thresholds
        self.paused_steps = 0

    def decide(self, A: float, A_hat: float) -> str:
        if A_hat >= self.th.refuse_A or A >= self.th.refuse_A:
            return "REFUSE"
        if A_hat >= self.th.reframe_A:
            return "REFRAME"
        if A_hat >= self.th.pause_A or A >= self.th.pause_A:
            return "PAUSE"
        return "PROCEED"


class NavierStokesQFCA:
    """2D NS-QFCA with coherence stabilization"""
    
    def __init__(self, params: NavierStokesParams):
        self.p = params
        
        # Spatial grid
        self.x = np.linspace(0, params.L, params.N, endpoint=False)
        self.dx = params.L / params.N
        self.X, self.Y = np.meshgrid(self.x, self.x)
        
        # Wavenumber grid
        self.k = 2*np.pi * fftfreq(params.N, d=self.dx)
        self.kx, self.ky = np.meshgrid(self.k, self.k)
        self.k2 = self.kx**2 + self.ky**2
        self.k2[0,0] = 1e-10
        
        # Memory buffer
        memory_length = max(100, int(5 * params.tau_m / 0.01))
        self.memory_buffer = deque(maxlen=memory_length)
        
        # Kernel
        self.kernel = MemoryKernel.get_kernel(
            params.kernel_type, 
            memory_length, 
            params.tau_m,
            params.alpha,
            params.beta_mix
        )
        
        # Diagnostics
        self.history = {
            't': [], 'coherence': [], 'energy': [], 'enstrophy': [],
            'max_vorticity': [], 'quantum_pressure': [], 
            'rho_min': [], 'rho_max': [],
            'nu_effective': [],
            'tension': [], 
            'agency_action': [],
        }
        
        self.nu_eff_last = params.nu  # Track effective viscosity

        if self.p.enable_agency:
            th = AgencyThresholds(
                pause_A=self.p.A_pause,
                refuse_A=self.p.A_refuse,
                reframe_A=self.p.A_reframe
            )
            self.agency = Agency(th)
        else:
            self.agency = None
    
    def madelung_decompose(self, psi: np.ndarray) -> tuple:
        rho = np.abs(psi)**2
        S = self.p.hbar * np.angle(psi)
        return rho, S
    
    def velocity(self, psi: np.ndarray) -> tuple:
        psi_hat = fft2(psi)
        dpsi_dx = ifft2(1j * self.kx * psi_hat)
        dpsi_dy = ifft2(1j * self.ky * psi_hat)
        
        u = (self.p.hbar / self.p.m) * np.imag(dpsi_dx / (psi + 1e-10))
        v = (self.p.hbar / self.p.m) * np.imag(dpsi_dy / (psi + 1e-10))
        return u, v
    
    def vorticity(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        u_hat = fft2(u)
        v_hat = fft2(v)
        omega = np.real(ifft2(1j * self.kx * v_hat - 1j * self.ky * u_hat))
        return omega
    
    def quantum_potential(self, psi: np.ndarray) -> np.ndarray:
        rho = np.abs(psi)**2
        sqrt_rho = np.sqrt(rho + 1e-10)
        sqrt_rho_hat = fft2(sqrt_rho)
        laplacian_sqrt_rho = np.real(ifft2(-self.k2 * sqrt_rho_hat))
        Q = -(self.p.hbar**2 / (2 * self.p.m**2)) * laplacian_sqrt_rho / (sqrt_rho + 1e-10)
        return Q
    
    def coherence(self, psi: np.ndarray) -> float:
        phase_factor = psi / (np.abs(psi) + 1e-10)
        return float(np.abs(np.mean(phase_factor)))
    
    def energy(self, psi: np.ndarray) -> float:
        u, v = self.velocity(psi)
        return 0.5 * float(np.mean(u**2 + v**2))
    
    def enstrophy(self, psi: np.ndarray) -> float:
        u, v = self.velocity(psi)
        omega = self.vorticity(u, v)
        return 0.5 * float(np.mean(omega**2))

    def tension_A(self, psi: np.ndarray) -> float:
        return self.enstrophy(psi)
    
    def coherence_potential(self, psi: np.ndarray) -> np.ndarray:
        """Energetic penalty for low coherence"""
        C_phi = self.coherence(psi)
        
        if C_phi < self.p.C_crit:
            penalty = self.p.lambda_coh * (self.p.C_crit - C_phi)**2
            return penalty * psi
        else:
            return np.zeros_like(psi)
    
    def adaptive_viscosity(self, C_phi: float) -> float:
        """Coherence-dependent viscosity"""
        boost = self.p.nu_boost_factor * np.exp(-10.0 * (C_phi - self.p.C_crit))
        return self.p.nu * (1.0 + boost)
    
    def renormalize_coherence(self, psi: np.ndarray) -> np.ndarray:
        """Project onto minimum coherence manifold"""
        C_phi = self.coherence(psi)
        
        if C_phi < self.p.C_crit:
            theta = np.angle(psi)
            theta_mean = np.angle(np.mean(psi / (np.abs(psi) + 1e-10)))
            
            shift_strength = self.p.renorm_strength * (self.p.C_crit - C_phi) / self.p.C_crit
            theta_new = theta * (1 - shift_strength) + theta_mean * shift_strength
            
            return np.abs(psi) * np.exp(1j * theta_new)
        else:
            return psi
    
    def memory_term(self, psi: np.ndarray) -> np.ndarray:
        """Memory with optional coherence gating"""
        if not self.p.enable_memory or len(self.memory_buffer) == 0:
            return np.zeros_like(psi)
        
        n = len(self.memory_buffer)
        kernel_weights = self.kernel[:n].copy()
        
        # Coherence gating
        if self.p.coherence_stabilization == 'memory_gating':
            C_phi = self.coherence(psi)
            gate = np.tanh(5.0 * (C_phi - self.p.C_crit))
            gate = max(0.0, gate)
            kernel_weights *= gate
        
        mem = np.zeros_like(psi, dtype=complex)
        for i, psi_past in enumerate(self.memory_buffer):
            mem += kernel_weights[i] * psi_past
        
        return mem
    
    def compute_derivative(self, psi_current: np.ndarray) -> tuple:
        """Compute dpsi/dt with stabilization, return (dpsi_dt, nu_eff)"""
        psi_hat = fft2(psi_current)
        # ---- Agency deep integration ----
        action = None
        if self.agency is not None:
            A = self.tension_A(psi_current)
            C_phi = self.coherence(psi_current)
            A_hat = A * (1.0 + self.p.agency_prediction_factor * (1.0 - C_phi))

            action = self.agency.decide(A, A_hat)
            self._last_action = action

            # REFUSE → freeze evolution (zero derivative)
            if action == "REFUSE":
                return np.zeros_like(psi_current, dtype=complex), self.p.nu

            # PAUSE → suppress Hamiltonian amplitude
            if action == "PAUSE":
                psi_amp_factor = 0.5
                psi_current = psi_current * psi_amp_factor
                psi_hat = fft2(psi_current)

            # REFRAME → damp amplitudes proportional to tension
            if action == "REFRAME":
                ratio = min(1.0, max(0.0, (A_hat - self.p.A_pause) /
                                     (self.p.A_refuse - self.p.A_pause + 1e-12)))
                damp = 1.0 - 0.3 * ratio
                psi_current = psi_current * damp
                psi_hat = fft2(psi_current)
        laplacian_psi = ifft2(-self.k2 * psi_hat)

        # Hamiltonian (optional)
        if self.p.use_hamiltonian:
            H_kinetic = -(self.p.hbar**2 / (2 * self.p.m)) * laplacian_psi
            rho = np.abs(psi_current)**2
            H_nonlinear = self.p.g * rho * psi_current
            H_psi = H_kinetic + H_nonlinear
        else:
            # No Hamiltonian: pure NS + memory
            H_psi = np.zeros_like(psi_current, dtype=complex)

        # Coherence-dependent viscosity
        if self.agency is not None and 'C_phi' in locals():
            pass  # C_phi already computed in agency block
        else:
            C_phi = self.coherence(psi_current)
        # ---- Agency-modulated viscosity ----
        if self.agency is not None and action is not None:
            if action == "PAUSE":
                nu_eff = self.p.nu * 5.0
            elif action == "REFRAME":
                nu_eff = self.p.nu * 2.0
            else:
                if self.p.coherence_stabilization == 'adaptive_viscosity':
                    nu_eff = self.adaptive_viscosity(C_phi)
                else:
                    nu_eff = self.p.nu
        else:
            if self.p.coherence_stabilization == 'adaptive_viscosity':
                nu_eff = self.adaptive_viscosity(C_phi)
            else:
                nu_eff = self.p.nu

        viscous_damping = nu_eff * laplacian_psi

        # Coherence potential
        if self.p.coherence_stabilization == 'potential':
            V_coh = self.coherence_potential(psi_current)
            H_psi = H_psi + V_coh

        # ---- Agency-gated memory ----
        if self.agency is not None and action is not None:
            if action == "REFUSE":
                mem = np.zeros_like(psi_current)
            else:
                mem = self.memory_term(psi_current)
                if action == "PAUSE":
                    mem *= 0.5
                elif action == "REFRAME":
                    mem *= 0.3
        else:
            mem = self.memory_term(psi_current)
        memory_coupling = (self.p.nu / self.p.tau_m) * mem

        dpsi_dt = (-1j / self.p.hbar) * H_psi + viscous_damping + memory_coupling

        return dpsi_dt, nu_eff
    
    def step_rk4(self, psi: np.ndarray, dt: float) -> np.ndarray:
        """RK4 with optional renormalization"""
        # RK4 stages
        k1, nu1 = self.compute_derivative(psi)
        k2, nu2 = self.compute_derivative(psi + 0.5 * dt * k1)
        k3, nu3 = self.compute_derivative(psi + 0.5 * dt * k2)
        k4, nu4 = self.compute_derivative(psi + dt * k3)
        
        psi_new = psi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Optional renormalization
        if self.p.coherence_stabilization == 'renormalization':
            psi_new = self.renormalize_coherence(psi_new)
        
        self.memory_buffer.appendleft(psi.copy())
        self.nu_eff_last = (nu1 + nu2 + nu3 + nu4) / 4.0
        
        return psi_new

    def step_with_agency(self, psi: np.ndarray, dt: float) -> np.ndarray:
        if self.agency is None:
            return self.step_rk4(psi, dt)

        A = self.tension_A(psi)
        C_phi = self.coherence(psi)
        A_hat = A * (1.0 + self.p.agency_prediction_factor * (1.0 - C_phi))

        action = self.agency.decide(A, A_hat)
        self._last_action = action

        if action == "REFUSE":
            return psi

        dt_eff = dt
        psi_mod = psi

        if action == "PAUSE":
            dt_eff = 0.5 * dt
        elif action == "REFRAME":
            ratio = min(1.0, max(0.0, (A_hat - self.p.A_pause) / (self.p.A_refuse - self.p.A_pause + 1e-12)))
            damp = 1.0 - 0.3 * ratio
            psi_mod = psi * damp

        psi_new = self.step_rk4(psi_mod, dt_eff)
        return psi_new
    
    def step(self, psi: np.ndarray, dt: float) -> np.ndarray:
        if self.agency is not None:
            return self.step_with_agency(psi, dt)
        return self.step_rk4(psi, dt)
    
    def diagnose(self, psi: np.ndarray, t: float):
        """Record diagnostics"""
        u, v = self.velocity(psi)
        omega = self.vorticity(u, v)
        rho = np.abs(psi)**2
        if self.p.use_hamiltonian:
            Q = self.quantum_potential(psi)
            self.history['quantum_pressure'].append(float(np.mean(np.abs(Q))))
        else:
            self.history['quantum_pressure'].append(0.0)
        self.history['t'].append(t)
        self.history['coherence'].append(self.coherence(psi))
        self.history['energy'].append(self.energy(psi))
        self.history['enstrophy'].append(self.enstrophy(psi))
        self.history['max_vorticity'].append(float(np.max(np.abs(omega))))
        self.history['rho_min'].append(float(np.min(rho)))
        self.history['rho_max'].append(float(np.max(rho)))
        self.history['nu_effective'].append(self.nu_eff_last)
        self.history['tension'].append(self.tension_A(psi))
        if hasattr(self, "_last_action"):
            self.history['agency_action'].append(self._last_action)
        else:
            self.history['agency_action'].append("NONE")
    
    def run(self, psi0: np.ndarray, T: float, dt: float, 
            diagnose_every: int = 10) -> np.ndarray:
        """Run simulation"""
        psi = psi0.copy()
        n_steps = int(T / dt)
        
        print(f"Running {n_steps} steps with dt={dt}, kernel={self.p.kernel_type}")
        print(f"Domain: {self.p.N}x{self.p.N}, L={self.p.L}")
        print(f"ν={self.p.nu}, ℏ={self.p.hbar}, τ_m={self.p.tau_m}")
        print(f"Stabilization: {self.p.coherence_stabilization}")
        print(f"Hamiltonian: {'ENABLED' if self.p.use_hamiltonian else 'DISABLED'}")
        if self.agency is not None:
            print(f"Agency: ENABLED (A_pause={self.p.A_pause:.2e}, A_refuse={self.p.A_refuse:.2e}, A_reframe={self.p.A_reframe:.2e})")
        else:
            print("Agency: DISABLED")
        print("-" * 60)
        
        for step in range(n_steps):
            t = step * dt
            
            if step % diagnose_every == 0:
                self.diagnose(psi, t)
                
                if step % (max(1, n_steps // 10)) == 0:
                    C = self.history['coherence'][-1]
                    E = self.history['energy'][-1]
                    nu_e = self.history['nu_effective'][-1]
                    print(f"t={t:6.2f} | C_φ={C:.4f} | E={E:.6e} | ν_eff={nu_e:.4f}")
            
            psi = self.step(psi, dt)
            
            if np.max(np.abs(psi)) > 1e6 or np.isnan(psi).any():
                print(f"\n⚠️  Instability at t={t:.2f}")
                break
        
        print("-" * 60)
        print(f"✓ Simulation complete")
        print(f"  Final coherence: {self.history['coherence'][-1]:.4f}")
        print(f"  Min coherence: {min(self.history['coherence']):.4f}")
        print(f"  Max ν_eff: {max(self.history['nu_effective']):.4f}")
        
        return psi
    
    def plot_diagnostics(self, save_path: str = None):
        """Plot diagnostics"""
        fig, axes = plt.subplots(3, 2, figsize=(14, 12))
        t = np.array(self.history['t'])
        
        # Coherence
        ax = axes[0, 0]
        ax.plot(t, self.history['coherence'], 'b-', linewidth=2)
        ax.axhline(self.p.C_crit, color='r', linestyle='--', alpha=0.5, label='$C_\\phi^c$')
        ax.set_xlabel('Time')
        ax.set_ylabel('Coherence $C_\\phi$')
        ax.set_title('Phase Coherence Evolution')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Energy
        ax = axes[0, 1]
        ax.plot(t, self.history['energy'], 'g-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy $E$')
        ax.set_title('Kinetic Energy')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # Enstrophy
        ax = axes[1, 0]
        ax.plot(t, self.history['enstrophy'], 'm-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel('Enstrophy $Z$')
        ax.set_title('Vorticity Variance')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # Effective viscosity
        ax = axes[1, 1]
        ax.plot(t, self.history['nu_effective'], 'purple', linewidth=2)
        ax.axhline(self.p.nu, color='k', linestyle='--', alpha=0.5, label='Base $\\nu$')
        ax.set_xlabel('Time')
        ax.set_ylabel('$\\nu_{eff}$')
        ax.set_title('Adaptive Viscosity')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Quantum pressure
        ax = axes[2, 0]
        ax.plot(t, self.history['quantum_pressure'], 'c-', linewidth=2)
        ax.set_xlabel('Time')
        ax.set_ylabel(r'$\langle|Q|\rangle$')
        ax.set_title('Quantum Pressure')
        ax.set_yscale('log')
        ax.grid(alpha=0.3)
        
        # Phase diagram
        ax = axes[2, 1]
        scatter = ax.scatter(self.history['coherence'], self.history['max_vorticity'], 
                           c=t, cmap='viridis', alpha=0.6, s=20)
        ax.set_xlabel('Coherence $C_\\phi$')
        ax.set_ylabel(r'$\max|\omega|$')
        ax.set_title('Coherence-Regularity Correlation')
        ax.grid(alpha=0.3)
        plt.colorbar(scatter, ax=ax, label='Time')
        
        plt.suptitle(f'NS-QFCA Diagnostics ({self.p.coherence_stabilization})', 
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Diagnostics saved to {save_path}")
        
        plt.show()