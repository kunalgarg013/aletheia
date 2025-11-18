import numpy as np
from scipy.fft import fft2, ifft2

def taylor_green_vortex(solver, amplitude: float = 1.0) -> np.ndarray:
    """Taylor-Green vortex initial condition"""
    S = -amplitude * np.sin(solver.X) * np.sin(solver.Y) * solver.p.hbar
    rho = np.ones_like(S)
    psi = np.sqrt(rho) * np.exp(1j * S / solver.p.hbar)
    return psi

def kelvin_helmholtz(solver, delta: float = 0.05, epsilon: float = 0.01) -> np.ndarray:
    """Kelvin-Helmholtz shear layer"""
    u_shear = np.tanh((solver.Y - np.pi) / delta)
    u_pert = epsilon * np.sin(2 * solver.X) * np.exp(-(solver.Y - np.pi)**2 / delta**2)
    
    S_base = delta * np.log(np.cosh((solver.Y - np.pi) / delta))
    S_pert = -epsilon * delta * np.cos(2 * solver.X) * np.exp(-(solver.Y - np.pi)**2 / delta**2)
    S = (S_base + S_pert) * solver.p.hbar * solver.p.m
    
    rho = np.ones_like(S)
    psi = np.sqrt(rho) * np.exp(1j * S / solver.p.hbar)
    return psi

def random_vorticity(solver, k_peak: int = 8, energy: float = 1.0) -> np.ndarray:
    """Random vorticity field"""
    omega_hat = (np.random.randn(solver.p.N, solver.p.N) + 
                 1j * np.random.randn(solver.p.N, solver.p.N))
    
    k_mag = np.sqrt(solver.k2)
    spectrum = k_mag**4 * np.exp(-k_mag**2 / k_peak**2)
    omega_hat *= np.sqrt(spectrum)
    omega_hat *= np.sqrt(energy / (np.sum(np.abs(omega_hat)**2) + 1e-10))
    
    omega = np.real(ifft2(omega_hat))
    psi_stream_hat = fft2(omega) / (-solver.k2)
    psi_stream_hat[0, 0] = 0
    psi_stream = np.real(ifft2(psi_stream_hat))
    
    S = psi_stream * solver.p.hbar * solver.p.m
    rho = np.ones_like(S)
    psi = np.sqrt(rho) * np.exp(1j * S / solver.p.hbar)
    return psi
