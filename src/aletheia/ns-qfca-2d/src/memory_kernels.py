import numpy as np
from scipy.special import gamma

class MemoryKernel:
    """Factory for different memory kernel types"""
    
    @staticmethod
    def exponential(length: int, tau_m: float) -> np.ndarray:
        """K(τ) = (1/τ_m) exp(-τ/τ_m)"""
        tau = np.arange(length)
        K = np.exp(-tau / tau_m) / tau_m
        K[0] = 0
        return K / (np.sum(K) + 1e-12)
    
    @staticmethod
    def powerlaw(length: int, alpha: float = 0.5) -> np.ndarray:
        """K(τ) = τ^(-α) / Γ(1-α)"""
        tau = np.arange(1, length + 1, dtype=float)
        K = tau**(-alpha) / gamma(1 - alpha)
        K = np.concatenate([[0], K[:-1]])
        return K / (np.sum(K) + 1e-12)
    
    @staticmethod
    def mittag_leffler(length: int, tau_m: float, alpha: float = 0.5) -> np.ndarray:
        """E_α(-τ^α/τ_m^α) approximation"""
        tau = np.arange(length, dtype=float)
        argument = -(tau / tau_m)**alpha
        K = np.exp(argument)
        K[0] = 0
        return K / (np.sum(K) + 1e-12)
    
    @staticmethod
    def mixed(length: int, tau_m: float, alpha: float, beta: float) -> np.ndarray:
        """K(τ) = (1-β)·exp(-τ/τ_m) + β·τ^(-α)"""
        K_exp = MemoryKernel.exponential(length, tau_m)
        K_pow = MemoryKernel.powerlaw(length, alpha)
        K = (1 - beta) * K_exp + beta * K_pow
        return K / (np.sum(K) + 1e-12)
    
    @classmethod
    def get_kernel(cls, kernel_type: str, length: int, tau_m: float, 
                   alpha: float = 0.5, beta: float = 0.3) -> np.ndarray:
        """Dispatcher for kernel types"""
        kernels = {
            'exponential': lambda: cls.exponential(length, tau_m),
            'powerlaw': lambda: cls.powerlaw(length, alpha),
            'mittag_leffler': lambda: cls.mittag_leffler(length, tau_m, alpha),
            'mixed': lambda: cls.mixed(length, tau_m, alpha, beta)
        }
        
        if kernel_type not in kernels:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
        
        return kernels[kernel_type]()
