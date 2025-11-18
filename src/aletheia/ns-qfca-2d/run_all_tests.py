#!/usr/bin/env python
"""
Run all NS-QFCA benchmark tests
"""

import sys
import os

# Add src to path
sys.path.append('src')

# Create results directory
os.makedirs('figures', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("\n" + "="*60)
print("NS-QFCA 2D BENCHMARK SUITE")
print("="*60 + "\n")

# Import tests
from tests.test_taylor_green import test_taylor_green
from tests.test_kelvin_helmholtz import test_kelvin_helmholtz
from tests.test_kernel_comparison import test_kernel_comparison

# Run all tests
print("\n🚀 Starting benchmark suite...\n")

try:
    print("\n[1/3] Taylor-Green Vortex (Validation)")
    test_taylor_green()
    
    print("\n[2/3] Kelvin-Helmholtz Instability")
    test_kelvin_helmholtz()
    
    print("\n[3/3] Kernel Comparison")
    test_kernel_comparison()
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*60)
print("✅ ALL TESTS COMPLETE")
print("="*60)
print("\nResults saved in figures/")
print("\nTo view plots:")
print("  python -c 'import matplotlib.pyplot as plt; plt.show()'")
print("\n")
