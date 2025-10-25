import numpy as np
import matplotlib.pyplot as plt
from aletheia.core.multifield_engine import MultiFieldEngine, AdaptiveGates, GateParams
from aletheia.core.field import Field, FieldConfig
from aletheia.core.memory import exp_kernel

# --- dummy setup ---
N = 3
cfgs = [FieldConfig(shape=(8, 8), dt=0.05, seed=10+i) for i in range(N)]
fields = [Field(cfg) for cfg in cfgs]

# random initial psi
for f in fields:
    f.psi = np.random.randn(*f.cfg.shape) + 1j * np.random.randn(*f.cfg.shape)

# boundaries: everyone shares the same subset of 10 elements
shared_idx = np.arange(10)
boundaries = {(i,j):(shared_idx, shared_idx) for i in range(N) for j in range(i+1,N)}
lambdas = np.ones((N,N)) - np.eye(N)
gates = AdaptiveGates(GateParams(alpha=5, beta=5, floor=0.05))
kernel = exp_kernel(length=64, tau=20.0)

engine = MultiFieldEngine(fields, boundaries, lambdas, gates, eta=0.3)

# run a few steps and record coherence evolution
steps = 300
Cbar = []
for t in range(steps):
    diag = engine.step(t, kernel=kernel)
    Cbar.append(engine.mean_coherence())
    if t % 50 == 0:
        print(f"t={t}, ⟨C⟩={engine.mean_coherence():.3f}")

plt.figure(figsize=(8,4))
plt.plot(Cbar, lw=2)
plt.xlabel("step"); plt.ylabel("mean coherence")
plt.title("Multi-field synchronization test")
plt.grid(alpha=0.3)
plt.tight_layout()
# plt.show()
plt.savefig("test_multifield_engine_coherence.png", dpi=150)