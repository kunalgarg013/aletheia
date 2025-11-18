# Multi-Field TSP: Quick Start Guide

## What This Does

Uses your **actual** multi-field engine (`multifield_engine.py`) to solve TSP with:
- K independent fields exploring solution space
- Agency-gated coupling based on coherence and tension
- Retrocausal memory kernels for non-Markovian dynamics
- Adaptive gates that open/close based on field similarity

## File Structure

```
multi_field_tsp_solver.py  ← Main implementation

Imports from your aletheia package:
├── aletheia/core/field.py          ← Complex field evolution
├── aletheia/core/affect.py         ← Tension metric
├── aletheia/core/agency.py         ← Agency decisions
├── aletheia/core/memory.py         ← Exponential kernels
└── aletheia/core/multifield_engine.py  ← Your actual engine
```

## Quick Start

```python
from multi_field_tsp_solver import run_experiment

# Small test (30 seconds)
results = run_experiment(n_cities=100, K=8, steps=500)

# Your 2k scale (~2-5 minutes)
results = run_experiment(n_cities=2000, K=10, steps=500)

# Your 10k scale (~10-20 minutes)
results = run_experiment(n_cities=10000, K=15, steps=1000)
```

## Key Parameters

### Number of Fields (K)
```python
K=5    # Few fields, fast, less exploration
K=10   # Balanced (recommended for 1k-5k cities)
K=15   # Many fields, slow, thorough exploration
K=20   # Maximum useful (diminishing returns after this)
```

**Rule of thumb:** K ≈ sqrt(n_cities) / 10

### Evolution Steps
```python
steps=300   # Quick test
steps=500   # Standard (recommended)
steps=1000  # Thorough (for large problems)
steps=2000  # Overkill (won't improve much)
```

**Convergence:** Watch the "best_costs" plot. If it's still dropping at the end, increase steps.

### Cost Bias (alpha)
```python
alpha=0.10  # Weak bias toward short edges
alpha=0.15  # Balanced (recommended)
alpha=0.20  # Strong bias (may get stuck)
```

**Effect:** Higher alpha makes fields prefer short edges more aggressively. Too high → gets stuck in local minima.

### Memory Timescale (tau)
```python
tau=16   # Short memory (more exploration)
tau=32   # Balanced (recommended)
tau=64   # Long memory (more exploitation)
```

**Effect:** Longer memory → fields remember further back → smoother but slower adaptation.

## Gate Parameters

```python
from aletheia.core.multifield_engine import GateParams

# Conservative coupling (fields mostly independent)
gate_params = GateParams(
    alpha=2.0,   # Coherence sharpness
    beta=1.0,    # Affect penalty
    floor=0.01,  # Min gate strength
    cap=0.9      # Max gate strength
)

# Aggressive coupling (fields strongly influence each other)
gate_params = GateParams(
    alpha=5.0,   # Higher → gates open only for high coherence
    beta=5.0,    # Higher → gates close for dissimilar fields
    floor=0.05,  
    cap=0.95
)

# Then pass to solver:
results = multi_field_tsp_solve(
    distance_matrix, 
    K=10, 
    gate_params=gate_params
)
```

## Understanding the Output

### Console Output
```
Step  100: Coherence=0.543, Tension=0.0234, Gates=0.456, Best=1234.5
         │            │             │            │          │
         └─ Progress  │             │            │          └─ Best tour so far
                      │             │            └─ Mean coupling strength
                      │             └─ Mean field tension (lower = calmer)
                      └─ Mean field coherence (higher = more structured)
```

### What To Look For

**Good run:**
- Coherence increases steadily
- Tension decreases over time
- Gates stabilize around 0.3-0.7 (moderate coupling)
- Best cost improves rapidly early, then plateaus

**Bad run:**
- Coherence stuck low (<0.3)
- Tension oscillating wildly
- Gates all near 0 (no coupling) or 1 (over-coupling)
- Best cost not improving after step 200

## Tuning Strategy

### 1. Start Conservative
```python
# Your first run
results = run_experiment(
    n_cities=100,
    K=8,
    steps=500
)
```

Check: Did it beat nearest neighbor? What's the ratio?

### 2. Increase Fields If Quality Poor
```python
# If ratio > 1.2 (worse than NN), try more fields
results = run_experiment(
    n_cities=100,
    K=12,    # More exploration
    steps=500
)
```

### 3. Tune Gates If Coupling Not Working
```python
# If "Gates" staying near 0 or 1, adjust parameters
gate_params = GateParams(
    alpha=3.0,  # Try different values
    beta=3.0,
    floor=0.1,  # Raise floor if gates too closed
    cap=0.8     # Lower cap if gates too open
)
```

### 4. Scale Up Gradually
```python
# Once tuned on 100 cities, scale:
n_cities = [100, 500, 1000, 2000, 5000, 10000]
for n in n_cities:
    K = max(8, int(np.sqrt(n) / 10))
    steps = 500 if n < 2000 else 1000
    results = run_experiment(n, K, steps)
```

## Ablation Studies

### Test Agency Effect
```python
# Run with agency
results_with = multi_field_tsp_solve(distance_matrix, K=10)

# Run without (pass agency_hook=None to engine)
# Need to modify code slightly to disable agency
```

### Test Coupling Effect
```python
# Run with coupling
lambdas[i,j] = 0.5  # Normal

# Run without coupling
lambdas[:,:] = 0.0  # Fields independent
```

### Test Memory Effect
```python
# With memory
kernel = exp_kernel(length=128, tau=32)

# Without memory
kernel = None  # Pass to engine.step()
```

## Expected Performance

Based on your existing results:

| Cities | K | Steps | Expected Ratio | Time |
|--------|---|-------|----------------|------|
| 100 | 8 | 500 | 0.95-1.10× | 30s |
| 500 | 10 | 500 | 0.90-1.00× | 2min |
| 2000 | 10 | 500 | 0.85-0.95× | 10min |
| 10000 | 15 | 1000 | 0.80-0.90× | 2-3min (GPU) |

**Key insight from your results:** Performance IMPROVES with scale.
- 2k cities: 0.94× NN (6% better)
- 10k cities: 0.82× NN (18% better)

This is unusual and suggests the multi-field approach exploits large-scale structure.

## Debugging Tips

### Field Not Evolving
**Symptom:** Coherence stays constant, best cost doesn't change
**Fix:** 
- Increase alpha (stronger cost bias)
- Check that distance_matrix is symmetric
- Verify TSPField.step() is being called

### Gates All Zero
**Symptom:** Mean gate strength near 0, no coupling happening
**Fix:**
- Lower gate_params.beta (less penalty for dissimilarity)
- Raise gate_params.floor (minimum coupling)
- Check tension values aren't too different between fields

### Memory Errors
**Symptom:** Crash in tensordot or shape mismatch
**Fix:**
- Check kernel length matches memory_history length
- Verify shape compatibility in TSPField.step()
- Try shorter kernel (length=64 instead of 128)

### Too Slow
**Symptom:** Taking forever on large problems
**Fix:**
- Reduce K (fewer fields)
- Reduce steps
- Reduce boundary density in create_tsp_boundaries()
- Use GPU if available (fields are embarassingly parallel)

## What To Expect

### First Run (100 cities, K=8):
- Takes ~30 seconds
- Might be slightly worse than NN (ratio ~1.05)
- This is fine! You're establishing baseline.

### After Tuning:
- Ratio should improve to 0.95-1.00
- Coupling should stabilize (gates 0.3-0.7)
- Coherence should grow smoothly

### At Scale (2k-10k cities):
- **This is where multi-field shines**
- Should see ratio < 1.0 (beating NN)
- Improvement increases with scale (your observation!)

## Next Steps

1. **Run the code as-is** on 100 cities
2. **Check the plots** (evolution and tour)
3. **Tune one parameter at a time** based on diagnostics
4. **Scale up gradually** once it's working on small problems
5. **Compare to your single-field results** (your retro_tsp code)

The goal: beat your existing 0.82× result on 10k cities using agency-gated coupling!

## Questions?

- **Gates not coupling?** → Lower beta, raise floor
- **Too slow?** → Reduce K or steps
- **Quality poor?** → Increase K or alpha
- **Memory issues?** → Reduce kernel length
- **Not beating NN?** → Try larger problems (scales better)

**Most important:** Look at the evolution plots. They tell you what's happening.