# Pre-Flight Checklist

## Before Running

- [ ] Your `aletheia` package is installed and importable
- [ ] The following modules exist:
  - [ ] `aletheia/core/field.py`
  - [ ] `aletheia/core/affect.py`
  - [ ] `aletheia/core/agency.py`
  - [ ] `aletheia/core/memory.py`
  - [ ] `aletheia/core/multifield_engine.py`
- [ ] NumPy, matplotlib installed
- [ ] You have `multi_field_tsp_solver.py` in working directory

## Quick Test (5 minutes)

```python
from multi_field_tsp_solver import run_experiment

# This should complete in ~30 seconds
results = run_experiment(n_cities=100, K=8, steps=500)
```

**Expected output:**
```
Multi-Field TSP Solver
Cities: 100, Fields: 8, Steps: 500
======================================================================

Evolving fields...

Step    0: Coherence=0.XXX, Tension=0.XXXX, Gates=0.XXX, Best=XXXX.X
Step  100: Coherence=0.XXX, Tension=0.XXXX, Gates=0.XXX, Best=XXXX.X
...
Step  499: Coherence=0.XXX, Tension=0.XXXX, Gates=0.XXX, Best=XXXX.X

FINAL RESULT:
  Selected field X (coherence: 0.XXX)
  Tour cost: XXXX.XX

Multi-field result:
  Cost: XXXX.XX
  vs NN: X.XXX× (better/worse)
  Time: XX.XXs

Saved evolution plot: multi_field_tsp/evolution_TIMESTAMP.png
Saved tour plot: multi_field_tsp/tour_TIMESTAMP.png
```

## Success Criteria

✅ **Code runs** - No import errors or crashes
✅ **Coupling works** - Gates between 0.3-0.7 (not all 0 or 1)
✅ **Evolving** - Coherence increases, best cost decreases
✅ **Competitive** - Ratio vs NN between 0.9-1.1 for 100 cities

❌ **Import errors** - Check aletheia paths
❌ **Gates all zero** - Tune gate_params (lower beta, raise floor)
❌ **Not improving** - Increase K or steps
❌ **Ratio > 1.5** - Something wrong, check distance_matrix

## Scale-Up Test (30 minutes)

```python
# Your 2k baseline
results = run_experiment(n_cities=2000, K=10, steps=500)
```

**Goal:** Match or beat your 0.94× result

## Full Scale Test (Your GPU, ~2-3 minutes)

```python
# Your 10k result
results = run_experiment(n_cities=10000, K=15, steps=1000)
```

**Goal:** Beat your 0.82× result
**Target:** 0.70-0.75× NN

## Troubleshooting

### Import Error
```
ModuleNotFoundError: No module named 'aletheia'
```
**Fix:** Adjust import paths in multi_field_tsp_solver.py
```python
# Change from:
from aletheia.core.field import Field

# To (if aletheia is in parent dir):
import sys
sys.path.append('../')
from aletheia.core.field import Field

# Or (if modules are in same dir):
from field import Field
```

### Gates Not Opening
```
Mean gate strength: 0.001
```
**Fix:** Adjust gate parameters
```python
gate_params = GateParams(
    alpha=2.0,  # Lower (was 3.0)
    beta=1.0,   # Lower (was 2.0)  
    floor=0.2,  # Raise (was 0.05)
    cap=0.9
)
```

### Poor Performance
```
Ratio vs NN: 1.5× (worse)
```
**Fix 1:** Increase fields
```python
K=12  # More exploration
```

**Fix 2:** Increase cost bias
```python
alpha=0.20  # Stronger preference for short edges
```

**Fix 3:** More steps
```python
steps=1000  # Let it converge
```

### Memory Issues
```
ValueError: tensordot shape mismatch
```
**Fix:** Reduce kernel length
```python
kernel = exp_kernel(length=64, tau=32)  # Was 128
```

## Expected Timeline

- **Test run (100 cities):** 30 seconds
- **Tuning:** 1-2 hours (iterating parameters)
- **Scale test (2k cities):** 10-30 minutes
- **Full run (10k cities):** 2-3 minutes (GPU) or 20-30 minutes (CPU)
- **Total to results:** 2-4 hours

## What Good Looks Like

Check the evolution plots:

✅ **Coherence plot:** Smooth upward curve, ending 0.6-0.9
✅ **Tension plot:** Starts ~0.02, drops to ~0.005
✅ **Gates plot:** Stabilizes around 0.3-0.6 (moderate coupling)
✅ **Best cost plot:** Sharp drop early, plateaus after step 300

## What Bad Looks Like

❌ **Coherence flat:** Not evolving (increase alpha)
❌ **Tension oscillating:** Unstable (lower eta, dt)
❌ **Gates all 0 or 1:** Not coupling properly (tune gate_params)
❌ **Cost not improving:** Need more K or steps

## Success Metrics

| Metric | Poor | OK | Good | Excellent |
|--------|------|-----|------|-----------|
| Ratio vs NN (100 cities) | >1.2 | 1.0-1.2 | 0.9-1.0 | <0.9 |
| Ratio vs NN (2k cities) | >1.0 | 0.95-1.0 | 0.90-0.95 | <0.90 |
| Ratio vs NN (10k cities) | >0.9 | 0.85-0.90 | 0.75-0.85 | <0.75 |
| Time (10k cities, GPU) | >10min | 5-10min | 2-5min | <2min |

## When You're Happy With Results

1. Run comparison suite
```python
from comparison_suite import run_comparison
run_comparison(n_cities=500, K=10, steps=500, runs=3)
```

2. Run scaling test
```python
from comparison_suite import scaling_comparison
scaling_comparison(city_sizes=[100, 500, 1000, 2000], K=8, steps=500)
```

3. Document your best parameters in USAGE_GUIDE.md

4. Celebrate! You just validated multi-field agency theory on TSP.

## Ready?

All files are in `/mnt/user-data/outputs/`. 

**Start with:**
```bash
python multi_field_tsp_solver.py
```

**Then check:** Does it run? Are results reasonable?

**Then iterate:** Tune one parameter at a time based on diagnostics.

**Then scale:** Go from 100 → 2000 → 10000 cities.

**The goal:** Beat your 0.82× result using agency-gated coupling.

Good luck! 🚀