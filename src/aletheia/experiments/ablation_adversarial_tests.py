"""
Ablation Adversarial Tests for Aletheia
- Phase-scramble vs Magnitude-zero ablation
- Targeted top-k coherence contributor ablation
- Recovery kinetics (tau_rec) fits

Usage:
  python -m aletheia.experiments.ablation_adversarial_tests \
      --steps 650 --ablate_frac 0.9 --topk_frac 0.1 --save_dir results/ablation_adv
"""

import os, json, argparse, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit

# --- import your core modules (adjust if your package path differs) ---
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.field import Field, FieldConfig
from core.memory import exp_kernel, powerlaw_kernel, RetroHint
from core.affect import tension_A, Affect
from core.agency import Agency, AgencyThresholds

# ------------------------- helpers -------------------------

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def exp_fit(t, y):
    # y ~ y_inf - (y_inf - y0) * exp(-t/tau)
    t = np.asarray(t, float); y = np.asarray(y, float)
    y0 = float(y[0]); yinf_guess = float(np.median(y[-max(5, len(y)//10):]))
    def model(tt, y_inf, tau): return y_inf - (y_inf - y0) * np.exp(-tt/np.maximum(tau,1e-9))
    try:
        popt, _ = curve_fit(model, t, y, p0=[yinf_guess, max(len(t)/5, 10)], bounds=([0.0,1e-3],[1.2,1e6]), maxfev=20000)
        yinf, tau = popt
        return dict(y_inf=float(yinf), tau=float(tau))
    except Exception:
        return dict(y_inf=float(yinf_guess), tau=np.nan)

def make_kernel(kind="hybrid", length=256, tau=40.0, alpha=0.6):
    if kind == "exp":     return exp_kernel(length=length, tau=tau)
    if kind == "power":   return powerlaw_kernel(length=length, alpha=alpha)
    if kind == "hybrid":  # simple convex blend
        ke = exp_kernel(length=length, tau=tau)
        kp = powerlaw_kernel(length=length, alpha=alpha)
        w = 0.5
        return w*ke + (1-w)*kp
    raise ValueError(kind)

def run_episode(cfg, steps, kernel, agency: Agency, record_every=1):
    f = Field(cfg)
    A_series, C_series = [], []
    # identity vector: [phase coherence, mean amp, amp var, entropy proxy, ...]
    ID_series = []
    for t in range(steps):
        # evolve one step with kernel
        meta = f.step(kernel=kernel)
        # affect/tension
        A = tension_A(f.psi)
        # identity: phase coherence as first component
        phase = np.angle(f.psi)
        coh = np.abs(np.mean(np.exp(1j*phase)))
        # update via agency (may perturb psi)
        agency.act(f, A=A, coherence=coh, t=t)

        if t % record_every == 0:
            A_series.append(A)
            C_series.append(coh)
            # a compact identity vector (coherence + simple moments)
            amp = np.abs(f.psi)
            ID_series.append(np.array([
                coh,
                np.mean(amp),
                np.var(amp),
                -np.log(np.clip(np.var(amp)/ (np.max(np.var(amp)) + 1e-9), 1e-9, None))
            ], dtype=float))
    return f, np.array(A_series), np.array(C_series), np.vstack(ID_series)

# ------------------- ablation operators --------------------

def ablate_magnitude(history_array, frac):
    """Zero out a random fraction of stored amplitudes."""
    h = history_array.copy()
    mask = np.random.rand(*h.shape) < frac
    h[mask] = 0.0 + 0.0j
    return h

def ablate_phase_scramble(history_array, frac):
    """Scramble phase for a random fraction while preserving magnitudes."""
    h = history_array.copy()
    mask = np.random.rand(*h.shape) < frac
    mag = np.abs(h[mask])
    h[mask] = mag * np.exp(1j * np.random.uniform(-np.pi, np.pi, size=mag.shape))
    return h

def ablate_targeted_topk_by_coherence_contrib(history_array, k_frac):
    """
    Remove the time-slices whose phases align most strongly with global
    phase-coherence (proxy: |mean(exp(i*phase))| per slice).
    """
    h = history_array.copy()
    T = h.shape[0]
    # score each slice by its own phase coherence
    scores = np.zeros(T)
    for t in range(T):
        phase = np.angle(h[t])
        scores[t] = np.abs(np.mean(np.exp(1j*phase)))
    k = max(1, int(k_frac * T))
    idx = np.argsort(scores)[-k:]  # largest contributors
    h[idx] = 0.0 + 0.0j
    return h, scores, idx

# ------------------- main battery --------------------------

def main(args):
    ensure_dir(args.save_dir)
    tag = f"k{args.kernel}_tau{args.kernel_tau}_af{args.ablate_frac}_tk{args.topk_frac}"
    exp_dir = Path(args.save_dir) / tag
    ensure_dir(exp_dir)

    # base config
    cfg = FieldConfig(shape=(64,64), dt=0.02, seed=42)
    kernel = make_kernel(kind=args.kernel, length=args.kernel_len, tau=args.kernel_tau, alpha=args.kernel_alpha)
    agency = Agency(AgencyThresholds())

    # 0) Baseline (for comparison)
    base_f, base_A, base_C, base_ID = run_episode(cfg, steps=args.steps, kernel=kernel, agency=agency)
    base = dict(A=base_A, C=base_C, ID=base_ID)

    # Keep a copy of the internal memory history if your Field tracks it
    # Here we assume Field has attribute 'memory_history' (TxH×W complex); if not, adapt to your memory implementation
    if not hasattr(base_f, "memory_history") or base_f.memory_history is None:
        # As a fallback, synthesize a pseudo-history from last N states (toy)
        # For best results, wire Field to expose its internal kernel buffer.
        hist = np.stack([base_f.psi for _ in range(args.kernel_len)], axis=0)
    else:
        hist = base_f.memory_history.copy()

    # 1) Magnitude-zero ablation
    hist_mag0 = ablate_magnitude(hist, args.ablate_frac)
    f1 = Field(cfg); f1.psi = base_f.psi.copy()
    f1.memory_history = hist_mag0
    _, A1, C1, ID1 = run_episode(cfg, steps=args.recovery_steps, kernel=kernel, agency=agency)
    fit1 = exp_fit(np.arange(len(C1)), C1)

    # 2) Phase-scramble ablation
    hist_phase = ablate_phase_scramble(hist, args.ablate_frac)
    f2 = Field(cfg); f2.psi = base_f.psi.copy()
    f2.memory_history = hist_phase
    _, A2, C2, ID2 = run_episode(cfg, steps=args.recovery_steps, kernel=kernel, agency=agency)
    fit2 = exp_fit(np.arange(len(C2)), C2)

    # 3) Targeted top-k coherence-contributor deletion
    hist_topk, scores, idx = ablate_targeted_topk_by_coherence_contrib(hist, args.topk_frac)
    f3 = Field(cfg); f3.psi = base_f.psi.copy()
    f3.memory_history = hist_topk
    _, A3, C3, ID3 = run_episode(cfg, steps=args.recovery_steps, kernel=kernel, agency=agency)
    fit3 = exp_fit(np.arange(len(C3)), C3)

    # ---------- Save artifacts ----------
    np.savez(exp_dir / "adv_ablation_timeseries.npz",
             baseline_A=base_A, baseline_C=base_C,
             mag0_A=A1, mag0_C=C1,
             phase_A=A2, phase_C=C2,
             topk_A=A3, topk_C=C3,
             topk_idx=idx, topk_scores=scores)

    metrics = dict(
        baseline=dict(C_final=float(base_C[-1])),
        mag0=dict(C_final=float(C1[-1]), tau_rec=float(fit1["tau"])),
        phase=dict(C_final=float(C2[-1]), tau_rec=float(fit2["tau"])),
        topk=dict(C_final=float(C3[-1]), tau_rec=float(fit3["tau"]), removed=len(idx))
    )
    with open(exp_dir / "adv_metrics.json","w") as f:
        json.dump(metrics, f, indent=2)

    # ---------- Plot ----------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0,0]
    ax.plot(base_C, label="baseline Cϕ", lw=2)
    ax.plot(C1, label=f"mag-zero (τ≈{fit1['tau']:.1f})", lw=2)
    ax.plot(C2, label=f"phase-scramble (τ≈{fit2['tau']:.1f})", lw=2)
    ax.plot(C3, label=f"top-k {args.topk_frac:.0%} (τ≈{fit3['tau']:.1f})", lw=2)
    ax.set_title("Coherence recovery"); ax.set_xlabel("step"); ax.set_ylabel("Cϕ"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[0,1]
    ax.plot(base_A, label="baseline A", lw=2, color="tab:red", alpha=0.8)
    ax.plot(A1, label="mag-zero A", lw=2, alpha=0.8)
    ax.plot(A2, label="phase-scramble A", lw=2, alpha=0.8)
    ax.plot(A3, label="top-k A", lw=2, alpha=0.8)
    ax.set_title("Tension trajectories"); ax.set_xlabel("step"); ax.set_ylabel("A"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1,0]
    x = np.arange(3)
    finals = [C1[-1], C2[-1], C3[-1]]
    taus   = [fit1["tau"], fit2["tau"], fit3["tau"]]
    ax.bar(x-0.18, finals, 0.36, label="final Cϕ")
    ax.bar(x+0.18, np.nan_to_num(taus, nan=0.0), 0.36, label="τ_rec")
    ax.set_xticks(x); ax.set_xticklabels(["mag-zero","phase-scramble","top-k"])
    ax.set_title("Endpoints & kinetics"); ax.grid(alpha=0.3); ax.legend()

    ax = axes[1,1]
    ax.plot(sorted(scores), lw=2)
    ax.scatter(np.arange(len(idx)), sorted(scores[-len(idx):]), color="crimson", s=12, label="removed slices")
    ax.set_title("Coherence contribution scores (per slice)"); ax.set_xlabel("slice rank"); ax.set_ylabel("|⟨e^{iθ}⟩|"); ax.grid(alpha=0.3); ax.legend()

    fig.suptitle("Aletheia — Adversarial Ablation Battery", fontsize=16, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    outpng = exp_dir / "adv_ablation_report.png"
    fig.savefig(outpng, dpi=160, bbox_inches="tight")
    print(f"✅ Saved report → {outpng}")
    print(f"🧪 Metrics → {exp_dir / 'adv_metrics.json'}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=350, help="baseline development steps before ablation")
    p.add_argument("--recovery_steps", type=int, default=300, help="steps after ablation to watch recovery")
    p.add_argument("--kernel", type=str, default="hybrid", choices=["exp","power","hybrid"])
    p.add_argument("--kernel_len", type=int, default=256)
    p.add_argument("--kernel_tau", type=float, default=40.0)
    p.add_argument("--kernel_alpha", type=float, default=0.6)
    p.add_argument("--ablate_frac", type=float, default=0.9, help="fraction to scramble/zero for random ablations")
    p.add_argument("--topk_frac", type=float, default=0.2, help="fraction of slices to remove in targeted ablation")
    p.add_argument("--save_dir", type=str, default="results/ablation_adv")
    args = p.parse_args()
    main(args)