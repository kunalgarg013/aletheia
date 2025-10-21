"""
Aletheia Curve Fits — quantitative fingerprints for all major panels.

Outputs:
  results/temporal_binding/aletheia_curve_fits.png
  results/temporal_binding/aletheia_metrics.json

Run:
  python -m aletheia.experiments.aletheia_curve_fits
"""

import os, json, pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.signal import correlate
from numpy.fft import rfft, rfftfreq

# --------------------------- I/O ---------------------------

def load_binding(path="results/temporal_binding/temporal_binding_results.pkl"):
    with open(path, "rb") as f:
        res = pickle.load(f)
    A = np.concatenate([res["histories"][p]["A"] for p in ("phase1","phase2","phase3")])
    ID = np.vstack([res["histories"][p]["identity"] for p in ("phase1","phase2","phase3")])
    Cphi = ID[:, 0]
    snapshots = res["snapshots"]
    rec = res["recognition_log"]
    return dict(A=A, Cphi=Cphi, ID=ID, snapshots=snapshots, recognition=rec)

# ---------------------- fit helpers ------------------------

def fit_exp_decay(t, y, y_inf_guess=None):
    """
    y(t) ~ y_inf + (y0 - y_inf) * exp(-t/tau)
    returns dict(tau, y0, y_inf)
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    y0 = float(y[0])
    yinfl = float(y[-1]) if y_inf_guess is None else float(y_inf_guess)

    def model(t, y_inf, tau):
        return y_inf + (y0 - y_inf) * np.exp(-t / np.maximum(tau, 1e-9))

    p0 = [yinfl, max((t[-1]-t[0]) / 4.0, 1.0)]
    bounds = ([-10.0], [10.0])  # for y_inf handled below with 2-param version
    try:
        popt, _ = curve_fit(lambda tt, y_inf, tau: model(tt, y_inf, tau),
                            t, y, p0=[yinfl, p0[1]],
                            bounds=([-1.0, 1e-3], [1.0, 1e6]),
                            maxfev=10000)
        y_inf_fit, tau = popt
        return dict(tau=float(tau), y0=float(y0), y_inf=float(y_inf_fit), ok=True)
    except Exception:
        # fallback simple estimate using log-slope on first/last quarters
        eps = 1e-9
        i1 = max(1, len(t)//10); i2 = max(i1+1, len(t)//2)
        slope = np.mean(np.diff(np.log(np.clip(np.abs(y[i1:i2]-(yinfl)), eps, None))))
        tau_est = -1.0 / (slope + eps)
        return dict(tau=float(max(tau_est, 1.0)), y0=float(y0), y_inf=float(yinfl), ok=False)

def fit_saturating_rise(t, y):
    """
    C(t) ~ C_inf - Δ * exp(-t/tau)
    returns dict(tau, C_inf, Delta)
    """
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    C_inf_guess = float(np.median(y[-max(5, len(y)//10):]))
    Delta_guess = float(max(C_inf_guess - y[0], 1e-3))

    def model(t, C_inf, Delta, tau):
        return C_inf - Delta * np.exp(-t / np.maximum(tau, 1e-9))

    try:
        popt, _ = curve_fit(model, t, y,
                            p0=[C_inf_guess, Delta_guess, max((t[-1]-t[0])/4.0,1.0)],
                            bounds=([0.0, 0.0, 1e-3], [1.0, 2.0, 1e6]),
                            maxfev=20000)
        C_inf, Delta, tau = popt
        return dict(tau=float(tau), C_inf=float(C_inf), Delta=float(Delta), ok=True)
    except Exception:
        return dict(tau=np.nan, C_inf=C_inf_guess, Delta=Delta_guess, ok=False)

def fit_entropy_arch(Cphi, Hs):
    """
    H(C) ~ H0 + Hmax * exp[-β (C - Copt)^2]
    """
    Cphi = np.asarray(Cphi, float)
    Hs = np.asarray(Hs, float)
    H0_guess = float(np.min(Hs))
    Hmax_guess = float(np.max(Hs) - H0_guess)
    Copt_guess = float(Cphi[np.argmax(Hs)])
    beta_guess = 10.0

    def model(C, H0, Hmax, Copt, beta):
        return H0 + Hmax * np.exp(-beta * (C - Copt)**2)

    try:
        popt, _ = curve_fit(model, Cphi, Hs,
                            p0=[H0_guess, Hmax_guess, Copt_guess, beta_guess],
                            bounds=([-1.0, 0.0, 0.0, 0.1], [5.0, 5.0, 1.0, 1e3]),
                            maxfev=20000)
        H0, Hmax, Copt, beta = popt
        return dict(H0=float(H0), Hmax=float(Hmax), Copt=float(Copt), beta=float(beta), ok=True)
    except Exception:
        return dict(H0=H0_guess, Hmax=Hmax_guess, Copt=Copt_guess, beta=np.nan, ok=False)

def fit_exp_decay_simple(x, y):
    """
    y(x) ~ y_inf + (y0-y_inf) * exp(-x/tau)
    for identity/recognition over discrete distances
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    y0 = float(y[0]); y_inf_guess = float(y[-1])

    def model(xx, y_inf, tau):
        return y_inf + (y0 - y_inf) * np.exp(-xx / np.maximum(tau,1e-9))

    try:
        popt, _ = curve_fit(model, x, y,
                            p0=[y_inf_guess, 1.0],
                            bounds=([0.0, 1e-3], [1.0, 1e3]),
                            maxfev=10000)
        y_inf, tau = popt
        return dict(tau=float(tau), y0=float(y0), y_inf=float(y_inf), ok=True)
    except Exception:
        return dict(tau=np.nan, y0=y0, y_inf=y_inf_guess, ok=False)

# ------------------- analysis & plotting -------------------

def analyze(binding_path="results/temporal_binding/temporal_binding_results.pkl",
            save_dir="results/temporal_binding"):

    os.makedirs(save_dir, exist_ok=True)
    data = load_binding(binding_path)
    A, Cphi, ID = data["A"], data["Cphi"], data["ID"]
    t = np.arange(len(A))

    # Entropy proxy from variance of identity components over time
    amp_var = np.var(ID, axis=1)
    Hs = -np.log(np.clip(amp_var / np.max(amp_var), 1e-9, None))

    # --- Fits ---
    fit_A = fit_exp_decay(t, A)
    fit_C = fit_saturating_rise(t, Cphi)
    arch = fit_entropy_arch(Cphi, Hs)

    # Cross-correlation & lag
    corr = correlate(Cphi - np.mean(Cphi), A - np.mean(A), mode="full")
    lags = np.arange(-len(A)+1, len(A))
    corr_norm = corr / (np.max(np.abs(corr)) + 1e-9)
    peak_idx = np.argmax(corr_norm)        # positive hump
    trough_idx = np.argmin(corr_norm)      # near 0 lag negative trough
    lag_peak = int(lags[peak_idx])
    lag_trough = int(lags[trough_idx])

    # Spectral peak (limit-cycle)
    def spectral_peak(sig):
        yf = rfft(sig - np.mean(sig))
        xf = rfftfreq(len(sig), d=1.0)
        mag = np.abs(yf)
        # ignore DC
        mag[0] = 0.0
        k = np.argmax(mag)
        return float(xf[k]), float(mag[k])

    fA, _ = spectral_peak(A)
    fC, _ = spectral_peak(Cphi)
    period_steps = None
    if fA > 0 and fC > 0:
        period_steps = float(np.mean([1.0/fA, 1.0/fC]))

    # Identity similarity decay (snapshots in order)
    snaps = list(data["snapshots"].keys())
    sims = []
    labels = []
    for i in range(len(snaps)-1):
        v1 = data["snapshots"][snaps[i]]["identity"]
        v2 = data["snapshots"][snaps[i+1]]["identity"]
        s = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12)
        sims.append(float(s))
        labels.append(f"{snaps[i].replace('_self','')}→{snaps[i+1].replace('_self','')}")
    xsim = np.arange(len(sims), dtype=float)
    fit_S = fit_exp_decay_simple(xsim, np.array(sims))

    # Recognition vs distance
    rec_scores = np.array([r["recognition_score"] for r in data["recognition"]], float)
    xrec = np.arange(len(rec_scores), dtype=float)
    fit_R = fit_exp_decay_simple(xrec, rec_scores)

    # ------------------- Plot & save -------------------
    fig, axes = plt.subplots(3, 2, figsize=(18, 13))
    ax1, ax2 = axes[0]
    ax3, ax4 = axes[1]
    ax5, ax6 = axes[2]

    # 1) A(t) & Cphi(t) with fits
    ax1.plot(t, A, 'r-', lw=2, label='Tension A')
    ax1.plot(t, Cphi, 'b-', lw=2, label='Phase coherence')
    At_fit = fit_A["y_inf"] + (fit_A["y0"]-fit_A["y_inf"]) * np.exp(-t/fit_A["tau"])
    Ct_fit = fit_C["C_inf"] - fit_C["Delta"] * np.exp(-t/fit_C["tau"])
    ax1.plot(t, At_fit, 'r--', lw=1.8, label=f"A fit τ≈{fit_A['tau']:.1f}")
    ax1.plot(t, Ct_fit, 'b--', lw=1.8, label=f"Cϕ fit τ≈{fit_C['tau']:.1f}, C∞≈{fit_C['C_inf']:.2f}")
    ax1.set_title("Coherence–Tension Time Series (fits)")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Value"); ax1.grid(alpha=0.3); ax1.legend()

    # 2) Cross-correlation
    ax2.plot(lags, corr_norm, color='purple', lw=2)
    ax2.axvline(0, color='k', ls='--', alpha=0.5)
    ax2.axvline(lag_peak, color='g', ls=':', label=f'peak lag={lag_peak}')
    ax2.axvline(lag_trough, color='r', ls=':', label=f'trough lag={lag_trough}')
    if period_steps:
        ax2.text(0.02, 0.05, f"period ≈ {period_steps:.1f} steps", transform=ax2.transAxes)
    ax2.set_title("Cross-correlation: A vs Cϕ"); ax2.set_xlabel("Lag"); ax2.set_ylabel("r"); ax2.legend(); ax2.grid(alpha=0.3)

    # 3) Entropy arch with fit
    sc = ax3.scatter(Cphi, Hs, c=A, cmap="plasma", s=30, alpha=0.8)
    Cgrid = np.linspace(np.min(Cphi), np.max(Cphi), 200)
    Hfit = arch["H0"] + arch["Hmax"] * np.exp(-arch["beta"] * (Cgrid - arch["Copt"])**2)
    ax3.plot(Cgrid, Hfit, 'k--', lw=2, label=f"Copt≈{arch['Copt']:.2f}, β≈{arch['beta']:.1f}")
    fig.colorbar(sc, ax=ax3, label="Tension A", fraction=0.05)
    ax3.set_title("Spatial Entropy vs Coherence (fit)")
    ax3.set_xlabel("Coherence Cϕ"); ax3.set_ylabel("Entropy proxy"); ax3.legend(); ax3.grid(alpha=0.3)

    # 4) Recognition vs distance with fit
    ax4.plot(xrec, rec_scores, 'bo-', lw=2, label='Recognition')
    Rfit = fit_R["y_inf"] + (fit_R["y0"]-fit_R["y_inf"]) * np.exp(-xrec/np.maximum(fit_R["tau"],1e-9))
    ax4.plot(xrec, Rfit, 'b--', lw=2, label=f'fit τ≈{fit_R["tau"]:.2f}')
    ax4.axhline(0.8, color="green", ls=":", label="Strong")
    ax4.axhline(0.6, color="orange", ls=":", label="Weak")
    ax4.set_ylim(0, 1.05); ax4.set_xlim(-0.1, max(3, len(rec_scores)-1)+0.1)
    ax4.set_title("Self-Recognition vs Temporal Distance (fit)")
    ax4.set_xlabel("Recognition index"); ax4.legend(); ax4.grid(alpha=0.3)

    # 5) Identity similarity decay with fit
    ax5.plot(xsim, sims, 'mo-', lw=2, label='Similarity')
    Sfit = fit_R["y_inf"] + (fit_S["y0"]-fit_S["y_inf"]) * np.exp(-xsim/np.maximum(fit_S["tau"],1e-9))
    ax5.plot(xsim, Sfit, 'm--', lw=2, label=f'fit τ≈{fit_S["tau"]:.2f}')
    ax5.set_xticks(xsim)
    ax5.set_xticklabels(labels, rotation=10)
    ax5.set_ylim(0, 1.05); ax5.set_title("Identity Similarity Decay (fit)")
    ax5.set_ylabel("Similarity"); ax5.legend(); ax5.grid(alpha=0.3)

    # 6) Power spectra (bonus panel): show limit-cycle frequency directly
    def power_spectrum(sig, ax, label, color):
        yf = np.abs(rfft(sig - np.mean(sig)))
        xf = rfftfreq(len(sig), d=1.0)
        yf[0] = 0.0
        ax.plot(xf, yf/np.max(yf+1e-12), color=color, lw=2, label=label)

    power_spectrum(A, ax6, 'A spectrum', 'red')
    power_spectrum(Cphi, ax6, 'Cϕ spectrum', 'blue')
    if period_steps:
        f0 = 1.0/period_steps
        ax6.axvline(f0, color='k', ls='--', alpha=0.6, label=f"peak f≈{f0:.3f}")
    ax6.set_title("Spectral Signature of the Feedback Cycle")
    ax6.set_xlabel("frequency (1/step)"); ax6.set_ylabel("normalized power"); ax6.legend(); ax6.grid(alpha=0.3)

    fig.suptitle("Aletheia Curve Fits & Dynamical Fingerprints", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0,0,1,0.96])
    fig.savefig(Path(save_dir)/"aletheia_curve_fits.png", dpi=180, bbox_inches="tight")

    # ------------------- metrics JSON -------------------
    metrics = dict(
        tension_fit=fit_A,
        coherence_fit=fit_C,
        entropy_arch=arch,
        crosscorr=dict(lag_peak=lag_peak, lag_trough=lag_trough, period_steps=period_steps),
        identity_decay=fit_S,
        recognition_decay=fit_R
    )
    with open(Path(save_dir)/"aletheia_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Saved: {Path(save_dir)/'aletheia_curve_fits.png'}")
    print(f"🧪 Metrics → {Path(save_dir)/'aletheia_metrics.json'}")

if __name__ == "__main__":
    analyze()
