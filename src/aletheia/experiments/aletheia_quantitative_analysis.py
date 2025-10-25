"""
Aletheia Quantitative Analysis — Six-Panel Research Poster
Combines coherence–tension dynamics, entropy relations,
recognition metrics, identity decay, and ablation comparisons.

Usage:
    python -m aletheia.experiments.aletheia_quantitative_analysis
"""

import numpy as np, pickle, json, os, matplotlib.pyplot as plt
from scipy.signal import correlate
from pathlib import Path
from glob import glob

# ------------------------------------------------------------
# Utility loaders
# ------------------------------------------------------------
def load_any(path):
    ext = Path(path).suffix
    if ext == ".pkl":
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ext == ".npz":
        return dict(np.load(path, allow_pickle=True))
    elif ext == ".json":
        with open(path) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file format: {ext}")

# ------------------------------------------------------------
# Core extractors
# ------------------------------------------------------------
def extract_series(results):
    A = np.concatenate([results["histories"][p]["A"]
                        for p in ("phase1","phase2","phase3")])
    ID = np.vstack([results["histories"][p]["identity"]
                    for p in ("phase1","phase2","phase3")])
    coherence = ID[:,0]
    return np.array(A), np.array(coherence), ID

def identity_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2) + 1e-12)

# ------------------------------------------------------------
# Ablation summary loader
# ------------------------------------------------------------
def load_ablation_summaries(ablation_dir="results/ablation"):
    summaries = {}
    if not os.path.exists(ablation_dir):
        return summaries

    for file in sorted(glob(os.path.join(ablation_dir, "ablation_*.npz"))):
        try:
            data = dict(np.load(file, allow_pickle=True))
            label = Path(file).stem.replace("ablation_", "")

            # Unwrap possible object arrays
            def unwrap(x):
                if isinstance(x, np.ndarray) and x.shape == () and isinstance(x.item(), dict):
                    return x.item()
                elif isinstance(x, np.ndarray) and x.dtype == object and len(x) == 1 and isinstance(x[0], dict):
                    return x[0]
                elif isinstance(x, dict):
                    return x
                return None

            sim = unwrap(data.get("similarities"))
            mem = unwrap(data.get("memory"))
            abl = unwrap(data.get("ablation"))

            if sim and mem:
                summaries[label] = {
                    "id_post": sim.get("baseline_to_post", np.nan),
                    "id_rec": sim.get("baseline_to_recovery", np.nan),
                    "gain": sim.get("recovery_gain", np.nan),
                    "mem_post": mem.get("post_ablation", np.nan),
                    "mem_rec": mem.get("recovery", np.nan),
                    "frac": abl.get("actual_fraction", np.nan) if abl else np.nan
                }

        except Exception as e:
            print(f"⚠ Could not read {file}: {e}")

    return summaries


# ------------------------------------------------------------
# Main plotting
# ------------------------------------------------------------
def plot_all(binding_path="results/temporal_binding/temporal_binding_results.pkl",
             ablation_dir="results/ablation",
             save_path="results/temporal_binding/aletheia_quantitative_analysis_poster.png"):

    res = load_any(binding_path)
    A, Cphi, ID = extract_series(res)
    n = len(A); t = np.arange(n)

    # --- Cross-correlation
    corr = correlate(Cphi - Cphi.mean(), A - A.mean(), mode="full")
    lags = np.arange(-n + 1, n)
    corr /= np.max(np.abs(corr))

    # --- Entropy proxy (variance-based)
    amp_var = np.var(ID, axis=1)
    Hs = -np.log(np.clip(amp_var / np.max(amp_var), 1e-6, None))

    # --- Recognition log
    rec = res["recognition_log"]
    dist = np.arange(len(rec))
    recog = [r["recognition_score"] for r in rec]
    conf  = [r["confidence"] for r in rec]

    # --- Identity decay
    snaps = list(res["snapshots"].keys())
    sims = []
    for i in range(len(snaps)-1):
        v1 = res["snapshots"][snaps[i]]["identity"]
        v2 = res["snapshots"][snaps[i+1]]["identity"]
        sims.append(identity_similarity(v1, v2))
    sims = np.array(sims)

    # --- Load ablation summaries
    ablations = load_ablation_summaries(ablation_dir)

    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(18, 13))
    gs = gridspec.GridSpec(3, 2, figure=fig,
                           width_ratios=[1.05, 0.95],
                           height_ratios=[1.0, 1.0, 1.0],
                           hspace=0.4, wspace=0.35)

    # 1. Coherence–Tension
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, A, color="red", lw=2, label="Tension A")
    ax1.plot(t, Cphi, color="blue", lw=2, label="Phase coherence")
    ax1.set_xlabel("Step"); ax1.set_ylabel("Value")
    ax1.set_title("Coherence–Tension Time Series", weight="bold")
    ax1.legend(); ax1.grid(alpha=0.3)

    # 2. Cross-correlation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(lags, corr, color="purple", lw=2)
    ax2.axvline(0, color="k", ls="--", alpha=0.4)
    ax2.set_title("Cross-correlation: A vs Coherence", weight="bold")
    ax2.set_xlabel("Lag"); ax2.set_ylabel("r")
    ax2.grid(alpha=0.3)

    # 3. Spatial entropy
    ax3 = fig.add_subplot(gs[1, 0])
    sc = ax3.scatter(Cphi[:len(Hs)], Hs, c=A[:len(Hs)],
                     cmap="plasma", s=35, alpha=0.8)
    fig.colorbar(sc, ax=ax3, label="Tension A", fraction=0.05)
    ax3.set_title("Spatial Entropy vs Coherence", weight="bold")
    ax3.set_xlabel("Coherence Cφ"); ax3.set_ylabel("Entropy proxy")
    ax3.grid(alpha=0.3)

    # 4. Recognition vs distance
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(dist, recog, "bo-", lw=2, label="Recognition")
    ax4.plot(dist, conf, "ro--", lw=2, label="Confidence")
    ax4.axhline(0.8, color="green", ls=":", label="Strong")
    ax4.axhline(0.6, color="orange", ls=":", label="Weak")
    ax4.set_ylim(0, 1); ax4.set_xlim(-0.2, len(rec))
    ax4.set_xlabel("Recognition test index")
    ax4.set_title("Self-Recognition vs Temporal Distance", weight="bold")
    ax4.legend(); ax4.grid(alpha=0.3)

    # 5. Identity similarity decay
    ax5 = fig.add_subplot(gs[2, 0])
    phase_names = [s.replace("_self", "") for s in snaps]
    ax5.plot(np.arange(len(sims)), sims, "mo-", lw=2)
    ax5.set_xticks(np.arange(len(sims)))
    ax5.set_xticklabels([f"{phase_names[i]}→{phase_names[i+1]}" for i in range(len(sims))])
    ax5.set_ylim(0, 1)
    ax5.set_ylabel("Similarity")
    ax5.set_title("Identity Similarity Decay", weight="bold")
    ax5.grid(alpha=0.3)

    # 6. Ablation summary comparison
    ax6 = fig.add_subplot(gs[2, 1])
    if ablations:
        labels, id_rec, mem_rec, gains = [], [], [], []
        for lbl, d in ablations.items():
            labels.append(lbl)
            id_rec.append(d["id_rec"])
            mem_rec.append(d["mem_rec"])
            gains.append(d["gain"])

        x = np.arange(len(labels))
        width = 0.25
        ax6.bar(x - width, id_rec, width, label="Identity recovery")
        ax6.bar(x, mem_rec, width, label="Memory recovery")
        ax6.bar(x + width, gains, width, label="Gain")
        ax6.set_xticks(x)
        ax6.set_xticklabels(labels, rotation=30, ha="right")
        ax6.set_ylim(0, 1.1)
        ax6.set_ylabel("Normalized Value")
        ax6.set_title("Ablation Impact and Recovery", weight="bold")
        ax6.legend(fontsize=8)
        ax6.grid(alpha=0.3)
    else:
        ax6.text(0.5, 0.5, "No ablation .npz files found",
                 ha="center", va="center", fontsize=11, color="gray")
        ax6.set_axis_off()

    fig.suptitle("Aletheia Quantitative Research Summary", fontsize=18, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    print(f"✅ Saved six-panel analysis to {save_path}")

# ------------------------------------------------------------
if __name__ == "__main__":
    plot_all()
