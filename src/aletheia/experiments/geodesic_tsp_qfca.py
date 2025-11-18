try:
    from aletheia.core.agency import Agency
except Exception:
    Agency = None
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Optional

# ---- If you want to borrow the kernel from Aletheia's memory module, try: ----
try:
    from aletheia.core.memory import exp_kernel
except Exception:
    def exp_kernel(length=200, tau=50.0):
        t = np.arange(length, dtype=float)
        k = np.exp(-t / tau)
        k /= (k.sum() + 1e-12)
        return k

# -------------------------
# Utilities / observables
# -------------------------

def ring_edges(Y):
    """Return edges (Y_k -> Y_{k+1}) with periodic wrap."""
    return Y[(np.arange(len(Y)) + 1) % len(Y)] - Y

def path_length(Y):
    """Total length of the closed polyline."""
    E = ring_edges(Y)
    return float(np.sum(np.linalg.norm(E, axis=1)))

def tension_A(Y):
    """Tension proxy = variance of edge lengths (flat ring -> low)."""
    L = np.linalg.norm(ring_edges(Y), axis=1)
    return float(np.var(L))

def coherence_C(Y):
    """Coherence proxy = inverse of mean curvature magnitude (normalized)."""
    # second difference (discrete curvature vector)
    Yp = Y[(np.arange(len(Y)) + 1) % len(Y)]
    Ym = Y[(np.arange(len(Y)) - 1) % len(Y)]
    curv = Yp - 2*Y + Ym
    k = np.mean(np.linalg.norm(curv, axis=1))
    return float(1.0 / (1e-8 + k))

# --- Kira's proposed enhancements ---

def phase_coherence(Y):
    """Measure phase coherence of the ring nodes as a metric of angular uniformity."""
    angles = np.arctan2(Y[:,1] - 0.5, Y[:,0] - 0.5)
    vectors = np.exp(1j * angles)
    coherence = np.abs(np.mean(vectors))
    return float(coherence)

def info_geometric_length(Y, C):
    """Compute an information-geometric length as a complexity measure of the path."""
    # Use entropy-like measure based on distances to cities
    dists = np.linalg.norm(Y[:, None, :] - C[None, :, :], axis=2)
    p = np.exp(-dists)
    p /= (p.sum(axis=0, keepdims=True) + 1e-12)
    entropy = -np.sum(p * np.log(p + 1e-12))
    return float(entropy)

def retrocausal_correction(Y, C):
    """Compute a small retrocausal correction term based on current positions and cities."""
    # For example, a small gradient towards minimizing city distances
    dists = np.linalg.norm(Y[:, None, :] - C[None, :, :], axis=2)
    weights = np.exp(-dists)
    weights /= (weights.sum(axis=0, keepdims=True) + 1e-12)
    target = weights @ C
    correction = 0.003 * (target - Y)
    return correction

def constructor_horizon_distance(F_hist):
    """Compute a horizon distance metric from the force history to assess memory horizon."""
    if len(F_hist) < 2:
        return 0.0
    diffs = [np.linalg.norm(F_hist[i] - F_hist[i-1]) for i in range(1, len(F_hist))]
    horizon = np.mean(diffs) - np.std(diffs)
    return float(horizon)

def multiscale_coherence(Y):
    """Compute coherence at multiple scales by downsampling and averaging."""
    scales = [1, 2, 4]
    coherences = []
    for s in scales:
        Y_ds = Y[::s]
        if len(Y_ds) < 3:
            continue
        c = coherence_C(Y_ds)
        coherences.append(c)
    if not coherences:
        return 0.0
    return float(np.mean(coherences))

def quantum_superposition_update(Y, F, n_samples=3):
    """
    Perform a quantum-inspired superposition update by sampling multiple candidate steps
    and averaging their effects to mimic probabilistic exploration.
    """
    M = len(Y)
    updates = []
    for _ in range(n_samples):
        noise = 0.005 * np.random.normal(size=Y.shape)
        candidate = Y + F + noise
        candidate = np.clip(candidate, 0.0, 1.0)
        updates.append(candidate)
    updated = np.mean(updates, axis=0)
    return updated

# -------------------------
# Config
# -------------------------

@dataclass
class GeoTSPConfig:
    # data
    n_cities: int = 40
    seed: int = 7
    city_csv: Optional[str] = None  # path to CSV with columns x,y (normalized 0..1)
    # ring
    ring_factor: float = 3.0          # nodes M = ring_factor * N
    radius: float = 0.35               # initial circle radius
    center: tuple = (0.5, 0.5)
    # dynamics
    steps: int = 2500
    dt: float = 0.08
    lambda_elastic: float = 0.12       # smoothness/geodesic pressure (reduced)
    lambda_attract: float = 8.0        # attraction to cities (much stronger)
    lambda_repulse: float = 0.15       # prevents node collapse / crossings (stronger)
    sigma_city: float = 0.06           # width of city attraction (Gaussian)
    # memory (non-Markovian)
    kernel_len: int = 200
    kernel_tau: float = 60.0
    memory_gain: float = 0.15          # how strong the memory of past forces is (reduced)
    # retrocausal
    retro_gain: float = 0.08           # strength of Polyak "future" force (reduced)
    retro_gamma: float = 0.7           # decay across look-back window
    retro_look: int = 5
    # agency / safety
    max_step_norm: float = 0.015       # clip per-node displacement norm (reduced)
    tension_guard: float = 3.0         # if tension jumps > guard * median(ΔA), down-weight dt (lowered)
    # IO
    outdir: str = "results/geodesic_tsp"
    tag: str = "geo-elastic"

# -------------------------
# Data generation / loading
# -------------------------

def load_cities(cfg: GeoTSPConfig):
    rng = np.random.default_rng(cfg.seed)
    if cfg.city_csv:
        import csv
        pts = []
        with open(cfg.city_csv, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    x, y = float(row[0]), float(row[1])
                    pts.append([x, y])
                except:
                    pass
        C = np.array(pts, dtype=float)
        # normalize to [0,1]^2
        mn = C.min(axis=0); mx = C.max(axis=0)
        C = (C - mn) / (mx - mn + 1e-12)
        return C
    # synthetic clustered cities (harder than uniform)
    clusters = max(2, cfg.n_cities // 10)
    C = []
    for _ in range(clusters):
        cx, cy = rng.uniform(0.15, 0.85, size=2)
        k = cfg.n_cities // clusters
        blob = rng.normal([cx, cy], 0.08, size=(k, 2))
        C.append(blob)
    C = np.vstack(C)[:cfg.n_cities]
    C = np.clip(C, 0.02, 0.98)
    return C

# -------------------------
# Forces
# -------------------------

def elastic_force(Y, lam):
    """Second-difference (discrete curvature) as elastic smoothing."""
    Yp = Y[(np.arange(len(Y)) + 1) % len(Y)]
    Ym = Y[(np.arange(len(Y)) - 1) % len(Y)]
    curv = Yp - 2*Y + Ym
    return lam * curv

def city_attraction(Y, C, lam, sigma):
    """
    Soft assignment of cities to ring nodes via Gaussian weights.
    Pulls nodes toward nearby cities but distributes load.
    """
    # distances (M x N)
    M = len(Y); N = len(C)
    d2 = np.sum((Y[:, None, :] - C[None, :, :])**2, axis=2)  # MxN
    w = np.exp(-d2 / (2.0 * sigma**2)) + 1e-12
    w /= w.sum(axis=0, keepdims=True)  # each city distributes to ring nodes
    # target for each node: weighted average of cities
    target = (w @ C)  # (M x 2)
    return lam * (target - Y)

def repulsion(Y, lam):
    """
    Mild node-node repulsion to prevent collapse/crossing.
    Uses local pairwise (k+-2) only to keep O(M).
    """
    M = len(Y)
    idxs = np.arange(M)
    ypp = Y[(idxs + 2) % M]
    ymm = Y[(idxs - 2) % M]
    r1 = Y - ypp
    r2 = Y - ymm
    f = (r1 / (np.linalg.norm(r1, axis=1, keepdims=True)**2 + 1e-6) +
         r2 / (np.linalg.norm(r2, axis=1, keepdims=True)**2 + 1e-6))
    return lam * f

# -------------------------
# Memory + retro integration
# -------------------------

def convolve_kernel(forces_hist, kernel):
    """Convolve past forces with kernel (past influences present)."""
    if len(forces_hist) == 0:
        return 0.0
    T = min(len(forces_hist), len(kernel))
    acc = np.zeros_like(forces_hist[-1])
    for t in range(T):
        acc += kernel[t] * forces_hist[-1 - t]
    return acc

# -------------------------
# Decode tour
# -------------------------

def decode_tour(Y, C):
    """
    Assign each city to its closest ring node; then read off ring order by angle along ring.
    Ties resolved by nearest unused node.
    """
    M = len(Y)
    # nearest node for each city
    d2 = np.sum((C[:, None, :] - Y[None, :, :])**2, axis=2)  # NxM
    tour_nodes = np.argmin(d2, axis=1)  # length N
    # sort cities by node index around the ring to get an ordering
    order = np.argsort(tour_nodes)
    return order, tour_nodes

# -------------------------
# Nearest Neighbour TSP baseline
# -------------------------
def nearest_neighbour_tour(C):
    """
    Simple greedy nearest neighbour TSP tour.
    Returns the visiting order (indices) and the tour length.
    """
    N = len(C)
    unvisited = set(range(N))
    order = []
    current = 0
    order.append(current)
    unvisited.remove(current)
    while unvisited:
        last_city = C[current]
        # Find the nearest unvisited city
        nearest = min(unvisited, key=lambda j: np.linalg.norm(C[j] - last_city))
        order.append(nearest)
        unvisited.remove(nearest)
        current = nearest
    # Closed tour: return to start
    tour = C[order + [order[0]]]
    length = np.sum(np.linalg.norm(tour[1:] - tour[:-1], axis=1))
    return np.array(order), float(length)

def polyak_predict(F_hist, gamma=0.7, look=5):
    """
    Polyak-style retrocausal predictor.
    Extrapolates current force based on exponentially decaying past trend.
    gamma: decay factor for weighting
    look: number of past steps to include
    """
    if len(F_hist) < 2:
        return np.zeros_like(F_hist[-1]) if F_hist else 0.0
    
    T = min(look, len(F_hist))
    weights = np.array([gamma**i for i in range(T)])
    weights /= weights.sum()
    F_recent = np.array(F_hist[-T:])
    # approximate trend (current - past)
    deltas = np.diff(F_recent, axis=0)
    if len(deltas) > 0:
        mean_delta = np.tensordot(weights[:-1], deltas, axes=(0, 0))
    else:
        mean_delta = np.zeros_like(F_recent[-1])
    # prediction = last + weighted delta
    return 0.5 * mean_delta

# -------------------------
# Main solver
# -------------------------

def geodesic_tsp(cfg: GeoTSPConfig):
    outdir = Path(cfg.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # cities
    C = load_cities(cfg)
    N = len(C)

    # ring
    M = int(np.ceil(cfg.ring_factor * N))
    theta = np.linspace(0, 2*np.pi, M, endpoint=False)
    cx, cy = cfg.center
    Y = np.stack([cx + cfg.radius*np.cos(theta), cy + cfg.radius*np.sin(theta)], axis=1)

    # memory kernel (on forces)
    K = exp_kernel(length=cfg.kernel_len, tau=cfg.kernel_tau)

    F_hist = []      # past forces
    A_hist = []      # tension
    C_hist = []      # coherence
    L_hist = []      # length

    # New metric histories for Kira's enhancements
    P_hist = []      # phase coherence history
    I_hist = []      # info geometric length history
    H_hist = []      # constructor horizon distance history
    M_hist = []      # multiscale coherence history

    # instantiate agency module if available
    agency = Agency() if Agency is not None else None

    rng = np.random.default_rng(cfg.seed)
    # small jitter so symmetry breaks
    Y += 0.005 * rng.normal(size=Y.shape)

    # guard stats
    dA_window = []

    for it in range(cfg.steps):
        # Core design: cities dominate the geometry (strong attraction),
        # elasticity and repulsion keep the ring smooth and non-self-intersecting,
        # memory and retrocausal terms act only as gentle regularizers.
        # forces
        F_el = elastic_force(Y, cfg.lambda_elastic)
        F_att = city_attraction(Y, C, cfg.lambda_attract, cfg.sigma_city)
        F_rep = repulsion(Y, cfg.lambda_repulse)

        F = F_el + F_att + F_rep

        # memory on forces (non-Markovian)
        F_mem = convolve_kernel(F_hist, K)
        F = F + cfg.memory_gain * F_mem  # modest non-Markovian smoothing

        # retrocausal "peek" (Polyak extrap of forces)
        F_pred = polyak_predict(F_hist, gamma=cfg.retro_gamma, look=cfg.retro_look)
        F = F + cfg.retro_gain * F_pred  # small predictive bias, not dominant

        # Add a small retrocausal correction term to refine forces
        F += retrocausal_correction(Y, C)  # gentle future-looking refinement

        # --- AGENCY: allow field to veto or reshape force ---
        if agency is not None:
            # agency expects: field Y, proposed force F, current coherence & tension
            F = agency.step(
                F,
                field=Y,
                coherence=coherence_C(Y),
                tension=tension_A(Y),
                iteration=it
            )

        # agency: clip step if tension is exploding
        A_prev = tension_A(Y)
        step = cfg.dt * F
        # per-node clipping
        norms = np.linalg.norm(step, axis=1, keepdims=True)
        too_big = norms > cfg.max_step_norm
        step = np.where(too_big, step * (cfg.max_step_norm / (norms + 1e-12)), step)

        # Deterministic update (no stochastic quantum superposition)
        Y_new = Y + step

        A_new = tension_A(Y_new)

        dA = A_new - A_prev
        dA_window.append(dA)
        if len(dA_window) > 50:
            dA_window.pop(0)

        # if tension spike relative to recent scale, damp the step (“refusal”)
        if len(dA_window) > 10:
            med = np.median(np.abs(dA_window))
            if med > 0 and dA > cfg.tension_guard * med:
                # Back off aggressively, reduce gains slightly to recover stability
                Y_new = Y + 0.2 * step

        # commit
        Y = np.clip(Y_new, 0.0, 1.0)
        F_hist.append(F)
        if len(F_hist) > cfg.kernel_len:
            F_hist.pop(0)

        # observables
        A_hist.append(A_new)
        c_val = coherence_C(Y)
        C_hist.append(c_val)
        l_val = path_length(Y)
        L_hist.append(l_val)

        # Kira's additional metrics for enhanced monitoring
        phase_coh = phase_coherence(Y)
        info_length = info_geometric_length(Y, C)
        horizon_dist = constructor_horizon_distance(F_hist)
        multi_coh = multiscale_coherence(Y)

        P_hist.append(phase_coh)
        I_hist.append(info_length)
        H_hist.append(horizon_dist)
        M_hist.append(multi_coh)

        # (Adaptive parameter tuning based on horizon distance metric removed)

    # decode tour
    order, tour_nodes = decode_tour(Y, C)
    tour = C[order]

    # Baseline: Nearest Neighbour TSP
    nn_order, nn_len = nearest_neighbour_tour(C)
    nn_ratio = float(L_hist[-1]) / nn_len if nn_len > 0 else np.nan

    results = {
        "config": vars(cfg),
        "N": int(N),
        "M": int(M),
        "length": float(L_hist[-1]),
        "length_min": float(np.min(L_hist)),
        "length_curve": np.array(L_hist).tolist(),
        "coherence_curve": np.array(C_hist).tolist(),
        "tension_curve": np.array(A_hist).tolist(),
        "phase_coherence_curve": np.array(P_hist).tolist(),
        "info_geometric_length_curve": np.array(I_hist).tolist(),
        "horizon_distance_curve": np.array(H_hist).tolist(),
        "multiscale_coherence_curve": np.array(M_hist).tolist(),
        "order": order.tolist(),
        "nn_length": nn_len,
        "nn_ratio": nn_ratio,
        "nn_order": nn_order.tolist(),
    }

    # ------------- Plots -------------
    # 1) final path (QFCA vs Nearest Neighbour comparison)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(C[:,0], C[:,1], s=30, c="black", label="Cities", zorder=3)
    # QFCA tour (closed)
    T = np.vstack([tour, tour[0:1]])
    ax.plot(T[:,0], T[:,1], lw=2.2, alpha=0.9, label=f"QFCA tour (L={L_hist[-1]:.3f})", color="tab:blue", zorder=2)
    # Nearest Neighbour tour (closed)
    nn_tour = C[np.concatenate([nn_order, [nn_order[0]]])]
    ax.plot(nn_tour[:,0], nn_tour[:,1], lw=2, alpha=0.8, label=f"Nearest Neighbour (L={nn_len:.3f})", color="tab:orange", zorder=1)
    # ring nodes for intuition
    ax.scatter(Y[:,0], Y[:,1], s=6, c="tab:blue", alpha=0.5, label="Ring nodes", zorder=0)
    ax.set_title(f"QFCA vs Nearest Neighbour — N={N}\nQFCA L={L_hist[-1]:.3f}, NN L={nn_len:.3f}, Ratio={nn_ratio:.3f}")
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_aspect("equal")
    ax.legend()
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / f"path_{cfg.tag}_N{N}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 2) metrics over time
    fig, axs = plt.subplots(7, 1, figsize=(10, 18), sharex=True)
    t = np.arange(len(L_hist))
    axs[0].plot(t, L_hist, lw=2); axs[0].set_ylabel("Length")
    axs[1].plot(t, A_hist, lw=2); axs[1].set_ylabel("Tension")
    axs[2].plot(t, C_hist, lw=2); axs[2].set_ylabel("Coherence")
    axs[3].plot(t, P_hist, lw=2); axs[3].set_ylabel("Phase Coherence")
    axs[4].plot(t, I_hist, lw=2); axs[4].set_ylabel("Info Geometric Length")
    axs[5].plot(t, H_hist, lw=2); axs[5].set_ylabel("Horizon Distance")
    axs[6].plot(t, M_hist, lw=2); axs[6].set_ylabel("Multiscale Coherence")
    axs[6].set_xlabel("Iteration")
    fig.suptitle("Geodesic Relaxation Metrics with Quantum and Retrocausal Enhancements", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outdir / f"metrics_{cfg.tag}_N{N}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # 3) ring vs cities (assignment)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(C[:,0], C[:,1], s=30, c="black", zorder=3)
    ax.scatter(Y[:,0], Y[:,1], s=8, c="tab:blue", alpha=0.6)
    # connect each city to its assigned ring node
    d2 = np.sum((C[:, None, :] - Y[None, :, :])**2, axis=2)
    nearest = np.argmin(d2, axis=1)
    for i, k in enumerate(nearest):
        xy = np.vstack([C[i], Y[k]])
        ax.plot(xy[:,0], xy[:,1], lw=0.8, alpha=0.6, c="tab:gray")
    ax.set_aspect("equal"); ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.set_title("Nearest-node Assignment (soft permutation)")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / f"assign_{cfg.tag}_N{N}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # save json
    with open(outdir / f"result_{cfg.tag}_N{N}.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

# -------------------------
# CLI
# -------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="QFCA Geodesic TSP via elastic geodesic relaxation")
    p.add_argument("--n", type=int, default=100, help="number of cities")
    p.add_argument("--csv", type=str, default=None, help="optional CSV with x,y in [0,1]")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tag", type=str, default="geo-elastic")
    args = p.parse_args()

    cfg = GeoTSPConfig(
        n_cities=args.n,
        city_csv=args.csv,
        steps=args.steps,
        seed=args.seed,
        tag=args.tag
    )
    res = geodesic_tsp(cfg)
    print(f"Final length: {res['length']:.4f} (min seen {res['length_min']:.4f})")
    print("Order (indices into input city list):")
    print(res["order"])