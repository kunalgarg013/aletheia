# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, itertools, json, os
from pathlib import Path
import time
import yaml
import numpy as np
import matplotlib.pyplot as plt

from aletheia.solvers.system_roots import QFCASystemConfig, qfca_solve_system
from aletheia.experiments.scaling_utils import (
    make_poly_kostlan, make_mixed_trig, make_sparse_phys, patch_knn_repulsion
)

FAMILY_BUILDERS = {
    "poly": make_poly_kostlan,
    "mixed": make_mixed_trig,
    "sparse_phys": make_sparse_phys,
}

def build_family(name: str, dim: int, params: dict):
    if name == "poly":
        return make_poly_kostlan(dim, degree=params.get("degree",3), density=params.get("density",0.3))
    elif name == "mixed":
        return make_mixed_trig(dim, density=params.get("density",0.2))
    elif name == "sparse_phys":
        return make_sparse_phys(dim, block=params.get("block",5), density=params.get("density",0.05))
    else:
        raise ValueError(f"Unknown family {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/scaling.yaml")
    args = ap.parse_args()

    with open(args.cfg, "r") as f:
        C = yaml.safe_load(f)

    outdir = Path(C.get("output_dir", "results/scaling"))
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "summary.csv"

    # global solver defaults
    S = C["solver"]
    base_cfg = QFCASystemConfig(
        steps=int(S["steps"]), tol=float(S["tol"]),
        keep_history=bool(S["keep_history"]), refine_roots=bool(S["refine_roots"]),
        eta_decay=float(S["eta_decay"]), max_step=float(S["max_step"]),
        grad_cap=float(S["grad_cap"]), rmax_scale=float(S["rmax_scale"]),
        cluster_eps=float(S["cluster_eps"]),
        # the rest get filled per-run
    )

    sweep = C["sweep"]
    keys = ["walkers","L","tau","eta","retro_every","retro_mu","gamma","kNN"]
    grid = list(itertools.product(*[sweep[k] for k in keys]))

    # CSV header
    if not csv_path.exists():
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "family","dim","run_id","walkers","L","tau","eta","retro_every","retro_mu","gamma","kNN",
                "success","iterations","n_solutions","max_resid","median_resid",
                "wall_time_s"
            ])

    run_id = 0
    for fam in C["families"]:
        name = fam["name"]
        for d in fam["dims"]:
            for combo in grid:
                run_id += 1
                walkers,L,tau,eta,retro_every,retro_mu,gamma,kNN = combo

                # low-dimensional overrides
                if d <= 5:
                    eta = min(0.05, eta)
                    tau = 3.0
                    gamma = 0.0
                    retro_mu = 0.3
                    retro_every = 6

                # mid-dimensional overrides: 6 <= d <= 20
                if 6 <= d <= 20:
                    # scale step size with sqrt(d)
                    eta = min(eta, 0.08 / np.sqrt(d))
                    # ensure some memory for damping
                    L = max(L, 8)
                    tau = max(tau, 6.0)
                    # retro helps stiff couplings but keep gentle
                    retro_every = 12 if retro_every == 0 else retro_every
                    retro_mu = min(retro_mu, 0.5)
                    # disable all-pairs repulsion in mid-d if kNN=0
                    if kNN == 0:
                        gamma = 0.0

                F,J,z0 = build_family(name, d, fam)
                # config per run
                cfg = base_cfg
                cfg = cfg.__class__(**{**cfg.__dict__,
                    "walkers": walkers, "L": L, "tau": float(tau), "eta": float(eta),
                    "retro_every": int(retro_every), "retro_mu": float(retro_mu),
                    "gamma": float(gamma), "lam_mem": 0.6
                })
                if kNN and kNN > 0:
                    cfg = patch_knn_repulsion(cfg, kNN)

                t0 = time.perf_counter()
                roots, traj, info = qfca_solve_system(F, J, z0, cfg)
                dt = time.perf_counter() - t0

                res = np.array(info.get("final_residuals", []), dtype=float)
                max_resid = float(res.max()) if res.size else float("inf")
                med_resid = float(np.median(res)) if res.size else float("inf")
                nsol = int(info.get("n_solutions", 0))
                success = bool(info.get("success", False))
                iters = int(info.get("iterations", 0))

                # per-run JSON
                jdir = outdir / name / f"d{d}"
                jdir.mkdir(parents=True, exist_ok=True)
                jpath = jdir / f"run_{run_id:06d}.json"
                payload = {
                    "family": name, "dim": d, "run_id": run_id,
                    "config": {**info.get("config", {}), "kNN": kNN},
                    "n_solutions": nsol,
                    "final_residuals": info.get("final_residuals", []),
                    "iterations": iters, "success": success,
                }
                with open(jpath, "w") as jf:
                    json.dump(payload, jf, indent=2)

                # append to CSV
                with open(csv_path, "a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        name, d, run_id, walkers, L, tau, eta, retro_every, retro_mu, gamma, kNN,
                        int(success), iters, nsol, max_resid, med_resid, f"{dt:.3f}"
                    ])

                print(f"[{name} d={d}] run#{run_id} → success={success} sols={nsol} "
                      f"max|F|={max_resid:.2e} iters={iters} {dt:.2f}s")

    # quick plots: success rate vs dimension (by family), using last CSV
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7,5))
    for fam, g in df.groupby("family"):
        H = g.groupby("dim")["success"].mean().reset_index()
        ax.plot(H["dim"], H["success"], marker="o", label=fam)
    ax.set_xlabel("dimension d"); ax.set_ylabel("success rate")
    ax.set_title("QFCA scaling: success vs dimension")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "success_vs_dimension.png", dpi=160)
    print("Saved →", outdir / "success_vs_dimension.png")

if __name__ == "__main__":
    main()