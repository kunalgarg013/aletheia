# -*- coding: utf-8 -*-
"""
Plot trajectories for the system solver from a saved JSON (solve_system.py).
Example:
  python -m aletheia.experiments.plot_system_trajectories --in results/system_poly.json --out results/system_poly_traj.png
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    with open(args.inp, "r") as f:
        data = json.load(f)

    if "traj_real" not in data:
        raise SystemExit("No trajectories in JSON. Re-run with history enabled.")

    Xr = np.array(data["traj_real"])   # (T, M, d)
    Xi = np.array(data["traj_imag"])
    T, M, d = Xr.shape

    fig, axs = plt.subplots(1, 2, figsize=(12,5))
    for v in range(min(2, d)):
        ax = axs[v]
        for j in range(M):
            ax.plot(Xr[:, j, v], Xi[:, j, v], lw=1.1, alpha=0.9, label=f"walker {j+1}")
            ax.scatter([Xr[0, j, v]], [Xi[0, j, v]], s=25, marker="o", color="blue", alpha=0.85, label="start" if j==0 else "")
            ax.scatter([Xr[-1, j, v]],[Xi[-1, j, v]],s=35, marker="x", color="red",  alpha=0.9,  label="final" if j==0 else "")
        ax.set_xlabel(f"Re(var{v})"); ax.set_ylabel(f"Im(var{v})")
        ax.set_title(f"Variable {v}")
        ax.grid(alpha=0.25)
        ax.legend(loc="best", fontsize=8, framealpha=0.85)

    fig.suptitle(f"QFCA trajectories (d={d}, M={M})")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    print("Saved →", args.out)

if __name__ == "__main__":
    main()