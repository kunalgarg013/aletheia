# -*- coding: utf-8 -*-
"""
Plot complex-plane trajectories of QFCA walkers (from JSON produced by solve_polynomial.py).
Example:
  python -m aletheia.experiments.plot_poly_trajectories --in results/poly.json --out results/poly_traj.png
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

    X = np.array(data["traj_real"])
    Y = np.array(data["traj_imag"])
    roots = [complex(s.replace("i","j")) if isinstance(s, str) else complex(s) for s in data["roots"]]

    plt.figure(figsize=(7,7))
    T, N = X.shape
    for j in range(N):
        plt.plot(X[:, j], Y[:, j], lw=1.2, alpha=0.9)
        plt.scatter([X[0, j]], [Y[0, j]], s=20, marker="o", label=None, alpha=0.8)
        plt.scatter([X[-1, j]], [Y[-1, j]], s=30, marker="x", label=None, alpha=0.9)
    for r in roots:
        plt.scatter([r.real], [r.imag], s=60, marker="*", alpha=0.9)

    plt.axhline(0, color="k", lw=0.5, alpha=0.3)
    plt.axvline(0, color="k", lw=0.5, alpha=0.3)
    plt.xlabel("Re(z)")
    plt.ylabel("Im(z)")
    plt.title("QFCA polynomial root trajectories")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.out, dpi=160)
    print("Saved →", args.out)

if __name__ == "__main__":
    main()