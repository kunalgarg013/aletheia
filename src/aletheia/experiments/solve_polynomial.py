# -*- coding: utf-8 -*-
"""
CLI for QFCA polynomial root finding.
Example:
  python -m aletheia.experiments.solve_polynomial --coeffs 1,0,-2,5 --steps 800 --save results/poly_z3m2z5.json
"""

from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path
from aletheia.solvers.poly_roots import QFCAPolyConfig, solve_polynomial_qfca
import matplotlib.pyplot as plt

def parse_coeffs(s: str):
    """
    Parse comma-separated coefficients into complex numbers.
    Supports 'a+bi' or plain reals. Highest degree first.
    """
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if tok.lower().endswith("i") and ("+" in tok or "-" in tok[1:]):
            tok = tok.replace("i", "j")
            out.append(complex(tok))
        else:
            out.append(complex(float(tok)))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", required=True, help="Comma-separated coeffs [a_n,...,a_0], e.g. 1,0,-2,5")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--lam_mem", type=float, default=0.6)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--tau", type=float, default=3.0)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--retro_every", type=int, default=10)
    ap.add_argument("--retro_mu", type=float, default=0.6)
    ap.add_argument("--tol", type=float, default=1e-12)
    ap.add_argument("--no_history", action="store_true")
    ap.add_argument("--save", default=None, help="Path to save JSON diagnostics")
    args = ap.parse_args()

    coeffs = parse_coeffs(args.coeffs)
    cfg = QFCAPolyConfig(
        steps=args.steps, eta=args.eta, lam_mem=args.lam_mem, L=args.L, tau=args.tau,
        gamma=args.gamma, retro_every=args.retro_every, retro_mu=args.retro_mu,
        tol=args.tol, keep_history=not args.no_history
    )
    roots, traj, info = solve_polynomial_qfca(coeffs, None, cfg)

    print("Degree:", info["degree"])
    print("Iterations:", info["iterations"], "Success:", info["success"])
    print("Roots:")
    for r, rr in zip(roots, info["roots_residuals"]):
        print(f"  {r.real:+.12f}{r.imag:+.12f}i   |p|={rr:.3e}")

    # Plot trajectories if available
    if traj is not None:
        plt.figure(figsize=(7,7))
        T, N = traj.shape
        X = np.real(traj)
        Y = np.imag(traj)
        for j in range(N):
            plt.plot(X[:, j], Y[:, j], lw=1.2, alpha=0.9, label=f"walker {j+1}")
            plt.scatter([X[0, j]], [Y[0, j]], s=30, marker="o", color="blue", alpha=0.8, label="start" if j == 0 else "")
            plt.scatter([X[-1, j]], [Y[-1, j]], s=40, marker="x", color="red", alpha=0.9, label="final" if j == 0 else "")
        for i, r in enumerate(roots):
            plt.scatter([r.real], [r.imag], s=80, marker="*", color="gold", edgecolor="black", alpha=0.9, label="root" if i == 0 else "")
        plt.axhline(0, color="k", lw=0.5, alpha=0.3)
        plt.axvline(0, color="k", lw=0.5, alpha=0.3)
        plt.xlabel("Re(z)")
        plt.ylabel("Im(z)")
        plt.title("QFCA Polynomial Root Trajectories")
        plt.legend(loc="best", fontsize=9, framealpha=0.8)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plot_path = None
        if args.save:
            plot_path = str(Path(args.save).with_suffix(".png"))
            plt.savefig(plot_path, dpi=160)
            print("Saved plot →", plot_path)
        else:
            plt.show()

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "coeffs": [complex(c).__repr__() for c in coeffs],
            "roots": [complex(r).__repr__() for r in roots],
            "roots_residuals": [float(x) for x in info["roots_residuals"]],
            "iterations": info["iterations"],
            "success": bool(info["success"]),
            "config": info["config"],
        }
        if traj is not None:
            payload["traj_real"] = np.real(traj).tolist()
            payload["traj_imag"] = np.imag(traj).tolist()
        with open(args.save, "w") as f:
            json.dump(payload, f, indent=2)
        print("Saved →", args.save)

if __name__ == "__main__":
    main()