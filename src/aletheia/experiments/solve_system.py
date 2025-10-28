# -*- coding: utf-8 -*-
"""
CLI for QFCA system root solving (vector case).
Examples:
  python -m aletheia.experiments.solve_system --demo poly --save results/system_poly.json
  python -m aletheia.experiments.solve_system --demo mixed --save results/system_mixed.json
"""

from __future__ import annotations
import argparse, json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from aletheia.solvers.system_roots import QFCASystemConfig, qfca_solve_system

def build_demo_system(name: str):
    name = name.lower()
    if name == "poly":
        # Solve:
        # f1(x,y) = x^3 - 2x + 5 = 0
        # f2(x,y) = y^2 + x - 3 = 0
        def F(z):
            # z: (M, 2)
            x = z[:, 0]; y = z[:, 1]
            f1 = x**3 - 2*x + 5
            f2 = y**2 + x - 3
            return np.stack([f1, f2], axis=1)
        def J(z):
            x = z[:, 0]; y = z[:, 1]
            # Jacobian per walker:
            # [df1/dx, df1/dy]
            # [df2/dx, df2/dy]
            J = np.zeros((z.shape[0], 2, 2), dtype=np.complex128)
            J[:, 0, 0] = 3*x**2 - 2
            J[:, 0, 1] = 0
            J[:, 1, 0] = 1
            J[:, 1, 1] = 2*y
            return J
        # sensible seeds around the cubic roots and y ~ sqrt(3 - x)
        M = 8
        ang = np.linspace(0, 2*np.pi, M, endpoint=False)
        x0 = (2.2*np.exp(1j*ang))
        # pick y≈sqrt(3 - Re(x)), allow complex where needed
        y0 = np.sqrt(3 - x0 + 0j)
        z0 = np.stack([x0, y0], axis=1).astype(np.complex128)
        return F, J, z0

    elif name == "mixed":
        # Solve:
        # f1(x,y) = sin(x) + y - 1 = 0
        # f2(x,y) = x^2 + y^2 - 2 = 0   (circle of radius sqrt(2))
        def F(z):
            x = z[:, 0]; y = z[:, 1]
            f1 = np.sin(x) + y - 1
            f2 = x**2 + y**2 - 2
            return np.stack([f1, f2], axis=1)
        def J(z):
            x = z[:, 0]; y = z[:, 1]
            J = np.zeros((z.shape[0], 2, 2), dtype=np.complex128)
            J[:, 0, 0] = np.cos(x)
            J[:, 0, 1] = 1
            J[:, 1, 0] = 2*x
            J[:, 1, 1] = 2*y
            return J
        M = 10
        ang = np.linspace(0, 2*np.pi, M, endpoint=False)
        x0 = np.sqrt(2)*np.cos(ang)
        y0 = np.sqrt(2)*np.sin(ang) + 0.2j*np.random.randn(M)  # slight complex jitter
        z0 = np.stack([x0, y0], axis=1).astype(np.complex128)
        return F, J, z0

    else:
        raise ValueError(f"Unknown demo system: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=["poly", "mixed"], default="poly")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--lam_mem", type=float, default=0.6)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--gamma", type=float, default=0.02)
    ap.add_argument("--retro_every", type=int, default=12)
    ap.add_argument("--retro_mu", type=float, default=0.6)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--walkers", type=int, default=8)
    ap.add_argument("--no_history", action="store_true")
    ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--save", default=None, help="Path to save JSON + plot")
    args = ap.parse_args()

    F, J, z0 = build_demo_system(args.demo)

    cfg = QFCASystemConfig(
        steps=args.steps, eta=args.eta, lam_mem=args.lam_mem, L=args.L, tau=args.tau,
        gamma=args.gamma, retro_every=args.retro_every, retro_mu=args.retro_mu,
        tol=args.tol, keep_history=not args.no_history, walkers=args.walkers,
        refine_roots=not args.no_refine
    )

    roots, traj, info = qfca_solve_system(F, J, z0, cfg)

    print(f"Demo: {args.demo}")
    print("Iterations:", info["iterations"], "Success:", info["success"])
    print("#Solutions:", info["n_solutions"])
    if roots.size:
        for i, r in enumerate(roots):
            print(f"  root[{i}]:", ", ".join([f"{v.real:+.10f}{v.imag:+.10f}i" for v in r]))
        print("  residuals:", info["final_residuals"])

    # Save JSON
    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "demo": args.demo,
            "iterations": info["iterations"],
            "success": bool(info["success"]),
            "config": info["config"],
            "roots": [[complex(v).__repr__() for v in r] for r in roots] if roots.size else [],
            "final_residuals": info["final_residuals"],
        }
        if traj is not None:
            payload["traj_real"] = np.real(traj).tolist()
            payload["traj_imag"] = np.imag(traj).tolist()
        with open(args.save, "w") as f:
            json.dump(payload, f, indent=2)
        print("Saved →", args.save)

    # Plot (delegate to a separate script for prettier multi-panel), but give a quick look here too
    if traj is not None:
        T, M, d = traj.shape
        assert d == 2, "Quick plot only supports d=2; use the plot script for general d."
        X0, Y0 = np.real(traj[:, :, 0]), np.imag(traj[:, :, 0])
        X1, Y1 = np.real(traj[:, :, 1]), np.imag(traj[:, :, 1])

        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        for j in range(M):
            axs[0].plot(X0[:, j], Y0[:, j], lw=1.1, alpha=0.9, label=f"walker {j+1}")
            axs[0].scatter([X0[0, j]], [Y0[0, j]], s=25, marker="o", color="blue", alpha=0.85, label="start" if j==0 else "")
            axs[0].scatter([X0[-1, j]],[Y0[-1, j]],s=35, marker="x", color="red",  alpha=0.9,  label="final" if j==0 else "")
        axs[0].set_title("Variable x (Re vs Im)")
        axs[0].set_xlabel("Re(x)"); axs[0].set_ylabel("Im(x)")
        axs[0].grid(alpha=0.25)
        axs[0].legend(loc="best", fontsize=8, framealpha=0.85)

        for j in range(M):
            axs[1].plot(X1[:, j], Y1[:, j], lw=1.1, alpha=0.9, label=f"walker {j+1}")
            axs[1].scatter([X1[0, j]], [Y1[0, j]], s=25, marker="o", color="blue", alpha=0.85, label="start" if j==0 else "")
            axs[1].scatter([X1[-1, j]],[Y1[-1, j]],s=35, marker="x", color="red",  alpha=0.9,  label="final" if j==0 else "")
        axs[1].set_title("Variable y (Re vs Im)")
        axs[1].set_xlabel("Re(y)"); axs[1].set_ylabel("Im(y)")
        axs[1].grid(alpha=0.25)
        axs[1].legend(loc="best", fontsize=8, framealpha=0.85)

        fig.suptitle("QFCA multi-variable trajectories")
        fig.tight_layout()
        if args.save:
            png = str(Path(args.save).with_suffix(".png"))
            fig.savefig(png, dpi=160)
            print("Saved plot →", png)
        else:
            plt.show()

if __name__ == "__main__":
    main()