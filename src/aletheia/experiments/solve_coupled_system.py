# -*- coding: utf-8 -*-
"""
Three-variable coupled nonlinear system solved by QFCA.
Example:
  python -m aletheia.experiments.solve_coupled_system --save results/coupled_system.json
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from aletheia.solvers.system_roots import QFCASystemConfig, qfca_solve_system

def build_coupled_system():
    # F: (M,3) -> (M,3)
    def F(z):
        x, y, z_ = z[:,0], z[:,1], z[:,2]
        f1 = x**2 + y + z_ - 3
        f2 = np.sin(y) + x*z_ - 1
        f3 = y**2 + z_**2 - 2*x
        return np.stack([f1,f2,f3],axis=1)

    def J(z):
        x, y, z_ = z[:,0], z[:,1], z[:,2]
        J = np.zeros((z.shape[0],3,3),dtype=np.complex128)
        J[:,0,0] = 2*x            # df1/dx
        J[:,0,1] = 1              # df1/dy
        J[:,0,2] = 1              # df1/dz
        J[:,1,0] = z_             # df2/dx
        J[:,1,1] = np.cos(y)      # df2/dy
        J[:,1,2] = x              # df2/dz
        J[:,2,0] = -2             # df3/dx
        J[:,2,1] = 2*y            # df3/dy
        J[:,2,2] = 2*z_           # df3/dz
        return J

    M = 10
    ang = np.linspace(0,2*np.pi,M,endpoint=False)
    x0 = 1.5*np.cos(ang) + 0.2j*np.random.randn(M)
    y0 = 1.5*np.sin(ang) + 0.2j*np.random.randn(M)
    z0 = 0.5*np.random.randn(M) + 0.2j*np.random.randn(M)
    init = np.stack([x0,y0,z0],axis=1)
    return F,J,init

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",type=int,default=1500)
    ap.add_argument("--eta",type=float,default=0.12)
    ap.add_argument("--lam_mem",type=float,default=0.6)
    ap.add_argument("--L",type=int,default=8)
    ap.add_argument("--tau",type=float,default=6.0)
    ap.add_argument("--gamma",type=float,default=0.02)
    ap.add_argument("--retro_every",type=int,default=12)
    ap.add_argument("--retro_mu",type=float,default=0.6)
    ap.add_argument("--tol",type=float,default=1e-9)
    ap.add_argument("--walkers",type=int,default=10)
    ap.add_argument("--no_refine",action="store_true")
    ap.add_argument("--save",default=None)
    args = ap.parse_args()

    F,J,z0 = build_coupled_system()
    cfg = QFCASystemConfig(
        steps=args.steps, eta=args.eta, lam_mem=args.lam_mem, L=args.L, tau=args.tau,
        gamma=args.gamma, retro_every=args.retro_every, retro_mu=args.retro_mu,
        tol=args.tol, walkers=args.walkers, refine_roots=not args.no_refine
    )

    roots,traj,info = qfca_solve_system(F,J,z0,cfg)
    print("Iterations:", info["iterations"], "Success:", info["success"])
    print("#Solutions:", info["n_solutions"])
    for i,r in enumerate(roots):
        print(f"root[{i}]:", ", ".join([f"{v.real:+.6f}{v.imag:+.6f}i" for v in r]))

    if args.save:
        Path(args.save).parent.mkdir(parents=True,exist_ok=True)
        payload=dict(roots=[[complex(v).__repr__() for v in r] for r in roots],
                     config=info["config"],final_residuals=info["final_residuals"])
        if traj is not None:
            payload["traj_real"]=np.real(traj).tolist()
            payload["traj_imag"]=np.imag(traj).tolist()
        with open(args.save,"w") as f: json.dump(payload,f,indent=2)
        print("Saved →",args.save)

    # quick 3D visualization of real parts
    if traj is not None:
        T,M,d = traj.shape
        from mpl_toolkits.mplot3d import Axes3D  # noqa
        fig = plt.figure(figsize=(7,6))
        ax = fig.add_subplot(111,projection='3d')
        for j in range(M):
            X,Y,Z = np.real(traj[:,j,0]), np.real(traj[:,j,1]), np.real(traj[:,j,2])
            ax.plot(X,Y,Z,lw=1.1,alpha=0.9)
            ax.scatter(X[0],Y[0],Z[0],color='blue',s=15)
            ax.scatter(X[-1],Y[-1],Z[-1],color='red',s=25)
        ax.set_xlabel('Re(x)'); ax.set_ylabel('Re(y)'); ax.set_zlabel('Re(z)')
        ax.set_title('QFCA Coupled-System Trajectories (real parts)')
        plt.tight_layout()
        if args.save:
            out = str(Path(args.save).with_suffix(".png"))
            plt.savefig(out,dpi=160)
            print("Saved plot →",out)
        else:
            plt.show()

if __name__ == "__main__":
    main()