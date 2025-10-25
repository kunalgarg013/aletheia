# src/experiments/proto_avoidance.py
# Experiment A1: preference from persistence (avoid internally harmful trajectory).

import numpy as np
from src.core.field import Field, FieldConfig
from src.core.memory import exp_kernel, RetroHint
from src.core.affect import tension_A, Affect
from src.core.agency import Agency, AgencyThresholds
import argparse
import os
import numpy as np


def stimulus(shape, kind="S1", strength=0.05):
    h, w = shape
    grid_y, grid_x = np.mgrid[0:h, 0:w]
    cx, cy = (w//3, h//3) if kind == "S1" else (2*w//3, 2*h//3)
    gauss = np.exp(-(((grid_x-cx)**2 + (grid_y-cy)**2) / float((w*0.12)**2)))
    phase = np.exp(1j * (0.6 if kind == "S1" else -0.6))
    return strength * gauss * phase

def run(steps=3000, s1_rate=0.5, seed=7):
    cfg = FieldConfig(shape=(48, 48), seed=seed)
    field = Field(cfg)
    ker = exp_kernel(length=128, tau=28.0)
    affect = Affect(beta_plasticity=0.8, beta_gain=0.6)
    agency = Agency(AgencyThresholds(pause_A=0.10, refuse_A=0.16, reframe_A=0.20))
    retro = RetroHint(gain=0.04, threshold=0.15)

    history = {"A": [], "act": [], "kind": []}
    base_gain = 1.0

    for t in range(steps):
        # pick stimulus kind
        kind = "S1" if np.random.rand() < s1_rate else "S2"
        stim = stimulus(cfg.shape, kind, strength=0.05)

        # predict crude future tension (cheap proxy: last A + norm of stim)
        A_now = tension_A(field.psi)
        A_hat = A_now + 0.5 * float(np.mean(np.abs(stim)))

        # agency decision
        decision = agency.decide(A=A_now, A_hat=A_hat)
        gain = affect.modulate_input_gain(base_gain, A_now)

        if decision == "REFRAME":
            # invert phase of stimulus: try different path
            stim = np.conj(stim) * 0.8 * gain
        elif decision == "REFUSE":
            stim = np.zeros_like(stim)
        elif decision == "PAUSE":
            stim = np.zeros_like(stim)  # hold state, no new drive
        else:
            stim = stim * gain

        # retro-hint: if predicted harm high, add small global bias to stabilize
        bias = retro.bias(predicted_tension=A_hat)
        stim = stim + bias

        # step field
        metrics = field.step(kernel=ker, input_field=stim)

        # log
        history["A"].append(A_now)
        history["act"].append(decision)
        history["kind"].append(kind)

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run proto-avoidance test")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--save", type=str, default=None, help="path to save .npz history")
    args = parser.parse_args()

    hist = run(steps=args.steps)

    A = np.array(hist["A"])
    acts = np.array(hist["act"], dtype=str)
    kinds = np.array(hist["kind"], dtype=str)

    s1_mask = kinds == "S1"
    refuse_rate_s1 = np.mean(acts[s1_mask] == "REFUSE")
    refuse_rate_s2 = np.mean(acts[~s1_mask] == "REFUSE")
    pause_rate_s1 = np.mean(acts[s1_mask] == "PAUSE")
    pause_rate_s2 = np.mean(acts[~s1_mask] == "PAUSE")

    print(f"Mean tension A: {A.mean():.4f} ± {A.std():.4f}")
    print(f"REFUSE rate S1 vs S2: {refuse_rate_s1:.3f} vs {refuse_rate_s2:.3f}")
    print(f"PAUSE  rate S1 vs S2: {pause_rate_s1:.3f} vs {pause_rate_s2:.3f}")

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        np.savez(args.save, A=A, act=acts, kind=kinds)
        print(f"[saved] → {args.save}")

