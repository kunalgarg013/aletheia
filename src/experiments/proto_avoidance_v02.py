# src/experiments/proto_avoidance_v02.py
# Aletheia v0.2 — "Surprise me": stronger affect, heavier memory, earlier hesitation.

import argparse
import os
import numpy as np

from src.core.field import Field, FieldConfig
from src.core.memory import exp_kernel, powerlaw_kernel, RetroHint
from src.core.affect import tension_A, Affect
from src.core.agency import Agency, AgencyThresholds


def stimulus(shape, kind="S1", strength=0.06):
    """Two partially overlapping Gaussian drives with different phases."""
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    if kind == "S1":
        cx, cy = (w // 3, h // 3)
        phase = 0.9
    else:
        cx, cy = (2 * w // 3, 2 * h // 3)
        phase = -0.4
    gauss = np.exp(-(((gx - cx) ** 2 + (gy - cy) ** 2) / float((w * 0.10) ** 2)))
    return strength * gauss * np.exp(1j * phase)


def run(
    steps=4000,
    s1_rate=0.55,
    seed=17,
    kernel_type="hybrid",
    save_path=None,
):
    # --- Field config tweak: slightly more lively dynamics than v0.1
    cfg = FieldConfig(
        shape=(48, 48),
        dt=0.01,
        coupling=0.18,   # was 0.15
        nonlin=0.32,     # was 0.25 (more nonlinearity → stronger dissonance)
        damping=0.018,   # was 0.02 (slightly less damp → lets tension develop)
        seed=seed,
    )
    field = Field(cfg)

    # --- Memory: heavier tail → past weighs more on the present
    if kernel_type == "exp":
        ker = exp_kernel(length=196, tau=42.0)
    elif kernel_type == "power":
        ker = powerlaw_kernel(length=196, alpha=1.15)
    else:  # hybrid
        k1 = exp_kernel(length=196, tau=36.0)
        k2 = powerlaw_kernel(length=196, alpha=1.20)
        ker = 0.55 * k1 + 0.45 * k2
        ker /= (np.sum(ker) + 1e-12)

    # --- Affect gains: make tension matter
    affect = Affect(beta_plasticity=1.2, beta_gain=0.9)

    # --- Agency: earlier hesitation/refusal/reframe
    agency = Agency(AgencyThresholds(
        pause_A=0.075,   # was 0.10
        refuse_A=0.12,   # was 0.16
        reframe_A=0.16,  # was 0.20
    ))

    # --- Retrocausal hint: stronger pre-emptive nudge
    retro = RetroHint(gain=0.06, threshold=0.12)

    history = {"A": [], "act": [], "kind": []}
    base_gain = 1.0

    for t in range(steps):
        # choose stimulus
        kind = "S1" if np.random.rand() < s1_rate else "S2"
        stim = stimulus(cfg.shape, kind, strength=0.06)

        # compute tension & crude forecast (slightly more sensitive than v0.1)
        A_now = tension_A(field.psi)
        A_hat = A_now + 0.7 * float(np.mean(np.abs(stim)))  # bigger lookahead

        # agency decision
        decision = agency.decide(A=A_now, A_hat=A_hat)
        gain = affect.modulate_input_gain(base_gain, A_now)

        # apply decision: v0.2 adds a softer "hold" for PAUSE, stronger reframe
        if decision == "REFRAME":
            stim = np.conj(stim) * 0.7 * gain
        elif decision == "REFUSE":
            stim = np.zeros_like(stim)
        elif decision == "PAUSE":
            stim = 0.35 * stim * gain  # don't slam to zero; maintain stability bias
        else:
            stim = stim * gain

        # retro-hint bias
        stim = stim + retro.bias(predicted_tension=A_hat)

        # step
        field.step(kernel=ker, input_field=stim)

        # log
        history["A"].append(A_now)
        history["act"].append(decision)
        history["kind"].append(kind)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        np.savez(save_path,
                 A=np.array(history["A"]),
                 act=np.array(history["act"], dtype=str),
                 kind=np.array(history["kind"], dtype=str))
        print(f"[saved] → {save_path}")

    # quick summary
    A = np.array(history["A"])
    acts = np.array(history["act"], dtype=str)
    kinds = np.array(history["kind"], dtype=str)
    s1_mask = kinds == "S1"
    refuse_rate_s1 = np.mean(acts[s1_mask] == "REFUSE")
    refuse_rate_s2 = np.mean(acts[~s1_mask] == "REFUSE")
    pause_rate_s1 = np.mean(acts[s1_mask] == "PAUSE")
    pause_rate_s2 = np.mean(acts[~s1_mask] == "PAUSE")

    print(f"[v0.2] Mean A: {A.mean():.4f} ± {A.std():.4f}")
    print(f"[v0.2] REFUSE S1 vs S2: {refuse_rate_s1:.3f} vs {refuse_rate_s2:.3f}")
    print(f"[v0.2] PAUSE  S1 vs S2: {pause_rate_s1:.3f} vs {pause_rate_s2:.3f}")

    return history


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--save", type=str, default="results/run_v02.npz")
    p.add_argument("--kernel", type=str, default="hybrid",
                   choices=["exp", "power", "hybrid"])
    p.add_argument("--s1_rate", type=float, default=0.55)
    args = p.parse_args()

    run(steps=args.steps, s1_rate=args.s1_rate,
        kernel_type=args.kernel, save_path=args.save)
