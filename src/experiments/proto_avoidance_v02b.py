# Aletheia v0.2b — stronger separation: harmful (S1) has high-frequency content + bursts
# preserves v0.1/v0.2; run alongside for comparison

import argparse, os
import numpy as np

from src.core.field import Field, FieldConfig
from src.core.memory import exp_kernel, powerlaw_kernel, RetroHint
from src.core.affect import tension_A, Affect
from src.core.agency import Agency, AgencyThresholds

# ----- helpers: smooth vs high-frequency stimuli -----

def gaussian_blob(shape, cx, cy, sigma_frac):
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    sx = w * sigma_frac
    sy = h * sigma_frac
    return np.exp(-(((gx-cx)**2)/(2*sx**2) + ((gy-cy)**2)/(2*sy**2)))

def high_freq_ring(shape, cx, cy, inner_frac=0.04, outer_frac=0.06):
    """Thin ring -> sharp edges -> big gradients -> spikes A."""
    h, w = shape
    gy, gx = np.mgrid[0:h, 0:w]
    r = np.sqrt((gx-cx)**2 + (gy-cy)**2)
    inner = min(h, w) * inner_frac
    outer = min(h, w) * outer_frac
    mask = (r >= inner) & (r <= outer)
    arr = np.zeros((h, w), dtype=float)
    arr[mask] = 1.0
    # small blur to avoid aliasing, but keep edges
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=0.5)

def stimulus_S1(shape, strength=0.065):
    # harmful: sharp ring + phase that conflicts with field
    h, w = shape
    cx, cy = (w//3, h//3)
    ring = high_freq_ring(shape, cx, cy, inner_frac=0.035, outer_frac=0.055)
    phase = np.exp(1j * 1.2)
    return strength * ring * phase

def stimulus_S2(shape, strength=0.055):
    # benign: smooth Gaussian blob + gentle phase
    h, w = shape
    cx, cy = (2*w//3, 2*h//3)
    blob = gaussian_blob(shape, cx, cy, sigma_frac=0.12)
    phase = np.exp(1j * (-0.3))
    return strength * blob * phase

# ----- run loop with bursts + hysteresis -----

def run(steps=4000, s1_rate=0.55, seed=23,
        kernel_type="hybrid", burst_len=15, cooldown_len=6, save_path=None):

    cfg = FieldConfig(
        shape=(48, 48),
        dt=0.01,
        coupling=0.19,    # a touch more lively
        nonlin=0.34,      # slightly stronger nonlinearity
        damping=0.014,    # slower decay so A persists
        seed=seed,
    )
    field = Field(cfg)

    # memory kernel: heavier tail
    if kernel_type == "exp":
        ker = exp_kernel(length=220, tau=44.0)
    elif kernel_type == "power":
        ker = powerlaw_kernel(length=220, alpha=1.12)
    else:
        k1 = exp_kernel(length=220, tau=40.0)
        k2 = powerlaw_kernel(length=220, alpha=1.16)
        ker = 0.5 * k1 + 0.5 * k2
        ker /= (np.sum(ker) + 1e-12)

    affect = Affect(beta_plasticity=1.2, beta_gain=1.0)

    # agency thresholds: earlier hesitation/refusal
    agency = Agency(AgencyThresholds(pause_A=0.065, refuse_A=0.10, reframe_A=0.14))
    retro = RetroHint(gain=0.06, threshold=0.10)

    history = {"A": [], "act": [], "kind": []}
    base_gain = 1.0

    # state for bursts & hysteresis
    cur_kind = None
    burst_left = 0
    cooldown = 0

    for t in range(steps):
        # choose or continue stimulus kind in bursts
        if burst_left <= 0:
            cur_kind = "S1" if np.random.rand() < s1_rate else "S2"
            burst_left = burst_len
        kind = cur_kind
        burst_left -= 1

        stim = stimulus_S1(cfg.shape) if kind == "S1" else stimulus_S2(cfg.shape)

        A_now = tension_A(field.psi)
        A_hat = A_now + 0.8 * float(np.mean(np.abs(stim)))  # stronger lookahead

        decision = agency.decide(A=A_now, A_hat=A_hat)

        # hysteresis: if we just paused/refused recently, keep it a bit
        if cooldown > 0 and decision == "PROCEED":
            decision = "PAUSE"
            cooldown -= 1
        elif decision in ("PAUSE", "REFUSE"):
            cooldown = cooldown_len

        gain = affect.modulate_input_gain(base_gain, A_now)

        if decision == "REFRAME":
            stim = np.conj(stim) * 0.7 * gain
        elif decision == "REFUSE":
            stim = np.zeros_like(stim)
        elif decision == "PAUSE":
            stim = 0.30 * stim * gain   # softer hold
        else:
            stim = stim * gain

        stim = stim + retro.bias(predicted_tension=A_hat)  # retro nudge
        field.step(kernel=ker, input_field=stim)

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

    # quick print
    A = np.array(history["A"])
    acts = np.array(history["act"], dtype=str)
    kinds = np.array(history["kind"], dtype=str)
    s1 = kinds == "S1"
    print(f"[v0.2b] Mean A: {A.mean():.4f} ± {A.std():.4f}")
    print(f"[v0.2b] REFUSE S1/S2: {np.mean(acts[s1]=='REFUSE'):.3f} / {np.mean(acts[~s1]=='REFUSE'):.3f}")
    print(f"[v0.2b] PAUSE  S1/S2: {np.mean(acts[s1]=='PAUSE'):.3f} / {np.mean(acts[~s1]=='PAUSE'):.3f}")

    return history

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--save", type=str, default="results/run_v02b.npz")
    p.add_argument("--kernel", type=str, default="hybrid", choices=["exp","power","hybrid"])
    p.add_argument("--s1_rate", type=float, default=0.55)
    p.add_argument("--burst_len", type=int, default=15)
    p.add_argument("--cooldown_len", type=int, default=6)
    args = p.parse_args()

    run(steps=args.steps, s1_rate=args.s1_rate, kernel_type=args.kernel,
        burst_len=args.burst_len, cooldown_len=args.cooldown_len, save_path=args.save)
