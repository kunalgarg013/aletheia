# src/core/agency.py
# Agency primitives implemented as dynamical gates on outputs/updates.

from __future__ import annotations
from dataclasses import dataclass

@dataclass
class AgencyThresholds:
    pause_A: float = 0.12
    refuse_A: float = 0.18
    reframe_A: float = 0.22

class Agency:
    """
    Aletheia-style agency module.

    Given current tension A and predicted future tension A_hat, choose one of:
    - PROCEED  : allow update as-is
    - PAUSE    : damp update
    - REFUSE   : zero-out update (protect identity)
    - REFRAME  : shrink update proportional to coherence

    This file originally acted only on fields (act()), but the geodesic
    solver requires agency gating over force arrays F. The new step() method
    provides that functionality and keeps backward compatibility.
    """

    def __init__(self, th: AgencyThresholds = AgencyThresholds()):
        self.th = th
        self.paused_steps = 0

    # ---------------------------------------------------------
    #  DECISION LOGIC
    # ---------------------------------------------------------
    def decide(self, A: float, A_hat: float) -> str:
        if A_hat >= self.th.reframe_A:
            return "REFRAME"
        if A_hat >= self.th.refuse_A or A >= self.th.refuse_A:
            return "REFUSE"
        if A_hat >= self.th.pause_A or A >= self.th.pause_A:
            return "PAUSE"
        return "PROCEED"

    # ---------------------------------------------------------
    #  FIELD-LEVEL AGENCY (original Aletheia API)
    # ---------------------------------------------------------
    def act(self, field, A=None, coherence=None, t=None):
        """
        Modify field.psi in place according to agency decisions.
        Used in Aletheia experiments.
        """
        A_hat = A * (1 + 0.1 * (1 - coherence))
        action = self.decide(A, A_hat)

        if action == "REFUSE":
            field.psi *= 0.0

        elif action == "PAUSE":
            self.paused_steps += 1
            # field.psi remains unchanged

        elif action == "REFRAME":
            field.psi *= (1.0 - 0.2 * coherence)

        else:  # PROCEED
            self.paused_steps = 0

        return action

    # ---------------------------------------------------------
    #  FORCE-LEVEL AGENCY (NEW — REQUIRED BY GEODESIC TSP)
    # ---------------------------------------------------------
    def step(self, F, field=None, coherence=None, tension=None, iteration=None):
        """
        Apply agency gating to *force arrays* instead of field.psi.

        This is the TSP version of agency:
        the membrane's update direction is modified based on tension.
        
        Parameters:
        - F : ndarray (forces to be gated)
        - field : ignored; kept for compatibility
        - coherence : float
        - tension : float
        - iteration : optional int

        Returns:
        - updated F after agency gating.
        """

        # Predictive tension
        A_hat = tension * (1 + 0.1 * (1 - coherence))
        action = self.decide(tension, A_hat)

        if action == "REFUSE":
            # complete rejection of the move
            return 0 * F

        elif action == "PAUSE":
            # allow only a tiny nudge
            return 0.2 * F

        elif action == "REFRAME":
            # soften update proportional to coherence
            return F * (1 - 0.2 * coherence)

        else:  # PROCEED
            return F