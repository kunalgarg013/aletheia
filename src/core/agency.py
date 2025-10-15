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
    Given current tension A and predicted future A_hat, decide:
    - PAUSE: hold outputs (stabilize) for one or more steps
    - REFUSE: zero outputs / cancel actuation this step
    - REFRAME: request objective/gain change upstream
    """
    def __init__(self, th: AgencyThresholds = AgencyThresholds()):
        self.th = th
        self.paused_steps = 0

    def decide(self, A: float, A_hat: float) -> str:
        if A_hat >= self.th.reframe_A:
            return "REFRAME"
        if A_hat >= self.th.refuse_A or A >= self.th.refuse_A:
            return "REFUSE"
        if A_hat >= self.th.pause_A or A >= self.th.pause_A:
            return "PAUSE"
        return "PROCEED"
