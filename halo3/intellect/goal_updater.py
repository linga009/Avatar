"""GoalUpdater — decays Kuramoto coupling toward default."""
from __future__ import annotations
from halo3.config import Halo3Config
from halo3.kuramoto import KuramotoState


class GoalUpdater:
    def __init__(self, cfg: Halo3Config):
        self.cfg = cfg

    def decay(self, state: KuramotoState, alpha: float = 0.99) -> KuramotoState:
        """Decay coupling 1% toward init_coupling each tick."""
        new_coupling = alpha * state.coupling + (1.0 - alpha) * self.cfg.init_coupling
        return state._replace(coupling=new_coupling)
