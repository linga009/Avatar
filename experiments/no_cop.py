"""Null COP engine for ablation — returns fixed values, no SOC control."""
from __future__ import annotations
from halo3.config import Halo3Config


class NullCOP:
    """Drop-in replacement for CriticalDynamics that does nothing.

    Returns fixed chi=0.5, tau=0.5, unity=0.5, and passes K through
    unchanged. This isolates the COP contribution by removing it.
    """

    def __init__(self, cfg: Halo3Config) -> None:
        pass

    def observe(self, *, r_mean: float, r_a: float, r_c: float,
                fe_delta: float, K: float, theta) -> dict:
        return {
            "chi": 0.5,
            "tau": 0.5,
            "unity": 0.5,
            "gap": 0.5,
            "K_new": K,
            "f_dot": -fe_delta,
            "T_body": abs(r_a - r_c),
            "U_product": r_mean * 0.5,
        }
