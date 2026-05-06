"""StateCompressor — formats HALO+FEP carry state into an LLM prompt string.

No neural network. Pure deterministic formatting.
"""
from __future__ import annotations

import jax.numpy as jnp

from halo_fep.config import HaloFEPConfig
from halo_fep.model import HaloFEPCarry

_SURPRISE_LEVELS = [
    (1.0,  "low"),
    (2.0,  "medium"),
    (3.0,  "high"),
    (1e9,  "very high"),
]


def _surprise_label(fe: float) -> str:
    for threshold, label in _SURPRISE_LEVELS:
        if fe <= threshold:
            return label
    return "very high"


class StateCompressor:
    def __init__(self, cfg: HaloFEPConfig) -> None:
        self.cfg = cfg

    def compress(
        self,
        carry: HaloFEPCarry,
        recent_memories: list,
        current_query: str,
        free_energy: float,
    ) -> str:
        mean_mu     = jnp.mean(carry.swarm_mu, axis=0)       # (n_hidden,)
        belief_idx  = int(jnp.argmax(mean_mu))
        mean_action = jnp.mean(carry.swarm_action, axis=0)   # (n_actions,)
        action_idx  = int(jnp.argmax(mean_action))
        surprise    = _surprise_label(free_energy)

        lines = [
            "CURRENT STATE",
            f"Query: {current_query}",
            f"Surprise level: {surprise} (FE={free_energy:.2f})",
            f"Dominant belief: cluster {belief_idx} of {self.cfg.n_hidden}",
            f"Dominant action: action {action_idx} of {self.cfg.n_actions}",
            "",
        ]

        if recent_memories:
            lines.append("RECENT MEMORY (most similar past episodes)")
            for i, ep in enumerate(recent_memories[:5], 1):
                lines.append(
                    f"[{i}] query={ep.query!r} | FE_delta={ep.free_energy_delta:.3f}"
                )
            lines.append("")

        lines += [
            "GOAL: minimize surprise — keep exploring",
            "",
            "What should I do next? Reply with exactly one of:",
            "SEARCH: <new search query>",
            "GOAL: <new goal description>",
            "LEARN: <structured fact to remember>",
            "IDLE",
        ]

        return "\n".join(lines)
