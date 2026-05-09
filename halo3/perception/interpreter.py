"""Output interpreter — reads Kuramoto state, decides explore/exploit, generates next query."""
from __future__ import annotations
import logging
import jax.numpy as jnp
from halo3.kuramoto import order_parameter

log = logging.getLogger(__name__)


class Interpreter:
    """Interprets Kuramoto state into findings and next actions."""

    def __init__(
        self,
        seed_topics: list[str],
        r_exploit: float = 0.6,
        r_explore: float = 0.4,
    ) -> None:
        self.seed_topics = seed_topics
        self.r_exploit = r_exploit
        self.r_explore = r_explore
        self.topic_idx = 0
        self.current_query = seed_topics[0] if seed_topics else "AI research"
        self.exploit_count = 0

    def interpret(
        self,
        theta: jnp.ndarray,
        texts: list[str],
        current_query: str,
    ) -> dict:
        """Interpret Kuramoto state after a tick.

        Returns dict with: r_mean, mode, finding, next_query
        """
        r = order_parameter(theta)  # (n_hidden,)
        r_mean = float(jnp.mean(r))

        # Determine mode
        if r_mean > self.r_exploit:
            mode = "exploit"
            self.exploit_count += 1
        else:
            mode = "explore"
            self.exploit_count = 0

        # Generate finding if pattern detected
        finding = None
        if mode == "exploit" and texts:
            # Summarize: top texts that drove synchronization
            finding = f"Pattern detected (r={r_mean:.3f}): {'; '.join(texts[:3])}"

        # Generate next query
        if mode == "exploit" and self.exploit_count < 5:
            # Refine current query — add specificity
            if texts:
                # Use first result title as refinement
                refinement = texts[0].split()[:4]
                next_query = current_query + " " + " ".join(refinement)
            else:
                next_query = current_query
        else:
            # Explore: rotate to next seed topic
            self.topic_idx = (self.topic_idx + 1) % len(self.seed_topics)
            next_query = self.seed_topics[self.topic_idx]
            self.exploit_count = 0

        self.current_query = next_query

        return {
            "r_mean": r_mean,
            "r_per_dim": r,
            "mode": mode,
            "finding": finding,
            "next_query": next_query,
        }
