"""Black-Scholes Volatility Surface for Query Valuation.

The organism treats each potential research topic as a call option:
  - S (spot price) = current competence (r history) for that topic
  - K (strike) = discovery threshold (r > 0.6 → finding)
  - sigma = implied volatility of prediction errors on that topic
  - T = ticks until next dream (time value remaining)
  - V = Black-Scholes call value = expected information gain

High-IV topics with moderate competence (near the money) have the
highest option value — they're where discoveries are most likely.
Low-IV topics are "priced in" — the organism already predicts them well.

This replaces heuristic query selection with principled explore/exploit.
"""
from __future__ import annotations
import math
import os
from collections import defaultdict


def _norm_cdf(x: float) -> float:
    """Standard normal CDF approximation (Abramowitz & Stegun)."""
    if x > 6.0:
        return 1.0
    if x < -6.0:
        return 0.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(S: float, K: float, sigma: float, T: float, r: float = 0.0) -> float:
    """Black-Scholes call option price.

    Args:
        S: spot price (current topic competence, 0-1)
        K: strike price (discovery threshold, typically 0.6)
        sigma: volatility (prediction error std for this topic)
        T: time to expiry (ticks until dream)
        r: risk-free rate (set to 0 — no discounting in organism)

    Returns:
        Call value in [0, 1] — expected probability of discovery.
    """
    if sigma < 1e-6 or T < 1e-6:
        # Zero vol or zero time: intrinsic value only
        return max(0.0, S - K)

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    call = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    return max(0.0, call)


class VolatilitySurface:
    """Tracks per-topic prediction error volatility and computes option values.

    The "implied volatility" for a topic is the rolling standard deviation
    of prediction errors (FE deltas) observed while exploring that topic.
    """

    def __init__(
        self,
        strike: float = 0.6,
        window: int = 20,
        min_samples: int = 3,
        recency_window: int = 30,
        recency_penalty: float = 0.7,
    ) -> None:
        self._strike = strike          # K: discovery threshold
        self._window = window          # rolling window for σ computation
        self._min_samples = min_samples
        self._recency_window = recency_window
        self._recency_penalty = recency_penalty
        # topic_key -> list of (r_mean, fe_delta) observations
        self._history: dict[str, list[tuple[float, float]]] = defaultdict(list)
        self._recent_topics: list[str] = []  # last N topics explored
        self._ticks_since_dream = 0
        self._dream_interval = 90      # approximate ticks between dreams

    def observe(self, topic: str, r_mean: float, fe_delta: float) -> None:
        """Record an observation for a topic. Called every tick."""
        self._ticks_since_dream += 1
        self._recent_topics.append(topic)
        if len(self._recent_topics) > self._recency_window:
            self._recent_topics = self._recent_topics[-self._recency_window:]
        hist = self._history[topic]
        hist.append((r_mean, fe_delta))
        # Keep bounded
        if len(hist) > self._window * 2:
            self._history[topic] = hist[-self._window:]

    def reset_dream_clock(self) -> None:
        """Called after dream — resets time-to-expiry."""
        self._ticks_since_dream = 0

    def implied_volatility(self, topic: str) -> float:
        """Compute implied volatility for a topic.

        IV = std(fe_deltas) scaled by sqrt of observation frequency.
        Higher IV means the topic produces unpredictable outcomes.
        """
        hist = self._history.get(topic, [])
        if len(hist) < self._min_samples:
            # Unknown topic → high default IV (optimistic prior: explore the unknown)
            # Set high enough that unexplored topics compete with near-money knowns
            return 1.0

        recent = hist[-self._window:]
        fe_deltas = [abs(fe) for _, fe in recent]

        n = len(fe_deltas)
        mean_fe = sum(fe_deltas) / n
        variance = sum((x - mean_fe) ** 2 for x in fe_deltas) / n
        sigma = math.sqrt(variance) if variance > 0 else 0.01

        # Normalize: typical FE deltas are O(1e-2), scale to option-friendly range
        # Clamp to [0.05, 2.0] — prevents degenerate pricing
        return max(0.05, min(2.0, sigma * 10.0))

    def topic_competence(self, topic: str) -> float:
        """Current competence estimate (spot price S) for a topic.

        Uses EMA of recent r values.
        """
        hist = self._history.get(topic, [])
        if not hist:
            return 0.3  # unknown topic: below strike → OTM but with potential

        recent = hist[-self._window:]
        r_values = [r for r, _ in recent]

        # EMA with alpha=0.3 (more weight to recent)
        alpha = 0.3
        ema = r_values[0]
        for r in r_values[1:]:
            ema = alpha * r + (1.0 - alpha) * ema
        return ema

    def time_to_expiry(self) -> float:
        """Ticks remaining until next dream (option expiry).

        Normalized to [0, 1] range for Black-Scholes.
        """
        remaining = max(1, self._dream_interval - self._ticks_since_dream)
        return remaining / self._dream_interval  # T in [0, 1]

    def value_topic(self, topic: str) -> float:
        """Compute the option value of exploring a topic.

        Uses Black-Scholes call price with a novelty correction:
        - Deep-ITM topics (mastered, low vol) get penalized — they're
          "priced in", no new information expected
        - Near-the-money topics with high vol get boosted — the organism
          is at the edge of discovery

        This produces the research value curve:
          low S, low vol → worthless (can't reach discovery)
          low S, high vol → moderate (volatile = might surprise)
          mid S, any vol → highest (edge of understanding)
          high S, high vol → good (can still learn)
          high S, low vol → low (mastered, boring)
        """
        S = self.topic_competence(topic)
        K = self._strike
        sigma = self.implied_volatility(topic)
        T = self.time_to_expiry()

        bs_value = black_scholes_call(S, K, sigma, T)

        # Novelty correction: penalize deep-ITM, low-vol topics (mastered)
        # Time value = BS price - intrinsic value
        intrinsic = max(0.0, S - K)
        time_value = bs_value - intrinsic

        # If mostly intrinsic (mastered) and low vol, reduce value
        if intrinsic > 0 and sigma < 0.2:
            # Discount proportional to how deep ITM we are
            satiation_discount = 1.0 - min(1.0, (S - K) / 0.3)
            return bs_value * satiation_discount + time_value

        # Exploration exhaustion: many observations but r stuck in narrow band
        # means high sigma is noise, not discovery potential — apply decay
        n_obs = len(self._history.get(topic, []))
        if n_obs > 20:
            hist = self._history.get(topic, [])
            r_values = [r for r, _ in hist[-self._window:]]
            r_range = max(r_values) - min(r_values) if len(r_values) > 1 else 0.0
            if r_range < 0.15:
                # Value halves every 20 observations beyond the first 20
                exhaustion = 0.5 ** ((n_obs - 20) / 20.0)
                bs_value *= exhaustion

        # Recency penalty: topics explored recently get discounted
        recency_count = self._recent_topics.count(topic)
        if recency_count > 0:
            bs_value *= self._recency_penalty

        return bs_value

    def rank_topics(self, candidates: list[str]) -> list[tuple[str, float]]:
        """Rank candidate topics by option value. Best first."""
        valued = [(topic, self.value_topic(topic)) for topic in candidates]
        valued.sort(key=lambda x: x[1], reverse=True)
        return valued

    def best_topic(self, candidates: list[str]) -> str | None:
        """Return the highest-value topic from candidates."""
        if not candidates:
            return None
        ranked = self.rank_topics(candidates)
        return ranked[0][0]

    def save_state(self, path: str) -> None:
        """Serialize volatility surface state to JSON for subprocess use."""
        import json
        state = {
            "strike": self._strike,
            "window": self._window,
            "dream_interval": self._dream_interval,
            "ticks_since_dream": self._ticks_since_dream,
            "history": {k: v for k, v in self._history.items()},
            "recent_topics": self._recent_topics,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f)

    @classmethod
    def load_state(cls, path: str) -> "VolatilitySurface":
        """Deserialize volatility surface from JSON."""
        import json
        with open(path) as f:
            state = json.load(f)
        vs = cls(
            strike=state.get("strike", 0.6),
            window=state.get("window", 20),
        )
        vs._dream_interval = state.get("dream_interval", 90)
        vs._ticks_since_dream = state.get("ticks_since_dream", 0)
        vs._recent_topics = state.get("recent_topics", [])
        for topic, hist in state.get("history", {}).items():
            vs._history[topic] = [tuple(h) for h in hist]
        return vs

    def summary(self) -> dict[str, dict]:
        """Return volatility surface snapshot for logging."""
        result = {}
        for topic in self._history:
            result[topic] = {
                "S": round(self.topic_competence(topic), 3),
                "sigma": round(self.implied_volatility(topic), 3),
                "V": round(self.value_topic(topic), 4),
                "n_obs": len(self._history[topic]),
            }
        return result
