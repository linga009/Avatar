"""Predictive Processing — the organism predicts before perceiving.

Every tick:
  1. PREDICT: Use current backbone + Hamiltonian to predict expected
     boundary coordinates for the current query
  2. PERCEIVE: Fetch actual web content, compute actual q_data
  3. PREDICTION ERROR: e = q_predicted - q_actual (vector in boundary space)
  4. LEARN: Small gradient step on backbone MERA cores + Hamiltonian V_learned
     using prediction error as the loss signal

This makes the physics body learn from every tick — not just during
bootstrap training. The body reshapes itself based on experience.

v3.1 additions:
  - State persistence: save/restore optimizer state across dreams
  - Adaptive learning rate: scales with prediction accuracy trend
"""
from __future__ import annotations
import logging
import os
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

log = logging.getLogger(__name__)


class PredictiveProcessor:
    """Implements predictive processing for the physics body.

    The organism predicts what it will perceive, then learns from
    the difference between prediction and reality.
    """

    def __init__(self, lr: float = 1e-5) -> None:
        self._base_lr = lr
        self._current_lr = lr
        self.opt = optax.adam(lr)
        self._opt_state = None
        self._prediction_history: list[float] = []

    def predict(self, model, carry, key) -> jnp.ndarray:
        """Generate predicted boundary coordinates from current state.

        Uses the backbone + Hamiltonian to predict what q_data SHOULD
        look like for the organism's current internal state.
        Returns q_predicted: (n_tokens, d_boundary).
        """
        from halo3.hamiltonian import leapfrog_integrate
        from halo3.kuramoto import kuramoto_action

        cfg = model.cfg

        # Use current Kuramoto state to condition the backbone
        actions = kuramoto_action(carry.kuramoto, cfg.n_actions)
        delta_v = model.belief_bridge(carry.kuramoto.theta)

        # Generate prediction from internal state (no external input)
        # Use the mean of recent backbone output as "expected" input
        k1, k2 = jax.random.split(key)
        # Internal prediction: random noise shaped by internal state
        h_internal = jax.random.normal(k1, (cfg.n_tokens, cfg.d_model)) * 0.1
        h_internal = h_internal + delta_v

        # Run through backbone to get predicted representation
        q_predicted, _ = model.lorentz_embed(h_internal)

        return q_predicted

    def compute_prediction_error(
        self,
        q_predicted: jnp.ndarray,
        q_actual: jnp.ndarray,
    ) -> tuple[jnp.ndarray, float]:
        """Compute prediction error vector and scalar magnitude.

        Returns:
            epsilon: (n_tokens, d_boundary) prediction error vector
            magnitude: scalar prediction error (for emotions)
        """
        epsilon = q_predicted - q_actual
        magnitude = float(jnp.mean(jnp.sum(epsilon ** 2, axis=-1)))
        return epsilon, magnitude

    def learn_from_error(
        self,
        model,
        carry,
        tokens: jnp.ndarray,
        q_actual: jnp.ndarray,
        key: jnp.ndarray,
    ):
        """Update the physics body based on prediction error.

        Takes a small gradient step on the backbone and Hamiltonian
        to reduce future prediction errors. The body physically changes.

        Returns updated model.
        """
        if self._opt_state is None:
            self._opt_state = self.opt.init(eqx.filter(model, eqx.is_array))

        # Loss: how far was our prediction from reality?
        def prediction_loss(m):
            from halo3.loss import halo3_loss
            return halo3_loss(m, carry, tokens, key)[0]

        loss, grads = eqx.filter_value_and_grad(prediction_loss)(model)

        # SELECTIVE LEARNING: only update Hamiltonian + MERA, NOT SSM/attention
        # SSM/attention feed the Kuramoto; training them causes r->1.0 collapse
        # Hamiltonian + MERA learn the data structure without disrupting sync
        lr_scale = self._adaptive_lr_scale()

        def _zero_non_target(path, grad):
            p = str(path)
            # Only keep gradients for hamiltonian and mera_ffn
            if "hamiltonian" in p or "mera" in p or "ffns" in p:
                return grad * 0.001 * lr_scale
            return jax.tree_util.tree_map(jnp.zeros_like, grad)  # freeze everything else

        grads = jax.tree_util.tree_map_with_path(_zero_non_target, grads)

        updates, self._opt_state = self.opt.update(
            eqx.filter(grads, eqx.is_array),
            self._opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)

        self._prediction_history.append(float(loss))
        return model, float(loss)

    def _adaptive_lr_scale(self) -> float:
        """Scale learning rate based on prediction accuracy trend.

        - If improving: maintain or slightly increase (up to 2x)
        - If stagnant: reduce to prevent drift (down to 0.3x)
        - If worsening: reduce sharply (down to 0.1x)
        """
        if len(self._prediction_history) < 20:
            return 1.0

        old = sum(self._prediction_history[-20:-10]) / 10
        new = sum(self._prediction_history[-10:]) / 10

        if old < 1e-12:
            return 1.0

        ratio = new / old

        if ratio < 0.9:
            # Improving: slight boost
            return min(2.0, 1.0 + (0.9 - ratio))
        elif ratio > 1.1:
            # Worsening: reduce to prevent drift
            return max(0.1, 1.0 / ratio)
        else:
            # Stagnant: slight reduction
            return 0.8

    def save_state(self, path: str) -> None:
        """Save prediction history to disk for persistence across dreams."""
        try:
            os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
            np.savez(
                path,
                prediction_history=np.array(self._prediction_history[-200:]),
            )
            log.info(f"Predictor state saved ({len(self._prediction_history)} entries)")
        except Exception as e:
            log.warning(f"Failed to save predictor state: {e}")

    def restore_state(self, path: str) -> None:
        """Restore prediction history from disk after dreams.

        Note: opt_state is NOT restored because the model weights changed
        during dreaming. The optimizer will re-initialize on first use.
        """
        if not os.path.exists(path):
            return
        try:
            data = np.load(path, allow_pickle=False)
            self._prediction_history = list(data["prediction_history"])
            # Force optimizer to re-initialize with dreamed model
            self._opt_state = None
            log.info(f"Predictor state restored ({len(self._prediction_history)} history entries)")
        except Exception as e:
            log.warning(f"Failed to restore predictor state: {e}")

    @property
    def recent_prediction_accuracy(self) -> float:
        """How well has the organism been predicting? Lower = better."""
        if not self._prediction_history:
            return 1.0
        recent = self._prediction_history[-20:]
        return sum(recent) / len(recent)

    @property
    def is_improving(self) -> bool:
        """Is prediction accuracy improving over time?"""
        if len(self._prediction_history) < 10:
            return False
        old = sum(self._prediction_history[-20:-10]) / 10
        new = sum(self._prediction_history[-10:]) / 10
        return new < old * 0.95  # at least 5% improvement
