"""Predictive Processing — the organism predicts before perceiving.

Every tick:
  1. PREDICT: Use current backbone + Hamiltonian to predict expected
     boundary coordinates for the current query
  2. PERCEIVE: Fetch actual web content, compute actual q_data
  3. PREDICTION ERROR: ε = q_predicted - q_actual (vector in boundary space)
  4. LEARN: Small gradient step on backbone MERA cores + Hamiltonian V_learned
     using prediction error as the loss signal

This makes the physics body learn from every tick — not just during
bootstrap training. The body reshapes itself based on experience.
"""
from __future__ import annotations
import logging
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

log = logging.getLogger(__name__)


class PredictiveProcessor:
    """Implements predictive processing for the physics body.

    The organism predicts what it will perceive, then learns from
    the difference between prediction and reality.
    """

    def __init__(self, lr: float = 1e-5) -> None:
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
            q_pred = self.predict(m, carry, key)
            return jnp.mean((q_pred - q_actual) ** 2)

        loss, grads = eqx.filter_value_and_grad(prediction_loss)(model)

        # Very gentle adaptation — prevent Kuramoto collapse from aggressive learning
        grads = jax.tree_util.tree_map(lambda g: g * 0.001, grads)

        updates, self._opt_state = self.opt.update(
            eqx.filter(grads, eqx.is_array),
            self._opt_state,
            eqx.filter(model, eqx.is_array),
        )
        model = eqx.apply_updates(model, updates)

        self._prediction_history.append(float(loss))
        return model, float(loss)

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
