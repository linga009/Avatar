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

v3.7 additions:
  - SenseModule integration: FNO + VQ-VAE + gated injection replaces
    old SenseProjections (3 linear layers)
  - Commitment loss included in per-tick backward pass
  - Codebook embeddings zeroed (EMA only, not gradient-updated)
  - Decoder gradients zeroed (trained separately during critical period)
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
        self._sense_opt = optax.adam(lr * 10)  # sense projections train 10x faster
        self._sense_opt_state = None

    def predict(self, model, carry, key) -> jnp.ndarray:
        """Generate predicted boundary coordinates from recent experience.

        Uses the Page memory ring buffer: the organism predicts that
        the next observation will resemble recent observations. This is
        a genuine temporal prediction — if the environment is stable,
        prediction error is low; if something new appears, error spikes.

        Returns q_predicted: (n_tokens, d_boundary).
        """
        cfg = model.cfg
        page_mem = carry.page_mem

        # Number of valid entries in the ring buffer
        n_valid = jnp.minimum(page_mem.n_cached, cfg.max_cache)

        # Use the most recent n_tokens entries as the prediction
        # (circular buffer: write pointer is at n_cached % max_cache)
        write_ptr = page_mem.n_cached % cfg.max_cache
        indices = (write_ptr - jnp.arange(1, cfg.n_tokens + 1)) % cfg.max_cache
        h_predicted = page_mem.cache[indices]  # (n_tokens, d_model)

        # Fallback for empty buffer (first tick)
        is_empty = n_valid < cfg.n_tokens
        k1, _ = jax.random.split(key)
        h_fallback = jax.random.normal(k1, (cfg.n_tokens, cfg.d_model)) * 0.01
        h_predicted = jnp.where(is_empty, h_fallback, h_predicted)

        # Project to boundary coordinates
        q_predicted, _ = model.lorentz_embed(h_predicted)
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
        sense_module,
        carry,
        text_tokens: jnp.ndarray,
        audio_raw: jnp.ndarray,
        vision_raw: jnp.ndarray,
        q_actual: jnp.ndarray,
        key: jnp.ndarray,
        contrastive_aligner=None,
        text_paired: bool = False,
        contrastive_weight: float = 0.0,
        dF_dt: float = 0.0,
    ):
        """Update the physics body and sense module based on prediction error.

        Returns updated (model, sense_module, loss, info).
        """
        if self._opt_state is None:
            self._opt_state = self.opt.init(eqx.filter(model, eqx.is_array))
        if self._sense_opt_state is None:
            self._sense_opt_state = self._sense_opt.init(
                eqx.filter(sense_module, eqx.is_array))

        # Forward pass outside grad for info extraction
        _, info = sense_module.process_and_inject(text_tokens, audio_raw, vision_raw)

        def prediction_loss(params):
            m, sm = params
            tokens_out, _ = sm.process_and_inject(text_tokens, audio_raw, vision_raw)
            from halo3.loss import halo3_loss
            body_loss = halo3_loss(m, carry, tokens_out, key)[0]
            _, _, commit_a = sm.audio_codebook.quantize(sm.audio_fno(audio_raw))
            _, _, commit_v = sm.vision_codebook.quantize(sm.vision_fno(vision_raw))
            commitment = 0.25 * (commit_a + commit_v)
            total = body_loss + commitment
            return total

        params = (model, sense_module)
        loss, grads = eqx.filter_value_and_grad(prediction_loss)(params)
        m_grads, sm_grads = grads

        # SELECTIVE LEARNING for model: only Hamiltonian + MERA, NOT SSM/attention
        lr_scale = self._adaptive_lr_scale(dF_dt=dF_dt)

        def _zero_non_target(path, grad):
            p = str(path)
            if "hamiltonian" in p or "mera" in p or "ffns" in p:
                return grad * 0.001 * lr_scale
            return jax.tree_util.tree_map(jnp.zeros_like, grad)

        m_grads = jax.tree_util.tree_map_with_path(_zero_non_target, m_grads)

        # Update model
        m_updates, self._opt_state = self.opt.update(
            eqx.filter(m_grads, eqx.is_array),
            self._opt_state,
            eqx.filter(model, eqx.is_array),
        )
        new_model = eqx.apply_updates(model, m_updates)

        # Zero codebook + decoder gradients
        def _zero_codebook(path, grad):
            p = str(path)
            if "codebook" in p and "embeddings" in p:
                return jax.tree_util.tree_map(jnp.zeros_like, grad)
            if "decoder" in p:
                return jax.tree_util.tree_map(jnp.zeros_like, grad)
            return grad

        sm_grads = jax.tree_util.tree_map_with_path(_zero_codebook, sm_grads)

        sp_updates, self._sense_opt_state = self._sense_opt.update(
            eqx.filter(sm_grads, eqx.is_array),
            self._sense_opt_state,
            eqx.filter(sense_module, eqx.is_array),
        )
        new_sm = eqx.apply_updates(sense_module, sp_updates)

        # Contrastive alignment (Phase B — optional)
        if text_paired and contrastive_aligner is not None and not contrastive_aligner.matured:
            audio_emb_mean = jnp.mean(
                jax.vmap(sense_module.spectral_proj)(info["audio_z_q"]), axis=0)
            text_emb_mean = jnp.mean(text_tokens, axis=0)
            c_loss = contrastive_aligner.compute_loss(audio_emb_mean, text_emb_mean)
            # Note: c_loss involves numpy buffer, not fully differentiable
            # We add it to the reported loss for logging
            loss = float(loss) + contrastive_weight * float(c_loss)

        info["text_paired"] = text_paired
        self._prediction_history.append(float(loss))
        return new_model, new_sm, float(loss), info

    def _adaptive_lr_scale(self, dF_dt: float = 0.0) -> float:
        """Scale learning rate based on prediction accuracy + thermodynamic efficiency.

        Two signals:
        1. Prediction accuracy trend (existing): ratio of recent vs older loss
        2. Thermodynamic efficiency (new): dF/dt from COP engine
           - dF/dt < 0: free energy decreasing = efficient learning, boost lr
           - dF/dt > 0: free energy increasing = wasteful, reduce lr
           - dF/dt ~ 0: thermally equilibrated, maintain
        """
        if len(self._prediction_history) < 20:
            return 1.0

        old = sum(self._prediction_history[-20:-10]) / 10
        new = sum(self._prediction_history[-10:]) / 10

        if old < 1e-12:
            return 1.0

        ratio = new / old

        if ratio < 0.9:
            base_scale = min(2.0, 1.0 + (0.9 - ratio))
        elif ratio > 1.1:
            base_scale = max(0.1, 1.0 / ratio)
        else:
            base_scale = 0.8

        # Thermodynamic modulation: dF/dt adjusts the base scale
        if abs(dF_dt) > 10.0:  # only modulate when F is meaningfully changing
            if dF_dt < 0:
                # F decreasing = efficient thermalization, boost up to 1.3x
                thermo_factor = min(1.3, 1.0 + abs(dF_dt) / 10000.0)
            else:
                # F increasing = wasteful, reduce down to 0.7x
                thermo_factor = max(0.7, 1.0 - dF_dt / 10000.0)
        else:
            thermo_factor = 1.0

        return max(0.1, min(2.5, base_scale * thermo_factor))

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
            # Force optimizers to re-initialize with dreamed model/sense_proj
            self._opt_state = None
            self._sense_opt_state = None
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
