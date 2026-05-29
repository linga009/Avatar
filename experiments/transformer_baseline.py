"""Standard transformer baseline (~100M params) for ablation comparison.

12-layer pre-norm transformer operating on continuous embeddings from
Avatar's perception pipeline.  Uses cosine similarity as the r-equivalent
metric so results are directly comparable with Avatar's order parameter.

Usage (standalone):
    python -m experiments.transformer_baseline --ticks 50

Or via the experiment runner:
    python -m experiments.experiment_runner transformer_baseline --ticks 200
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import yaml

from experiments.configs import ExperimentConfig
from experiments.metrics_logger import MetricsLogger

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class TransformerBlock(eqx.Module):
    """Pre-norm transformer block: LN -> MHA -> residual -> LN -> FFN -> residual."""

    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: eqx.nn.MultiheadAttention
    ffn_up: eqx.nn.Linear
    ffn_down: eqx.nn.Linear

    def __init__(self, d_inner: int, n_heads: int, ff_dim: int, *, key):
        k1, k2, k3 = jax.random.split(key, 3)
        self.ln1 = eqx.nn.LayerNorm(d_inner)
        self.ln2 = eqx.nn.LayerNorm(d_inner)
        self.attn = eqx.nn.MultiheadAttention(
            num_heads=n_heads,
            query_size=d_inner,
            key_size=d_inner,
            value_size=d_inner,
            output_size=d_inner,
            key=k1,
        )
        self.ffn_up = eqx.nn.Linear(d_inner, ff_dim, key=k2)
        self.ffn_down = eqx.nn.Linear(ff_dim, d_inner, key=k3)

    def __call__(self, x, mask=None):
        # x: (seq, d_inner)
        # Pre-norm self-attention
        normed = jax.vmap(self.ln1)(x)
        attn_out = self.attn(normed, normed, normed, mask=mask)
        x = x + attn_out

        # Pre-norm FFN
        normed = jax.vmap(self.ln2)(x)
        h = jax.vmap(self.ffn_up)(normed)
        h = jax.nn.gelu(h)
        h = jax.vmap(self.ffn_down)(h)
        x = x + h
        return x


class BaselineTransformer(eqx.Module):
    """Standard 12-layer pre-norm transformer for continuous embeddings.

    Input:  (seq_len, d_model) continuous embeddings from perception pipeline.
    Output: (d_model,) prediction of the last embedding from preceding context.

    Architecture:
        - Input projection: Linear(d_model, d_inner) — projects from embedding
          space into the transformer's internal dimension
        - Learnable positional embeddings: (seq_len, d_inner)
        - N transformer blocks (pre-norm) at d_inner width
        - Final LayerNorm
        - Output head: Linear(d_inner, d_model) — projects back to embedding space

    ~100M params at d_model=2048, d_inner=768, n_layers=12, n_heads=16, ff_mult=4.

    The bottleneck design (d_model=2048 -> d_inner=768 -> d_model=2048) keeps
    the param count manageable while accepting the full 2048-dim perception
    embeddings and predicting in the same space.
    """

    input_proj: eqx.nn.Linear
    pos_embed: jnp.ndarray
    blocks: list[TransformerBlock]
    final_ln: eqx.nn.LayerNorm
    output_head: eqx.nn.Linear

    def __init__(
        self,
        d_model: int = 2048,
        n_layers: int = 12,
        n_heads: int = 16,
        ff_mult: int = 4,
        max_seq_len: int = 32,
        d_inner: int = 768,
        *,
        key,
    ):
        keys = jax.random.split(key, n_layers + 3)
        ff_dim = d_inner * ff_mult

        self.input_proj = eqx.nn.Linear(d_model, d_inner, key=keys[0])
        self.pos_embed = jax.random.normal(keys[1], (max_seq_len, d_inner)) * 0.02
        self.blocks = [
            TransformerBlock(d_inner, n_heads, ff_dim, key=keys[i + 2])
            for i in range(n_layers)
        ]
        self.final_ln = eqx.nn.LayerNorm(d_inner)
        self.output_head = eqx.nn.Linear(d_inner, d_model, key=keys[-1])

    def __call__(self, x):
        """Forward pass.

        Args:
            x: (seq_len, d_model) continuous embeddings.

        Returns:
            prediction: (d_model,) predicted embedding for the last position.
        """
        seq_len = x.shape[0]

        # Project into transformer's internal dimension + add positional embedding
        x = jax.vmap(self.input_proj)(x)
        x = x + self.pos_embed[:seq_len]

        # Causal mask: (seq, seq) — True where attention is allowed
        causal_mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))

        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask=causal_mask)

        # Final norm on the second-to-last position (predict last from context)
        context_repr = self.final_ln(x[-2])

        # Output head projects back to embedding space
        prediction = self.output_head(context_repr)
        return prediction


# ---------------------------------------------------------------------------
# Loss and train step
# ---------------------------------------------------------------------------

def _cosine_similarity(a, b):
    """Cosine similarity between two vectors."""
    a_norm = jnp.sqrt(jnp.sum(a ** 2) + 1e-8)
    b_norm = jnp.sqrt(jnp.sum(b ** 2) + 1e-8)
    return jnp.sum(a * b) / (a_norm * b_norm)


def _loss_fn(model, tokens):
    """Negative cosine similarity loss.

    Args:
        model: BaselineTransformer
        tokens: (seq_len, d_model) continuous embeddings

    Returns:
        loss: scalar (negative cosine similarity)
        cos_sim: scalar (raw cosine similarity for logging)
    """
    prediction = model(tokens)
    target = tokens[-1]
    cos_sim = _cosine_similarity(prediction, target)
    loss = -cos_sim  # minimize negative cosine similarity
    return loss, cos_sim


@eqx.filter_jit
def train_step(model, opt_state, tokens, opt):
    """Single JIT-compiled training step.

    Args:
        model: BaselineTransformer
        opt_state: optax optimizer state
        tokens: (seq_len, d_model) embeddings
        opt: optax optimizer (static via closure in filter_jit)

    Returns:
        (new_model, new_opt_state, loss, cosine_sim)
    """
    grad_fn = eqx.filter_value_and_grad(lambda m: _loss_fn(m, tokens), has_aux=True)
    (loss, cos_sim), grads = grad_fn(model)

    updates, new_opt_state = opt.update(
        eqx.filter(grads, eqx.is_array),
        opt_state,
        eqx.filter(model, eqx.is_array),
    )
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_opt_state, loss, cos_sim


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_transformer_baseline(exp: ExperimentConfig) -> Path:
    """Run the transformer baseline experiment.

    Builds a ~100M param standard transformer, trains it on the same
    FineWeb-Edu data via Avatar's perception pipeline, and logs
    cosine-similarity-based r-equivalent metrics to CSV.

    Args:
        exp: ExperimentConfig with n_ticks and name fields.

    Returns:
        Path to the generated CSV file.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )

    log.info("=" * 60)
    log.info("  Transformer Baseline")
    log.info(f"  {exp.description}")
    log.info(f"  Ticks: {exp.n_ticks}")
    log.info("=" * 60)

    # --- Build model ---
    key = jax.random.PRNGKey(42)
    key, model_key = jax.random.split(key)

    model = BaselineTransformer(
        d_model=2048,
        n_layers=12,
        n_heads=16,
        ff_mult=4,
        max_seq_len=32,
        key=model_key,
    )

    n_params = sum(p.size for p in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
    log.info(f"Transformer baseline: {n_params:,} parameters")

    # --- Optimizer ---
    opt = optax.adamw(learning_rate=3e-4)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # --- Load topics ---
    topics_path = os.path.join(os.path.dirname(__file__), "..", "halo3", "topics.yaml")
    if os.path.exists(topics_path):
        with open(topics_path) as f:
            topics_cfg = yaml.safe_load(f)
        seed_topics = topics_cfg.get("seed_topics", ["artificial intelligence research"])
        max_results = topics_cfg.get("max_results_per_query", 5)
    else:
        seed_topics = ["artificial intelligence research"]
        max_results = 5

    log.info(f"Loaded {len(seed_topics)} seed topics")

    # --- Perception pipeline ---
    from halo3.perception.pipeline import PerceptionPipeline

    perception = PerceptionPipeline(d_model=2048, n_tokens=32)

    # --- Metrics logger ---
    csv_path = Path("data/experiments") / f"{exp.name}.csv"
    logger = MetricsLogger(csv_path)

    # --- Tick loop ---
    current_query = seed_topics[0]
    topic_idx = 0

    log.info(f"Starting {exp.n_ticks} ticks...")
    log.info("JIT compiling train step (first tick will be slow)...")
    t_start = time.time()

    for tick in range(1, exp.n_ticks + 1):
        tick_start = time.time()

        # Topic rotation every 20 ticks
        if tick > 1 and tick % 20 == 0:
            topic_idx = (topic_idx + 1) % len(seed_topics)
            current_query = seed_topics[topic_idx]

        # Perceive
        try:
            key, subkey = jax.random.split(key)
            tokens, texts = perception.perceive(current_query, max_results)
        except Exception as e:
            log.warning(f"Perception failed at tick {tick}: {e}")
            tokens = None
            texts = []

        # Check if we got usable tokens
        r_equiv = 0.0
        loss_val = 0.0

        if tokens is not None and tokens.shape[0] > 1:
            # Train step
            try:
                model, opt_state, loss_val, cos_sim = train_step(
                    model, opt_state, tokens, opt,
                )
                loss_val = float(loss_val)
                cos_sim_val = float(cos_sim)

                # r-equivalent: map cosine similarity from [-1, 1] to [0, 1]
                r_equiv = (cos_sim_val + 1.0) / 2.0
            except Exception as e:
                log.warning(f"Train step failed at tick {tick}: {e}")
                r_equiv = 0.0
                loss_val = 0.0
        else:
            log.debug(f"Tick {tick}: no tokens available, skipping train step")

        # Log metrics
        logger.log(
            tick=tick,
            r_mean=round(r_equiv, 6),
            fe_delta=round(loss_val, 6),
            chi=0.0,
            tau=0.0,
            K=0.0,
            unity=0.0,
            emotion="n/a",
            intensity=0.0,
            query=current_query[:80],
            prediction_error=round(abs(loss_val), 6),
            discovery="",
            topic_diversity=topic_idx + 1,
        )

        # Console log
        elapsed = time.time() - tick_start
        r_bar = "#" * int(max(0, min(1, r_equiv)) * 20) + "." * (20 - int(max(0, min(1, r_equiv)) * 20))
        log.info(
            f"[baseline] Tick {tick:4d}/{exp.n_ticks} | "
            f"r=[{r_bar}] {r_equiv:.3f} | "
            f"loss={loss_val:+.4f} | "
            f"q={current_query[:40]} | {elapsed:.1f}s"
        )

    # --- Finalize ---
    logger.close()
    total_time = time.time() - t_start
    log.info("=" * 60)
    log.info(f"  Transformer baseline complete")
    log.info(f"  {exp.n_ticks} ticks in {total_time:.1f}s ({total_time / max(1, exp.n_ticks):.2f}s/tick)")
    log.info(f"  Results: {csv_path}")
    log.info("=" * 60)

    return csv_path


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run transformer baseline experiment")
    parser.add_argument("--ticks", type=int, default=200, help="Number of ticks")
    args = parser.parse_args()

    exp = ExperimentConfig(
        name="transformer_baseline",
        n_ticks=args.ticks,
        is_transformer_baseline=True,
        description="Standard 12-layer transformer (~100M params), same data and schedule",
    )
    csv = run_transformer_baseline(exp)
    print(f"Results: {csv}")
