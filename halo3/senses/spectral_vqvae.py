"""Spectral VQ-VAE — vector quantization in Fourier space.

Each codebook entry is a 64-dim vector representing a learned spectral
pattern (frequency signature). Quantization uses L2 distance, straight-through
estimator for gradients, and EMA updates for codebook entries.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx


class SpectralCodebook(eqx.Module):
    """Codebook for spectral VQ-VAE quantization."""
    embeddings: jnp.ndarray  # (codebook_size, codebook_dim)
    codebook_size: int
    codebook_dim: int

    def __init__(self, codebook_size: int, codebook_dim: int,
                 *, key: jnp.ndarray) -> None:
        self.embeddings = jax.random.normal(key, (codebook_size, codebook_dim)) * 0.1
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim

    def quantize(self, z_e: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Quantize encoder output to nearest codebook entries.

        Args:
            z_e: (n_tokens, codebook_dim) — encoder output

        Returns:
            z_q: (n_tokens, codebook_dim) — quantized (with straight-through grad)
            indices: (n_tokens,) int32 — codebook indices
            commitment_loss: scalar — ||z_e - sg(z_q)||^2
        """
        if z_e.shape[-1] != self.codebook_dim:
            raise ValueError(
                f"Codebook dim mismatch: z_e has dim {z_e.shape[-1]}, "
                f"codebook expects {self.codebook_dim} "
                f"(z_e shape={z_e.shape}, codebook_size={self.codebook_size})")
        z_e_sq = jnp.sum(z_e ** 2, axis=-1, keepdims=True)
        e_sq = jnp.sum(self.embeddings ** 2, axis=-1)
        dots = z_e @ self.embeddings.T
        dists = z_e_sq - 2 * dots + e_sq[None, :]

        indices = jnp.argmin(dists, axis=-1)
        z_q = self.embeddings[indices]

        commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)

        # Straight-through estimator
        z_q = z_e + jax.lax.stop_gradient(z_q - z_e)

        return z_q, indices, commitment_loss

    def ema_update(self, z_e: jnp.ndarray, indices: jnp.ndarray,
                   decay: float = 0.99) -> "SpectralCodebook":
        """Update codebook entries via exponential moving average."""
        one_hot = jax.nn.one_hot(indices, self.codebook_size)
        counts = jnp.sum(one_hot, axis=0)
        sums = one_hot.T @ z_e

        has_assignment = counts > 0
        safe_counts = jnp.maximum(counts, 1.0)
        new_means = sums / safe_counts[:, None]
        updated = decay * self.embeddings + (1 - decay) * new_means
        new_embeddings = jnp.where(has_assignment[:, None], updated, self.embeddings)

        return SpectralCodebook.__new_from_embeddings(
            new_embeddings, self.codebook_size, self.codebook_dim)

    def revive_dead_codes(self, usage_counts: jnp.ndarray, z_e: jnp.ndarray,
                          threshold: float, key: jnp.ndarray) -> "SpectralCodebook":
        """Reinitialize codes that haven't been used."""
        is_dead = usage_counts < threshold
        n_tokens = z_e.shape[0]
        sample_indices = jax.random.randint(key, (self.codebook_size,), 0, n_tokens)
        k1, k2 = jax.random.split(key)
        noise = jax.random.normal(k2, self.embeddings.shape) * 0.01
        sampled = z_e[sample_indices] + noise
        new_embeddings = jnp.where(is_dead[:, None], sampled, self.embeddings)

        return SpectralCodebook.__new_from_embeddings(
            new_embeddings, self.codebook_size, self.codebook_dim)

    @staticmethod
    def __new_from_embeddings(embeddings, codebook_size, codebook_dim):
        """Create a new SpectralCodebook with given embeddings (bypasses __init__)."""
        obj = object.__new__(SpectralCodebook)
        object.__setattr__(obj, "embeddings", embeddings)
        object.__setattr__(obj, "codebook_size", codebook_size)
        object.__setattr__(obj, "codebook_dim", codebook_dim)
        return obj
