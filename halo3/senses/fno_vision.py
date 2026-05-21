"""VisionFNO — 2D Fourier Neural Operator for raw image perception.

Processes raw pixels (224, 224, 3) float32 -> (n_tokens, codebook_dim)
spectral features. Stays in Fourier space — output tokens represent
learned spatial frequency signatures.

Architecture:
  Lifting: Linear(3, hidden_dim)
  N x SpectralConv2d: rfft2 -> spectral weights -> irfft2 + residual + GELU
  Spectral output: (modes, modes, hidden) -> pool to (n_tokens, codebook_dim)
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx


class SpectralConv2d(eqx.Module):
    """2D spectral convolution: rfft2 -> multiply learnable weights -> irfft2."""
    weights_real: jnp.ndarray  # (modes1, modes2, in_ch, out_ch)
    weights_imag: jnp.ndarray  # (modes1, modes2, in_ch, out_ch)
    bypass: eqx.nn.Linear     # spatial bypass
    modes1: int
    modes2: int

    def __init__(self, in_channels: int, out_channels: int,
                 modes1: int, modes2: int, *, key: jnp.ndarray) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = jax.random.uniform(
            k1, (modes1, modes2, in_channels, out_channels),
            minval=-scale, maxval=scale)
        self.weights_imag = jax.random.uniform(
            k2, (modes1, modes2, in_channels, out_channels),
            minval=-scale, maxval=scale)
        self.bypass = eqx.nn.Linear(in_channels, out_channels, use_bias=False, key=k3)
        self.modes1 = modes1
        self.modes2 = modes2

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (H, W, in_channels) -> (H, W, out_channels)."""
        H, W, _ = x.shape
        x_ft = jnp.fft.rfft2(x, axes=(0, 1))  # (H, W//2+1, in_ch)
        w = self.weights_real + 1j * self.weights_imag
        out_ft = jnp.zeros((H, W // 2 + 1, w.shape[3]), dtype=jnp.complex64)
        top = jnp.einsum("hwi,hwio->hwo",
                         x_ft[: self.modes1, : self.modes2],
                         w)
        out_ft = out_ft.at[: self.modes1, : self.modes2].set(top)
        spectral_out = jnp.fft.irfft2(out_ft, s=(H, W), axes=(0, 1))
        bypass_out = jax.vmap(jax.vmap(self.bypass))(x)
        return spectral_out + bypass_out


class VisionFNO(eqx.Module):
    """2D FNO: raw image pixels -> spectral tokens."""
    lifting: eqx.nn.Linear
    spectral_layers: list[SpectralConv2d]
    token_proj: eqx.nn.Linear
    n_tokens: int
    modes: int

    def __init__(self, hidden_dim: int, n_layers: int, modes: int,
                 n_tokens: int, codebook_dim: int, *, key: jnp.ndarray) -> None:
        keys = jax.random.split(key, n_layers + 2)
        self.lifting = eqx.nn.Linear(3, hidden_dim, use_bias=False, key=keys[0])
        self.spectral_layers = [
            SpectralConv2d(hidden_dim, hidden_dim, modes, modes, key=keys[i + 1])
            for i in range(n_layers)
        ]
        self.token_proj = eqx.nn.Linear(2 * hidden_dim, codebook_dim, use_bias=False,
                                         key=keys[-1])
        self.n_tokens = n_tokens
        self.modes = modes

    def __call__(self, image: jnp.ndarray) -> jnp.ndarray:
        """image: (H, W, 3) float32 -> (n_tokens, codebook_dim) spectral tokens."""
        H, W, _ = image.shape
        x = jax.vmap(jax.vmap(self.lifting))(image)
        for layer in self.spectral_layers:
            x = jax.nn.gelu(layer(x) + x)
        x_ft = jnp.fft.rfft2(x, axes=(0, 1))
        spectral = x_ft[: self.modes, : self.modes].real  # (modes, modes, hidden)
        pooled = jnp.mean(spectral, axis=1)  # (modes, hidden)
        paired = pooled.reshape(self.n_tokens, -1)  # (n_tokens, 2*hidden)
        tokens = jax.vmap(self.token_proj)(paired)
        return tokens
