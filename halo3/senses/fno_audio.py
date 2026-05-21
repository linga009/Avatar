"""AudioFNO — 1D Fourier Neural Operator for raw waveform perception.

Processes raw audio (32000,) float32 at 16kHz -> (n_tokens, codebook_dim)
spectral features. Stays in Fourier space — output tokens represent
learned frequency band signatures.

Architecture:
  Lifting: Linear(1, hidden_dim)
  N x SpectralConv1d: rfft -> spectral weights -> irfft + residual + GELU
  Spectral output: take top modes, reshape to tokens
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import equinox as eqx


class SpectralConv1d(eqx.Module):
    """1D spectral convolution: rfft -> multiply learnable weights -> irfft."""
    weights_real: jnp.ndarray  # (modes, in_ch, out_ch)
    weights_imag: jnp.ndarray  # (modes, in_ch, out_ch)
    bypass: eqx.nn.Linear     # spatial bypass (residual path)
    modes: int

    def __init__(self, in_channels: int, out_channels: int, modes: int,
                 *, key: jnp.ndarray) -> None:
        k1, k2, k3 = jax.random.split(key, 3)
        scale = 1.0 / (in_channels * out_channels)
        self.weights_real = jax.random.uniform(k1, (modes, in_channels, out_channels),
                                                minval=-scale, maxval=scale)
        self.weights_imag = jax.random.uniform(k2, (modes, in_channels, out_channels),
                                                minval=-scale, maxval=scale)
        self.bypass = eqx.nn.Linear(in_channels, out_channels, use_bias=False, key=k3)
        self.modes = modes

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """x: (N, in_channels) -> (N, out_channels)."""
        N = x.shape[0]
        x_ft = jnp.fft.rfft(x, axis=0)  # (N//2+1, in_ch)
        w = self.weights_real + 1j * self.weights_imag  # (modes, in_ch, out_ch)
        out_ft = jnp.zeros((N // 2 + 1, w.shape[2]), dtype=jnp.complex64)
        top = jnp.einsum("mi,mio->mo", x_ft[: self.modes], w)
        out_ft = out_ft.at[: self.modes].set(top)
        spectral_out = jnp.fft.irfft(out_ft, n=N, axis=0)  # (N, out_ch)
        bypass_out = jax.vmap(self.bypass)(x)  # (N, out_ch)
        return spectral_out + bypass_out


class AudioFNO(eqx.Module):
    """1D FNO: raw waveform -> spectral tokens."""
    lifting: eqx.nn.Linear
    spectral_layers: list[SpectralConv1d]
    token_proj: eqx.nn.Linear
    n_tokens: int
    modes: int

    def __init__(self, hidden_dim: int, n_layers: int, modes: int,
                 n_tokens: int, codebook_dim: int, *, key: jnp.ndarray) -> None:
        keys = jax.random.split(key, n_layers + 2)
        self.lifting = eqx.nn.Linear(1, hidden_dim, use_bias=False, key=keys[0])
        self.spectral_layers = [
            SpectralConv1d(hidden_dim, hidden_dim, modes, key=keys[i + 1])
            for i in range(n_layers)
        ]
        self.token_proj = eqx.nn.Linear(2 * hidden_dim, codebook_dim, use_bias=False,
                                         key=keys[-1])
        self.n_tokens = n_tokens
        self.modes = modes

    def __call__(self, waveform: jnp.ndarray) -> jnp.ndarray:
        """waveform: (N,) float32 -> (n_tokens, codebook_dim) spectral tokens."""
        N = waveform.shape[0]
        x = jax.vmap(self.lifting)(waveform[:, None])  # (N, hidden)
        for layer in self.spectral_layers:
            x = jax.nn.gelu(layer(x) + x)
        x_ft = jnp.fft.rfft(x, axis=0)  # (N//2+1, hidden)
        spectral = x_ft[: self.modes].real  # (modes, hidden)
        paired = spectral.reshape(self.n_tokens, -1)  # (n_tokens, 2*hidden)
        tokens = jax.vmap(self.token_proj)(paired)  # (n_tokens, codebook_dim)
        return tokens
