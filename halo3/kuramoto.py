"""Kuramoto oscillators on n-torus — collective inference via synchronization."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
from halo3.config import Halo3Config


class KuramotoState(NamedTuple):
    theta: jnp.ndarray       # (K, n_hidden) phases in [0, 2π)
    omega: jnp.ndarray       # (K, n_hidden) natural frequencies
    coupling: float           # scalar K
    key: jnp.ndarray


def init_kuramoto(cfg: Halo3Config, key: jnp.ndarray) -> KuramotoState:
    k1, k2 = jax.random.split(key)
    return KuramotoState(
        theta=jax.random.uniform(k1, (cfg.n_clusters, cfg.n_hidden)) * 2 * jnp.pi,
        omega=jax.random.normal(k2, (cfg.n_clusters, cfg.n_hidden)) * 0.1,
        coupling=cfg.init_coupling,
        key=key,
    )


def kuramoto_step(state, obs, cfg):
    theta_mean = jnp.mean(state.theta, axis=0)
    coupling_force = state.coupling * jnp.sin(theta_mean[None, :] - state.theta)
    n_obs, n_hid = obs.shape[1], state.theta.shape[1]
    if n_obs >= n_hid:
        obs_drive = obs[:, :n_hid]
    else:
        obs_drive = jnp.pad(obs, ((0, 0), (0, n_hid - n_obs)))
    dtheta = state.omega + coupling_force + obs_drive
    new_theta = (state.theta + cfg.kuramoto_dt * dtheta) % (2 * jnp.pi)
    return state._replace(theta=new_theta, key=jax.random.split(state.key)[0])


def kuramoto_action(state, n_actions):
    theta_mean = jnp.mean(state.theta, axis=0)
    phase_velocity = jnp.sin(theta_mean[None, :] - state.theta)
    return jax.nn.softmax(phase_velocity[:, :n_actions], axis=-1)


def order_parameter(theta):
    return jnp.abs(jnp.mean(jnp.exp(1j * theta), axis=0))
