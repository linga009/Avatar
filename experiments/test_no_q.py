import jax
import jax.numpy as jnp
from halo3.config import Halo3Config
from halo3.kuramoto import kuramoto_step, init_kuramoto


def test_quantum_potential_disabled():
    cfg = Halo3Config(disable_quantum_potential=True)
    state = init_kuramoto(cfg, jax.random.PRNGKey(0))
    obs = jnp.zeros((cfg.n_clusters, cfg.n_hidden))
    key = jax.random.PRNGKey(1)
    new_state = kuramoto_step(state, obs, cfg)
    assert new_state.theta.shape == state.theta.shape
