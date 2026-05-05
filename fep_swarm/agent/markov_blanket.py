from typing import NamedTuple
import jax.numpy as jnp


class AgentState(NamedTuple):
    """State of one agent (or batch of N when arrays have leading N dim)."""
    mu: jnp.ndarray      # [n_hidden] beliefs
    action: jnp.ndarray  # [n_actions] last action taken
    obs: jnp.ndarray     # [n_obs] current observation


def check_blanket_independence(
    mu: jnp.ndarray,   # [N, n_hidden]
    eta: jnp.ndarray,  # [N, n_hidden]
    obs: jnp.ndarray,  # [N, n_obs]
) -> float:
    """
    Proxy for I(mu; eta | obs).
    Measures how much eta explains mu beyond what obs already explains.
    Returns near 0 when blanket independence P(mu,eta|s,a)=P(mu|s,a)P(eta|s,a) holds.
    """
    mu_c = mu - mu.mean(0)
    obs_c = obs - obs.mean(0)
    eta_c = eta - eta.mean(0)

    # Residual variance of mu unexplained by obs
    coef_obs, _, _, _ = jnp.linalg.lstsq(obs_c, mu_c, rcond=None)
    res_obs = mu_c - obs_c @ coef_obs
    var_given_obs = jnp.mean(res_obs ** 2)

    # Residual variance unexplained by (obs, eta)
    obs_eta = jnp.concatenate([obs_c, eta_c], axis=-1)
    coef_full, _, _, _ = jnp.linalg.lstsq(obs_eta, mu_c, rcond=None)
    res_full = mu_c - obs_eta @ coef_full
    var_given_both = jnp.mean(res_full ** 2)

    return float(var_given_obs - var_given_both)
