from typing import NamedTuple
import jax.numpy as jnp
from fep_swarm.config import FEPConfig


class MacroState(NamedTuple):
    M: jnp.ndarray   # [N//k, n_hidden]  macro internal states
    S: jnp.ndarray   # [N//k, n_obs]     macro sensory boundary states
    A: jnp.ndarray   # [N//k, n_actions] macro active boundary states


def coarse_grain(
    mu: jnp.ndarray,       # [N, n_hidden]
    obs: jnp.ndarray,      # [N, n_obs]
    actions: jnp.ndarray,  # [N, n_actions]
    W: jnp.ndarray,        # [N, N] coupling matrix
    cfg: FEPConfig,
) -> MacroState:
    """
    Apply coarse-graining operator R.
    Groups N agents into N//k clusters.
    M   = mean of internal beliefs within each group.
    S,A = boundary-weighted mean for inter-group edges.
    """
    N, k = cfg.n_agents, cfg.coarse_k
    n_groups = N // k

    # M: simple group mean
    M = mu.reshape(n_groups, k, -1).mean(axis=1)  # [n_groups, n_hidden]

    # Group assignment for each agent
    group_ids = jnp.repeat(jnp.arange(n_groups), k)  # [N]

    # out_group_mask[i,j] = 1 if i and j belong to different groups
    out_group_mask = (group_ids[:, None] != group_ids[None, :]).astype(float)  # [N,N]

    # Boundary weight for agent i = sum of coupling from agents outside its group
    boundary_w = (W * out_group_mask).sum(axis=1)  # [N]

    # group_mask[g, i] = 1 if agent i is in group g
    group_mask = (group_ids[None, :] == jnp.arange(n_groups)[:, None]).astype(float)  # [n_groups, N]

    # Weighted mean per group
    denom = (group_mask @ boundary_w[:, None]) + 1e-8  # [n_groups, 1]
    S = (group_mask @ (boundary_w[:, None] * obs)) / denom     # [n_groups, n_obs]
    A_mac = (group_mask @ (boundary_w[:, None] * actions)) / denom  # [n_groups, n_actions]

    return MacroState(M=M, S=S, A=A_mac)
