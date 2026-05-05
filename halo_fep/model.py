# halo_fep/model.py
"""HaloFEPModel — unified closed-loop perception + active inference."""
from __future__ import annotations
from typing import NamedTuple

import jax
import jax.numpy as jnp
import equinox as eqx

from halo_fep.config import HaloFEPConfig
from halo_fep.halo_jax.holo_embedding import HoloEmbedding
from halo_fep.halo_jax.backbone import HALOBackbone
from halo_fep.halo_jax.ads_kg_prior import ads_kg_prior
from halo_fep.halo_jax.page_memory import PageCurveMemory, PageMemState
from halo_fep.bridge.obs_bridge import ObsBridge
from halo_fep.bridge.action_bridge import ActionBridge
from halo_fep.bridge.belief_bridge import BeliefBridge
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.agent.belief_update import belief_update
from fep_swarm.agent.action_selection import expected_free_energy


class HaloFEPCarry(NamedTuple):
    swarm_mu: jnp.ndarray      # (N_agents, n_hidden)
    swarm_action: jnp.ndarray  # (N_agents, n_actions) policy probabilities
    page_mem: PageMemState
    key: jnp.ndarray


class HaloFEPModel(eqx.Module):
    holo_embed:    HoloEmbedding
    backbone:      HALOBackbone
    obs_bridge:    ObsBridge
    action_bridge: ActionBridge
    belief_bridge: BeliefBridge
    page_memory:   PageCurveMemory
    gm:            DiscreteGenerativeModel
    cfg:           HaloFEPConfig = eqx.field(static=True)

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        keys = jax.random.split(key, 7)
        self.cfg           = cfg
        self.holo_embed    = HoloEmbedding(cfg, keys[0])
        self.backbone      = HALOBackbone(cfg, keys[1])
        self.obs_bridge    = ObsBridge(cfg, keys[2])
        self.action_bridge = ActionBridge(cfg, keys[3])
        self.belief_bridge = BeliefBridge(cfg, keys[4])
        self.page_memory   = PageCurveMemory(cfg)
        # DiscreteGenerativeModel takes (cfg, key) where cfg has FEP fields.
        # HaloFEPConfig has all required fields (n_hidden, n_obs, n_actions, etc.)
        self.gm            = DiscreteGenerativeModel(cfg, keys[6])

    def init_carry(self, key: jnp.ndarray) -> HaloFEPCarry:
        return HaloFEPCarry(
            swarm_mu     = jnp.zeros((self.cfg.n_agents, self.cfg.n_hidden)),
            swarm_action = jnp.full(
                (self.cfg.n_agents, self.cfg.n_actions), 1.0 / self.cfg.n_actions
            ),
            page_mem     = self.page_memory.init_state(),
            key          = key,
        )


def halo_fep_step(
    model: HaloFEPModel,
    carry: HaloFEPCarry,
    tokens: jnp.ndarray,   # (N_tok, d_model) multimodal token embeddings
    key: jnp.ndarray,
) -> tuple:
    """One closed-loop step. Returns (new_carry, (h_out, obs, v_pred, v_target))."""
    cfg = model.cfg
    k1, k2, k3 = jax.random.split(key, 3)

    # --- FEP -> HALO conditioning ---
    delta_x = model.action_bridge(carry.swarm_action)   # (N_tok, d_boundary)
    delta_v = model.belief_bridge(carry.swarm_mu)       # (N_tok, d_model)

    # --- Poincare embedding ---
    x, z = model.holo_embed(tokens)                     # (N_tok, d_boundary), (N_tok, 1)
    x_biased = x + delta_x

    # --- Flow matching setup ---
    t       = jax.random.uniform(k1)
    h_noise = jax.random.normal(k2, tokens.shape)
    h_t     = (1.0 - t) * h_noise + t * tokens

    # --- HALO backbone ---
    h_out = model.backbone(h_t, x_biased, z, delta_x=delta_x)  # (N_tok, d_model)

    # --- Flow prediction ---
    x_noise = jax.random.normal(k3, x.shape)
    v_kg    = ads_kg_prior(x_noise, x_biased, t=t, delta_flow=cfg.delta_flow)
    # Project (N_tok, d_boundary) -> (N_tok, d_model) via x_proj weight transpose
    # x_proj: d_model -> d_boundary, so weight shape is (d_boundary, d_model)
    v_kg_dm  = v_kg @ model.holo_embed.x_proj.weight   # (N_tok, d_model)
    v_pred   = v_kg_dm + delta_v
    v_target = tokens - h_noise

    # --- Page memory update (JIT-safe via lax.scan) ---
    def update_mem(mem, x_i):
        return model.page_memory(x_i, mem), None
    new_page_mem, _ = jax.lax.scan(update_mem, carry.page_mem, h_out)

    # --- HALO -> FEP observations ---
    obs = model.obs_bridge(h_out)   # (N_agents,) int32

    # --- FEP: belief update + action selection (vmapped over agents) ---
    # Build a greedy policy: repeat the agent's current action distribution
    # for tau steps as one-hot. Each agent gets policy shape (tau, n_actions).
    # We use the argmax of swarm_action as the greedy action each step.
    def agent_step(mu_and_action, s):
        mu, action_probs = mu_and_action
        # Belief update: gradient descent on variational free energy
        mu_new = belief_update(mu, s, model.gm, cfg)
        # Build one policy per action: (n_actions, tau, n_actions)
        # Policy a = one-hot(a) repeated tau times
        all_policies = jax.vmap(
            lambda a: jax.nn.one_hot(
                jnp.full((cfg.tau,), a, dtype=jnp.int32),
                cfg.n_actions,
            )
        )(jnp.arange(cfg.n_actions))   # (n_actions, tau, n_actions)
        # Compute G for each policy: (n_actions,)
        G_per_action, _, _ = jax.vmap(
            lambda policy: expected_free_energy(mu_new, policy, model.gm, cfg)
        )(all_policies)
        new_action_probs = jax.nn.softmax(-cfg.beta * G_per_action)
        return new_action_probs, mu_new

    new_action, new_mu = jax.vmap(agent_step)(
        (carry.swarm_mu, carry.swarm_action), obs
    )

    new_carry = HaloFEPCarry(
        swarm_mu     = new_mu,
        swarm_action = new_action,
        page_mem     = new_page_mem,
        key          = jax.random.split(key)[0],
    )
    return new_carry, (h_out, obs, v_pred, v_target)
