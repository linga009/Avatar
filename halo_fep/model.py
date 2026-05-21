# halo_fep/model.py
"""HaloFEPModel — unified closed-loop perception + active inference.

Architecture overview
---------------------
One ``halo_fep_step`` call performs:

1. **FEP → HALO conditioning**
   - ``ActionBridge``:  swarm action probs  → (N_tok, d_boundary) bias Δx
   - ``BeliefBridge``:  swarm belief logits → (N_tok, d_model)   bias Δv

2. **Poincaré embedding**
   - ``HoloEmbedding``:  tokens → (x ∈ Poincaré disk, z curvature scalar)

3. **Flow matching setup**
   - Sample t ~ U(0,1); interpolate noisy path h_t = (1-t)*noise + t*tokens

4. **HALO backbone forward**
   - ``HALOBackbone``: (h_t, x_biased, z, Δx) → h_out (N_tok, d_model)

5. **Flow target**
   - ``ads_kg_prior``: analytical AdS-KG optimal flow → v_kg
   - Project to d_model; add Δv → v_pred; target = tokens - noise

6. **Page memory update**  (via lax.scan — JIT-safe)

7. **HALO → FEP**
   - ``ObsBridge``: h_out → soft_obs (N_agents, n_obs)

8. **Per-agent belief update + action selection**  (vmapped over agents)
   - ``belief_update``:    minimise VFE over inf_steps iterations
   - ``expected_free_energy``: compute G for each candidate policy
   - New action probs = softmax(-beta * G)

**Return value correction (Bug fix)**
The inner ``agent_step`` function returns ``(new_action_probs, mu_new)`` to
match the ``jax.vmap`` leading-axis scan convention used here.  The caller
unpacks in the same order: ``new_action, new_mu``.  This was previously
reversed, meaning ``swarm_mu`` received action probabilities and vice-versa.
"""
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
    """Immutable JAX carry encapsulating the full subconscious cognitive state.

    All fields are JAX arrays; any update returns a *new* ``HaloFEPCarry``
    instance, preserving functional purity for ``jax.jit`` compatibility.

    Attributes
    ----------
    swarm_mu     : (N_agents, n_hidden) — belief logits over discrete hidden states η.
                   Apply ``jax.nn.softmax`` along axis=-1 to get probabilities.
    swarm_action : (N_agents, n_actions) — policy probability vectors.
                   Each row sums to 1 (softmax-normalised after every step).
    page_mem     : Holographic page-memory state (JIT-compatible NamedTuple).
    key          : JAX PRNGKey (shape (2,)) for stochastic forward passes.
    """
    swarm_mu: jnp.ndarray      # (N_agents, n_hidden)
    swarm_action: jnp.ndarray  # (N_agents, n_actions) policy probabilities
    page_mem: PageMemState
    key: jnp.ndarray


class HaloFEPModel(eqx.Module):
    """Unified HoloBiont model combining HALO backbone with FEP active inference.

    All sub-modules are Equinox modules; the model is a valid JAX pytree and
    can be passed through ``eqx.filter_jit``, ``eqx.filter_grad``, etc.

    Attributes
    ----------
    holo_embed    : Poincaré disk embedding layer.
    backbone      : HALO SSM-Attention backbone.
    obs_bridge    : Maps backbone output to per-agent observations.
    action_bridge : Maps agent actions to boundary displacement Δx.
    belief_bridge : Maps agent beliefs to flow conditioning Δv.
    page_memory   : Page-curve memory module.
    gm            : Discrete generative model (A, B, C, D matrices).
    v_proj        : Linear projection d_boundary → d_model for KG prior.
    cfg           : Frozen config (static field, not a JAX leaf).
    """
    holo_embed:    HoloEmbedding
    backbone:      HALOBackbone
    obs_bridge:    ObsBridge
    action_bridge: ActionBridge
    belief_bridge: BeliefBridge
    page_memory:   PageCurveMemory
    gm:            DiscreteGenerativeModel
    v_proj:        eqx.nn.Linear  # d_boundary -> d_model, for KG prior projection
    cfg:           HaloFEPConfig = eqx.field(static=True)

    def __init__(self, cfg: HaloFEPConfig, key: jnp.ndarray) -> None:
        keys = jax.random.split(key, 8)
        self.cfg           = cfg
        self.holo_embed    = HoloEmbedding(cfg, keys[0])
        self.backbone      = HALOBackbone(cfg, keys[1])
        self.obs_bridge    = ObsBridge(cfg, keys[2])
        self.action_bridge = ActionBridge(cfg, keys[3])
        self.belief_bridge = BeliefBridge(cfg, keys[4])
        self.page_memory   = PageCurveMemory(cfg)
        self.v_proj        = eqx.nn.Linear(cfg.d_boundary, cfg.d_model, use_bias=False, key=keys[5])
        # DiscreteGenerativeModel requires cfg with n_hidden, n_obs, n_actions, n_policies, tau.
        # HaloFEPConfig satisfies all these fields.
        self.gm            = DiscreteGenerativeModel(cfg, keys[6])

    def init_carry(self, key: jnp.ndarray) -> HaloFEPCarry:
        """Initialise a zeroed carry for a fresh run.

        Beliefs are zero-logits (uniform after softmax).
        Actions are uniform (1/n_actions each).
        """
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
    """One closed-loop subconscious tick.

    Parameters
    ----------
    model  : Current HaloFEPModel (JAX pytree).
    carry  : Current cognitive state (HaloFEPCarry).
    tokens : (N_tok, d_model) float32 — packed multimodal token embeddings.
    key    : JAX PRNGKey for stochastic operations.

    Returns
    -------
    new_carry : Updated HaloFEPCarry with new beliefs, actions, and page memory.
    outputs   : Tuple ``(h_out, soft_obs, v_pred, v_target)`` where:
                - ``h_out``    (N_tok, d_model) — backbone hidden states.
                - ``soft_obs`` (N_agents, n_obs) — per-agent observation probs.
                - ``v_pred``   (N_tok, d_model) — predicted flow field.
                - ``v_target`` (N_tok, d_model) — flow-matching target.

    Notes
    -----
    **vmap return-order fix**: ``agent_step`` returns ``(new_action_probs, mu_new)``
    and the vmap result is unpacked in the same order as ``new_action, new_mu``.
    Previously the two outputs were swapped, causing ``swarm_mu`` to hold action
    probabilities and ``swarm_action`` to hold belief logits.
    """
    cfg = model.cfg
    k1, k2, k3 = jax.random.split(key, 3)

    # ------------------------------------------------------------------
    # 1. FEP → HALO conditioning
    # ------------------------------------------------------------------
    delta_x = model.action_bridge(carry.swarm_action)   # (N_tok, d_boundary)
    delta_v = model.belief_bridge(carry.swarm_mu)       # (N_tok, d_model)

    # ------------------------------------------------------------------
    # 2. Poincaré embedding
    # ------------------------------------------------------------------
    x, z = model.holo_embed(tokens)                     # (N_tok, d_boundary), (N_tok, 1)
    x_biased = x + delta_x

    # ------------------------------------------------------------------
    # 3. Flow matching setup
    # ------------------------------------------------------------------
    t       = jax.random.uniform(k1)
    h_noise = jax.random.normal(k2, tokens.shape)
    h_t     = (1.0 - t) * h_noise + t * tokens

    # ------------------------------------------------------------------
    # 4. HALO backbone
    # ------------------------------------------------------------------
    h_out = model.backbone(h_t, x_biased, z, delta_x=delta_x)  # (N_tok, d_model)

    # ------------------------------------------------------------------
    # 5. Flow prediction
    # ------------------------------------------------------------------
    x_noise = jax.random.normal(k3, x.shape)
    v_kg    = ads_kg_prior(x_noise, x_biased, t=t, delta_flow=cfg.delta_flow)
    # Project (N_tok, d_boundary) → (N_tok, d_model)
    v_kg_dm  = jax.vmap(model.v_proj)(v_kg)              # (N_tok, d_model)
    v_pred   = v_kg_dm + delta_v
    v_target = tokens - h_noise

    # ------------------------------------------------------------------
    # 6. Page memory update (JIT-safe via lax.scan)
    # ------------------------------------------------------------------
    def update_mem(mem, x_i):
        return model.page_memory(x_i, mem), None
    new_page_mem, _ = jax.lax.scan(update_mem, carry.page_mem, h_out)

    # ------------------------------------------------------------------
    # 7. HALO → FEP: observation bridge
    # ------------------------------------------------------------------
    soft_obs = model.obs_bridge(h_out)   # (N_agents, n_obs) float32

    # ------------------------------------------------------------------
    # 8. Per-agent FEP: belief update + action selection (vmapped)
    # ------------------------------------------------------------------
    # Build n_actions deterministic one-hot policies, each of shape (tau, n_actions).
    # This avoids storing a full (n_policies, tau, n_actions) tensor and is
    # sufficient for greedy policy evaluation.

    def agent_step(mu_and_action, obs_i):
        """Process a single agent.

        Parameters
        ----------
        mu_and_action : tuple (mu, action_probs) for this agent.
        obs_i         : (n_obs,) soft observation probabilities for this agent.

        Returns
        -------
        (new_action_probs, mu_new) — ORDER IS IMPORTANT for vmap unpacking.
        new_action_probs : (n_actions,) softmax-normalised policy probabilities.
        mu_new           : (n_hidden,) updated belief logits.
        """
        mu, action_probs = mu_and_action

        # Belief update: iterative gradient descent on variational free energy
        mu_new = belief_update(mu, obs_i, model.gm, cfg)

        # Build one deterministic policy per action: (n_actions, tau, n_actions)
        # Policy a = one-hot(a) repeated tau times along the time axis.
        all_policies = jax.vmap(
            lambda a: jax.nn.one_hot(
                jnp.full((cfg.tau,), a, dtype=jnp.int32),
                cfg.n_actions,
            )
        )(jnp.arange(cfg.n_actions))   # (n_actions, tau, n_actions)

        # Compute expected free energy G for each policy: returns (G,  ambiguity, risk)
        G_per_action, _, _ = jax.vmap(
            lambda policy: expected_free_energy(mu_new, policy, model.gm, cfg)
        )(all_policies)                # (n_actions,)

        # Softmax over negative G: lower G → higher probability (Eq. 9, Friston 2017)
        new_action_probs = jax.nn.softmax(-cfg.beta * G_per_action)

        # NOTE: return order (new_action_probs, mu_new) must match the vmap
        # unpacking below: ``new_action, new_mu = jax.vmap(agent_step)(...)``.
        return new_action_probs, mu_new

    # vmap over all N_agents simultaneously.
    # Inputs  : (carry.swarm_mu, carry.swarm_action) each (N_agents, ...)
    # Outputs : new_action (N_agents, n_actions), new_mu (N_agents, n_hidden)
    new_action, new_mu = jax.vmap(agent_step)(
        (carry.swarm_mu, carry.swarm_action), soft_obs
    )

    new_carry = HaloFEPCarry(
        swarm_mu     = new_mu,
        swarm_action = new_action,
        page_mem     = new_page_mem,
        key          = jax.random.split(key)[0],
    )
    return new_carry, (h_out, soft_obs, v_pred, v_target)
