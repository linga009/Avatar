"""Halo3Model — physics engine: backbone + Hamiltonian ODE + Kuramoto."""
from __future__ import annotations
from typing import NamedTuple
import jax
import jax.numpy as jnp
import equinox as eqx
from halo3.config import Halo3Config
from halo3.lorentz_embedding import LorentzEmbedding
from halo3.backbone import Halo3Backbone
from halo3.hamiltonian import LearnedHamiltonian, leapfrog_integrate, MomentumInitializer
from halo3.kuramoto import KuramotoState, init_kuramoto, kuramoto_step, kuramoto_action, order_parameter
from halo3.page_memory import PageCurveMemory, PageMemState
from halo3.bridge.obs_bridge import ObsBridge
from halo3.bridge.action_bridge import ActionBridge
from halo3.bridge.belief_bridge import BeliefBridge


class Halo3Carry(NamedTuple):
    kuramoto: KuramotoState
    page_mem: PageMemState
    key: jnp.ndarray


class Halo3Model(eqx.Module):
    lorentz_embed: LorentzEmbedding
    backbone: eqx.Module  # Halo3Backbone
    hamiltonian: LearnedHamiltonian
    momentum_init: MomentumInitializer
    obs_bridge: ObsBridge
    action_bridge: ActionBridge
    belief_bridge: BeliefBridge
    page_memory: PageCurveMemory
    cfg: Halo3Config = eqx.field(static=True)

    def __init__(self, cfg: Halo3Config, key: jnp.ndarray):
        keys = jax.random.split(key, 10)
        self.cfg = cfg
        self.lorentz_embed = LorentzEmbedding(cfg, keys[0])
        self.backbone = Halo3Backbone(cfg, keys[1])
        self.hamiltonian = LearnedHamiltonian(cfg, keys[2])
        self.momentum_init = MomentumInitializer(cfg, keys[3])
        self.obs_bridge = ObsBridge(cfg, keys[4])
        self.action_bridge = ActionBridge(cfg, keys[5])
        self.belief_bridge = BeliefBridge(cfg, keys[6])
        self.page_memory = PageCurveMemory(cfg)

    def init_carry(self, key: jnp.ndarray) -> Halo3Carry:
        return Halo3Carry(
            kuramoto=init_kuramoto(self.cfg, key),
            page_mem=self.page_memory.init_state(),
            key=key,
        )


def halo3_step(model: Halo3Model, carry: Halo3Carry, tokens: jnp.ndarray, key: jnp.ndarray,
               coherence_weights: jnp.ndarray | None = None):
    """One closed-loop step. Returns (new_carry, (h_out, obs, q_final, q_data))."""
    cfg = model.cfg
    k1, k2 = jax.random.split(key)

    # Kuramoto -> conditioning vectors
    actions = kuramoto_action(carry.kuramoto, cfg.n_actions)   # (K, n_actions)
    delta_x = model.action_bridge(actions)                      # (n_tokens, d_boundary)
    delta_v = model.belief_bridge(carry.kuramoto.theta)         # (n_tokens, d_model)

    # Lorentz embedding of raw tokens
    q_data, z = model.lorentz_embed(tokens)                    # q_data: (n_tokens, d_boundary)

    # Backbone: condition tokens with phase belief, then encode
    h_conditioned = tokens + delta_v                            # (n_tokens, d_model)
    h_out = model.backbone(h_conditioned, q_data, z)           # (n_tokens, d_model)

    # Hamiltonian dynamics on embedded positions
    p0 = model.momentum_init(h_out)                            # (n_tokens, d_boundary)
    q_final, p_final = leapfrog_integrate(
        model.hamiltonian, q_data, p0,
        cfg.n_leapfrog_steps, cfg.leapfrog_step_size
    )

    # Page memory update (one entry per token)
    def update_mem(mem, x_i):
        return model.page_memory(x_i, mem), None
    new_page_mem, _ = jax.lax.scan(update_mem, carry.page_mem, h_out)

    # Observations from backbone -> drive Kuramoto
    obs = model.obs_bridge(h_out)                              # (K, n_obs)

    # Bohmian pilot wave: momentum field from Hamiltonian guides the swarm
    # p_final is the evolved momentum (∇S of the wave function)
    # Project from (n_tokens, d_boundary) to (K, n_hidden) via assignment
    assignment = jax.nn.softmax(model.obs_bridge.assignment_logits, axis=-1)  # (K, n_tokens)
    pilot_wave = assignment @ p_final                          # (K, d_boundary)

    new_kuramoto = kuramoto_step(carry.kuramoto, obs, cfg, pilot_wave=pilot_wave,
                                 coherence_weights=coherence_weights)

    new_carry = Halo3Carry(
        kuramoto=new_kuramoto,
        page_mem=new_page_mem,
        key=jax.random.split(key)[0],
    )
    return new_carry, (h_out, obs, q_final, q_data)
