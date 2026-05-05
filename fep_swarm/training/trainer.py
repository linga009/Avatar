import jax
import jax.numpy as jnp
from fep_swarm.config import FEPConfig
from fep_swarm.agent.markov_blanket import AgentState
from fep_swarm.agent.belief_update import belief_update, free_energy
from fep_swarm.agent.action_selection import select_action, build_policies
from fep_swarm.swarm.environment import init_env, observe, step_env
from fep_swarm.swarm.coupling import build_coupling_matrix, apply_coupling
from fep_swarm.swarm.synchrony import synchrony_metric, mutual_information_estimate
from fep_swarm.macro.renormalization import coarse_grain
from fep_swarm.macro.macro_blanket import (
    macro_free_energy, micro_free_energy_sum, check_macro_bound,
)
from fep_swarm.macro.eigenanalysis import compute_jacobian, temporal_horizons
from fep_swarm.generative_model.discrete_gm import DiscreteGenerativeModel
from fep_swarm.data.synthetic_world import make_tmaze


def run_episode(
    cfg: FEPConfig,
    gm: DiscreteGenerativeModel,
    key: jax.random.PRNGKey,
    n_steps: int = 500,
    compute_jacobian_at_end: bool = True,
) -> dict:
    """
    Run full 4-layer episode. Returns history dict with keys:
    F_history, S_history, F_macro_history, F_micro_sum_history,
    I_sync_history, mu_history, eigenvalue_magnitudes (if requested).
    """
    A_true, B_true, _, _ = make_tmaze(cfg)
    k0, k1, k2 = jax.random.split(key, 3)

    env = init_env(cfg, k0)
    mu = jax.random.normal(k1, (cfg.n_agents, cfg.n_hidden))
    actions = jax.nn.softmax(
        jax.random.normal(k2, (cfg.n_agents, cfg.n_actions)), axis=-1
    )
    W = build_coupling_matrix(cfg, jax.random.PRNGKey(cfg.seed))

    F_history, S_history = [], []
    F_macro_history, F_micro_sum_history, I_sync_history = [], [], []
    mu_history = []
    mu_prev = mu

    for t in range(n_steps):
        key, k_obs, k_act, k_env = jax.random.split(key, 4)

        # Observations
        obs_idx = observe(env, cfg, A_true, k_obs)              # [N]
        obs_soft = jax.vmap(
            lambda o: jax.nn.one_hot(o, cfg.n_obs)
        )(obs_idx)                                               # [N, n_obs]

        # Coupling
        obs_coupled = apply_coupling(obs_soft, actions, W, cfg) # [N, n_obs]
        obs_coupled_idx = jax.vmap(jnp.argmax)(obs_coupled)     # [N]

        # Belief update (vmapped)
        mu = jax.vmap(
            lambda m, o: belief_update(m, o, gm, cfg)
        )(mu, obs_coupled_idx)

        # Action selection (vmapped)
        policies = build_policies(cfg, key)
        keys_act = jax.random.split(k_act, cfg.n_agents)
        actions, _ = jax.vmap(
            lambda m, k: select_action(m, policies, gm, cfg, k)
        )(mu, keys_act)

        # World step
        env = step_env(env, actions, B_true, cfg, k_env)

        # Metrics -- Layer 2
        F_vals = jax.vmap(
            lambda m, o: free_energy(m, o, gm)
        )(mu, obs_coupled_idx)
        F_history.append(float(F_vals.mean()))

        # Metrics -- Layer 3
        S = synchrony_metric(mu, mu_prev)
        S_history.append(float(S))

        # Metrics -- Layer 4
        macro = coarse_grain(mu, obs_coupled, actions, W, cfg)
        F_m = macro_free_energy(macro, gm, cfg)
        F_sum = micro_free_energy_sum(mu, obs_coupled_idx, gm, cfg)
        # Compute MI from sliding window of mu history
        if len(mu_history) >= 10:
            mu_window = jnp.stack(mu_history[-10:] + [mu])  # [11, N, n_hidden]
            I_sync = mutual_information_estimate(mu_window)
        else:
            I_sync = jnp.array(0.0)
        F_macro_history.append(float(F_m))
        F_micro_sum_history.append(float(F_sum))
        I_sync_history.append(float(I_sync))

        mu_history.append(mu)
        mu_prev = mu

    result = dict(
        F_history=F_history,
        S_history=S_history,
        F_macro_history=F_macro_history,
        F_micro_sum_history=F_micro_sum_history,
        I_sync_history=I_sync_history,
        mu_history=mu_history,
    )

    if compute_jacobian_at_end:
        obs_idx_final = observe(env, cfg, A_true, key)
        _, gap, magnitudes = compute_jacobian(mu, obs_idx_final, gm, cfg)
        micro_h, macro_h = temporal_horizons(magnitudes, cfg)
        result["eigenvalue_magnitudes"] = magnitudes
        result["eig_gap"] = float(gap)
        result["micro_horizon"] = float(micro_h)
        result["macro_horizon"] = float(macro_h)

    return result
